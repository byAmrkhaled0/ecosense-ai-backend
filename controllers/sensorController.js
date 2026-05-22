const mongoose = require("mongoose");
const axios = require("axios");

// استدعاء الموديلات مرة واحدة وبشكل صحيح
const SensorData = require("../models/SensorData");
const Device = require("../models/Device");
const Sector = require("../models/Sector");
const Notification = require("../models/Notification");
const User = require("../models/User");

/* ============================================================
   HELPERS (التنسيق لمتطلبات الـ AI)
   ============================================================ */

// 🛠️ دالة مساعدة لتنظيف وتوحيد أسماء المحاصيل قبل إرسالها للـ AI
const formatCropType = (crop) => {
  const crops = {
    maize: "Corn",
    corn: "Corn",
    tomato: "Tomato",
    pepper: "Pepper",
    mint: "Mint",
  };
  const normalized = String(crop || "").toLowerCase();
  return crops[normalized] || "Corn"; // إذا لم يجد المحصول، يضع Corn كقيمة افتراضية
};

// 🛠️ دالة مساعدة لتنظيف وتوحيد قيم الإضاءة
const formatLightValue = (light) => {
  const lightMap = {
    high: "Sufficient",
    medium: "Medium",
    low: "Low",
    sufficient: "Sufficient",
  };
  const normalized = String(light || "").toLowerCase();
  return lightMap[normalized] || "Medium"; // قيمة افتراضية في حال عدم المطابقة
};

/* ============================================================ 
   1️⃣ مسار استقبال بيانات الحساسات (خاص بالجهاز فقط ⚡ FormData)
   ============================================================ */
exports.uploadDataOnly = async (req, res) => {
  try {
    const { deviceSerial } = req.body;

    // تحويل القيم إلى أرقام وضمان عدم وجود قيم فارغة
    const temp = parseFloat(req.body.temp) || 0;
    const hum = parseFloat(req.body.hum) || 0;
    const Soil = parseFloat(req.body.Soil) || 0;
    const light = req.body.light || "Unknown";

    if (!deviceSerial) {
      return res
        .status(400)
        .json({ success: false, message: "deviceSerial مطلوب" });
    }

    // البحث عن الجهاز والقطاع المرتبط به
    const device = await Device.findOne({ deviceSerial }).populate("sectorId");
    if (!device || !device.sectorId) {
      return res
        .status(404)
        .json({ success: false, message: "الجهاز غير مربوط بقطاع" });
    }

    const sector = device.sectorId;
    const finalOwnerId = device.ownerId || sector.ownerId;

    // تحديث حالة الجهاز لـ online وتسجيل وقت التفاعل فوراً
    await Device.findByIdAndUpdate(device._id, {
      status: "online",
      lastPing: Date.now(),
    });

    // إنشاء سجل القراءات بقيم تحليل افتراضية لضمان سرعة الرد على جهاز الـ IoT
    const newData = await SensorData.create({
      ownerId: finalOwnerId,
      sectorId: sector._id,
      deviceId: device._id,
      air: { temperature: temp, humidity: hum },
      soil: { moisture: Soil },
      light: String(light),
      analysis: {
        status: "Safe",
        recommendation: "في انتظار طلب التحليل من التطبيق...",
      },
    });

    // الرد الفوري والسريع على الميكروكنترولر
    return res.status(200).json({
      success: true,
      message: "Data Saved Successfully",
      dataId: newData._id,
    });
  } catch (err) {
    console.error("❌ Upload Error:", err.message);
    return res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================ 
   2️⃣ مسار طلب التحليل من الـ AI (يطلبه الويب أو الفلاتر 🧠 المحمي بتوكن)
   ============================================================ */
exports.analyzeLastReading = async (req, res) => {
  try {
    const { sectorId } = req.params;

    // 1. جلب آخر قراءة مسجلة لهذا القطاع تحديداً
    const lastReading = await SensorData.findOne({ sectorId })
      .sort({ createdAt: -1 })
      .populate("sectorId");

    if (!lastReading) {
      return res.status(404).json({
        success: false,
        message: "لا توجد قراءات مسجلة لهذا القطاع بعد",
      });
    }

    const sector = lastReading.sectorId;
    const finalOwnerId = lastReading.ownerId;
    const assignedWorkerId = sector.assignedWorker;

    // 2. تجهيز بيانات الـ AI الافتراضية
    let aiAnalysis = {
      status: "Safe",
      recommendation: "سيرفر الـ AI لم يستجب",
    };

    try {
      // تطبيق دالات التنظيف والـ Formatting للـ JSON بـقيمة متوافقة مع الموديل
      const formattedCrop = formatCropType(sector.cropType);
      const formattedLight = formatLightValue(lastReading.light);

      console.log(
        `📡 Sending to AI -> Crop: ${formattedCrop}, Light: ${formattedLight}`,
      );

      const aiResponse = await axios.post(
        "https://amr2004-ecosense-ai.hf.space/api/predict_sensors",
        {
          cropType: formattedCrop,
          temperature: Number(lastReading.air.temperature),
          humidity: Number(lastReading.air.humidity),
          soilMoisture: Number(lastReading.soil.moisture),
          soilTemp: Number(lastReading.air.temperature) - 2,
          light: formattedLight, // تأكد إذا كان السيرفر يتوقعها نصية أو حقل رقمي
        },
        {
          headers: {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
          },
          timeout: 10000, // رفعنا الـ timeout لـ 10 ثواني لضمان الرد
        },
      );

      if (aiResponse.data) {
        const data = aiResponse.data;
        console.log("✅ AI Response Data:", data);

        // معالجة مرنة للتوصيات لو رجعت Array أو نص عادي
        let recs = "لا توجد توصيات حالياً";
        if (data.recommendations) {
          recs = Array.isArray(data.recommendations)
            ? data.recommendations.join(" | ")
            : data.recommendations;
        } else if (data.summary) {
          recs = data.summary;
        }

        aiAnalysis = {
          status: data.final_status || data.status || "Safe",
          recommendation: recs,
        };
      }
    } catch (aiErr) {
      // طباعة تفاصيل الخطأ كاملة في الـ Console عندك عشان تعرف سيرفر الـ AI زعلان من إيه بالظبط
      console.error(
        "⚠️ AI Server Error Details:",
        aiErr.response ? aiErr.response.data : aiErr.message,
      );
      aiAnalysis.recommendation =
        "تعذر الاتصال بسيرفر الـ AI، تم استخدام التقييم التلقائي المحدود.";
    }

    // 3. تحديث نفس السجل المسترجع ببيانات الـ AI الجديدة وحفظه
    lastReading.analysis = aiAnalysis;
    await lastReading.save();

    // 4. الرد المباشر على فرونت الويب أو الفلاتر بالبيانات الكاملة والمحدثة
    res
      .status(200)
      .json({ success: true, message: "Analysis updated", data: lastReading });

    // 5. نظام التنبيهات الفورية والإشعارات في الخلفية
    (async () => {
      try {
        const currentStatus = aiAnalysis.status;
        const criticalStatuses = [
          "High Stress",
          "Danger",
          "Critical",
          "Warning",
        ];

        const isCritical =
          lastReading.air.temperature > 45 ||
          lastReading.soil.moisture < 10 ||
          criticalStatuses.includes(currentStatus);

        if (isCritical) {
          const io = req.app.get("io");
          const socketPayload = {
            title: "🚨 تنبيه خطر فوري",
            message: `القطاع: ${sector.name} | الحالة: ${currentStatus} | حرارة: ${lastReading.air.temperature}°C`,
            sectorId: sector._id,
            createdAt: new Date(),
          };

          if (io) {
            // 👇 ضيف السطر ده فوراً فوق الـ Rooms عشان يبعت للكل وتشوفه في شاشتك
            io.emit("newNotification", socketPayload);
            console.log("📢 تم بث الإشعار بنجاح للجميع عبر السوكت");

            if (finalOwnerId)
              io.to(finalOwnerId.toString()).emit(
                "newNotification",
                socketPayload,
              );
            if (assignedWorkerId)
              io.to(assignedWorkerId.toString()).emit(
                "newNotification",
                socketPayload,
              );
          }
          const usersToNotify = await User.find({
            _id: { $in: [finalOwnerId, assignedWorkerId].filter(Boolean) },
          }).select("fcmToken");

          for (const user of usersToNotify) {
            if (user.fcmToken) {
              try {
                await axios.post(
                  "https://fcm.googleapis.com/fcm/send",
                  {
                    to: user.fcmToken,
                    notification: {
                      title: socketPayload.title,
                      body: socketPayload.message,
                      sound: "default",
                    },
                    priority: "high",
                  },
                  {
                    headers: {
                      Authorization: `key=${process.env.FIREBASE_SERVER_KEY}`,
                      "Content-Type": "application/json",
                    },
                    timeout: 4000,
                  },
                );
              } catch (fcmErr) {
                console.log("Firebase Send Error");
              }
            }
          }

          const notificationsToSave = [
            {
              recipient: finalOwnerId,
              sectorId: sector._id,
              title: socketPayload.title,
              message: socketPayload.message,
              type: "warning",
            },
          ];

          if (assignedWorkerId) {
            notificationsToSave.push({
              recipient: assignedWorkerId,
              sectorId: sector._id,
              title: socketPayload.title,
              message: socketPayload.message,
              type: "warning",
            });
          }

          await Notification.insertMany(notificationsToSave);
        }
      } catch (bgErr) {
        console.error("❌ Background Task Error:", bgErr.message);
      }
    })();
  } catch (err) {
    console.error("❌ Controller Error:", err.message);
    if (!res.headersSent) {
      res.status(500).json({ success: false, error: err.message });
    }
  }
};

/* ============================================================
3️⃣ GET LATEST READING (آخر قراءة محدثة للـ Dashboard)
============================================================ */
exports.getLatest = async (req, res) => {
  try {
    const { sectorId } = req.query;
    let filter = {};

    if (req.user.role === "worker") {
      const workerSectors = await Sector.find({
        assignedWorker: req.user._id,
      }).select("_id");
      const workerSectorIds = workerSectors.map((s) => s._id);

      if (sectorId) {
        if (!workerSectorIds.map((id) => id.toString()).includes(sectorId)) {
          return res.status(403).json({
            success: false,
            message: "غير مسموح لك بالوصول لهذا القطاع",
          });
        }
        filter = { sectorId };
      } else {
        filter = { sectorId: { $in: workerSectorIds } };
      }
    } else {
      filter = sectorId
        ? { ownerId: req.user._id, sectorId }
        : { ownerId: req.user._id };
    }

    const latestData = await SensorData.findOne(filter)
      .sort({ createdAt: -1 })
      .populate("sectorId", "name cropType location")
      .populate("deviceId", "deviceSerial status")
      .lean();

    if (!latestData) {
      return res
        .status(404)
        .json({ success: false, message: "لا توجد بيانات حالياً" });
    }

    res.status(200).json({ success: true, data: latestData });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
4️⃣ GET HISTORY (سجل البيانات مع الفلترة)
============================================================ */
exports.getHistory = async (req, res) => {
  try {
    const {
      sectorId,
      status,
      startDate,
      endDate,
      page = 1,
      limit = 10,
    } = req.query;

    let filter = {};

    if (req.user.role === "worker") {
      const workerSectors = await Sector.find({
        assignedWorker: req.user._id,
      }).select("_id");
      const workerSectorIds = workerSectors.map((s) => s._id);

      if (sectorId) {
        if (!workerSectorIds.map((id) => id.toString()).includes(sectorId)) {
          return res
            .status(403)
            .json({ success: false, message: "هذا القطاع ليس تحت مسؤوليتك" });
        }
        filter.sectorId = sectorId;
      } else {
        filter.sectorId = { $in: workerSectorIds };
      }
    } else {
      filter.ownerId = req.user._id;
      if (sectorId) filter.sectorId = sectorId;
    }

    if (status) filter["analysis.status"] = status;
    if (startDate || endDate) {
      filter.createdAt = {};
      if (startDate) filter.createdAt.$gte = new Date(startDate);
      if (endDate) filter.createdAt.$lte = new Date(endDate);
    }

    const history = await SensorData.find(filter)
      .sort({ createdAt: -1 })
      .skip((page - 1) * limit)
      .limit(Number(limit))
      .populate("sectorId", "name cropType")
      .lean();

    const total = await SensorData.countDocuments(filter);

    res.status(200).json({
      success: true,
      totalRecords: total,
      data: history,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
5️⃣ GET ANALYTICS (تحليلات الأداء اليومي)
============================================================ */
exports.getAnalytics = async (req, res) => {
  try {
    const { sectorId } = req.query;

    if (!sectorId) {
      return res
        .status(400)
        .json({ success: false, message: "يجب تحديد معرف القطاع" });
    }

    if (req.user.role === "worker") {
      const isAssigned = await Sector.findOne({
        _id: sectorId,
        assignedWorker: req.user._id,
      });
      if (!isAssigned) {
        return res.status(403).json({
          success: false,
          message: "غير مسموح لك برؤية تحليلات هذا القطاع",
        });
      }
    } else {
      const isOwner = await Sector.findOne({
        _id: sectorId,
        ownerId: req.user._id,
      });
      if (!isOwner)
        return res
          .status(403)
          .json({ success: false, message: "هذا القطاع لا ينتمي لمزرعتك" });
    }

    const startOfDay = new Date();
    startOfDay.setHours(0, 0, 0, 0);

    const analytics = await SensorData.aggregate([
      {
        $match: {
          sectorId: new mongoose.Types.ObjectId(sectorId),
          createdAt: { $gte: startOfDay },
        },
      },
      {
        $group: {
          _id: null,
          avgAirTemp: { $avg: "$air.temperature" },
          avgSoilMoist: { $avg: "$soil.moisture" },
          readingsCount: { $sum: 1 },
        },
      },
    ]);

    res.status(200).json({
      success: true,
      data: analytics[0] || { message: "لا توجد بيانات اليوم" },
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
