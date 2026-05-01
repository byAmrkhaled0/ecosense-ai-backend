const SensorData = require("../models/SensorData");
const Device = require("../models/Device");
const Sector = require("../models/Sector");
const Notification = require("../models/Notification");
const axios = require("axios");
const mongoose = require("mongoose");
const User = require("../models/User");

/* ============================================================
   HELPERS (التنسيق لمتطلبات الـ AI)
   ============================================================ */
const formatCropType = (crop) => {
  const crops = {
    maize: "Corn",
    corn: "Corn",
    tomato: "Tomato",
    pepper: "Pepper",
    mint: "Mint",
  };
  const normalized = String(crop || "").toLowerCase();
  return crops[normalized] || "Corn";
};

const formatLightValue = (light) => {
  const lightMap = {
    high: "Sufficient",
    medium: "Medium",
    low: "Low",
    sufficient: "Sufficient",
  };
  const normalized = String(light || "").toLowerCase();
  return lightMap[normalized] || "Medium";
};

/* ============================================================ 
   المسؤول عن استقبال بيانات الحساسات - مشروع EcoSense
   ============================================================ */

exports.uploadData = async (req, res) => {
  try {
    const { deviceSerial } = req.body;
    const temp = parseFloat(req.body.temp) || 0;
    const hum = parseFloat(req.body.hum) || 0;
    const Soil = parseFloat(req.body.Soil) || 0;
    const light = req.body.light || "Unknown";

    // 1. التحقق من البيانات الأساسية
    if (!deviceSerial) {
      return res
        .status(400)
        .json({ success: false, message: "deviceSerial مطلوب" });
    }

    const device = await Device.findOne({ deviceSerial }).populate("sectorId");
    if (!device || !device.sectorId) {
      return res
        .status(404)
        .json({ success: false, message: "الجهاز غير مربوط بقطاع" });
    }

    const sector = device.sectorId;
    const finalOwnerId = device.ownerId || sector.ownerId;
    const assignedWorkerId = sector.assignedWorker;

    // تحديث حالة الجهاز (Ping)
    await Device.findByIdAndUpdate(device._id, {
      status: "online",
      lastPing: Date.now(),
    });

    // 2. طلب تحليل الـ AI أولاً (قبل التخزين)

    try {
      const aiResponse = await axios.post(
        "https://Amrkhaled2004.pythonanywhere.com/api/predict_with_image",
        {
          cropType: sector.cropType,
          temperature: temp,
          humidity: hum,
          soilMoisture: Soil,
          light: light,
          // لو فيه صورة مبعوتة بتضيفها هنا
        },
        {
          headers: { "ngrok-skip-browser-warning": "true" },
          timeout: 8000,
        },
      );

      if (aiResponse.data) {
        const data = aiResponse.data;

        aiAnalysis = {
          // بناخد الحالة النهائية اللي الـ AI قررها بعد طبقة الحماية
          status: data.final_status || data.status || "Safe",

          // بنجمع التوصيات العامة وتوصيات الصورة في نص واحد
          recommendation: data.recommendations
            ? data.recommendations.join(" | ")
            : data.general_recommendation || "لا توجد توصيات",

          // إضافة اختيارية: لو عايز تخزن التشخيص البصري بالتحديد
          visualDiagnosis: data.diagnosis
            ? data.diagnosis.visual_problem_ar
            : null,
        };

        // تحديث السجل في قاعدة البيانات
        await SensorData.findByIdAndUpdate(newData._id, {
          analysis: aiAnalysis,
          // لو حابب تخزن الـ JSON كامل للرجوع ليه مستقبلاً
          aiRawResponse: data,
        });
      }
    } catch (aiErr) {
      console.log("⚠️ AI Error: " + aiErr.message);
      // في حالة الفشل، سيتم استخدام القيم الافتراضية لـ aiAnalysis المحددة فوق
    }

    // 3. تخزين الداتا النهائية (بعد ما معانا نتيجة الـ AI)
    // كده السجل هينزل بـ updatedAt و createdAt مطابقين ومعاهم النتيجة فوراً
    const newData = await SensorData.create({
      ownerId: finalOwnerId,
      sectorId: sector._id,
      deviceId: device._id,
      air: { temperature: temp, humidity: hum },
      soil: { moisture: Soil },
      light: String(light),
      analysis: aiAnalysis, // النتيجة جاهزة هنا
    });

    // 4. الرد على الجهاز (Accepted)
    res
      .status(200)
      .json({ success: true, message: "Accepted", dataId: newData._id });

    // 5. نظام التنبيهات (في الخلفية عشان ميعطلش الرد)
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
          temp > 45 || Soil < 10 || criticalStatuses.includes(currentStatus);

        if (isCritical) {
          const io = req.app.get("io");
          const socketPayload = {
            title: "🚨 تنبيه خطر فوري",
            message: `القطاع: ${sector.name} | الحالة: ${currentStatus} | حرارة: ${temp}°C`,
            sectorId: sector._id,
            createdAt: new Date(),
          };

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
                console.log("Firebase Notify Error");
              }
            }
          }

          if (io) {
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
2️⃣ GET LATEST READING (آخر قراءة محدثة للـ Dashboard)
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
3️⃣ GET HISTORY (سجل البيانات مع الفلترة)
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
4️⃣ GET ANALYTICS (تحليلات الأداء اليومي)
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
