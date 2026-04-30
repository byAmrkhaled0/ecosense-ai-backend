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
   1️⃣ UPLOAD SENSOR DATA (استقبال البيانات وتحليلها)
   ============================================================ */
exports.uploadData = async (req, res) => {
  try {
    // تم إزالة soilTemp من هنا
    const { deviceSerial, temp, hum, Soil, light } = req.body;

    if (!deviceSerial) {
      return res
        .status(400)
        .json({ success: false, message: "deviceSerial مطلوب" });
    }

    // 1. البحث عن الجهاز وعمل Populate للقطاع
    const device = await Device.findOne({ deviceSerial }).populate("sectorId");

    if (!device || !device.sectorId) {
      return res.status(404).json({
        success: false,
        message: "الجهاز غير مربوط بقطاع أو غير مسجل",
      });
    }

    const sector = device.sectorId;
    const finalOwnerId = device.ownerId || sector.ownerId;
    const finalSectorId = sector._id;
    const assignedWorkerId = sector.assignedWorker;

    // 2. محاولة الاتصال بسيرفر الـ AI
    let aiAnalysis = {
      status: "Unknown",
      recommendation: "سيرفر الـ AI غير متصل",
    };

    try {
      const aiResponse = await axios.post(
        process.env.AI_API_URL ||
          "https://Amrkhaled2004.pythonanywhere.com/api/mobile_predict",
        {
          cropType: formatCropType(sector.cropType),
          temperature: Number(temp),
          humidity: Number(hum),
          soilMoisture: Number(Soil),
          soilTemp: 0, // نرسل قيمة صفرية طالما السنسور غير متاح تقنياً
          light: formatLightValue(light),
        },
        {
          headers: { "ngrok-skip-browser-warning": "true" },
          timeout: 8000,
        },
      );

      if (aiResponse.data) {
        aiAnalysis = {
          status: aiResponse.data.status || "Unknown",
          recommendation: aiResponse.data.recommendations
            ? aiResponse.data.recommendations.join(" | ")
            : "لا توجد توصيات",
        };
      }
    } catch (aiErr) {
      console.log("⚠️ AI Server unreachable:", aiErr.message);
    }

    // 3. حفظ القراءة في الداتابيز
    const newData = await SensorData.create({
      ownerId: finalOwnerId,
      sectorId: finalSectorId,
      deviceId: device._id,
      air: { temperature: temp, humidity: hum },
      soil: { moisture: Soil, temperature: null }, // نضعها null لأنها غير متوفرة
      light: String(light),
      analysis: aiAnalysis,
    });

    // 4. تحديث حالة الجهاز (Online)
    await Device.findByIdAndUpdate(device._id, {
      status: "online",
      lastPing: Date.now(),
    });

    // 5. 🔔 نظام التنبيهات الذكي
    const criticalStatuses = ["High Stress", "Danger", "Critical", "Warning"];
    const isCritical =
      Number(temp) > 45 ||
      Number(Soil) < 10 ||
      criticalStatuses.includes(aiAnalysis.status);

    if (isCritical) {
      const io = req.app.get("io");

      const socketPayload = {
        title: "🚨 تنبيه خطر فوري",
        message: `القطاع: ${sector.name} | الحالة: ${aiAnalysis.status} | حرارة: ${temp}°C`,
        sectorId: finalSectorId,
        createdAt: new Date(),
      };

      // جلب التوكنز للمالك والعامل
      const usersToNotify = await User.find({
        _id: { $in: [finalOwnerId, assignedWorkerId].filter(Boolean) },
      }).select("fcmToken");

      // إرسال Firebase Push
      usersToNotify.forEach(async (user) => {
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
              },
            );
          } catch (fcmErr) {
            console.error("❌ Firebase Error:", fcmErr.message);
          }
        }
      });

      // إرسال Socket.io
      if (finalOwnerId)
        io.to(finalOwnerId.toString()).emit("newNotification", socketPayload);
      if (assignedWorkerId)
        io.to(assignedWorkerId.toString()).emit(
          "newNotification",
          socketPayload,
        );

      // حفظ في الهيستوري
      const notificationsToSave = [
        {
          recipient: finalOwnerId,
          sectorId: finalSectorId,
          title: socketPayload.title,
          message: socketPayload.message,
          type: "warning",
        },
      ];

      if (assignedWorkerId) {
        notificationsToSave.push({
          recipient: assignedWorkerId,
          sectorId: finalSectorId,
          title: socketPayload.title,
          message: socketPayload.message,
          type: "warning",
        });
      }

      await Notification.insertMany(notificationsToSave);
    }

    res.status(201).json({ success: true, data: newData });
  } catch (err) {
    console.error("❌ General Error:", err.message);
    res.status(500).json({ success: false, error: err.message });
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
