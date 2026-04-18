const SensorData = require("../models/SensorData");
const Device = require("../models/Device");
const Sector = require("../models/Sector");
const Notification = require("../models/Notification");
const axios = require("axios");
const mongoose = require("mongoose");

/* ============================================================
1️⃣ UPLOAD SENSOR DATA (استقبال البيانات وتحليلها)
============================================================ */
exports.uploadData = async (req, res) => {
  try {
    const { deviceSerial, temp, hum, Soil, soilTemp, light } = req.body;

    if (!deviceSerial) {
      return res
        .status(400)
        .json({ success: false, message: "deviceSerial مطلوب" });
    }

    // 1. البحث عن الجهاز وعمل Populate للقطاع عشان نوصل للعامل والمالك
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
    const assignedWorkerId = sector.assignedWorker; // 👈 هنجيب الـ ID بتاع العامل من هنا

    // 2. محاولة الاتصال بسيرفر الـ AI (نفس الكود بتاعك)
    let aiAnalysis = {
      status: "Unknown",
      recommendation: "سيرفر الـ AI غير متصل",
    };
    try {
      const aiResponse = await axios.post(
        process.env.AI_API_URL || "http://127.0.0.1:8000/predict",
        {
          cropType: sector.cropType || "general",
          temperature: temp,
          humidity: hum,
          soilMoisture: Soil,
          soilTemp: soilTemp,
          light: light,
        },
        { timeout: 4000 },
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
      console.log("⚠️ AI Server unreachable");
    }

    // 3. حفظ القراءة
    const newData = await SensorData.create({
      ownerId: finalOwnerId,
      sectorId: finalSectorId,
      deviceId: device._id,
      air: { temperature: temp, humidity: hum },
      soil: { moisture: Soil, temperature: soilTemp },
      light: String(light),
      analysis: aiAnalysis,
    });

    // 4. تحديث حالة الجهاز
    await Device.findByIdAndUpdate(device._id, {
      status: "online",
      lastPing: Date.now(),
    });

    // 5. 🔔 نظام التنبيهات المطور (للمالك والعامل)
    const isCritical =
      temp > 45 ||
      Soil < 10 ||
      aiAnalysis.status === "Danger" ||
      aiAnalysis.status === "Critical";

    if (isCritical) {
      const io = req.app.get("io");

      const socketPayload = {
        title: "🚨 تنبيه خطر فوري",
        message: `خطر في قطاع (${sector.name}). الحالة: ${aiAnalysis.status}`,
        sectorId: finalSectorId,
        createdAt: new Date(),
      };

      // 1️⃣ إرسال الإشعار للمالك (Owner)
      // تأكد إن المالك عمل join لغرفة باسم الـ ID بتاعه عند الاتصال
      io.to(finalOwnerId.toString()).emit("newNotification", socketPayload);

      // 2️⃣ إرسال الإشعار للعامل (Worker) إذا وجد
      if (assignedWorkerId) {
        io.to(assignedWorkerId.toString()).emit(
          "newNotification",
          socketPayload,
        );
      }

      // 3️⃣ خطوة احترافية: حفظ الإشعارات في قاعدة البيانات
      // عشان تظهر في "سجل التنبيهات" للمالك والعامل لما يفتحوا الأبلكيشن
      const notificationData = [
        {
          recipient: finalOwnerId,
          sectorId: finalSectorId,
          title: socketPayload.title,
          message: socketPayload.message,
          type: "warning",
        },
      ];

      if (assignedWorkerId) {
        notificationData.push({
          recipient: assignedWorkerId,
          sectorId: finalSectorId,
          title: socketPayload.title,
          message: socketPayload.message,
          type: "warning",
        });
      }

      // حفظ الكل مرة واحدة في الداتابيز
      await Notification.insertMany(notificationData);

      console.log(
        `✅ تم إرسال التنبيه للمالك ${finalOwnerId} ${assignedWorkerId ? "والعامل " + assignedWorkerId : ""}`,
      );
    }

    res.status(201).json({ success: true, data: newData });
  } catch (err) {
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
      // 🚩 نجيب كل الـ IDs للقطاعات اللي العامل مسؤول عنها
      const workerSectors = await Sector.find({
        assignedWorker: req.user._id,
      }).select("_id");
      const workerSectorIds = workerSectors.map((s) => s._id);

      // لو باعت ID معين، نتأكد إنه ضمن حاجته، لو مش باعت، ندور في كل حاجته
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
      // المالك
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

    // بقية الفلاتر (التاريخ والحالة)
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

    // 🚩 التحقق من الصلاحية قبل البدء
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
      // التأكد إن القطاع يخص المالك
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
