const SensorData = require("../models/SensorData");
const Device = require("../models/Device");
const Sector = require("../models/Sector");
const User = require("../models/User");
const Notification = require("../models/Notification");
const axios = require("axios");
const mongoose = require("mongoose");

/* ============================================================
1️⃣ UPLOAD SENSOR DATA (استقبال البيانات وتحليلها)
============================================================ */
exports.uploadData = async (req, res) => {
  try {
    const { deviceSerial, temperature, humidity, soilMoisture, soilPH, light } =
      req.body;

    // البحث عن الجهاز ومعرفة القطاع المرتبط به
    const device = await Device.findOne({ deviceSerial }).populate("sector");

    if (!device || !device.sector) {
      return res.status(404).json({
        success: false,
        message: "الجهاز غير مسجل أو غير مربوط بقطاع معين",
      });
    }

    const cropType = device.sector.cropType;

    // إرسال البيانات لسيرفر الـ AI
    let aiAnalysis = {
      status: "Unknown",
      recommendation: "سيرفر الـ AI غير متصل",
    };
    try {
      const aiResponse = await axios.post(
        process.env.AI_API_URL || "http://127.0.0.1:8000/predict",
        {
          cropType,
          temperature,
          humidity,
          soilMoisture,
          soilPH,
          light,
        },
      );

      aiAnalysis = {
        status: aiResponse.data.status,
        recommendation: aiResponse.data.recommendations.join(" | "),
      };
    } catch (aiErr) {
      console.log("AI Server Error: ", aiErr.message);
    }

    // حفظ القراءة في الداتابيز
    const newData = await SensorData.create({
      ownerId: device.owner,
      sectorId: device.sector._id,
      deviceId: device._id,
      air: { temperature, humidity },
      soil: { moisture: soilMoisture, ph: soilPH },
      light,
      analysis: aiAnalysis,
    });

    // تحديث حالة الجهاز وتوقيت النشاط
    device.status = "online";
    device.lastPing = Date.now();
    await device.save();

    // نظام التنبيهات الفوري (Logic)
    if (temperature > 45 || soilMoisture < 10) {
      await Notification.create({
        title: "🚨 تنبيه خطر",
        message: `تم رصد قراءات حرجة في قطاع ${device.sector.name}. الحالة: ${aiAnalysis.status}`,
        // تأكد من نوع الـ type المسموح به في الموديل عندك (لو alert مش شغال جرب warning)
        type: "warning",
        // غيرنا ownerId لـ recipient عشان ده اللي الموديل طالبه
        recipient: device.owner,
        sectorId: device.sector._id,
      });
    }

    return res
      .status(201)
      .json({ success: true, analysis: aiAnalysis, data: newData });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
2️⃣ GET LATEST READING (آخر قراءة محدثة)
============================================================ */
exports.getLatest = async (req, res) => {
  try {
    let filter =
      req.user.role === "worker"
        ? { sectorId: req.user.assignedSector }
        : { ownerId: req.user._id };

    const latestData = await SensorData.findOne(filter)
      .sort({ timestamp: -1 })
      .populate("sectorId", "name");

    if (!latestData)
      return res
        .status(404)
        .json({ success: false, message: "لا توجد بيانات" });

    res.status(200).json({ success: true, data: latestData });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
3️⃣ GET HISTORY (سجل البيانات مع البحث والصفحات)
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
    let filter =
      req.user.role === "worker"
        ? { sectorId: req.user.assignedSector }
        : { ownerId: req.user._id };

    if (sectorId) filter.sectorId = sectorId;
    if (status) filter["analysis.status"] = status;
    if (startDate || endDate) {
      filter.timestamp = {};
      if (startDate) filter.timestamp.$gte = new Date(startDate);
      if (endDate) filter.timestamp.$lte = new Date(endDate);
    }

    const history = await SensorData.find(filter)
      .sort({ timestamp: -1 })
      .skip((page - 1) * limit)
      .limit(Number(limit))
      .populate("sectorId", "name cropType");

    const total = await SensorData.countDocuments(filter);

    res.status(200).json({
      success: true,
      totalRecords: total,
      totalPages: Math.ceil(total / limit),
      data: history,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
4️⃣ [جديد] GET ANALYTICS (تحليلات الأداء اليومي)
============================================================ */
exports.getAnalytics = async (req, res) => {
  try {
    const { sectorId } = req.query;

    if (!sectorId) {
      return res
        .status(400)
        .json({ success: false, message: "يجب تحديد معرف القطاع" });
    }

    // 1. حساب بداية اليوم (الساعة 00:00:00) ونهاية اليوم (23:59:59)
    const startOfDay = new Date();
    startOfDay.setHours(0, 0, 0, 0);

    const endOfDay = new Date();
    endOfDay.setHours(23, 59, 59, 999);

    // 2. التحويل لـ ObjectId للتأكد إن الماتش شغال صح
    const sectorObjectId = new mongoose.Types.ObjectId(sectorId);

    const analytics = await SensorData.aggregate([
      {
        $match: {
          sectorId: sectorObjectId,
          createdAt: { $gte: startOfDay, $lte: endOfDay }, // البحث في نطاق اليوم بالكامل
        },
      },
      {
        $group: {
          _id: null,
          maxTemp: { $max: "$air.temperature" },
          minTemp: { $min: "$air.temperature" },
          avgMoisture: { $avg: "$soil.moisture" },
          readingsCount: { $sum: 1 },
        },
      },
      {
        $project: {
          _id: 0, // عشان نشيل الـ null اللي بتظهر
          maxTemp: { $round: ["$maxTemp", 1] }, // تقريب الأرقام لخانة واحدة
          minTemp: { $round: ["$minTemp", 1] },
          avgMoisture: { $round: ["$avgMoisture", 1] },
          readingsCount: 1,
        },
      },
    ]);

    res.status(200).json({
      success: true,
      analytics:
        analytics.length > 0
          ? analytics[0]
          : { message: "لا توجد قراءات لهذا القطاع اليوم" },
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
