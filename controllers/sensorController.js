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
    const { deviceSerial, temp, hum, soilMoist, soilTemp, light } = req.body;

    if (!deviceSerial) {
      return res
        .status(400)
        .json({ success: false, message: "deviceSerial مطلوب" });
    }

    // 1. البحث عن الجهاز وعمل Populate للقطاع
    const device = await Device.findOne({ deviceSerial }).populate("sectorId");

    if (!device) {
      return res.status(404).json({
        success: false,
        message: `الجهاز ${deviceSerial} غير مسجل، يرجى إضافته أولاً`,
      });
    }

    // 🚩 الخطوة السحرية: استخراج الـ IDs بشكل مرن (Robust Extraction)
    // بنسحب الـ ownerId سواء كان مخزن كـ ID أو كائن، ونفس الكلام للقطاع
    const finalOwnerId =
      device.ownerId || (device.sectorId ? device.sectorId.ownerId : null);
    const finalSectorId = device.sectorId
      ? device.sectorId._id || device.sectorId
      : null;

    // التحقق من وجود البيانات الأساسية قبل الحفظ لمنع الـ Validation Error
    if (!finalOwnerId || !finalSectorId) {
      return res.status(400).json({
        success: false,
        message:
          "بيانات الجهاز غير مكتملة (يفتقد للمالك أو القطاع في الداتابيز)",
      });
    }

    // 2. محاولة الاتصال بسيرفر الـ AI
    let aiAnalysis = {
      status: "Unknown",
      recommendation: "سيرفر الـ AI غير متصل",
    };

    try {
      const aiResponse = await axios.post(
        process.env.AI_API_URL || "http://127.0.0.1:8000/predict",
        {
          cropType: device.sectorId ? device.sectorId.cropType : "general",
          temperature: temp,
          humidity: hum,
          soilMoisture: soilMoist,
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
            : "لا توجد توصيات حالياً",
        };
      }
    } catch (aiErr) {
      console.log("⚠️ AI Server unreachable, using defaults.");
    }

    // 3. حفظ القراءة في SensorData (استخدام القيم المستخرجة)
    const newData = await SensorData.create({
      ownerId: finalOwnerId,
      sectorId: finalSectorId,
      deviceId: device._id,
      air: { temperature: temp, humidity: hum },
      soil: { moisture: soilMoist, temperature: soilTemp },
      light: String(light),
      analysis: aiAnalysis,
    });

    // 4. تحديث حالة الجهاز لـ Online
    await Device.findByIdAndUpdate(device._id, {
      status: "online",
      lastPing: Date.now(),
    });

    // 5. نظام التنبيهات الفوري
    if (temp > 45 || soilMoist < 10) {
      await Notification.create({
        title: "🚨 تنبيه خطر",
        message: `قراءات حرجة في ${device.sectorId ? device.sectorId.name : "قطاعك"}. حرارة: ${temp}°C، رطوبة: ${soilMoist}%`,
        type: "alert",
        recipient: finalOwnerId,
        sectorId: finalSectorId,
      });
    }

    res.status(201).json({
      success: true,
      message: "تم استقبال البيانات ومعالجتها بنجاح",
      data: newData,
    });
  } catch (err) {
    console.error("❌ Upload Error:", err.message);
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
      filter = { sectorId: req.user.assignedSector };
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

    res.status(200).json({
      success: true,
      data: {
        timestamp: latestData.createdAt,
        sector: latestData.sectorId,
        device: latestData.deviceId,
        readings: {
          air: latestData.air,
          soil: latestData.soil,
          light: latestData.light,
        },
        ai_analysis: latestData.analysis,
      },
    });
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

    let filter =
      req.user.role === "worker"
        ? { sectorId: req.user.assignedSector }
        : { ownerId: req.user._id };

    if (sectorId) filter.sectorId = sectorId;
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

    const formattedHistory = history.map((item) => ({
      _id: item._id,
      timestamp: item.createdAt,
      sector: item.sectorId,
      readings: {
        air: item.air,
        soil: item.soil,
        light: item.light,
      },
      analysis: item.analysis,
    }));

    res.status(200).json({
      success: true,
      totalRecords: total,
      totalPages: Math.ceil(total / limit),
      currentPage: Number(page),
      data: formattedHistory,
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

    const startOfDay = new Date();
    startOfDay.setHours(0, 0, 0, 0);
    const endOfDay = new Date();
    endOfDay.setHours(23, 59, 59, 999);

    const analytics = await SensorData.aggregate([
      {
        $match: {
          sectorId: new mongoose.Types.ObjectId(sectorId),
          createdAt: { $gte: startOfDay, $lte: endOfDay },
        },
      },
      {
        $group: {
          _id: null,
          maxAirTemp: { $max: "$air.temperature" },
          minAirTemp: { $min: "$air.temperature" },
          avgAirHum: { $avg: "$air.humidity" },
          maxSoilTemp: { $max: "$soil.temperature" },
          avgSoilMoist: { $avg: "$soil.moisture" },
          readingsCount: { $sum: 1 },
        },
      },
      {
        $project: {
          _id: 0,
          air: {
            maxTemp: { $round: ["$maxAirTemp", 1] },
            minTemp: { $round: ["$minAirTemp", 1] },
            avgHumidity: { $round: ["$avgAirHum", 1] },
          },
          soil: {
            maxTemp: { $round: ["$maxSoilTemp", 1] },
            avgMoisture: { $round: ["$avgSoilMoist", 1] },
          },
          totalReadings: "$readingsCount",
        },
      },
    ]);

    res.status(200).json({
      success: true,
      data:
        analytics.length > 0
          ? analytics[0]
          : { message: "لا توجد قراءات مسجلة اليوم" },
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
