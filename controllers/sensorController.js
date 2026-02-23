const SensorData = require("../models/SensorData");
const Device = require("../models/Device");
const Sector = require("../models/Sector");
const User = require("../models/User");
const Notification = require("../models/Notification"); // ضفنا التنبيهات
const axios = require("axios");
const mongoose = require("mongoose");

/* ============================================================
1️⃣ UPLOAD SENSOR DATA (من الجهاز)
============================================================ */
exports.uploadData = async (req, res) => {
  try {
    const {
      deviceSerial,
      temperature,
      humidity,
      soilMoisture,
      soilTemperature,
      soilPH,
      light,
    } = req.body;

    // البحث عن الجهاز والتأكد من ربطه بقطاع
    const device = await Device.findOne({ deviceSerial }).populate("sector");
    if (!device || !device.sector) {
      return res.status(404).json({
        success: false,
        message: "الجهاز غير مسجل أو غير مربوط بقطاع",
      });
    }

    // إنشاء القراءة وربطها بالقطاع وصاحب المزرعة أوتوماتيكياً
    const newData = await SensorData.create({
      ownerId: device.owner,
      sectorId: device.sector._id,
      deviceId: device._id,
      air: { temperature, humidity },
      soil: {
        moisture: soilMoisture,
        temperature: soilTemperature,
        ph: soilPH,
      },
      light,
    });

    // تحديث حالة الجهاز (Heartbeat)
    device.status = "online";
    device.lastPing = Date.now();
    await device.save();

    // [إضافة احترافية]: لو الحرارة عالية جداً كريت تنبيه فوراً
    if (temperature > 45) {
      await Notification.create({
        title: "⚠️ تحذير حرارة",
        message: `الحرارة مرتفعة جداً في قطاع ${device.sector.name}: ${temperature}°C`,
        type: "alert",
        ownerId: device.owner,
        sectorId: device.sector._id,
      });
    }

    return res.status(201).json({ success: true, data: newData });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
2️⃣ GET LATEST READING (للمالك والعامل)
============================================================ */
exports.getLatest = async (req, res) => {
  try {
    let filter = {};

    // لو عامل، نجيب آخر قراءة لقطاعه بس
    if (req.user.role === "worker") {
      filter = { sectorId: req.user.assignedSector };
    } else {
      filter = { ownerId: req.user._id };
    }

    const latestData = await SensorData.findOne(filter)
      .sort({ timestamp: -1 })
      .populate("sectorId", "name");

    if (!latestData) {
      return res
        .status(404)
        .json({ success: false, message: "لا توجد بيانات متاحة" });
    }

    res.status(200).json({ success: true, data: latestData });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
3️⃣ AI PREDICTION (توقع حالة النبات)
============================================================ */
exports.predictStatus = async (req, res) => {
  try {
    const { sensorId } = req.body;
    const sensorData = await SensorData.findById(sensorId).populate("sectorId");

    if (!sensorData)
      return res.status(404).json({ message: "البيانات غير موجودة" });

    const cropType = sensorData.sectorId.cropType;

    // الاتصال بموديل الـ AI (Flask/FastAPI)
    const aiResponse = await axios
      .post(process.env.AI_API_URL || "http://127.0.0.1:8000/predict", {
        cropType,
        temperature: sensorData.air.temperature,
        humidity: sensorData.air.humidity,
        soilMoisture: sensorData.soil.moisture,
        soilPH: sensorData.soil.ph,
        light: sensorData.light,
      })
      .catch(() => {
        // لو الـ AI وقع، ندي نتيجة افتراضية عشان الكود ميفصلش
        return {
          data: {
            status: "Unknown",
            recommendations: ["تأكد من تشغيل سيرفر الـ AI"],
          },
        };
      });

    const { status, recommendations } = aiResponse.data;

    sensorData.analysis = {
      status: status,
      recommendation: recommendations.join(" | "),
    };
    await sensorData.save();

    res.status(200).json({ success: true, analysis: sensorData.analysis });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
4️⃣ GET HISTORY (With Search, Sort, and Pagination)
============================================================ */
exports.getHistory = async (req, res) => {
  try {
    // 1. استلام بارامترات البحث والترتيب والصفحات من الـ Query
    const {
      sectorId,
      status,
      startDate,
      endDate,
      sortBy = "timestamp",
      order = "desc",
      page = 1,
      limit = 10,
    } = req.query;

    // 2. بناء الفلتر (Search Logic)
    let filter = {};

    // أمان: لو عامل يشوف قطاعه بس، لو مالك يشوف مزارعه
    if (req.user.role === "worker") {
      filter.sectorId = req.user.assignedSector;
    } else {
      filter.ownerId = req.user._id;
      if (sectorId) filter.sectorId = sectorId; // فلتر بقطاع محدد للمالك
    }

    // فلتر بالحالة (مثلاً Healthy أو Warning)
    if (status) filter["analysis.status"] = status;

    // فلتر بالتاريخ (Search by Date Range)
    if (startDate || endDate) {
      filter.timestamp = {};
      if (startDate) filter.timestamp.$gte = new Date(startDate);
      if (endDate) filter.timestamp.$lte = new Date(endDate);
    }

    // 3. الترتيب (Sorting Logic)
    // بنخلي الـ order إما 1 (تصاعدي) أو -1 (تنازلي)
    const sortOrder = order === "desc" ? -1 : 1;
    const sortOptions = {};
    sortOptions[sortBy] = sortOrder;

    // 4. التنفيذ مع الـ Pagination
    const history = await SensorData.find(filter)
      .sort(sortOptions)
      .skip((page - 1) * limit) // تخطي البيانات السابقة
      .limit(Number(limit)) // تحديد كمية البيانات في الصفحة
      .populate("sectorId", "name cropType");

    // 5. حساب إجمالي النتائج والصفحات
    const total = await SensorData.countDocuments(filter);
    const totalPages = Math.ceil(total / limit);

    res.status(200).json({
      success: true,
      results: history.length,
      totalRecords: total,
      totalPages: totalPages,
      currentPage: Number(page),
      data: history,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
5️⃣ GET SECTOR STATS (إحصائيات مجمعة)
============================================================ */
exports.getSectorStats = async (req, res) => {
  try {
    const { sectorId } = req.query;

    if (!sectorId) {
      return res
        .status(400)
        .json({ success: false, message: "يجب تحديد معرف القطاع (sectorId)" });
    }

    const stats = await SensorData.aggregate([
      { $match: { sectorId: new mongoose.Types.ObjectId(sectorId) } },
      {
        $group: {
          _id: "$analysis.status",
          count: { $sum: 1 },
          avgTemp: { $avg: "$air.temperature" },
          avgMoisture: { $avg: "$soil.moisture" },
          avgHumidity: { $avg: "$air.humidity" },
        },
      },
    ]);

    res.status(200).json({ success: true, stats });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
