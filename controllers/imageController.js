const ImageLog = require("../models/ImageLog");
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");
const Sector = require("../models/Sector");
const Notification = require("../models/Notification"); // ضفنا الموديل ده
const mongoose = require("mongoose");
const Device = require("../models/Device");
/* ============================================================
    1️⃣ UPLOAD & AI ANALYZE (رفع الصورة وتحليلها)
============================================================ */
exports.uploadImage = async (req, res) => {
  try {
    const { deviceSerial, sectorId } = req.body;
    let finalSectorId,
      finalOwnerId,
      finalWorkerId,
      uploadedBy,
      device,
      cropType;
    let captureReason = "Manual Scan";

    if (!req.file) {
      return res.status(400).json({ success: false, message: "يرجى رفع صورة" });
    }

    // --- حالة 1: تصوير آلي (IoT - ESP32-CAM) ---
    if (deviceSerial) {
      // بنجيب الجهاز ونعمل Populate للقطاع عشان نجيب نوع الزرعة والمكان
      device = await Device.findOne({ deviceSerial }).populate("sectorId");

      if (!device || !device.sectorId) {
        return res.status(404).json({
          success: false,
          message: "هذا الجهاز غير مسجل أو غير مربوط بقطاع زراعي",
        });
      }

      finalSectorId = device.sectorId._id;
      finalOwnerId = device.sectorId.ownerId;
      finalWorkerId = device.sectorId.assignedWorker;
      cropType = device.sectorId.cropType || "Unknown"; // هنجيب نوع الزرعة أوتوماتيك
      uploadedBy = null;
      captureReason = "Automatic Camera";

      console.log(
        `📡 IoT Device [${deviceSerial}] detected in Sector: ${device.sectorId.name} | Crop: ${cropType}`,
      );
    }
    // --- حالة 2: تصوير يدوي (Mobile App) ---
    else if (req.user) {
      // في حالة الموبايل، لسه محتاجين الـ sectorId عشان نعرف اليوزر بيصور في أنهي حتة
      if (!sectorId)
        return res
          .status(400)
          .json({ success: false, message: "sectorId مطلوب للفحص اليدوي" });

      const sector = await Sector.findById(sectorId);
      if (!sector)
        return res
          .status(404)
          .json({ success: false, message: "القطاع غير موجود" });

      finalSectorId = sector._id;
      finalOwnerId = sector.ownerId;
      finalWorkerId = sector.assignedWorker;
      cropType = sector.cropType || "Unknown";
      uploadedBy = req.user._id;
      captureReason = "Manual Scan";
    } else {
      return res.status(401).json({
        success: false,
        message: "يجب توفير Serial الجهاز أو Token المستخدم",
      });
    }

    const imageUrl = `${req.protocol}://${req.get("host")}/uploads/${req.file.filename}`;

    // --- (جزء تحليل الـ AI) ---
    let aiAnalysis = {
      status: "Unknown",
      diseaseName: "تحليل غير متاح",
      confidence: 0,
      recommendation: "سيرفر الـ AI لا يستجيب",
    };

    try {
      const formData = new FormData();
      // إرفاق الصورة
      formData.append(
        "file",
        req.file.buffer || require("fs").createReadStream(req.file.path),
        {
          filename: req.file.originalname,
          contentType: req.file.mimetype,
        },
      );

      // 🚀 إضافة نوع المحصول لعمرو
      formData.append("cropType", cropType);

      const aiResponse = await axios.post(
        "https://Amrkhaled2004.pythonanywhere.com/api/predict_with_image",
        formData,
        {
          headers: {
            ...formData.getHeaders(),
            "ngrok-skip-browser-warning": "true",
          },
          timeout: 15000,
        },
      );

      // ... (باقي كود معالجة الرد aiResponse.data يفضل كما هو)
      if (aiResponse.data) {
        const analysisData =
          aiResponse.data.analysis || aiResponse.data.image_analysis;
        let conf = parseFloat(aiResponse.data.confidence);
        conf = isNaN(conf) ? 0 : conf;

        aiAnalysis = {
          status: aiResponse.data.status || "Infected",
          diseaseName: aiResponse.data.disease_name || "Severe Plant Stress",
          confidence: conf,
          recommendation: aiResponse.data.recommendations
            ? aiResponse.data.recommendations.join(" | ")
            : "يرجى مراجعة المختص",
        };

        if (analysisData) {
          const g = parseFloat(analysisData.green_ratio || 0) * 100;
          const y = parseFloat(analysisData.yellow_ratio || 0) * 100;
          const b = parseFloat(analysisData.brown_ratio || 0) * 100;
          aiAnalysis.recommendation += ` [تحليل الألوان: أخضر ${g.toFixed(1)}% | أصفر ${y.toFixed(1)}% | بني ${b.toFixed(1)}%]`;
        }
      }
    } catch (aiErr) {
      console.log("⚠️ Image AI Server Error:", aiErr.message);
    }

    // حفظ السجل في الداتابيز
    const newImageLog = await ImageLog.create({
      ownerId: finalOwnerId,
      sectorId: finalSectorId,
      imageUrl: imageUrl,
      capturedBy: uploadedBy || finalOwnerId,
      analysisResult: aiAnalysis,
      deviceId: device?._id,
      captureReason: captureReason,
    });

    // 4️⃣ الإشعارات (تخصيص الرسالة)
    if (aiAnalysis.status !== "Healthy") {
      const io = req.app.get("io");
      const notificationData = {
        title: "🚨 تنبيه صحة النبات",
        // الرسالة تتغير حسب مين اللي صور
        message:
          captureReason === "Manual Scan"
            ? `نتائج الفحص اليدوي: رصد (${aiAnalysis.diseaseName}).`
            : `الكاميرا الآلية رصدت إصابة (${aiAnalysis.diseaseName}).`,
        type: "disease",
        sectorId: finalSectorId,
      };

      const nOwner = await Notification.create({
        ...notificationData,
        recipient: finalOwnerId,
      });
      io.to(finalOwnerId.toString()).emit("newNotification", nOwner);

      if (finalWorkerId) {
        const nWorker = await Notification.create({
          ...notificationData,
          recipient: finalWorkerId,
        });
        io.to(finalWorkerId.toString()).emit("newNotification", nWorker);
      }
    }

    res.status(201).json({ success: true, data: newImageLog });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
/* ============================================================
    2️⃣ GET IMAGE HISTORY (عرض التاريخ بتعدد القطاعات)
============================================================ */
exports.getImageHistory = async (req, res) => {
  try {
    const { sectorId, page = 1, limit = 10 } = req.query;
    let filter = {};

    if (req.user.role === "worker") {
      // 🚩 نجيب كل القطاعات اللي العامل مسؤول عنها
      const workerSectors = await Sector.find({
        assignedWorker: req.user._id,
      }).select("_id");
      const sectorIds = workerSectors.map((s) => s._id);

      if (sectorId) {
        if (!sectorIds.map((id) => id.toString()).includes(sectorId)) {
          return res
            .status(403)
            .json({ success: false, message: "ليس لديك صلاحية لهذا القطاع" });
        }
        filter.sectorId = sectorId;
      } else {
        filter.sectorId = { $in: sectorIds }; // صور كل قطاعاته
      }
    } else {
      filter.ownerId = req.user._id;
      if (sectorId) filter.sectorId = sectorId;
    }

    const images = await ImageLog.find(filter)
      .sort({ createdAt: -1 })
      .skip((page - 1) * Number(limit))
      .limit(Number(limit))
      .populate("sectorId", "name cropType")
      .populate("capturedBy", "firstName lastName")
      .lean();

    const total = await ImageLog.countDocuments(filter);

    res.status(200).json({
      success: true,
      totalRecords: total,
      data: images,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
/* ============================================================
    4️⃣ DELETE IMAGE LOG (حذف الصورة من السيرفر والداتابيز)
    DELETE /api/images/:id
============================================================ */
exports.deleteImageLog = async (req, res) => {
  try {
    // 1. البحث عن السجل والتأكد إن اللي بيمسح هو صاحب المزرعة (أو عامل له صلاحية)
    const log = await ImageLog.findById(req.params.id);

    if (!log) {
      return res
        .status(404)
        .json({ success: false, message: "لم يتم العثور على سجل هذه الصورة." });
    }

    // 2. التحقق من الصلاحيات (المالك فقط أو صاحب الصورة هو اللي يمسح)
    const ownerId = req.user.role === "owner" ? req.user._id : req.user.ownerId;
    if (log.ownerId.toString() !== ownerId.toString()) {
      return res
        .status(403)
        .json({ success: false, message: "ليس لديك صلاحية لحذف هذه الصورة." });
    }

    // 3. حذف الملف الفعلي من السيرفر (Physical Delete)
    // بنجيب اسم الملف من الـ URL المسجل
    const fileName = log.imageUrl.split("/").pop();
    const filePath = `./uploads/${fileName}`;

    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath); // حذف الملف من الهارد ديسك
    }

    // 4. حذف السجل من الداتابيز
    await log.deleteOne();

    res.status(200).json({
      success: true,
      message: "✅ تم حذف الصورة من السيرفر ومن السجلات بنجاح.",
    });
  } catch (err) {
    console.error("❌ Delete Image Error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
};
