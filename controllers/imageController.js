const ImageLog = require("../models/ImageLog");
const axios = require("axios");
const FormData = require("form-data");
const Sector = require("../models/Sector");
const Notification = require("../models/Notification");
const Device = require("../models/Device");
const User = require("../models/User");
const cloudinary = require("cloudinary").v2;

// دالة مساعدة لإرسال الـ Push Notification
const sendFirebasePush = async (user, title, message) => {
  if (user && user.fcmToken) {
    try {
      await axios.post(
        "https://fcm.googleapis.com/fcm/send",
        {
          to: user.fcmToken,
          notification: {
            title: title,
            body: message,
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
    } catch (err) {
      console.error("❌ Firebase Push Error:", err.message);
    }
  }
};

/* ============================================================
    1️⃣ UPLOAD & AI ANALYZE (نسخة Cloudinary المعدلة)
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

    // ✅ التعديل: الحصول على رابط الصورة مباشرة من Cloudinary
    const imageUrl = req.file.path;

    // --- حالة 1: تصوير آلي (IoT - ESP32-CAM) ---
    if (deviceSerial) {
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
      cropType = device.sectorId.cropType || "Unknown";
      uploadedBy = null;
      captureReason = "Automatic Camera";
    }
    // --- حالة 2: تصوير يدوي (Mobile App) ---
    else if (req.user) {
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

    // --- (جزء تحليل الـ AI) ---
    let aiAnalysis = {
      status: "Unknown",
      diseaseName: "تحليل غير متاح",
      confidence: 0,
      recommendation: "سيرفر الـ AI لا يستجيب",
    };

    try {
      const formData = new FormData();

      // ✅ التعديل: إرسال الصورة لسيرفر عمرو (تحميلها من Cloudinary كـ Stream)
      const imageResponse = await axios.get(imageUrl, {
        responseType: "stream",
      });

      formData.append("file", imageResponse.data, {
        filename: req.file.originalname,
        contentType: req.file.mimetype,
      });

      formData.append("cropType", cropType);

      const aiResponse = await axios.post(
        "https://Amrkhaled2004.pythonanywhere.com/api/predict_with_image",
        formData,
        {
          headers: {
            ...formData.getHeaders(),
            "ngrok-skip-browser-warning": "true",
          },
          timeout: 20000, // زيادة المهلة قليلاً للمعالجة السحابية
        },
      );

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
      imageUrl: imageUrl, // رابط Cloudinary
      capturedBy: uploadedBy || finalOwnerId,
      analysisResult: aiAnalysis,
      deviceId: device?._id,
      captureReason: captureReason,
    });

    // 4️⃣ الإشعارات
    if (aiAnalysis.status !== "Healthy") {
      const io = req.app.get("io");
      const title = "🚨 تنبيه صحة النبات";
      const message =
        captureReason === "Manual Scan"
          ? `نتائج الفحص اليدوي: رصد (${aiAnalysis.diseaseName})`
          : `الكاميرا الآلية رصدت إصابة (${aiAnalysis.diseaseName})`;

      const owner = await User.findById(finalOwnerId);
      const worker = finalWorkerId ? await User.findById(finalWorkerId) : null;

      await sendFirebasePush(owner, title, message);
      if (worker) await sendFirebasePush(worker, title, message);

      const notificationData = {
        title,
        message,
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
    2️⃣ GET IMAGE HISTORY (عرض التاريخ)
============================================================ */
exports.getImageHistory = async (req, res) => {
  try {
    const { sectorId, page = 1, limit = 10 } = req.query;
    let filter = {};

    if (req.user.role === "worker") {
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
        filter.sectorId = { $in: sectorIds };
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
    3️⃣ DELETE IMAGE LOG (حذف الصورة من Cloudinary والداتابيز)
============================================================ */
exports.deleteImageLog = async (req, res) => {
  try {
    const log = await ImageLog.findById(req.params.id);

    if (!log) {
      return res
        .status(404)
        .json({ success: false, message: "لم يتم العثور على سجل الصورة." });
    }

    const ownerId = req.user.role === "owner" ? req.user._id : req.user.ownerId;
    if (log.ownerId.toString() !== ownerId.toString()) {
      return res
        .status(403)
        .json({ success: false, message: "ليس لديك صلاحية لحذف هذه الصورة." });
    }

    // ✅ التعديل: حذف الصورة من Cloudinary باستخدام الـ Public ID
    // استخراج الـ ID من الرابط: ecosense/images/filename
    const urlParts = log.imageUrl.split("/");
    const fileNameWithExt = urlParts[urlParts.length - 1];
    const publicIdWithoutExt = fileNameWithExt.split(".")[0];

    // تأكد من مسار الفولدر كما عرفته في ملف الـ upload.js
    const fullPublicId = `ecosense/images/${publicIdWithoutExt}`;

    await cloudinary.uploader.destroy(fullPublicId);

    // حذف السجل من الداتابيز
    await log.deleteOne();

    res.status(200).json({
      success: true,
      message: "✅ تم حذف الصورة من السحاب ومن السجلات بنجاح.",
    });
  } catch (err) {
    console.error("❌ Delete Image Error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
};
