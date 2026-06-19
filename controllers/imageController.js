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
    1️⃣ UPLOAD & AI ANALYZE
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

    const imageUrl = req.file.path;

    // 🎯 حالة 1: الـ deviceSerial مبعوتة صراحة (من الـ ESP32-CAM)
    if (deviceSerial && deviceSerial !== "ESP32-GENERIC-UNIT") {
      device = await Device.findOne({ deviceSerial }).populate("sectorId");

      if (!device || !device.sectorId) {
        return res.status(404).json({
          success: false,
          message: "هذا الجهاز غير مسجل أو غير مربوط بقطاع زراعي متاح",
        });
      }

      finalSectorId = device.sectorId._id;
      finalOwnerId = device.sectorId.ownerId;
      finalWorkerId = device.sectorId.assignedWorker;
      cropType = device.sectorId.cropType || "Unknown";
      uploadedBy = null;
      captureReason = "Automatic Camera";
    }
    // 🌿 حالة 2: التعرف الذكي عن طريق الـ sectorId (الرفع اليدوي)
    else if (sectorId) {
      const sector = await Sector.findById(sectorId);
      if (!sector) {
        return res.status(404).json({
          success: false,
          message: "القطاع المختار غير موجود بالنظام",
        });
      }

      device = await Device.findOne({ sectorId: sector._id });

      finalSectorId = sector._id;
      finalOwnerId = sector.ownerId;
      finalWorkerId = sector.assignedWorker;
      cropType = sector.cropType || "Unknown";
      uploadedBy = req.user ? req.user._id : sector.ownerId;
      captureReason = "Manual Scan";
    }
    // ⚠️ حالة 3: لو الطلب ناقص
    else {
      return res.status(400).json({
        success: false,
        message:
          "يجب تزويد السيرفر بـ deviceSerial أو معرف القطاع sectorId لإتمام العملية",
      });
    }

    // --- (جزء تحليل الـ AI) ---
    let aiAnalysis = {
      status: "Unknown",
      diseaseName: "تحليل غير متاح",
      confidence: 0,
      recommendations: ["سيرفر الـ AI لا يستجيب"],
      treatmentPlan: [],
      captureTips: [],
      ratios: { green: 0, yellow: 0, brown: 0, damaged: 0 },
      note: "",
    };

    try {
      const formData = new FormData();
      const imageResponse = await axios.get(imageUrl, {
        responseType: "stream",
      });

      formData.append("file", imageResponse.data, {
        filename: req.file.originalname || "image.jpg",
        contentType: req.file.mimetype || "image/jpeg",
      });
      formData.append("cropType", cropType);

      const aiResponse = await axios.post(
        "https://amr2004-ecosense-ai.hf.space/api/predict_image",
        formData,
        {
          headers: {
            // 🛡️ الطريقة الآمنة لطلب الـ Headers في البيئات المرفوعة سحابياً لعدم تجميد الدالة
            ...formData.getHeaders(),
            "ngrok-skip-browser-warning": "true",
          },
          timeout: 25000, // زيادة المهلة لضمان استقرار الاستجابة الكبيرة
        },
      );

      if (aiResponse.data) {
        const fullAnalysis =
          aiResponse.data.analysis || aiResponse.data.image_analysis || {};

        let conf = parseFloat(
          aiResponse.data.confidence ||
            aiResponse.data.final_confidence ||
            fullAnalysis.confidence,
        );
        conf = isNaN(conf) ? 0 : conf;

        aiAnalysis = {
          status:
            aiResponse.data.status ||
            fullAnalysis.status ||
            aiResponse.data.final_status ||
            fullAnalysis.final_status ||
            "Infected",
          diseaseName:
            aiResponse.data.disease_name_ar ||
            fullAnalysis.disease_name_ar ||
            aiResponse.data.disease_name ||
            fullAnalysis.disease_name ||
            "Severe Plant Stress",
          confidence: conf,
          recommendations:
            aiResponse.data.recommendations ||
            fullAnalysis.recommendations ||
            aiResponse.data.image_recommendations ||
            fullAnalysis.image_recommendations ||
            [],
          treatmentPlan: fullAnalysis.treatment_plan || [],
          captureTips: fullAnalysis.capture_tips || [],
          ratios: {
            green: parseFloat(fullAnalysis.green_ratio ?? 0) * 100,
            yellow: parseFloat(fullAnalysis.yellow_ratio ?? 0) * 100,
            brown: parseFloat(fullAnalysis.brown_ratio ?? 0) * 100,
            damaged: parseFloat(fullAnalysis.damaged_ratio ?? 0) * 100,
          },
          note: fullAnalysis.note || "",
        };
      }
    } catch (aiErr) {
      console.log("⚠️ Image AI Server Error:", aiErr.message);
      aiAnalysis = {
        status: "Unknown",
        diseaseName: "تعذر فحص الصورة حالياً",
        confidence: 0,
        recommendations: ["سيرفر تحليل الصور لا يستجيب، يرجى المحاولة لاحقاً."],
        treatmentPlan: [],
        captureTips: [],
        ratios: { green: 0, yellow: 0, brown: 0, damaged: 0 },
        note: "خطأ في الاتصال بسيرفر الذكاء الاصطناعي.",
      };
    }

    try {
      // حفظ السجل في الداتابيز بالإصدار المحدث والشامل
      const newImageLog = await ImageLog.create({
        ownerId: finalOwnerId,
        sectorId: finalSectorId,
        imageUrl: imageUrl,
        capturedBy: uploadedBy,
        analysisResult: aiAnalysis,
        deviceId: device ? device._id : null,
        captureReason: captureReason,
      });

      // 4️⃣ الإشعارات الفورية والـ Socket.io
      if (aiAnalysis.status !== "Healthy" && aiAnalysis.status !== "Unknown") {
        const io = req.app.get("io");
        const title = "🚨 تنبيه صحة النبات";
        const message =
          captureReason === "Manual Scan"
            ? `نتائج الفحص اليدوي: رصد (${aiAnalysis.diseaseName})`
            : `الكاميرا الآلية رصدت إصابة (${aiAnalysis.diseaseName})`;

        const owner = await User.findById(finalOwnerId);
        const worker = finalWorkerId
          ? await User.findById(finalWorkerId)
          : null;

        if (typeof sendFirebasePush === "function") {
          if (owner) await sendFirebasePush(owner, title, message);
          if (worker) await sendFirebasePush(worker, title, message);
        }

        const notificationData = {
          title,
          message,
          type: "disease",
          sectorId: finalSectorId,
        };

        if (finalOwnerId) {
          const nOwner = await Notification.create({
            ...notificationData,
            recipient: finalOwnerId,
          });
          if (io)
            io.to(finalOwnerId.toString()).emit("newNotification", nOwner);
        }

        if (finalWorkerId) {
          const nWorker = await Notification.create({
            ...notificationData,
            recipient: finalWorkerId,
          });
          if (io)
            io.to(finalWorkerId.toString()).emit("newNotification", nWorker);
        }
      }

      return res.status(201).json({ success: true, data: newImageLog });
    } catch (err) {
      console.error(
        "❌ Error in ImageLog.create inside Controller:",
        err.message,
      );
      return res.status(500).json({ success: false, error: err.message });
    }
  } catch (globalErr) {
    console.error("❌ Global Error in Controller:", globalErr.message);
    return res.status(500).json({ success: false, error: globalErr.message });
  }
};

/*======================================
    2️⃣ GET IMAGE HISTORY 
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
      const ownerSectors = await Sector.find({ ownerId: req.user._id }).select(
        "_id",
      );
      const ownerSectorIds = ownerSectors.map((s) => s._id);

      if (sectorId) {
        if (!ownerSectorIds.map((id) => id.toString()).includes(sectorId)) {
          return res
            .status(403)
            .json({ success: false, message: "هذا القطاع لا ينتمي لمزرعتك" });
        }
        filter.sectorId = sectorId;
      } else {
        filter.sectorId = { $in: ownerSectorIds };
      }
    }

    const images = await ImageLog.find(filter)
      .sort({ createdAt: -1 })
      .skip((page - 1) * Number(limit))
      .limit(Number(limit))
      .populate("sectorId", "name cropType")
      .populate("capturedBy", "firstName lastName role")
      .lean();

    const total = await ImageLog.countDocuments(filter);

    res.status(200).json({
      success: true,
      totalRecords: total,
      currentPage: Number(page),
      totalPages: Math.ceil(total / Number(limit)),
      data: images,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
    3️⃣ DELETE IMAGE LOG
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

    const urlParts = log.imageUrl.split("/");
    const fileNameWithExt = urlParts[urlParts.length - 1];
    const publicIdWithoutExt = fileNameWithExt.split(".")[0];
    const fullPublicId = `ecosense/images/${publicIdWithoutExt}`;

    await cloudinary.uploader.destroy(fullPublicId);
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
