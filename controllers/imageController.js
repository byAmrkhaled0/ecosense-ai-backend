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

    // 1️⃣ تعريف المتغيرات بره الـ if عشان تكون مرئية للكود كله
    let finalSectorId, finalOwnerId, finalWorkerId, uploadedBy, device;

    if (!req.file) {
      return res.status(400).json({ success: false, message: "يرجى رفع صورة" });
    }

    // --- حالة 1: الرفع عن طريق جهاز IoT (Serial) ---
    if (deviceSerial) {
      device = await Device.findOne({ deviceSerial }).populate("sectorId");
      if (!device || !device.sectorId) {
        return res
          .status(404)
          .json({ success: false, message: "الجهاز غير مربوط بقطاع" });
      }
      finalSectorId = device.sectorId._id;
      finalOwnerId = device.sectorId.ownerId;
      finalWorkerId = device.sectorId.assignedWorker;
      uploadedBy = null;
    }

    // --- حالة 2: الرفع يدوي (Token) ---
    else if (req.user) {
      const targetSectorId = sectorId; // تأكد إنك باعت الـ id في الـ body
      if (!targetSectorId)
        return res
          .status(400)
          .json({ success: false, message: "sectorId مطلوب" });

      const sector = await Sector.findById(targetSectorId);
      if (!sector)
        return res
          .status(404)
          .json({ success: false, message: "القطاع غير موجود" });

      finalSectorId = sector._id;
      finalOwnerId = sector.ownerId;
      finalWorkerId = sector.assignedWorker;
      uploadedBy = req.user._id;
    } else {
      return res
        .status(401)
        .json({ success: false, message: "يجب توفير Serial أو Token" });
    }

    // 2️⃣ دلوقتي المتغيرات دي (finalOwnerId, etc) بقت متشافة هنا
    const imageUrl = `${req.protocol}://${req.get("host")}/uploads/${req.file.filename}`;

    // تحليل الـ AI (كود الـ axios بتاعك هنا)
    let aiAnalysis = {
      status: "Healthy",
      diseaseName: "None",
      confidence: 0,
      recommendation: "النبات سليم",
    };

    // 3️⃣ حفظ السجل في الداتابيز
    const newImageLog = await ImageLog.create({
      ownerId: finalOwnerId,
      sectorId: finalSectorId,
      imageUrl: imageUrl,
      capturedBy: uploadedBy || finalOwnerId,
      analysisResult: aiAnalysis,
      deviceId: device?._id, // الـ Optional Chaining مهم هنا
    });

    // 4️⃣ الإشعارات (إرسال للمالك والعامل)
    if (aiAnalysis.status !== "Healthy") {
      const io = req.app.get("io");
      const notificationData = {
        title: "🚨 تنبيه صحة النبات",
        message: `تم رصد إصابة (${aiAnalysis.diseaseName}) في قطاعك.`,
        type: "disease",
        sectorId: finalSectorId,
      };

      // المالك
      const nOwner = await Notification.create({
        ...notificationData,
        recipient: finalOwnerId,
      });
      io.to(finalOwnerId.toString()).emit("newNotification", nOwner);

      // العامل
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
    // لو حصل Error هنا هيرجع رسالة واضحة
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
