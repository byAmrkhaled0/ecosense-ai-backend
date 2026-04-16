const ImageLog = require("../models/ImageLog");
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");
const Sector = require("../models/Sector");
const Notification = require("../models/Notification"); // ضفنا الموديل ده
const mongoose = require("mongoose");

/* ============================================================
    1️⃣ UPLOAD & AI ANALYZE (رفع الصورة وتحليلها)
============================================================ */
exports.uploadImage = async (req, res) => {
  try {
    const { sectorId, captureReason } = req.body;
    const userId = req.user._id;

    if (!req.file) {
      return res
        .status(400)
        .json({ success: false, message: "يرجى رفع صورة للتحليل" });
    }

    // 1. التأكد من أن القطاع يخص المستخدم (عامل أو مالك) 🚩 (الأمان)
    const sector = await Sector.findById(sectorId);
    if (!sector) {
      return res
        .status(404)
        .json({ success: false, message: "القطاع غير موجود" });
    }

    if (
      req.user.role === "worker" &&
      sector.assignedWorker.toString() !== userId.toString()
    ) {
      return res.status(403).json({
        success: false,
        message: "عذراً، هذا القطاع ليس تحت مسؤوليتك",
      });
    }

    if (
      req.user.role === "owner" &&
      sector.ownerId.toString() !== userId.toString()
    ) {
      return res
        .status(403)
        .json({ success: false, message: "هذا القطاع لا ينتمي لمزرعتك" });
    }

    const imageUrl = `${req.protocol}://${req.get("host")}/uploads/${req.file.filename}`;

    // 2. محاولة الاتصال بسيرفر الـ AI (نفس كودك)
    let aiAnalysis = {
      status: "Healthy",
      diseaseName: "None",
      confidence: 0,
      recommendation: "النبات سليم",
    };
    try {
      const formData = new FormData();
      formData.append("image", fs.createReadStream(req.file.path));
      formData.append("cropType", sector.cropType);

      const aiResponse = await axios.post(
        process.env.AI_IMAGE_SERVER_URL || "http://127.0.0.1:8000/predict",
        formData,
        { headers: formData.getHeaders(), timeout: 5000 },
      );

      if (aiResponse.data) {
        aiAnalysis = {
          status: aiResponse.data.status || "Healthy",
          diseaseName: aiResponse.data.disease || "None",
          confidence: aiResponse.data.confidence || 0,
          recommendation: aiResponse.data.recommendation || "لا توجد توصيات",
        };
      }
    } catch (aiErr) {
      console.error("⚠️ AI Image Server Error");
    }

    // 3. حفظ السجل
    const newImageLog = await ImageLog.create({
      ownerId: sector.ownerId, // نضمن إن صاحب المزرعة هو المالك هنا
      capturedBy: userId,
      sectorId: sectorId,
      imageUrl: imageUrl,
      captureReason: captureReason || "Manual Scan",
      analysisResult: aiAnalysis,
    });

    // 4. 🔔 إرسال إشعار فوري (Socket.io) لو فيه مرض 🚩
    if (aiAnalysis.status !== "Healthy") {
      const io = req.app.get("io");
      const notificationData = {
        title: "⚠️ اكتشاف إصابة نبات",
        message: `تم رصد (${aiAnalysis.diseaseName}) في قطاع (${sector.name}).`,
        type: "disease",
        sectorId: sectorId,
      };

      // حفظ الإشعارات في الداتابيز وإرسالها عبر Socket
      const notifyOwner = await Notification.create({
        ...notificationData,
        recipient: sector.ownerId,
      });
      io.to(sector.ownerId.toString()).emit("newNotification", notifyOwner);

      if (sector.assignedWorker) {
        const notifyWorker = await Notification.create({
          ...notificationData,
          recipient: sector.assignedWorker,
        });
        io.to(sector.assignedWorker.toString()).emit(
          "newNotification",
          notifyWorker,
        );
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
