const ImageLog = require("../models/ImageLog");
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");
const Sector = require("../models/Sector");
const mongoose = require("mongoose");

/* ============================================================
    1️⃣ UPLOAD & AI ANALYZE (رفع الصورة وتحليلها)
============================================================ */
exports.uploadImage = async (req, res) => {
  try {
    const { sectorId, captureReason } = req.body;
    const userId = req.user._id;

    // 1. التأكد من وجود الصورة المرفوعة
    if (!req.file) {
      return res
        .status(400)
        .json({ success: false, message: "يرجى رفع صورة للتحليل" });
    }

    // 2. التحقق من وجود القطاع (هنا كان بيحصل الخطأ)
    const sector = await Sector.findById(sectorId);
    if (!sector) {
      return res
        .status(404)
        .json({ success: false, message: "القطاع المحدد غير موجود" });
    }

    // 3. تجهيز رابط الصورة للرد به (Full URL)
    const imageUrl = `${req.protocol}://${req.get("host")}/uploads/${req.file.filename}`;

    // 4. إعداد بيانات التحليل (افتراضية في حال تعذر الاتصال بسيرفر AI)
    let aiAnalysis = {
      status: "Unknown",
      diseaseName: "None",
      confidence: 0,
      recommendation: "سيرفر التحليل غير متاح حالياً",
    };

    // 5. محاولة الاتصال بسيرفر الـ AI
    try {
      const formData = new FormData();
      formData.append("image", fs.createReadStream(req.file.path));
      formData.append("cropType", sector.cropType);

      const aiResponse = await axios.post(
        process.env.AI_IMAGE_SERVER_URL || "http://127.0.0.1:8000/predict",
        formData,
        { headers: formData.getHeaders() },
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
      console.error("⚠️ AI Server Error:", aiErr.message);
    }

    // 6. حفظ السجل في قاعدة البيانات
    const newImageLog = await ImageLog.create({
      ownerId: userId,
      capturedBy: userId,
      sectorId: sectorId,
      imageUrl: imageUrl,
      captureReason: captureReason || "Manual Scan",
      analysisResult: aiAnalysis,
    });

    // 7. جلب البيانات المحدثة مع عمل Populate لاسم القطاع
    const finalLog = await ImageLog.findById(newImageLog._id).populate(
      "sectorId",
      "name cropType",
    );

    res.status(201).json({
      success: true,
      message: "✅ تم الرفع والتحليل بنجاح",
      data: finalLog,
    });
  } catch (err) {
    // في حالة حدوث أي خطأ آخر
    res.status(500).json({ success: false, error: err.message });
  }
};
/* ============================================================
    2️⃣ GET IMAGE HISTORY (عرض التاريخ بشكل منظم)
============================================================ */

exports.getImageHistory = async (req, res) => {
  try {
    const { sectorId, page = 1, limit = 10 } = req.query;

    // 1. تحديد الفلتر الأساسي بناءً على الصلاحيات
    let filter = {};

    if (req.user.role === "worker") {
      // العامل يشوف قطاعه فقط
      filter.sectorId = req.user.assignedSector;
    } else {
      // المالك يشوف حاجته كلها
      filter.ownerId = req.user._id;
    }

    // 2. 🚩 التعديل الجوهري: لو المالك باعت sectorId معين في الـ Query
    if (sectorId) {
      // بنحول الـ String لـ ObjectId عشان نضمن إن الـ Match يتم صح في الداتابيز
      filter.sectorId = new mongoose.Types.ObjectId(sectorId);
    }

    const images = await ImageLog.find(filter)
      .sort({ createdAt: -1 })
      .skip((page - 1) * Number(limit))
      .limit(Number(limit))
      .populate("sectorId", "name cropType")
      .populate("capturedBy", "firstName lastName")
      .lean();

    // 3. تنسيق الرد
    const formattedData = images.map((img) => ({
      _id: img._id,
      url: img.imageUrl,
      capturedAt: img.createdAt,
      sector: img.sectorId, // هيرجع كائن فيه الـ name و الـ cropType بسبب الـ populate
      info: {
        by: img.capturedBy,
        reason: img.captureReason,
      },
      analysis: img.analysisResult,
    }));

    const total = await ImageLog.countDocuments(filter);

    res.status(200).json({
      success: true,
      totalRecords: total,
      currentPage: Number(page),
      totalPages: Math.ceil(total / Number(limit)),
      data: formattedData,
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
