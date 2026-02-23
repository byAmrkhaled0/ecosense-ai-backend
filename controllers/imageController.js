const ImageLog = require("../models/ImageLog");
const fs = require("fs");

/* ============================================================
    1️⃣ UPLOAD IMAGE (رفع الصورة وربطها بالقطاع)
    POST /api/images/upload
============================================================ */
exports.uploadImage = async (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      message: "يرجى اختيار صورة لرفعها.",
    });
  }

  try {
    const { sectorId, captureReason, relatedReadingId } = req.body;

    // التحقق من وجود معرف القطاع (إلزامي في الهيكلية الجديدة)
    if (!sectorId) {
      return res.status(400).json({
        success: false,
        message: "يجب تحديد القطاع التابع له هذه الصورة.",
      });
    }

    const baseUrl = `${req.protocol}://${req.get("host")}`;
    const finalFileUrl = `${baseUrl}/uploads/${req.file.filename}`;

    // تحديد مين المالك (Owner) بناءً على دور المستخدم الحالي
    const ownerId = req.user.role === "owner" ? req.user._id : req.user.ownerId;
    if (
      req.user.role === "worker" &&
      req.body.sectorId !== req.user.assignedSector.toString()
    ) {
      return res
        .status(403)
        .json({ message: "لا يمكنك رفع صور لقطاع لست مسؤولاً عنه" });
    }
    // 💾 حفظ سجل الصورة
    const newImageLog = await ImageLog.create({
      capturedBy: req.user._id, // من قام بالتصوير (صاحب أو عامل)
      ownerId: ownerId, // المالك الأساسي للمزرعة
      sectorId: sectorId, // القطاع المستهدف
      imageUrl: finalFileUrl,
      captureReason: captureReason || "Manual",
      relatedReadingId: relatedReadingId || null,
    });

    res.status(201).json({
      success: true,
      message: "✅ تم رفع الصورة بنجاح وربطها بالقطاع.",
      data: newImageLog,
    });
  } catch (err) {
    console.error("Image Upload Error:", err.message);
    // حذف الملف لو حصل خطأ في الداتابيز
    if (req.file && req.file.path) {
      fs.unlink(req.file.path, (e) => {});
    }
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
    2️⃣ GET IMAGE HISTORY (عرض صور القطاع أو المزرعة)
    GET /api/images/history?sectorId=xxx
============================================================ */
exports.getImageHistory = async (req, res) => {
  try {
    // 1. استلام بارامترات البحث والترتيب والصفحات من الـ Query
    const {
      sectorId,
      capturedBy, // فلتر باللي صور الصورة (ID)
      status, // لو فيه تحليل للصور (مثل Healthy, Infected)
      startDate,
      endDate,
      sortBy = "createdAt", // الترتيب الافتراضي حسب تاريخ الرفع
      order = "desc",
      page = 1,
      limit = 10,
    } = req.query;

    // 2. بناء الفلتر (Search Logic)
    let filter = {};

    // أمان: لو عامل يشوف صور قطاعه بس، لو مالك يشوف مزارعه
    if (req.user.role === "worker") {
      filter.sectorId = req.user.assignedSector;
    } else {
      filter.ownerId = req.user._id;
      if (sectorId) filter.sectorId = sectorId; // فلتر بقطاع محدد للمالك
    }

    // فلتر باللي صور الصورة
    if (capturedBy) filter.capturedBy = capturedBy;
    // فلتر بحالة تحليل الصورة (لو فيه AI بيحلل الصور)
    if (status) filter["analysisResult.status"] = status;

    // فلتر بالتاريخ (Search by Date Range)
    if (startDate || endDate) {
      filter.createdAt = {};
      if (startDate) filter.createdAt.$gte = new Date(startDate);
      if (endDate) filter.createdAt.$lte = new Date(endDate);
    }

    // 3. الترتيب (Sorting Logic)
    const sortOrder = order === "desc" ? -1 : 1;
    const sortOptions = {};
    sortOptions[sortBy] = sortOrder;

    // 4. التنفيذ مع الـ Pagination
    const images = await Image.find(filter)
      .sort(sortOptions)
      .skip((Number(page) - 1) * Number(limit))
      .limit(Number(limit))
      .populate("sectorId", "name cropType")
      .populate("capturedBy", "firstName lastName"); // عشان يظهرلك مين اللي صور

    // 5. حساب إجمالي النتائج والصفحات
    const total = await Image.countDocuments(filter);
    const totalPages = Math.ceil(total / Number(limit));

    res.status(200).json({
      success: true,
      results: images.length,
      totalRecords: total,
      totalPages: totalPages,
      currentPage: Number(page),
      data: images,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
/* ============================================================
    3️⃣ UPDATE IMAGE ANALYSIS (تحديث نتيجة تحليل الـ AI للصور)
    PATCH /api/images/:id/analyze
============================================================ */
exports.updateAnalysis = async (req, res) => {
  try {
    const { status, diseaseName, confidence } = req.body;

    const log = await ImageLog.findByIdAndUpdate(
      req.params.id,
      {
        analysisResult: {
          status,
          diseaseName,
          confidence,
        },
      },
      { new: true },
    );

    if (!log) return res.status(404).json({ message: "السجل غير موجود." });

    res.status(200).json({ success: true, data: log });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

/* ============================================================
    4️⃣ DELETE IMAGE LOG
============================================================ */
exports.deleteImageLog = async (req, res) => {
  try {
    const log = await ImageLog.findOne({
      _id: req.params.id,
      ownerId: req.user.role === "owner" ? req.user._id : req.user.ownerId,
    });

    if (!log)
      return res.status(404).json({ message: "لم يتم العثور على الصورة." });

    // حذف الملف الفعلي من السيرفر
    const filePath = `./uploads/${log.imageUrl.split("/").pop()}`;
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);

    await log.deleteOne();

    res
      .status(200)
      .json({ success: true, message: "✅ تم حذف الصورة وسجلها بنجاح." });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
