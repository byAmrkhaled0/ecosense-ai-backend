const User = require("../models/User");
const Sector = require("../models/Sector"); // استيراد موديل القطاعات

// جلب العمال التابعين للمالك
exports.getMyWorkers = async (req, res) => {
  try {
    const workers = await User.find({
      ownerId: req.user._id,
      role: "worker",
    }).select("-password"); // نصيحة: بلاش تبعت الـ Password حتى لو مشفر في الـ API

    res.status(200).json({
      success: true,
      count: workers.length,
      data: workers,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// حذف عامل مع تنظيف القطاعات المرتبطة به
exports.deleteWorker = async (req, res) => {
  try {
    const workerId = req.params.id;

    const worker = await User.findOne({
      _id: workerId,
      ownerId: req.user._id,
    });

    if (!worker) {
      return res
        .status(404)
        .json({ success: false, message: "العامل غير موجود" });
    }

    // 🚩 الخطوة الإضافية: تصفير حقل العامل في أي قطاع كان مسؤول عنه
    await Sector.updateMany(
      { assignedWorker: workerId },
      { $set: { assignedWorker: null } },
    );

    await worker.deleteOne();

    res.status(200).json({
      success: true,
      message: "تم حذف العامل وتحديث القطاعات المرتبطة به بنجاح",
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
