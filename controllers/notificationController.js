const Notification = require("../models/Notification");

// @desc    جلب تنبيهات المستخدم الحالي
exports.getNotifications = async (req, res) => {
  try {
    let query = {};

    // التعديل هنا: السيرفر لازم يدور بكلمة recipient
    // لأن ده الاسم اللي احنا عرفناه في الموديل
    if (req.user.role === "worker") {
      // العامل يشوف التنبيهات اللي تخص قطاعه
      query = { sectorId: req.user.assignedSector };
    } else {
      // المالك يشوف التنبيهات اللي هو المستلم بتاعها
      query = { recipient: req.user._id };
    }

    const notifications = await Notification.find(query)
      .sort("-createdAt") // الأحدث فوق
      .limit(20)
      .populate("sectorId", "name");

    res.status(200).json({
      success: true,
      count: notifications.length,
      data: notifications,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// @desc    تحديد التنبيه كـ "مقروء"
exports.markAsRead = async (req, res) => {
  try {
    const notification = await Notification.findByIdAndUpdate(
      req.params.id,
      { isRead: true },
      { new: true, runValidators: true },
    );

    if (!notification)
      return res
        .status(404)
        .json({ success: false, message: "التنبيه غير موجود" });

    res.status(200).json({ success: true, data: notification });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
