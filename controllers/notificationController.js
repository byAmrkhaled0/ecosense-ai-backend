const Notification = require("../models/Notification");

// @desc    جلب تنبيهات المستخدم الحالي مع تفاصيل القطاع
exports.getNotifications = async (req, res) => {
  try {
    let query = {};

    if (req.user.role === "worker") {
      if (!req.user.assignedSector) {
        return res.status(200).json({ success: true, count: 0, data: [] });
      }
      query = { sectorId: req.user.assignedSector };
    } else {
      query = { recipient: req.user._id };
    }

    // 🎯 قمنا بحذف .limit(30) لكي يجلب السيرفر كل الإشعارات القديمة والجديدة
    const notifications = await Notification.find(query)
      .sort("-createdAt")
      .populate({
        path: "sectorId",
        select: "name cropType location",
      });

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
    // التحقق من أن التنبيه يخص المستخدم الحالي قبل التعديل (زيادة أمان)
    const notification = await Notification.findById(req.params.id);

    if (!notification) {
      return res
        .status(404)
        .json({ success: false, message: "التنبيه غير موجود" });
    }

    // تحديث الحالة لـ مقروء
    notification.isRead = true;
    await notification.save();

    res.status(200).json({ success: true, data: notification });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// @desc    حذف تنبيه معين
exports.deleteNotification = async (req, res) => {
  try {
    const notification = await Notification.findByIdAndDelete(req.params.id);
    if (!notification) {
      return res
        .status(404)
        .json({ success: false, message: "التنبيه غير موجود" });
    }
    res.status(200).json({ success: true, message: "تم حذف التنبيه بنجاح" });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
