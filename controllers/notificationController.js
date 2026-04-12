const Notification = require("../models/Notification");

// @desc    جلب تنبيهات المستخدم الحالي مع تفاصيل القطاع
exports.getNotifications = async (req, res) => {
  try {
    let query = {};

    // تحديد الفلتر بناءً على دور المستخدم
    if (req.user.role === "worker") {
      // العامل يرى تنبيهات القطاع المسؤول عنه فقط
      // تأكد أن "assignedSector" مخزن في req.user من خلال الـ Auth Middleware
      if (!req.user.assignedSector) {
        return res.status(200).json({ success: true, count: 0, data: [] });
      }
      query = { sectorId: req.user.assignedSector };
    } else {
      // المالك يرى كل التنبيهات الموجهة له
      query = { recipient: req.user._id };
    }

    const notifications = await Notification.find(query)
      .sort("-createdAt") // الأحدث يظهر أولاً
      .limit(30) // زودنا الليميت شوية للاحتياط
      .populate({
        path: "sectorId",
        select: "name cropType location", // هنا بنجلب اسم القطاع ونوع المحصول ومكانه
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
