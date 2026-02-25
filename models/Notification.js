const mongoose = require("mongoose");

const notificationSchema = new mongoose.Schema({
  // المستخدم المستلم للإشعار (صاحب المزرعة أو العامل)
  recipient: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
    index: true, // إضافة index للسرعة في جلب إشعارات مستخدم معين
  },
  title: {
    type: String,
    required: true,
  },
  message: {
    type: String,
    required: true,
  },
  type: {
    type: String,
    // تم إضافة 'alert' و 'success' للقائمة عشان الكود ميضربش
    enum: ["info", "warning", "critical", "alert", "success"],
    default: "info",
  },
  isRead: {
    type: Boolean,
    default: false,
  },
  // ربط الإشعار بقطاع معين (اختياري)
  sectorId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Sector",
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

// ترتيب الإشعارات من الأحدث للأقدم تلقائياً
notificationSchema.index({ createdAt: -1 });

module.exports = mongoose.model("Notification", notificationSchema);
