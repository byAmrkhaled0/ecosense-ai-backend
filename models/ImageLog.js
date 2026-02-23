const mongoose = require("mongoose");

const ImageLogSchema = new mongoose.Schema(
  {
    // ✅ الشخص اللي صور (سواء صاحب المزرعة أو العامل)
    capturedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },
    // ✅ صاحب المزرعة الأصلي (عشان يظهر في الداشبورد بتاعته)
    ownerId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },
    // ✅ القطاع اللي الصورة اتأخدت فيه (عشان نعرف نوع الزرعة)
    sectorId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Sector",
      required: true,
    },
    // ✅ رابط الصورة (Cloudinary / S3 / Local)
    imageUrl: {
      type: String,
      required: [true, "Image URL is required"],
    },
    // ✅ تحليل الذكاء الاصطناعي (اكتشاف الأمراض)
    analysisResult: {
      status: {
        type: String,
        enum: ["Healthy", "Infected", "Pending"],
        default: "Pending",
      },
      diseaseName: { type: String, default: "None" }, // اسم المرض المكتشف
      confidence: { type: Number, default: 0 }, // نسبة التأكد (0-100)
    },
    // السبب وراء التقاط الصورة
    captureReason: {
      type: String,
      enum: ["Scheduled", "Warning_Trigger", "Manual"],
      default: "Manual",
    },
    // ربط الصورة بآخر قراءة مستشعر مسجلة
    relatedReadingId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "SensorData",
      default: null,
    },
    // الجهاز اللي التقط الصورة (لو كاميرا ثابتة مثلاً)
    deviceId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Device",
      default: null,
    },
  },
  {
    timestamps: true, // بيغنينا عن حقل الـ timestamp اليدوي لأنه بيعمل createdAt و updatedAt تلقائياً
  },
);

// تحسين سرعة البحث باليوزر والوقت والقطاع
ImageLogSchema.index({ ownerId: 1, sectorId: 1, createdAt: -1 });

module.exports = mongoose.model("ImageLog", ImageLogSchema);
