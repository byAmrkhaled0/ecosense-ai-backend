const mongoose = require("mongoose");

/**
 * ImageLog Model
 * سجل صور المحاصيل وتحليلات الذكاء الاصطناعي
 * يدعم الفحص الآلي (عبر الكاميرات) والفحص اليدوي (عبر تطبيق الموبايل)
 */
const ImageLogSchema = new mongoose.Schema(
  {
    // 👤 الشخص الذي قام بالالتقاط (صاحب المزرعة أو العامل)
    capturedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },

    // 👨‍🌾 صاحب المزرعة (المالك الأساسي للبيانات)
    ownerId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },

    // 📍 القطاع الذي تمت فيه عملية التصوير
    sectorId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Sector",
      required: true,
      index: true,
    },

    // 🖼️ رابط الصورة المخزنة على السيرفر
    imageUrl: {
      type: String,
      required: [true, "Image URL is required"],
    },

    // 🧠 نتائج تحليل الذكاء الاصطناعي (AI Analysis)
    // أضف هذه الحقول داخل analysisResult في ملف الموديل (ImageLog.js)
    analysisResult: {
      status: String,
      diseaseName: String,
      confidence: Number,
      recommendation: String,
      // الحقول الجديدة 👇
      greenRatio: Number,
      yellowRatio: Number,
      brownRatio: Number,
      healthScore: Number,
    },

    // ❓ سبب أو طريقة التقاط الصورة (مهم جداً للتقارير)
    captureReason: {
      type: String,
      enum: {
        values: [
          "Routine",
          "Alert",
          "Scheduled",
          "Manual Scan",
          "Automatic Camera",
          "Unknown",
        ],
        message: "{VALUE} غير مدعوم في سبب الالتقاط",
      },
      default: "Manual Scan",
    },

    // 🔗 ربط الصورة بآخر قراءة مستشعر (لربط الحالة البيئية بحالة النبات)
    relatedReadingId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "SensorData",
      default: null,
    },

    // 🔌 الجهاز الذي التقط الصورة (في حالة استخدام ESP32-CAM)
    deviceId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Device",
      default: null,
    },
  },
  {
    // إضافة timestamps لتسجيل وقت الإنشاء والتعديل تلقائياً
    timestamps: true,
  },
);

// --- الفهارس (Indexes) لتحسين أداء البحث في الداشبورد ---
// ترتيب زمني تنازلي للصور الخاصة بمالك معين في قطاع معين
ImageLogSchema.index({ ownerId: 1, sectorId: 1, createdAt: -1 });

module.exports = mongoose.model("ImageLog", ImageLogSchema);
