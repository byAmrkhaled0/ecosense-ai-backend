const mongoose = require("mongoose");

const ImageLogSchema = new mongoose.Schema(
  {
    // 👤 الشخص اللي صور (سواء صاحب المزرعة أو العامل)
    capturedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },

    // 👨‍🌾 صاحب المزرعة الأصلي (المالك الأساسي للداتا)
    ownerId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },

    // 📍 القطاع اللي الصورة اتأخدت فيه
    sectorId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Sector",
      required: true,
      index: true,
    },

    // 🖼️ رابط الصورة على السيرفر
    imageUrl: {
      type: String,
      required: [true, "Image URL is required"],
    },

    // 🧠 نتائج تحليل الذكاء الاصطناعي (AI Analysis)
    analysisResult: {
      status: {
        type: String,
        // تم تحديث القائمة لتشمل حالات المعالجة لضمان مرونة النظام
        enum: ["Healthy", "Infected", "Pending", "Processing", "Unknown"],
        default: "Pending",
      },
      diseaseName: {
        type: String,
        default: "None",
      }, // اسم المرض المكتشف (مثلاً: Tomato Blight)
      confidence: {
        type: Number,
        default: 0,
      }, // نسبة التأكد من 0 لـ 100
      recommendation: {
        type: String,
        default: null,
      },
    },

    // ❓ سبب التقاط الصورة
    captureReason: {
      type: String,
      enum: {
        values: ["Routine", "Alert", "Scheduled", "Manual Scan", "Unknown"], // أضفنا Manual Scan هنا
        message: "{VALUE} غير مدعوم في سبب الالتقاط",
      },
      default: "Manual Scan",
    },

    // 🔗 ربط الصورة بآخر قراءة مستشعر (عشان نربط الأرقام بالصور)
    relatedReadingId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "SensorData",
      default: null,
    },

    // 🔌 الجهاز اللي التقط الصورة (لو فيه كاميرا ESP32-Cam مثلاً)
    deviceId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Device",
      default: null,
    },
  },
  {
    // بيعمل createdAt و updatedAt تلقائياً
    timestamps: true,
  },
);

// تحسين البحث للداشبورد (البحث بالمالك والقطاع مع ترتيب زمني تنازلي)
ImageLogSchema.index({ ownerId: 1, sectorId: 1, createdAt: -1 });

module.exports = mongoose.model("ImageLog", ImageLogSchema);
