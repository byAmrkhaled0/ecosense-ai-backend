const mongoose = require("mongoose");

const SensorDataSchema = new mongoose.Schema(
  {
    // 👤 صاحب المزرعة (المالك الأساسي للداتا)
    ownerId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },

    // 📍 القطاع (عشان الـ AI يعرف نوع الزرعة والموسم ويحلل صح)
    sectorId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Sector",
      required: true,
      index: true,
    },

    // 🔌 الجهاز (عشان نعرف أنهي Kit اللي بعتت القراءة دي)
    deviceId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Device",
      required: true,
      index: true,
    },

    // 🌡️ قراءات الهواء
    air: {
      temperature: { type: Number, default: null },
      humidity: { type: Number, default: null },
    },

    // 🌱 قراءات التربة
    soil: {
      moisture: { type: Number, default: null }, // رطوبة التربة
      temperature: { type: Number, default: null }, // حرارة التربة
      ph: { type: Number, default: null }, // حموضة التربة
    },

    // ☀️ الإضاءة
    light: { type: Number, default: null },

    // 🧠 نتائج تحليل الذكاء الاصطناعي (AI Analysis)
    analysis: {
      status: {
        type: String,
        enum: ["Healthy", "Warning", "Critical", "Processing"],
        default: "Processing",
      },
      recommendation: { type: String, default: null }, // نصيحة الـ AI (مثلاً: قلل الري)
    },
  },
  {
    // timestamps: true بتضيف لنا createdAt و updatedAt تلقائياً
    // وده بيغنينا عن حقل timestamp اليدوي وبيكون أدق
    timestamps: true,
  },
);

// تحسين البحث للـ Dashboard: جلب أحدث قراءات لقطاع معين تبع صاحب مزرعة معين
SensorDataSchema.index({ ownerId: 1, sectorId: 1, createdAt: -1 });

module.exports = mongoose.model("SensorData", SensorDataSchema);
