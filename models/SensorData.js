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

    // 🌱 قراءات التربة (تم تحديثها لتناسب السنسورات المتاحة)
    soil: {
      moisture: { type: Number, default: null }, // رطوبة التربة (المتوفرة)
      temperature: { type: Number, default: null }, // حرارة التربة (ستبقى null حالياً)

      // لو قررت مستقبلاً تضيف كذا سنسور رطوبة
      individual_sensors: [{ type: Number }],
    },

    // ☀️ الإضاءة
    light: { type: String, default: null },

    // 🧠 نتائج تحليل الذكاء الاصطناعي (AI Analysis)
    analysis: {
      status: {
        type: String,
        default: "Unknown",
      },
      recommendation: { type: String },
    },
  },
  {
    // بيضيف تلقائياً وقت إنشاء القراءة (createdAt) ووقت تحديثها (updatedAt)
    timestamps: true,
  },
);

// تحسين أداء البحث للـ Dashboard والتقارير
SensorDataSchema.index({ ownerId: 1, sectorId: 1, createdAt: -1 });

module.exports = mongoose.model("SensorData", SensorDataSchema);
