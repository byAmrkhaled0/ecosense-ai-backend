const mongoose = require("mongoose");

const SensorDataSchema = new mongoose.Schema(
  {
    ownerId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },
    sectorId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Sector",
      required: true,
      index: true,
    },
    deviceId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Device",
      required: true,
      index: true,
    },

    // 👇 التعديل هنا: نقطتين فوق بعض وليس يساوي
    air: {
      temperature: { type: Number, default: null },
      humidity: { type: Number, default: null },
    },

    soil: {
      moisture: { type: Number, default: null },
      temperature: { type: Number, default: null },
      individual_sensors: [{ type: Number }],
    },
    light: { type: String, default: null },

    // 🧠 نتائج تحليل الذكاء الاصطناعي الشاملة والمحدثة
    analysis: {
      status: { type: String, default: "Unknown" },
      recommendation: { type: String }, // الحفاظ عليه للفرونت إند الحالي كـ string مدمج

      final_status: { type: String },
      final_confidence: { type: Number },
      general_recommendation: { type: String },

      // التوصيات كـ مصفوفة منفصلة
      recommendations: [{ type: String }],

      // المصفوفات الديناميكية للأفعال وعوامل الخطورة
      actions: [
        {
          code: { type: String },
          title: { type: String },
          details: { type: String },
          priority: { type: Number },
        },
      ],
      risk_factors: [
        {
          code: { type: String },
          label: { type: String },
          value: { type: Number },
          ideal_range: { type: String },
          severity: { type: String },
        },
      ],

      // طبقة الحماية والإشعارات الجاهزة من الـ AI
      safety_layer: {
        applied: { type: Boolean, default: false },
        sensor_model_status: { type: String },
        status_after_safety: { type: String },
        flags: [{ type: String }],
      },
      notification: {
        send: { type: Boolean, default: false },
        title: { type: String },
        message: { type: String },
        type: { type: String },
      },
      timestamp: { type: String },
    },
  },
  {
    timestamps: true,
  },
);

SensorDataSchema.index({ ownerId: 1, sectorId: 1, createdAt: -1 });

module.exports = mongoose.model("SensorData", SensorDataSchema);
