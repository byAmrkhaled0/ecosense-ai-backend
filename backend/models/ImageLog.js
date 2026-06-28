const mongoose = require("mongoose");

const ImageLogSchema = new mongoose.Schema(
  {
    capturedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: false, // تم تعديلها لتسمح بـ null في حالة الكاميرات التلقائية (Automatic Camera)
      index: true,
    },
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
    imageUrl: {
      type: String,
      required: [true, "Image URL is required"],
    },
    analysisResult: {
      status: { type: String, default: "Unknown" },
      diseaseName: { type: String, default: "Severe Plant Stress" },
      confidence: { type: Number, default: 0 },
      recommendations: [{ type: String }],
      treatmentPlan: [
        {
          priority: Number,
          title: String,
          details: String,
        },
      ],
      captureTips: [{ type: String }],
      ratios: {
        green: { type: Number, default: 0 },
        yellow: { type: Number, default: 0 },
        brown: { type: Number, default: 0 },
        damaged: { type: Number, default: 0 },
      },
      note: { type: String, default: "" },
    },
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
    relatedReadingId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "SensorData",
      default: null,
    },
    deviceId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Device",
      default: null,
    },
  },
  { timestamps: true },
);

ImageLogSchema.index({ ownerId: 1, sectorId: 1, createdAt: -1 });

module.exports = mongoose.model("ImageLog", ImageLogSchema);
