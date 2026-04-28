const mongoose = require("mongoose");

const ImageLogSchema = new mongoose.Schema(
  {
    capturedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
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
      status: String,
      diseaseName: String,
      confidence: Number,
      recommendation: String,
      // تأكد من تطابق هذه الأسماء مع ما يتم حفظه في الـ Controller
      greenRatio: Number,
      yellowRatio: Number,
      brownRatio: Number,
      healthScore: Number,
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
