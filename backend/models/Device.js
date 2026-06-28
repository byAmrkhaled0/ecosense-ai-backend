const mongoose = require("mongoose");

const deviceSchema = new mongoose.Schema(
  {
    deviceSerial: {
      type: String,
      required: true,
      unique: true, // رقم التسلسل الفريد للجهاز
    },
    deviceName: { type: String, default: "Smart Node" },

    status: {
      type: String,
      enum: ["online", "offline", "maintenance"],
      default: "offline",
    },

    lastPing: Date, // آخر وقت الجهاز بعت فيه داتا

    // الجهاز ده محطوط في أنهي قطاع؟
    sectorId: {
      // خليناها sectorId بدل sector للتوحيد
      type: mongoose.Schema.Types.ObjectId,
      ref: "Sector",
      required: true, // الجهاز لازم يكون مربوط بقطاع عشان الداتا تتسجل صح
    },

    ownerId: {
      // خليناها ownerId بدل owner
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
  },
  { timestamps: true },
);

module.exports = mongoose.model("Device", deviceSchema);
