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
    sector: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Sector",
    },

    owner: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
    },
  },
  { timestamps: true },
);

module.exports = mongoose.model("Device", deviceSchema);
