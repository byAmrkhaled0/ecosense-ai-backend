const mongoose = require("mongoose");

const sectorSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: [true, "يرجى إدخال اسم القطاع"],
      trim: true,
    },
    description: String,

    cropType: {
      type: String,
      required: [true, "يرجى تحديد نوع المحصول"],
    },

    area: {
      type: Number,
      default: 0,
    },

    location: {
      type: String,
      default: "غير محدد",
    },

    season: {
      type: String,
    },

    // صاحب المزرعة
    ownerId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },

    // العامل المسؤول
    assignedWorker: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      default: null,
    },

    healthStatus: {
      type: String,
      enum: ["Healthy", "Warning", "Critical"],
      default: "Healthy",
    },
  },
  {
    timestamps: true,
  },
);

/* ⚠️ تصحيح الـ Indexes: 
إنت كنت كاتب owner: 1 والحقل عندك اسمه ownerId
*/
sectorSchema.index({ ownerId: 1 }); // تم التصحيح
sectorSchema.index({ assignedWorker: 1 });

module.exports = mongoose.model("Sector", sectorSchema);
