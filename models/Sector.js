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

    // المساحة (مهمة للحسابات الزراعية والداشبورد)
    area: {
      type: Number, // بالمتر المربع أو الفدان
      default: 0,
    },

    // الموقع (إحداثيات أو وصف مكاني)
    location: {
      type: String,
      default: "غير محدد",
    },

    season: {
      type: String,
    },

    // صاحب المزرعة (المالك)
    owner: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },

    // العامل المسؤول عن هذا القطاع
    // الربط ده هو اللي بيخلينا نفلتر الداشبورد للعامل
    assignedWorker: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      default: null,
    },

    // حالة القطاع (هل فيه مشكلة حالياً؟)
    healthStatus: {
      type: String,
      enum: ["Healthy", "Warning", "Critical"],
      default: "Healthy",
    },
  },
  {
    timestamps: true, // بيضيف createdAt و updatedAt أوتوماتيك
  },
);

// إضافة Index لتحسين سرعة البحث بالمالك أو العامل
sectorSchema.index({ owner: 1 });
sectorSchema.index({ assignedWorker: 1 });

module.exports = mongoose.model("Sector", sectorSchema);
