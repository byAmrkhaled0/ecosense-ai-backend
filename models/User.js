const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

const userSchema = new mongoose.Schema({
  firstName: {
    type: String,
    required: [true, "Please add a first name"],
    trim: true,
  },
  lastName: {
    type: String,
    required: [true, "Please add a last name"],
    trim: true,
  },
  address: {
    type: String,
    required: [true, "Please add an address"],
  },
  phoneNumber: {
    type: String,
    required: [true, "Please add a phone number"],
    unique: true,
    match: [/^\+?\d{10,15}$/, "Please add a valid phone number"],
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
  },
  password: {
    type: String,
    minlength: 6,
    select: false, // لا يظهر الباسورد في الاستعلامات العادية لزيادة الأمان
  },

  // 🛡️ نظام الصلاحيات الجديد
  role: {
    type: String,
    enum: ["admin", "owner", "worker"],
    default: "owner", // صاحب المزرعة هو الدور الافتراضي عند التسجيل
  },

  assignedSector: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Sector",
    default: null, // المالك بيبقى null، أما العامل بنحدد له قطاع
  },

  // 🔗 ربط العامل بصاحب المزرعة (يستخدم فقط إذا كان الدور worker)
  ownerId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    default: null,
  },

  // 🌍 Social Login fields
  provider: {
    type: String,
    enum: ["local", "google"],
    default: "local",
  },
  googleId: {
    type: String,
    default: null,
  },

  // ✅ التحقق بالرمز السري (OTP)
  isVerified: {
    type: Boolean,
    default: false,
  },
  verificationCode: String,
  verificationCodeExpires: Date,

  createdAt: {
    type: Date,
    default: Date.now,
  },
});

// 🔐 تشفير الباسورد قبل الحفظ
userSchema.pre("save", async function (next) {
  if (!this.isModified("password")) return next();
  // تأكد إنك بتستخدم ملح (Salt) مناسب
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

// 🎫 إنشاء JWT token (أضفنا الـ role للـ payload لسهولة استخدامه في الفرونت إند)
userSchema.methods.getSignedToken = function () {
  return jwt.sign({ id: this._id, role: this.role }, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRE || "7d",
  });
};

// مقارنة الباسورد أثناء الـ login
userSchema.methods.matchPassword = async function (password) {
  return await bcrypt.compare(password, this.password);
};

userSchema.methods.getSignedJwtToken = function () {
  const jwt = require("jsonwebtoken");
  return jwt.sign({ id: this._id }, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRE,
  });
};

// ✅ إنشاء رمز OTP رقمي من 6 أرقام
userSchema.methods.getVerificationCode = function () {
  const verificationCode = Math.floor(
    100000 + Math.random() * 900000,
  ).toString();
  this.verificationCode = verificationCode;

  // صلاحية الرمز (10 دقائق)
  this.verificationCodeExpires = Date.now() + 10 * 60 * 1000;

  return verificationCode;
};

module.exports = mongoose.model("User", userSchema);
