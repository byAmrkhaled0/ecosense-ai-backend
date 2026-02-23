const jwt = require("jsonwebtoken");
const User = require("../models/User");

// 1️⃣ حماية المسارات (التأكد من تسجيل الدخول)
exports.protect = async (req, res, next) => {
  let token;

  // التأكد من وجود الـ Token في الـ Headers
  if (
    req.headers.authorization &&
    req.headers.authorization.startsWith("Bearer")
  ) {
    token = req.headers.authorization.split(" ")[1];
  }

  if (!token) {
    return res
      .status(401)
      .json({
        success: false,
        message: "غير مسموح لك بالدخول، يرجى تسجيل الدخول أولاً.",
      });
  }

  try {
    // فك تشفير الـ Token
    const decoded = jwt.verify(token, process.env.JWT_SECRET);

    // جلب بيانات المستخدم وإلحاقها بالطلب (req.user)
    req.user = await User.findById(decoded.id);

    if (!req.user) {
      return res
        .status(404)
        .json({
          success: false,
          message: "المستخدم المرتبط بهذا الـ Token غير موجود.",
        });
    }

    next();
  } catch (err) {
    return res
      .status(401)
      .json({
        success: false,
        message: "الـ Token غير صالح أو انتهت صلاحيته.",
      });
  }
};

// 2️⃣ التحكم في الصلاحيات (Roles)
// بنستخدمها كده: authorize('owner')
exports.authorize = (...roles) => {
  return (req, res, next) => {
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({
        success: false,
        message: `الدور (${req.user.role}) غير مسموح له بالقيام بهذا الإجراء.`,
      });
    }
    next();
  };
};
