const User = require("../models/User");
const sendEmail = require("../utils/sendEmail");
const { OAuth2Client } = require("google-auth-library");
const jwt = require("jsonwebtoken");

const client = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);

// مساعد لإرسال الـ Token والبيانات الأساسية
const sendTokenResponse = (user, statusCode, res, message = "Success") => {
  const token = user.getSignedToken();
  res.status(statusCode).json({
    success: true,
    message: message,
    token: token,
    user: {
      id: user._id,
      firstName: user.firstName,
      lastName: user.lastName,
      email: user.email,
      role: user.role, // 🛡️ مهم جداً للـ Frontend
      ownerId: user.ownerId, // 🔗 عشان العامل يعرف هو تبع مين
    },
  });
};

// 📝 Register (تسجيل صاحب مزرعة جديد)
exports.registerUser = async (req, res) => {
  try {
    let { email, password, firstName, lastName, address, phoneNumber } =
      req.body;

    if (email) email = email.trim().toLowerCase();

    // التحقق من الحقول الإلزامية
    if (
      !email ||
      !password ||
      !firstName ||
      !lastName ||
      !address ||
      !phoneNumber
    ) {
      return res
        .status(400)
        .json({ success: false, message: "يرجى ملء جميع الحقول المطلوبة." });
    }

    const existing = await User.findOne({ email });
    if (existing) {
      return res
        .status(400)
        .json({ success: false, message: "المستخدم موجود بالفعل." });
    }

    // إنشاء المستخدم كـ Owner (مالك مزرعة)
    const user = await User.create({
      email,
      password,
      firstName,
      lastName,
      address,
      phoneNumber,
      role: "owner", // القيمة الافتراضية للتسجيل الخارجي
    });

    const verificationCode = user.getVerificationCode();
    await user.save({ validateBeforeSave: false });

    const emailMessage = `
        <h3>👋 مرحباً بك في EcoSense!</h3>
        <p>لإكمال تفعيل حسابك كصاحب مزرعة، يرجى استخدام الرمز السري:</p>
        <h1 style="color: #4CAF50; text-align: center;">${verificationCode}</h1>
        <p>الرمز صالح لمدة 10 دقائق فقط.</p>
    `;

    try {
      await sendEmail({
        email: user.email,
        subject: "رمز تفعيل حساب EcoSense",
        message: emailMessage,
      });

      res.status(200).json({
        success: true,
        message: "تم التسجيل بنجاح. يرجى تفعيل الحساب بالرمز المرسل للإيميل.",
      });
    } catch (err) {
      await User.findByIdAndDelete(user._id);
      return res
        .status(500)
        .json({ success: false, message: "فشل إرسال الإيميل، حاول لاحقاً." });
    }
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// 👷 إضافة عامل جديد (بواسطة صاحب المزرعة فقط)
exports.addWorker = async (req, res) => {
  try {
    // 1. استلام البيانات (بما فيها الـ assignedSector اللي بعتناه من Postman)
    const {
      email,
      password,
      firstName,
      lastName,
      address,
      phoneNumber,
      assignedSector,
    } = req.body;

    // 2. التأكد أن الذي يضيف هو Owner (إجراء إضافي للأمان)
    if (req.user.role !== "owner") {
      return res.status(403).json({
        success: false,
        message: "غير مسموح لغير أصحاب المزارع بإضافة عمال.",
      });
    }

    // 3. التأكد من عدم تكرار الإيميل
    const existing = await User.findOne({ email });
    if (existing) {
      return res
        .status(400)
        .json({ success: false, message: "هذا الإيميل مسجل مسبقاً." });
    }

    // 4. إنشاء العامل وربطه بالمالك وبالقطاع
    const worker = await User.create({
      email,
      password,
      firstName,
      lastName,
      address,
      phoneNumber,
      role: "worker",
      ownerId: req.user._id,
      assignedSector: assignedSector || null, // ربط القطاع بالعامل
      isVerified: true, // العامل بيتكريه مفعل جاهز لأن المالك هو اللي ضايفه
    });

    // 5. [خطوة احترافية] تحديث القطاع ليصبح هذا العامل هو المسؤول عنه
    if (assignedSector) {
      const Sector = require("../models/Sector"); // تأكد من عمل import للموديل
      await Sector.findByIdAndUpdate(assignedSector, {
        assignedWorker: worker._id,
      });
    }

    res.status(201).json({
      success: true,
      message: "تم إضافة العامل بنجاح وربطه بالقطاع المحدد.",
      data: worker,
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};
// 🔑 Login
exports.loginUser = async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res
        .status(400)
        .json({ success: false, message: "يرجى إدخال الإيميل والباسورد" });

    const user = await User.findOne({ email }).select("+password");
    if (!user || !(await user.matchPassword(password))) {
      return res
        .status(401)
        .json({ success: false, message: "بيانات الدخول غير صحيحة" });
    }

    if (!user.isVerified) {
      return res
        .status(401)
        .json({ success: false, message: "الحساب غير مفعل" });
    }

    sendTokenResponse(user, 200, res, "تم تسجيل الدخول بنجاح");
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// 📧 Verify Code (OTP)
exports.verifyCode = async (req, res) => {
  try {
    const { email, code } = req.body;
    const user = await User.findOne({
      email,
      verificationCode: code,
      verificationCodeExpires: { $gt: Date.now() },
    });

    if (!user)
      return res
        .status(400)
        .json({ success: false, message: "الرمز خاطئ أو منتهي الصلاحية" });

    user.isVerified = true;
    user.verificationCode = undefined;
    user.verificationCodeExpires = undefined;
    await user.save();

    sendTokenResponse(user, 200, res, "تم تفعيل الحساب بنجاح");
  } catch (err) {
    res.status(500).json({ success: false, message: "خطأ أثناء عملية التحقق" });
  }
};

// 👤 Get Me
exports.getMe = async (req, res) => {
  try {
    const user = await User.findById(req.user._id).select("-password");
    res.status(200).json({ success: true, user });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// 🌍 Native Google Auth
exports.nativeGoogleAuth = async (req, res) => {
  const { idToken, address, phoneNumber } = req.body;
  try {
    const ticket = await client.verifyIdToken({
      idToken,
      audience: process.env.GOOGLE_CLIENT_ID,
    });
    const payload = ticket.getPayload();
    const email = payload.email.toLowerCase();

    let user = await User.findOne({ email });

    if (!user) {
      user = await User.create({
        email,
        firstName: payload.given_name || "Google", // جوجل بتبعت الاسم الأول منفصل
        lastName: payload.family_name || "User", // واللقب منفصل
        role: "owner",
        isVerified: true,
        address: address || "Social Account",
        phoneNumber: phoneNumber || "0000000000",
        googleId: payload.sub, // حتة ذكية: خزن الـ ID بتاع جوجل عشان لو غير إيميله
      });
    }

    sendTokenResponse(user, 200, res, "Google login successful");
  } catch (error) {
    res
      .status(401)
      .json({ success: false, message: "فشلت عملية الدخول عبر جوجل" });
  }
};

// 🚪 Logout (تسجيل الخروج)
exports.logout = async (req, res) => {
  try {
    // في الـ JWT، السيرفر لا يحتاج لفعل الكثير، فقط نؤكد للـ Client حذف التوكن
    res.status(200).json({
      success: true,
      message: "تم تسجيل الخروج بنجاح. يرجى حذف التوكن من الجهاز.",
    });
  } catch (err) {
    res.status(500).json({ success: false, message: "خطأ أثناء تسجيل الخروج" });
  }
};

// 🌍 Social Login (Google / Facebook)
// @desc    معالجة نجاح تسجيل الدخول عبر جوجل وإرسال التوكن
exports.socialAuthSuccess = async (req, res) => {
  try {
    const user = req.user;

    if (!user) {
      return res
        .status(401)
        .json({ success: false, message: "User not found" });
    }

    // توليد التوكن مباشرة هنا عشان نضمن إنه يشتغل
    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRE || "30d",
    });

    res.status(200).json({
      success: true,
      token,
      user: {
        id: user._id,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        role: user.role,
      },
    });
  } catch (err) {
    console.error("Social Auth Success Error:", err.message);
    res.status(500).json({
      success: false,
      message: "Server error during token generation",
    });
  }
};
