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

exports.registerUser = async (req, res) => {
  try {
    let { email, password, firstName, lastName, address, phoneNumber } =
      req.body;

    if (email) email = email.trim().toLowerCase();

    // 1. التأكد من أن المستخدم مش موجود فعلاً
    const existing = await User.findOne({ email });
    if (existing) {
      return res
        .status(400)
        .json({ success: false, message: "المستخدم موجود بالفعل ومفعل." });
    }

    // 2. توليد رمز تفعيل عشوائي
    const verificationCode = Math.floor(
      100000 + Math.random() * 900000,
    ).toString();

    // 3. تشفير بيانات المستخدم + الرمز في JWT
    const tempToken = jwt.sign(
      {
        userData: {
          email,
          password,
          firstName,
          lastName,
          address,
          phoneNumber,
          role: "owner",
        },
        verificationCode,
      },
      process.env.JWT_SECRET,
      { expiresIn: "10m" },
    );

    const emailMessage = `
        <h3>👋 مرحباً بك في EcoSense!</h3>
        <p>استخدم الرمز التالي لإتمام عملية التسجيل وتفعيل حسابك:</p>
        <h1 style="color: #4CAF50; text-align: center;">${verificationCode}</h1>
    `;

    try {
      await sendEmail({
        email,
        subject: "رمز تفعيل حساب EcoSense",
        message: emailMessage,
      });

      // 🔥 التعديل هنا: بنبعت الـ tempToken جوه الـ json عشان الفرونت إند يستلمه ويمسكه في إيده
      return res.status(200).json({
        success: true,
        message: "تم إرسال رمز التفعيل للإيميل.",
        registrationToken: tempToken, // 👈 أهو ده اللي هينقذنا
      });
    } catch (err) {
      return res
        .status(500)
        .json({ success: false, message: "فشل إرسال الإيميل." });
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

exports.verifyAndRegister = async (req, res) => {
  try {
    // 🔥 التعديل هنا: بنستقبل الـ registrationToken من الـ body اللي جاي من الفرونت إند مباشرة
    const { code, registrationToken } = req.body;

    const tempToken = registrationToken || req.cookies?.registrationToken;

    if (!tempToken) {
      return res.status(400).json({
        success: false,
        message:
          "انتهت صلاحية جلسة التسجيل أو التوكن مفقود، يرجى إعادة المحاولة.",
      });
    }

    // 🔐 فك التوكن
    let decoded;
    try {
      decoded = jwt.verify(tempToken, process.env.JWT_SECRET);
    } catch (err) {
      if (err.name === "TokenExpiredError") {
        return res.status(400).json({
          success: false,
          message: "انتهت صلاحية الجلسة (10 دقائق)، يرجى التسجيل من جديد.",
        });
      }
      return res.status(400).json({
        success: false,
        message: "جلسة غير صالحة، يرجى إعادة المحاولة.",
      });
    }

    // 🔥 التحقق من الكود
    if (!decoded?.verificationCode) {
      return res
        .status(400)
        .json({ success: false, message: "بيانات التحقق غير مكتملة." });
    }

    if (decoded.verificationCode !== code) {
      return res
        .status(400)
        .json({ success: false, message: "الرمز غير صحيح، حاول مرة أخرى." });
    }

    // 🚀 إنشاء المستخدم
    const newUser = await User.create({
      ...decoded.userData,
      isVerified: true,
    });

    // 🔥 الحتة السحرية: توليد توكن تسجيل الدخول الأساسي فوراً للمستخدم الجديد
    const authToken = jwt.sign(
      { id: newUser._id },
      process.env.JWT_SECRET,
      { expiresIn: "1d" }, // صلاحيته يوم أو حسب نظام سيستمك
    );

    return res.status(201).json({
      success: true,
      message: "تم تفعيل الحساب ودخولك تلقائياً 🚀",
      token: authToken, // 👈 بنبعت التوكن هنا عشان الفرونت يمسكه
      user: {
        id: newUser._id,
        email: newUser.email,
        name: `${newUser.firstName} ${newUser.lastName}`,
      },
    });
  } catch (err) {
    console.error("verifyAndRegister error:", err);
    return res
      .status(500)
      .json({ success: false, message: "حدث خطأ داخلي أثناء التفعيل." });
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

    // حدد رابط الفرونت إند بتاعك (محلي أو المرفوع على المخدم)
    const frontendUrl = process.env.FRONTEND_URL || "http://localhost:3000";

    if (!user) {
      // لو مفيش مستخدم رجعه لصفحة اللوجن مع رسالة خطأ في الرابط
      return res.redirect(`${frontendUrl}/login?error=user_not_found`);
    }

    // توليد التوكن
    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRE || "30d",
    });

    // تحويل المستخدم للفرونت إند وتمرير التوكن في الرابط بشكل آمن
    return res.redirect(`${frontendUrl}/login?token=${token}`);
  } catch (err) {
    console.error("Social Auth Success Error:", err.message);
    const frontendUrl = process.env.FRONTEND_URL || "http://localhost:3000";
    return res.redirect(`${frontendUrl}/login?error=server_error`);
  }
};
