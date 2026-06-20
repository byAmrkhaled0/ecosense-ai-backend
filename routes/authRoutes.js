const express = require("express");
const router = express.Router();
const passport = require("passport"); // ✅ لازم تعمل import لباسبورت هنا
const {
  registerUser,
  loginUser,
  verifyAndRegister,
  getMe,
  nativeGoogleAuth,
  socialAuthSuccess, // ✅ لازم تضيفها هنا عشان الـ Route يشوفها
  logout,
  forgotPassword,
  resetPassword,
} = require("../controllers/authController");

const { protect } = require("../middleware/authMiddleware");

// --- المسارات العادية ---
router.post("/register", registerUser);
router.post("/login", loginUser);
router.post("/verify-otp", verifyAndRegister);
router.post("/google-auth", nativeGoogleAuth);

router.get(
  "/google",
  (req, res, next) => {
    // نقرأ الرابط اللي الفرونت إند باعته في الـ Query Parameter
    const redirectTo = req.query.redirect_to;

    if (redirectTo) {
      // بنحفظ الرابط في الكوكيز لمدة 5 دقائق مثلاً لحين العودة من جوجل
      res.cookie("returnTo", redirectTo, {
        maxAge: 5 * 60 * 1000, // 5 دقائق
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "none",
      });
    }
    next();
  },
  passport.authenticate("google", {
    scope: ["profile", "email"],
    prompt: "select_account",
  }),
);

// 2. العودة من جوجل (الرابط المتسجل في Google Console)
router.get(
  "/google/callback",
  passport.authenticate("google", {
    session: false,
    failureRedirect: "/login",
  }),
  socialAuthSuccess,
);

router.post("/forgot-password", forgotPassword);
router.post("/reset-password", resetPassword);

// للموبايل (Token-based)
router.get("/me", protect, getMe);

router.post("/logout", protect, logout);

// --- مسارات جوجل (Passport Flow) ---

// 1. بداية عملية الدخول (الويب بيضغط على اللينك ده)

module.exports = router;
