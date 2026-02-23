const express = require("express");
const router = express.Router();
const passport = require("passport"); // ✅ لازم تعمل import لباسبورت هنا
const {
  registerUser,
  loginUser,
  verifyCode,
  getMe,
  nativeGoogleAuth,
  socialAuthSuccess, // ✅ لازم تضيفها هنا عشان الـ Route يشوفها
  logout,
} = require("../controllers/authController");

const { protect } = require("../middleware/authMiddleware");

// --- المسارات العادية ---
router.post("/register", registerUser);
router.post("/login", loginUser);
router.post("/verify-otp", verifyCode);
router.post("/google-auth", nativeGoogleAuth);
router.get(
  "/google",
  passport.authenticate("google", {
    scope: ["profile", "email"],
    prompt: "select_account",
  }),
);
// 2. العودة من جوجل (الرابط اللي متسجل في Google Console)
router.get(
  "/google/callback",
  passport.authenticate("google", {
    session: false,
    failureRedirect: "/login",
  }),
  socialAuthSuccess,
);

// للموبايل (Token-based)
router.get("/me", protect, getMe);

router.post("/logout", protect, logout);

// --- مسارات جوجل (Passport Flow) ---

// 1. بداية عملية الدخول (الويب بيضغط على اللينك ده)

module.exports = router;
