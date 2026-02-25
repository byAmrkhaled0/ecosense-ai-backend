const express = require("express");
const router = express.Router();
const sensorController = require("../controllers/sensorController");
const { protect, restrictTo } = require("../middleware/authMiddleware");

// 1. استقبال البيانات من الـ ESP32 (دي مش محتاجة Token لو الجهاز بيبعت سريال بس)
// ملاحظة: لو مأمنها بـ API Key يكون أحسن، بس حالياً هنخليها مفتوحة للجهاز
router.post("/upload", sensorController.uploadData);

// --- كل الروابط اللي جاية محتاجة تسجيل دخول (protect) ---
router.use(protect);

// 2. جلب آخر قراءة (للموبايل والويب دشبورد)
router.get("/latest", sensorController.getLatest);

// 3. جلب سجل البيانات (History) مع البحث والفلترة
router.get("/history", sensorController.getHistory);

// 4. جلب الإحصائيات التحليلية (Analytics)
router.get("/analytics", sensorController.getAnalytics);

module.exports = router;
