const express = require("express");
const router = express.Router();
const sensorController = require("../controllers/sensorController");
const { protect, restrictTo } = require("../middleware/authMiddleware");
const upload = require("../config/upload");

// 1. تعديل السطر ده عشان يستقبل form-data
// استخدم upload.none() لو الداتا عبارة عن نصوص بس (حرارة، رطوبة، سيريال)
router.post("/upload", upload.none(), sensorController.uploadData);

// --- كل الروابط اللي جاية محتاجة تسجيل دخول (protect) ---
router.use(protect);

// 2. جلب آخر قراءة (للموبايل والويب دشبورد)
router.get("/latest", sensorController.getLatest);

// 3. جلب سجل البيانات (History) مع البحث والفلترة
router.get("/history", sensorController.getHistory);

// 4. جلب الإحصائيات التحليلية (Analytics)
router.get("/analytics", sensorController.getAnalytics);

module.exports = router;
