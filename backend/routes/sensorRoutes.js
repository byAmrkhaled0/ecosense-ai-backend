const express = require("express");

const router = express.Router();

const sensorController = require("../controllers/sensorController");

const { protect } = require("../middleware/authMiddleware");

const upload = require("../config/upload");

// Upload Sensor Data
router.post("/upload", upload.none(), sensorController.uploadDataOnly);

// 2. مسار التحليل (يطلبه الويب أو الفلاتر لتحديث وتحليل آخر قراءة بناءً على الـ Sector)
router.post("/analyze/:sectorId", sensorController.analyzeLastReading);
// Background AI Processing

// Protected Routes
router.use(protect);

// Dashboard APIs
router.get("/latest", sensorController.getLatest);

router.get("/history", sensorController.getHistory);

router.get("/analytics", sensorController.getAnalytics);

module.exports = router;
