const express = require("express");
const router = express.Router();
const {
  uploadData,
  getLatest,
  predictStatus,
  getHistory,
  getSectorStats, // ✅ تأكد إن الاسم ده هو اللي مكتوب هنا
} = require("../controllers/sensorController");

const { protect } = require("../middleware/authMiddleware");

router.post("/upload", uploadData);
router.get("/latest", protect, getLatest);
router.post("/predict", protect, predictStatus);
router.get("/history", protect, getHistory);

// السطر 16 - تأكد أن getSectorStats معرفة فوق
router.get("/stats", protect, getSectorStats);

module.exports = router;
