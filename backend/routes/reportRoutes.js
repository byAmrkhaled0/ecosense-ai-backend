const express = require("express");
const router = express.Router();
const {
  getSectorStatsReport,
  exportToCSV,
} = require("../controllers/reportController");
const { protect, authorize } = require("../middleware/authMiddleware");

// مسارات التقارير - متاحة للمالك والعمال (أو المالك فقط حسب رغبتك)
router.get("/stats", protect, authorize("owner"), getSectorStatsReport);
router.get("/export", protect, authorize("owner"), exportToCSV);

module.exports = router;
