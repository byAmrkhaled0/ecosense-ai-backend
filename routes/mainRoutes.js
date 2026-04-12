const express = require("express");
const router = express.Router();
const { getDashboard } = require("../controllers/dashboardController");
const {
  getNotifications,
  markAsRead,
  deleteNotification,
} = require("../controllers/notificationController");
const { protect } = require("../middleware/authMiddleware");

router.get("/dashboard", protect, getDashboard);
router.get("/notifications", protect, getNotifications);
router.patch("/notifications/:id", protect, markAsRead);
router.delete("/notifications/:id", protect, deleteNotification);

module.exports = router;
