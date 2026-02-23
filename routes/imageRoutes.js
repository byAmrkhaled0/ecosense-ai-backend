// routes/imageRoutes.js

const express = require("express");
const router = express.Router();
const {
  uploadImage,
  getImageHistory,
  updateAnalysis,
  deleteImageLog,
} = require("../controllers/imageController");
const { protect } = require("../middleware/authMiddleware");
const upload = require("../config/upload"); // ملف تهيئة Multer

router.post("/upload", protect, upload.single("image"), uploadImage);
router.get("/history", protect, getImageHistory);
router.patch("/:id/analyze", protect, updateAnalysis);
router.delete("/:id", protect, deleteImageLog);

module.exports = router;
