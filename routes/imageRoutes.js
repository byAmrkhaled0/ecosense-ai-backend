// routes/imageRoutes.js

const express = require("express");
const router = express.Router();
const {
  uploadImage,
  getImageHistory,
  deleteImageLog,
} = require("../controllers/imageController");
const { protect, authorize } = require("../middleware/authMiddleware");
const upload = require("../config/upload"); // ملف تهيئة Multer

router.post("/upload", protect, upload.single("image"), uploadImage);
router.get("/history", protect, getImageHistory);

router.delete("/:id", protect, authorize("owner"), deleteImageLog);

module.exports = router;
