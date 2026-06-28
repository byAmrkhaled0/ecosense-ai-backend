const express = require("express");
const router = express.Router();
const {
  uploadImage,
  getImageHistory,
  deleteImageLog,
} = require("../controllers/imageController");
const { protect, authorize } = require("../middleware/authMiddleware");
const upload = require("../config/upload"); // تأكد أن هذا الملف يستخدم CloudinaryStorage الآن

// 1. رفع الصور (مفتوح Public لسهولة التعامل مع IoT والموبايل)
router.post("/upload", upload.single("image"), uploadImage);

// 2. تاريخ الصور (محمي - لازم يكون مسجل دخول)
router.get("/history", protect, getImageHistory);

// 3. حذف سجل (محمي للمالك فقط)
router.delete("/:id", protect, authorize("owner"), deleteImageLog);

module.exports = router;
