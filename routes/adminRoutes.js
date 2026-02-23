const express = require("express");
const router = express.Router();
const {
  getAllUsers,
  getSystemStats,
  deleteAnyUser,
} = require("../controllers/adminController"); // هنكريته تحت

const { protect, authorize } = require("../middleware/authMiddleware");

// 🔒 حماية مزدوجة: لازم يكون مسجل دخول (protect) ولازم يكون أدمن (authorize)
router.use(protect);
router.use(authorize("admin"));

// المسارات الخاصة بالأدمن فقط
router.get("/users", getAllUsers); // عرض كل الناس اللي في السيستم
router.get("/system-stats", getSystemStats); // إحصائيات السيرفر (كام جهاز، كام قراءة)
router.delete("/user/:id", deleteAnyUser); // حذف أي مستخدم مخالف

module.exports = router;
