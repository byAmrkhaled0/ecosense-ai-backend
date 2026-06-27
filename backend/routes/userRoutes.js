const express = require("express");
const router = express.Router();
// تأكد أن الوظائف دي موجودة فعلياً في userController
const { getMyWorkers, deleteWorker } = require("../controllers/userController");
const { addWorker } = require("../controllers/authController"); // هذي موجودة في authController
const { protect, authorize } = require("../middleware/authMiddleware");

router.use(protect);

// المسارات
router.post("/add-worker", authorize("owner"), addWorker);
router.get("/workers", authorize("owner"), getMyWorkers);
router.delete("/worker/:id", authorize("owner"), deleteWorker);

module.exports = router;
