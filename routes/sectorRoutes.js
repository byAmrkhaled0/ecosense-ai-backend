const express = require("express");
const router = express.Router();
const {
  createSector,
  getSectors,
  deleteSector,
  getAllSectorsAdmin,
} = require("../controllers/sectorController");
const { protect, authorize } = require("../middleware/authMiddleware");

router.use(protect); // لازم يكون مسجل دخول

// عرض القطاعات المتاحة للشخص (مالك أو عامل)
router.get("/", getSectors);

// المالك فقط هو من ينشئ أو يحذف
router.post("/", authorize("owner"), createSector);
router.delete("/:id", authorize("owner"), deleteSector);

module.exports = router;
