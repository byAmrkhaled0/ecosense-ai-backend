const express = require("express");
const router = express.Router();
const {
  createSector,
  getSectors,
  deleteSector,
  updateSector,
} = require("../controllers/sectorController");
const { protect, authorize } = require("../middleware/authMiddleware");

router.use(protect); // لازم يكون مسجل دخول

// عرض القطاعات المتاحة للشخص (مالك أو عامل)
router.get("/", authorize("owner"), getSectors);

// المالك فقط هو من ينشئ أو يحذف
router.post("/", authorize("owner"), createSector);
router.delete("/:id", authorize("owner"), deleteSector);
router.put("/:id", authorize("owner"), updateSector);

module.exports = router;
