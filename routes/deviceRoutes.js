const express = require("express");
const router = express.Router();
const {
  registerDevice,
  getDevices,
  deleteDevice,
} = require("../controllers/deviceController");
const { protect, authorize } = require("../middleware/authMiddleware");

router
  .route("/")
  .post(protect, authorize("owner"), registerDevice)
  .get(protect, authorize("owner"), getDevices);

router.delete("/:id", protect, authorize("owner"), deleteDevice);

module.exports = router;
