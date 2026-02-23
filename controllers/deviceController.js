const Device = require("../models/Device");

// @desc    تسجيل جهاز جديد وربطه بقطاع
exports.registerDevice = async (req, res) => {
  try {
    const { deviceSerial, deviceType, sectorId } = req.body;

    const existingDevice = await Device.findOne({ deviceSerial });
    if (existingDevice)
      return res
        .status(400)
        .json({ message: "Device serial already registered" });

    const device = await Device.create({
      deviceSerial,
      deviceType: deviceType || "Sensor Kit",
      sector: sectorId,
      owner: req.user._id,
    });

    res.status(201).json({ success: true, data: device });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

// @desc    جلب حالة الأجهزة في المزرعة
exports.getDevices = async (req, res) => {
  try {
    const devices = await Device.find({ owner: req.user._id }).populate(
      "sector",
      "name cropType",
    );
    res.status(200).json({ success: true, data: devices });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// @desc    حذف جهاز من المزرعة
exports.deleteDevice = async (req, res) => {
  try {
    const device = await Device.findById(req.params.id);

    if (!device) {
      return res
        .status(404)
        .json({ success: false, message: "الجهاز غير موجود" });
    }

    // التأكد إن المالك هو اللي بيمسح جهازه مش حد تاني
    if (device.owner.toString() !== req.user._id.toString()) {
      return res
        .status(401)
        .json({ success: false, message: "غير مسموح لك بحذف هذا الجهاز" });
    }

    await device.deleteOne();

    res
      .status(200)
      .json({ success: true, message: "تم حذف الجهاز بنجاح من النظام" });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
