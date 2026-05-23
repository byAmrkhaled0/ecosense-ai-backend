const Sector = require("../models/Sector");

// @desc    إنشاء قطاع جديد في المزرعة (للمالك فقط)
exports.createSector = async (req, res) => {
  try {
    const { name, cropType, area, assignedWorker, location } = req.body;

    const sector = await Sector.create({
      name,
      cropType,
      area,
      location,
      assignedWorker: assignedWorker || null,
      ownerId: req.user._id, // 👈 التعديل هنا (ownerId بدل owner)
    });

    res.status(201).json({ success: true, data: sector });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

// @desc    جلب القطاعات (للمالك يرى الكل، وللعامل يرى ما يخصه فقط)
exports.getSectors = async (req, res) => {
  try {
    let query = {};

    if (req.user.role === "owner") {
      // المالك يشوف كل القطاعات اللي هو أنشأها
      query = { ownerId: req.user._id }; // 👈 التعديل هنا
    } else if (req.user.role === "worker") {
      // العامل يشوف فقط القطاعات اللي هو متسجل فيها كـ assignedWorker
      query = { assignedWorker: req.user._id };
    }

    const sectors = await Sector.find(query).populate(
      "assignedWorker",
      "firstName lastName phoneNumber",
    );

    res.status(200).json({
      success: true,
      count: sectors.length,
      data: sectors,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// @desc    تعديل بيانات قطاع (للمالك فقط) - [PUT ]
exports.updateSector = async (req, res) => {
  try {
    let sector = await Sector.findById(req.params.id);

    if (!sector) {
      return res
        .status(404)
        .json({ success: false, message: "القطاع غير موجود" });
    }

    // ✅ التعديل: استخراج الـ ID بأمان (سواء ownerId أو owner)
    const sectorOwner = sector.ownerId || sector.owner;

    if (!sectorOwner) {
      return res.status(400).json({
        success: false,
        message:
          "هذا القطاع لا يمتلك مالك مسجل، يرجى تحديثه يدوياً من الداتابيز أو حذفه.",
      });
    }

    // التأكد من الملكية
    if (sectorOwner.toString() !== req.user._id.toString()) {
      return res
        .status(401)
        .json({ success: false, message: "غير مسموح لك بتعديل هذا القطاع" });
    }

    // التحديث
    sector = await Sector.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
      runValidators: true,
    });

    res.status(200).json({ success: true, data: sector });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
// @desc    حذف قطاع (للمالك فقط)
exports.deleteSector = async (req, res) => {
  try {
    const sector = await Sector.findById(req.params.id);

    if (!sector) {
      return res
        .status(404)
        .json({ success: false, message: "القطاع غير موجود" });
    }

    // التأكد من الملكية (ownerId)
    if (sector.ownerId.toString() !== req.user._id.toString()) {
      return res
        .status(401)
        .json({ success: false, message: "غير مسموح لك بحذف هذا القطاع" });
    }

    await sector.deleteOne();

    res.status(200).json({ success: true, message: "تم حذف القطاع بنجاح" });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
