const Device = require("../models/Device");
const SensorData = require("../models/SensorData"); // 👈 لازم تستورد موديل البيانات

// @desc    تسجيل جهاز جديد وربطه بقطاع
exports.registerDevice = async (req, res) => {
  try {
    const { deviceSerial, deviceName, sectorId } = req.body;

    const existingDevice = await Device.findOne({ deviceSerial });
    if (existingDevice) {
      return res.status(400).json({
        success: false,
        message: "هذا الجهاز مسجل مسبقاً في النظام برقم التسلسل هذا",
      });
    }

    const device = await Device.create({
      deviceSerial,
      deviceName: deviceName || "Smart Node",
      sectorId,
      ownerId: req.user._id,
      status: "offline",
    });

    res.status(201).json({
      success: true,
      message: "تم تسجيل الجهاز بنجاح",
      data: device,
    });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

// @desc    جلب حالة الأجهزة في المزرعة
exports.getDevices = async (req, res) => {
  try {
    // 1. البحث باستخدام ownerId (تأكد إنه بنفس الاسم في الموديل)
    const devices = await Device.find({ ownerId: req.user._id })
      .populate({
        path: "sectorId",
        select: "name cropType location",
      })
      .lean(); // بنستخدم lean عشان نعدل على الكائن بسهولة لو فيه نقص

    // 2. معالجة البيانات قبل الإرسال (اختياري بس بيحل مشاكل العرض)
    const formattedDevices = devices.map((device) => {
      return {
        ...device,
        // لو الـ sectorId رجع null (بسبب داتا قديمة)، بنحط قيم افتراضية بدل ما الفرونت إند يضرب
        sectorInfo: device.sectorId
          ? device.sectorId
          : { name: "قطاع غير معروف", cropType: "N/A" },
      };
    });

    res.status(200).json({
      success: true,
      count: formattedDevices.length,
      data: formattedDevices,
    });
  } catch (err) {
    console.error("❌ Get Devices Error:", err.message);
    res.status(500).json({ success: false, error: err.message });
  }
};
// @desc    حذف جهاز من المزرعة نهائياً مع بياناته
exports.deleteDevice = async (req, res) => {
  try {
    const device = await Device.findById(req.params.id);

    if (!device) {
      return res
        .status(404)
        .json({ success: false, message: "الجهاز غير موجود" });
    }

    // التأكد من الملكية
    if (device.ownerId.toString() !== req.user._id.toString()) {
      return res
        .status(401)
        .json({ success: false, message: "غير مسموح لك بحذف هذا الجهاز" });
    }

    // 🔥 الخطوة الأهم: مسح كل قراءات السنسورات المرتبطة بهذا الجهاز
    // ده بيضمن إنك لو ضفت جهاز جديد بنفس السيريال، يبدأ على نظافة ملوش تاريخ قديم
    await SensorData.deleteMany({ deviceId: device._id });

    // مسح الجهاز نفسه
    await device.deleteOne();

    res.status(200).json({
      success: true,
      message: "تم حذف الجهاز وكل بيانات الحساسات المرتبطة به نهائياً",
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
