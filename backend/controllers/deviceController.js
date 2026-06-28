const Device = require("../models/Device");
const SensorData = require("../models/SensorData");
const mongoose = require("mongoose");
const Sector = require("../models/Sector"); // 👈 لازم تستورد موديل البيانات

// @desc    تسجيل جهاز جديد وربطه بقطاع
exports.registerDevice = async (req, res) => {
  try {
    const { deviceSerial, deviceName, sectorId } = req.body;

    // 1️⃣ التحقق من المدخلات الأساسية وتجنب الـ ObjectId المشوه
    if (!deviceSerial || !deviceSerial.trim()) {
      return res
        .status(400)
        .json({ success: false, message: "الرقم التسلسلي للجهاز مطلوب" });
    }

    if (!sectorId || !mongoose.Types.ObjectId.isValid(sectorId)) {
      return res.status(400).json({
        success: false,
        message: "معرف القطاع (Sector ID) غير صحيح أو غير موجود",
      });
    }

    // 2️⃣ التحقق من أن الجهاز مش مسجل قبل كده
    const existingDevice = await Device.findOne({ deviceSerial });
    if (existingDevice) {
      return res.status(400).json({
        success: false,
        message: "هذا الجهاز مسجل مسبقاً في النظام برقم التسلسل هذا",
      });
    }

    // 3️⃣ التأكد أن القطاع موجود فعلاً وينتمي للمستخدم الحالي قبل التعديل عليه
    const sectorExists = await Sector.findById(sectorId);
    if (!sectorExists) {
      return res.status(404).json({
        success: false,
        message: "القطاع المختار غير موجود في النظام",
      });
    }

    // 4️⃣ إنشاء الجهاز في موديل الـ Device
    const device = await Device.create({
      deviceSerial,
      deviceName: deviceName || "Smart Node",
      sectorId,
      ownerId: req.user._id, // تأكد أن الـ Auth Middleware بيمرر الـ user صح
      status: "offline",
    });

    // 5️⃣ تحديث مصفوفة الـ devices جوه القطاع (Sector) فوراً
    await Sector.findByIdAndUpdate(
      sectorId,
      { $push: { devices: device._id } },
      { new: true },
    );

    // 6️⃣ خطوة الأمان للفرونت إند: جلب الجهاز بالـ populate عشان الـ Table يقراه فوراً
    const populatedDevice = await Device.findById(device._id)
      .populate("sectorId", "name")
      .lean();

    // 7️⃣ الرد بالنجاح الصريح
    return res.status(201).json({
      success: true,
      message: "تم تسجيل الجهاز بنجاح وربطه بالقطاع 🎉",
      data: populatedDevice,
    });
  } catch (err) {
    console.error("❌ Error in registerDevice Backend:", err.message);

    // ✅ تعديل جوهري: إرجاع الـ حقل كـ message ليتوافق مع الـ Toast في الفرونت إند
    return res.status(500).json({
      success: false,
      message: err.message || "حدث خطأ غير متوقع أثناء تسجيل الجهاز",
    });
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
