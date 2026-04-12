const Sector = require("../models/Sector");
const User = require("../models/User");
const Device = require("../models/Device");
const SensorData = require("../models/SensorData");
const Notification = require("../models/Notification");

exports.getDashboard = async (req, res) => {
  try {
    const userRole = req.user.role;
    const userId = req.user._id;

    // --- 1️⃣ حالة المالك (Owner) ---
    if (userRole === "owner") {
      // تعديل: البحث بـ ownerId بدل owner ليتوافق مع الموديلات الجديدة
      const totalSectors = await Sector.countDocuments({ ownerId: userId });
      const totalWorkers = await User.countDocuments({
        ownerId: userId,
        role: "worker",
      });
      const totalDevices = await Device.countDocuments({ ownerId: userId });
      const onlineDevices = await Device.countDocuments({
        ownerId: userId,
        status: "online",
      });

      // جلب القطاعات مع بيانات العامل
      const sectors = await Sector.find({ ownerId: userId }).populate(
        "assignedWorker",
        "firstName lastName phoneNumber",
      );

      // التنبيهات (Notification تستخدم recipient أو ownerId حسب تصميمك، هنا ثبتناها ownerId)
      const recentAlerts = await Notification.find({ recipient: userId })
        .sort("-createdAt")
        .limit(5)
        .populate("sectorId", "name");

      return res.status(200).json({
        success: true,
        role: "owner",
        data: {
          summary: {
            sectorsCount: totalSectors,
            workersCount: totalWorkers,
            devicesCount: totalDevices,
            onlineDevices: onlineDevices,
          },
          sectors: sectors,
          notifications: recentAlerts,
        },
      });
    }

    // --- 2️⃣ حالة العامل (Worker) ---
    if (userRole === "worker") {
      // تعديل: العامل بيبحث عن القطاعات اللي هو "assignedWorker" فيها
      const sector = await Sector.findOne({ assignedWorker: userId });

      if (!sector) {
        return res.status(404).json({
          success: false,
          message: "لم يتم تعيين قطاع لك بعد.",
        });
      }

      // جلب آخر قراءة حساسات (تعديل: السنسور بيستخدم createdAt مش timestamp)
      const latestReadings = await SensorData.findOne({ sectorId: sector._id })
        .sort("-createdAt")
        .lean();

      // التنبيهات الخاصة بقطاعه
      const sectorNotifications = await Notification.find({
        sectorId: sector._id,
      })
        .sort("-createdAt")
        .limit(5);

      return res.status(200).json({
        success: true,
        role: "worker",
        data: {
          assignedSector: sector,
          latestReadings: latestReadings || "لا توجد قراءات حتى الآن",
          notifications: sectorNotifications,
        },
      });
    }

    // --- 3️⃣ حالة الأدمن ---
    if (userRole === "admin") {
      const totalUsers = await User.countDocuments();
      const totalAllDevices = await Device.countDocuments();
      return res.status(200).json({
        success: true,
        role: "admin",
        data: { totalUsers, totalAllDevices },
      });
    }
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};
