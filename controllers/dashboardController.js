const Sector = require("../models/Sector");
const User = require("../models/User");
const Device = require("../models/Device");
const SensorData = require("../models/SensorData");
const Notification = require("../models/Notification"); // تأكد من استيراد الموديل

exports.getDashboard = async (req, res) => {
  try {
    const userRole = req.user.role;
    const userId = req.user._id;

    // --- حالة المالك (Owner) ---
    if (userRole === "owner") {
      // 1. الإحصائيات العامة للمالك
      const totalSectors = await Sector.countDocuments({ owner: userId });
      const totalWorkers = await User.countDocuments({
        ownerId: userId,
        role: "worker",
      });
      const totalDevices = await Device.countDocuments({ owner: userId });
      const onlineDevices = await Device.countDocuments({
        owner: userId,
        status: "online",
      });

      // 2. جلب كل القطاعات التابعة له
      const sectors = await Sector.find({ owner: userId }).populate(
        "assignedWorker",
        "firstName lastName",
      );

      // 3. التنبيهات الحرجة لكل المزرعة
      const recentAlerts = await Notification.find({ ownerId: userId })
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

    // --- حالة العامل (Worker) ---
    if (userRole === "worker") {
      // التأكد من أن العامل مرتبط بقطاع
      if (!req.user.assignedSector) {
        return res.status(403).json({
          success: false,
          message: "لم يتم تعيين قطاع لك بعد، يرجى مراجعة صاحب المزرعة.",
        });
      }

      const sectorId = req.user.assignedSector;

      // 1. جلب بيانات القطاع الخاص به فقط
      const sector = await Sector.findById(sectorId);

      // 2. جلب آخر قراءات الحساسات لهذا القطاع فقط
      const latestReadings = await SensorData.find({ sectorId: sectorId })
        .sort("-timestamp")
        .limit(1);

      // 3. التنبيهات الخاصة بقطاعه فقط
      const sectorNotifications = await Notification.find({
        sectorId: sectorId,
      })
        .sort("-createdAt")
        .limit(5);

      return res.status(200).json({
        success: true,
        role: "worker",
        data: {
          assignedSector: sector,
          latestReadings: latestReadings,
          notifications: sectorNotifications,
        },
      });
    }

    // --- حالة الأدمن (Admin) - اختياري ---
    if (userRole === "admin") {
      const totalUsers = await User.countDocuments();
      const totalAllDevices = await Device.countDocuments();
      return res.status(200).json({
        success: true,
        role: "admin",
        data: {
          totalUsers,
          totalAllDevices,
        },
      });
    }
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};
