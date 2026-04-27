const Sector = require("../models/Sector");
const User = require("../models/User");
const Device = require("../models/Device");
const SensorData = require("../models/SensorData");
const Notification = require("../models/Notification");
const ImageLog = require("../models/ImageLog"); // ضيفنا موديل الصور

exports.getDashboard = async (req, res) => {
  try {
    const { role: userRole, _id: userId } = req.user;

    // --- 1️⃣ حالة المالك (Owner) - النظرة الشاملة ---
    if (userRole === "owner") {
      // تنفيذ كل الاستعلامات في نفس الوقت (أسرع بكتير)
      const [
        sectorsCount,
        workersCount,
        devices,
        sectors,
        recentNotifications,
        aiStats,
      ] = await Promise.all([
        Sector.countDocuments({ ownerId: userId }),
        User.countDocuments({ ownerId: userId, role: "worker" }),
        Device.find({ ownerId: userId }).select("status"),
        Sector.find({ ownerId: userId }).populate(
          "assignedWorker",
          "firstName lastName",
        ),
        Notification.find({ recipient: userId })
          .sort("-createdAt")
          .limit(5)
          .populate("sectorId", "name"),
        // إحصائية سريعة: كام صورة تم تحليلها ووجد بها إصابة؟
        ImageLog.countDocuments({
          ownerId: userId,
          "analysisResult.status": "Infected",
        }),
      ]);

      const onlineDevices = devices.filter((d) => d.status === "online").length;

      return res.status(200).json({
        success: true,
        role: "owner",
        data: {
          summary: {
            totalSectors: sectorsCount,
            totalWorkers: workersCount,
            totalDevices: devices.length,
            onlineDevices,
            infectedPlantsAlerts: aiStats, // عدد الحالات المصابة المكتشفة
          },
          sectors,
          notifications: recentNotifications,
        },
      });
    }

    // --- 2️⃣ حالة العامل (Worker) - التشغيل الميداني ---
    if (userRole === "worker") {
      const sector = await Sector.findOne({ assignedWorker: userId });

      if (!sector) {
        return res
          .status(404)
          .json({ success: false, message: "لم يتم تعيين قطاع لك بعد." });
      }

      // جلب القراءات والصور والتنبيهات للقطاع في وقت واحد
      const [latestReadings, recentImages, notifications] = await Promise.all([
        SensorData.findOne({ sectorId: sector._id }).sort("-createdAt").lean(),
        ImageLog.find({ sectorId: sector._id }).sort("-createdAt").limit(3), // آخر 3 صور تم التقاطها
        Notification.find({ sectorId: sector._id, recipient: userId })
          .sort("-createdAt")
          .limit(5),
      ]);

      // إضافة "تقييم حالة" سريع للعامل
      const healthStatus = {
        isSystemCritical:
          latestReadings?.temperature > 40 || latestReadings?.humidity < 20,
        lastDiagnostic: recentImages[0]?.analysisResult?.status || "No Data",
      };

      return res.status(200).json({
        success: true,
        role: "worker",
        data: {
          sectorInfo: sector,
          sensorInsights: latestReadings || "لا توجد قراءات",
          recentDiagnostics: recentImages,
          healthStatus, // ملخص الحالة (خطر أم آمن)
          notifications,
        },
      });
    }

    // --- 3️⃣ حالة الأدمن ---
    if (userRole === "admin") {
      const [totalUsers, totalDevices, totalSectors] = await Promise.all([
        User.countDocuments(),
        Device.countDocuments(),
        Sector.countDocuments(),
      ]);
      return res.status(200).json({
        success: true,
        role: "admin",
        data: { totalUsers, totalDevices, totalSectors },
      });
    }
  } catch (err) {
    res
      .status(500)
      .json({ success: false, message: "خطأ في تحميل بيانات الداشبورد" });
  }
};
