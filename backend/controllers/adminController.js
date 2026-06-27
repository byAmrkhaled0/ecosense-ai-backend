const User = require("../models/User");
const Device = require("../models/Device");
const SensorData = require("../models/SensorData");

// جلب كل المستخدمين (ملاك وعمال)
exports.getAllUsers = async (req, res) => {
  try {
    const users = await User.find().select("-password").sort("-createdAt");
    res.status(200).json({ success: true, count: users.length, data: users });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// إحصائيات النظام الشاملة (Global Dashboard)
exports.getSystemStats = async (req, res) => {
  try {
    const totalUsers = await User.countDocuments();
    const totalDevices = await Device.countDocuments();
    const totalReadings = await SensorData.countDocuments();
    const totalOwners = await User.countDocuments({ role: "owner" });

    res.status(200).json({
      success: true,
      data: {
        usersCount: totalUsers,
        ownersCount: totalOwners,
        devicesCount: totalDevices,
        readingsCount: totalReadings,
      },
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// حذف أي مستخدم (Admin Power)
exports.deleteAnyUser = async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (!user) return res.status(404).json({ message: "User not found" });

    await user.deleteOne();
    res.status(200).json({ success: true, message: "User deleted by admin" });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
