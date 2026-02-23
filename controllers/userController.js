const User = require("../models/User");

// تأكد من وجود exports.اسم_الوظيفة
exports.getMyWorkers = async (req, res) => {
  try {
    const workers = await User.find({ ownerId: req.user._id, role: "worker" });
    res
      .status(200)
      .json({ success: true, count: workers.length, data: workers });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.deleteWorker = async (req, res) => {
  try {
    const worker = await User.findOne({
      _id: req.params.id,
      ownerId: req.user._id,
    });
    if (!worker) return res.status(404).json({ message: "Worker not found" });

    await worker.deleteOne();
    res
      .status(200)
      .json({ success: true, message: "Worker removed successfully" });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
