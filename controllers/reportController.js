const SensorData = require("../models/SensorData");
const mongoose = require("mongoose");

// 1. تقرير إحصائيات القطاع (للرسم البياني)
exports.getSectorStatsReport = async (req, res) => {
  try {
    const { sectorId, days = 7 } = req.query; // الافتراضي آخر أسبوع

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(days));

    const report = await SensorData.aggregate([
      {
        $match: {
          sectorId: new mongoose.Types.ObjectId(sectorId),
          timestamp: { $gte: startDate },
        },
      },
      {
        $group: {
          _id: {
            $dateToString: { format: "%Y-%m-%d", date: "$timestamp" },
          },
          avgTemp: { $avg: "$air.temperature" },
          avgMoisture: { $avg: "$soil.moisture" },
          avgHumidity: { $avg: "$air.humidity" },
          alertsCount: {
            $sum: { $cond: [{ $eq: ["$analysis.status", "Critical"] }, 1, 0] },
          },
        },
      },
      { $sort: { _id: 1 } }, // الترتيب حسب التاريخ من الأقدم للأحدث
    ]);

    res.status(200).json({ success: true, count: report.length, data: report });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

// 2. تصدير البيانات لملف CSV (للتحميل)
exports.exportToCSV = async (req, res) => {
  try {
    const { sectorId } = req.query;
    const data = await SensorData.find({ sectorId })
      .sort("-timestamp")
      .limit(100);

    let csv = "\ufeff"; // عشان يدعم العربي في Excel
    csv += "التاريخ,الحرارة,الرطوبة,رطوبة التربة,الحالة\n";

    data.forEach((item) => {
      csv += `${item.timestamp.toISOString()},${item.air.temperature},${item.air.humidity},${item.soil.moisture},${item.analysis.status}\n`;
    });

    res.header("Content-Type", "text/csv; charset=utf-8");
    res.attachment(`sector-report-${Date.now()}.csv`);
    return res.send(csv);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
