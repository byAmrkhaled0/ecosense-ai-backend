const SensorData = require("../models/SensorData");
const mongoose = require("mongoose");

exports.getSectorStatsReport = async (req, res) => {
  try {
    const { sectorId, days = 7 } = req.query;

    if (!sectorId) {
      return res
        .status(400)
        .json({ success: false, message: "يجب تحديد معرف القطاع" });
    }

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(days));
    startDate.setHours(0, 0, 0, 0);

    const report = await SensorData.aggregate([
      {
        $match: {
          // ✅ تأكدنا إن الـ ID بيتحول لـ ObjectId صح
          sectorId: new mongoose.Types.ObjectId(sectorId),
          createdAt: { $gte: startDate },
        },
      },
      {
        $group: {
          _id: {
            $dateToString: { format: "%Y-%m-%d", date: "$createdAt" },
          },
          avgTemp: { $avg: "$air.temperature" },
          avgMoisture: { $avg: "$soil.moisture" },
          avgHumidity: { $avg: "$air.humidity" },
          alertsCount: {
            $sum: {
              $cond: [
                // ✅ تعديل الـ $in لتكون متوافقة مع الـ Aggregation
                {
                  $in: ["$analysis.status", ["Critical", "Warning", "Danger"]],
                },
                1,
                0,
              ],
            },
          },
        },
      },
      // ✅ تقريب الأرقام العشرية عشان المنظر في الداشبورد يبقى نظيف
      {
        $project: {
          _id: 1,
          avgTemp: { $round: ["$avgTemp", 1] },
          avgMoisture: { $round: ["$avgMoisture", 1] },
          avgHumidity: { $round: ["$avgHumidity", 1] },
          alertsCount: 1,
        },
      },
      { $sort: { _id: 1 } },
    ]);

    res.status(200).json({
      success: true,
      daysRequested: days,
      count: report.length,
      data: report,
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.exportToCSV = async (req, res) => {
  try {
    const { sectorId } = req.query;

    if (!sectorId) {
      return res
        .status(400)
        .json({ success: false, message: "يجب تحديد معرف القطاع" });
    }

    // ✅ التعديل: التأكد من تحويل الـ String لـ ObjectId في الـ Find برضه لو واجهت مشاكل
    const data = await SensorData.find({
      sectorId: new mongoose.Types.ObjectId(sectorId),
    })
      .sort("-createdAt")
      .limit(500);

    if (data.length === 0) {
      return res.status(404).json({
        success: false,
        message: "لا توجد بيانات لتصديرها لهذا القطاع",
      });
    }

    let csv = "\ufeff";
    csv +=
      "التاريخ والوقت,درجة الحرارة,الرطوبة الجوية,رطوبة التربة,حالة النبات,التوصية\n";

    data.forEach((item) => {
      // ✅ استخدام توقيت القاهرة أو توقيت محلي ثابت
      const date = item.createdAt
        ? item.createdAt.toLocaleString("ar-EG", { timeZone: "Africa/Cairo" })
        : "غير مسجل";

      const temp = item.air?.temperature ?? "N/A";
      const hum = item.air?.humidity ?? "N/A";
      const soilMoist = item.soil?.moisture ?? "N/A";
      const status = item.analysis?.status ?? "N/A";

      // تنظيف التوصية من أي فواصل قد تبوظ ملف الـ CSV
      const recommendation = item.analysis?.recommendation
        ? item.analysis.recommendation.replace(/,/g, " - ")
        : "لا يوجد";

      csv += `${date},${temp},${hum},${soilMoist},${status},${recommendation}\n`;
    });

    res.setHeader("Content-Type", "text/csv; charset=utf-8");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename=Report-${sectorId}-${Date.now()}.csv`,
    );

    return res.status(200).send(csv);
  } catch (err) {
    console.error("Export Error:", err);
    res.status(500).json({ success: false, error: err.message });
  }
};
