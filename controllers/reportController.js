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
    startDate.setHours(0, 0, 0, 0); // نبدأ من أول اليوم

    const report = await SensorData.aggregate([
      {
        $match: {
          sectorId: new mongoose.Types.ObjectId(sectorId),
          // التعديل هنا: نستخدم createdAt بدل timestamp
          createdAt: { $gte: startDate },
        },
      },
      {
        $group: {
          _id: {
            // التعديل هنا: استخراج التاريخ من createdAt
            $dateToString: { format: "%Y-%m-%d", date: "$createdAt" },
          },
          avgTemp: { $avg: "$air.temperature" },
          avgMoisture: { $avg: "$soil.moisture" },
          avgHumidity: { $avg: "$air.humidity" },
          // نعد أي حالة مش Healthy عشان تبان كـ Alert
          alertsCount: {
            $sum: {
              $cond: [
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
      { $sort: { _id: 1 } }, // ترتيب الأيام (السبت، الأحد، الاثنين...)
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

    // بنجيب البيانات وبنرتبها من الأحدث للأقدم
    const data = await SensorData.find({ sectorId })
      .sort("-createdAt") // التعديل هنا: نستخدم createdAt
      .limit(500); // زودنا الليميت شوية عشان التقرير يبقى محترم

    if (data.length === 0) {
      return res
        .status(404)
        .json({
          success: false,
          message: "لا توجد بيانات لتصديرها لهذا القطاع",
        });
    }

    // \ufeff ده الـ Byte Order Mark (BOM) عشان Excel يفهم إن الملف UTF-8 ويقرأ العربي صح
    let csv = "\ufeff";
    csv +=
      "التاريخ والوقت,درجة الحرارة,الرطوبة الجوية,رطوبة التربة,حالة النبات,التوصية\n";

    data.forEach((item) => {
      // 1. استخراج التاريخ (التعديل الأساسي هنا)
      const date = item.createdAt
        ? item.createdAt.toLocaleString("ar-EG")
        : "غير مسجل";

      // 2. تجهيز البيانات (بنتأكد إن مفيش قيم ناقصة عشان الملف ميبوظش)
      const temp = item.air?.temperature ?? "N/A";
      const hum = item.air?.humidity ?? "N/A";
      const soilMoist = item.soil?.moisture ?? "N/A";
      const status = item.analysis?.status ?? "N/A";
      const recommendation = item.analysis?.recommendation
        ? item.analysis.recommendation.replace(/,/g, "-")
        : "لا يوجد";

      // إضافة السطر للـ CSV
      csv += `${date},${temp},${hum},${soilMoist},${status},${recommendation}\n`;
    });

    // إرسال الملف للمتصفح أو Postman
    res.setHeader("Content-Type", "text/csv; charset=utf-8");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename=sector-report-${Date.now()}.csv`,
    );

    return res.status(200).send(csv);
  } catch (err) {
    console.error("Export Error:", err);
    res.status(500).json({ success: false, error: err.message });
  }
};
