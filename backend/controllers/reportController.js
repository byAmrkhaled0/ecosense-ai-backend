const SensorData = require("../models/SensorData");
const ImageLog = require("../models/ImageLog");
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

    const sId = new mongoose.Types.ObjectId(sectorId);

    const report = await SensorData.aggregate([
      {
        $match: {
          sectorId: sId,
          createdAt: { $gte: startDate },
        },
      },
      {
        $group: {
          _id: { $dateToString: { format: "%Y-%m-%d", date: "$createdAt" } },
          avgTemp: { $avg: "$air.temperature" },
          maxTemp: { $max: "$air.temperature" }, // التعديل: إضافة أعلى حرارة
          minTemp: { $min: "$air.temperature" }, // التعديل: إضافة أقل حرارة
          avgMoisture: { $avg: "$soil.moisture" },
          avgHumidity: { $avg: "$air.humidity" },
        },
      },
      // التعديل الجوهري: ربط تقرير السنسورز بتقرير الصور (AI)
      {
        $lookup: {
          from: "imagelogs",
          let: { reportDate: "$_id" },
          pipeline: [
            {
              $match: {
                $expr: {
                  $and: [
                    { $eq: ["$sectorId", sId] },
                    {
                      $eq: [
                        {
                          $dateToString: {
                            format: "%Y-%m-%d",
                            date: "$createdAt",
                          },
                        },
                        "$$reportDate",
                      ],
                    },
                    { $eq: ["$analysisResult.status", "Infected"] },
                  ],
                },
              },
            },
            { $count: "count" },
          ],
          as: "diseaseIncidents",
        },
      },
      {
        $project: {
          _id: 1,
          avgTemp: { $round: ["$avgTemp", 1] },
          maxTemp: { $round: ["$maxTemp", 1] },
          minTemp: { $round: ["$minTemp", 1] },
          avgMoisture: { $round: ["$avgMoisture", 1] },
          avgHumidity: { $round: ["$avgHumidity", 1] },
          // تحويل نتيجة الـ lookup لرقم بسيط
          totalDiseasesFound: {
            $ifNull: [{ $arrayElemAt: ["$diseaseIncidents.count", 0] }, 0],
          },
        },
      },
      { $sort: { _id: 1 } },
    ]);

    res.status(200).json({
      success: true,
      sectorId,
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

    // جلب آخر 1000 قراءة لضمان شمولية التقرير
    const data = await SensorData.find({
      sectorId: new mongoose.Types.ObjectId(sectorId),
    })
      .sort("-createdAt")
      .limit(1000)
      .lean();

    if (data.length === 0) {
      return res
        .status(404)
        .json({ success: false, message: "لا توجد بيانات متاحة لهذا القطاع" });
    }

    // إضافة Byte Order Mark (BOM) عشان الـ Excel يفهم إن الملف UTF-8 ويظهر العربي صح
    let csv = "\ufeff";
    csv +=
      "التاريخ,الوقت,الحرارة (C),الرطوبة الجوية (%),رطوبة التربة (%),الحالة التحذيرية\n";

    data.forEach((item) => {
      const dateObj = new Date(item.createdAt);
      const date = dateObj.toLocaleDateString("ar-EG");
      const time = dateObj.toLocaleTimeString("ar-EG");

      const temp = item.air?.temperature ?? "-";
      const hum = item.air?.humidity ?? "-";
      const soil = item.soil?.moisture ?? "-";
      const status = item.analysis?.status || "Normal";

      csv += `${date},${time},${temp},${hum},${soil},${status}\n`;
    });

    // إعدادات الرد لتحميل الملف
    res.setHeader("Content-Type", "text/csv; charset=utf-8");
    res.setHeader(
      "Content-Disposition",
      `attachment; filename=Sector-Report-${Date.now()}.csv`,
    );

    return res.status(200).send(csv);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};
