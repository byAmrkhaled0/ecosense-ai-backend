const SensorData = require("../models/SensorData");

const { getAIAnalysis } = require("./ai.service");

const { sendCriticalAlert } = require("./notification.service");

exports.processSensorData = async (sensorId, io) => {
  try {
    const sensor = await SensorData.findById(sensorId)
      .populate("sectorId")
      .populate("deviceId");

    if (!sensor) return;

    const sector = sensor.sectorId;

    // ================= AI ANALYSIS =================
    const aiAnalysis = await getAIAnalysis({
      cropType: sector.cropType,

      temperature: sensor.air.temperature,

      humidity: sensor.air.humidity,

      soilMoisture: sensor.soil.moisture,

      light: sensor.light,
    });

    // ================= SAVE =================
    sensor.analysis = aiAnalysis;

    await sensor.save();

    console.log("✅ AI Analysis Saved:", aiAnalysis.status);

    // ================= CRITICAL LOGIC (UPDATED) =================
    const isCritical =
      aiAnalysis.alert === true ||
      aiAnalysis.severity === "high" ||
      aiAnalysis.status === "High Stress" ||
      aiAnalysis.status === "Moderate Stress" ||
      sensor.air.temperature > 45 ||
      sensor.soil.moisture < 10;

    if (isCritical) {
      console.log("🚨 CRITICAL DETECTED");

      await sendCriticalAlert({
        io,

        ownerId: sensor.ownerId,

        workerId: sector.assignedWorker,

        sectorName: sector.name,

        sectorId: sector._id,

        status: aiAnalysis.status,

        temperature: sensor.air.temperature,

        severity: aiAnalysis.severity,

        riskFactors: aiAnalysis.riskFactors,
      });
    }

    // ================= REALTIME UPDATE (OPTIONAL BUT IMPORTANT) =================
    if (io && sensor.ownerId) {
      io.to(sensor.ownerId.toString()).emit("sensorUpdated", {
        sensorId: sensor._id,
        analysis: aiAnalysis,
      });
    }

    console.log("✅ Sensor Processing Complete");
  } catch (err) {
    console.error("Processor Error:", err.message);
  }
};
