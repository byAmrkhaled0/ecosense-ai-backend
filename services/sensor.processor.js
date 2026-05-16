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

    // AI Analysis
    const aiAnalysis = await getAIAnalysis({
      cropType: sector.cropType,

      temperature: sensor.air.temperature,

      humidity: sensor.air.humidity,

      soilMoisture: sensor.soil.moisture,

      light: sensor.light,
    });

    // Update Sensor Data
    sensor.analysis = aiAnalysis;

    await sensor.save();

    console.log("✅ AI Analysis Saved");

    // Critical Check
    const isCritical =
      aiAnalysis.status === "Critical" ||
      aiAnalysis.status === "Danger" ||
      sensor.air.temperature > 45 ||
      sensor.soil.moisture < 10;

    if (isCritical) {
      await sendCriticalAlert({
        io,

        ownerId: sensor.ownerId,

        workerId: sector.assignedWorker,

        sectorName: sector.name,

        sectorId: sector._id,

        status: aiAnalysis.status,

        temperature: sensor.air.temperature,
      });
    }

    console.log("✅ Sensor Processing Complete");
  } catch (err) {
    console.error("Processor Error:", err.message);
  }
};
