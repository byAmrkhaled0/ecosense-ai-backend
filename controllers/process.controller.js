const { processSensorData } = require("../services/sensor.processor");

exports.processSensor = async (req, res) => {
  try {
    const { id } = req.params;

    await processSensorData(id, req.app.get("io"));

    res.status(200).json({
      success: true,
      message: "Processed",
    });
  } catch (err) {
    console.log(err);

    res.status(500).json({
      success: false,
    });
  }
};
