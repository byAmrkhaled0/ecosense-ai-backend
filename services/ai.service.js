const axios = require("axios");

exports.getAIAnalysis = async ({
  cropType,
  temperature,
  humidity,
  soilMoisture,
  light,
}) => {
  try {
    const response = await axios.post(
      "https://amr2004-ecosense-ai.hf.space/api/predict_sensors",
      {
        cropType,
        temperature,
        humidity,
        soilMoisture,
        light,
      },
      {
        timeout: 6000,
      },
    );

    const data = response.data;
    console.log("AI RESPONSE:", response.data);

    return {
      status: data.final_status || "Safe",

      recommendation: data.recommendations
        ? data.recommendations.join(" | ")
        : "No Recommendation",
    };
  } catch (err) {
    console.log("❌ AI ERROR");

    if (err.response) {
      console.log(err.response.data);
    }

    console.log(err.message);

    return {
      status: "Warning",
      recommendation: "AI Server Offline",
    };
  }
};
