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
        timeout: 8000,
      },
    );

    const data = response.data;

    console.log("✅ AI RESPONSE:", data.final_status);

    return {
      status: data.final_status || data.status || "Safe",

      recommendation:
        data.general_recommendation ||
        data.summary ||
        data.recommendations?.join(" | ") ||
        "No Recommendation",

      severity: data.severity || "low",

      alert: data.alert || false,

      riskFactors: data.risk_factors || [],
    };
  } catch (err) {
    console.log("❌ AI ERROR:", err.message);

    return {
      status: "Warning",
      recommendation: "AI Server Offline",
      severity: "unknown",
      alert: false,
    };
  }
};
