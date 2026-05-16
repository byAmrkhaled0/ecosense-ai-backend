const axios = require("axios");

// ================= Crop Formatting =================
const formatCropType = (crop) => {
  const crops = {
    maize: "Corn",
    corn: "Corn",
    tomato: "Tomato",
    pepper: "Pepper",
    mint: "Mint",
  };

  const normalized = String(crop || "")
    .trim()
    .toLowerCase();

  return crops[normalized] || "Corn";
};

// ================= Light Formatting =================
const formatLightValue = (light) => {
  const lightMap = {
    high: "Sufficient",
    sufficient: "Sufficient",

    medium: "Medium",

    low: "Low",
  };

  const normalized = String(light || "")
    .trim()
    .toLowerCase();

  return lightMap[normalized] || "Medium";
};

// ================= AI SERVICE =================
exports.getAIAnalysis = async ({
  cropType,
  temperature,
  humidity,
  soilMoisture,
  light,
}) => {
  try {
    // تنظيف البيانات
    const payload = {
      cropType: formatCropType(cropType),

      temperature: Number(temperature) || 0,

      humidity: Number(humidity) || 0,

      soilMoisture: Number(soilMoisture) || 0,

      light: formatLightValue(light),
    };

    console.log("📤 AI REQUEST:", payload);

    // Request
    const response = await axios.post(
      "https://amr2004-ecosense-ai.hf.space/api/predict_sensors",
      payload,
      {
        timeout: 8000,

        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
      },
    );

    const data = response.data;

    console.log("✅ AI RESPONSE:", data);

    return {
      status: data.final_status || data.status || "Safe",

      recommendation: Array.isArray(data.recommendations)
        ? data.recommendations.join(" | ")
        : data.summary || "No Recommendation",
    };
  } catch (err) {
    console.log("❌ AI ERROR");

    // Axios Response Error
    if (err.response) {
      console.log("STATUS:", err.response.status);

      console.log("DATA:", err.response.data);
    }

    // Timeout
    if (err.code === "ECONNABORTED") {
      console.log("⏰ AI TIMEOUT");
    }

    // Network Error
    if (err.code === "ENOTFOUND") {
      console.log("🌐 NETWORK ERROR");
    }

    console.log("MESSAGE:", err.message);

    // Fallback Response
    return {
      status: "Warning",

      recommendation: "AI Server Offline",
    };
  }
};
