const axios = require("axios");

// ✅ يجب تعريف مفتاح السيرفر في .env
const FIREBASE_SERVER_KEY = process.env.FIREBASE_SERVER_KEY;

exports.sendMobileNotification = async (token, title, message) => {
  try {
    // ⚠️ تحقق من وجود المفتاح أولاً
    if (
      !FIREBASE_SERVER_KEY ||
      FIREBASE_SERVER_KEY === "YOUR_FIREBASE_SERVER_KEY"
    ) {
      console.warn(
        "FCM Warning: FIREBASE_SERVER_KEY is not set or is a placeholder."
      );
      return;
    }

    await axios.post(
      "https://fcm.googleapis.com/fcm/send",
      {
        to: token, // رمز الجهاز المستهدف (FCM Token)
        notification: {
          title,
          body: message,
        },
      },
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `key=${FIREBASE_SERVER_KEY}`, // ✅ استخدام المتغير البيئي
        },
      }
    );
    console.log("🔔 FCM Notification sent successfully");
  } catch (err) {
    console.log("FCM Error:", err.message);
  }
};
