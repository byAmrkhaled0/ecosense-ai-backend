const Notification = require("../models/Notification");
const User = require("../models/User");

exports.sendCriticalAlert = async ({
  io,
  ownerId,
  workerId,
  sectorName,
  sectorId,
  status,
  temperature,
}) => {
  try {
    const payload = {
      title: "🚨 تنبيه خطر",

      message: `القطاع: ${sectorName} | الحالة: ${status} | الحرارة: ${temperature}°C`,

      sectorId,

      createdAt: new Date(),
    };

    // Socket
    if (io) {
      if (ownerId) {
        io.to(ownerId.toString()).emit("newNotification", payload);
      }

      if (workerId) {
        io.to(workerId.toString()).emit("newNotification", payload);
      }
    }

    // Save DB
    const notifications = [];

    if (ownerId) {
      notifications.push({
        recipient: ownerId,
        sectorId,
        title: payload.title,
        message: payload.message,
        type: "warning",
      });
    }

    if (workerId) {
      notifications.push({
        recipient: workerId,
        sectorId,
        title: payload.title,
        message: payload.message,
        type: "warning",
      });
    }

    if (notifications.length) {
      await Notification.insertMany(notifications);
    }

    console.log("✅ Notifications Sent");
  } catch (err) {
    console.log("Notification Error:", err.message);
  }
};
