require("dotenv").config();
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const cors = require("cors");
const path = require("path");
const connectDB = require("./config/db");
const passport = require("passport");
const cookieParser = require("cookie-parser");

// 🛡️ Security Libraries
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const sanitizeHtml = require("sanitize-html");
const hpp = require("hpp");
const mongoSanitize = require("express-mongo-sanitize");
const morgan = require("morgan");

// Swagger
const swaggerUi = require("swagger-ui-express");
const swaggerSpec = require("./swagger");

// 1️⃣ إنشاء تطبيق Express أولاً
const app = express();
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

// 2️⃣ إنشاء سيرفر HTTP وربطه بـ Express
const server = http.createServer(app);

// 3️⃣ إعداد Socket.io
const io = new Server(server, {
  cors: {
    origin: (origin, callback) => callback(null, true),
    credentials: true,
  },
});

// تخزين الـ IO في الـ app لاستخدامه في الـ Controllers
app.set("io", io);

// تعريف منطق Socket.io
io.on("connection", (socket) => {
  console.log("🟢 A user connected: ", socket.id);

  socket.on("join", (userId) => {
    socket.join(userId);
    console.log(`👤 User ${userId} joined their private room`);
  });

  socket.on("disconnect", () => {
    console.log("🔴 User disconnected");
  });
});

// 🔌 الاتصال بقاعدة البيانات
connectDB();

// 🛡️ إعدادات Passport
require("./config/passport")(passport);
app.use(passport.initialize());

// ============================
// 🛡️ Security Middlewares
// ============================
app.use(helmet());
app.use(hpp());
app.use(mongoSanitize());
app.use(
  cors({
    origin: (origin, callback) => callback(null, true), // بيوافق على أي Origin باعت الطلب
    credentials: true,
  }),
);
app.use(morgan("dev"));

// 🧹 HTML Sanitization (XSS Prevention)
app.use((req, res, next) => {
  if (req.body) {
    Object.keys(req.body).forEach((key) => {
      if (typeof req.body[key] === "string") {
        req.body[key] = sanitizeHtml(req.body[key], {
          allowedTags: [],
          allowedAttributes: {},
        });
      }
    });
  }
  next();
});

// Rate Limiter
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
  message: "Too many requests from this IP, please try again after 15 minutes",
});
app.use("/api", limiter);

// Body Parser
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// ============================
// 📂 Static Folders
// ============================
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

// ============================
// 📖 Swagger Documentation
// ============================
app.use(
  "/api-docs",
  swaggerUi.serve,
  swaggerUi.setup(swaggerSpec, {
    swaggerOptions: { persistAuthorization: true },
    customSiteTitle: "EcoSense API Documentation",
  }),
);

// ============================
// 🛤️ Routes Mapping
// ============================
app.get("/", (req, res) => res.send("EcoSense Backend Running Securely 🔐🚀"));

app.use("/api/auth", require("./routes/authRoutes"));
app.use("/api/users", require("./routes/userRoutes"));
app.use("/api/sectors", require("./routes/sectorRoutes"));
app.use("/api/devices", require("./routes/deviceRoutes"));
app.use("/api/sensors", require("./routes/sensorRoutes"));
app.use("/api/images", require("./routes/imageRoutes"));
app.use("/api/main", require("./routes/mainRoutes"));
app.use("/api/admin", require("./routes/adminRoutes"));
app.use("/api/reports", require("./routes/reportRoutes"));

// ============================
// ❌ 404 Handler
// ============================
app.use((req, res, next) => {
  res.status(404).json({
    success: false,
    message: `Route not found: ${req.originalUrl}`,
  });
});

// ============================
// 🔥 Global Error Handler
// ============================
app.use((err, req, res, next) => {
  console.error("🔥 Server Error:", err.stack);
  res.status(500).json({
    success: false,
    error: err.message || "Internal Server Error",
  });
});

// ============================
// 🚀 Start Server (استخدام server.listen وليس app.listen)
// ============================
const PORT = process.env.PORT || 6000;
server.listen(PORT, () => {
  console.log(`
    *****************************************
    🌐 EcoSense Secure Server is LIVE
    🚀 Port: ${PORT}
    🛡️ Mode: ${process.env.NODE_ENV || "development"}
    📡 Socket.io: Enabled
    *****************************************
  `);
});
module.exports = app;
