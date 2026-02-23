require("dotenv").config();
const express = require("express");
const cors = require("cors");
const path = require("path");
const connectDB = require("./config/db");
const passport = require("passport");
require("./config/passport")(passport);
// 🛡️ Security Libraries
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const sanitizeHtml = require("sanitize-html");
const hpp = require("hpp");
const mongoSanitize = require("express-mongo-sanitize");
const morgan = require("morgan");

// Swagger (اختياري لو مفعله)
const swaggerUi = require("swagger-ui-express");
const swaggerSpec = require("./swagger");

const app = express();

// ============================
// 🔌 Connect Database
// ============================
connectDB();

// ============================
// 🛡️ Security Middlewares
// ============================
app.use(helmet()); // حماية الـ Headers
app.use(hpp()); // منع HTTP Parameter Pollution
app.use(mongoSanitize()); // منع هجمات NoSQL Injection
app.use(cors()); // السماح بالاتصال من الـ Flutter / Frontend
app.use(morgan("dev")); // لطباعة الـ Logs في الـ Console أثناء التطوير

app.use(passport.initialize());

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

// Rate Limiter — حد أقصى 100 طلب كل 15 دقيقة لكل IP
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
// جعل مجلد الصور متاحاً للوصول عبر الرابط
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
// 🛤️ Routes Mapping (الربط النهائي)
// ============================

// 1. الأساسيات (التشغيل)
app.get("/", (req, res) => res.send("EcoSense Backend Running Securely 🔐🚀"));

// 2. مستخدمين، عمال، وتوثيق (Auth & Users)
app.use("/api/auth", require("./routes/authRoutes"));
app.use("/api/users", require("./routes/userRoutes"));

// 3. إدارة المزرعة (Sectors & Devices)
app.use("/api/sectors", require("./routes/sectorRoutes"));
app.use("/api/devices", require("./routes/deviceRoutes"));

// 4. البيانات والتحليل (Sensors & AI)
app.use("/api/sensors", require("./routes/sensorRoutes"));

// 5. الصور (Plant Disease Detection)
app.use("/api/images", require("./routes/imageRoutes"));

// 6. لوحة التحكم والتنبيهات (Main Dashboard & Notifications)
app.use("/api/main", require("./routes/mainRoutes"));

app.use("/api/admin", require("./routes/adminRoutes"));

// تفعيل الروابط
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
// 🚀 Start Server
// ============================
const PORT = process.env.PORT || 6000;
app.listen(PORT, () => {
  console.log(`
    *****************************************
    🌐 EcoSense Secure Server is LIVE
    🚀 Port: ${PORT}
    🛡️ Mode: ${process.env.NODE_ENV || "development"}
    *****************************************
  `);
});
