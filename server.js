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

const app = express();

// ============================
// 🛡️ Middlewares & Security (الترتيب الصحيح والمعدل القاطع للـ CORS)
// ============================

// 1️⃣ الـ Middleware اليدوي الصارم للرد على الـ Preflight والـ CORS فوراً قبل أي مكتبة تانية
app.use((req, res, next) => {
  const origin = req.headers.origin;
  if (origin) {
    res.setHeader("Access-Control-Allow-Origin", origin);
  }
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader(
    "Access-Control-Allow-Methods",
    "GET,HEAD,PUT,PATCH,POST,DELETE,OPTIONS",
  );
  res.setHeader(
    "Access-Control-Allow-Headers",
    "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization",
  );

  // إذا كان الطلب OPTIONS رد بـ 200 فوراً وقفل السكة عشان المتصفح يعدي الـ Preflight
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }
  next();
});

// تشغيل مكتبة CORS كأمان إضافي للطلبات الفعلية
app.use(cors({ origin: true, credentials: true }));
app.options("*", cors());

// الـ Body Parsers والـ Cookie Parser
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

app.use(morgan("dev"));

// 🛑 التعديل الجوهري: تعطيل الـ crossOriginResourcePolicy عشان Helmet ميبلوکش الـ Requests الخارجية
app.use(
  helmet({
    crossOriginResourcePolicy: { policy: "cross-origin" },
    crossOriginOpenerPolicy: { policy: "unsafe-none" },
  }),
);

app.use(hpp());
app.use(mongoSanitize());

// 🧹 دالة الـ HTML Sanitization (XSS Prevention)
app.use((req, res, next) => {
  if (
    req.body &&
    typeof req.body === "object" &&
    !req.headers["content-type"]?.includes("multipart/form-data")
  ) {
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

// 2️⃣ إنشاء سيرفر HTTP وربطه بـ Express
const server = http.createServer(app);

// 3️⃣ إعداد Socket.io (تعديل الـ origin ليكون مرن ويقبل الفرونت الجديد)
const io = new Server(server, {
  cors: {
    origin: true,
    credentials: true,
  },
  allowEIO3: true,
  transports: ["polling", "websocket"],
});

app.set("io", io);

app.get("/socket.io/", (req, res) => {
  res.status(200).end();
});

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

connectDB();
require("./config/passport")(passport);
app.use(passport.initialize());

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

app.use((req, res, next) => {
  res.status(404).json({
    success: false,
    message: `Route not found: ${req.originalUrl}`,
  });
});

app.use((err, req, res, next) => {
  console.error("🔥 Server Error:", err.stack);
  res.status(500).json({
    success: false,
    error: err.message || "Internal Server Error",
  });
});

const PORT = process.env.PORT || 6000;

module.exports = app;
