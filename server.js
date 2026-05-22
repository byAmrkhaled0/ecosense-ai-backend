require("dotenv").config();
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const cors = require("cors");
const path = require("path");
const connectDB = require("./config/db");
const passport = require("passport");
const cookieParser = require("cookie-parser");

// 🛡️ Security
const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const sanitizeHtml = require("sanitize-html");
const hpp = require("hpp");
const mongoSanitize = require("express-mongo-sanitize");
const morgan = require("morgan");

// Swagger
const swaggerUi = require("swagger-ui-express");
const swaggerSpec = require("./swagger");

// ============================
// 1️⃣ App init
// ============================
const app = express();
const server = http.createServer(app);

// ============================
// 🛡️ Middlewares
// ============================

app.use(
  cors({
    origin: [
      "http://localhost:3000",
      "http://localhost:5173",
      "https://your-frontend-domain.com",
    ],
    methods: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
    credentials: true,
  }),
);

app.options("*", cors());

app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

app.use(morgan("dev"));
app.use(helmet());
app.use(hpp());
app.use(mongoSanitize());

// 🧹 XSS clean
app.use((req, res, next) => {
  if (req.body && typeof req.body === "object") {
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

// Rate limit
app.use(
  "/api",
  rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100,
  }),
);

// ============================
// 2️⃣ Socket.io Setup
// ============================

const io = new Server(server, {
  cors: {
    origin: [
      "http://localhost:3000",
      "http://localhost:5173",
      "https://your-frontend-domain.com",
    ],
    credentials: true,
  },
  transports: ["websocket"],
});

app.set("io", io);

// ❌ IMPORTANT: DO NOT add /socket.io route (it breaks socket handshake)

// ============================
// Socket logic
// ============================
io.on("connection", (socket) => {
  console.log("🟢 Connected:", socket.id);

  // join room
  socket.on("join", (userId) => {
    if (!userId) return;

    socket.join(userId);
    console.log(`👤 User joined room: ${userId}`);
  });

  socket.on("disconnect", (reason) => {
    console.log("🔴 Disconnected:", socket.id, reason);
  });
});

// ============================
// 3️⃣ DB
// ============================
connectDB();

// ============================
// 4️⃣ Passport
// ============================
require("./config/passport")(passport);
app.use(passport.initialize());

// ============================
// 5️⃣ Static
// ============================
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

// ============================
// 6️⃣ Swagger
// ============================
app.use("/api-docs", swaggerUi.serve, swaggerUi.setup(swaggerSpec));

// ============================
// 7️⃣ Routes
// ============================
app.get("/", (req, res) => res.send("EcoSense Backend Running 🚀"));

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
// 8️⃣ Test Socket Route (IMPORTANT)
// ============================
app.get("/test-socket/:id", (req, res) => {
  const { id } = req.params;

  io.to(id).emit("newNotification", {
    title: "Test Notification",
    message: "Socket is working perfectly 🚀",
  });

  res.json({ success: true });
});

// ============================
// 9️⃣ Error handling
// ============================
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: `Route not found: ${req.originalUrl}`,
  });
});

app.use((err, req, res, next) => {
  console.error("🔥 Error:", err);
  res.status(500).json({
    success: false,
    message: err.message || "Server Error",
  });
});

// ============================
// 🚀 Start server
// ============================
const PORT = process.env.PORT || 6000;

server.listen(PORT, () => {
  console.log(`
🚀 Server running on port ${PORT}
🔌 Socket.io enabled
🛡️ Secure mode: ${process.env.NODE_ENV || "dev"}
  `);
});

module.exports = app;
