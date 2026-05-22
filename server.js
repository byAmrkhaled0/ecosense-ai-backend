require("dotenv").config();
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const cors = require("cors");
const path = require("path");
const connectDB = require("./config/db");
const passport = require("passport");
const cookieParser = require("cookie-parser");

const helmet = require("helmet");
const rateLimit = require("express-rate-limit");
const sanitizeHtml = require("sanitize-html");
const hpp = require("hpp");
const mongoSanitize = require("express-mongo-sanitize");
const morgan = require("morgan");

const swaggerUi = require("swagger-ui-express");
const swaggerSpec = require("./swagger");

// ======================
// APP INIT
// ======================
const app = express();
const server = http.createServer(app);

// ======================
// CORS
// ======================
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

// ======================
// BASIC MIDDLEWARES
// ======================
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

app.use(morgan("dev"));
app.use(helmet());
app.use(hpp());
app.use(mongoSanitize());

// XSS sanitize
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

// ======================
// SOCKET.IO SETUP (Vercel-safe)
// ======================
const io = new Server(server, {
  cors: {
    origin: [
      "http://localhost:3000",
      "http://localhost:5173",
      "https://your-frontend-domain.com",
    ],
    credentials: true,
  },
  transports: ["polling", "websocket"], // IMPORTANT for Vercel
});

app.set("io", io);

// ======================
// SOCKET EVENTS
// ======================
io.on("connection", (socket) => {
  console.log("🟢 Connected:", socket.id);

  socket.on("join", (userId) => {
    if (!userId) return;

    socket.join(userId);
    console.log(`👤 Joined room: ${userId}`);
  });

  socket.on("disconnect", (reason) => {
    console.log("🔴 Disconnected:", socket.id, reason);
  });
});

// ======================
// DB
// ======================
connectDB();

// ======================
// PASSPORT
// ======================
require("./config/passport")(passport);
app.use(passport.initialize());

// ======================
// STATIC
// ======================
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

// ======================
// SWAGGER
// ======================
app.use("/api-docs", swaggerUi.serve, swaggerUi.setup(swaggerSpec));

// ======================
// ROUTES
// ======================
app.get("/", (req, res) => {
  res.send("EcoSense Backend Running 🚀");
});

app.use("/api/auth", require("./routes/authRoutes"));
app.use("/api/users", require("./routes/userRoutes"));
app.use("/api/sectors", require("./routes/sectorRoutes"));
app.use("/api/devices", require("./routes/deviceRoutes"));
app.use("/api/sensors", require("./routes/sensorRoutes"));
app.use("/api/images", require("./routes/imageRoutes"));
app.use("/api/main", require("./routes/mainRoutes"));
app.use("/api/admin", require("./routes/adminRoutes"));
app.use("/api/reports", require("./routes/reportRoutes"));

// ======================
// TEST SOCKET ROUTE
// ======================
app.get("/test-socket/:id", (req, res) => {
  const { id } = req.params;

  io.to(id).emit("newNotification", {
    title: "Test Notification",
    message: "Socket is working 🚀",
  });

  res.json({ success: true });
});

// ======================
// 404
// ======================
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: "Route not found",
  });
});

// ======================
// ERROR HANDLER
// ======================
app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({
    success: false,
    message: err.message || "Server Error",
  });
});

// ======================
// START SERVER
// ======================
const PORT = process.env.PORT || 6000;

server.listen(PORT, () => {
  console.log(`
🚀 Server running on ${PORT}
🔌 Socket.io enabled
🛡️ Secure mode: ${process.env.NODE_ENV || "dev"}
  `);
});

module.exports = app;
