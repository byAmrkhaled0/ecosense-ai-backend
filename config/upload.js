
const multer = require("multer");
const path = require("path");



// إعداد التخزين
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/"); // كل الملفات تتخزن هنا
  },
  filename: (req, file, cb) => {
    const uniqueName = Date.now() + path.extname(file.originalname);
    cb(null, uniqueName);
  }
});

// فلترة الملفات (تأكد إن النوع مسموح بيه)
const fileFilter = (req, file, cb) => {
  const allowedTypes = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "video/mp4",
    "video/mkv",
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  ];

  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true); // الملف مقبول
  } else {
    cb(new Error("File type not allowed!"), false);
  }
};

const upload = multer({ storage, fileFilter });

module.exports = upload;