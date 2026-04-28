const multer = require("multer");
const cloudinary = require("cloudinary").v2;
const { CloudinaryStorage } = require("multer-storage-cloudinary");

// 1️⃣ إعداد إعدادات Cloudinary
// (المفاتيح دي هتضيفها في الـ Environment Variables في Vercel)
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

// 2️⃣ إعداد التخزين على Cloudinary
const storage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: async (req, file) => {
    // تحديد نوع الفولدر بناءً على نوع الملف
    let folderName = "ecosense/others";
    if (file.mimetype.startsWith("image/")) folderName = "ecosense/images";
    if (file.mimetype.startsWith("video/")) folderName = "ecosense/videos";
    if (file.mimetype === "application/pdf") folderName = "ecosense/docs";

    return {
      folder: folderName,
      resource_type: "auto", // مهم جداً عشان يقبل (صور، فيديو، أو ملفات)
      public_id: Date.now() + "-" + file.originalname.split(".")[0],
    };
  },
});

// 3️⃣ فلترة الملفات (زي ما هي عندك)
const fileFilter = (req, file, cb) => {
  const allowedTypes = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "video/mp4",
    "video/mkv",
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ];

  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error("File type not allowed!"), false);
  }
};

const upload = multer({ storage, fileFilter });

module.exports = upload;
