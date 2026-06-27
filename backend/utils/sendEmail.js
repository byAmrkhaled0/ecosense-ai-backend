const nodemailer = require("nodemailer");

// ✅ تم تعديل المدخلات لاستقبال كائن options كما في الكنترولر
const sendEmail = async (options) => {
  try {
    const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS,
      },
    });

    await transporter.sendMail({
      from: `"EcoSense Alerts" <${process.env.EMAIL_USER}>`,
      to: options.email, // ⬅️ يستخدم options.email
      subject: options.subject, // ⬅️ يستخدم options.subject
      html: `
      <div style="font-family: Arial; padding: 10px;"> 
<h2 style="color: red;">🌱 EcoSense Alert</h2>
<p>${options.message}</p> // ⬅️ يستخدم options.message
<hr/>
<small>This is an automated alert from EcoSense.</small>
 </div>
`,
    });

    console.log("📧 Email sent successfully");
  } catch (error) {
    console.log("Email Error:", error.message);
    // ⚠️ من المهم تمرير الخطأ في وضع الإنتاج
    throw new Error(`Email sending failed: ${error.message}`);
  }
};

module.exports = sendEmail;
