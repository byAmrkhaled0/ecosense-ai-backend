const GoogleStrategy = require("passport-google-oauth20").Strategy;
const User = require("../models/User");

module.exports = function (passport) {
  passport.use(
    new GoogleStrategy(
      {
        clientID: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
        // callbackURL: "http://localhost:3000/api/auth/google/callback",
        callbackURL:
          "https://ecosense-backend.vercel.app/api/auth/google/callback",
        proxy: true, // مهم جداً لو رفعت الموقع على (Heroku/Render) عشان الـ SSL
      },

      async (accessToken, refreshToken, profile, done) => {
        try {
          const email = profile.emails[0].value.toLowerCase();

          // ابحث عن المستخدم بالإيميل
          let user = await User.findOne({ email });

          if (user) {
            // لو المستخدم موجود، حدث الـ googleId والـ provider كنوع من التأكيد
            user.googleId = profile.id;
            user.provider = "google";
            await user.save({ validateBeforeSave: false });
            return done(null, user);
          }

          // لو المستخدم جديد، انشئه بالبيانات اللي جاية من جوجل مباشرة
          user = await User.create({
            email,
            // جوجل بتبعت الاسم الأول والأخير جاهزين هنا
            firstName: profile.name.givenName || "Google",
            lastName: profile.name.familyName || "User",
            provider: "google",
            googleId: profile.id,
            role: "owner", // افتراضياً أي حد يسجل بجوجل هو صاحب مزرعة
            isVerified: true,
            address: "Social Account",
            phoneNumber: "0000000000",
            // بنحط باسوورد عشوائي عشان الموديل ميعملش Validation error
            password: Math.random().toString(36).slice(-10),
          });

          return done(null, user);
        } catch (err) {
          console.error("Google Auth Strategy Error:", err);
          return done(err, null);
        }
      },
    ),
  );

  // السيرياليزيشن عشان الجلسات (Sessions)
  passport.serializeUser((user, done) => {
    done(null, user.id);
  });

  passport.deserializeUser(async (id, done) => {
    try {
      const user = await User.findById(id);
      done(null, user);
    } catch (err) {
      done(err, null);
    }
  });
};
