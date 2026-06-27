// ========================
// LOAD NAVBAR + INIT PAGES
// ========================

document.addEventListener("DOMContentLoaded", function () {
  loadNavbar();
  handleSignup();
  handleOTP();
  handleLogin();
  handleDashboardActions();

});


function loadNavbar() {
  fetch("navbar.html")
    .then(res => res.text())
    .then(data => {
      const container = document.getElementById("navbar-container");
      if (container) {
        container.innerHTML = data;

        customizeNavbar();

      }
      initLang();
    })
    .catch(err => console.log("Navbar not loaded:", err));
}

// ========================
// CUSTOMIZE NAVBAR
// ========================

function customizeNavbar() {

  const currentPage = window.location.pathname;
  const navbar = document.querySelector(".navbar");
  if (!navbar) return;

  const rightSide = navbar.lastElementChild;

  // ===== صفحة Signup فقط =====
  if (currentPage.includes("signup.html")) {

    const langBtn = document.getElementById("langToggle");

    rightSide.innerHTML = "";

    if (langBtn) rightSide.appendChild(langBtn);

    const btn = document.createElement("button");
    btn.className = "already-btn";
    btn.setAttribute("data-i18n", "alreadyAccount");
    btn.innerText = "Already have an account?";
    btn.onclick = () => window.location.href = "login.html";

    rightSide.appendChild(btn);
  }

  // ===== صفحة OTP فقط =====
  if (currentPage.includes("verify-otp.html")) {

    const loginBtn = rightSide.querySelector(".login");
    const signupBtn = rightSide.querySelector(".signup");

    if (loginBtn) loginBtn.remove();
    if (signupBtn) signupBtn.remove();
  }
  // ===== صفحة Login فقط =====
  if (currentPage.includes("login.html")) {

    const rightSide = navbar.querySelector(".nav-actions");

    const loginBtn = rightSide.querySelector(".login");
    const signupBtn = rightSide.querySelector(".signup");

    if (loginBtn) loginBtn.remove();
    if (signupBtn) signupBtn.remove();

  }

}
// ========================
// LANGUAGE SYSTEM
// ========================


function initLang() {

  const btn = document.getElementById("langToggle");
  const savedLang = localStorage.getItem("lang") || "en";

  applyLang(savedLang);

  if (btn) {
    btn.onclick = function () {

      const currentLang = localStorage.getItem("lang") || "en";
      const newLang = currentLang === "en" ? "ar" : "en";

      applyLang(newLang);
    };
  }
}
function applyLang(lang) {

  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (translations[lang][key]) {
      el.innerText = translations[lang][key];
    }
  });

  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";

  localStorage.setItem("lang", lang);
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
    const key = el.getAttribute("data-i18n-placeholder");
    if (translations[lang] && translations[lang][key]) {
      el.placeholder = translations[lang][key];
    }
  });
}

// ========================
// NAVIGATION FUNCTIONS
// ========================

function goLogin() {
  window.location.href = "login.html";
}

function goSignup() {
  window.location.href = "signup.html";
}

// ========================
// SIGNUP LOGIC
// ========================

function handleSignup() {

  const signupForm = document.getElementById("signupForm");
  if (!signupForm) return;

  signupForm.addEventListener("submit", function (e) {
    e.preventDefault();

    const passwords = signupForm.querySelectorAll("input[type='password']");
    const password = passwords[0].value;
    const confirm = passwords[1].value;

    if (password !== confirm) {
      alert("Passwords do not match ❌");
      return;
    }

    const email = document.getElementById("email").value;

    localStorage.setItem("pendingEmail", email);

    window.location.href = "verify-otp.html";
  });
}

// ========================
// OTP LOGIC
// ========================

function handleOTP() {

  const otpForm = document.getElementById("otpForm");
  if (!otpForm) return;

  const inputs = document.querySelectorAll(".otp-inputs input");

  inputs.forEach((input, index) => {

    input.addEventListener("input", (e) => {

      // يمنع أي حاجة غير رقم
      input.value = input.value.replace(/[^0-9]/g, '');

      // لو كتب رقم يتحرك تلقائي
      if (input.value !== "" && index < inputs.length - 1) {
        inputs[index + 1].focus();
      }

    });

    // لو مسح يرجع للخانة اللي قبلها
    input.addEventListener("keydown", (e) => {
      if (e.key === "Backspace" && input.value === "" && index > 0) {
        inputs[index - 1].focus();
      }
    });

  });

  otpForm.addEventListener("submit", function (e) {
    e.preventDefault();

    let otp = "";
    inputs.forEach(input => otp += input.value);

    if (otp.length < 6) {
      alert("Enter full code ❌");
      return;
    }

    if (otp === "123456") {
      alert("OTP Verified ✅");
      window.location.href = "login.html";
    } else {
      alert("Wrong Code ❌");
    }

  });
}

// ========================
// LOGIN LOGIC
// ========================

function handleLogin() {

  const loginForm = document.getElementById("loginForm");
  if (!loginForm) return;

  loginForm.addEventListener("submit", function (e) {
    e.preventDefault();

    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    if (!email || !password) {
      alert("Please fill all fields");
      return;
    }

    // مؤقتاً: ندخل على الداشبورد مباشرة
    localStorage.setItem("isLoggedIn", "true");

    window.location.href = "dashboard.html";
  });

}
// GOOGLE

const googleBtn = document.getElementById("googleLogin");

if (googleBtn) {
  googleBtn.addEventListener("click", () => {

    // تحويل لصفحة تسجيل جوجل الرسمية
    window.location.href = "https://accounts.google.com/signin";

  });
}
// ========================
// DASHBOARD QUICK ACTIONS
// ========================

function handleDashboardActions() {

  const uploadBtn = document.getElementById("uploadBtn");
  const predictBtn = document.getElementById("predictBtn");

  if (uploadBtn) {
    uploadBtn.addEventListener("click", () => {
      window.location.href = "upload-data.html";
    });
  }

  if (predictBtn) {
    predictBtn.addEventListener("click", () => {
      window.location.href = "Ai-prediction.html";
    });
  }
}

/**************** AI Prediction Page ****************/

document.addEventListener("DOMContentLoaded", function () {

  const analyzeBtn = document.getElementById("analyzeBtn");

  if (analyzeBtn) {   // مهم عشان ميأثرش على باقي الصفحات

    const loading = document.getElementById("loadingBadge");
    const skeleton = document.getElementById("skeleton");

    analyzeBtn.addEventListener("click", function () {

      loading.style.display = "inline-block";
      skeleton.style.display = "block";

      setTimeout(() => {
        loading.style.display = "none";
        skeleton.style.display = "none";
      }, 2000);

    });

  }

});

/**************** IMAGE PREVIEW ****************/

document.addEventListener("DOMContentLoaded", function () {

  const input = document.getElementById("imageInput");
  const preview = document.getElementById("previewImage");

  if (input) {
    input.addEventListener("change", function () {
      const file = this.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
          preview.src = e.target.result;
          preview.hidden = false;
        }
        reader.readAsDataURL(file);
      }
    });
  }

});

/************** GO TO IMAGE HISTORY **************/

document.addEventListener("DOMContentLoaded", function () {

  const historyBtn = document.getElementById("goHistory");

  if (historyBtn) {
    historyBtn.addEventListener("click", function () {
      window.location.href = "image-history.html";
    });
  }

});

/******** DELETE IMAGE (UI ONLY) ********/

document.addEventListener("DOMContentLoaded", function () {

    const deleteBtns = document.querySelectorAll(".delete-btn");

    deleteBtns.forEach(btn => {
        btn.addEventListener("click", function () {
            btn.parentElement.remove();
        });
    });

}); 

/******** PROFESSIONAL MOBILE SIDEBAR ********/

document.addEventListener("DOMContentLoaded", function () {

    const menuBtn = document.getElementById("menuToggle");
    const sidebar = document.querySelector(".sidebar");
    const overlay = document.getElementById("sidebarOverlay");

    if (menuBtn) {
        menuBtn.addEventListener("click", function () {
            sidebar.classList.toggle("active");
            overlay.classList.toggle("active");
        });
    }

    if (overlay) {
        overlay.addEventListener("click", function () {
            sidebar.classList.remove("active");
            overlay.classList.remove("active");
        });
    }

});

// ===== Mobile Sidebar Toggle =====

document.addEventListener("DOMContentLoaded", function () {

  const menuToggle = document.getElementById("menuToggle");
  const sidebar = document.querySelector(".sidebar");

  if (!menuToggle) {
    console.log("menuToggle NOT FOUND");
    return;
  }

  if (!sidebar) {
    console.log("sidebar NOT FOUND");
    return;
  }

  menuToggle.addEventListener("click", function () {
    sidebar.classList.toggle("active");
  });

});
// ===== Sidebar Toggle (Bulletproof Version) =====

document.addEventListener("click", function (e) {

  const menuBtn = e.target.closest("#menuToggle");
  const sidebar = document.querySelector(".sidebar");

  if (menuBtn && sidebar) {
    sidebar.classList.toggle("active");
  }

});

function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");

    const message = input.value.trim();
    if (!message) return;

    chatBox.innerHTML += `<p><strong>You:</strong> ${message}</p>`;

    // رد مؤقت
    chatBox.innerHTML += `<p><strong>AI:</strong> ...</p>`;

    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;
}
// ===== Enter send =====
const input = document.getElementById("user-input");

if (input) {
    input.addEventListener("keypress", function (e) {
        if (e.key === "Enter") {
            sendMessage();
        }
    });
}

// ===== detect Arabic =====
function isArabic(text) {
    return /[\u0600-\u06FF]/.test(text);
}

// ===== chat function =====
function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");

    if (!input || !chatBox) return;

    const message = input.value.trim();
    if (!message) return;

    // user message
    const userMsg = document.createElement("div");
    userMsg.classList.add("message", "user-message");
    userMsg.textContent = message;
    chatBox.appendChild(userMsg);

    input.value = "";

    // ai message
    const aiMsg = document.createElement("div");
    aiMsg.classList.add("message", "ai-message");
    aiMsg.textContent = "...";
    chatBox.appendChild(aiMsg);

    chatBox.scrollTop = chatBox.scrollHeight;

    setTimeout(() => {
        if (isArabic(message)) {
            aiMsg.textContent = "تمام 👌 فهمت كلامك بالعربي!";
        } else {
            aiMsg.textContent = "Got it! I understand you 👍";
        }

        chatBox.scrollTop = chatBox.scrollHeight;
    }, 800);
}