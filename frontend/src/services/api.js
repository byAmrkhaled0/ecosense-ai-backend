import axios from "axios";

export const PRODUCTION_BACKEND_ORIGIN = "https://ecosense-backend.vercel.app";
export const PRODUCTION_BACKEND_URL = `${PRODUCTION_BACKEND_ORIGIN}/api`;
export const AI_MODEL_BASE_URL = import.meta.env.VITE_AI_MODEL_BASE_URL || "/ai-model/api";
export const AI_COMBINED_MODEL_BASE_URL = import.meta.env.VITE_AI_COMBINED_MODEL_BASE_URL || "/ai-combined/api";

// Browser requests use a same-origin /api proxy by default to avoid CORS/Network Error.
// Vite dev server proxies /api to the real backend, and Vercel rewrites /api to:
// https://ecosense-backend.vercel.app/api
// Google OAuth still uses the real backend origin for the full browser redirect.
const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL;
export const API_BASE_URL = configuredApiBaseUrl || "/api";
// Default sector used by the hardware demo endpoints supplied with the project.
// It is only used as a fallback when the page is not already scoped to a selected sector.
export const DEFAULT_HARDWARE_SECTOR_ID = import.meta.env.VITE_DEFAULT_HARDWARE_SECTOR_ID || "";

export const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || PRODUCTION_BACKEND_ORIGIN;

const API = axios.create({
  baseURL: API_BASE_URL,
  timeout: Number(import.meta.env.VITE_API_TIMEOUT || 15000),
  withCredentials: import.meta.env.VITE_API_WITH_CREDENTIALS === "true",
});

const AI_MODEL_API = axios.create({
  baseURL: AI_MODEL_BASE_URL,
  timeout: Number(import.meta.env.VITE_AI_MODEL_TIMEOUT || 45000),
});

const AI_COMBINED_MODEL_API = axios.create({
  baseURL: AI_COMBINED_MODEL_BASE_URL,
  timeout: Number(import.meta.env.VITE_AI_MODEL_TIMEOUT || 60000),
});

export const normalizeStoredToken = (rawToken = "") => {
  let token = String(rawToken || "").trim();
  if (!token) return "";
  try { token = decodeURIComponent(token); } catch {}
  return token.trim().replace(/^['\"]|['\"]$/g, "").replace(/^Bearer\s+/i, "").trim();
};

export const getAuthToken = () =>
  normalizeStoredToken(
    localStorage.getItem("ecosense_token") ||
    localStorage.getItem("sph_token") ||
    localStorage.getItem("token") ||
    localStorage.getItem("user_token") ||
    localStorage.getItem("accessToken")
  );

export const saveAuthSession = (token, user) => {
  const cleanToken = normalizeStoredToken(token);
  if (cleanToken) {
    localStorage.setItem("ecosense_token", cleanToken);
    localStorage.setItem("sph_token", cleanToken);
  }
  if (user) localStorage.setItem("ecosense_user", JSON.stringify(user));
};

export const clearAuthSession = () => {
  const accountDataPrefixes = [
    "sph_worker_accounts", "sph_sectors", "sph_readings", "sph_plants", "sph_tasks_global",
    "sph_notifications", "sph_settings", "sph_account_settings", "sph_onboarding",
    "sph_devices", "sph_reports", "sph_alerts", "sph_farm_data", "sph_diagnoses",
    "sph_images_history", "sph_sensors_history"
  ];
  Object.keys(localStorage).forEach((key) => {
    if (accountDataPrefixes.some((prefix) => key === prefix || key.startsWith(`${prefix}_`) || key.startsWith(`${prefix}__account__`))) {
      localStorage.removeItem(key);
    }
  });
  [
    "ecosense_token",
    "ecosense_user",
    "sph_auth",
    "sph_token",
    "sph_role",
    "sph_user_email",
    "token",
    "user_token",
    "accessToken",
  ].forEach((key) => localStorage.removeItem(key));
};

API.interceptors.request.use((config) => {
  const token = getAuthToken();
  if (token && token !== "local-demo-token" && token !== "local-worker-token") {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

API.interceptors.response.use(
  (response) => response,
  (error) => {
    const url = error.config?.url || "";
    const isAuthRoute =
      url.includes("/auth/login") ||
      url.includes("/auth/register") ||
      url.includes("/auth/verify-otp") ||
      url.includes("/auth/verify") ||
      url.includes("/auth/google") ||
      url.includes("/auth/google-auth") ||
      url.includes("/auth/forgot-password") ||
      url.includes("/auth/reset-password");

    // Let auth pages show the exact backend validation message.
    if (isAuthRoute) return Promise.reject(error);

    if (error.response?.status === 401) {
      clearAuthSession();
      if (window.location.pathname !== "/login") window.location.href = "/login";
    }
    return Promise.reject(error);
  },
);

export const getApiError = (error) => {
  const status = error?.response?.status || 0;
  const backendMessage =
    error?.response?.data?.message ||
    error?.response?.data?.error ||
    error?.response?.data?.details ||
    error?.message;
  return {
    endpoint: error?.config?.url || "unknown",
    method: String(error?.config?.method || "GET").toUpperCase(),
    status,
    message: backendMessage || (status ? "Request failed" : "Network Error"),
    data: error?.response?.data,
  };
};

const toFetchResponse = (axiosPromise) =>
  axiosPromise.then((res) => ({
    ok: res.status >= 200 && res.status < 300,
    status: res.status,
    json: async () => res.data,
  }));

const splitName = (fullName = "") => {
  const parts = String(fullName || "").trim().split(/\s+/).filter(Boolean);
  return {
    firstName: parts[0] || "EcoSense",
    lastName: parts.slice(1).join(" ") || "User",
  };
};

const normalizeRegisterPayload = (data = {}) => {
  const names = splitName(
    data.fullName || data.name || `${data.firstName || ""} ${data.lastName || ""}`,
  );
  return {
    firstName: data.firstName || names.firstName,
    lastName: data.lastName || names.lastName,
    email: data.email,
    password: data.password,
    phoneNumber: data.phoneNumber || data.phone,
    phone: data.phone || data.phoneNumber,
    address: data.address,
    role: data.role,
  };
};

const normalizeLoginResponse = (res) => {
  const payload = res.data || {};
  const token = payload.token || payload.accessToken || payload.data?.token || payload.data?.accessToken;
  const rawUser = payload.user || payload.data?.user || payload.currentUser || payload.data || {};
  const fullName =
    rawUser.name ||
    rawUser.fullName ||
    rawUser.displayName ||
    [rawUser.firstName, rawUser.lastName].filter(Boolean).join(" ");
  return {
    ...res,
    data: {
      ...payload,
      token,
      user: {
        ...rawUser,
        name: fullName || rawUser.email || rawUser.username || "EcoSense User",
        phone: rawUser.phone || rawUser.phoneNumber,
        role: rawUser.role || rawUser.accountType || "owner",
        sectorId: rawUser.sectorId || rawUser.assignedSectorId || rawUser.sector?._id,
        sector: rawUser.sector || rawUser.assignedSector || rawUser.sectorName,
      },
    },
  };
};

const normalizeImageFormData = (formData) => {
  if (!(formData instanceof FormData)) return formData;
  const image = formData.get("image") || formData.get("photo") || formData.get("file");
  if (image && !formData.has("image")) formData.append("image", image);
  return formData;
};

const normalizeModelImageFormData = (formData) => {
  if (!(formData instanceof FormData)) return formData;
  const file = formData.get("file") || formData.get("image") || formData.get("photo") || formData.get("plantImage");
  if (file && !formData.has("file")) formData.append("file", file);
  return formData;
};

export const authAPI = {
  login: (data = {}) => {
    const identifier = String(data.identifier || data.email || data.username || '').trim();
    return API.post("/auth/login", {
      ...data,
      email: data.email || identifier,
      username: data.username || identifier,
      identifier,
      login: identifier,
    }).then(normalizeLoginResponse);
  },
  register: (data) => API.post("/auth/register", normalizeRegisterPayload(data)),
  verifyOTP: (data) => API.post("/auth/verify-otp", data),
  getMe: () => API.get("/auth/me").then(normalizeLoginResponse),
  logout: () => API.post("/auth/logout"),
  forgotPassword: (emailOrData) => {
    const payload = typeof emailOrData === "string" ? { email: emailOrData } : emailOrData;
    return API.post("/auth/forgot-password", payload);
  },
  resetPassword: ({ code, newPassword, resetToken }) =>
    API.post("/auth/reset-password", { code, newPassword, resetToken }),
  // Google OAuth must be a full browser redirect, not axios/fetch, to avoid CORS blocks.
  // Login/Register buttons should navigate to this URL with redirect_to=<frontend-origin>.
  googleAuth: (data) => API.post("/auth/google-auth", data).then(normalizeLoginResponse),
  googleRedirectUrl: () => `${PRODUCTION_BACKEND_ORIGIN}/api/auth/google?redirect_to=${encodeURIComponent(window.location.origin)}`,
};


// Production diagnosis uses the real Ecosense backend only: /sensors/analyze/:sectorId and /images/upload.

export const dashboardAPI = {
  get: () => API.get("/main/dashboard"),
};

let notificationsCache = { time: 0, promise: null, response: null };
const NOTIFICATIONS_MIN_INTERVAL = 45000;

export const notificationsAPI = {
  getAll: ({ force = false } = {}) => {
    const now = Date.now();
    if (!force && notificationsCache.response && now - notificationsCache.time < NOTIFICATIONS_MIN_INTERVAL) {
      return Promise.resolve(notificationsCache.response);
    }
    if (!force && notificationsCache.promise) return notificationsCache.promise;

    notificationsCache.promise = API.get("/main/notifications")
      .then((response) => {
        notificationsCache = { time: Date.now(), promise: null, response };
        return response;
      })
      .catch((error) => {
        notificationsCache.promise = null;
        throw error;
      });

    return notificationsCache.promise;
  },
  markRead: (id) => { notificationsCache = { time: 0, promise: null, response: null }; return API.patch(`/main/notifications/${id}`); },
  delete: (id) => { notificationsCache = { time: 0, promise: null, response: null }; return API.delete(`/main/notifications/${id}`); },
};

export const sectorsAPI = {
  getAll: () => API.get("/sectors"),
  create: (data) => API.post("/sectors", data),
  update: (id, data) => API.put(`/sectors/${id}`, data),
  delete: (id) => API.delete(`/sectors/${id}`),
};

export const devicesAPI = {
  getAll: () => API.get("/devices"),
  create: (data) => API.post("/devices", data),
  delete: (id) => API.delete(`/devices/${id}`),
};

const cleanParams = (params = {}) => Object.fromEntries(
  Object.entries(params || {}).filter(([, value]) => value !== undefined && value !== null && value !== "")
);

const shouldRetryWithoutSector = (error) => [403, 404].includes(Number(error?.response?.status));
const removeSectorFilter = (params = {}) => {
  const next = { ...(params || {}) };
  delete next.sectorId;
  delete next.sector;
  delete next.assignedSector;
  return next;
};

export const sensorsAPI = {
  getLatest: (sectorId) => API.get("/sensors/latest", { params: cleanParams(sectorId ? { sectorId } : {}) }),
  getLatestWithFallback: async (sectorId) => {
    try { return await API.get("/sensors/latest", { params: cleanParams(sectorId ? { sectorId } : {}) }); }
    catch (error) { if (sectorId && shouldRetryWithoutSector(error)) return API.get("/sensors/latest"); throw error; }
  },
  getLatestHardware: (sectorId = "") => API.get("/sensors/latest", { params: cleanParams(sectorId ? { sectorId } : {}) }),
  getHistory: (params) => API.get("/sensors/history", { params: cleanParams(params) }),
  getHistoryWithFallback: async (params = {}) => {
    try { return await API.get("/sensors/history", { params: cleanParams(params) }); }
    catch (error) { if ((params?.sectorId || params?.sector) && shouldRetryWithoutSector(error)) return API.get("/sensors/history", { params: cleanParams(removeSectorFilter(params)) }); throw error; }
  },
  getHardwareHistory: (params = {}) => API.get("/sensors/history", { params: cleanParams({ limit: 10, ...params }) }),
  getHistoryBySector: (sectorId, params = {}) => API.get("/sensors/history", { params: cleanParams({ ...params, sectorId }) }),
  getAnalytics: (sectorId) => API.get("/sensors/analytics", { params: cleanParams(sectorId ? { sectorId } : {}) }),
  analyze: (sectorId, data = {}) => API.post(`/sensors/analyze/${sectorId}`, data),
  upload: (formData) =>
    API.post("/sensors/upload", formData, { headers: { "Content-Type": "multipart/form-data" } }),
};

export const imagesAPI = {
  upload: (formData) =>
    API.post("/images/upload", normalizeImageFormData(formData), {
      headers: { "Content-Type": "multipart/form-data" },
    }),
  getHistory: (params) => API.get("/images/history", { params: cleanParams(params) }),
  getHistoryWithFallback: async (params = {}) => {
    try { return await API.get("/images/history", { params: cleanParams(params) }); }
    catch (error) { if ((params?.sectorId || params?.sector) && shouldRetryWithoutSector(error)) return API.get("/images/history", { params: cleanParams(removeSectorFilter(params)) }); throw error; }
  },
  getHardwareHistory: (params = {}) => API.get("/images/history", { params: cleanParams({ limit: 10, ...params }) }),
  // Sector Details uses this to show the latest image from the camera assigned to the opened sector.
  // It still calls the existing backend endpoint, so no new backend route is required.
  getLatestForSector: (sectorId) => imagesAPI.getHistoryWithFallback({ sectorId, limit: 1 }),
  delete: (id) => API.delete(`/images/${id}`),
};

const isRouteMissing = (error) => [404, 405, 501].includes(Number(error?.response?.status));

const postWithRouteFallback = async (primaryUrl, fallbackUrl, payload, config) => {
  try {
    return await API.post(primaryUrl, payload, config);
  } catch (error) {
    if (!isRouteMissing(error)) throw error;
    return API.post(fallbackUrl, payload, config);
  }
};

export const diagnosisAPI = {
  // Backend diagnosis routes stay available for saved/backend-driven flows.
  analyzeManualSensors: (data = {}) =>
    postWithRouteFallback("/diagnosis/analyze-sensors-manual", "/predict_sensors", data),
  analyzeCombined: (formData) =>
    postWithRouteFallback("/diagnosis/analyze-combined", "/predict_with_image", normalizeImageFormData(formData), {
      headers: { "Content-Type": "multipart/form-data" },
    }),

  // Direct AI model links supplied by the model/backend team.
  // These do not require sectorId or deviceSerial and must be used for manual/simulated diagnosis.
  predictSensorsModel: (data = {}) => AI_MODEL_API.post("/predict_sensors", data),
  predictImageModel: (formData) =>
    AI_MODEL_API.post("/predict_image", normalizeModelImageFormData(formData), {
      headers: { "Content-Type": "multipart/form-data" },
    }),
  predictWithImageModel: (formData) =>
    AI_COMBINED_MODEL_API.post("/predict_with_image", normalizeModelImageFormData(formData), {
      headers: { "Content-Type": "multipart/form-data" },
    }),
  predictWithImageModelHF: (formData) =>
    AI_MODEL_API.post("/predict_with_image", normalizeModelImageFormData(formData), {
      headers: { "Content-Type": "multipart/form-data" },
    }),
};

const normalizeWorkerPayload = (data = {}) => {
  const names = splitName(
    data.fullName || data.name || data.workerName || `${data.firstName || ""} ${data.lastName || ""}`,
  );
  const sectorId = data.assignedSector || data.assignedSectorId || data.sectorId;
  const password = data.password || data.generatedPassword || data.tempPassword || data.temporaryPassword;
  const firstName = String(data.firstName || names.firstName || "").trim();
  const lastName = String(data.lastName || names.lastName || "Worker").trim();
  const email = String(data.email || data.username || "").trim().toLowerCase();

  // The backend endpoint is specifically /users/add-worker, so role must be worker.
  // jobTitle can still describe the job, but authentication/authorization must be worker.
  const payload = {
    firstName,
    lastName,
    email,
    ...(password ? { password } : {}),
    ...(password ? { confirmPassword: data.confirmPassword || password, passwordConfirm: data.passwordConfirm || password } : {}),
    phoneNumber: data.phoneNumber || data.phone || "",
    address: data.address || data.location || "",
    role: "worker",
    accountType: "worker",
    jobTitle: data.jobTitle || data.roleLabel || "Plant Care Worker",
    isVerified: true,
    verified: true,
  };
  if (sectorId) {
    payload.assignedSector = sectorId;
    payload.sectorId = sectorId;
    payload.assignedSectorId = sectorId;
  }
  return payload;
};

export const usersAPI = {
  getWorkers: () => API.get("/users/workers"),
  addWorker: (data) => API.post("/users/add-worker", normalizeWorkerPayload(data)),
  assignWorker: (id, data) => API.patch(`/users/worker/${id}`, data),
  updateWorker: (id, data) => API.patch(`/users/worker/${id}`, normalizeWorkerPayload(data)),
  deleteWorker: (id) => API.delete(`/users/worker/${id}`),
};


const extractDiagnosisList = (payload = {}) => {
  if (Array.isArray(payload)) return payload;
  const keys = ['diagnoses','history','items','rows','records','reports','logs','sensorData','sensor_data','images','imageLogs','image_logs','results','data'];
  for (const key of keys) if (Array.isArray(payload?.[key])) return payload[key];
  if (Array.isArray(payload?.data)) return payload.data;
  for (const key of keys) if (Array.isArray(payload?.data?.[key])) return payload.data[key];
  if (Array.isArray(payload?.result)) return payload.result;
  return [];
};

const diagnosisCountFromPayload = (payload = {}) => {
  const direct = payload.count ?? payload.total ?? payload.totalCount ?? payload.totalRecords ?? payload.data?.count ?? payload.data?.total ?? payload.data?.totalCount ?? payload.data?.totalRecords;
  if (direct !== undefined && direct !== null) return Number(direct) || 0;
  return extractDiagnosisList(payload).length;
};

const withDiagnosisFallback = async (primary, fallbacks = []) => {
  try { return await primary(); } catch (firstError) {
    let lastError = firstError;
    for (const call of fallbacks) {
      try { return await call(); } catch (err) { lastError = err; }
    }
    throw lastError;
  }
};

export const diagnosisHistoryAPI = {
  // Diagnosis history is read from /sensors/history and /images/history.
  getMyDiagnoses: (params = {}) => API.get('/sensors/history', { params }),
  getDiagnosisById: (id) => API.get('/sensors/history', { params: { id } }),
  downloadDiagnosisReport: (id) => API.get('/reports/export', { params: { diagnosisId: id }, responseType: 'blob' }),
  getDiagnosisCount: async (params = {}) => {
    const [sensorRes, imageRes] = await Promise.allSettled([
      API.get('/sensors/history', { params }),
      API.get('/images/history', { params }),
    ]);
    const sensorCount = sensorRes.status === 'fulfilled' ? diagnosisCountFromPayload(sensorRes.value.data || {}) : 0;
    const imageCount = imageRes.status === 'fulfilled' ? diagnosisCountFromPayload(imageRes.value.data || {}) : 0;
    return { data: { count: sensorCount + imageCount } };
  },
};

export const reportsAPI = {
  getStats: (sectorId, days = 7) => API.get("/reports/stats", { params: { sectorId, days } }),
  exportCSV: (sectorId) =>
    API.get("/reports/export", { params: { sectorId }, responseType: "blob" }),
  delete: (id) => API.delete(`/reports/${id}`),
};

export const tasksAPI = {
  getAll: (params) => API.get("/tasks", { params }),
  create: (data) => API.post("/tasks", data),
  update: (id, data) => API.patch(`/tasks/${id}`, data),
  markDone: (id, data = {}) => API.patch(`/tasks/${id}`, { ...data, status: data.status || "completed" }),
  delete: (id) => API.delete(`/tasks/${id}`),
};

// Direct /predict_* endpoints are intentionally not used by the production frontend.
// Diagnosis must go through /sensors/analyze/:sectorId and /images/upload so backend history is saved.


export default API;
