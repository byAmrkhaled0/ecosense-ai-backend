import axios from "axios";

export const API_BASE_URL =
  import.meta.env.VITE_BACKEND_API_URL ||
  import.meta.env.VITE_AI_API_URL ||
  "https://amr2004-ecosense-ai.hf.space/api";

const API = axios.create({
  baseURL: API_BASE_URL,
  timeout: 12000,
});

API.interceptors.request.use((config) => {
  const token =
    localStorage.getItem("ecosense_token") ||
    localStorage.getItem("sph_token");
  if (token && token !== "local-demo-token" && token !== "local-worker-token") {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

API.interceptors.response.use(
  (response) => response,
  (error) => {
    const isAuthRoute =
      error.config?.url?.includes("/auth/login") ||
      error.config?.url?.includes("/auth/register") ||
      error.config?.url?.includes("/auth/verify");

    if (isAuthRoute) return Promise.reject(error);

    if (error.response?.status === 401) {
      localStorage.removeItem("ecosense_token");
      localStorage.removeItem("ecosense_user");
      localStorage.removeItem("sph_auth");
      if (window.location.pathname !== "/login") window.location.href = "/login";
    }

    return Promise.reject(error);
  },
);

const toFetchResponse = (axiosPromise) =>
  axiosPromise.then((res) => ({
    ok: res.status >= 200 && res.status < 300,
    status: res.status,
    json: async () => res.data,
  }));

export const aiDiagnosisAPI = {
  predictSensors: (data) => toFetchResponse(API.post("/predict_sensors", data)),
  predictImage: (formData) => toFetchResponse(API.post("/predict_image", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  })),
  predictWithImage: (formData) => toFetchResponse(API.post("/predict_with_image", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  })),
};

export const authAPI = {
  login: (data) => API.post("/auth/login", data),
  register: (data) => API.post("/auth/register", data),
  verifyOTP: (data) => API.post("/auth/verify-otp", data),
  getMe: () => API.get("/auth/me"),
  logout: () => API.post("/auth/logout"),
};

export const dashboardAPI = {
  get: () => API.get("/main/dashboard"),
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

export const sensorsAPI = {
  getLatest: (sectorId) => API.get("/sensors/latest", { params: { sectorId } }),
  getHistory: (params) => API.get("/sensors/history", { params }),
  getAnalytics: (sectorId) => API.get("/sensors/analytics", { params: { sectorId } }),
  analyze: (sectorId) => API.post(`/sensors/analyze/${sectorId}`),
};

export const imagesAPI = {
  upload: (formData) => API.post("/images/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  }),
  getHistory: (params) => API.get("/images/history", { params }),
  delete: (id) => API.delete(`/images/${id}`),
};

export const notificationsAPI = {
  getAll: () => API.get("/main/notifications"),
  markRead: (id) => API.patch(`/main/notifications/${id}`),
  delete: (id) => API.delete(`/main/notifications/${id}`),
};

export const usersAPI = {
  getWorkers: () => API.get("/users/workers"),
  addWorker: (data) => API.post("/users/add-worker", data),
  deleteWorker: (id) => API.delete(`/users/worker/${id}`),
};

export const reportsAPI = {
  getStats: (sectorId, days) => API.get("/reports/stats", { params: { sectorId, days } }),
  exportCSV: (sectorId) => API.get("/reports/export", {
    params: { sectorId },
    responseType: "blob",
  }),
};

export default API;
