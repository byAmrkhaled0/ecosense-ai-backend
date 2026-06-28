import AsyncStorage from "@react-native-async-storage/async-storage";

export const MAIN_API_BASE_URL = "https://ecosense-backend.vercel.app/api";
export const AI_API_BASE_URL = "https://amr2004-ecosense-ai.hf.space/api";
export const API_BASE_URL = MAIN_API_BASE_URL;

const TOKEN_KEY = "ecosense_token";
const USER_KEY = "ecosense_user";
const DIAGNOSES_KEY = "ecosense_diagnoses";
const REGISTRATION_TOKEN_KEY = "ecosense_registration_token";

type RequestOptions = {
  method?: string;
  body?: any;
  headers?: Record<string, string>;
  auth?: boolean;
};

function normalizeEndpoint(endpoint: string) {
  return endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
}

function getErrorMessage(status: number, data: any) {
  if (status === 429) {
    return "Too many requests. Please wait a little and try again.";
  }

  if (typeof data === "string" && data.trim()) {
    return data;
  }

  if (Array.isArray(data?.errors) && data.errors.length > 0) {
    const firstError = data.errors[0];

    if (typeof firstError === "string") {
      return firstError;
    }

    return (
      firstError?.msg ||
      firstError?.message ||
      firstError?.error ||
      "Request failed. Please check your data."
    );
  }

  return (
    data?.message ||
    data?.error ||
    data?.details ||
    data?.data?.message ||
    data?.data?.error ||
    "Request failed. Please try again."
  );
}

async function request(
  baseUrl: string,
  endpoint: string,
  options: RequestOptions = {},
) {
  const url = `${baseUrl}${normalizeEndpoint(endpoint)}`;

  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(options.headers || {}),
  };

  if (options.auth) {
    const token = await AsyncStorage.getItem(TOKEN_KEY);

    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
  }

  const isFormData =
    typeof FormData !== "undefined" && options.body instanceof FormData;

  let body: any = undefined;

  if (options.body !== undefined) {
    if (isFormData) {
      body = options.body;
    } else {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(options.body);
    }
  }

  const response = await fetch(url, {
    method: options.method || "GET",
    headers,
    body,
  });

  const rawText = await response.text();

  let data: any = null;

  try {
    data = rawText ? JSON.parse(rawText) : null;
  } catch {
    data = rawText;
  }

  if (!response.ok) {
    throw new Error(getErrorMessage(response.status, data));
  }

  return data;
}

export async function apiRequest(
  endpoint: string,
  options: RequestOptions = {},
) {
  return request(MAIN_API_BASE_URL, endpoint, options);
}

export async function aiRequest(
  endpoint: string,
  options: RequestOptions = {},
) {
  return request(AI_API_BASE_URL, endpoint, options);
}

/* =========================
   Auth
========================= */

export async function loginUser(email: string, password: string) {
  const data = await apiRequest("/auth/login", {
    method: "POST",
    body: {
      email,
      password,
    },
  });

  const token =
    data?.token ||
    data?.authToken ||
    data?.accessToken ||
    data?.data?.token ||
    data?.data?.authToken ||
    data?.data?.accessToken ||
    data?.user?.token;

  const user =
    data?.user ||
    data?.data?.user ||
    data?.account ||
    data?.data?.account ||
    null;

  if (!token) {
    throw new Error("Login succeeded but no token was returned from backend.");
  }

  await AsyncStorage.setItem(TOKEN_KEY, token);

  if (user) {
    await AsyncStorage.setItem(USER_KEY, JSON.stringify(user));
  }

  return {
    token,
    user,
    raw: data,
  };
}

export async function logoutUser() {
  try {
    await apiRequest("/auth/logout", {
      method: "POST",
      auth: true,
    });
  } catch {
    // حتى لو logout endpoint فشل، لازم نمسح التوكن محليًا
  }

  await AsyncStorage.removeItem(TOKEN_KEY);
  await AsyncStorage.removeItem(USER_KEY);
  await AsyncStorage.removeItem(REGISTRATION_TOKEN_KEY);
}

export async function getStoredToken() {
  return AsyncStorage.getItem(TOKEN_KEY);
}

export async function getStoredUser() {
  const user = await AsyncStorage.getItem(USER_KEY);

  if (!user) return null;

  try {
    return JSON.parse(user);
  } catch {
    return null;
  }
}

export async function getCurrentUserFromBackend() {
  const data = await apiRequest("/auth/me", {
    method: "GET",
    auth: true,
  });

  const user =
    data?.user ||
    data?.data?.user ||
    data?.account ||
    data?.data?.account ||
    data?.data ||
    data;

  if (user) {
    await AsyncStorage.setItem(USER_KEY, JSON.stringify(user));
  }

  return user;
}

export async function registerUser(payload: any) {
  const firstName =
    payload?.firstName ||
    payload?.name?.split(" ")?.[0] ||
    payload?.fullName?.split(" ")?.[0] ||
    "";

  const lastName =
    payload?.lastName ||
    payload?.name?.split(" ")?.slice(1).join(" ") ||
    payload?.fullName?.split(" ")?.slice(1).join(" ") ||
    "";

  const email = payload?.email || "";
  const password = payload?.password || "";
  const address = payload?.address || "";
  const phoneNumber = payload?.phoneNumber || payload?.phone || "";

  const data = await apiRequest("/auth/register", {
    method: "POST",
    body: {
      email,
      password,
      firstName,
      lastName,
      address,
      phoneNumber,
    },
  });

  const registrationToken =
    data?.registrationToken ||
    data?.data?.registrationToken ||
    data?.result?.registrationToken ||
    data?.token ||
    data?.data?.token;

  if (!registrationToken) {
    throw new Error(
      "Account request was created but registrationToken was not returned from backend.",
    );
  }

  await AsyncStorage.setItem(REGISTRATION_TOKEN_KEY, registrationToken);

  return {
    registrationToken,
    raw: data,
  };
}

export async function verifyOtp(code: string, registrationToken?: string) {
  const finalRegistrationToken =
    registrationToken ||
    (await AsyncStorage.getItem(REGISTRATION_TOKEN_KEY)) ||
    "";

  if (!code) {
    throw new Error("OTP code is missing.");
  }

  if (!finalRegistrationToken) {
    throw new Error("registrationToken is missing. Please register again.");
  }

  const data = await apiRequest("/auth/verify-otp", {
    method: "POST",
    body: {
      code: String(code).trim(),
      registrationToken: String(finalRegistrationToken).trim(),
    },
  });

  const token =
    data?.token ||
    data?.authToken ||
    data?.accessToken ||
    data?.data?.token ||
    data?.data?.authToken ||
    data?.data?.accessToken ||
    data?.result?.token ||
    data?.result?.accessToken;

  const user =
    data?.user ||
    data?.data?.user ||
    data?.account ||
    data?.data?.account ||
    data?.result?.user ||
    null;

  if (!token) {
    throw new Error(
      "OTP verified but no auth token was returned from backend.",
    );
  }

  await AsyncStorage.setItem(TOKEN_KEY, token);
  await AsyncStorage.removeItem(REGISTRATION_TOKEN_KEY);

  if (user) {
    await AsyncStorage.setItem(USER_KEY, JSON.stringify(user));
  }

  return {
    token,
    user,
    raw: data,
  };
}

/* =========================
   Sensors Backend
========================= */

function pickLatestRecord(data: any) {
  const possible =
    data?.reading ||
    data?.latestReading ||
    data?.latestSensorReading ||
    data?.latest ||
    data?.sensorReading ||
    data?.record ||
    data?.result ||
    data?.data?.reading ||
    data?.data?.latestReading ||
    data?.data?.latestSensorReading ||
    data?.data?.latest ||
    data?.data?.sensorReading ||
    data?.data?.record ||
    data?.data?.result ||
    data?.data?.readings ||
    data?.data?.history ||
    data?.readings ||
    data?.history ||
    data?.items ||
    data?.data?.items ||
    data?.data ||
    data;

  if (Array.isArray(possible)) {
    return possible[0] || null;
  }

  return possible || null;
}

function readField(source: any, names: string[]) {
  for (const name of names) {
    if (source?.[name] !== undefined && source?.[name] !== null) {
      return source[name];
    }
  }

  return undefined;
}

function normalizeNumber(value: any) {
  if (value === undefined || value === null || value === "") return 0;

  if (typeof value === "string") {
    const cleaned = value.replace("%", "").replace("°C", "").trim();
    const parsed = Number(cleaned);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function getRiskFactorValue(readings: any, code: string) {
  const riskFactors =
    readings?.analysis?.risk_factors ||
    readings?.analysis?.riskFactors ||
    readings?.risk_factors ||
    readings?.riskFactors ||
    [];

  if (!Array.isArray(riskFactors)) return undefined;

  const factor = riskFactors.find(
    (item: any) =>
      String(item?.code || "").toLowerCase() === code.toLowerCase(),
  );

  return factor?.value;
}

function normalizeLightForDisplay(value: any) {
  if (value === undefined || value === null || value === "") {
    return "Sufficient";
  }

  return String(value);
}

function normalizeSensorReading(data: any) {
  const readings = pickLatestRecord(data);

  if (!readings) {
    throw new Error("No hardware sensor readings found from backend.");
  }

  const airData = readings?.air || {};
  const soilData = readings?.soil || {};

  const sectorObject =
    typeof readings?.sectorId === "object" && readings?.sectorId !== null
      ? readings.sectorId
      : null;

  const deviceObject =
    typeof readings?.deviceId === "object" && readings?.deviceId !== null
      ? readings.deviceId
      : null;

  const temperature = normalizeNumber(
    readField(readings, [
      "temperature",
      "temp",
      "Temp",
      "TEMP",
      "airTemperature",
      "air_temp",
    ]) ??
      readField(airData, [
        "temperature",
        "temp",
        "Temp",
        "TEMP",
        "airTemperature",
        "air_temp",
      ]),
  );

  const humidity = normalizeNumber(
    readField(readings, ["humidity", "hum", "Hum", "HUM", "airHumidity"]) ??
      readField(airData, ["humidity", "hum", "Hum", "HUM", "airHumidity"]),
  );

  const soilMoisture = 0;

  const soilTempFromBackend =
    readField(readings, [
      "soilTemp",
      "soilTemperature",
      "soil_temp",
      "SoilTemp",
      "SOIL_TEMP",
      "soil_temp_c",
      "rootTemp",
      "rootTemperature",
    ]) ??
    readField(soilData, [
      "temperature",
      "soilTemp",
      "soilTemperature",
      "soil_temp",
      "SoilTemp",
      "SOIL_TEMP",
      "soil_temp_c",
      "rootTemp",
      "rootTemperature",
    ]);

  const soilTemp =
    soilTempFromBackend !== undefined && soilTempFromBackend !== null
      ? normalizeNumber(soilTempFromBackend)
      : normalizeNumber(getRiskFactorValue(readings, "HIGH_SOIL_TEMP"));

  const light = normalizeLightForDisplay(
    readField(readings, [
      "light",
      "Light",
      "LIGHT",
      "lightStatus",
      "light_status",
      "lux",
      "lightLevel",
      "light_level",
    ]),
  );

  return {
    id: readings.id || readings._id || readings.readingId || null,

    ownerId: readings.ownerId || null,

    deviceId:
      deviceObject?._id ||
      deviceObject?.id ||
      readings.deviceId ||
      readings.device_id ||
      null,

    deviceSerial:
      deviceObject?.deviceSerial ||
      deviceObject?.serial ||
      readings.deviceSerial ||
      readings.device_serial ||
      readings.serial ||
      null,

    deviceStatus:
      deviceObject?.status || readings.deviceStatus || readings.status || null,

    sectorId:
      sectorObject?._id ||
      sectorObject?.id ||
      readings.sectorId ||
      readings.sector_id ||
      null,

    sectorName:
      sectorObject?.name ||
      readings.sectorName ||
      readings.sector_name ||
      "Default Sector",

    sectorLocation:
      sectorObject?.location ||
      readings.sectorLocation ||
      readings.location ||
      null,

    cropType:
      sectorObject?.cropType ||
      readings.cropType ||
      readings.crop ||
      readings.plantType ||
      readings.plant_type ||
      "Tomato",

    temperature,
    humidity,
    soilMoisture,
    soilTemp,
    light,

    status:
      readings?.analysis?.final_status ||
      readings?.analysis?.status ||
      readings.status ||
      null,

    finalStatus:
      readings?.analysis?.final_status ||
      readings?.analysis?.status ||
      readings.status ||
      null,

    confidence:
      readings?.analysis?.final_confidence ||
      readings?.analysis?.confidence ||
      null,

    recommendation:
      readings?.analysis?.general_recommendation ||
      readings?.analysis?.recommendation ||
      null,

    recommendations: readings?.analysis?.recommendations || [],

    actions: readings?.analysis?.actions || [],

    riskFactors:
      readings?.analysis?.risk_factors || readings?.analysis?.riskFactors || [],

    notification: readings?.analysis?.notification || null,

    readingCount: Number(readings.readingCount ?? readings.count ?? 0),

    updatedAt:
      readings.updatedAt ||
      readings.lastUpdate ||
      readings.last_update ||
      readings.createdAt ||
      readings.timestamp ||
      readings.time ||
      readings.date ||
      readings?.analysis?.timestamp ||
      null,

    createdAt: readings.createdAt || null,

    aiAnalysis: readings.analysis || null,

    raw: readings,
  };
}

export async function getLatestSensorReadings(sectorId?: string) {
  const endpoint = sectorId
    ? `/sensors/latest?sectorId=${encodeURIComponent(sectorId)}`
    : "/sensors/latest";

  const data = await apiRequest(endpoint, {
    method: "GET",
    auth: true,
  });

  return normalizeSensorReading(data);
}

export async function getSensorHistory(sectorId?: string) {
  const endpoint = sectorId
    ? `/sensors/history?sectorId=${encodeURIComponent(sectorId)}`
    : "/sensors/history";

  return apiRequest(endpoint, {
    method: "GET",
    auth: true,
  });
}

export async function getSensorAnalytics(sectorId?: string) {
  const endpoint = sectorId
    ? `/sensors/analytics?sectorId=${encodeURIComponent(sectorId)}`
    : "/sensors/analytics";

  return apiRequest(endpoint, {
    method: "GET",
    auth: true,
  });
}

export async function analyzeLatestSensorReading(sectorId: string) {
  return apiRequest(`/sensors/analyze/${sectorId}`, {
    method: "POST",
    auth: true,
  });
}

export async function uploadSensorReading(payload: any) {
  return apiRequest("/sensors/upload", {
    method: "POST",
    body: payload,
  });
}

/* =========================
   Devices Backend
========================= */

export async function getDevices() {
  return apiRequest("/devices", {
    method: "GET",
    auth: true,
  });
}

export async function registerDevice(payload: any) {
  return apiRequest("/devices", {
    method: "POST",
    body: payload,
    auth: true,
  });
}

export async function deleteDevice(deviceId: string) {
  return apiRequest(`/devices/${deviceId}`, {
    method: "DELETE",
    auth: true,
  });
}

/* =========================
   Sectors / Farm
========================= */

export async function getSectors() {
  return apiRequest("/sectors", {
    method: "GET",
    auth: true,
  });
}

export async function createSector(payload: any) {
  return apiRequest("/sectors", {
    method: "POST",
    body: payload,
    auth: true,
  });
}

export async function updateSector(sectorId: string, payload: any) {
  return apiRequest(`/sectors/${sectorId}`, {
    method: "PUT",
    body: payload,
    auth: true,
  });
}

export async function deleteSector(sectorId: string) {
  return apiRequest(`/sectors/${sectorId}`, {
    method: "DELETE",
    auth: true,
  });
}

/* =========================
   Dashboard / Notifications
========================= */

export async function getMainDashboard() {
  return apiRequest("/main/dashboard", {
    method: "GET",
    auth: true,
  });
}

export async function getNotifications() {
  return apiRequest("/main/notifications", {
    method: "GET",
    auth: true,
  });
}

export async function markNotificationRead(notificationId: string) {
  return apiRequest(`/main/notifications/${notificationId}`, {
    method: "PATCH",
    auth: true,
  });
}

export async function deleteNotification(notificationId: string) {
  return apiRequest(`/main/notifications/${notificationId}`, {
    method: "DELETE",
    auth: true,
  });
}

/* =========================
   AI Model Endpoints
========================= */

function buildImageFile(imageUri: string) {
  const filename = imageUri.split("/").pop() || `plant-${Date.now()}.jpg`;
  const extension = filename.split(".").pop()?.toLowerCase();

  const type =
    extension === "png"
      ? "image/png"
      : extension === "webp"
        ? "image/webp"
        : "image/jpeg";

  return {
    uri: imageUri,
    name: filename,
    type,
  } as any;
}

export async function uploadPlantImageToBackend(
  imageUri: string,
  sectorId?: string | null,
  deviceSerial?: string | null,
) {
  if (!imageUri) {
    throw new Error("Image is missing.");
  }

  if (!sectorId) {
    throw new Error("Please select a sector before uploading the image.");
  }

  const formData = new FormData();

  // الباك إند مستني الصورة باسم image
  formData.append("image", buildImageFile(imageUri));
  formData.append("sectorId", String(sectorId));

  if (deviceSerial) {
    formData.append("deviceSerial", String(deviceSerial));
  }

  return apiRequest("/images/upload", {
    method: "POST",
    body: formData,
    auth: true,
  });
}

export async function predictSensors(payload: any) {
  return aiRequest("/predict_sensors", {
    method: "POST",
    body: payload,
  });
}

export async function predictImage(imageUri: string, cropType: string) {
  const formData = new FormData();

  // AI Model مستني الصورة باسم file
  formData.append("file", buildImageFile(imageUri));
  formData.append("cropType", cropType || "Tomato");

  return aiRequest("/predict_image", {
    method: "POST",
    body: formData,
  });
}

export async function predictWithImage(imageUri: string, payload: any) {
  const formData = new FormData();

  // AI Model مستني الصورة باسم file
  formData.append("file", buildImageFile(imageUri));
  formData.append("cropType", payload?.cropType || "Tomato");
  formData.append("temperature", String(payload?.temperature ?? 0));
  formData.append("humidity", String(payload?.humidity ?? 0));
  formData.append("soilMoisture", String(payload?.soilMoisture ?? 0));
  formData.append("soilTemp", String(payload?.soilTemp ?? 0));
  formData.append("light", String(payload?.light ?? "Sufficient"));

  return aiRequest("/predict_with_image", {
    method: "POST",
    body: formData,
  });
}

/* =========================
   Local Diagnosis History
========================= */

export async function getDiagnosisHistory() {
  const saved = await AsyncStorage.getItem(DIAGNOSES_KEY);

  if (!saved) return [];

  try {
    const parsed = JSON.parse(saved);

    if (Array.isArray(parsed)) {
      return parsed;
    }

    return [];
  } catch {
    return [];
  }
}

export async function saveDiagnosisResult(record: any) {
  const history = await getDiagnosisHistory();

  const newRecord = {
    id: record?.id || `${Date.now()}`,
    createdAt: record?.createdAt || new Date().toISOString(),
    ...record,
  };

  const nextHistory = [newRecord, ...history];

  await AsyncStorage.setItem(DIAGNOSES_KEY, JSON.stringify(nextHistory));

  return newRecord;
}

export async function clearDiagnosisHistory() {
  await AsyncStorage.removeItem(DIAGNOSES_KEY);
}

/* =========================
   Optional Backend Diagnosis APIs
========================= */

export async function saveDiagnosisToBackend(payload: any) {
  return apiRequest("/diagnoses", {
    method: "POST",
    body: payload,
    auth: true,
  });
}

export async function getMyDiagnosesFromBackend() {
  return apiRequest("/diagnoses/my", {
    method: "GET",
    auth: true,
  });
}

/* =========================
   Users / Workers / Tasks
========================= */

export async function getWorkers() {
  return apiRequest("/users/workers", {
    method: "GET",
    auth: true,
  });
}

export async function addWorker(payload: any) {
  return apiRequest("/users/add-worker", {
    method: "POST",
    body: payload,
    auth: true,
  });
}

export async function deleteWorker(workerId: string) {
  return apiRequest(`/users/worker/${workerId}`, {
    method: "DELETE",
    auth: true,
  });
}

export async function getTasks() {
  return apiRequest("/tasks", {
    method: "GET",
    auth: true,
  });
}

export async function createTask(payload: any) {
  return apiRequest("/tasks", {
    method: "POST",
    body: payload,
    auth: true,
  });
}

export async function updateTask(taskId: string, payload: any) {
  return apiRequest(`/tasks/${taskId}`, {
    method: "PUT",
    body: payload,
    auth: true,
  });
}

export async function deleteTask(taskId: string) {
  return apiRequest(`/tasks/${taskId}`, {
    method: "DELETE",
    auth: true,
  });
}
