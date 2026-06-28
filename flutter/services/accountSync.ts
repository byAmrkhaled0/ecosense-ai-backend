import {
    apiRequest,
    getDevices,
    getDiagnosisHistory,
    getLatestSensorReadings,
} from "./api";

async function safeCall<T>(
  callback: () => Promise<T>,
  fallback: T,
): Promise<T> {
  try {
    return await callback();
  } catch {
    return fallback;
  }
}

function extractArray(data: any): any[] {
  if (Array.isArray(data)) return data;

  if (Array.isArray(data?.data)) return data.data;
  if (Array.isArray(data?.items)) return data.items;
  if (Array.isArray(data?.results)) return data.results;
  if (Array.isArray(data?.notifications)) return data.notifications;
  if (Array.isArray(data?.devices)) return data.devices;
  if (Array.isArray(data?.sectors)) return data.sectors;
  if (Array.isArray(data?.diagnoses)) return data.diagnoses;
  if (Array.isArray(data?.readings)) return data.readings;

  if (Array.isArray(data?.data?.items)) return data.data.items;
  if (Array.isArray(data?.data?.results)) return data.data.results;
  if (Array.isArray(data?.data?.notifications)) return data.data.notifications;
  if (Array.isArray(data?.data?.devices)) return data.data.devices;
  if (Array.isArray(data?.data?.sectors)) return data.data.sectors;
  if (Array.isArray(data?.data?.diagnoses)) return data.data.diagnoses;
  if (Array.isArray(data?.data?.readings)) return data.data.readings;

  return [];
}

export async function getCurrentAccount() {
  return safeCall(async () => {
    const data = await apiRequest("/auth/me", {
      method: "GET",
      auth: true,
    });

    return data?.user || data?.data?.user || data?.data || data;
  }, null);
}

export async function getWebDashboardData() {
  return safeCall(async () => {
    const data = await apiRequest("/main/dashboard", {
      method: "GET",
      auth: true,
    });

    return data?.data || data?.dashboard || data;
  }, null);
}

export async function getWebNotifications() {
  return safeCall(async () => {
    const data = await apiRequest("/main/notifications", {
      method: "GET",
      auth: true,
    });

    return extractArray(data);
  }, []);
}

export async function getWebSectors() {
  const endpoints = ["/sectors", "/main/sectors", "/farm/sectors"];

  for (const endpoint of endpoints) {
    try {
      const data = await apiRequest(endpoint, {
        method: "GET",
        auth: true,
      });

      const list = extractArray(data);

      if (list.length > 0) {
        return list;
      }
    } catch {
      // try next possible endpoint
    }
  }

  return [];
}

export async function getWebDiagnoses() {
  const endpoints = [
    "/diagnoses/my",
    "/diagnoses",
    "/ai-diagnoses/my",
    "/ai-diagnoses",
    "/ai-diagnosis/my",
    "/ai-diagnosis",
  ];

  for (const endpoint of endpoints) {
    try {
      const data = await apiRequest(endpoint, {
        method: "GET",
        auth: true,
      });

      const list = extractArray(data);

      if (list.length > 0) {
        return list;
      }
    } catch {
      // try next possible endpoint
    }
  }

  return getDiagnosisHistory();
}

export async function getAccountSyncData() {
  const [
    account,
    dashboard,
    devices,
    latestReading,
    notifications,
    sectors,
    diagnoses,
  ] = await Promise.all([
    getCurrentAccount(),
    getWebDashboardData(),
    safeCall(() => getDevices(), []),
    safeCall(() => getLatestSensorReadings(), null),
    getWebNotifications(),
    getWebSectors(),
    getWebDiagnoses(),
  ]);

  return {
    account,
    dashboard,
    devices: extractArray(devices),
    latestReading,
    notifications,
    sectors,
    diagnoses,
  };
}
