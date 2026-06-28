import { router, useFocusEffect } from "expo-router";
import { useCallback, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { getWebSectors } from "../services/accountSync";
import {
  deleteDevice,
  getDevices,
  getLatestSensorReadings,
} from "../services/api";

export default function FarmScreen() {
  const [devices, setDevices] = useState<any[]>([]);
  const [sectors, setSectors] = useState<any[]>([]);
  const [latestReading, setLatestReading] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  async function loadFarmData() {
    try {
      setLoading(true);
      setErrorMessage("");

      const [devicesResponse, sectorsResponse, readingResponse] =
        await Promise.allSettled([
          getDevices(),
          getWebSectors(),
          getLatestSensorReadings(),
        ]);

      if (devicesResponse.status === "fulfilled") {
        setDevices(extractArray(devicesResponse.value));
      } else {
        setDevices([]);
      }

      if (sectorsResponse.status === "fulfilled") {
        setSectors(extractArray(sectorsResponse.value));
      } else {
        setSectors([]);
      }

      if (readingResponse.status === "fulfilled") {
        setLatestReading(readingResponse.value);
      } else {
        setLatestReading(null);
      }

      if (
        devicesResponse.status === "rejected" &&
        sectorsResponse.status === "rejected" &&
        readingResponse.status === "rejected"
      ) {
        setErrorMessage(
          "No farm data was returned from backend for this account yet.",
        );
      }
    } finally {
      setLoading(false);
    }
  }

  function confirmDeleteDevice(device: any) {
    const deviceId = safeText(device?._id || device?.id);

    if (!deviceId) {
      Alert.alert("Cannot Delete", "This device does not include an id.");
      return;
    }

    Alert.alert(
      "Delete Device",
      "Are you sure you want to delete this device from the backend?",
      [
        {
          text: "Cancel",
          style: "cancel",
        },
        {
          text: "Delete",
          style: "destructive",
          onPress: () => handleDeleteDevice(deviceId),
        },
      ],
    );
  }

  async function handleDeleteDevice(deviceId: string) {
    try {
      setDeletingId(deviceId);

      await deleteDevice(deviceId);

      setDevices((previous) =>
        previous.filter((item) => safeText(item?._id || item?.id) !== deviceId),
      );

      Alert.alert("Deleted", "Device deleted successfully.");
    } catch (error: any) {
      Alert.alert(
        "Delete Failed",
        error?.message || "Could not delete this device.",
      );
    } finally {
      setDeletingId(null);
    }
  }

  useFocusEffect(
    useCallback(() => {
      loadFarmData();
    }, []),
  );

  const onlineDevices = devices.filter((device) => isOnline(device)).length;
  const offlineDevices = devices.length - onlineDevices;

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <View style={{ flex: 1 }}>
          <Text style={styles.kicker}>ECOSENSE AI</Text>
          <Text style={styles.title}>Farm Management</Text>
          <Text style={styles.subtitle}>
            Devices, sectors and hardware readings synced from the same web
            account.
          </Text>
        </View>

        <Pressable
          style={styles.refreshButton}
          onPress={loadFarmData}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#118A5B" />
          ) : (
            <Text style={styles.refreshText}>Refresh</Text>
          )}
        </Pressable>
      </View>

      {errorMessage ? (
        <View style={styles.errorBox}>
          <Text style={styles.errorTitle}>Backend Notice</Text>
          <Text style={styles.errorText}>{errorMessage}</Text>
        </View>
      ) : null}

      <View style={styles.heroCard}>
        <Text style={styles.heroLabel}>Farm Overview</Text>
        <Text style={styles.heroTitle}>Smart Farm Control</Text>

        <View style={styles.heroGrid}>
          <HeroStat label="Devices" value={String(devices.length)} />
          <HeroStat label="Sectors" value={String(sectors.length)} />
          <HeroStat label="Online" value={String(onlineDevices)} />
          <HeroStat label="Offline" value={String(offlineDevices)} danger />
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Latest Hardware Reading</Text>

        {latestReading ? (
          <>
            <View style={styles.metaRow}>
              <View style={styles.metaBox}>
                <Text style={styles.metaLabel}>Sector</Text>
                <Text style={styles.metaValue}>
                  {safeText(latestReading?.sectorName) || "Default Sector"}
                </Text>
              </View>

              <View style={styles.metaBox}>
                <Text style={styles.metaLabel}>Crop</Text>
                <Text style={styles.metaValue}>
                  {safeText(latestReading?.cropType) || "Tomato"}
                </Text>
              </View>
            </View>

            <View style={styles.readingsGrid}>
              <ReadingBox
                label="Temperature"
                value={`${safeNumber(latestReading?.temperature)}°C`}
              />
              <ReadingBox
                label="Humidity"
                value={`${safeNumber(latestReading?.humidity)}%`}
              />
              <ReadingBox
                label="Soil Moisture"
                value={`${safeNumber(latestReading?.soilMoisture)}%`}
              />
              <ReadingBox
                label="Soil Temp"
                value={`${safeNumber(latestReading?.soilTemp)}°C`}
              />
              <ReadingBox
                label="Light"
                value={safeText(latestReading?.light) || "Sufficient"}
                full
              />
            </View>
          </>
        ) : (
          <Text style={styles.cardText}>
            No sensor reading found yet. When the ESP32 uploads data to
            /api/sensors/upload, the latest reading will appear here.
          </Text>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Devices</Text>
        <Text style={styles.cardSmallText}>
          Loaded from GET /api/devices for the logged in account.
        </Text>

        {devices.length > 0 ? (
          devices.map((device, index) => {
            const deviceId = safeText(device?._id || device?.id);
            const deleting = deletingId === deviceId;

            return (
              <View key={deviceId || String(index)} style={styles.deviceCard}>
                <View style={styles.deviceTop}>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.deviceName}>
                      {getDeviceName(device, index)}
                    </Text>

                    <Text style={styles.deviceType}>
                      {getDeviceType(device)}
                    </Text>
                  </View>

                  <Text
                    style={[
                      styles.statusPill,
                      isOnline(device) ? styles.onlinePill : styles.offlinePill,
                    ]}
                  >
                    {isOnline(device) ? "Online" : "Offline"}
                  </Text>
                </View>

                <View style={styles.deviceInfoGrid}>
                  <InfoBox
                    label="Device ID"
                    value={safeText(device?.deviceId || deviceId) || "No ID"}
                  />

                  <InfoBox
                    label="Sector"
                    value={getDeviceSectorLabel(device)}
                  />

                  <InfoBox
                    label="Last Contact"
                    value={formatDate(
                      device?.lastContact ||
                        device?.lastSeen ||
                        device?.updatedAt ||
                        device?.createdAt,
                    )}
                  />

                  <InfoBox
                    label="Status"
                    value={
                      safeText(device?.status || device?.connectionStatus) ||
                      "Unknown"
                    }
                  />
                </View>

                <View style={styles.deviceActions}>
                  <Pressable
                    style={styles.smallButton}
                    onPress={() =>
                      Alert.alert(
                        "Device Details",
                        JSON.stringify(device, null, 2).slice(0, 900),
                      )
                    }
                  >
                    <Text style={styles.smallButtonText}>Details</Text>
                  </Pressable>

                  <Pressable
                    style={[
                      styles.deleteButton,
                      deleting && styles.disabledButton,
                    ]}
                    onPress={() => confirmDeleteDevice(device)}
                    disabled={deleting}
                  >
                    {deleting ? (
                      <ActivityIndicator color="#FFFFFF" />
                    ) : (
                      <Text style={styles.deleteButtonText}>Delete</Text>
                    )}
                  </Pressable>
                </View>
              </View>
            );
          })
        ) : (
          <Text style={styles.cardText}>
            No devices found yet. Devices added on the web using the same
            account should appear here.
          </Text>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Sectors</Text>
        <Text style={styles.cardSmallText}>
          Sectors are loaded from backend using possible web routes.
        </Text>

        {sectors.length > 0 ? (
          sectors.map((sector, index) => (
            <View
              key={safeText(sector?._id || sector?.id) || String(index)}
              style={styles.sectorCard}
            >
              <View style={{ flex: 1 }}>
                <Text style={styles.sectorName}>
                  {getSectorName(sector, index)}
                </Text>

                <Text style={styles.sectorText}>{getSectorCrop(sector)}</Text>
              </View>

              <View style={styles.sectorBadge}>
                <Text style={styles.sectorBadgeText}>
                  {safeText(sector?.status || sector?.healthStatus) || "Active"}
                </Text>
              </View>
            </View>
          ))
        ) : (
          <Text style={styles.cardText}>
            No sectors found yet. If sectors exist on the web, ask backend for
            the exact sector route.
          </Text>
        )}
      </View>

      <View style={styles.quickActions}>
        <Pressable
          style={styles.actionCard}
          onPress={() => router.push("/(tabs)/sensors" as any)}
        >
          <Text style={styles.actionTitle}>Open Sensors</Text>
          <Text style={styles.actionText}>View readings and analytics</Text>
        </Pressable>

        <Pressable
          style={styles.actionCard}
          onPress={() => router.push("/(tabs)/diagnosis" as any)}
        >
          <Text style={styles.actionTitle}>Diagnose Now</Text>
          <Text style={styles.actionText}>Run AI plant diagnosis</Text>
        </Pressable>
      </View>
    </ScrollView>
  );
}

function HeroStat({
  label,
  value,
  danger,
}: {
  label: string;
  value: string;
  danger?: boolean;
}) {
  return (
    <View style={styles.heroStat}>
      <Text style={[styles.heroStatValue, danger && styles.dangerText]}>
        {value}
      </Text>
      <Text style={styles.heroStatLabel}>{label}</Text>
    </View>
  );
}

function ReadingBox({
  label,
  value,
  full,
}: {
  label: string;
  value: string;
  full?: boolean;
}) {
  return (
    <View style={full ? styles.readingBoxFull : styles.readingBox}>
      <Text style={styles.readingLabel}>{label}</Text>
      <Text style={styles.readingValue}>{value}</Text>
    </View>
  );
}

function InfoBox({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.infoBox}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue} numberOfLines={1}>
        {safeText(value) || "N/A"}
      </Text>
    </View>
  );
}

function extractArray(data: any): any[] {
  if (Array.isArray(data)) return data;

  if (Array.isArray(data?.data)) return data.data;
  if (Array.isArray(data?.items)) return data.items;
  if (Array.isArray(data?.results)) return data.results;
  if (Array.isArray(data?.devices)) return data.devices;
  if (Array.isArray(data?.sectors)) return data.sectors;

  if (Array.isArray(data?.data?.items)) return data.data.items;
  if (Array.isArray(data?.data?.results)) return data.data.results;
  if (Array.isArray(data?.data?.devices)) return data.data.devices;
  if (Array.isArray(data?.data?.sectors)) return data.data.sectors;

  return [];
}

function safeText(value: any): string {
  if (value === null || value === undefined) return "";

  if (typeof value === "string") return value;
  if (typeof value === "number") return String(value);
  if (typeof value === "boolean") return value ? "Yes" : "No";

  if (typeof value === "object") {
    return (
      value.name ||
      value.title ||
      value.label ||
      value.cropType ||
      value.deviceId ||
      value._id ||
      value.id ||
      ""
    );
  }

  return String(value);
}

function safeNumber(value: any) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : 0;
}

function getDeviceName(device: any, index: number) {
  return (
    safeText(device?.name) ||
    safeText(device?.deviceName) ||
    safeText(device?.deviceId) ||
    `Device ${index + 1}`
  );
}

function getDeviceType(device: any) {
  return (
    safeText(device?.type) ||
    safeText(device?.deviceType) ||
    safeText(device?.model) ||
    "IoT Device"
  );
}

function getDeviceSectorLabel(device: any) {
  return (
    safeText(device?.sector) ||
    safeText(device?.sectorId) ||
    safeText(device?.sectorName) ||
    "No sector"
  );
}

function getSectorName(sector: any, index: number) {
  return (
    safeText(sector?.name) ||
    safeText(sector?.sectorName) ||
    safeText(sector?.title) ||
    `Sector ${index + 1}`
  );
}

function getSectorCrop(sector: any) {
  return (
    safeText(sector?.cropType) ||
    safeText(sector?.crop) ||
    safeText(sector?.plantType) ||
    "No crop type"
  );
}

function isOnline(device: any) {
  return (
    device?.online === true ||
    device?.isOnline === true ||
    device?.status === "online" ||
    device?.status === "Online" ||
    device?.connectionStatus === "online" ||
    device?.connectionStatus === "Online"
  );
}

function formatDate(value: any) {
  if (!value) return "No date";

  try {
    return new Date(value).toLocaleDateString();
  } catch {
    return "No date";
  }
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#F3F8F1",
  },
  content: {
    padding: 22,
    paddingTop: 50,
    paddingBottom: 40,
  },
  header: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
  },
  kicker: {
    color: "#118A5B",
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 2,
  },
  title: {
    color: "#082A1F",
    fontSize: 36,
    fontWeight: "900",
    marginTop: 6,
  },
  subtitle: {
    color: "#65786D",
    fontSize: 14,
    lineHeight: 22,
    marginTop: 10,
  },
  refreshButton: {
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#DDE9E2",
    borderRadius: 16,
    paddingVertical: 10,
    paddingHorizontal: 14,
  },
  refreshText: {
    color: "#118A5B",
    fontWeight: "900",
  },
  errorBox: {
    backgroundColor: "#FFF7E6",
    borderColor: "#F6C56B",
    borderWidth: 1,
    borderRadius: 22,
    padding: 16,
    marginTop: 18,
    marginBottom: 16,
  },
  errorTitle: {
    color: "#B66A00",
    fontSize: 15,
    fontWeight: "900",
    marginBottom: 6,
  },
  errorText: {
    color: "#7A4B00",
    fontSize: 13,
    lineHeight: 20,
    fontWeight: "700",
  },
  heroCard: {
    backgroundColor: "#0B2A22",
    borderRadius: 30,
    padding: 22,
    marginTop: 20,
    marginBottom: 16,
  },
  heroLabel: {
    color: "#8BE0B3",
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 1.5,
  },
  heroTitle: {
    color: "#FFFFFF",
    fontSize: 30,
    fontWeight: "900",
    marginTop: 8,
  },
  heroGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 18,
  },
  heroStat: {
    width: "48%",
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  heroStatValue: {
    color: "#FFFFFF",
    fontSize: 25,
    fontWeight: "900",
  },
  heroStatLabel: {
    color: "#BFD4C8",
    fontSize: 12,
    fontWeight: "800",
    marginTop: 4,
  },
  dangerText: {
    color: "#FFB4AB",
  },
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    borderColor: "#DDE9E2",
    marginBottom: 16,
  },
  cardTitle: {
    color: "#0B2A22",
    fontSize: 20,
    fontWeight: "900",
    marginBottom: 8,
  },
  cardSmallText: {
    color: "#65786D",
    fontSize: 13,
    lineHeight: 20,
    marginBottom: 12,
  },
  cardText: {
    color: "#65786D",
    fontSize: 14,
    lineHeight: 22,
  },
  metaRow: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 14,
  },
  metaBox: {
    flex: 1,
    backgroundColor: "#F8FBF7",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  metaLabel: {
    color: "#65786D",
    fontSize: 12,
    fontWeight: "900",
  },
  metaValue: {
    color: "#0B2A22",
    fontSize: 15,
    fontWeight: "900",
    marginTop: 5,
  },
  readingsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  readingBox: {
    width: "48%",
    backgroundColor: "#F3F8F1",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  readingBoxFull: {
    width: "100%",
    backgroundColor: "#F3F8F1",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  readingLabel: {
    color: "#65786D",
    fontSize: 12,
    fontWeight: "900",
  },
  readingValue: {
    color: "#0B2A22",
    fontSize: 22,
    fontWeight: "900",
    marginTop: 6,
  },
  deviceCard: {
    backgroundColor: "#F8FBF7",
    borderRadius: 22,
    padding: 16,
    borderWidth: 1,
    borderColor: "#DDE9E2",
    marginBottom: 12,
  },
  deviceTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 12,
  },
  deviceName: {
    color: "#0B2A22",
    fontSize: 17,
    fontWeight: "900",
  },
  deviceType: {
    color: "#65786D",
    fontSize: 13,
    fontWeight: "800",
    marginTop: 4,
  },
  statusPill: {
    overflow: "hidden",
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 6,
    fontSize: 11,
    fontWeight: "900",
  },
  onlinePill: {
    backgroundColor: "#E6F7EE",
    color: "#118A5B",
  },
  offlinePill: {
    backgroundColor: "#FEF3F2",
    color: "#B42318",
  },
  deviceInfoGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 14,
  },
  infoBox: {
    width: "48%",
    backgroundColor: "#FFFFFF",
    borderRadius: 16,
    padding: 12,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  infoLabel: {
    color: "#65786D",
    fontSize: 11,
    fontWeight: "900",
  },
  infoValue: {
    color: "#0B2A22",
    fontSize: 13,
    fontWeight: "900",
    marginTop: 5,
  },
  deviceActions: {
    flexDirection: "row",
    gap: 10,
    marginTop: 14,
  },
  smallButton: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    borderRadius: 16,
    paddingVertical: 12,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  smallButtonText: {
    color: "#118A5B",
    fontSize: 13,
    fontWeight: "900",
  },
  deleteButton: {
    flex: 1,
    backgroundColor: "#B42318",
    borderRadius: 16,
    paddingVertical: 12,
    alignItems: "center",
  },
  deleteButtonText: {
    color: "#FFFFFF",
    fontSize: 13,
    fontWeight: "900",
  },
  disabledButton: {
    opacity: 0.6,
  },
  sectorCard: {
    backgroundColor: "#F8FBF7",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "#DDE9E2",
    marginBottom: 10,
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 12,
  },
  sectorName: {
    color: "#0B2A22",
    fontSize: 16,
    fontWeight: "900",
  },
  sectorText: {
    color: "#65786D",
    fontSize: 13,
    fontWeight: "800",
    marginTop: 4,
  },
  sectorBadge: {
    alignSelf: "flex-start",
    backgroundColor: "#E6F7EE",
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  sectorBadgeText: {
    color: "#118A5B",
    fontSize: 11,
    fontWeight: "900",
  },
  quickActions: {
    flexDirection: "row",
    gap: 12,
  },
  actionCard: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    borderRadius: 24,
    padding: 16,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  actionTitle: {
    color: "#0B2A22",
    fontSize: 17,
    fontWeight: "900",
  },
  actionText: {
    color: "#65786D",
    fontSize: 13,
    marginTop: 8,
  },
});
