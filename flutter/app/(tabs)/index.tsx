import { router, useFocusEffect } from "expo-router";
import { useCallback, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { getAccountSyncData } from "../../services/accountSync";

export default function DashboardScreen() {
  const [account, setAccount] = useState<any | null>(null);
  const [dashboard, setDashboard] = useState<any | null>(null);
  const [devices, setDevices] = useState<any[]>([]);
  const [latestReading, setLatestReading] = useState<any | null>(null);
  const [notifications, setNotifications] = useState<any[]>([]);
  const [sectors, setSectors] = useState<any[]>([]);
  const [diagnoses, setDiagnoses] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  async function loadDashboard() {
    try {
      setLoading(true);

      const data = await getAccountSyncData();

      setAccount(data.account);
      setDashboard(data.dashboard);
      setDevices(data.devices || []);
      setLatestReading(data.latestReading);
      setNotifications(data.notifications || []);
      setSectors(data.sectors || []);
      setDiagnoses(data.diagnoses || []);
    } finally {
      setLoading(false);
    }
  }

  useFocusEffect(
    useCallback(() => {
      loadDashboard();
    }, []),
  );

  const userName =
    account?.name ||
    account?.fullName ||
    account?.username ||
    account?.email ||
    dashboard?.user?.name ||
    "Owner";

  const farmName =
    account?.farmName ||
    dashboard?.farmName ||
    dashboard?.farm?.name ||
    dashboard?.farm?.farmName ||
    latestReading?.sectorName ||
    "Smart Farm";

  const totalDiagnoses =
    Number(dashboard?.totalDiagnoses) ||
    Number(dashboard?.diagnosesCount) ||
    diagnoses.length;

  const totalDevices =
    Number(dashboard?.totalDevices) ||
    Number(dashboard?.devicesCount) ||
    devices.length;

  const totalSectors =
    Number(dashboard?.totalSectors) ||
    Number(dashboard?.sectorsCount) ||
    sectors.length;

  const alertsCount =
    Number(dashboard?.alertsCount) ||
    Number(dashboard?.notificationsCount) ||
    notifications.length;

  const healthyCount = diagnoses.filter(
    (item) => getFinalStatus(item) === "Healthy",
  ).length;

  const healthScore =
    Number(dashboard?.healthScore) ||
    Number(dashboard?.farmHealthScore) ||
    (totalDiagnoses === 0
      ? 0
      : Math.round((healthyCount / totalDiagnoses) * 100));

  const lastDiagnosis = diagnoses[0];
  const lastStatus =
    dashboard?.lastStatus ||
    dashboard?.finalStatus ||
    dashboard?.latestDiagnosis?.final_status ||
    getFinalStatus(lastDiagnosis) ||
    "No diagnosis yet";

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <View style={{ flex: 1 }}>
          <Text style={styles.kicker}>ECOSENSE AI</Text>
          <Text style={styles.title}>Dashboard</Text>
          <Text style={styles.welcome}>Welcome, {userName}</Text>
        </View>

        <Pressable style={styles.refreshButton} onPress={loadDashboard}>
          {loading ? (
            <ActivityIndicator color="#118A5B" />
          ) : (
            <Text style={styles.refreshText}>Refresh</Text>
          )}
        </Pressable>
      </View>

      <Text style={styles.subtitle}>
        Synced with your web account using the same backend and token.
      </Text>

      <View style={styles.healthCard}>
        <Text style={styles.healthLabel}>{farmName}</Text>
        <Text style={styles.healthValue}>{healthScore}%</Text>
        <Text style={styles.healthText}>
          Farm health score from account data
        </Text>
      </View>

      <View style={styles.statsGrid}>
        <StatCard title="Diagnoses" value={String(totalDiagnoses)} />
        <StatCard title="Devices" value={String(totalDevices)} />
        <StatCard title="Sectors" value={String(totalSectors)} />
        <StatCard title="Alerts" value={String(alertsCount)} danger />
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Latest Account Status</Text>
        <Text style={styles.bigText}>{lastStatus}</Text>
        <Text style={styles.cardText}>
          This status is loaded from saved diagnoses or dashboard data connected
          to the same account used on the web.
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Latest Hardware Reading</Text>

        {latestReading ? (
          <>
            <View style={styles.readingsGrid}>
              <ReadingBox
                label="Temperature"
                value={`${latestReading.temperature}°C`}
              />
              <ReadingBox
                label="Humidity"
                value={`${latestReading.humidity}%`}
              />
              <ReadingBox
                label="Soil Moisture"
                value={`${latestReading.soilMoisture}%`}
              />
              <ReadingBox
                label="Soil Temp"
                value={`${latestReading.soilTemp}°C`}
              />
              <ReadingBox
                label="Light"
                value={String(latestReading.light)}
                full
              />
            </View>

            <Text style={styles.dateText}>
              Sector: {latestReading.sectorName || "Default Sector"}
            </Text>
          </>
        ) : (
          <Text style={styles.cardText}>
            No hardware readings found for this account yet. Once the ESP32
            sends readings to the backend, they will appear here.
          </Text>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Connected Devices</Text>

        {devices.length > 0 ? (
          devices.slice(0, 3).map((device, index) => (
            <View
              key={device?._id || device?.id || index}
              style={styles.listItem}
            >
              <View>
                <Text style={styles.listTitle}>
                  {device?.name ||
                    device?.deviceName ||
                    device?.deviceId ||
                    `Device ${index + 1}`}
                </Text>
                <Text style={styles.listText}>
                  {device?.sector?.name ||
                    device?.sectorName ||
                    device?.status ||
                    "Connected device"}
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
          ))
        ) : (
          <Text style={styles.cardText}>
            No devices found yet for this account.
          </Text>
        )}
      </View>

      <View style={styles.quickActions}>
        <Pressable
          style={styles.actionCard}
          onPress={() => router.push("/(tabs)/diagnosis" as any)}
        >
          <Text style={styles.actionTitle}>Run Diagnosis</Text>
          <Text style={styles.actionText}>Analyze plant health</Text>
        </Pressable>

        <Pressable
          style={styles.actionCard}
          onPress={() => router.push("/diagnoses" as any)}
        >
          <Text style={styles.actionTitle}>My Diagnoses</Text>
          <Text style={styles.actionText}>View account results</Text>
        </Pressable>
      </View>
    </ScrollView>
  );
}

function StatCard({
  title,
  value,
  danger,
}: {
  title: string;
  value: string;
  danger?: boolean;
}) {
  return (
    <View style={styles.statCard}>
      <Text style={[styles.statValue, danger && styles.dangerText]}>
        {value}
      </Text>
      <Text style={styles.statTitle}>{title}</Text>
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

function getFinalStatus(item: any) {
  const result = item?.result || item?.aiResult || item?.analysis || item;

  return (
    result?.final_status ||
    result?.finalStatus ||
    result?.status ||
    result?.data?.final_status ||
    result?.result?.final_status ||
    "Unknown"
  );
}

function isOnline(device: any) {
  return (
    device?.online === true ||
    device?.isOnline === true ||
    device?.status === "online" ||
    device?.status === "Online" ||
    device?.connectionStatus === "online"
  );
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
    justifyContent: "space-between",
    alignItems: "flex-start",
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
  welcome: {
    color: "#65786D",
    fontSize: 14,
    fontWeight: "800",
    marginTop: 6,
  },
  subtitle: {
    color: "#65786D",
    fontSize: 14,
    lineHeight: 22,
    marginTop: 10,
    marginBottom: 20,
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
  healthCard: {
    backgroundColor: "#0B2A22",
    borderRadius: 30,
    padding: 24,
    marginBottom: 18,
  },
  healthLabel: {
    color: "#8BE0B3",
    fontSize: 13,
    fontWeight: "900",
    letterSpacing: 1,
  },
  healthValue: {
    color: "#FFFFFF",
    fontSize: 58,
    fontWeight: "900",
    marginTop: 8,
  },
  healthText: {
    color: "#CFE3D7",
    fontSize: 14,
    marginTop: 4,
  },
  statsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 16,
  },
  statCard: {
    width: "48%",
    backgroundColor: "#FFFFFF",
    borderRadius: 22,
    padding: 16,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  statValue: {
    color: "#118A5B",
    fontSize: 28,
    fontWeight: "900",
  },
  dangerText: {
    color: "#B42318",
  },
  statTitle: {
    color: "#65786D",
    fontSize: 13,
    fontWeight: "800",
    marginTop: 6,
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
    marginBottom: 10,
  },
  bigText: {
    color: "#118A5B",
    fontSize: 26,
    fontWeight: "900",
    marginBottom: 8,
  },
  cardText: {
    color: "#65786D",
    fontSize: 14,
    lineHeight: 22,
  },
  dateText: {
    color: "#7A8B82",
    fontSize: 12,
    marginTop: 12,
    fontWeight: "800",
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
  listItem: {
    backgroundColor: "#F8FBF7",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "#DDE9E2",
    marginBottom: 10,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  listTitle: {
    color: "#0B2A22",
    fontSize: 15,
    fontWeight: "900",
  },
  listText: {
    color: "#65786D",
    fontSize: 12,
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
