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
import { getAccountSyncData } from "../services/accountSync";

export default function ReportsScreen() {
  const [dashboard, setDashboard] = useState<any | null>(null);
  const [diagnoses, setDiagnoses] = useState<any[]>([]);
  const [latestReading, setLatestReading] = useState<any | null>(null);
  const [devices, setDevices] = useState<any[]>([]);
  const [sectors, setSectors] = useState<any[]>([]);
  const [notifications, setNotifications] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  async function loadReports() {
    try {
      setLoading(true);
      setErrorMessage("");

      const data = await getAccountSyncData();

      setDashboard(data.dashboard || null);
      setDiagnoses(Array.isArray(data.diagnoses) ? data.diagnoses : []);
      setLatestReading(data.latestReading || null);
      setDevices(Array.isArray(data.devices) ? data.devices : []);
      setSectors(Array.isArray(data.sectors) ? data.sectors : []);
      setNotifications(
        Array.isArray(data.notifications) ? data.notifications : [],
      );
    } catch (error: any) {
      setErrorMessage(
        error?.message || "Could not load reports from this account.",
      );
    } finally {
      setLoading(false);
    }
  }

  useFocusEffect(
    useCallback(() => {
      loadReports();
    }, []),
  );

  const totalDiagnoses =
    Number(dashboard?.totalDiagnoses) ||
    Number(dashboard?.diagnosesCount) ||
    diagnoses.length;

  const healthyCount = diagnoses.filter(
    (item) => normalizeStatus(getFinalStatus(item)) === "healthy",
  ).length;

  const moderateCount = diagnoses.filter(
    (item) => normalizeStatus(getFinalStatus(item)) === "moderate",
  ).length;

  const highStressCount = diagnoses.filter(
    (item) => normalizeStatus(getFinalStatus(item)) === "high",
  ).length;

  const healthScore =
    Number(dashboard?.healthScore) ||
    Number(dashboard?.farmHealthScore) ||
    (totalDiagnoses === 0
      ? 0
      : Math.round((healthyCount / totalDiagnoses) * 100));

  const lastDiagnosis = diagnoses[0] || null;
  const lastStatus =
    safeText(dashboard?.lastStatus) ||
    safeText(dashboard?.finalStatus) ||
    safeText(dashboard?.latestDiagnosis?.final_status) ||
    getFinalStatus(lastDiagnosis) ||
    "No diagnosis yet";

  const mostCommonProblem = getMostCommonProblem(diagnoses);
  const reportLevel = getReportLevel(healthScore, highStressCount);
  const recommendations = buildRecommendations({
    healthScore,
    highStressCount,
    latestReading,
    totalDiagnoses,
  });

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backText}>‹ Back</Text>
        </Pressable>

        <Pressable
          style={styles.refreshButton}
          onPress={loadReports}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#118A5B" />
          ) : (
            <Text style={styles.refreshText}>Refresh</Text>
          )}
        </Pressable>
      </View>

      <Text style={styles.kicker}>ECOSENSE AI</Text>
      <Text style={styles.title}>Reports</Text>
      <Text style={styles.subtitle}>
        Account report generated from web dashboard data, diagnoses, sensors,
        devices and farm status.
      </Text>

      {errorMessage ? (
        <View style={styles.noticeBox}>
          <Text style={styles.noticeTitle}>Backend Notice</Text>
          <Text style={styles.noticeText}>{errorMessage}</Text>
        </View>
      ) : null}

      <View style={styles.heroCard}>
        <Text style={styles.heroLabel}>Farm Health Report</Text>
        <Text style={styles.heroScore}>{healthScore}%</Text>
        <Text style={styles.heroText}>{reportLevel}</Text>

        <View style={styles.heroFooter}>
          <Text style={styles.heroFooterText}>
            Last status: {safeText(lastStatus)}
          </Text>
        </View>
      </View>

      <View style={styles.summaryGrid}>
        <SummaryCard label="Diagnoses" value={String(totalDiagnoses)} />
        <SummaryCard label="Healthy" value={String(healthyCount)} />
        <SummaryCard label="Moderate" value={String(moderateCount)} />
        <SummaryCard
          label="High Stress"
          value={String(highStressCount)}
          danger
        />
        <SummaryCard label="Devices" value={String(devices.length)} />
        <SummaryCard label="Sectors" value={String(sectors.length)} />
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Diagnosis Summary</Text>

        {totalDiagnoses > 0 ? (
          <>
            <ReportRow
              label="Latest Status"
              value={safeText(lastStatus)}
              highlight
            />

            <ReportRow label="Most Common Problem" value={mostCommonProblem} />

            <ReportRow
              label="Critical Cases"
              value={`${highStressCount} high stress result(s)`}
            />

            <ReportRow label="Report Source" value="Synced account diagnoses" />
          </>
        ) : (
          <Text style={styles.cardText}>
            No diagnosis records found yet. When you run diagnosis from mobile
            or web and save it to the backend, the report will update.
          </Text>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Latest Sensor Summary</Text>

        {latestReading ? (
          <>
            <View style={styles.sensorGrid}>
              <SensorBox
                label="Temperature"
                value={`${safeNumber(latestReading.temperature)}°C`}
                warning={safeNumber(latestReading.temperature) >= 32}
              />
              <SensorBox
                label="Humidity"
                value={`${safeNumber(latestReading.humidity)}%`}
                warning={safeNumber(latestReading.humidity) <= 35}
              />
              <SensorBox
                label="Soil Moisture"
                value={`${safeNumber(latestReading.soilMoisture)}%`}
                warning={safeNumber(latestReading.soilMoisture) <= 30}
              />
              <SensorBox
                label="Soil Temp"
                value={`${safeNumber(latestReading.soilTemp)}°C`}
                warning={safeNumber(latestReading.soilTemp) >= 30}
              />
              <SensorBox
                label="Light"
                value={safeText(latestReading.light) || "N/A"}
                full
              />
            </View>

            <Text style={styles.timestamp}>
              Sector: {safeText(latestReading.sectorName) || "Default Sector"}
            </Text>

            <Text style={styles.timestamp}>
              Last update:{" "}
              {latestReading.updatedAt
                ? new Date(latestReading.updatedAt).toLocaleString()
                : "No timestamp"}
            </Text>
          </>
        ) : (
          <Text style={styles.cardText}>
            No hardware readings found yet for this account. Once ESP32 uploads
            readings, sensor reports will appear here.
          </Text>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recommendations</Text>

        {recommendations.map((item, index) => (
          <View key={index} style={styles.recommendationItem}>
            <View style={styles.recommendationDot} />
            <Text style={styles.recommendationText}>{item}</Text>
          </View>
        ))}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recent Diagnoses</Text>

        {diagnoses.length > 0 ? (
          diagnoses.slice(0, 4).map((item, index) => {
            const status = getFinalStatus(item);
            const disease = getDiseaseName(item);

            return (
              <View key={getItemId(item, index)} style={styles.diagnosisItem}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.diagnosisTitle}>{disease}</Text>
                  <Text style={styles.diagnosisText}>
                    {getDiagnosisDate(item)} • {getCropType(item)}
                  </Text>
                </View>

                <Text style={[styles.statusBadge, getStatusStyle(status)]}>
                  {safeText(status)}
                </Text>
              </View>
            );
          })
        ) : (
          <Text style={styles.cardText}>
            No recent diagnoses available for this account.
          </Text>
        )}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>System Overview</Text>

        <ReportRow label="Connected Devices" value={String(devices.length)} />
        <ReportRow label="Farm Sectors" value={String(sectors.length)} />
        <ReportRow label="Notifications" value={String(notifications.length)} />
        <ReportRow
          label="Data Source"
          value="Same backend account used on web"
        />
      </View>

      <View style={styles.quickActions}>
        <Pressable
          style={styles.actionCard}
          onPress={() => router.push("/(tabs)/diagnosis" as any)}
        >
          <Text style={styles.actionTitle}>Run Diagnosis</Text>
          <Text style={styles.actionText}>Create a new AI report</Text>
        </Pressable>

        <Pressable
          style={styles.actionCard}
          onPress={() => router.push("/diagnoses" as any)}
        >
          <Text style={styles.actionTitle}>My Diagnoses</Text>
          <Text style={styles.actionText}>View all diagnosis records</Text>
        </Pressable>
      </View>
    </ScrollView>
  );
}

function SummaryCard({
  label,
  value,
  danger,
}: {
  label: string;
  value: string;
  danger?: boolean;
}) {
  return (
    <View style={styles.summaryCard}>
      <Text style={[styles.summaryValue, danger && styles.dangerText]}>
        {value}
      </Text>
      <Text style={styles.summaryLabel}>{label}</Text>
    </View>
  );
}

function ReportRow({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <View style={styles.reportRow}>
      <Text style={styles.reportLabel}>{label}</Text>
      <Text style={[styles.reportValue, highlight && styles.highlightValue]}>
        {safeText(value) || "N/A"}
      </Text>
    </View>
  );
}

function SensorBox({
  label,
  value,
  warning,
  full,
}: {
  label: string;
  value: string;
  warning?: boolean;
  full?: boolean;
}) {
  return (
    <View style={full ? styles.sensorBoxFull : styles.sensorBox}>
      <Text style={styles.sensorLabel}>{label}</Text>
      <Text style={[styles.sensorValue, warning && styles.sensorWarning]}>
        {safeText(value)}
      </Text>
    </View>
  );
}

function getItemId(item: any, index: number) {
  return safeText(item?._id || item?.id || item?.diagnosisId) || String(index);
}

function getFinalStatus(item: any) {
  if (!item) return "Unknown";

  const result = item?.result || item?.aiResult || item?.analysis || item;

  return (
    safeText(result?.final_status) ||
    safeText(result?.finalStatus) ||
    safeText(result?.status) ||
    safeText(result?.plantStatus) ||
    safeText(result?.data?.final_status) ||
    safeText(result?.result?.final_status) ||
    "Unknown"
  );
}

function normalizeStatus(status: any) {
  const value = safeText(status).toLowerCase();

  if (value.includes("healthy")) return "healthy";
  if (value.includes("moderate")) return "moderate";
  if (value.includes("high")) return "high";
  if (value.includes("stress")) return "high";

  return "unknown";
}

function getStatusStyle(status: any) {
  const normalized = normalizeStatus(status);

  if (normalized === "healthy") {
    return {
      backgroundColor: "#E6F7EE",
      color: "#118A5B",
    };
  }

  if (normalized === "moderate") {
    return {
      backgroundColor: "#FFF7E6",
      color: "#B66A00",
    };
  }

  if (normalized === "high") {
    return {
      backgroundColor: "#FEF3F2",
      color: "#B42318",
    };
  }

  return {
    backgroundColor: "#EEF2F0",
    color: "#65786D",
  };
}

function getDiseaseName(item: any) {
  const result = item?.result || item?.aiResult || item?.analysis || item;

  return (
    safeText(result?.disease_name) ||
    safeText(result?.diseaseName) ||
    safeText(result?.disease) ||
    safeText(result?.primary_issue) ||
    safeText(result?.primaryIssue) ||
    safeText(item?.title) ||
    "Plant Diagnosis Result"
  );
}

function getCropType(item: any) {
  const result = item?.result || item?.aiResult || item?.analysis || item;

  return (
    safeText(item?.cropType) ||
    safeText(item?.crop) ||
    safeText(result?.cropType) ||
    safeText(result?.crop) ||
    "Tomato"
  );
}

function getDiagnosisDate(item: any) {
  const value =
    item?.createdAt ||
    item?.updatedAt ||
    item?.date ||
    item?.timestamp ||
    item?.result?.createdAt;

  if (!value) return "No date";

  try {
    return new Date(value).toLocaleDateString();
  } catch {
    return "No date";
  }
}

function getMostCommonProblem(items: any[]) {
  if (!items.length) return "No diagnosis data yet";

  const counts: Record<string, number> = {};

  items.forEach((item) => {
    const name = getDiseaseName(item);
    counts[name] = (counts[name] || 0) + 1;
  });

  let topName = "No common problem";
  let topCount = 0;

  Object.keys(counts).forEach((name) => {
    if (counts[name] > topCount) {
      topName = name;
      topCount = counts[name];
    }
  });

  return topName;
}

function getReportLevel(score: number, highStressCount: number) {
  if (highStressCount > 0) {
    return "Critical attention required for one or more plant cases.";
  }

  if (score >= 80) {
    return "Farm condition looks good based on current diagnosis data.";
  }

  if (score >= 50) {
    return "Farm condition is moderate and needs regular monitoring.";
  }

  return "Farm condition needs attention. Run diagnosis and check sensors.";
}

function buildRecommendations({
  healthScore,
  highStressCount,
  latestReading,
  totalDiagnoses,
}: {
  healthScore: number;
  highStressCount: number;
  latestReading: any | null;
  totalDiagnoses: number;
}) {
  const list: string[] = [];

  if (totalDiagnoses === 0) {
    list.push("Run at least one AI diagnosis to generate a real farm report.");
  }

  if (highStressCount > 0) {
    list.push(
      "Review high stress diagnosis results and apply recommended actions.",
    );
  }

  if (healthScore < 60 && totalDiagnoses > 0) {
    list.push("Increase monitoring frequency and review plant health history.");
  }

  if (!latestReading) {
    list.push(
      "Connect ESP32 sensors and upload readings to improve report accuracy.",
    );
  } else {
    const temp = safeNumber(latestReading.temperature);
    const humidity = safeNumber(latestReading.humidity);
    const moisture = safeNumber(latestReading.soilMoisture);

    if (temp >= 32) {
      list.push("Temperature is high. Consider cooling or shade actions.");
    }

    if (humidity <= 35) {
      list.push("Humidity is low. Consider humidity adjustment.");
    }

    if (moisture <= 30) {
      list.push("Soil moisture is low. Irrigation may be required.");
    }
  }

  if (list.length === 0) {
    list.push(
      "Farm condition is stable. Keep monitoring sensors and diagnosis history.",
    );
  }

  return list;
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
      value.message ||
      value.details ||
      value.description ||
      value.final_status ||
      value.finalStatus ||
      value.status ||
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
    alignItems: "center",
  },
  backButton: {
    backgroundColor: "#FFFFFF",
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  backText: {
    color: "#118A5B",
    fontSize: 14,
    fontWeight: "900",
  },
  refreshButton: {
    backgroundColor: "#FFFFFF",
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  refreshText: {
    color: "#118A5B",
    fontSize: 14,
    fontWeight: "900",
  },
  kicker: {
    color: "#118A5B",
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 2,
    marginTop: 26,
  },
  title: {
    color: "#082A1F",
    fontSize: 38,
    fontWeight: "900",
    marginTop: 8,
  },
  subtitle: {
    color: "#65786D",
    fontSize: 14,
    lineHeight: 22,
    marginTop: 10,
    marginBottom: 18,
  },
  noticeBox: {
    backgroundColor: "#FFF7E6",
    borderColor: "#F6C56B",
    borderWidth: 1,
    borderRadius: 22,
    padding: 16,
    marginBottom: 16,
  },
  noticeTitle: {
    color: "#B66A00",
    fontSize: 15,
    fontWeight: "900",
    marginBottom: 6,
  },
  noticeText: {
    color: "#7A4B00",
    fontSize: 13,
    lineHeight: 20,
    fontWeight: "700",
  },
  heroCard: {
    backgroundColor: "#0B2A22",
    borderRadius: 30,
    padding: 24,
    marginBottom: 18,
  },
  heroLabel: {
    color: "#8BE0B3",
    fontSize: 13,
    fontWeight: "900",
    letterSpacing: 1,
  },
  heroScore: {
    color: "#FFFFFF",
    fontSize: 62,
    fontWeight: "900",
    marginTop: 8,
  },
  heroText: {
    color: "#CFE3D7",
    fontSize: 14,
    lineHeight: 22,
    marginTop: 4,
  },
  heroFooter: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 18,
    padding: 14,
    marginTop: 18,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  heroFooterText: {
    color: "#FFFFFF",
    fontSize: 13,
    fontWeight: "900",
  },
  summaryGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 16,
  },
  summaryCard: {
    width: "48%",
    backgroundColor: "#FFFFFF",
    borderRadius: 22,
    padding: 16,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  summaryValue: {
    color: "#118A5B",
    fontSize: 28,
    fontWeight: "900",
  },
  dangerText: {
    color: "#B42318",
  },
  summaryLabel: {
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
  cardText: {
    color: "#65786D",
    fontSize: 14,
    lineHeight: 22,
  },
  reportRow: {
    backgroundColor: "#F8FBF7",
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: "#DDE9E2",
    marginBottom: 10,
  },
  reportLabel: {
    color: "#65786D",
    fontSize: 12,
    fontWeight: "900",
  },
  reportValue: {
    color: "#0B2A22",
    fontSize: 15,
    fontWeight: "900",
    marginTop: 5,
  },
  highlightValue: {
    color: "#118A5B",
    fontSize: 18,
  },
  sensorGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  sensorBox: {
    width: "48%",
    backgroundColor: "#F3F8F1",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  sensorBoxFull: {
    width: "100%",
    backgroundColor: "#F3F8F1",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  sensorLabel: {
    color: "#65786D",
    fontSize: 12,
    fontWeight: "900",
  },
  sensorValue: {
    color: "#0B2A22",
    fontSize: 22,
    fontWeight: "900",
    marginTop: 6,
  },
  sensorWarning: {
    color: "#B42318",
  },
  timestamp: {
    color: "#7A8B82",
    fontSize: 12,
    fontWeight: "800",
    marginTop: 12,
  },
  recommendationItem: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 10,
    backgroundColor: "#F8FBF7",
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: "#DDE9E2",
    marginBottom: 10,
  },
  recommendationDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: "#118A5B",
    marginTop: 5,
  },
  recommendationText: {
    color: "#0B2A22",
    fontSize: 13,
    lineHeight: 21,
    fontWeight: "800",
    flex: 1,
  },
  diagnosisItem: {
    backgroundColor: "#F8FBF7",
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "#DDE9E2",
    marginBottom: 10,
    flexDirection: "row",
    gap: 12,
    alignItems: "center",
  },
  diagnosisTitle: {
    color: "#0B2A22",
    fontSize: 15,
    fontWeight: "900",
  },
  diagnosisText: {
    color: "#65786D",
    fontSize: 12,
    fontWeight: "800",
    marginTop: 4,
  },
  statusBadge: {
    overflow: "hidden",
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 7,
    fontSize: 11,
    fontWeight: "900",
    maxWidth: 120,
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
