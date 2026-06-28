import { useRouter } from "expo-router";
import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import {
  clearDiagnosisHistory,
  getDiagnosisHistory,
  getMyDiagnosesFromBackend,
} from "../services/api";

const COLORS = {
  dark: "#0F3D2E",
  primary: "#1F7A4D",
  mint: "#E8F5EE",
  soft: "#F6FAF7",
  white: "#FFFFFF",
  text: "#1F2A24",
  muted: "#6B7A70",
  border: "#DDE8E1",
  warning: "#F59E0B",
  danger: "#DC2626",
  success: "#16A34A",
  blue: "#2563EB",
};

export default function MyDiagnosesScreen() {
  const router = useRouter();

  const [diagnoses, setDiagnoses] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [source, setSource] = useState<"backend" | "local" | "empty">("empty");
  const [filter, setFilter] = useState<"all" | "healthy" | "moderate" | "high">(
    "all",
  );

  useEffect(() => {
    loadDiagnoses();
  }, []);

  async function loadDiagnoses() {
    try {
      setLoading(true);

      let backendList: any[] = [];

      try {
        const backendResponse = await getMyDiagnosesFromBackend();
        backendList = extractArray(backendResponse);
      } catch {
        backendList = [];
      }

      if (backendList.length > 0) {
        setDiagnoses(backendList);
        setSource("backend");
        return;
      }

      const localList = await getDiagnosisHistory();

      if (Array.isArray(localList) && localList.length > 0) {
        setDiagnoses(localList);
        setSource("local");
        return;
      }

      setDiagnoses([]);
      setSource("empty");
    } catch {
      setDiagnoses([]);
      setSource("empty");
    } finally {
      setLoading(false);
    }
  }

  async function handleClearLocalHistory() {
    Alert.alert(
      "Clear Local Diagnoses",
      "This will clear local saved diagnoses from this phone only.",
      [
        {
          text: "Cancel",
          style: "cancel",
        },
        {
          text: "Clear",
          style: "destructive",
          onPress: async () => {
            await clearDiagnosisHistory();
            await loadDiagnoses();
          },
        },
      ],
    );
  }

  const filteredDiagnoses = useMemo(() => {
    if (filter === "all") return diagnoses;

    return diagnoses.filter((item) => {
      const status = getFinalStatus(item).toLowerCase();

      if (filter === "healthy") return status.includes("healthy");
      if (filter === "moderate") return status.includes("moderate");
      if (filter === "high") return status.includes("high");

      return true;
    });
  }, [diagnoses, filter]);

  const stats = useMemo(() => {
    const total = diagnoses.length;

    const healthy = diagnoses.filter((item) =>
      getFinalStatus(item).toLowerCase().includes("healthy"),
    ).length;

    const moderate = diagnoses.filter((item) =>
      getFinalStatus(item).toLowerCase().includes("moderate"),
    ).length;

    const high = diagnoses.filter((item) =>
      getFinalStatus(item).toLowerCase().includes("high"),
    ).length;

    return {
      total,
      healthy,
      moderate,
      high,
    };
  }, [diagnoses]);

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={loading} onRefresh={loadDiagnoses} />
      }
    >
      <View style={styles.header}>
        <Text style={styles.kicker}>ECOSENSE AI</Text>
        <Text style={styles.title}>My Diagnoses</Text>
        <Text style={styles.subtitle}>
          Saved plant diagnosis results from image, sensors, and combined AI
          analysis.
        </Text>
      </View>

      <View style={styles.summaryCard}>
        <View>
          <Text style={styles.summaryLabel}>Saved Records</Text>
          <Text style={styles.summaryNumber}>{stats.total}</Text>
          <Text style={styles.summarySource}>
            Source:{" "}
            {source === "backend"
              ? "Backend"
              : source === "local"
                ? "Local Storage"
                : "No Data"}
          </Text>
        </View>

        <TouchableOpacity
          style={styles.refreshButton}
          onPress={loadDiagnoses}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color={COLORS.white} />
          ) : (
            <Text style={styles.refreshButtonText}>Refresh</Text>
          )}
        </TouchableOpacity>
      </View>

      <View style={styles.statsGrid}>
        <StatBox label="Healthy" value={stats.healthy} type="healthy" />
        <StatBox label="Moderate" value={stats.moderate} type="moderate" />
        <StatBox label="High" value={stats.high} type="high" />
      </View>

      <View style={styles.filterCard}>
        <Text style={styles.cardTitle}>Filters</Text>

        <View style={styles.filterRow}>
          <FilterButton
            title="All"
            active={filter === "all"}
            onPress={() => setFilter("all")}
          />
          <FilterButton
            title="Healthy"
            active={filter === "healthy"}
            onPress={() => setFilter("healthy")}
          />
          <FilterButton
            title="Moderate"
            active={filter === "moderate"}
            onPress={() => setFilter("moderate")}
          />
          <FilterButton
            title="High"
            active={filter === "high"}
            onPress={() => setFilter("high")}
          />
        </View>
      </View>

      {filteredDiagnoses.length === 0 ? (
        <View style={styles.emptyCard}>
          <Text style={styles.emptyTitle}>No diagnoses found</Text>
          <Text style={styles.emptyText}>
            Run a diagnosis first from the Diagnosis Center, then come back
            here.
          </Text>

          <TouchableOpacity
            style={styles.mainButton}
            onPress={() => router.push("/(tabs)/diagnosis" as any)}
          >
            <Text style={styles.mainButtonText}>Open Diagnosis Center</Text>
          </TouchableOpacity>
        </View>
      ) : (
        filteredDiagnoses.map((item, index) => (
          <DiagnosisCard
            key={getRecordId(item, index)}
            item={item}
            index={index}
          />
        ))
      )}

      {source === "local" && diagnoses.length > 0 && (
        <TouchableOpacity
          style={styles.clearButton}
          onPress={handleClearLocalHistory}
        >
          <Text style={styles.clearButtonText}>Clear Local History</Text>
        </TouchableOpacity>
      )}
    </ScrollView>
  );
}

function DiagnosisCard({ item, index }: { item: any; index: number }) {
  const status = getFinalStatus(item);
  const mode = getMode(item);
  const crop = getCropType(item);
  const sector = getSectorName(item);
  const diagnosis = getDiagnosisText(item);
  const confidence = getConfidence(item);
  const recommendations = getRecommendations(item);
  const actions = getActions(item);
  const imageUri = getImageUri(item);
  const createdAt = getCreatedAt(item);

  return (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={styles.cardHeaderText}>
          <Text style={styles.cardTitle}>Diagnosis #{index + 1}</Text>
          <Text style={styles.cardDate}>{formatDate(createdAt)}</Text>
        </View>

        <View style={[styles.statusBadge, getStatusStyle(status)]}>
          <Text style={styles.statusText}>{safeText(status || "Unknown")}</Text>
        </View>
      </View>

      {imageUri ? (
        <Image source={{ uri: imageUri }} style={styles.previewImage} />
      ) : null}

      <View style={styles.infoGrid}>
        <InfoItem label="Mode" value={mode} />
        <InfoItem label="Crop" value={crop} />
        <InfoItem label="Sector" value={sector} />
        <InfoItem label="Confidence" value={confidence} />
      </View>

      <Text style={styles.sectionTitle}>Diagnosis</Text>
      <Text style={styles.bodyText}>{safeText(diagnosis)}</Text>

      {recommendations.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Recommendations</Text>
          {recommendations.map((rec, recIndex) => (
            <Text key={`rec-${recIndex}`} style={styles.bulletText}>
              • {safeText(rec)}
            </Text>
          ))}
        </>
      )}

      {actions.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Actions</Text>
          {actions.map((action, actionIndex) => (
            <Text key={`action-${actionIndex}`} style={styles.bulletText}>
              • {safeText(action?.title || action?.code || action)}
            </Text>
          ))}
        </>
      )}
    </View>
  );
}

function StatBox({
  label,
  value,
  type,
}: {
  label: string;
  value: number;
  type: "healthy" | "moderate" | "high";
}) {
  return (
    <View style={[styles.statBox, getStatStyle(type)]}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
    </View>
  );
}

function FilterButton({
  title,
  active,
  onPress,
}: {
  title: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity
      style={[styles.filterButton, active && styles.filterButtonActive]}
      onPress={onPress}
    >
      <Text
        style={[
          styles.filterButtonText,
          active && styles.filterButtonTextActive,
        ]}
      >
        {title}
      </Text>
    </TouchableOpacity>
  );
}

function InfoItem({ label, value }: { label: string; value: any }) {
  return (
    <View style={styles.infoItem}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{safeText(value || "Unknown")}</Text>
    </View>
  );
}

function extractArray(response: any) {
  if (Array.isArray(response)) return response;
  if (Array.isArray(response?.data)) return response.data;
  if (Array.isArray(response?.diagnoses)) return response.diagnoses;
  if (Array.isArray(response?.items)) return response.items;
  if (Array.isArray(response?.results)) return response.results;
  if (Array.isArray(response?.data?.diagnoses)) return response.data.diagnoses;
  if (Array.isArray(response?.data?.items)) return response.data.items;
  if (Array.isArray(response?.data?.results)) return response.data.results;
  if (Array.isArray(response?.data?.data)) return response.data.data;

  return [];
}

function getRecordId(item: any, index: number) {
  return String(item?._id || item?.id || item?.createdAt || index);
}

function getFinalStatus(item: any) {
  return safeText(
    item?.finalStatus ||
      item?.final_status ||
      item?.status ||
      item?.analysis?.final_status ||
      item?.analysis?.status ||
      item?.analysisResult?.status ||
      item?.raw?.final_status ||
      item?.raw?.status ||
      "Unknown",
  );
}

function getMode(item: any) {
  return safeText(
    item?.mode || item?.analysisType || item?.type || "Diagnosis",
  );
}

function getCropType(item: any) {
  return safeText(
    item?.cropType ||
      item?.crop ||
      item?.plantType ||
      item?.sector?.cropType ||
      item?.sectorId?.cropType ||
      item?.sensorReadings?.cropType ||
      "Unknown",
  );
}

function getSectorName(item: any) {
  return safeText(
    item?.sectorName ||
      item?.sector?.name ||
      item?.sectorId?.name ||
      item?.sector ||
      item?.sensorReadings?.sectorName ||
      "Unknown",
  );
}

function getDiagnosisText(item: any) {
  return safeText(
    item?.diagnosisText ||
      item?.diagnosis ||
      item?.recommendation ||
      item?.general_recommendation ||
      item?.analysis?.recommendation ||
      item?.analysisResult?.diseaseName ||
      item?.analysisResult?.note ||
      item?.raw?.diagnosis ||
      item?.raw?.recommendation ||
      "No diagnosis text available.",
  );
}

function getConfidence(item: any) {
  const value =
    item?.confidence ||
    item?.final_confidence ||
    item?.finalConfidence ||
    item?.analysis?.final_confidence ||
    item?.analysisResult?.confidence ||
    item?.raw?.confidence;

  if (value === undefined || value === null) return "Unknown";

  const num = Number(value);

  if (!Number.isFinite(num)) return safeText(value);

  if (num <= 1) return `${Math.round(num * 100)}%`;

  return `${Math.round(num)}%`;
}

function getRecommendations(item: any) {
  const value =
    item?.recommendations ||
    item?.analysis?.recommendations ||
    item?.analysisResult?.recommendations ||
    item?.raw?.recommendations ||
    [];

  if (Array.isArray(value)) return value;

  if (typeof value === "string" && value.trim()) return [value];

  return [];
}

function getActions(item: any) {
  const value =
    item?.actions ||
    item?.analysis?.actions ||
    item?.analysisResult?.treatmentPlan ||
    item?.treatmentPlan ||
    item?.raw?.actions ||
    [];

  if (Array.isArray(value)) return value;

  return [];
}

function getImageUri(item: any) {
  const uri =
    item?.imageUri ||
    item?.imageUrl ||
    item?.image ||
    item?.photoUrl ||
    item?.raw?.imageUrl ||
    null;

  if (!uri) return null;

  return String(uri);
}

function getCreatedAt(item: any) {
  return (
    item?.createdAt ||
    item?.updatedAt ||
    item?.timestamp ||
    item?.raw?.createdAt ||
    null
  );
}

function safeText(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }

  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  if (Array.isArray(value)) {
    const textValues: string[] = [];

    for (const entry of value) {
      textValues.push(safeText(entry));
    }

    return textValues.join(", ");
  }

  if (typeof value === "object") {
    const objectValue = value as Record<string, unknown>;

    if (objectValue.title) return safeText(objectValue.title);
    if (objectValue.name) return safeText(objectValue.name);
    if (objectValue.label) return safeText(objectValue.label);
    if (objectValue.message) return safeText(objectValue.message);
    if (objectValue.text) return safeText(objectValue.text);

    try {
      return JSON.stringify(objectValue);
    } catch {
      return "Object";
    }
  }

  return String(value);
}

function formatDate(value: any) {
  if (!value) return "No date";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return safeText(value);
  }

  return date.toLocaleString();
}

function getStatusStyle(status: string) {
  const value = String(status || "").toLowerCase();

  if (value.includes("high")) {
    return {
      backgroundColor: "#FEE2E2",
      borderColor: COLORS.danger,
    };
  }

  if (value.includes("moderate")) {
    return {
      backgroundColor: "#FEF3C7",
      borderColor: COLORS.warning,
    };
  }

  if (value.includes("healthy")) {
    return {
      backgroundColor: "#DCFCE7",
      borderColor: COLORS.success,
    };
  }

  if (value.includes("detected")) {
    return {
      backgroundColor: "#E0F2FE",
      borderColor: COLORS.blue,
    };
  }

  return {
    backgroundColor: COLORS.mint,
    borderColor: COLORS.primary,
  };
}

function getStatStyle(type: "healthy" | "moderate" | "high") {
  if (type === "healthy") {
    return {
      backgroundColor: "#DCFCE7",
      borderColor: "#BBF7D0",
    };
  }

  if (type === "moderate") {
    return {
      backgroundColor: "#FEF3C7",
      borderColor: "#FDE68A",
    };
  }

  return {
    backgroundColor: "#FEE2E2",
    borderColor: "#FECACA",
  };
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.soft,
  },
  content: {
    padding: 18,
    paddingBottom: 36,
  },
  header: {
    marginBottom: 16,
  },
  kicker: {
    color: COLORS.primary,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 1,
  },
  title: {
    color: COLORS.dark,
    fontSize: 30,
    fontWeight: "900",
    marginTop: 6,
  },
  subtitle: {
    color: COLORS.muted,
    fontSize: 14,
    lineHeight: 21,
    marginTop: 8,
  },
  summaryCard: {
    backgroundColor: COLORS.dark,
    borderRadius: 24,
    padding: 20,
    marginBottom: 14,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  summaryLabel: {
    color: COLORS.mint,
    fontSize: 13,
    fontWeight: "800",
  },
  summaryNumber: {
    color: COLORS.white,
    fontSize: 42,
    fontWeight: "900",
    marginTop: 6,
  },
  summarySource: {
    color: COLORS.mint,
    fontSize: 12,
    marginTop: 6,
    fontWeight: "700",
  },
  refreshButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 16,
    minWidth: 86,
    alignItems: "center",
  },
  refreshButtonText: {
    color: COLORS.white,
    fontSize: 13,
    fontWeight: "900",
  },
  statsGrid: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 14,
  },
  statBox: {
    flex: 1,
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
  },
  statLabel: {
    color: COLORS.muted,
    fontSize: 12,
    fontWeight: "900",
  },
  statValue: {
    color: COLORS.dark,
    fontSize: 26,
    fontWeight: "900",
    marginTop: 8,
  },
  filterCard: {
    backgroundColor: COLORS.white,
    borderRadius: 20,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  cardTitle: {
    color: COLORS.dark,
    fontSize: 18,
    fontWeight: "900",
  },
  filterRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 12,
  },
  filterButton: {
    paddingVertical: 9,
    paddingHorizontal: 14,
    borderRadius: 999,
    backgroundColor: COLORS.soft,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  filterButtonActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  filterButtonText: {
    color: COLORS.muted,
    fontSize: 12,
    fontWeight: "900",
  },
  filterButtonTextActive: {
    color: COLORS.white,
  },
  emptyCard: {
    backgroundColor: COLORS.white,
    borderRadius: 22,
    padding: 18,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  emptyTitle: {
    color: COLORS.dark,
    fontSize: 20,
    fontWeight: "900",
  },
  emptyText: {
    color: COLORS.muted,
    fontSize: 14,
    lineHeight: 21,
    marginTop: 8,
  },
  mainButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: 16,
  },
  mainButtonText: {
    color: COLORS.white,
    fontSize: 14,
    fontWeight: "900",
  },
  card: {
    backgroundColor: COLORS.white,
    borderRadius: 22,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  cardHeader: {
    flexDirection: "row",
    gap: 10,
    alignItems: "flex-start",
    justifyContent: "space-between",
  },
  cardHeaderText: {
    flex: 1,
  },
  cardDate: {
    color: COLORS.muted,
    fontSize: 12,
    marginTop: 4,
    fontWeight: "700",
  },
  statusBadge: {
    borderRadius: 999,
    paddingVertical: 7,
    paddingHorizontal: 11,
    borderWidth: 1,
  },
  statusText: {
    color: COLORS.dark,
    fontSize: 12,
    fontWeight: "900",
  },
  previewImage: {
    width: "100%",
    height: 210,
    borderRadius: 18,
    backgroundColor: COLORS.mint,
    marginTop: 14,
  },
  infoGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 14,
  },
  infoItem: {
    width: "47%",
    backgroundColor: COLORS.soft,
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  infoLabel: {
    color: COLORS.muted,
    fontSize: 12,
    fontWeight: "800",
  },
  infoValue: {
    color: COLORS.dark,
    fontSize: 14,
    fontWeight: "900",
    marginTop: 6,
  },
  sectionTitle: {
    color: COLORS.dark,
    fontSize: 15,
    fontWeight: "900",
    marginTop: 16,
    marginBottom: 6,
  },
  bodyText: {
    color: COLORS.text,
    fontSize: 14,
    lineHeight: 21,
  },
  bulletText: {
    color: COLORS.text,
    fontSize: 14,
    lineHeight: 22,
    marginTop: 4,
  },
  clearButton: {
    backgroundColor: "#FEE2E2",
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#FECACA",
  },
  clearButtonText: {
    color: COLORS.danger,
    fontSize: 14,
    fontWeight: "900",
  },
});
