import React, { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";

import {
  analyzeLatestSensorReading,
  getLatestSensorReadings,
  getSensorAnalytics,
  getSensorHistory,
} from "../../services/api";

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
};

export default function SensorsScreen() {
  const [latestReading, setLatestReading] = useState<any>(null);
  const [historyResponse, setHistoryResponse] = useState<any>(null);
  const [analyticsResponse, setAnalyticsResponse] = useState<any>(null);

  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const historyList = useMemo(() => {
    return extractArray(historyResponse).slice(0, 6);
  }, [historyResponse]);

  useEffect(() => {
    loadSensorsData();
  }, []);

  async function loadSensorsData() {
    try {
      setLoading(true);
      setErrorMessage(null);

      const latest = await getLatestSensorReadings();
      setLatestReading(latest);

      try {
        const history = await getSensorHistory(latest?.sectorId || undefined);
        setHistoryResponse(history);
      } catch {
        setHistoryResponse(null);
      }

      try {
        const analytics = await getSensorAnalytics(
          latest?.sectorId || undefined,
        );
        setAnalyticsResponse(analytics);
      } catch {
        setAnalyticsResponse(null);
      }
    } catch (error: any) {
      setLatestReading(null);
      setErrorMessage(
        error?.message || "Could not load hardware readings from backend.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleAnalyzeLatestReading() {
    try {
      if (!latestReading?.sectorId) {
        Alert.alert(
          "Sector Required",
          "No sector ID found for the latest hardware reading.",
        );
        return;
      }

      setAnalyzing(true);

      await analyzeLatestSensorReading(latestReading.sectorId);

      await loadSensorsData();

      Alert.alert(
        "Analysis Completed",
        "Backend sensor analysis was updated successfully.",
      );
    } catch (error: any) {
      Alert.alert(
        "Analysis Failed",
        error?.message || "Could not analyze latest sensor values.",
      );
    } finally {
      setAnalyzing(false);
    }
  }

  const connected = !!latestReading;

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={loading} onRefresh={loadSensorsData} />
      }
    >
      <View style={styles.header}>
        <Text style={styles.kicker}>ECOSENSE AI</Text>
        <Text style={styles.title}>Connected Sensors</Text>
        <Text style={styles.subtitle}>
          Hardware readings are loaded from the backend account used on the web.
        </Text>
      </View>

      <View
        style={[styles.statusCard, connected && styles.statusCardConnected]}
      >
        <Text style={styles.statusLabel}>Hardware Status</Text>
        <Text style={[styles.statusTitle, connected && styles.connectedText]}>
          {connected ? "Connected" : "No Reading"}
        </Text>
        <Text style={styles.statusDescription}>
          {connected
            ? "The mobile app received the latest sensor reading from the backend."
            : errorMessage || "No hardware reading was received yet."}
        </Text>
      </View>

      <View style={styles.card}>
        <View style={styles.rowBetween}>
          <Text style={styles.cardTitle}>Latest Reading</Text>

          <TouchableOpacity
            style={styles.smallButton}
            onPress={loadSensorsData}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color={COLORS.white} />
            ) : (
              <Text style={styles.smallButtonText}>Refresh</Text>
            )}
          </TouchableOpacity>
        </View>

        {latestReading ? (
          <>
            <View style={styles.topGrid}>
              <InfoBox
                label="Sector"
                value={latestReading.sectorName || "Unknown"}
              />
              <InfoBox
                label="Crop"
                value={latestReading.cropType || "Unknown"}
              />
            </View>

            <View style={styles.readingGrid}>
              <ReadingBox
                label="Temperature"
                value={`${latestReading.temperature ?? 0}°C`}
                note={getTemperatureNote(latestReading.temperature)}
              />

              <ReadingBox
                label="Humidity"
                value={`${latestReading.humidity ?? 0}%`}
                note={getHumidityNote(latestReading.humidity)}
              />

              <ReadingBox
                label="Soil Moisture"
                value={`${latestReading.soilMoisture ?? 0}%`}
                note={getSoilMoistureNote(latestReading.soilMoisture)}
              />

              <ReadingBox
                label="Soil Temp"
                value={`${latestReading.soilTemp ?? 0}°C`}
                note={getSoilTempNote(latestReading.soilTemp)}
              />

              <View style={styles.fullWidthBox}>
                <Text style={styles.readingLabel}>Light</Text>
                <Text style={styles.readingValue}>
                  {String(latestReading.light || "Unknown")}
                </Text>
                <Text style={styles.readingNote}>
                  Light level from hardware reading
                </Text>
              </View>
            </View>

            <Text style={styles.metaText}>
              Device: {latestReading.deviceSerial || "Unknown"}
            </Text>

            <Text style={styles.metaText}>
              Last update: {formatDate(latestReading.updatedAt)}
            </Text>

            <SensorBackendAnalysis reading={latestReading} />

            <TouchableOpacity
              style={[styles.analyzeButton, analyzing && styles.disabledButton]}
              onPress={handleAnalyzeLatestReading}
              disabled={analyzing}
            >
              {analyzing ? (
                <ActivityIndicator color={COLORS.white} />
              ) : (
                <Text style={styles.analyzeButtonText}>
                  Analyze Latest Values
                </Text>
              )}
            </TouchableOpacity>
          </>
        ) : (
          <View style={styles.emptyBox}>
            <Text style={styles.emptyTitle}>No hardware data</Text>
            <Text style={styles.emptyText}>
              Press Refresh after the hardware sends a new reading to the
              backend.
            </Text>
          </View>
        )}
      </View>

      {analyticsResponse && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Sensor Analytics</Text>
          <Text style={styles.cardSubtitle}>
            Summary data returned from the backend analytics endpoint.
          </Text>

          <AnalyticsPreview data={analyticsResponse} />
        </View>
      )}

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recent Hardware Readings</Text>
        <Text style={styles.cardSubtitle}>
          Latest records returned from sensor history.
        </Text>

        {historyList.length > 0 ? (
          historyList.map((item: any, index: number) => {
            const normalized = normalizeHistoryItem(item);

            return (
              <View key={normalized.id || index} style={styles.historyItem}>
                <View style={styles.historyHeader}>
                  <Text style={styles.historyTitle}>
                    {normalized.sectorName || "Sensor Reading"}
                  </Text>
                  <Text style={styles.historyDate}>
                    {formatDate(normalized.updatedAt)}
                  </Text>
                </View>

                <Text style={styles.historyText}>
                  Temp: {normalized.temperature}°C · Humidity:{" "}
                  {normalized.humidity}% · Soil: {normalized.soilMoisture}% ·
                  Soil Temp: {normalized.soilTemp}°C
                </Text>

                {normalized.finalStatus && (
                  <View
                    style={[
                      styles.statusBadgeSmall,
                      getStatusStyle(normalized.finalStatus),
                    ]}
                  >
                    <Text style={styles.statusTextSmall}>
                      {normalized.finalStatus}
                    </Text>
                  </View>
                )}
              </View>
            );
          })
        ) : (
          <Text style={styles.emptyText}>
            No recent history was returned from the backend.
          </Text>
        )}
      </View>
    </ScrollView>
  );
}

function InfoBox({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.infoBox}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

function ReadingBox({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <View style={styles.readingBox}>
      <Text style={styles.readingLabel}>{label}</Text>
      <Text style={styles.readingValue}>{value}</Text>
      <Text style={styles.readingNote}>{note}</Text>
    </View>
  );
}

function SensorBackendAnalysis({ reading }: { reading: any }) {
  const status = reading?.finalStatus || reading?.status;
  const confidence = reading?.confidence;
  const recommendation = reading?.recommendation;
  const recommendations = Array.isArray(reading?.recommendations)
    ? reading.recommendations
    : [];
  const actions = Array.isArray(reading?.actions) ? reading.actions : [];
  const riskFactors = Array.isArray(reading?.riskFactors)
    ? reading.riskFactors
    : [];

  if (!status && !recommendation && recommendations.length === 0) {
    return null;
  }

  return (
    <View style={styles.analysisBox}>
      <Text style={styles.analysisTitle}>Backend Sensor Analysis</Text>

      {status && (
        <View style={[styles.statusBadgeSmall, getStatusStyle(status)]}>
          <Text style={styles.statusTextSmall}>{status}</Text>
        </View>
      )}

      {confidence !== null && confidence !== undefined && (
        <Text style={styles.analysisText}>
          Confidence: {formatConfidence(confidence)}
        </Text>
      )}

      {recommendation && (
        <Text style={styles.analysisText}>{String(recommendation)}</Text>
      )}

      {recommendations.length > 0 && (
        <>
          <Text style={styles.analysisSubTitle}>Recommendations</Text>
          {recommendations.map((item: any, index: number) => (
            <Text key={`rec-${index}`} style={styles.analysisText}>
              • {String(item)}
            </Text>
          ))}
        </>
      )}

      {actions.length > 0 && (
        <>
          <Text style={styles.analysisSubTitle}>Actions</Text>
          {actions.map((action: any, index: number) => (
            <Text key={`action-${index}`} style={styles.analysisText}>
              • {action?.title || action?.code || String(action)}
            </Text>
          ))}
        </>
      )}

      {riskFactors.length > 0 && (
        <>
          <Text style={styles.analysisSubTitle}>Risk Factors</Text>
          {riskFactors.map((factor: any, index: number) => (
            <Text key={`risk-${index}`} style={styles.analysisText}>
              • {factor?.label || factor?.code || "Risk"}{" "}
              {factor?.value !== undefined ? `(${factor.value})` : ""}
            </Text>
          ))}
        </>
      )}
    </View>
  );
}

function AnalyticsPreview({ data }: { data: any }) {
  const source = data?.data || data?.analytics || data;

  const totalReadings =
    source?.totalReadings ||
    source?.count ||
    source?.readingsCount ||
    source?.total ||
    null;

  const averageTemp =
    source?.averageTemperature ||
    source?.avgTemperature ||
    source?.avgTemp ||
    source?.temperatureAvg ||
    null;

  const averageHumidity =
    source?.averageHumidity ||
    source?.avgHumidity ||
    source?.humidityAvg ||
    null;

  const averageSoil =
    source?.averageSoilMoisture ||
    source?.avgSoilMoisture ||
    source?.soilMoistureAvg ||
    null;

  return (
    <View style={styles.analyticsGrid}>
      <InfoBox
        label="Total Readings"
        value={totalReadings !== null ? String(totalReadings) : "Available"}
      />
      <InfoBox
        label="Avg Temp"
        value={averageTemp !== null ? `${roundValue(averageTemp)}°C` : "-"}
      />
      <InfoBox
        label="Avg Humidity"
        value={
          averageHumidity !== null ? `${roundValue(averageHumidity)}%` : "-"
        }
      />
      <InfoBox
        label="Avg Soil"
        value={averageSoil !== null ? `${roundValue(averageSoil)}%` : "-"}
      />
    </View>
  );
}

function extractArray(response: any) {
  if (Array.isArray(response)) return response;
  if (Array.isArray(response?.data)) return response.data;
  if (Array.isArray(response?.history)) return response.history;
  if (Array.isArray(response?.readings)) return response.readings;
  if (Array.isArray(response?.data?.history)) return response.data.history;
  if (Array.isArray(response?.data?.readings)) return response.data.readings;
  if (Array.isArray(response?.data?.data)) return response.data.data;
  return [];
}

function normalizeHistoryItem(item: any) {
  const air = item?.air || {};
  const soil = item?.soil || {};
  const sector =
    typeof item?.sectorId === "object" && item?.sectorId !== null
      ? item.sectorId
      : null;

  const temperature = numberValue(
    item?.temperature ?? item?.temp ?? air?.temperature ?? 0,
  );

  const humidity = numberValue(
    item?.humidity ?? item?.hum ?? air?.humidity ?? 0,
  );

  const soilMoisture = numberValue(
    item?.soilMoisture ?? item?.Soil ?? item?.soil ?? soil?.moisture ?? 0,
  );

  const soilTemp = numberValue(
    item?.soilTemp ??
      item?.soilTemperature ??
      soil?.temperature ??
      getRiskFactorValue(item, "HIGH_SOIL_TEMP") ??
      0,
  );

  return {
    id: item?._id || item?.id || `${Date.now()}`,
    sectorName: sector?.name || item?.sectorName || "Unknown Sector",
    temperature,
    humidity,
    soilMoisture,
    soilTemp,
    light: item?.light || "Unknown",
    finalStatus:
      item?.analysis?.final_status ||
      item?.analysis?.status ||
      item?.finalStatus ||
      item?.status ||
      null,
    updatedAt:
      item?.updatedAt ||
      item?.createdAt ||
      item?.timestamp ||
      item?.analysis?.timestamp ||
      null,
  };
}

function getRiskFactorValue(reading: any, code: string) {
  const riskFactors =
    reading?.analysis?.risk_factors ||
    reading?.analysis?.riskFactors ||
    reading?.riskFactors ||
    reading?.risk_factors ||
    [];

  if (!Array.isArray(riskFactors)) return undefined;

  const factor = riskFactors.find(
    (item: any) =>
      String(item?.code || "").toLowerCase() === code.toLowerCase(),
  );

  return factor?.value;
}

function numberValue(value: any) {
  const num = Number(value);

  if (!Number.isFinite(num)) return 0;

  return num;
}

function roundValue(value: any) {
  const num = Number(value);

  if (!Number.isFinite(num)) return "-";

  return Math.round(num * 10) / 10;
}

function formatDate(value: any) {
  if (!value) return "Not available";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString();
}

function formatConfidence(value: any) {
  const num = Number(value);

  if (!Number.isFinite(num)) {
    return String(value);
  }

  if (num <= 1) {
    return `${Math.round(num * 100)}%`;
  }

  return `${Math.round(num)}%`;
}

function getTemperatureNote(value: any) {
  const num = Number(value);

  if (!Number.isFinite(num) || num === 0) return "No temperature reading";
  if (num >= 35) return "High temperature warning";
  if (num <= 10) return "Low temperature warning";

  return "Temperature is available";
}

function getHumidityNote(value: any) {
  const num = Number(value);

  if (!Number.isFinite(num) || num === 0) return "No humidity reading";
  if (num <= 30) return "Low humidity detected";
  if (num >= 80) return "High humidity detected";

  return "Humidity is available";
}

function getSoilMoistureNote(value: any) {
  const num = Number(value);

  if (!Number.isFinite(num)) return "No soil moisture reading";
  if (num === 0) return "Soil moisture value is 0 from hardware";
  if (num <= 20) return "Irrigation may be required";

  return "Soil moisture is available";
}

function getSoilTempNote(value: any) {
  const num = Number(value);

  if (!Number.isFinite(num) || num === 0) return "No soil temperature reading";
  if (num >= 32) return "Root zone temperature is high";
  if (num <= 10) return "Root zone temperature is low";

  return "Soil temperature is available";
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
      borderColor: "#0284C7",
    };
  }

  return {
    backgroundColor: COLORS.mint,
    borderColor: COLORS.primary,
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
    fontSize: 28,
    fontWeight: "900",
    marginTop: 6,
  },
  subtitle: {
    color: COLORS.muted,
    fontSize: 14,
    lineHeight: 21,
    marginTop: 8,
  },
  statusCard: {
    backgroundColor: COLORS.white,
    borderRadius: 24,
    padding: 20,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  statusCardConnected: {
    backgroundColor: COLORS.mint,
    borderColor: "#BDEBD0",
  },
  statusLabel: {
    color: COLORS.muted,
    fontSize: 14,
    fontWeight: "900",
  },
  statusTitle: {
    color: COLORS.danger,
    fontSize: 36,
    fontWeight: "900",
    marginTop: 12,
  },
  connectedText: {
    color: COLORS.primary,
  },
  statusDescription: {
    color: COLORS.muted,
    fontSize: 15,
    lineHeight: 22,
    marginTop: 10,
  },
  card: {
    backgroundColor: COLORS.white,
    borderRadius: 22,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  rowBetween: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  cardTitle: {
    color: COLORS.dark,
    fontSize: 21,
    fontWeight: "900",
  },
  cardSubtitle: {
    color: COLORS.muted,
    fontSize: 13,
    lineHeight: 20,
    marginTop: 6,
  },
  smallButton: {
    backgroundColor: COLORS.primary,
    paddingVertical: 9,
    paddingHorizontal: 14,
    borderRadius: 12,
  },
  smallButtonText: {
    color: COLORS.white,
    fontSize: 12,
    fontWeight: "900",
  },
  topGrid: {
    flexDirection: "row",
    gap: 10,
    marginTop: 16,
  },
  infoBox: {
    flex: 1,
    backgroundColor: COLORS.soft,
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    minHeight: 82,
  },
  infoLabel: {
    color: COLORS.muted,
    fontSize: 12,
    fontWeight: "800",
  },
  infoValue: {
    color: COLORS.dark,
    fontSize: 16,
    fontWeight: "900",
    marginTop: 8,
  },
  readingGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 14,
  },
  readingBox: {
    width: "47%",
    backgroundColor: COLORS.soft,
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    minHeight: 128,
  },
  fullWidthBox: {
    width: "100%",
    backgroundColor: COLORS.soft,
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    minHeight: 110,
  },
  readingLabel: {
    color: COLORS.muted,
    fontSize: 13,
    fontWeight: "900",
  },
  readingValue: {
    color: COLORS.dark,
    fontSize: 28,
    fontWeight: "900",
    marginTop: 10,
  },
  readingNote: {
    color: COLORS.muted,
    fontSize: 13,
    lineHeight: 19,
    marginTop: 8,
  },
  metaText: {
    color: COLORS.muted,
    fontSize: 12,
    marginTop: 12,
    fontWeight: "700",
  },
  analysisBox: {
    backgroundColor: COLORS.mint,
    borderRadius: 16,
    padding: 14,
    marginTop: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  analysisTitle: {
    color: COLORS.dark,
    fontSize: 16,
    fontWeight: "900",
  },
  analysisSubTitle: {
    color: COLORS.dark,
    fontSize: 13,
    fontWeight: "900",
    marginTop: 10,
  },
  analysisText: {
    color: COLORS.text,
    fontSize: 13,
    lineHeight: 20,
    marginTop: 6,
  },
  statusBadgeSmall: {
    alignSelf: "flex-start",
    borderRadius: 999,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderWidth: 1,
    marginTop: 10,
  },
  statusTextSmall: {
    color: COLORS.dark,
    fontSize: 12,
    fontWeight: "900",
  },
  analyzeButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 16,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: 14,
  },
  disabledButton: {
    opacity: 0.65,
  },
  analyzeButtonText: {
    color: COLORS.white,
    fontSize: 14,
    fontWeight: "900",
  },
  emptyBox: {
    backgroundColor: COLORS.soft,
    borderRadius: 16,
    padding: 16,
    marginTop: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  emptyTitle: {
    color: COLORS.dark,
    fontSize: 16,
    fontWeight: "900",
  },
  emptyText: {
    color: COLORS.muted,
    fontSize: 13,
    lineHeight: 20,
    marginTop: 8,
  },
  analyticsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 14,
  },
  historyItem: {
    backgroundColor: COLORS.soft,
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    marginTop: 12,
  },
  historyHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 10,
  },
  historyTitle: {
    flex: 1,
    color: COLORS.dark,
    fontSize: 14,
    fontWeight: "900",
  },
  historyDate: {
    color: COLORS.muted,
    fontSize: 11,
    fontWeight: "700",
  },
  historyText: {
    color: COLORS.muted,
    fontSize: 12,
    lineHeight: 18,
    marginTop: 8,
  },
});
