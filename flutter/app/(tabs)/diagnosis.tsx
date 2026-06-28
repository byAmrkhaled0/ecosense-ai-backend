import * as ImagePicker from "expo-image-picker";
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
  analyzeLatestSensorReading,
  getLatestSensorReadings,
  getSectors,
  predictImage,
  predictSensors,
  predictWithImage,
  saveDiagnosisResult,
  uploadPlantImageToBackend,
} from "../../services/api";

type DiagnosisMode = "image" | "sensors" | "combined";

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

export default function DiagnosisScreen() {
  const router = useRouter();

  const [mode, setMode] = useState<DiagnosisMode>("image");

  const [sectors, setSectors] = useState<any[]>([]);
  const [selectedSectorId, setSelectedSectorId] = useState<string | null>(null);
  const [loadingSectors, setLoadingSectors] = useState(false);

  const [latestReading, setLatestReading] = useState<any>(null);
  const [loadingReadings, setLoadingReadings] = useState(false);

  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const [analyzing, setAnalyzing] = useState(false);
  const [uploadingImage, setUploadingImage] = useState(false);
  const [aiUnavailable, setAiUnavailable] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const selectedSector = useMemo(() => {
    return sectors.find((sector) => getSectorId(sector) === selectedSectorId);
  }, [sectors, selectedSectorId]);

  useEffect(() => {
    loadInitialData();
  }, []);

  async function loadInitialData() {
    await Promise.all([loadSectors(), loadLatestReading()]);
  }

  async function loadSectors() {
    try {
      setLoadingSectors(true);

      const response = await getSectors();
      const list = extractArray(response);

      setSectors(list);

      if (list.length > 0 && !selectedSectorId) {
        const firstId = getSectorId(list[0]);

        if (firstId) {
          setSelectedSectorId(firstId);
        }
      }
    } catch {
      setErrorMessage("Could not load sectors from backend.");
    } finally {
      setLoadingSectors(false);
    }
  }

  async function loadLatestReading() {
    try {
      setLoadingReadings(true);

      const reading = await getLatestSensorReadings();
      setLatestReading(reading);
    } catch {
      setLatestReading(null);
    } finally {
      setLoadingReadings(false);
    }
  }

  async function pickImageFromGallery() {
    try {
      const permission =
        await ImagePicker.requestMediaLibraryPermissionsAsync();

      if (!permission.granted) {
        Alert.alert(
          "Permission Required",
          "Please allow gallery access to choose plant images for diagnosis.",
        );
        return;
      }

      const picked = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ["images"] as any,
        allowsEditing: true,
        quality: 0.85,
      });

      if (!picked.canceled && picked.assets?.[0]?.uri) {
        setImageUri(picked.assets[0].uri);
        setResult(null);
        setAiUnavailable(false);
        setErrorMessage(null);
      }
    } catch (error: any) {
      Alert.alert("Gallery Error", error?.message || "Could not choose image.");
    }
  }

  async function openCamera() {
    try {
      const permission = await ImagePicker.requestCameraPermissionsAsync();

      if (!permission.granted) {
        Alert.alert(
          "Permission Required",
          "Please allow camera access to capture plant images for diagnosis.",
        );
        return;
      }

      const captured = await ImagePicker.launchCameraAsync({
        mediaTypes: ["images"] as any,
        allowsEditing: true,
        quality: 0.85,
      });

      if (!captured.canceled && captured.assets?.[0]?.uri) {
        setImageUri(captured.assets[0].uri);
        setResult(null);
        setAiUnavailable(false);
        setErrorMessage(null);
      }
    } catch (error: any) {
      Alert.alert("Camera Error", error?.message || "Could not open camera.");
    }
  }

  function getCropTypeForDiagnosis() {
    return (
      selectedSector?.cropType ||
      selectedSector?.crop ||
      latestReading?.cropType ||
      "Tomato"
    );
  }

  function buildSensorPayload() {
    if (!latestReading) {
      return null;
    }

    return {
      sectorId: selectedSectorId || latestReading.sectorId || null,
      cropType: getCropTypeForDiagnosis(),
      temperature: Number(latestReading.temperature ?? 0),
      humidity: Number(latestReading.humidity ?? 0),
      soilMoisture: Number(latestReading.soilMoisture ?? 0),
      soilTemp: Number(latestReading.soilTemp ?? 0),
      light: normalizeLightForAI(latestReading.light),
    };
  }

  function isValidSensorPayload(payload: any) {
    if (!payload) return false;

    const temperature = Number(payload.temperature);
    const humidity = Number(payload.humidity);
    const soilMoisture = Number(payload.soilMoisture);
    const soilTemp = Number(payload.soilTemp);

    return (
      Number.isFinite(temperature) &&
      Number.isFinite(humidity) &&
      Number.isFinite(soilMoisture) &&
      Number.isFinite(soilTemp) &&
      temperature > 0 &&
      humidity > 0 &&
      soilTemp > 0 &&
      soilMoisture >= 0
    );
  }

  async function uploadImageToBackendForStorageAndFallback() {
    if (!imageUri) return null;

    if (!selectedSectorId) {
      throw new Error("Please select a sector before uploading the image.");
    }

    setUploadingImage(true);

    try {
      const uploadResponse = await uploadPlantImageToBackend(
        imageUri,
        selectedSectorId,
        latestReading?.deviceSerial || null,
      );

      return uploadResponse;
    } finally {
      setUploadingImage(false);
    }
  }

  async function handleAnalyze() {
    try {
      setAnalyzing(true);
      setAiUnavailable(false);
      setErrorMessage(null);
      setResult(null);

      const needsImage = mode === "image" || mode === "combined";
      const needsSensors = mode === "sensors" || mode === "combined";

      if (needsImage && !imageUri) {
        Alert.alert(
          "Image Required",
          "Please take a plant photo or choose one from gallery first.",
        );
        return;
      }

      if (needsImage && !selectedSectorId) {
        Alert.alert(
          "Select Sector",
          "Please select the farm sector that this image belongs to.",
        );
        return;
      }

      const sensorPayload = buildSensorPayload();

      if (needsSensors && !isValidSensorPayload(sensorPayload)) {
        Alert.alert(
          "Sensor Readings Required",
          "Latest hardware readings are missing or incomplete. Please refresh sensors first.",
        );
        return;
      }

      let backendUploadResponse: any = null;

      if (needsImage) {
        try {
          backendUploadResponse =
            await uploadImageToBackendForStorageAndFallback();
        } catch {
          Alert.alert(
            "Image Upload Warning",
            "The image diagnosis will continue, but backend image storage failed.",
          );
        }
      }

      let aiResponse: any = null;

      if (mode === "image") {
        aiResponse = await predictImage(
          imageUri as string,
          getCropTypeForDiagnosis(),
        );
      }

      if (mode === "sensors") {
        const sectorId = selectedSectorId || latestReading?.sectorId;

        try {
          if (sectorId) {
            aiResponse = await analyzeLatestSensorReading(sectorId);
          } else {
            aiResponse = await predictSensors(sensorPayload);
          }
        } catch {
          aiResponse = await predictSensors(sensorPayload);
        }

        await loadLatestReading();
      }

      if (mode === "combined") {
        aiResponse = await predictWithImage(imageUri as string, sensorPayload);
      }

      if (isExplicitNoPlantResponse(aiResponse)) {
        Alert.alert(
          "Invalid Plant Image",
          getPlantValidationMessage(aiResponse),
        );
        return;
      }

      const finalResponseForDisplay = hasUsefulAIResult(aiResponse)
        ? aiResponse
        : backendUploadResponse || aiResponse;

      if (!finalResponseForDisplay) {
        throw new Error("No diagnosis response was returned.");
      }

      const normalized = normalizeAnalysisResult(finalResponseForDisplay, {
        mode,
        sector: selectedSector,
        sensorPayload,
        imageUri,
      });

      await saveDiagnosisResult(normalized);

      setResult(normalized);

      Alert.alert("Diagnosis Completed", "Diagnosis completed successfully.");
    } catch (error: any) {
      setAiUnavailable(true);
      setErrorMessage(error?.message || "AI service is currently unavailable.");
      Alert.alert(
        "AI Unavailable",
        error?.message || "Could not complete the diagnosis now.",
      );
    } finally {
      setAnalyzing(false);
      setUploadingImage(false);
    }
  }

  function handleShowDemoResult() {
    const demo = buildPresentationDemoResult({
      mode,
      sector: selectedSector,
      sensorPayload: buildSensorPayload(),
      imageUri,
    });

    setResult(demo);
    saveDiagnosisResult(demo);
  }

  const needsImage = mode === "image" || mode === "combined";
  const needsSensors = mode === "sensors" || mode === "combined";

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={false} onRefresh={loadInitialData} />
      }
    >
      <View style={styles.header}>
        <Text style={styles.kicker}>ECOSENSE AI</Text>
        <Text style={styles.title}>Plant Diagnosis Center</Text>
        <Text style={styles.subtitle}>
          Choose a diagnosis mode, select a sector, capture a plant image, and
          run AI analysis.
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Diagnosis Mode</Text>

        <View style={styles.modeGrid}>
          <ModeButton
            title="Image"
            subtitle="AI + fallback"
            active={mode === "image"}
            onPress={() => {
              setMode("image");
              setResult(null);
              setAiUnavailable(false);
            }}
          />

          <ModeButton
            title="Sensors"
            subtitle="Hardware readings"
            active={mode === "sensors"}
            onPress={() => {
              setMode("sensors");
              setResult(null);
              setAiUnavailable(false);
            }}
          />

          <ModeButton
            title="Combined"
            subtitle="Image + Sensors"
            active={mode === "combined"}
            onPress={() => {
              setMode("combined");
              setResult(null);
              setAiUnavailable(false);
            }}
          />
        </View>
      </View>

      {needsImage && (
        <View style={styles.card}>
          <View style={styles.rowBetween}>
            <View style={styles.flexOne}>
              <Text style={styles.cardTitle}>Select Sector</Text>
              <Text style={styles.cardSubtitle}>
                The image will be stored in this backend sector. If the AI model
                returns an incomplete result, backend image analysis will be
                used as fallback.
              </Text>
            </View>

            {loadingSectors && <ActivityIndicator color={COLORS.primary} />}
          </View>

          {sectors.length === 0 && !loadingSectors ? (
            <Text style={styles.emptyText}>
              No sectors found. Please create a sector from the web or farm
              page.
            </Text>
          ) : (
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.sectorList}
            >
              {sectors.map((sector) => {
                const id = getSectorId(sector);
                const active = selectedSectorId === id;

                return (
                  <TouchableOpacity
                    key={id}
                    style={[
                      styles.sectorButton,
                      active && styles.sectorButtonActive,
                    ]}
                    onPress={() => setSelectedSectorId(id)}
                  >
                    <Text
                      style={[
                        styles.sectorButtonText,
                        active && styles.sectorButtonTextActive,
                      ]}
                    >
                      {sector?.name || "Unnamed Sector"}
                    </Text>
                    <Text
                      style={[
                        styles.sectorCrop,
                        active && styles.sectorCropActive,
                      ]}
                    >
                      {sector?.cropType || sector?.crop || "Unknown Crop"}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </ScrollView>
          )}
        </View>
      )}

      {needsSensors && (
        <View style={styles.card}>
          <View style={styles.rowBetween}>
            <View style={styles.flexOne}>
              <Text style={styles.cardTitle}>Hardware Readings</Text>
              <Text style={styles.cardSubtitle}>
                Soil Moisture can be 0 and will still be treated as a valid
                hardware reading.
              </Text>
            </View>

            <TouchableOpacity
              style={styles.smallButton}
              onPress={loadLatestReading}
              disabled={loadingReadings}
            >
              {loadingReadings ? (
                <ActivityIndicator color={COLORS.white} />
              ) : (
                <Text style={styles.smallButtonText}>Refresh</Text>
              )}
            </TouchableOpacity>
          </View>

          {latestReading ? (
            <>
              <View style={styles.readingGrid}>
                <ReadingItem
                  label="Temperature"
                  value={`${latestReading.temperature ?? 0}°C`}
                />
                <ReadingItem
                  label="Humidity"
                  value={`${latestReading.humidity ?? 0}%`}
                />
                <ReadingItem
                  label="Soil Moisture"
                  value={`${latestReading.soilMoisture ?? 0}%`}
                />
                <ReadingItem
                  label="Soil Temp"
                  value={`${latestReading.soilTemp ?? 0}°C`}
                />
                <ReadingItem
                  label="Light"
                  value={String(latestReading.light)}
                />
                <ReadingItem
                  label="Crop"
                  value={String(latestReading.cropType || "Unknown")}
                />
              </View>

              <Text style={styles.metaText}>
                Sector: {latestReading.sectorName || "Unknown"} · Device:{" "}
                {latestReading.deviceSerial || "Unknown"}
              </Text>

              <SensorBackendAnalysis reading={latestReading} />
            </>
          ) : (
            <Text style={styles.emptyText}>
              No sensor readings available. Please refresh after hardware
              upload.
            </Text>
          )}
        </View>
      )}

      {needsImage && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Plant Image</Text>
          <Text style={styles.cardSubtitle}>
            Take a clear photo of the plant leaf or choose one from gallery.
          </Text>

          {imageUri ? (
            <Image source={{ uri: imageUri }} style={styles.previewImage} />
          ) : (
            <View style={styles.imagePlaceholder}>
              <Text style={styles.imagePlaceholderText}>No image selected</Text>
            </View>
          )}

          <View style={styles.actionRow}>
            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={openCamera}
            >
              <Text style={styles.secondaryButtonText}>Open Camera</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={pickImageFromGallery}
            >
              <Text style={styles.secondaryButtonText}>Gallery</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      <TouchableOpacity
        style={[
          styles.analyzeButton,
          (analyzing || uploadingImage) && styles.disabledButton,
        ]}
        onPress={handleAnalyze}
        disabled={analyzing || uploadingImage}
      >
        {analyzing || uploadingImage ? (
          <ActivityIndicator color={COLORS.white} />
        ) : (
          <Text style={styles.analyzeButtonText}>Analyze Plant</Text>
        )}
      </TouchableOpacity>

      {aiUnavailable && (
        <View style={styles.warningCard}>
          <Text style={styles.warningTitle}>AI Unavailable</Text>
          <Text style={styles.warningText}>
            {errorMessage ||
              "The service is not available now. You can retry or show presentation demo."}
          </Text>

          <View style={styles.actionRow}>
            <TouchableOpacity
              style={styles.retryButton}
              onPress={handleAnalyze}
              disabled={analyzing}
            >
              <Text style={styles.retryButtonText}>Try Again</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.demoButton}
              onPress={handleShowDemoResult}
            >
              <Text style={styles.demoButtonText}>Show Presentation Demo</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {result && (
        <View style={styles.resultCard}>
          {result.presentationDemo && (
            <View style={styles.demoBadge}>
              <Text style={styles.demoBadgeText}>Presentation Demo Result</Text>
            </View>
          )}

          {result.analysisSource && (
            <View style={styles.sourceBadge}>
              <Text style={styles.sourceBadgeText}>
                Source: {result.analysisSource}
              </Text>
            </View>
          )}

          <Text style={styles.resultTitle}>Diagnosis Result</Text>

          <View
            style={[
              styles.statusBadge,
              getStatusStyle(result.finalStatus || result.final_status),
            ]}
          >
            <Text style={styles.statusText}>
              {result.finalStatus || result.final_status || "Unknown"}
            </Text>
          </View>

          {result.confidence !== null && result.confidence !== undefined && (
            <Text style={styles.confidenceText}>
              Confidence: {formatConfidence(result.confidence)}
            </Text>
          )}

          <Text style={styles.resultSectionTitle}>Diagnosis</Text>
          <Text style={styles.resultText}>
            {result.diagnosisText || result.diagnosis || "No diagnosis text."}
          </Text>

          <Text style={styles.resultSectionTitle}>Recommendations</Text>
          {Array.isArray(result.recommendations) &&
          result.recommendations.length > 0 ? (
            result.recommendations.map((item: any, index: number) => (
              <Text key={`${item}-${index}`} style={styles.bulletText}>
                • {String(item)}
              </Text>
            ))
          ) : (
            <Text style={styles.resultText}>No recommendations.</Text>
          )}

          {Array.isArray(result.actions) && result.actions.length > 0 && (
            <>
              <Text style={styles.resultSectionTitle}>Actions</Text>
              {result.actions.map((action: any, index: number) => (
                <Text
                  key={`${action?.code || index}`}
                  style={styles.bulletText}
                >
                  • {action?.title || action?.code || String(action)}
                </Text>
              ))}
            </>
          )}

          <TouchableOpacity
            style={styles.historyButton}
            onPress={() => router.push("/diagnoses" as any)}
          >
            <Text style={styles.historyButtonText}>Open My Diagnoses</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
}

function ModeButton({
  title,
  subtitle,
  active,
  onPress,
}: {
  title: string;
  subtitle: string;
  active: boolean;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity
      style={[styles.modeButton, active && styles.modeButtonActive]}
      onPress={onPress}
    >
      <Text style={[styles.modeTitle, active && styles.modeTitleActive]}>
        {title}
      </Text>
      <Text style={[styles.modeSubtitle, active && styles.modeSubtitleActive]}>
        {subtitle}
      </Text>
    </TouchableOpacity>
  );
}

function ReadingItem({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.readingItem}>
      <Text style={styles.readingLabel}>{label}</Text>
      <Text style={styles.readingValue}>{value}</Text>
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

function extractArray(response: any) {
  if (Array.isArray(response)) return response;
  if (Array.isArray(response?.data)) return response.data;
  if (Array.isArray(response?.sectors)) return response.sectors;
  if (Array.isArray(response?.data?.sectors)) return response.data.sectors;
  if (Array.isArray(response?.data?.data)) return response.data.data;
  return [];
}

function getSectorId(sector: any) {
  return String(sector?._id || sector?.id || sector?.sectorId || "");
}

function normalizeLightForAI(value: any) {
  const light = String(value || "").toLowerCase();

  if (light === "low") return "Low";
  if (light === "medium") return "Medium";
  if (light === "sufficient") return "Sufficient";
  if (light === "high") return "Sufficient";
  if (light === "strong") return "Sufficient";

  return "Sufficient";
}

function isExplicitNoPlantResponse(response: any) {
  const data = response?.data || response;

  return (
    data?.is_plant === false ||
    data?.plant_detected === false ||
    data?.isPlant === false ||
    data?.plantDetected === false
  );
}

function getPlantValidationMessage(response: any) {
  const data = response?.data || response;

  return (
    data?.message ||
    data?.error ||
    "No plant detected. Please upload a clear plant leaf image."
  );
}

function hasUsefulAIResult(response: any) {
  if (!response) return false;

  const data = response?.data || response?.result || response;

  const status =
    data?.final_status ||
    data?.finalStatus ||
    data?.status ||
    data?.image_status ||
    data?.sensor_status ||
    data?.prediction ||
    data?.result?.final_status ||
    data?.result?.status;

  const diagnosis =
    data?.diagnosisText ||
    data?.diagnosis ||
    data?.explanation ||
    data?.disease_name ||
    data?.diseaseName ||
    data?.visual_problem ||
    data?.message ||
    data?.result?.diagnosis;

  const recommendations =
    data?.recommendations ||
    data?.actions ||
    data?.suggestions ||
    data?.result?.recommendations;

  if (status && String(status).toLowerCase() !== "unknown") {
    return true;
  }

  if (diagnosis) {
    return true;
  }

  if (Array.isArray(recommendations) && recommendations.length > 0) {
    return true;
  }

  return false;
}

function normalizeAnalysisResult(
  response: any,
  extra: {
    mode: DiagnosisMode;
    sector: any;
    sensorPayload: any;
    imageUri: string | null;
  },
) {
  const data = response?.data || response?.result || response;

  const analysisResult =
    data?.analysisResult ||
    data?.data?.analysisResult ||
    response?.analysisResult ||
    response?.data?.analysisResult ||
    {};

  const isBackendUploadFallback =
    !!analysisResult &&
    Object.keys(analysisResult).length > 0 &&
    !!(data?.imageUrl || response?.data?.imageUrl);

  const finalStatus =
    data?.final_status ||
    data?.finalStatus ||
    data?.status ||
    data?.image_status ||
    data?.sensor_status ||
    data?.prediction ||
    data?.result?.final_status ||
    data?.result?.status ||
    analysisResult?.final_status ||
    analysisResult?.finalStatus ||
    analysisResult?.status ||
    "Unknown";

  const diagnosisObject = data?.diagnosis || data?.result?.diagnosis || {};

  const diagnosisText =
    data?.diagnosisText ||
    data?.explanation ||
    data?.message ||
    data?.recommendation ||
    data?.general_recommendation ||
    data?.disease_name ||
    data?.diseaseName ||
    data?.visual_problem ||
    diagnosisObject?.explanation ||
    diagnosisObject?.primary_issue ||
    diagnosisObject?.secondary_issue ||
    diagnosisObject?.visual_problem ||
    analysisResult?.diseaseName ||
    analysisResult?.disease_name ||
    analysisResult?.note ||
    "AI diagnosis completed successfully.";

  const recommendations =
    data?.recommendations ||
    data?.actions_recommended ||
    data?.suggestions ||
    data?.result?.recommendations ||
    analysisResult?.recommendations ||
    [];

  const actions =
    data?.actions ||
    data?.required_actions ||
    data?.result?.actions ||
    analysisResult?.treatmentPlan ||
    analysisResult?.actions ||
    [];

  return {
    id: `${Date.now()}`,
    createdAt: new Date().toISOString(),
    mode: extra.mode,
    sectorId: getSectorId(extra.sector),
    sectorName: extra.sector?.name || "Unknown Sector",
    cropType:
      extra.sector?.cropType ||
      extra.sector?.crop ||
      extra.sensorPayload?.cropType ||
      "Tomato",
    imageUri:
      extra.imageUri || data?.imageUrl || response?.data?.imageUrl || null,
    sensorReadings: extra.sensorPayload,
    finalStatus,
    final_status: finalStatus,
    confidence:
      data?.final_confidence ||
      data?.confidence ||
      data?.image_confidence ||
      data?.sensor_confidence ||
      analysisResult?.confidence ||
      null,
    diagnosisText,
    diagnosis: diagnosisText,
    recommendations: Array.isArray(recommendations)
      ? recommendations
      : [String(recommendations)],
    actions: Array.isArray(actions) ? actions : [],
    imageAnalysis:
      data?.image_analysis ||
      data?.imageAnalysis ||
      data?.ratios ||
      analysisResult?.ratios ||
      null,
    analysisSource: isBackendUploadFallback
      ? "Backend Upload Fallback"
      : "AI Model",
    raw: data,
  };
}

function buildPresentationDemoResult(extra: {
  mode: DiagnosisMode;
  sector: any;
  sensorPayload: any;
  imageUri: string | null;
}) {
  return {
    id: `${Date.now()}`,
    createdAt: new Date().toISOString(),
    presentationDemo: true,
    mode: extra.mode,
    sectorId: getSectorId(extra.sector),
    sectorName: extra.sector?.name || "Demo Sector",
    cropType:
      extra.sector?.cropType ||
      extra.sector?.crop ||
      extra.sensorPayload?.cropType ||
      "Mint",
    imageUri: extra.imageUri,
    sensorReadings: extra.sensorPayload,
    finalStatus: "Moderate Stress",
    final_status: "Moderate Stress",
    confidence: 0.84,
    diagnosisText:
      "النبات يعاني من إجهاد متوسط ويحتاج متابعة. يجب مراجعة الري ودرجة الحرارة والإضاءة.",
    diagnosis:
      "النبات يعاني من إجهاد متوسط ويحتاج متابعة. يجب مراجعة الري ودرجة الحرارة والإضاءة.",
    recommendations: [
      "تابع حالة النبات خلال 24 إلى 48 ساعة.",
      "راجع رطوبة التربة ودرجة حرارة الجو.",
      "تأكد من وضوح صورة الورقة عند التحليل.",
    ],
    actions: [
      {
        code: "MONITOR",
        title: "متابعة حالة النبات",
      },
    ],
    analysisSource: "Presentation Demo",
  };
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
  card: {
    backgroundColor: COLORS.white,
    borderRadius: 20,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  flexOne: {
    flex: 1,
    paddingRight: 10,
  },
  cardTitle: {
    color: COLORS.dark,
    fontSize: 18,
    fontWeight: "900",
  },
  cardSubtitle: {
    color: COLORS.muted,
    fontSize: 13,
    lineHeight: 19,
    marginTop: 4,
  },
  modeGrid: {
    flexDirection: "row",
    gap: 8,
    marginTop: 14,
  },
  modeButton: {
    flex: 1,
    padding: 12,
    borderRadius: 16,
    backgroundColor: COLORS.soft,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  modeButtonActive: {
    backgroundColor: COLORS.dark,
    borderColor: COLORS.dark,
  },
  modeTitle: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "900",
  },
  modeTitleActive: {
    color: COLORS.white,
  },
  modeSubtitle: {
    color: COLORS.muted,
    fontSize: 11,
    marginTop: 4,
  },
  modeSubtitleActive: {
    color: COLORS.mint,
  },
  rowBetween: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  sectorList: {
    gap: 10,
    paddingTop: 14,
    paddingRight: 6,
  },
  sectorButton: {
    minWidth: 150,
    padding: 14,
    borderRadius: 16,
    backgroundColor: COLORS.soft,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  sectorButtonActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  sectorButtonText: {
    color: COLORS.text,
    fontSize: 14,
    fontWeight: "900",
  },
  sectorButtonTextActive: {
    color: COLORS.white,
  },
  sectorCrop: {
    color: COLORS.muted,
    fontSize: 12,
    marginTop: 5,
  },
  sectorCropActive: {
    color: COLORS.mint,
  },
  emptyText: {
    color: COLORS.muted,
    fontSize: 13,
    marginTop: 12,
    lineHeight: 20,
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
  readingGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 14,
  },
  readingItem: {
    width: "47%",
    backgroundColor: COLORS.soft,
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  readingLabel: {
    color: COLORS.muted,
    fontSize: 12,
    fontWeight: "700",
  },
  readingValue: {
    color: COLORS.dark,
    fontSize: 18,
    fontWeight: "900",
    marginTop: 5,
  },
  metaText: {
    color: COLORS.muted,
    fontSize: 12,
    marginTop: 10,
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
    fontSize: 15,
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
  previewImage: {
    width: "100%",
    height: 230,
    borderRadius: 18,
    marginTop: 14,
    backgroundColor: COLORS.mint,
  },
  imagePlaceholder: {
    height: 190,
    borderRadius: 18,
    backgroundColor: COLORS.soft,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 14,
  },
  imagePlaceholderText: {
    color: COLORS.muted,
    fontSize: 14,
    fontWeight: "700",
  },
  actionRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: 14,
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: COLORS.mint,
    borderRadius: 14,
    paddingVertical: 13,
    alignItems: "center",
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  secondaryButtonText: {
    color: COLORS.primary,
    fontSize: 14,
    fontWeight: "900",
  },
  analyzeButton: {
    backgroundColor: COLORS.primary,
    borderRadius: 18,
    paddingVertical: 16,
    alignItems: "center",
    marginBottom: 14,
  },
  disabledButton: {
    opacity: 0.65,
  },
  analyzeButtonText: {
    color: COLORS.white,
    fontSize: 16,
    fontWeight: "900",
  },
  warningCard: {
    backgroundColor: "#FFF7ED",
    borderRadius: 20,
    padding: 16,
    borderWidth: 1,
    borderColor: "#FDBA74",
    marginBottom: 14,
  },
  warningTitle: {
    color: "#9A3412",
    fontSize: 18,
    fontWeight: "900",
  },
  warningText: {
    color: "#9A3412",
    fontSize: 13,
    lineHeight: 20,
    marginTop: 8,
  },
  retryButton: {
    flex: 1,
    backgroundColor: COLORS.warning,
    borderRadius: 14,
    paddingVertical: 13,
    alignItems: "center",
  },
  retryButtonText: {
    color: COLORS.white,
    fontSize: 13,
    fontWeight: "900",
  },
  demoButton: {
    flex: 1,
    backgroundColor: COLORS.white,
    borderRadius: 14,
    paddingVertical: 13,
    alignItems: "center",
    borderWidth: 1,
    borderColor: COLORS.warning,
  },
  demoButtonText: {
    color: COLORS.warning,
    fontSize: 13,
    fontWeight: "900",
  },
  resultCard: {
    backgroundColor: COLORS.white,
    borderRadius: 22,
    padding: 18,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  demoBadge: {
    alignSelf: "flex-start",
    backgroundColor: "#FEF3C7",
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 999,
    marginBottom: 10,
  },
  demoBadgeText: {
    color: "#92400E",
    fontSize: 11,
    fontWeight: "900",
  },
  sourceBadge: {
    alignSelf: "flex-start",
    backgroundColor: COLORS.mint,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 999,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  sourceBadgeText: {
    color: COLORS.primary,
    fontSize: 11,
    fontWeight: "900",
  },
  resultTitle: {
    color: COLORS.dark,
    fontSize: 21,
    fontWeight: "900",
  },
  statusBadge: {
    alignSelf: "flex-start",
    borderRadius: 999,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderWidth: 1,
    marginTop: 12,
  },
  statusText: {
    color: COLORS.dark,
    fontSize: 14,
    fontWeight: "900",
  },
  confidenceText: {
    color: COLORS.muted,
    fontSize: 13,
    marginTop: 8,
    fontWeight: "700",
  },
  resultSectionTitle: {
    color: COLORS.dark,
    fontSize: 15,
    fontWeight: "900",
    marginTop: 18,
    marginBottom: 6,
  },
  resultText: {
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
  historyButton: {
    backgroundColor: COLORS.dark,
    borderRadius: 14,
    paddingVertical: 13,
    alignItems: "center",
    marginTop: 18,
  },
  historyButtonText: {
    color: COLORS.white,
    fontSize: 14,
    fontWeight: "900",
  },
});
