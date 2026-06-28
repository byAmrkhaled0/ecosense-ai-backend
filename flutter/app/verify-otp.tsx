import { router, useLocalSearchParams } from "expo-router";
import { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { verifyOtp } from "../services/api";

export default function VerifyOtpScreen() {
  const params = useLocalSearchParams();

  const email = getParam(params.email);
  const registrationToken = getParam(params.registrationToken);

  const [code, setCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  async function handleVerifyOtp() {
    const cleanCode = code.trim();

    setErrorMessage("");

    if (!cleanCode) {
      setErrorMessage("Please enter the OTP code.");
      return;
    }

    if (!registrationToken) {
      setErrorMessage(
        "Registration token is missing. Please register again from the beginning.",
      );
      return;
    }

    try {
      setLoading(true);

      await verifyOtp(cleanCode, registrationToken);

      Alert.alert(
        "Account Activated",
        "Your account has been activated successfully.",
      );

      router.replace("/(tabs)" as any);
    } catch (error: any) {
      const message =
        error?.message || "OTP verification failed. Please try again.";

      setErrorMessage(message);
      Alert.alert("Verification Failed", message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.keyboard}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
        <Pressable
          style={styles.backButton}
          onPress={() => router.replace("/register" as any)}
        >
          <Text style={styles.backText}>‹ Back</Text>
        </Pressable>

        <View style={styles.logoBox}>
          <Text style={styles.logoText}>EC</Text>
        </View>

        <Text style={styles.kicker}>VERIFY OTP</Text>
        <Text style={styles.title}>Activate your account</Text>

        <Text style={styles.subtitle}>
          Enter the OTP code sent to your email to activate your account and
          access the dashboard.
        </Text>

        <View style={styles.infoCard}>
          <Text style={styles.infoLabel}>Email</Text>
          <Text style={styles.infoValue}>{email || "No email provided"}</Text>
        </View>

        <View style={styles.formCard}>
          <Text style={styles.label}>OTP Code</Text>

          <TextInput
            style={styles.otpInput}
            placeholder="123456"
            placeholderTextColor="#9AABA1"
            value={code}
            onChangeText={setCode}
            keyboardType="number-pad"
            maxLength={6}
            textAlign="center"
          />

          {errorMessage ? (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{errorMessage}</Text>
            </View>
          ) : null}

          <Pressable
            style={[styles.verifyButton, loading && styles.disabledButton]}
            onPress={handleVerifyOtp}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#FFFFFF" />
            ) : (
              <Text style={styles.verifyButtonText}>Verify Account</Text>
            )}
          </Pressable>
        </View>

        <Pressable
          style={styles.loginButton}
          onPress={() => router.replace("/login" as any)}
        >
          <Text style={styles.loginButtonText}>Already verified? Login</Text>
        </Pressable>

        <Text style={styles.footer}>
          ECOSENSE AI • Secure Account Verification
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

function getParam(value: any) {
  if (Array.isArray(value)) return value[0] || "";
  if (typeof value === "string") return value;
  return "";
}

const styles = StyleSheet.create({
  keyboard: {
    flex: 1,
  },
  screen: {
    flex: 1,
    backgroundColor: "#F3F8F1",
  },
  content: {
    padding: 24,
    paddingTop: 54,
    paddingBottom: 40,
  },
  backButton: {
    alignSelf: "flex-start",
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#DDE9E2",
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 999,
    marginBottom: 22,
  },
  backText: {
    color: "#118A5B",
    fontSize: 14,
    fontWeight: "900",
  },
  logoBox: {
    width: 72,
    height: 72,
    borderRadius: 26,
    backgroundColor: "#0B2A22",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 18,
  },
  logoText: {
    color: "#8BE0B3",
    fontSize: 25,
    fontWeight: "900",
    letterSpacing: 1,
  },
  kicker: {
    color: "#118A5B",
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 2,
    marginBottom: 10,
  },
  title: {
    color: "#082A1F",
    fontSize: 36,
    lineHeight: 42,
    fontWeight: "900",
  },
  subtitle: {
    color: "#65786D",
    fontSize: 15,
    lineHeight: 24,
    marginTop: 14,
    marginBottom: 22,
  },
  infoCard: {
    backgroundColor: "#E6F7EE",
    borderRadius: 22,
    padding: 16,
    borderWidth: 1,
    borderColor: "#C7EBD8",
    marginBottom: 16,
  },
  infoLabel: {
    color: "#118A5B",
    fontSize: 12,
    fontWeight: "900",
    marginBottom: 6,
  },
  infoValue: {
    color: "#0B2A22",
    fontSize: 15,
    fontWeight: "900",
  },
  formCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 30,
    padding: 20,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  label: {
    color: "#0B2A22",
    fontSize: 14,
    fontWeight: "900",
    marginBottom: 10,
  },
  otpInput: {
    backgroundColor: "#F8FBF7",
    borderWidth: 1,
    borderColor: "#DDE9E2",
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 18,
    color: "#0B2A22",
    fontSize: 28,
    fontWeight: "900",
    letterSpacing: 6,
  },
  errorBox: {
    backgroundColor: "#FEF3F2",
    borderWidth: 1,
    borderColor: "#FDA29B",
    borderRadius: 16,
    padding: 12,
    marginTop: 14,
  },
  errorText: {
    color: "#B42318",
    fontSize: 13,
    fontWeight: "800",
    lineHeight: 20,
  },
  verifyButton: {
    backgroundColor: "#118A5B",
    borderRadius: 20,
    paddingVertical: 17,
    alignItems: "center",
    marginTop: 20,
  },
  disabledButton: {
    opacity: 0.7,
  },
  verifyButtonText: {
    color: "#FFFFFF",
    fontSize: 17,
    fontWeight: "900",
  },
  loginButton: {
    backgroundColor: "#FFFFFF",
    borderRadius: 20,
    paddingVertical: 16,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#DDE9E2",
    marginTop: 18,
  },
  loginButtonText: {
    color: "#118A5B",
    fontSize: 15,
    fontWeight: "900",
  },
  footer: {
    color: "#7A8B82",
    textAlign: "center",
    marginTop: 24,
    fontSize: 12,
    fontWeight: "800",
  },
});
