import { router } from "expo-router";
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
import { registerUser, verifyOtp } from "../services/api";

export default function RegisterScreen() {
  const [step, setStep] = useState(0);

  const [form, setForm] = useState({
    email: "",
    password: "",
    firstName: "",
    lastName: "",
    phoneNumber: "",
    address: "",
  });

  const [confirmPassword, setConfirmPassword] = useState("");
  const [otp, setOtp] = useState("");
  const [regToken, setRegToken] = useState("");

  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  function updateField(key: keyof typeof form, value: string) {
    setForm((previous) => ({
      ...previous,
      [key]: value,
    }));
  }

  async function handleRegister() {
    if (loading) return;

    const cleanEmail = form.email.trim().toLowerCase();
    const cleanPassword = form.password.trim();
    const cleanConfirmPassword = confirmPassword.trim();
    const cleanFirstName = form.firstName.trim();
    const cleanLastName = form.lastName.trim();
    const cleanPhoneNumber = form.phoneNumber.trim();
    const cleanAddress = form.address.trim();

    setErrorMessage("");

    if (
      !cleanEmail ||
      !cleanPassword ||
      !cleanConfirmPassword ||
      !cleanFirstName ||
      !cleanLastName ||
      !cleanPhoneNumber ||
      !cleanAddress
    ) {
      setErrorMessage("Please complete all required fields.");
      return;
    }

    if (cleanPassword.length < 6) {
      setErrorMessage("Password must be at least 6 characters.");
      return;
    }

    if (cleanPassword !== cleanConfirmPassword) {
      setErrorMessage("Passwords do not match.");
      return;
    }

    try {
      setLoading(true);

      const response = await registerUser({
        email: cleanEmail,
        password: cleanPassword,
        firstName: cleanFirstName,
        lastName: cleanLastName,
        phoneNumber: cleanPhoneNumber,
        address: cleanAddress,
      });

      if (!response?.registrationToken) {
        throw new Error("registrationToken was not returned from backend.");
      }

      setRegToken(response.registrationToken);
      setStep(1);

      Alert.alert(
        "Verification Code Sent",
        "Please check your email and enter the OTP code.",
      );
    } catch (error: any) {
      const message =
        error?.message ||
        "Registration failed. Please check your data or use another email.";

      setErrorMessage(message);
      Alert.alert("Registration Failed", message);
    } finally {
      setLoading(false);
    }
  }

  async function handleVerify() {
    if (loading) return;

    const cleanOtp = otp.trim();

    setErrorMessage("");

    if (!cleanOtp) {
      setErrorMessage("Please enter the OTP code.");
      return;
    }

    if (!regToken) {
      setErrorMessage("Registration token is missing. Please register again.");
      return;
    }

    try {
      setLoading(true);

      await verifyOtp(cleanOtp, regToken);

      Alert.alert(
        "Account Verified",
        "Account verified successfully. Welcome to Ecosense AI.",
      );

      router.replace("/(tabs)" as any);
    } catch (error: any) {
      const message = error?.message || "OTP code is incorrect or expired.";

      setErrorMessage(message);
      Alert.alert("Verification Failed", message);
    } finally {
      setLoading(false);
    }
  }

  function handleBack() {
    if (step === 1) {
      setStep(0);
      setOtp("");
      setErrorMessage("");
      return;
    }

    router.replace("/landing" as any);
  }

  return (
    <KeyboardAvoidingView
      style={styles.keyboard}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
        <Pressable style={styles.backButton} onPress={handleBack}>
          <Text style={styles.backText}>‹ Back</Text>
        </Pressable>

        <View style={styles.logoBox}>
          <Text style={styles.logoText}>EC</Text>
        </View>

        <Text style={styles.kicker}>
          {step === 0 ? "CREATE ACCOUNT" : "VERIFY OTP"}
        </Text>

        <Text style={styles.title}>
          {step === 0 ? "Start with Ecosense AI" : "Activate your account"}
        </Text>

        <Text style={styles.subtitle}>
          {step === 0
            ? "Create your account, then verify the OTP code sent to your email."
            : "Enter the OTP code sent to your email to activate your account."}
        </Text>

        {step === 0 ? (
          <View style={styles.formCard}>
            <Text style={styles.label}>First Name *</Text>
            <TextInput
              style={styles.input}
              placeholder="Mahmoud"
              placeholderTextColor="#9AABA1"
              value={form.firstName}
              onChangeText={(value) => updateField("firstName", value)}
            />

            <Text style={styles.label}>Last Name *</Text>
            <TextInput
              style={styles.input}
              placeholder="Mansour"
              placeholderTextColor="#9AABA1"
              value={form.lastName}
              onChangeText={(value) => updateField("lastName", value)}
            />

            <Text style={styles.label}>Email *</Text>
            <TextInput
              style={styles.input}
              placeholder="user@example.com"
              placeholderTextColor="#9AABA1"
              value={form.email}
              onChangeText={(value) => updateField("email", value)}
              autoCapitalize="none"
              keyboardType="email-address"
            />

            <Text style={styles.label}>Phone Number *</Text>
            <TextInput
              style={styles.input}
              placeholder="0123456789"
              placeholderTextColor="#9AABA1"
              value={form.phoneNumber}
              onChangeText={(value) => updateField("phoneNumber", value)}
              keyboardType="phone-pad"
            />

            <Text style={styles.label}>Address *</Text>
            <TextInput
              style={styles.input}
              placeholder="Cairo"
              placeholderTextColor="#9AABA1"
              value={form.address}
              onChangeText={(value) => updateField("address", value)}
            />

            <Text style={styles.label}>Password *</Text>
            <View style={styles.passwordRow}>
              <TextInput
                style={styles.passwordInput}
                placeholder="securepassword123"
                placeholderTextColor="#9AABA1"
                value={form.password}
                onChangeText={(value) => updateField("password", value)}
                secureTextEntry={!showPassword}
                autoCapitalize="none"
              />

              <Pressable
                style={styles.showButton}
                onPress={() => setShowPassword((value) => !value)}
              >
                <Text style={styles.showText}>
                  {showPassword ? "Hide" : "Show"}
                </Text>
              </Pressable>
            </View>

            <Text style={styles.label}>Confirm Password *</Text>
            <TextInput
              style={styles.input}
              placeholder="Confirm your password"
              placeholderTextColor="#9AABA1"
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              secureTextEntry={!showPassword}
              autoCapitalize="none"
            />

            {errorMessage ? (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>{errorMessage}</Text>
              </View>
            ) : null}

            <Pressable
              style={[styles.primaryButton, loading && styles.disabledButton]}
              onPress={handleRegister}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color="#FFFFFF" />
              ) : (
                <Text style={styles.primaryButtonText}>Create Account</Text>
              )}
            </Pressable>
          </View>
        ) : (
          <>
            <View style={styles.infoCard}>
              <Text style={styles.infoLabel}>Email</Text>
              <Text style={styles.infoValue}>
                {form.email.trim().toLowerCase()}
              </Text>
            </View>

            <View style={styles.formCard}>
              <Text style={styles.label}>OTP Code</Text>

              <TextInput
                style={styles.otpInput}
                placeholder="123456"
                placeholderTextColor="#9AABA1"
                value={otp}
                onChangeText={setOtp}
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
                style={[styles.primaryButton, loading && styles.disabledButton]}
                onPress={handleVerify}
                disabled={loading}
              >
                {loading ? (
                  <ActivityIndicator color="#FFFFFF" />
                ) : (
                  <Text style={styles.primaryButtonText}>Verify Account</Text>
                )}
              </Pressable>
            </View>
          </>
        )}

        <View style={styles.loginCard}>
          <Text style={styles.loginText}>
            {step === 0 ? "Already have an account?" : "Already verified?"}
          </Text>

          <Pressable
            style={styles.loginButton}
            onPress={() => router.replace("/login" as any)}
          >
            <Text style={styles.loginButtonText}>Login</Text>
          </Pressable>
        </View>

        <Text style={styles.footer}>
          ECOSENSE AI • Smart Plant Health System
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
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
    marginBottom: 8,
    marginTop: 12,
  },
  input: {
    backgroundColor: "#F8FBF7",
    borderWidth: 1,
    borderColor: "#DDE9E2",
    borderRadius: 18,
    paddingHorizontal: 16,
    paddingVertical: 15,
    color: "#0B2A22",
    fontSize: 15,
    fontWeight: "700",
  },
  passwordRow: {
    backgroundColor: "#F8FBF7",
    borderWidth: 1,
    borderColor: "#DDE9E2",
    borderRadius: 18,
    flexDirection: "row",
    alignItems: "center",
  },
  passwordInput: {
    flex: 1,
    paddingHorizontal: 16,
    paddingVertical: 15,
    color: "#0B2A22",
    fontSize: 15,
    fontWeight: "700",
  },
  showButton: {
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  showText: {
    color: "#118A5B",
    fontSize: 13,
    fontWeight: "900",
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
  primaryButton: {
    backgroundColor: "#118A5B",
    borderRadius: 20,
    paddingVertical: 17,
    alignItems: "center",
    marginTop: 20,
  },
  disabledButton: {
    opacity: 0.7,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 17,
    fontWeight: "900",
  },
  loginCard: {
    backgroundColor: "#E6F7EE",
    borderRadius: 24,
    padding: 18,
    marginTop: 18,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#C7EBD8",
  },
  loginText: {
    color: "#65786D",
    fontSize: 14,
    fontWeight: "800",
  },
  loginButton: {
    backgroundColor: "#FFFFFF",
    borderRadius: 18,
    paddingVertical: 14,
    paddingHorizontal: 22,
    marginTop: 12,
    borderWidth: 1,
    borderColor: "#DDE9E2",
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
