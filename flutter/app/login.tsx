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
import { loginUser } from "../services/api";

export default function LoginScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  async function handleLogin() {
    const cleanEmail = email.trim().toLowerCase();
    const cleanPassword = password.trim();

    setErrorMessage("");

    if (!cleanEmail || !cleanPassword) {
      setErrorMessage("Please enter your email and password.");
      return;
    }

    try {
      setLoading(true);

      await loginUser(cleanEmail, cleanPassword);

      router.replace("/(tabs)" as any);
    } catch (error: any) {
      const message =
        error?.message || "Login failed. Please check your account details.";

      setErrorMessage(message);

      Alert.alert("Login Failed", message);
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
          onPress={() => router.replace("/landing" as any)}
        >
          <Text style={styles.backText}>‹ Back</Text>
        </Pressable>

        <View style={styles.logoBox}>
          <Text style={styles.logoText}>EC</Text>
        </View>

        <Text style={styles.kicker}>WELCOME BACK</Text>
        <Text style={styles.title}>Login to Ecosense AI</Text>

        <Text style={styles.subtitle}>
          Access your smart farm dashboard, AI diagnosis history, sensors,
          reports and alerts.
        </Text>

        <View style={styles.formCard}>
          <Text style={styles.label}>Email Address</Text>
          <TextInput
            style={styles.input}
            placeholder="example@email.com"
            placeholderTextColor="#9AABA1"
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            keyboardType="email-address"
          />

          <Text style={styles.label}>Password</Text>

          <View style={styles.passwordRow}>
            <TextInput
              style={styles.passwordInput}
              placeholder="Enter your password"
              placeholderTextColor="#9AABA1"
              value={password}
              onChangeText={setPassword}
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

          {errorMessage ? (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>{errorMessage}</Text>
            </View>
          ) : null}

          <Pressable
            style={[styles.loginButton, loading && styles.disabledButton]}
            onPress={handleLogin}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#FFFFFF" />
            ) : (
              <Text style={styles.loginButtonText}>Login</Text>
            )}
          </Pressable>

          <Pressable
            style={styles.forgotButton}
            onPress={() =>
              Alert.alert(
                "Forgot Password",
                "Password reset endpoint is not connected yet.",
              )
            }
          >
            <Text style={styles.forgotText}>Forgot password?</Text>
          </Pressable>
        </View>

        <View style={styles.createCard}>
          <Text style={styles.createText}>No account yet?</Text>

          <Pressable
            style={styles.createButton}
            onPress={() => router.push("/register" as any)}
          >
            <Text style={styles.createButtonText}>Create Account</Text>
          </Pressable>
        </View>

        <Text style={styles.footer}>
          ECOSENSE AI • Smart Agriculture • AI + IoT
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
  loginButton: {
    backgroundColor: "#118A5B",
    borderRadius: 20,
    paddingVertical: 17,
    alignItems: "center",
    marginTop: 20,
  },
  disabledButton: {
    opacity: 0.7,
  },
  loginButtonText: {
    color: "#FFFFFF",
    fontSize: 17,
    fontWeight: "900",
  },
  forgotButton: {
    alignItems: "center",
    marginTop: 16,
  },
  forgotText: {
    color: "#118A5B",
    fontSize: 14,
    fontWeight: "900",
  },
  createCard: {
    backgroundColor: "#E6F7EE",
    borderRadius: 24,
    padding: 18,
    marginTop: 18,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#C7EBD8",
  },
  createText: {
    color: "#65786D",
    fontSize: 14,
    fontWeight: "800",
  },
  createButton: {
    backgroundColor: "#FFFFFF",
    borderRadius: 18,
    paddingVertical: 14,
    paddingHorizontal: 22,
    marginTop: 12,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  createButtonText: {
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
