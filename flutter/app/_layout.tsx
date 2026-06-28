import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import {
  AppSettingsProvider,
  useAppSettings,
} from "../contexts/AppSettingsContext";

function RootStack() {
  const { themeMode } = useAppSettings();

  return (
    <>
      <StatusBar style={themeMode === "dark" ? "light" : "dark"} />

      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="index" />
        <Stack.Screen name="splash" />
        <Stack.Screen name="landing" />
        <Stack.Screen name="login" />
        <Stack.Screen name="register" />
        <Stack.Screen name="verify-otp" />
        <Stack.Screen name="(tabs)" />
        <Stack.Screen name="diagnoses" />
        <Stack.Screen name="library" />
        <Stack.Screen name="reports" />
        <Stack.Screen name="alerts" />
        <Stack.Screen name="farm" />
        <Stack.Screen name="settings" />
      </Stack>
    </>
  );
}

export default function RootLayout() {
  return (
    <AppSettingsProvider>
      <RootStack />
    </AppSettingsProvider>
  );
}
