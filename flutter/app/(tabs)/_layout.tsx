import AsyncStorage from "@react-native-async-storage/async-storage";
import { Tabs, router } from "expo-router";
import { useEffect, useState } from "react";
import { ActivityIndicator, Text, View } from "react-native";
import { useAppSettings } from "../../contexts/AppSettingsContext";

export default function TabsLayout() {
  const { theme, t } = useAppSettings();
  const [checkingAuth, setCheckingAuth] = useState(true);

  useEffect(() => {
    let mounted = true;

    async function checkAuth() {
      try {
        const token = await AsyncStorage.getItem("ecosense_token");

        if (!mounted) return;

        if (!token) {
          router.replace("/splash" as any);
          return;
        }

        setCheckingAuth(false);
      } catch {
        router.replace("/splash" as any);
      }
    }

    checkAuth();

    return () => {
      mounted = false;
    };
  }, []);

  if (checkingAuth) {
    return (
      <View
        style={{
          flex: 1,
          backgroundColor: "#F3F8F1",
          alignItems: "center",
          justifyContent: "center",
          padding: 24,
        }}
      >
        <ActivityIndicator size="large" color="#118A5B" />
        <Text
          style={{
            marginTop: 14,
            color: "#65786D",
            fontSize: 14,
            fontWeight: "800",
          }}
        >
          Checking your account...
        </Text>
      </View>
    );
  }

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: theme.primary,
        tabBarInactiveTintColor: theme.muted,
        tabBarStyle: {
          backgroundColor: theme.tabBar,
          height: 72,
          paddingBottom: 10,
          paddingTop: 8,
          borderTopWidth: 1,
          borderTopColor: theme.border,
        },
        tabBarLabelStyle: {
          fontSize: 11,
          fontWeight: "800",
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: t("tab.dashboard"),
          tabBarIcon: ({ color }) => (
            <Text style={{ color, fontSize: 20 }}>⌂</Text>
          ),
        }}
      />

      <Tabs.Screen
        name="diagnosis"
        options={{
          title: t("tab.diagnosis"),
          tabBarIcon: ({ color }) => (
            <Text style={{ color, fontSize: 20 }}>⌕</Text>
          ),
        }}
      />

      <Tabs.Screen
        name="sensors"
        options={{
          title: t("tab.sensors"),
          tabBarIcon: ({ color }) => (
            <Text style={{ color, fontSize: 20 }}>⌁</Text>
          ),
        }}
      />

      <Tabs.Screen
        name="analysis"
        options={{
          title: t("tab.analysis"),
          tabBarIcon: ({ color }) => (
            <Text style={{ color, fontSize: 20 }}>◴</Text>
          ),
        }}
      />

      <Tabs.Screen
        name="menu"
        options={{
          title: t("tab.menu"),
          tabBarIcon: ({ color }) => (
            <Text style={{ color, fontSize: 20 }}>☰</Text>
          ),
        }}
      />

      <Tabs.Screen name="explore" options={{ href: null }} />
    </Tabs>
  );
}
