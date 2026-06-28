import { router } from "expo-router";
import {
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { logoutUser } from "../../services/api";

export default function MenuScreen() {
  async function handleLogout() {
    Alert.alert("Logout", "Are you sure you want to logout?", [
      {
        text: "Cancel",
        style: "cancel",
      },
      {
        text: "Logout",
        style: "destructive",
        onPress: async () => {
          await logoutUser();
          router.replace("/splash" as any);
        },
      },
    ]);
  }

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={styles.kicker}>ECOSENSE AI</Text>
      <Text style={styles.title}>Menu</Text>
      <Text style={styles.subtitle}>
        Manage your diagnoses, farm reports, alerts and application settings.
      </Text>

      <View style={styles.section}>
        <MenuButton
          title="My Diagnoses"
          description="View saved AI diagnosis history"
          onPress={() => router.push("/diagnoses" as any)}
        />

        <MenuButton
          title="Reports"
          description="Farm summaries and diagnosis reports"
          onPress={() => router.push("/reports" as any)}
        />

        <MenuButton
          title="Alerts"
          description="Warnings and important plant health alerts"
          onPress={() => router.push("/alerts" as any)}
        />

        <MenuButton
          title="Farm Management"
          description="Sectors, devices, camera and farm overview"
          onPress={() => router.push("/farm" as any)}
        />

        <MenuButton
          title="Plant Library"
          description="Plant diseases and diagnosis information"
          onPress={() => router.push("/library" as any)}
        />

        <MenuButton
          title="Settings"
          description="Profile, language, theme and app settings"
          onPress={() => router.push("/settings" as any)}
        />
      </View>

      <Pressable style={styles.logoutButton} onPress={handleLogout}>
        <Text style={styles.logoutText}>Logout</Text>
      </Pressable>
    </ScrollView>
  );
}

function MenuButton({
  title,
  description,
  onPress,
}: {
  title: string;
  description: string;
  onPress: () => void;
}) {
  return (
    <Pressable style={styles.menuButton} onPress={onPress}>
      <View>
        <Text style={styles.menuTitle}>{title}</Text>
        <Text style={styles.menuDescription}>{description}</Text>
      </View>

      <Text style={styles.arrow}>›</Text>
    </Pressable>
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
  kicker: {
    color: "#118A5B",
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 2,
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
    marginBottom: 22,
  },
  section: {
    gap: 12,
  },
  menuButton: {
    backgroundColor: "#FFFFFF",
    borderRadius: 22,
    padding: 18,
    borderWidth: 1,
    borderColor: "#DDE9E2",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  menuTitle: {
    color: "#0B2A22",
    fontSize: 17,
    fontWeight: "900",
  },
  menuDescription: {
    color: "#65786D",
    fontSize: 13,
    lineHeight: 20,
    marginTop: 4,
    maxWidth: 260,
  },
  arrow: {
    color: "#118A5B",
    fontSize: 32,
    fontWeight: "400",
  },
  logoutButton: {
    backgroundColor: "#B42318",
    borderRadius: 20,
    paddingVertical: 16,
    alignItems: "center",
    marginTop: 24,
  },
  logoutText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "900",
  },
});
