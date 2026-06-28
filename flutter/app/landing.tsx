import { router } from "expo-router";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

export default function LandingScreen() {
  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <View style={styles.topArea}>
        <View style={styles.logoBox}>
          <Text style={styles.logoText}>EC</Text>
        </View>

        <Text style={styles.brand}>ECOSENSE AI</Text>

        <Text style={styles.title}>Smart Plant Health Monitoring</Text>

        <Text style={styles.subtitle}>
          AI-powered mobile app for plant diagnosis, sensor monitoring, farm
          alerts and smart agriculture reports.
        </Text>
      </View>

      <View style={styles.heroCard}>
        <View style={styles.heroHeader}>
          <Text style={styles.heroTitle}>Plant Health System</Text>
          <Text style={styles.heroBadge}>AI + IoT</Text>
        </View>

        <Text style={styles.heroPercent}>86%</Text>
        <Text style={styles.heroText}>Estimated farm health overview</Text>

        <View style={styles.statsRow}>
          <View style={styles.statBox}>
            <Text style={styles.statValue}>AI</Text>
            <Text style={styles.statLabel}>Diagnosis</Text>
          </View>

          <View style={styles.statBox}>
            <Text style={styles.statValue}>5</Text>
            <Text style={styles.statLabel}>Sensors</Text>
          </View>

          <View style={styles.statBox}>
            <Text style={styles.statValue}>CAM</Text>
            <Text style={styles.statLabel}>ESP32</Text>
          </View>
        </View>
      </View>

      <View style={styles.featuresCard}>
        <FeatureItem
          title="Image Diagnosis"
          text="Analyze plant images using the AI model and show the final plant status."
        />

        <FeatureItem
          title="Hardware Sensors"
          text="Read temperature, humidity, soil moisture, soil temperature and light."
        />

        <FeatureItem
          title="Reports & Alerts"
          text="Track diagnosis history, warnings, farm reports and recommendations."
        />
      </View>

      <Pressable
        style={styles.primaryButton}
        onPress={() => router.push("/login" as any)}
      >
        <Text style={styles.primaryButtonText}>Login</Text>
      </Pressable>

      <Pressable
        style={styles.secondaryButton}
        onPress={() => router.push("/register" as any)}
      >
        <Text style={styles.secondaryButtonText}>Create Account</Text>
      </Pressable>

      <Text style={styles.footer}>
        Graduation Project • Smart Agriculture • AI Diagnosis
      </Text>
    </ScrollView>
  );
}

function FeatureItem({ title, text }: { title: string; text: string }) {
  return (
    <View style={styles.featureItem}>
      <View style={styles.featureIcon}>
        <Text style={styles.featureIconText}>✓</Text>
      </View>

      <View style={styles.featureContent}>
        <Text style={styles.featureTitle}>{title}</Text>
        <Text style={styles.featureText}>{text}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#F3F8F1",
  },
  content: {
    padding: 24,
    paddingTop: 60,
    paddingBottom: 40,
  },
  topArea: {
    marginBottom: 24,
  },
  logoBox: {
    width: 74,
    height: 74,
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
  brand: {
    color: "#118A5B",
    fontSize: 13,
    fontWeight: "900",
    letterSpacing: 2,
    marginBottom: 10,
  },
  title: {
    color: "#082A1F",
    fontSize: 40,
    lineHeight: 46,
    fontWeight: "900",
  },
  subtitle: {
    color: "#65786D",
    fontSize: 15,
    lineHeight: 24,
    marginTop: 14,
  },
  heroCard: {
    backgroundColor: "#0B2A22",
    borderRadius: 34,
    padding: 24,
    marginBottom: 18,
  },
  heroHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  heroTitle: {
    color: "#8BE0B3",
    fontSize: 13,
    fontWeight: "900",
    letterSpacing: 1,
  },
  heroBadge: {
    backgroundColor: "#DDF5E8",
    color: "#0B2A22",
    fontSize: 11,
    fontWeight: "900",
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 999,
    overflow: "hidden",
  },
  heroPercent: {
    color: "#FFFFFF",
    fontSize: 64,
    fontWeight: "900",
    marginTop: 14,
  },
  heroText: {
    color: "#CFE3D7",
    fontSize: 14,
    marginTop: 4,
  },
  statsRow: {
    flexDirection: "row",
    marginTop: 22,
  },
  statBox: {
    flex: 1,
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 18,
    padding: 12,
    marginRight: 10,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
  },
  statValue: {
    color: "#FFFFFF",
    fontSize: 19,
    fontWeight: "900",
  },
  statLabel: {
    color: "#BFD4C8",
    fontSize: 11,
    fontWeight: "800",
    marginTop: 4,
  },
  featuresCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 28,
    padding: 18,
    borderWidth: 1,
    borderColor: "#DDE9E2",
    marginBottom: 22,
  },
  featureItem: {
    flexDirection: "row",
    marginBottom: 16,
  },
  featureIcon: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: "#E6F7EE",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
    marginTop: 2,
  },
  featureIconText: {
    color: "#118A5B",
    fontSize: 15,
    fontWeight: "900",
  },
  featureContent: {
    flex: 1,
  },
  featureTitle: {
    color: "#0B2A22",
    fontSize: 16,
    fontWeight: "900",
  },
  featureText: {
    color: "#65786D",
    fontSize: 13,
    lineHeight: 20,
    marginTop: 4,
  },
  primaryButton: {
    backgroundColor: "#118A5B",
    paddingVertical: 17,
    borderRadius: 20,
    alignItems: "center",
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 17,
    fontWeight: "900",
  },
  secondaryButton: {
    backgroundColor: "#FFFFFF",
    paddingVertical: 16,
    borderRadius: 20,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#DDE9E2",
    marginTop: 12,
  },
  secondaryButtonText: {
    color: "#118A5B",
    fontSize: 16,
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
