import { ScrollView, StyleSheet, Text, View } from "react-native";

export default function AnalysisScreen() {
  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Text style={styles.kicker}>DATA ANALYSIS</Text>
      <Text style={styles.title}>Dataset Analyzer</Text>
      <Text style={styles.subtitle}>
        Analyze plant images, final_status distribution, diseases and farm
        insights.
      </Text>

      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statValue}>2,480</Text>
          <Text style={styles.statLabel}>Images</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statValue}>86%</Text>
          <Text style={styles.statLabel}>Healthy</Text>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>final_status Distribution</Text>

        <View style={styles.progressItem}>
          <Text style={styles.progressText}>Healthy</Text>
          <View style={styles.track}>
            <View style={[styles.fill, { width: "70%" }]} />
          </View>
        </View>

        <View style={styles.progressItem}>
          <Text style={styles.progressText}>Moderate Stress</Text>
          <View style={styles.track}>
            <View style={[styles.fillWarning, { width: "20%" }]} />
          </View>
        </View>

        <View style={styles.progressItem}>
          <Text style={styles.progressText}>High Stress</Text>
          <View style={styles.track}>
            <View style={[styles.fillDanger, { width: "10%" }]} />
          </View>
        </View>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>AI Insights</Text>
        <Text style={styles.insight}>
          • Leaf Spot is the most repeated disease this week.
        </Text>
        <Text style={styles.insight}>
          • North House A has the highest stress cases.
        </Text>
        <Text style={styles.insight}>
          • Soil moisture is below safe range in recent readings.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: "#F3F8F1" },
  content: { padding: 22, paddingBottom: 40 },
  kicker: {
    marginTop: 28,
    color: "#118A5B",
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 1.4,
  },
  title: { marginTop: 8, color: "#082A1F", fontSize: 30, fontWeight: "900" },
  subtitle: {
    marginTop: 8,
    color: "#65786D",
    fontSize: 14,
    lineHeight: 22,
    marginBottom: 18,
  },
  statsRow: { flexDirection: "row", gap: 12, marginBottom: 16 },
  statCard: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    borderRadius: 22,
    padding: 18,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  statValue: { fontSize: 28, fontWeight: "900", color: "#082A1F" },
  statLabel: { color: "#687A70", fontWeight: "800", marginTop: 6 },
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 26,
    padding: 18,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "900",
    color: "#0B2A22",
    marginBottom: 16,
  },
  progressItem: { marginBottom: 14 },
  progressText: { color: "#0B2A22", fontWeight: "800", marginBottom: 7 },
  track: {
    height: 12,
    backgroundColor: "#EAF2EC",
    borderRadius: 999,
    overflow: "hidden",
  },
  fill: { height: "100%", backgroundColor: "#118A5B" },
  fillWarning: { height: "100%", backgroundColor: "#F3A51C" },
  fillDanger: { height: "100%", backgroundColor: "#D92D20" },
  insight: {
    color: "#5F7167",
    lineHeight: 22,
    marginBottom: 8,
    fontWeight: "700",
  },
});
