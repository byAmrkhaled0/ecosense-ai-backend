import { router } from "expo-router";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

const diseases = [
  {
    name: "Leaf Spot",
    crop: "Tomato",
    desc: "Dark circular spots on leaves caused by fungal or bacterial infection.",
  },
  {
    name: "Chlorosis",
    crop: "Pepper",
    desc: "Yellowing leaves caused by nutrition imbalance, water stress, or low light.",
  },
  {
    name: "Powdery Mildew",
    crop: "Cucumber",
    desc: "White powder on leaves, usually related to humidity and poor ventilation.",
  },
];

export default function LibraryScreen() {
  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.content}>
      <Pressable onPress={() => router.back()}>
        <Text style={styles.back}>‹ Back</Text>
      </Pressable>

      <Text style={styles.kicker}>PLANT LIBRARY</Text>
      <Text style={styles.title}>Disease Guide</Text>
      <Text style={styles.subtitle}>
        Learn symptoms, causes and treatment recommendations for common plant
        diseases.
      </Text>

      {diseases.map((item) => (
        <View style={styles.card} key={item.name}>
          <View style={styles.iconBox}>
            <Text style={styles.icon}>🌿</Text>
          </View>

          <Text style={styles.name}>{item.name}</Text>
          <Text style={styles.crop}>{item.crop}</Text>
          <Text style={styles.desc}>{item.desc}</Text>

          <View style={styles.infoBox}>
            <Text style={styles.infoText}>
              Recommended: isolate affected leaves and improve monitoring.
            </Text>
          </View>
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: "#F3F8F1" },
  content: { padding: 22, paddingBottom: 40 },
  back: { marginTop: 28, color: "#118A5B", fontSize: 15, fontWeight: "900" },
  kicker: {
    marginTop: 22,
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
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 26,
    padding: 18,
    marginBottom: 15,
    borderWidth: 1,
    borderColor: "#DDE9E2",
  },
  iconBox: {
    width: 54,
    height: 54,
    borderRadius: 18,
    backgroundColor: "#E7F4EA",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 12,
  },
  icon: { fontSize: 26 },
  name: { color: "#0B2A22", fontSize: 21, fontWeight: "900" },
  crop: { marginTop: 4, color: "#118A5B", fontSize: 13, fontWeight: "900" },
  desc: { marginTop: 10, color: "#62766B", fontSize: 14, lineHeight: 22 },
  infoBox: {
    marginTop: 14,
    backgroundColor: "#F0F8F2",
    padding: 14,
    borderRadius: 18,
  },
  infoText: { color: "#446155", fontWeight: "700", lineHeight: 20 },
});
