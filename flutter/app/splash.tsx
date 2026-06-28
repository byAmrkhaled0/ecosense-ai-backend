import AsyncStorage from "@react-native-async-storage/async-storage";
import { router } from "expo-router";
import { useEffect, useRef } from "react";
import { Animated, Easing, Image, StyleSheet, Text, View } from "react-native";

const logo = require("../assets/logo.png");

export default function SplashScreen() {
  const logoScale = useRef(new Animated.Value(0.75)).current;
  const logoOpacity = useRef(new Animated.Value(0)).current;
  const titleOpacity = useRef(new Animated.Value(0)).current;
  const progress = useRef(new Animated.Value(0)).current;

  const progressWidth = progress.interpolate({
    inputRange: [0, 1],
    outputRange: ["0%", "100%"],
  });

  useEffect(() => {
    Animated.sequence([
      Animated.parallel([
        Animated.timing(logoOpacity, {
          toValue: 1,
          duration: 700,
          easing: Easing.out(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.spring(logoScale, {
          toValue: 1,
          friction: 5,
          tension: 70,
          useNativeDriver: true,
        }),
      ]),

      Animated.timing(titleOpacity, {
        toValue: 1,
        duration: 500,
        easing: Easing.out(Easing.ease),
        useNativeDriver: true,
      }),

      Animated.timing(progress, {
        toValue: 1,
        duration: 1300,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: false,
      }),
    ]).start();

    const timer = setTimeout(async () => {
      const token = await AsyncStorage.getItem("ecosense_token");

      if (token) {
        router.replace("/(tabs)" as any);
      } else {
        router.replace("/landing" as any);
      }
    }, 2800);

    return () => clearTimeout(timer);
  }, [logoOpacity, logoScale, progress, titleOpacity]);

  return (
    <View style={styles.screen}>
      <View style={styles.glowOne} />
      <View style={styles.glowTwo} />

      <Animated.View
        style={[
          styles.logoWrapper,
          {
            opacity: logoOpacity,
            transform: [{ scale: logoScale }],
          },
        ]}
      >
        <Image source={logo} style={styles.logo} resizeMode="contain" />
      </Animated.View>

      <Animated.View style={[styles.textArea, { opacity: titleOpacity }]}>
        <Text style={styles.brand}>ECOSENSE AI</Text>
        <Text style={styles.subtitle}>Smart Plant Health Monitoring</Text>
      </Animated.View>

      <View style={styles.loaderTrack}>
        <Animated.View style={[styles.loaderFill, { width: progressWidth }]} />
      </View>

      <Text style={styles.loadingText}>Preparing your smart farm...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#F3F8F1",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
    overflow: "hidden",
  },
  glowOne: {
    position: "absolute",
    width: 280,
    height: 280,
    borderRadius: 140,
    backgroundColor: "rgba(17, 138, 91, 0.16)",
    top: 80,
    right: -100,
  },
  glowTwo: {
    position: "absolute",
    width: 340,
    height: 340,
    borderRadius: 170,
    backgroundColor: "rgba(139, 224, 179, 0.18)",
    bottom: -120,
    left: -120,
  },
  logoWrapper: {
    width: 270,
    height: 270,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 8,
  },
  logo: {
    width: 270,
    height: 270,
  },
  textArea: {
    alignItems: "center",
  },
  brand: {
    color: "#0B2A22",
    fontSize: 25,
    fontWeight: "900",
    letterSpacing: 2,
  },
  subtitle: {
    color: "#65786D",
    fontSize: 14,
    fontWeight: "700",
    marginTop: 8,
    textAlign: "center",
  },
  loaderTrack: {
    width: "72%",
    height: 8,
    backgroundColor: "#DDE9E2",
    borderRadius: 999,
    overflow: "hidden",
    marginTop: 42,
  },
  loaderFill: {
    height: "100%",
    backgroundColor: "#118A5B",
    borderRadius: 999,
  },
  loadingText: {
    color: "#65786D",
    fontSize: 12,
    fontWeight: "800",
    marginTop: 16,
  },
});
