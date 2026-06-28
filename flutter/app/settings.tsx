import { router } from "expo-router";
import {
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import {
  AppLanguage,
  AppThemeMode,
  useAppSettings,
} from "../contexts/AppSettingsContext";

export default function SettingsScreen() {
  const { language, themeMode, theme, isArabic, setLanguage, setThemeMode, t } =
    useAppSettings();

  async function changeLanguage(value: AppLanguage) {
    await setLanguage(value);
    Alert.alert(t("settings.saved"), t("settings.savedDesc"));
  }

  async function changeTheme(value: AppThemeMode) {
    await setThemeMode(value);
    Alert.alert(t("settings.saved"), t("settings.savedDesc"));
  }

  return (
    <ScrollView
      style={[styles.screen, { backgroundColor: theme.background }]}
      contentContainerStyle={styles.content}
    >
      <Pressable onPress={() => router.back()}>
        <Text
          style={[
            styles.back,
            {
              color: theme.primary,
              textAlign: isArabic ? "right" : "left",
            },
          ]}
        >
          {t("common.back")}
        </Text>
      </Pressable>

      <Text
        style={[
          styles.kicker,
          {
            color: theme.primary,
            textAlign: isArabic ? "right" : "left",
          },
        ]}
      >
        {t("settings.kicker")}
      </Text>

      <Text
        style={[
          styles.title,
          {
            color: theme.text,
            textAlign: isArabic ? "right" : "left",
          },
        ]}
      >
        {t("settings.title")}
      </Text>

      <Text
        style={[
          styles.subtitle,
          {
            color: theme.muted,
            textAlign: isArabic ? "right" : "left",
          },
        ]}
      >
        {t("settings.subtitle")}
      </Text>

      <View
        style={[
          styles.card,
          {
            backgroundColor: theme.card,
            borderColor: theme.border,
          },
        ]}
      >
        <Text
          style={[
            styles.cardTitle,
            {
              color: theme.text,
              textAlign: isArabic ? "right" : "left",
            },
          ]}
        >
          {t("settings.language")}
        </Text>

        <Text
          style={[
            styles.cardText,
            {
              color: theme.muted,
              textAlign: isArabic ? "right" : "left",
            },
          ]}
        >
          {t("settings.languageDesc")}
        </Text>

        <View style={styles.optionRow}>
          <Pressable
            style={[
              styles.optionButton,
              {
                backgroundColor:
                  language === "en" ? theme.primary : theme.cardSoft,
                borderColor: theme.border,
              },
            ]}
            onPress={() => changeLanguage("en")}
          >
            <Text
              style={[
                styles.optionText,
                { color: language === "en" ? "#FFFFFF" : theme.text },
              ]}
            >
              {t("settings.english")}
            </Text>
          </Pressable>

          <Pressable
            style={[
              styles.optionButton,
              {
                backgroundColor:
                  language === "ar" ? theme.primary : theme.cardSoft,
                borderColor: theme.border,
              },
            ]}
            onPress={() => changeLanguage("ar")}
          >
            <Text
              style={[
                styles.optionText,
                { color: language === "ar" ? "#FFFFFF" : theme.text },
              ]}
            >
              {t("settings.arabic")}
            </Text>
          </Pressable>
        </View>
      </View>

      <View
        style={[
          styles.card,
          {
            backgroundColor: theme.card,
            borderColor: theme.border,
          },
        ]}
      >
        <Text
          style={[
            styles.cardTitle,
            {
              color: theme.text,
              textAlign: isArabic ? "right" : "left",
            },
          ]}
        >
          {t("settings.theme")}
        </Text>

        <Text
          style={[
            styles.cardText,
            {
              color: theme.muted,
              textAlign: isArabic ? "right" : "left",
            },
          ]}
        >
          {t("settings.themeDesc")}
        </Text>

        <View style={styles.optionRow}>
          <Pressable
            style={[
              styles.optionButton,
              {
                backgroundColor:
                  themeMode === "light" ? theme.primary : theme.cardSoft,
                borderColor: theme.border,
              },
            ]}
            onPress={() => changeTheme("light")}
          >
            <Text
              style={[
                styles.optionText,
                { color: themeMode === "light" ? "#FFFFFF" : theme.text },
              ]}
            >
              {t("settings.light")}
            </Text>
          </Pressable>

          <Pressable
            style={[
              styles.optionButton,
              {
                backgroundColor:
                  themeMode === "dark" ? theme.primary : theme.cardSoft,
                borderColor: theme.border,
              },
            ]}
            onPress={() => changeTheme("dark")}
          >
            <Text
              style={[
                styles.optionText,
                { color: themeMode === "dark" ? "#FFFFFF" : theme.text },
              ]}
            >
              {t("settings.dark")}
            </Text>
          </Pressable>
        </View>
      </View>

      <View
        style={[
          styles.card,
          {
            backgroundColor: theme.card,
            borderColor: theme.border,
          },
        ]}
      >
        <Text
          style={[
            styles.cardTitle,
            {
              color: theme.text,
              textAlign: isArabic ? "right" : "left",
            },
          ]}
        >
          {t("settings.account")}
        </Text>

        <Text
          style={[
            styles.cardText,
            {
              color: theme.muted,
              textAlign: isArabic ? "right" : "left",
            },
          ]}
        >
          {t("settings.accountDesc")}
        </Text>

        <View
          style={[
            styles.statusBox,
            {
              backgroundColor: theme.cardSoft,
              borderColor: theme.border,
            },
          ]}
        >
          <Text
            style={[
              styles.statusText,
              {
                color: theme.text,
                textAlign: isArabic ? "right" : "left",
              },
            ]}
          >
            {isArabic ? "اللغة الحالية: العربية" : "Current language: English"}
          </Text>

          <Text
            style={[
              styles.statusText,
              {
                color: theme.text,
                textAlign: isArabic ? "right" : "left",
              },
            ]}
          >
            {themeMode === "dark"
              ? isArabic
                ? "المظهر الحالي: الوضع الداكن"
                : "Current appearance: Dark Mode"
              : isArabic
                ? "المظهر الحالي: الوضع الفاتح"
                : "Current appearance: Light Mode"}
          </Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
  },
  content: {
    padding: 22,
    paddingBottom: 40,
  },
  back: {
    marginTop: 28,
    fontSize: 15,
    fontWeight: "900",
  },
  kicker: {
    marginTop: 22,
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 1.5,
  },
  title: {
    marginTop: 8,
    fontSize: 34,
    fontWeight: "900",
  },
  subtitle: {
    marginTop: 8,
    fontSize: 14,
    lineHeight: 22,
    marginBottom: 18,
  },
  card: {
    borderRadius: 24,
    padding: 18,
    borderWidth: 1,
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: "900",
    marginBottom: 8,
  },
  cardText: {
    fontSize: 14,
    lineHeight: 22,
    marginBottom: 16,
  },
  optionRow: {
    flexDirection: "row",
    gap: 10,
  },
  optionButton: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 16,
    borderWidth: 1,
    alignItems: "center",
  },
  optionText: {
    fontSize: 14,
    fontWeight: "900",
  },
  statusBox: {
    borderWidth: 1,
    borderRadius: 18,
    padding: 14,
    gap: 8,
  },
  statusText: {
    fontSize: 14,
    fontWeight: "800",
  },
});
