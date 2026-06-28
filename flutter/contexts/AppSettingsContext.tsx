import AsyncStorage from "@react-native-async-storage/async-storage";
import {
    createContext,
    ReactNode,
    useContext,
    useEffect,
    useMemo,
    useState,
} from "react";

export type AppLanguage = "en" | "ar";
export type AppThemeMode = "light" | "dark";

const LANGUAGE_KEY = "ecosense_language";
const THEME_KEY = "ecosense_theme";

const lightTheme = {
  background: "#F3F8F1",
  card: "#FFFFFF",
  cardSoft: "#F7FAF8",
  text: "#082A1F",
  muted: "#65786D",
  primary: "#118A5B",
  primaryDark: "#0B2A22",
  border: "#DDE9E2",
  danger: "#B42318",
  warning: "#D97706",
  success: "#087A4B",
  tabBar: "#FFFFFF",
};

const darkTheme = {
  background: "#071B15",
  card: "#0B2A22",
  cardSoft: "#12382D",
  text: "#FFFFFF",
  muted: "#BFD4C8",
  primary: "#19A36F",
  primaryDark: "#06140F",
  border: "#1F4A3D",
  danger: "#F97066",
  warning: "#FACC15",
  success: "#86EFAC",
  tabBar: "#0B2A22",
};

const translations: Record<string, { en: string; ar: string }> = {
  "tab.dashboard": { en: "Dashboard", ar: "الرئيسية" },
  "tab.diagnosis": { en: "Diagnosis", ar: "التشخيص" },
  "tab.sensors": { en: "Sensors", ar: "الحساسات" },
  "tab.analysis": { en: "Analysis", ar: "التحليل" },
  "tab.menu": { en: "Menu", ar: "القائمة" },

  "settings.kicker": { en: "ECOSENSE SETTINGS", ar: "إعدادات إيكوسينس" },
  "settings.title": { en: "Settings", ar: "الإعدادات" },
  "settings.subtitle": {
    en: "Control appearance, language and account preferences.",
    ar: "تحكم في المظهر واللغة وإعدادات الحساب.",
  },
  "settings.language": { en: "Language", ar: "اللغة" },
  "settings.languageDesc": {
    en: "Switch between English and Arabic.",
    ar: "بدّل بين اللغة الإنجليزية والعربية.",
  },
  "settings.theme": { en: "Appearance", ar: "المظهر" },
  "settings.themeDesc": {
    en: "Switch between light and dark mode.",
    ar: "بدّل بين الوضع الفاتح والوضع الداكن.",
  },
  "settings.english": { en: "English", ar: "English" },
  "settings.arabic": { en: "Arabic", ar: "العربية" },
  "settings.light": { en: "Light Mode", ar: "الوضع الفاتح" },
  "settings.dark": { en: "Dark Mode", ar: "الوضع الداكن" },
  "settings.account": { en: "Account", ar: "الحساب" },
  "settings.accountDesc": {
    en: "User account and saved preferences are stored locally.",
    ar: "بيانات الحساب والإعدادات محفوظة على الجهاز.",
  },
  "settings.saved": { en: "Saved", ar: "تم الحفظ" },
  "settings.savedDesc": {
    en: "Your preference has been saved successfully.",
    ar: "تم حفظ اختيارك بنجاح.",
  },
  "common.back": { en: "‹ Back", ar: "رجوع ›" },
};

type AppSettingsContextValue = {
  language: AppLanguage;
  themeMode: AppThemeMode;
  theme: typeof lightTheme;
  isArabic: boolean;
  isDark: boolean;
  setLanguage: (language: AppLanguage) => Promise<void>;
  setThemeMode: (themeMode: AppThemeMode) => Promise<void>;
  toggleLanguage: () => Promise<void>;
  toggleTheme: () => Promise<void>;
  t: (key: string) => string;
};

const AppSettingsContext = createContext<AppSettingsContextValue | null>(null);

export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<AppLanguage>("en");
  const [themeMode, setThemeModeState] = useState<AppThemeMode>("light");

  useEffect(() => {
    async function loadSettings() {
      const savedLanguage = await AsyncStorage.getItem(LANGUAGE_KEY);
      const savedTheme = await AsyncStorage.getItem(THEME_KEY);

      if (savedLanguage === "en" || savedLanguage === "ar") {
        setLanguageState(savedLanguage);
      }

      if (savedTheme === "light" || savedTheme === "dark") {
        setThemeModeState(savedTheme);
      }
    }

    loadSettings();
  }, []);

  async function setLanguage(newLanguage: AppLanguage) {
    setLanguageState(newLanguage);
    await AsyncStorage.setItem(LANGUAGE_KEY, newLanguage);
  }

  async function setThemeMode(newTheme: AppThemeMode) {
    setThemeModeState(newTheme);
    await AsyncStorage.setItem(THEME_KEY, newTheme);
  }

  async function toggleLanguage() {
    await setLanguage(language === "en" ? "ar" : "en");
  }

  async function toggleTheme() {
    await setThemeMode(themeMode === "light" ? "dark" : "light");
  }

  const value = useMemo(() => {
    const theme = themeMode === "dark" ? darkTheme : lightTheme;

    return {
      language,
      themeMode,
      theme,
      isArabic: language === "ar",
      isDark: themeMode === "dark",
      setLanguage,
      setThemeMode,
      toggleLanguage,
      toggleTheme,
      t: (key: string) => translations[key]?.[language] || key,
    };
  }, [language, themeMode]);

  return (
    <AppSettingsContext.Provider value={value}>
      {children}
    </AppSettingsContext.Provider>
  );
}

export function useAppSettings() {
  const context = useContext(AppSettingsContext);

  if (!context) {
    throw new Error("useAppSettings must be used inside AppSettingsProvider");
  }

  return context;
}
