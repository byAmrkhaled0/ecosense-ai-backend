import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../providers/language_provider.dart';

class AppTranslations {
  static final Map<String, Map<String, String>> _translations = {
    'en': {
      'welcome_back': 'Welcome Back',
      'sign_in_continue': 'Sign in to continue to Ecosense',
      'email': 'Email',
      'password': 'Password',
      'login': 'Login',
      'no_account': 'Don\'t have an account? ',
      'sign_up': 'Sign Up',
      'skip_test': 'Skip (Test Mode)',
      
      'home_dashboard': 'Home / Dashboard',
      'total_plants': 'Total Plants',
      'healthy': 'Healthy',
      'under_stress': 'Under Stress',
      'diseased': 'Diseased',
      'plant_health_dist': 'Plant Health Distribution',
      'chart_area': 'Chart Area',
      
      'nav_home': 'Home',
      'nav_datasets': 'Datasets',
      'nav_analytics': 'Analytics',
      'nav_predictions': 'Predictions',
      'nav_about': 'About',
      
      'menu_home': 'Home',
      'menu_profile': 'User Profile',
      'menu_sensor': 'Sensor Details',
      'menu_history': 'Prediction History',
      'menu_chat': 'AI Chat',
      'menu_upload': 'Upload Dataset',
      'menu_report': 'Plant Report',
      'menu_admin': 'Admin Panel',
      'menu_logout': 'Logout',
      
      'ai_chat_title': 'AI Chat (Dr. Farmer)',
      'type_message': 'Type your message...',
    },
    'ar': {
      'welcome_back': 'مرحباً بعودتك',
      'sign_in_continue': 'قم بتسجيل الدخول للمتابعة إلى Ecosense',
      'email': 'البريد الإلكتروني',
      'password': 'كلمة المرور',
      'login': 'تسجيل الدخول',
      'no_account': 'ليس لديك حساب؟ ',
      'sign_up': 'إنشاء حساب',
      'skip_test': 'تخطي (وضع الاختبار)',
      
      'home_dashboard': 'الرئيسية / لوحة التحكم',
      'total_plants': 'إجمالي النباتات',
      'healthy': 'سليمة',
      'under_stress': 'تحت ضغط',
      'diseased': 'مريضة',
      'plant_health_dist': 'توزيع صحة النباتات',
      'chart_area': 'منطقة الرسم البياني',
      
      'nav_home': 'الرئيسية',
      'nav_datasets': 'البيانات',
      'nav_analytics': 'التحليلات',
      'nav_predictions': 'التوقعات',
      'nav_about': 'حول',
      
      'menu_home': 'الرئيسية',
      'menu_profile': 'الملف الشخصي',
      'menu_sensor': 'تفاصيل الحساسات',
      'menu_history': 'سجل التوقعات',
      'menu_chat': 'المساعد الذكي',
      'menu_upload': 'رفع بيانات',
      'menu_report': 'تقرير النبات',
      'menu_admin': 'لوحة التحكم',
      'menu_logout': 'تسجيل الخروج',
      
      'ai_chat_title': 'المساعد الذكي (Dr. Farmer)',
      'type_message': 'اكتب رسالتك...',
    }
  };

  static String get(BuildContext context, String key) {
    final languageProvider = Provider.of<LanguageProvider>(context, listen: true);
    final langCode = languageProvider.locale.languageCode;
    return _translations[langCode]?[key] ?? key;
  }
}
