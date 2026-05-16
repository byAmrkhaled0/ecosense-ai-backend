class AppConstants {
  /// عنوان الـ API (بدون شرطة في النهاية).
  ///
  /// **مهم:** غيّر `_defaultApiBaseUrl` أدناه إلى رابط سيرفرك الحقيقي، أو عند التشغيل:
  /// `flutter run --dart-define=API_BASE_URL=https://api.موقعك.com/api`
  static const String _defaultApiBaseUrl = 'https://ecosense-backend.vercel.app/api';

  static String get apiBaseUrl {
    const fromEnv = String.fromEnvironment('API_BASE_URL');
    if (fromEnv.isNotEmpty) {
      return fromEnv.endsWith('/')
          ? fromEnv.substring(0, fromEnv.length - 1)
          : fromEnv;
    }
    return _defaultApiBaseUrl.endsWith('/')
        ? _defaultApiBaseUrl.substring(0, _defaultApiBaseUrl.length - 1)
        : _defaultApiBaseUrl;
  }

  static const String apiTimeout = '30000'; // milliseconds

  // Storage Keys
  static const String userTokenKey = 'user_token';
  static const String userDataKey = 'user_data';
  static const String themeKey = 'theme_mode';

  // Messages
  static const String loadingMessage = 'جاري التحميل...';
  static const String errorMessage = 'حدث خطأ، الرجاء المحاولة لاحقاً';
  static const String noDataMessage = 'لا توجد بيانات';

  // Pagination
  static const int pageSize = 20;
  static const int initialPage = 1;
}
