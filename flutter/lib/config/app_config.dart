enum Environment { development, staging, production }

class AppConfig {
  static const Environment environment = Environment.development;

  static bool get isDevelopment => environment == Environment.development;
  static bool get isProduction => environment == Environment.production;

  static const String appName = 'Ecosense';
  static const String appVersion = '1.0.0';
  static const String buildNumber = '1';
}
