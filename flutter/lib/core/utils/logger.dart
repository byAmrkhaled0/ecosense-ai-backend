import 'package:flutter/foundation.dart';

class AppLogger {
  static void info(String message) {
    if (kDebugMode) {
      print('[INFO] $message');
    }
  }

  static void debug(String message) {
    if (kDebugMode) {
      print('[DEBUG] $message');
    }
  }

  static void warning(String message) {
    if (kDebugMode) {
      print('[WARNING] $message');
    }
  }

  static void error(String message, [dynamic error, StackTrace? stackTrace]) {
    if (kDebugMode) {
      print('[ERROR] $message');
      if (error != null) print('Error: $error');
      if (stackTrace != null) print('StackTrace: $stackTrace');
    }
  }

  static void network(String method, String path, int? statusCode) {}
}
