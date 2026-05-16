// ignore_for_file: depend_on_referenced_packages

import 'dart:convert';

import 'package:http/http.dart' as http;

import '../../config/constants.dart';
import '../utils/logger.dart';

class ApiClient {
  final String baseUrl;
  late http.Client _client;
  String? _authToken;
  String? _cookie; // Added for registrationToken cookie handling

  ApiClient({required this.baseUrl}) {
    _client = http.Client();
  }

  void setAuthToken(String? token) {
    _authToken = token;
  }

  String? get authToken => _authToken;

  Future<dynamic> get(String endpoint, {Map<String, String>? headers}) async {
    final url = Uri.parse('$baseUrl$endpoint');
    try {
      final response = await _client
          .get(url, headers: _defaultHeaders(headers))
          .timeout(Duration(milliseconds: int.parse(AppConstants.apiTimeout)));

      return _handleResponse(response);
    } catch (e) {
      _throwMappedNetworkError(e, url, 'GET');
    }
  }

  Future<dynamic> post(
    String endpoint, {
    dynamic body,
    Map<String, String>? headers,
  }) async {
    final url = Uri.parse('$baseUrl$endpoint');
    try {
      final response = await _client
          .post(
            url,
            headers: _defaultHeaders(headers),
            body: body == null ? null : jsonEncode(body),
          )
          .timeout(Duration(milliseconds: int.parse(AppConstants.apiTimeout)));

      return _handleResponse(response);
    } catch (e) {
      _throwMappedNetworkError(e, url, 'POST');
    }
  }

  Future<dynamic> put(
    String endpoint, {
    dynamic body,
    Map<String, String>? headers,
  }) async {
    final url = Uri.parse('$baseUrl$endpoint');
    try {
      final response = await _client
          .put(
            url,
            headers: _defaultHeaders(headers),
            body: body == null ? null : jsonEncode(body),
          )
          .timeout(Duration(milliseconds: int.parse(AppConstants.apiTimeout)));

      return _handleResponse(response);
    } catch (e) {
      _throwMappedNetworkError(e, url, 'PUT');
    }
  }

  Future<dynamic> delete(
    String endpoint, {
    Map<String, String>? headers,
  }) async {
    final url = Uri.parse('$baseUrl$endpoint');
    try {
      final response = await _client
          .delete(url, headers: _defaultHeaders(headers))
          .timeout(Duration(milliseconds: int.parse(AppConstants.apiTimeout)));

      return _handleResponse(response);
    } catch (e) {
      _throwMappedNetworkError(e, url, 'DELETE');
    }
  }

  Never _throwMappedNetworkError(Object e, Uri url, String verb) {
    AppLogger.error('$verb ${url.toString()}: $e');
    final urlStr = url.toString();
    final msg = e.toString();

    if (urlStr.contains('your-api-domain.com')) {
      throw ConfigurationException(
        'عنوان الـ API ما زال افتراضيًا (your-api-domain) وليس سيرفرك الحقيقي.\n\n'
        '• افتح الملف lib/config/constants.dart وغيّر _defaultApiBaseUrl\n'
        '• أو شغّل: flutter run --dart-define=API_BASE_URL=https://سيرفرك/api',
      );
    }

    if (msg.contains('Failed to fetch') ||
        msg.contains('ClientException') ||
        msg.contains('SocketException') ||
        msg.contains('HandshakeException')) {
      throw NetworkException(
        'تعذر الاتصال بالخادم.\n\n'
        '• تأكد أن الـ backend يعمل وأن الرابط في constants صحيح\n'
        '• إذا التطبيق على الويب (Chrome)، غالبًا تحتاج تفعيل CORS على الـ API',
      );
    }

    throw e is Exception ? e : Exception(e.toString());
  }

  Map<String, String> _defaultHeaders(Map<String, String>? customHeaders) {
    final headers = <String, String>{
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };
    final token = _authToken;
    if (token != null && token.isNotEmpty) {
      headers['Authorization'] = 'Bearer $token';
    }
    if (_cookie != null && _cookie!.isNotEmpty) {
      // Send the stored cookies back to the server
      headers['cookie'] = _cookie!;
    }
    if (customHeaders != null) {
      headers.addAll(customHeaders);
    }
    return headers;
  }

  dynamic _handleResponse(http.Response response) {
    // Extract set-cookie to use in subsequent requests (like /verify)
    final setCookie = response.headers['set-cookie'];
    if (setCookie != null) {
      // If there are multiple cookies, they might be comma separated by http package
      // For a simple implementation, we just store what's given.
      // We only split by ';' if we just want the first part, but sending the whole thing
      // or parsing it properly is better. http package joins them with comma.
      // A quick fix is to just assign it:
      _cookie = setCookie;
    }

    final body = response.body;
    dynamic decoded;
    if (body.isNotEmpty) {
      try {
        decoded = jsonDecode(body);
      } catch (_) {
        decoded = body;
      }
    }

    final code = response.statusCode;
    if (code == 204) {
      return decoded;
    }
    if (code == 200 || code == 201) {
      return decoded;
    }

    final message = decoded is Map && decoded['message'] != null
        ? decoded['message'].toString()
        : body;

    if (code == 400) {
      throw BadRequestException(message.toString());
    }
    if (code == 401) {
      throw UnauthorizedException(message.toString());
    }
    if (code == 403) {
      throw ForbiddenException(message.toString());
    }
    if (code == 404) {
      throw NotFoundException(message.toString());
    }
    if (code == 500) {
      throw ServerException(message.toString());
    }
    throw Exception('HTTP $code: $message');
  }
}

/// Host / URL not configured (still using docs placeholder).
class ConfigurationException implements Exception {
  ConfigurationException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// No route to server, timeout, DNS, or browser CORS blocking the request.
class NetworkException implements Exception {
  NetworkException(this.message);
  final String message;

  @override
  String toString() => message;
}

class BadRequestException implements Exception {
  final String message;
  BadRequestException(this.message);

  @override
  String toString() => message;
}

class UnauthorizedException implements Exception {
  final String message;
  UnauthorizedException(this.message);

  @override
  String toString() => message;
}

class ForbiddenException implements Exception {
  final String message;
  ForbiddenException(this.message);

  @override
  String toString() => message;
}

class NotFoundException implements Exception {
  final String message;
  NotFoundException(this.message);

  @override
  String toString() => message;
}

class ServerException implements Exception {
  final String message;
  ServerException(this.message);

  @override
  String toString() => message;
}
