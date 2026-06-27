import 'dart:convert';

import '../core/network/api_client.dart';
import '../core/network/api_endpoints.dart';
import '../core/services/token_storage_service.dart';
import '../models/user_model.dart';

class AuthLoginResult {
  AuthLoginResult({required this.token, required this.user});

  final String token;
  final UserModel user;
}

class AuthRepository {
  AuthRepository({
    required this.apiClient,
    required this.tokenStorage,
  });

  final ApiClient apiClient;
  final TokenStorageService tokenStorage;

  Future<void> register({
    required String email,
    required String password,
    required String firstName,
    required String lastName,
    required String address,
    required String phoneNumber,
  }) async {
    await apiClient.post(
      ApiEndpoints.authRegister,
      body: {
        'email': email,
        'password': password,
        'firstName': firstName,
        'lastName': lastName,
        'address': address,
        'phoneNumber': phoneNumber,
      },
    );
  }

  Future<void> verify({required String code}) async {
    await apiClient.post(
      ApiEndpoints.authVerify,
      body: {'code': code},
    );
  }

  Future<AuthLoginResult> login({
    required String email,
    required String password,
  }) async {
    final raw = await apiClient.post(
      ApiEndpoints.authLogin,
      body: {
        'email': email,
        'password': password,
      },
    );

    if (raw is! Map<String, dynamic>) {
      throw Exception('Invalid login response');
    }

    final token = raw['token'] as String?;
    if (token == null || token.isEmpty) {
      throw Exception('Missing token in login response');
    }

    apiClient.setAuthToken(token);
    await tokenStorage.saveToken(token);

    final userRaw = raw['user'];
    if (userRaw is! Map<String, dynamic>) {
      throw Exception('Missing user in login response');
    }
    final user = UserModel.fromJson(userRaw);
    await tokenStorage.saveUserJson(jsonEncode(user.toJson()));

    return AuthLoginResult(token: token, user: user);
  }

  Future<AuthLoginResult> googleNative({
    required String idToken,
    String? address,
    String? phoneNumber,
  }) async {
    final raw = await apiClient.post(
      ApiEndpoints.authGoogleNative,
      body: {
        'idToken': idToken,
        'address': address ?? '',
        'phoneNumber': phoneNumber ?? '',
      },
    );

    if (raw is! Map<String, dynamic>) {
      throw Exception('Invalid Google login response');
    }

    final token = raw['token'] as String?;
    if (token == null || token.isEmpty) {
      throw Exception('Missing token in Google login response');
    }

    apiClient.setAuthToken(token);
    await tokenStorage.saveToken(token);

    final userRaw = raw['user'];
    if (userRaw is! Map<String, dynamic>) {
      throw Exception('Missing user in Google login response');
    }
    final user = UserModel.fromJson(userRaw);
    await tokenStorage.saveUserJson(jsonEncode(user.toJson()));

    return AuthLoginResult(token: token, user: user);
  }

  Future<UserModel> fetchMe() async {
    final raw = await apiClient.get(ApiEndpoints.authMe);

    Map<String, dynamic> userMap;
    if (raw is Map<String, dynamic>) {
      if (raw['user'] is Map<String, dynamic>) {
        userMap = Map<String, dynamic>.from(raw['user'] as Map);
      } else {
        userMap = raw;
      }
    } else {
      throw Exception('Invalid /me response');
    }

    final user = UserModel.fromJson(userMap);
    await tokenStorage.saveUserJson(jsonEncode(user.toJson()));
    return user;
  }

  Future<void> addWorker({
    required String email,
    required String password,
    required String firstName,
    required String lastName,
    required String assignedSector,
  }) async {
    await apiClient.post(
      ApiEndpoints.authAddWorker,
      body: {
        'email': email,
        'password': password,
        'firstName': firstName,
        'lastName': lastName,
        'assignedSector': assignedSector,
      },
    );
  }

  Future<void> logoutRemote() async {
    try {
      await apiClient.get(ApiEndpoints.authLogout);
    } catch (_) {
      // Still clear local session if server fails.
    } finally {
      apiClient.setAuthToken(null);
      await tokenStorage.clearSession();
    }
  }
}
