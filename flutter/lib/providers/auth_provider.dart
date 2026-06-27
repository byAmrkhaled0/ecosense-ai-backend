import 'dart:convert';

import 'package:flutter/material.dart';

import '../core/network/api_client.dart';
import '../core/services/token_storage_service.dart';
import '../models/user_model.dart';
import '../repositories/auth_repository.dart';

class AuthProvider extends ChangeNotifier {
  AuthProvider({
    required AuthRepository authRepository,
    required TokenStorageService tokenStorage,
    required ApiClient apiClient,
  })  : _authRepository = authRepository,
        _tokenStorage = tokenStorage,
        _apiClient = apiClient;

  final AuthRepository _authRepository;
  final TokenStorageService _tokenStorage;
  final ApiClient _apiClient;

  UserModel? _user;
  bool _isLoading = false;
  String? _error;

  UserModel? get user => _user;
  bool get isLoading => _isLoading;
  String? get error => _error;
  bool get isAuthenticated => _user != null;

  String _formatError(Object e) => e.toString();

  Future<void> restoreSession() async {
    _error = null;
    final token = await _tokenStorage.readToken();
    if (token == null || token.isEmpty) {
      _user = null;
      _apiClient.setAuthToken(null);
      notifyListeners();
      return;
    }

    _apiClient.setAuthToken(token);

    try {
      _user = await _authRepository.fetchMe();
    } catch (_) {
      final cached = await _tokenStorage.readUserJson();
      if (cached != null && cached.isNotEmpty) {
        try {
          _user = UserModel.fromJson(
            jsonDecode(cached) as Map<String, dynamic>,
          );
        } catch (_) {
          _user = null;
          await _tokenStorage.clearSession();
          _apiClient.setAuthToken(null);
        }
      } else {
        _user = null;
      }
    }
    notifyListeners();
  }

  Future<bool> login({
    required String email,
    required String password,
  }) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      final result = await _authRepository.login(
        email: email.trim(),
        password: password,
      );
      _user = result.user;
      _error = null;
      notifyListeners();
      return true;
    } catch (e) {
      _error = _formatError(e);
      notifyListeners();
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<bool> register({
    required String firstName,
    required String lastName,
    required String email,
    required String password,
    required String address,
    required String phoneNumber,
  }) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      await _authRepository.register(
        email: email.trim(),
        password: password,
        firstName: firstName.trim(),
        lastName: lastName.trim(),
        address: address.trim(),
        phoneNumber: phoneNumber.trim(),
      );
      _error = null;
      notifyListeners();
      return true;
    } catch (e) {
      _error = _formatError(e);
      notifyListeners();
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<bool> verifyOtp(String code) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      await _authRepository.verify(code: code.trim());
      _error = null;
      notifyListeners();
      return true;
    } catch (e) {
      _error = _formatError(e);
      notifyListeners();
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<bool> loginWithGoogle({
    required String idToken,
    String? address,
    String? phoneNumber,
  }) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      final result = await _authRepository.googleNative(
        idToken: idToken,
        address: address,
        phoneNumber: phoneNumber,
      );
      _user = result.user;
      _error = null;
      notifyListeners();
      return true;
    } catch (e) {
      _error = _formatError(e);
      notifyListeners();
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<bool> addWorker({
    required String email,
    required String password,
    required String firstName,
    required String lastName,
    required String assignedSector,
  }) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      await _authRepository.addWorker(
        email: email.trim(),
        password: password,
        firstName: firstName.trim(),
        lastName: lastName.trim(),
        assignedSector: assignedSector.trim(),
      );
      _error = null;
      notifyListeners();
      return true;
    } catch (e) {
      _error = _formatError(e);
      notifyListeners();
      return false;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> logout() async {
    await _authRepository.logoutRemote();
    _user = null;
    _error = null;
    notifyListeners();
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }

  void skipAuth() {
    _user = UserModel(
      id: 'mock_user_123',
      name: 'Guest User',
      email: 'guest@example.com',
      userType: 'owner',
      createdAt: DateTime.now(),
    );
    _error = null;
    notifyListeners();
  }
}
