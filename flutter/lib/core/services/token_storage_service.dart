import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../../config/constants.dart';

class TokenStorageService {
  TokenStorageService({FlutterSecureStorage? storage})
      : _storage = storage ?? const FlutterSecureStorage();

  final FlutterSecureStorage _storage;

  Future<String?> readToken() => _storage.read(key: AppConstants.userTokenKey);

  Future<void> saveToken(String token) =>
      _storage.write(key: AppConstants.userTokenKey, value: token);

  Future<String?> readUserJson() => _storage.read(key: AppConstants.userDataKey);

  Future<void> saveUserJson(String json) =>
      _storage.write(key: AppConstants.userDataKey, value: json);

  Future<void> clearSession() async {
    await _storage.delete(key: AppConstants.userTokenKey);
    await _storage.delete(key: AppConstants.userDataKey);
  }
}
