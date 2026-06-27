import '../core/network/api_client.dart';
import '../core/network/api_endpoints.dart';
import '../models/user_model.dart';

class UserRepository {
  final ApiClient apiClient;

  UserRepository({required this.apiClient});

  Future<UserModel> getUserById(String id) async {
    try {
      final response = await apiClient.get('/users/$id');
      return UserModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }

  Future<UserModel> updateUser({
    required String id,
    required String name,
    required String email,
  }) async {
    try {
      final response = await apiClient.put(
        '/users/$id',
        body: {
          'name': name,
          'email': email,
        },
      );
      return UserModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }

  Future<void> deleteUser(String id) async {
    try {
      await apiClient.delete('/users/$id');
    } catch (e) {
      rethrow;
    }
  }

  Future<UserModel> getUserProfile() async {
    try {
      final response = await apiClient.get(ApiEndpoints.getUserProfile);
      return UserModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }
}
