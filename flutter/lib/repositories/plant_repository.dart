import '../core/network/api_client.dart';
import '../models/plant_model.dart';

class PlantRepository {
  final ApiClient apiClient;

  PlantRepository({required this.apiClient});

  Future<List<PlantModel>> getPlants() async {
    try {
      final response = await apiClient.get('/plants');
      final plants = (response as List)
          .map((item) => PlantModel.fromJson(item))
          .toList();
      return plants;
    } catch (e) {
      rethrow;
    }
  }

  Future<PlantModel> getPlantById(String id) async {
    try {
      final response = await apiClient.get('/plants/$id');
      return PlantModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }

  Future<List<PlantModel>> getPlantsByUser(String userId) async {
    try {
      final response = await apiClient.get('/users/$userId/plants');
      final plants = (response as List)
          .map((item) => PlantModel.fromJson(item))
          .toList();
      return plants;
    } catch (e) {
      rethrow;
    }
  }

  Future<PlantModel> createPlant({
    required String userId,
    required String name,
    required String species,
    required String location,
  }) async {
    try {
      final response = await apiClient.post(
        '/plants',
        body: {
          'userId': userId,
          'name': name,
          'species': species,
          'location': location,
        },
      );
      return PlantModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }

  Future<PlantModel> updatePlant({
    required String id,
    required String name,
    required String species,
    required String location,
  }) async {
    try {
      final response = await apiClient.put(
        '/plants/$id',
        body: {
          'name': name,
          'species': species,
          'location': location,
        },
      );
      return PlantModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }

  Future<void> deletePlant(String id) async {
    try {
      await apiClient.delete('/plants/$id');
    } catch (e) {
      rethrow;
    }
  }
}
