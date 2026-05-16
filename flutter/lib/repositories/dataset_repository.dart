import '../core/network/api_client.dart';
import '../models/dataset_model.dart';

class DatasetRepository {
  final ApiClient apiClient;

  DatasetRepository({required this.apiClient});

  Future<List<DatasetModel>> getDatasets() async {
    try {
      final response = await apiClient.get('/datasets');
      final datasets = (response as List)
          .map((item) => DatasetModel.fromJson(item))
          .toList();
      return datasets;
    } catch (e) {
      rethrow;
    }
  }

  Future<DatasetModel> getDatasetById(String id) async {
    try {
      final response = await apiClient.get('/datasets/$id');
      return DatasetModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }

  Future<List<DatasetModel>> searchDatasets({
    required String query,
  }) async {
    try {
      final response = await apiClient.get(
        '/datasets/search',
        headers: {'q': query},
      );
      final datasets = (response as List)
          .map((item) => DatasetModel.fromJson(item))
          .toList();
      return datasets;
    } catch (e) {
      rethrow;
    }
  }

  Future<DatasetModel> createDataset({
    required String name,
    required String description,
    required List<String> features,
  }) async {
    try {
      final response = await apiClient.post(
        '/datasets',
        body: {
          'name': name,
          'description': description,
          'features': features,
        },
      );
      return DatasetModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }

  Future<DatasetModel> updateDataset({
    required String id,
    required String name,
    required String description,
  }) async {
    try {
      final response = await apiClient.put(
        '/datasets/$id',
        body: {
          'name': name,
          'description': description,
        },
      );
      return DatasetModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }

  Future<void> deleteDataset(String id) async {
    try {
      await apiClient.delete('/datasets/$id');
    } catch (e) {
      rethrow;
    }
  }
}
