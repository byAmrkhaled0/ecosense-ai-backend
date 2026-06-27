import '../core/network/api_client.dart';
import '../models/prediction_model.dart';

class PredictionRepository {
  final ApiClient apiClient;

  PredictionRepository({required this.apiClient});

  Future<List<PredictionModel>> getPredictions() async {
    try {
      final response = await apiClient.get('/predictions');
      final predictions = (response as List)
          .map((item) => PredictionModel.fromJson(item))
          .toList();
      return predictions;
    } catch (e) {
      rethrow;
    }
  }

  Future<PredictionModel> getPredictionById(String id) async {
    try {
      final response = await apiClient.get('/predictions/$id');
      return PredictionModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }

  Future<List<PredictionModel>> getPredictionsByPlant(String plantId) async {
    try {
      final response = await apiClient.get('/plants/$plantId/predictions');
      final predictions = (response as List)
          .map((item) => PredictionModel.fromJson(item))
          .toList();
      return predictions;
    } catch (e) {
      rethrow;
    }
  }

  Future<PredictionModel> createPrediction({
    required String plantId,
    required String disease,
    required double confidence,
    required String recommendation,
  }) async {
    try {
      final response = await apiClient.post(
        '/predictions',
        body: {
          'plantId': plantId,
          'disease': disease,
          'confidence': confidence,
          'recommendation': recommendation,
        },
      );
      return PredictionModel.fromJson(response);
    } catch (e) {
      rethrow;
    }
  }

  Future<List<PredictionModel>> getPredictionsByDateRange({
    required DateTime startDate,
    required DateTime endDate,
  }) async {
    try {
      final response = await apiClient.get(
        '/predictions/date-range',
        headers: {
          'startDate': startDate.toIso8601String(),
          'endDate': endDate.toIso8601String(),
        },
      );
      final predictions = (response as List)
          .map((item) => PredictionModel.fromJson(item))
          .toList();
      return predictions;
    } catch (e) {
      rethrow;
    }
  }

  Future<void> deletePrediction(String id) async {
    try {
      await apiClient.delete('/predictions/$id');
    } catch (e) {
      rethrow;
    }
  }
}
