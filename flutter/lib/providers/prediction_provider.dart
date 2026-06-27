import 'package:flutter/material.dart';
import '../models/prediction_model.dart';

class PredictionProvider extends ChangeNotifier {
  final List<PredictionModel> _predictions = [];
  bool _isLoading = false;
  String? _error;

  List<PredictionModel> get predictions => _predictions;
  bool get isLoading => _isLoading;
  String? get error => _error;

  Future<void> fetchPredictions() async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      // TODO: Call API to fetch predictions
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}
