import 'package:flutter/material.dart';
import '../models/dataset_model.dart';

class DatasetProvider extends ChangeNotifier {
  final List<DatasetModel> _datasets = [];
  bool _isLoading = false;
  String? _error;

  List<DatasetModel> get datasets => _datasets;
  bool get isLoading => _isLoading;
  String? get error => _error;

  Future<void> fetchDatasets() async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      // TODO: Call API to fetch datasets
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> uploadDataset(String filePath, String fileName) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      // TODO: Call API to upload dataset
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
