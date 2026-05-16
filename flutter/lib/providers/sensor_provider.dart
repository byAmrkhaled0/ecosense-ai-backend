import 'package:flutter/material.dart';
import '../models/sensor_model.dart';

class SensorProvider extends ChangeNotifier {
  final List<SensorModel> _sensors = [];
  bool _isLoading = false;
  String? _error;

  List<SensorModel> get sensors => _sensors;
  bool get isLoading => _isLoading;
  String? get error => _error;

  Future<void> fetchSensors() async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      // TODO: Call API to fetch sensors
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
