import 'package:flutter/material.dart';

class HomeProvider extends ChangeNotifier {
  bool _isLoading = false;
  int _totalPlants = 0;
  int _healthyPlants = 0;
  int _stressedPlants = 0;
  int _diseasedPlants = 0;

  bool get isLoading => _isLoading;
  int get totalPlants => _totalPlants;
  int get healthyPlants => _healthyPlants;
  int get stressedPlants => _stressedPlants;
  int get diseasedPlants => _diseasedPlants;

  Future<void> fetchHomeData() async {
    _isLoading = true;
    notifyListeners();

    try {
      // TODO: Call API to fetch home data
      _totalPlants = 240;
      _healthyPlants = 163;
      _stressedPlants = 53;
      _diseasedPlants = 24;
      notifyListeners();
    } catch (e) {
      debugPrint('Error fetching home data: $e');
      notifyListeners();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
}
