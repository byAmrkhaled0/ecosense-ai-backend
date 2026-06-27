import 'package:flutter/material.dart';
import '../models/plant_model.dart';

class PlantProvider extends ChangeNotifier {
  final List<PlantModel> _plants = [];
  bool _isLoading = false;
  String? _error;

  List<PlantModel> get plants => _plants;
  bool get isLoading => _isLoading;
  String? get error => _error;

  Future<void> fetchPlants() async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      // TODO: Call API to fetch plants
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<void> addPlant(PlantModel plant) async {
    try {
      // TODO: Call API to add plant
      _plants.add(plant);
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    }
  }

  Future<void> updatePlant(PlantModel plant) async {
    try {
      // TODO: Call API to update plant
      final index = _plants.indexWhere((p) => p.id == plant.id);
      if (index != -1) {
        _plants[index] = plant;
        notifyListeners();
      }
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    }
  }

  Future<void> deletePlant(String plantId) async {
    try {
      // TODO: Call API to delete plant
      _plants.removeWhere((p) => p.id == plantId);
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    }
  }
}
