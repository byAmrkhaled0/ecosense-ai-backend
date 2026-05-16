class SensorModel {
  final String id;
  final String name;
  final String type; // 'soil_moisture', 'temperature', 'humidity', etc
  final String plantId;
  final double currentValue;
  final String unit;
  final double minThreshold;
  final double maxThreshold;
  final DateTime lastReading;
  final bool isActive;

  SensorModel({
    required this.id,
    required this.name,
    required this.type,
    required this.plantId,
    required this.currentValue,
    required this.unit,
    required this.minThreshold,
    required this.maxThreshold,
    required this.lastReading,
    required this.isActive,
  });

  factory SensorModel.fromJson(Map<String, dynamic> json) {
    return SensorModel(
      id: json['id'] as String,
      name: json['name'] as String,
      type: json['type'] as String,
      plantId: json['plantId'] as String,
      currentValue: (json['currentValue'] as num).toDouble(),
      unit: json['unit'] as String,
      minThreshold: (json['minThreshold'] as num).toDouble(),
      maxThreshold: (json['maxThreshold'] as num).toDouble(),
      lastReading: DateTime.parse(json['lastReading'] as String),
      isActive: json['isActive'] as bool? ?? true,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'type': type,
      'plantId': plantId,
      'currentValue': currentValue,
      'unit': unit,
      'minThreshold': minThreshold,
      'maxThreshold': maxThreshold,
      'lastReading': lastReading.toIso8601String(),
      'isActive': isActive,
    };
  }
}