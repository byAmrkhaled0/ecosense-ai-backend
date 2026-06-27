class PlantModel {
  final String id;
  final String name;
  final String species;
  final String? description;
  final String? image;
  final String healthStatus; // 'healthy', 'stress', 'diseased'
  final double healthPercentage;
  final DateTime plantedDate;
  final String location;
  final List<String> sensorIds;
  final DateTime createdAt;

  PlantModel({
    required this.id,
    required this.name,
    required this.species,
    this.description,
    this.image,
    required this.healthStatus,
    required this.healthPercentage,
    required this.plantedDate,
    required this.location,
    required this.sensorIds,
    required this.createdAt,
  });

  factory PlantModel.fromJson(Map<String, dynamic> json) {
    return PlantModel(
      id: json['id'] as String,
      name: json['name'] as String,
      species: json['species'] as String,
      description: json['description'] as String?,
      image: json['image'] as String?,
      healthStatus: json['healthStatus'] as String,
      healthPercentage: (json['healthPercentage'] as num).toDouble(),
      plantedDate: DateTime.parse(json['plantedDate'] as String),
      location: json['location'] as String,
      sensorIds: List<String>.from(json['sensorIds'] as List? ?? []),
      createdAt: DateTime.parse(json['createdAt'] as String),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'species': species,
      'description': description,
      'image': image,
      'healthStatus': healthStatus,
      'healthPercentage': healthPercentage,
      'plantedDate': plantedDate.toIso8601String(),
      'location': location,
      'sensorIds': sensorIds,
      'createdAt': createdAt.toIso8601String(),
    };
  }
}