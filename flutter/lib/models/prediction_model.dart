class PredictionModel {
  final String id;
  final String plantId;
  final String predictedCondition;
  final double confidence;
  final String recommendation;
  final List<String> suggestedActions;
  final DateTime createdAt;
  final DateTime? actionTakenAt;

  PredictionModel({
    required this.id,
    required this.plantId,
    required this.predictedCondition,
    required this.confidence,
    required this.recommendation,
    required this.suggestedActions,
    required this.createdAt,
    this.actionTakenAt,
  });

  factory PredictionModel.fromJson(Map<String, dynamic> json) {
    return PredictionModel(
      id: json['id'] as String,
      plantId: json['plantId'] as String,
      predictedCondition: json['predictedCondition'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      recommendation: json['recommendation'] as String,
      suggestedActions: List<String>.from(json['suggestedActions'] as List? ?? []),
      createdAt: DateTime.parse(json['createdAt'] as String),
      actionTakenAt: json['actionTakenAt'] != null ? DateTime.parse(json['actionTakenAt'] as String) : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'plantId': plantId,
      'predictedCondition': predictedCondition,
      'confidence': confidence,
      'recommendation': recommendation,
      'suggestedActions': suggestedActions,
      'createdAt': createdAt.toIso8601String(),
      'actionTakenAt': actionTakenAt?.toIso8601String(),
    };
  }
}