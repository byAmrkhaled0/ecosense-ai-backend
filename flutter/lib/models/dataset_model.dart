class DatasetModel {
  final String id;
  final String name;
  final String fileName;
  final int rowCount;
  final List<String> columns;
  final double fileSize;
  final String uploadStatus; // 'pending', 'processing', 'completed', 'failed'
  final DateTime uploadedAt;
  final DateTime? processedAt;

  DatasetModel({
    required this.id,
    required this.name,
    required this.fileName,
    required this.rowCount,
    required this.columns,
    required this.fileSize,
    required this.uploadStatus,
    required this.uploadedAt,
    this.processedAt,
  });

  factory DatasetModel.fromJson(Map<String, dynamic> json) {
    return DatasetModel(
      id: json['id'] as String,
      name: json['name'] as String,
      fileName: json['fileName'] as String,
      rowCount: json['rowCount'] as int,
      columns: List<String>.from(json['columns'] as List? ?? []),
      fileSize: (json['fileSize'] as num).toDouble(),
      uploadStatus: json['uploadStatus'] as String,
      uploadedAt: DateTime.parse(json['uploadedAt'] as String),
      processedAt: json['processedAt'] != null ? DateTime.parse(json['processedAt'] as String) : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'fileName': fileName,
      'rowCount': rowCount,
      'columns': columns,
      'fileSize': fileSize,
      'uploadStatus': uploadStatus,
      'uploadedAt': uploadedAt.toIso8601String(),
      'processedAt': processedAt?.toIso8601String(),
    };
  }
}