class ResponseModel<T> {
  final bool success;
  final String? message;
  final T? data;
  final int? statusCode;
  final dynamic error;

  ResponseModel({
    required this.success,
    this.message,
    this.data,
    this.statusCode,
    this.error,
  });

  factory ResponseModel.success({
    required T data,
    String? message,
    int? statusCode,
  }) {
    return ResponseModel(
      success: true,
      message: message,
      data: data,
      statusCode: statusCode ?? 200,
    );
  }

  factory ResponseModel.error({
    required String message,
    dynamic error,
    int? statusCode,
  }) {
    return ResponseModel(
      success: false,
      message: message,
      error: error,
      statusCode: statusCode ?? 500,
    );
  }
}