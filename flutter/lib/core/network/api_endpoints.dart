class ApiEndpoints {
  // --- Auth (EcoSense API docs: POST/GET under /api/auth/...) ---
  static const String authRegister = '/auth/register';
  static const String authVerify = '/auth/verify';
  static const String authLogin = '/auth/login';
  static const String authGoogleNative = '/auth/google-native';
  static const String authAddWorker = '/auth/worker-add';
  static const String authMe = '/auth/me';
  static const String authLogout = '/auth/logout';

  /// Backward-compatible names
  static const String login = authLogin;
  static const String signup = authRegister;
  static const String logout = authLogout;

  static const String refreshToken = '/auth/refresh';

  // User endpoints
  static const String getUserProfile = '/users/profile';
  static const String updateUserProfile = '/users/profile';
  static const String getUserSettings = '/users/settings';

  // Plant endpoints
  static const String getPlants = '/plants';
  static const String getPlantById = '/plants/{id}';
  static const String createPlant = '/plants';
  static const String updatePlant = '/plants/{id}';
  static const String deletePlant = '/plants/{id}';

  // Sensor endpoints
  static const String getSensors = '/sensors';
  static const String getSensorById = '/sensors/{id}';
  static const String getSensorReadings = '/sensors/{id}/readings';

  // Prediction endpoints
  static const String getPredictions = '/predictions';
  static const String createPrediction = '/predictions';
  static const String getPredictionHistory = '/predictions/history';

  // Dataset endpoints
  static const String uploadDataset = '/datasets/upload';
  static const String getDatasets = '/datasets';
  static const String deleteDataset = '/datasets/{id}';

  // AI Chat endpoints
  static const String sendMessage = '/ai/chat';
  static const String getChatHistory = '/ai/chat/history';

  // Analytics endpoints
  static const String getAnalytics = '/analytics';
  static const String getHealthDistribution = '/analytics/health-distribution';
  static const String getPerformanceMetrics = '/analytics/performance';
}
