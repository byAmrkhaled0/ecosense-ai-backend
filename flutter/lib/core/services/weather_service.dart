import 'dart:convert';
import 'package:http/http.dart' as http;

class WeatherData {
  final double currentTemp;
  final double maxTempToday;
  final double minTempToday;
  final double maxTempTomorrow;
  final double minTempTomorrow;
  final int weatherCode;

  WeatherData({
    required this.currentTemp,
    required this.maxTempToday,
    required this.minTempToday,
    required this.maxTempTomorrow,
    required this.minTempTomorrow,
    required this.weatherCode,
  });

  factory WeatherData.fromJson(Map<String, dynamic> json) {
    return WeatherData(
      currentTemp: json['current_weather']['temperature'].toDouble(),
      weatherCode: json['current_weather']['weathercode'],
      maxTempToday: json['daily']['temperature_2m_max'][0].toDouble(),
      minTempToday: json['daily']['temperature_2m_min'][0].toDouble(),
      maxTempTomorrow: json['daily']['temperature_2m_max'][1].toDouble(),
      minTempTomorrow: json['daily']['temperature_2m_min'][1].toDouble(),
    );
  }

  String getWeatherDescription() {
    // Basic mapping based on WMO Weather interpretation codes
    if (weatherCode == 0) return 'Clear sky';
    if (weatherCode >= 1 && weatherCode <= 3) return 'Partly cloudy';
    if (weatherCode >= 45 && weatherCode <= 48) return 'Fog';
    if (weatherCode >= 51 && weatherCode <= 55) return 'Drizzle';
    if (weatherCode >= 61 && weatherCode <= 65) return 'Rain';
    if (weatherCode >= 71 && weatherCode <= 75) return 'Snow';
    if (weatherCode >= 80 && weatherCode <= 82) return 'Rain showers';
    if (weatherCode >= 95) return 'Thunderstorm';
    return 'Unknown';
  }

  String getWeatherIcon() {
    if (weatherCode == 0) return '☀️';
    if (weatherCode >= 1 && weatherCode <= 3) return '⛅';
    if (weatherCode >= 45 && weatherCode <= 48) return '🌫️';
    if (weatherCode >= 51 && weatherCode <= 55) return '🌧️';
    if (weatherCode >= 61 && weatherCode <= 65) return '🌧️';
    if (weatherCode >= 71 && weatherCode <= 75) return '❄️';
    if (weatherCode >= 80 && weatherCode <= 82) return '🌦️';
    if (weatherCode >= 95) return '⛈️';
    return '🌡️';
  }
}

class WeatherService {
  // Using Cairo, Egypt coordinates as default for the farm.
  // In a real app, this could come from device location or user profile.
  static const double _latitude = 30.0444;
  static const double _longitude = 31.2357;

  Future<WeatherData?> fetchWeather() async {
    try {
      final url = Uri.parse(
          'https://api.open-meteo.com/v1/forecast?latitude=$_latitude&longitude=$_longitude&current_weather=true&daily=temperature_2m_max,temperature_2m_min,weathercode&timezone=auto');

      final response = await http.get(url);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return WeatherData.fromJson(data);
      } else {
        // ignore: avoid_print
        print('Failed to load weather data: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      // ignore: avoid_print
      print('Error fetching weather: $e');
      return null;
    }
  }
}
