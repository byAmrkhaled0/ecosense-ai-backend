import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../core/services/weather_service.dart';
// ignore: unused_import
import '../../core/localization/app_translations.dart';

class WeatherCard extends StatefulWidget {
  const WeatherCard({super.key});

  @override
  State<WeatherCard> createState() => _WeatherCardState();
}

class _WeatherCardState extends State<WeatherCard> {
  final WeatherService _weatherService = WeatherService();
  WeatherData? _weatherData;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchWeather();
  }

  Future<void> _fetchWeather() async {
    final data = await _weatherService.fetchWeather();
    if (mounted) {
      setState(() {
        _weatherData = data;
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    if (_isLoading) {
      return Container(
        height: 120,
        decoration: BoxDecoration(
          color: Theme.of(context).cardColor,
          borderRadius: BorderRadius.circular(16),
        ),
        child: const Center(child: CircularProgressIndicator()),
      );
    }

    if (_weatherData == null) {
      return const SizedBox.shrink();
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: isDark
              ? [const Color(0xFF1E3C72), const Color(0xFF2A5298)]
              : [const Color(0xFF4CA1AF), const Color(0xFFC4E0E5)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            // ignore: deprecated_member_use
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Farm Weather Forecast',
                style: GoogleFonts.inter(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: isDark ? Colors.white : Colors.black87,
                ),
              ),
              Icon(
                Icons.location_on,
                color: isDark ? Colors.white70 : Colors.black54,
                size: 18,
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildForecastDay(
                context,
                'Today',
                _weatherData!.maxTempToday,
                _weatherData!.minTempToday,
                _weatherData!.getWeatherIcon(),
                isDark,
              ),
              Container(
                width: 1,
                height: 40,
                color: isDark ? Colors.white30 : Colors.black26,
              ),
              _buildForecastDay(
                context,
                'Tomorrow',
                _weatherData!.maxTempTomorrow,
                _weatherData!.minTempTomorrow,
                '🌤️', // Simple fallback icon for tomorrow
                isDark,
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Current: ${_weatherData!.currentTemp}°C - ${_weatherData!.getWeatherDescription()}',
            style: GoogleFonts.inter(
              fontSize: 14,
              color: isDark ? Colors.white70 : Colors.black87,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildForecastDay(
    BuildContext context,
    String day,
    double maxTemp,
    double minTemp,
    String icon,
    bool isDark,
  ) {
    return Column(
      children: [
        Text(
          day,
          style: GoogleFonts.inter(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: isDark ? Colors.white : Colors.black87,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          icon,
          style: const TextStyle(fontSize: 24),
        ),
        const SizedBox(height: 4),
        Text(
          '${maxTemp.toStringAsFixed(1)}° / ${minTemp.toStringAsFixed(1)}°',
          style: GoogleFonts.inter(
            fontSize: 14,
            color: isDark ? Colors.white70 : Colors.black87,
          ),
        ),
      ],
    );
  }
}
