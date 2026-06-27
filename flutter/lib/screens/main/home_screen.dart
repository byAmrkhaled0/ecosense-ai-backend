import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../providers/home_provider.dart';
import '../../widgets/cards/stat_card.dart';
import '../../core/localization/app_translations.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Provider.of<HomeProvider>(context, listen: false).getPlants();
    });
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Consumer<HomeProvider>(
      builder: (context, homeProvider, _) {
        return SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                AppTranslations.get(context, 'home_dashboard'),
                style: GoogleFonts.inter(
                  fontSize: 14,
                  color: Colors.grey,
                ),
              ),
              const SizedBox(height: 20),
              GridView.count(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                crossAxisCount: 2,
                crossAxisSpacing: 12,
                mainAxisSpacing: 12,
                childAspectRatio: 1.4,
                children: [
                  StatCard(
                    title: AppTranslations.get(context, 'total_plants'),
                    value: '${homeProvider.totalPlants}',
                    dotColor: Colors.grey,
                    bgColor:
                        isDark ? Theme.of(context).cardColor : Colors.white,
                    textColor: Theme.of(context).textTheme.bodyLarge?.color ??
                        Colors.black87,
                  ),
                  StatCard(
                    title: AppTranslations.get(context, 'healthy'),
                    value:
                        '${(homeProvider.healthyPlants / homeProvider.totalPlants * 100).toStringAsFixed(0)}%',
                    dotColor: const Color(0xFF4CAF50),
                    bgColor: isDark
                        ? const Color(0xFF1B3320)
                        : const Color(0xFFE8F5E9),
                    textColor:
                        isDark ? const Color(0xFF81C784) : Colors.black87,
                  ),
                  StatCard(
                    title: AppTranslations.get(context, 'under_stress'),
                    value:
                        '${(homeProvider.stressedPlants / homeProvider.totalPlants * 100).toStringAsFixed(0)}%',
                    dotColor: const Color(0xFFFFA726),
                    bgColor: isDark
                        ? const Color(0xFF332B1B)
                        : const Color(0xFFFFF8E1),
                    textColor:
                        isDark ? const Color(0xFFFFB74D) : Colors.black87,
                  ),
                  StatCard(
                    title: AppTranslations.get(context, 'diseased'),
                    value:
                        '${(homeProvider.diseasedPlants / homeProvider.totalPlants * 100).toStringAsFixed(0)}%',
                    dotColor: const Color(0xFFEF5350),
                    bgColor: isDark
                        ? const Color(0xFF331E1E)
                        : const Color(0xFFFFEBEE),
                    textColor:
                        isDark ? const Color(0xFFE57373) : Colors.black87,
                  ),
                ],
              ),
              const SizedBox(height: 24),
              Text(
                AppTranslations.get(context, 'plant_health_dist'),
                style: GoogleFonts.inter(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Theme.of(context).textTheme.bodyLarge?.color,
                ),
              ),
              const SizedBox(height: 12),
              Container(
                height: 250,
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Theme.of(context).cardColor,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Center(
                  child: Text(
                    AppTranslations.get(context, 'chart_area'),
                    style: TextStyle(
                      color: Colors.grey[500],
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

extension on HomeProvider {
  void getPlants() {}
}
