import 'package:flutter/material.dart';
import '../../config/theme.dart';
import '../../core/localization/app_translations.dart';

class CustomBottomNav extends StatelessWidget {
  final int currentIndex;
  final Function(int) onTap;

  const CustomBottomNav({
    required this.currentIndex,
    required this.onTap,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return BottomNavigationBar(
      currentIndex: currentIndex,
      onTap: onTap,
      type: BottomNavigationBarType.fixed,
      backgroundColor: theme.cardColor,
      selectedItemColor: AppTheme.primaryGreen,
      unselectedItemColor: Colors.grey,
      items: [
        BottomNavigationBarItem(
          icon: const Icon(Icons.home_outlined),
          activeIcon: const Icon(Icons.home),
          label: AppTranslations.get(context, 'nav_home'),
        ),
        BottomNavigationBarItem(
          icon: const Icon(Icons.dataset_outlined),
          activeIcon: const Icon(Icons.dataset),
          label: AppTranslations.get(context, 'nav_datasets'),
        ),
        BottomNavigationBarItem(
          icon: const Icon(Icons.analytics_outlined),
          activeIcon: const Icon(Icons.analytics),
          label: AppTranslations.get(context, 'nav_analytics'),
        ),
        BottomNavigationBarItem(
          icon: const Icon(Icons.psychology_outlined),
          activeIcon: const Icon(Icons.psychology),
          label: AppTranslations.get(context, 'nav_predictions'),
        ),
        BottomNavigationBarItem(
          icon: const Icon(Icons.info_outlined),
          activeIcon: const Icon(Icons.info),
          label: AppTranslations.get(context, 'nav_about'),
        ),
      ],
    );
  }
}