import 'package:flutter/material.dart';
import '../../widgets/common/header.dart';
import '../../widgets/common/sidebar.dart';
import '../../widgets/common/bottom_nav.dart';
import 'home_screen.dart';
import '../dataset_screen.dart';
import '../analysis_screen.dart';
import '../prediction_screen.dart';
import '../about_screen.dart';

class MainLayout extends StatefulWidget {
  const MainLayout({super.key});

  @override
  State<MainLayout> createState() => _MainLayoutState();
}

class _MainLayoutState extends State<MainLayout> {
  int _currentIndex = 0;

  final List<Widget> _screens = const [
    HomeScreen(),
    DatasetScreen(),
    AnalysisScreen(),
    PredictionScreen(),
    AboutScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomHeader(),
      drawer: Sidebar(
        onItemTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
      ),
      body: SafeArea(
        child: _screens[_currentIndex],
      ),
      bottomNavigationBar: CustomBottomNav(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() => _currentIndex = index);
        },
      ),
    );
  }
}
