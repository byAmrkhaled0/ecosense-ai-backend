import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class PlantReportScreen extends StatelessWidget {
  const PlantReportScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Plant Report',
          style: GoogleFonts.inter(fontWeight: FontWeight.w700),
        ),
      ),
      body: const Center(
        child: Text('Plant Report page (placeholder)'),
      ),
    );
  }
}

