import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class SensorDetailsScreen extends StatelessWidget {
  const SensorDetailsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Sensor Details',
          style: GoogleFonts.inter(fontWeight: FontWeight.w700),
        ),
      ),
      body: const Center(
        child: Text('Sensor Details page (placeholder)'),
      ),
    );
  }
}

