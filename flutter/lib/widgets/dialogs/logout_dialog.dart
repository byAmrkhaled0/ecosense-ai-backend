import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class LogoutDialog extends StatelessWidget {
  final VoidCallback onConfirm;

  const LogoutDialog({required this.onConfirm, super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return AlertDialog(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      title: Text(
        'Logout?',
        style: GoogleFonts.inter(
          fontSize: 20,
          fontWeight: FontWeight.bold,
          color: theme.textTheme.bodyLarge?.color,
        ),
      ),
      content: Text(
        'Are you sure you want to logout from your account?',
        style: GoogleFonts.inter(
          fontSize: 14,
          color: theme.textTheme.bodyMedium?.color,
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: Text(
            'No, Stay',
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              color: Colors.grey,
            ),
          ),
        ),
        ElevatedButton(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.red.shade700,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
          onPressed: () {
            // Call onConfirm before closing the dialog so the caller's `context.mounted`
            // checks (if any) still succeed.
            Navigator.pop(context);
            onConfirm();
          },
          child: Text(
            'Yes, Logout',
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ],
    );
  }
}
