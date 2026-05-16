import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../providers/theme_provider.dart';
import '../../providers/language_provider.dart';

class CustomHeader extends StatelessWidget implements PreferredSizeWidget {
  const CustomHeader({super.key});

  @override
  Size get preferredSize => const Size.fromHeight(60);

  @override
  Widget build(BuildContext context) {
    return AppBar(
      // ignore: prefer_const_constructors
      title: Row(
        children: const [
          Icon(
            Icons.eco,
            color: Color(0xFF2E7D32),
          ),
        ],
      ),
      actions: [
        Consumer<ThemeProvider>(
          builder: (context, themeProvider, _) {
            final isDark = themeProvider.themeMode == ThemeMode.dark ||
                (themeProvider.themeMode == ThemeMode.system &&
                    MediaQuery.of(context).platformBrightness == Brightness.dark);
            return IconButton(
              icon: Icon(isDark ? Icons.light_mode : Icons.dark_mode),
              onPressed: () {
                themeProvider.toggleTheme(!isDark);
              },
            );
          },
        ),
        PopupMenuButton<String>(
          icon: const Icon(Icons.menu),
          tooltip: 'Change Language',
          onSelected: (String value) {
            context.read<LanguageProvider>().changeLanguage(value);
          },
          itemBuilder: (BuildContext context) => <PopupMenuEntry<String>>[
            const PopupMenuItem<String>(
              enabled: false,
              child: Text('Change Language', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.grey)),
            ),
            const PopupMenuItem<String>(
              value: 'ar',
              child: Text('العربية'),
            ),
            const PopupMenuItem<String>(
              value: 'en',
              child: Text('English'),
            ),
          ],
        ),
      ],
    );
  }
}
