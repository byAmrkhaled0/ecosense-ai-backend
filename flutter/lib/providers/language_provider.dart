import 'package:flutter/material.dart';

class LanguageProvider extends ChangeNotifier {
  Locale _locale = const Locale('ar'); // Default to Arabic

  Locale get locale => _locale;

  void changeLanguage(String languageCode) {
    if (_locale.languageCode != languageCode) {
      _locale = Locale(languageCode);
      notifyListeners();
    }
  }
}
