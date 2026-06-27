import 'dart:convert';
import 'package:http/http.dart' as http;

import '../core/network/api_client.dart';

class AIChatRepository {
  AIChatRepository({required this.apiClient});

  final ApiClient apiClient;

  // قم بوضع مفتاح جيميناي الخاص بك هنا. (إنه مجاني تماماً)
  // يمكنك الحصول عليه من: https://aistudio.google.com/app/apikey
  static const String _geminiApiKey = "AIzaSyAeh2NTyyCWZC-YFqPwR4r9eivT9arhgU4";
  
  static const String _geminiEndpoint = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent';

  Future<String> sendMessage(String message) async {
    if (_geminiApiKey == 'YOUR_GEMINI_API_KEY_HERE') {
      return 'عذراً! أنت لم تقم بإضافة مفتاح Gemini الخاص بك في الكود بعد. يرجى الحصول على المفتاح المجاني من https://aistudio.google.com/ ووضعه في ملف ai_chat_repository.dart';
    }

    try {
      final response = await http.post(
        Uri.parse('$_geminiEndpoint?key=$_geminiApiKey'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'contents': [
            {
              'parts': [
                {
                  'text': 'أنت مساعد ذكي متخصص في مجال الزراعة والنباتات، اسمك دكتور فارمر (Dr. Farmer). أجب على هذا السؤال من المزارع: $message'
                }
              ]
            }
          ]
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        final candidates = data['candidates'] as List;
        if (candidates.isNotEmpty) {
          final content = candidates[0]['content']['parts'][0]['text'];
          return content.toString();
        }
        return 'لم أتمكن من إيجاد إجابة.';
      } else {
        return 'عذراً، حدث خطأ أثناء الاتصال بجيميناي: ${response.statusCode} - ${response.body}';
      }
    } catch (e) {
      return 'حدث خطأ: $e';
    }
  }
}
