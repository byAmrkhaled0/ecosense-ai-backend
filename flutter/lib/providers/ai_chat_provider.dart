import 'package:flutter/material.dart';

import '../repositories/ai_chat_repository.dart';

class AIChatProvider extends ChangeNotifier {
  AIChatProvider({required this.aiChatRepository});

  final AIChatRepository aiChatRepository;

  final List<ChatMessage> _messages = [];
  bool _isLoading = false;
  String? _error;
  int _remainingAttempts = 5;

  List<ChatMessage> get messages => _messages;
  bool get isLoading => _isLoading;
  String? get error => _error;
  int get remainingAttempts => _remainingAttempts;

  Future<void> sendMessage(String message) async {
    if (_remainingAttempts <= 0) {
      _error = 'لقد استنفدت جميع المحاولات المجانية.';
      notifyListeners();
      return;
    }

    // Add user message
    _messages.add(ChatMessage(
      text: message,
      isUser: true,
      timestamp: DateTime.now(),
    ));
    notifyListeners();

    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      final reply = await aiChatRepository.sendMessage(message);
      
      _messages.add(ChatMessage(
        text: reply,
        isUser: false,
        timestamp: DateTime.now(),
      ));
      _remainingAttempts--; // Decrease attempt
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void clearMessages() {
    _messages.clear();
    notifyListeners();
  }
}

class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.text,
    required this.isUser,
    required this.timestamp,
  });
}
