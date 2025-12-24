import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/types.dart';

class GeminiLivenessService {
  final String apiKey;

  GeminiLivenessService({required this.apiKey});

  Future<VerificationResult> verifyLiveness(
    String base64Image,
    GyroData gyroData,
    Challenge challenge,
  ) async {
    const model = 'gemini-3-flash-preview';
    final url = Uri.parse(
      'https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=$apiKey',
    );

    final prompt = '''
      Perform liveness detection.
      
      User Challenge: ${challenge.instruction}
      Target Motion: ${challenge.expectedMovement}
      
      Current Device Sensor Data (Gyroscope):
      - Alpha (Z-axis): ${gyroData.alpha?.toStringAsFixed(2) ?? 'N/A'}
      - Beta (X-axis/Tilt): ${gyroData.beta?.toStringAsFixed(2) ?? 'N/A'}
      - Gamma (Y-axis/Roll): ${gyroData.gamma?.toStringAsFixed(2) ?? 'N/A'}

      Analyze the provided image and the sensor data. 
      Determine if the visual movement in the image matches the physical motion suggested by the gyroscope.
      A live person will show depth, natural movement, and synchronization with the device sensors.
      A static photo or a screen re-broadcast will lack depth or have mismatched sensor data.
      
      Respond with a JSON object containing:
      - isLive: boolean
      - confidence: number (0-1)
      - reasoning: string
    ''';

    final requestBody = {
      'contents': [
        {
          'parts': [
            {'text': prompt},
            {
              'inline_data': {
                'mime_type': 'image/jpeg',
                'data': base64Image,
              }
            }
          ]
        }
      ],
      'generationConfig': {
        'response_mime_type': 'application/json',
        'response_schema': {
          'type': 'object',
          'properties': {
            'isLive': {'type': 'boolean'},
            'confidence': {'type': 'number'},
            'reasoning': {'type': 'string'}
          },
          'required': ['isLive', 'confidence', 'reasoning']
        }
      }
    };

    try {
      final response = await http.post(
        url,
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode(requestBody),
      );

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        final text = jsonResponse['candidates']?[0]?['content']?['parts']?[0]?['text'] as String?;
        
        if (text != null) {
          final result = jsonDecode(text) as Map<String, dynamic>;
          return VerificationResult.fromJson(result);
        }
      }

      throw Exception('Failed to verify liveness: ${response.statusCode}');
    } catch (error) {
      throw Exception('Liveness verification failed: $error');
    }
  }
}

