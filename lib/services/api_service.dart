import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ApiService {
  // ⚠️ TODO: Change this to your computer's IP address
  static const String baseUrl = 'http://YOUR_IPV4_ADDRESS:8000';

  Future<Map<String, dynamic>> verifyLiveness({
    required File videoFile,
    required File gyroFile,
    required File accelFile,
  }) async {
    var uri = Uri.parse('$baseUrl/verify');
    var request = http.MultipartRequest('POST', uri);

    // แนบไฟล์
    request.files.add(await http.MultipartFile.fromPath('video', videoFile.path));
    request.files.add(await http.MultipartFile.fromPath('gyroscope', gyroFile.path));
    request.files.add(await http.MultipartFile.fromPath('accelerometer', accelFile.path));

    try {
      print("🚀 Sending to AI Server...");
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        print("✅ AI Result: $result");
        return result;
      } else {
        return {"status": "error", "message": "Server error: ${response.statusCode}"};
      }
    } catch (e) {
      return {"status": "error", "message": "Connection failed: $e"};
    }
  }
}