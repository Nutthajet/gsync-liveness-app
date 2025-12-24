import 'dart:typed_data';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'dart:io';

class CameraService {
  CameraController? _controller;
  List<CameraDescription>? _cameras;

  CameraController? get controller => _controller;

  Future<void> initialize() async {
    _cameras = await availableCameras();
    if (_cameras == null || _cameras!.isEmpty) {
      throw Exception('No cameras available');
    }

    // Use front camera
    final frontCamera = _cameras!.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => _cameras!.first,
    );

    _controller = CameraController(
      frontCamera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.jpeg
          : ImageFormatGroup.bgra8888,
    );

    await _controller!.initialize();
  }

  Future<Uint8List> takePicture() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      throw Exception('Camera not initialized');
    }

    try {
      final image = await _controller!.takePicture();
      final bytes = await image.readAsBytes();
      return bytes;
    } catch (e) {
      throw Exception('Failed to take picture: $e');
    }
  }

  Future<String> takePictureBase64() async {
    final bytes = await takePicture();
    return base64Encode(bytes);
  }

  void dispose() {
    _controller?.dispose();
    _controller = null;
  }

  bool get isInitialized => _controller?.value.isInitialized ?? false;
}

String base64Encode(List<int> bytes) {
  return base64.encode(bytes);
}

