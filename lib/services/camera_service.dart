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
      enableAudio: false, // เราไม่ได้ใช้เสียง ปิดไว้ช่วยลดขนาดไฟล์ได้
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.jpeg
          : ImageFormatGroup.bgra8888,
    );

    await _controller!.initialize();
  }

  // --- ส่วนที่เพิ่มใหม่สำหรับ Video ---
  Future<void> startRecording() async {
    if (_controller == null || !_controller!.value.isInitialized) {
      return;
    }
    if (_controller!.value.isRecordingVideo) {
      return;
    }
    try {
      await _controller!.startVideoRecording();
    } catch (e) {
      throw Exception('Failed to start recording: $e');
    }
  }

  Future<XFile?> stopRecording() async {
    if (_controller == null || !_controller!.value.isRecordingVideo) {
      return null;
    }
    try {
      return await _controller!.stopVideoRecording();
    } catch (e) {
      throw Exception('Failed to stop recording: $e');
    }
  }
  // ------------------------------------

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
  }
}