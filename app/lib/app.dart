import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'models/types.dart';
import 'services/camera_service.dart';
import 'services/sensor_service.dart';
import 'services/api_service.dart';
import 'widgets/gyro_visualizer.dart';

class App extends StatefulWidget {
  const App({super.key});

  @override
  State<App> createState() => _AppState();
}

class _AppState extends State<App> with TickerProviderStateMixin {
  // Services
  final CameraService _cameraService = CameraService();
  final SensorService _sensorService = SensorService();
  final ApiService _apiService = ApiService();

  // State
  LivenessStatus _status = LivenessStatus.idle;
  GyroData _gyroData = GyroData();
  VerificationResult? _result;
  bool _isCameraReady = false;
  String? _error;
  Timer? _verificationTimer;

  // Animation controller for scanning effect
  late AnimationController _scanController;

  @override
  void initState() {
    super.initState();
    _initialize();

    // Initialize Animation Controller for the UI effect
    _scanController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    );

    // Stream listener for the Visualizer
    _sensorService.gyroStream.listen((event) {
      if (mounted) {
        setState(() {
          _gyroData = GyroData(x: event.x, y: event.y, z: event.z);
        });
      }
    });
  }

  Future<void> _initialize() async {
    try {
      await _cameraService.initialize();
      setState(() {
        _isCameraReady = true;
      });
    } catch (e) {
      setState(() {
        _error = "Camera access denied. Please allow camera permissions.";
      });
    }
  }

  @override
  void dispose() {
    _verificationTimer?.cancel();
    _cameraService.dispose();
    // _sensorService.dispose(); // Commented out: Method doesn't exist in SensorService
    _scanController.dispose();
    super.dispose();
  }

  // --- Logic ---

  Future<void> _startVerification() async {
    setState(() {
      _result = null;
      _error = null;
      _status = LivenessStatus.verifying;
    });

    _scanController.repeat(reverse: true);

    try {
      // 1. Start Recording (Video + Sensors)
      _sensorService.startRecording();
      await _cameraService.startRecording();

      // 2. Wait 4 seconds then verify
      _verificationTimer = Timer(const Duration(seconds: 4), _verify);
    } catch (e) {
      _handleError(e);
    }
  }

  Future<void> _verify() async {
    try {
      // 1. Stop Recording
      XFile? videoXFile = await _cameraService.stopRecording();
      var sensorFiles = await _sensorService.stopRecordingAndGetFiles();

      if (videoXFile == null) throw Exception("Video recording failed");

      // 2. Send to API
      var apiResult = await _apiService.verifyLiveness(
        videoFile: File(videoXFile.path),
        gyroFile: sensorFiles['gyro']!,
        accelFile: sensorFiles['accel']!,
      );

      // 3. Process Result
      _scanController.stop();

      if (apiResult['status'] == 'success') {
        bool isReal = apiResult['result'] == 'REAL';
        double confidence = (apiResult['pass_rate'] ?? 0.0).toDouble();
        

        if (mounted) {
          setState(() {
            _status = isReal ? LivenessStatus.success : LivenessStatus.failed;
            
            // Map API result to UI model
            _result = VerificationResult(
              isReal: isReal, // Fixed: changed from isLive to isReal
              confidence: confidence,
              reason: isReal 
                  ? "Motion analysis matches video context." 
                  : "Motion mismatch detected (Potential Replay Attack).",
            );
          });
        }
      } else {
        throw Exception(apiResult['message'] ?? "Unknown Error");
      }
    } catch (e) {
      _handleError(e);
    }
  }

  void _handleError(Object e) {
    _scanController.stop();
    _cameraService.stopRecording(); 
    _sensorService.stopRecordingAndGetFiles();
    
    if (mounted) {
      setState(() {
        _error = e.toString();
        _status = LivenessStatus.failed;
      });
    }
  }

  void _reset() {
    _verificationTimer?.cancel();
    _scanController.stop();
    setState(() {
      _status = LivenessStatus.idle;
      _result = null;
      _error = null;
    });
  }

  // --- UI Construction ---

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                // Header
                const SizedBox(height: 20),
                Text(
                  'Gsync Liveness Detection',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    foreground: Paint()
                      ..shader = const LinearGradient(
                        colors: [Color(0xFF60A5FA), Color(0xFF818CF8)],
                      ).createShader(const Rect.fromLTWH(0.0, 0.0, 200.0, 70.0)),
                  ),
                ),
                const SizedBox(height: 5),
                const Text(
                  'Sensor & Video Fusion Verification',
                  style: TextStyle(
                    fontSize: 14  ,
                    color: Colors.grey,
                  ),
                ),
                const SizedBox(height: 25),

                // Camera Preview Container
                AspectRatio(
                  aspectRatio: 3 / 4,
                  child: Container(
                    decoration: BoxDecoration(
                      color: const Color(0xFF18181B),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(
                        color: Colors.white.withOpacity(0.1),
                        width: 1,
                      ),
                    ),
                    clipBehavior: Clip.antiAlias,
                    child: Stack(
                      children: [
                        // Camera Preview (FIXED: Full Cover)
                        if (_isCameraReady && _cameraService.controller != null)
                           ClipRRect(
                            borderRadius: BorderRadius.circular(24),
                            child: SizedBox.expand(
                              child: FittedBox(
                                fit: BoxFit.cover,
                                child: SizedBox(
                                  // Swap width/height because camera sensor is usually landscape relative to portrait phone
                                  width: _cameraService.controller!.value.previewSize!.height,
                                  height: _cameraService.controller!.value.previewSize!.width,
                                  child: CameraPreview(_cameraService.controller!),
                                ),
                              ),
                            ),
                          )
                        else
                          Container(
                            color: const Color(0xFF18181B),
                            child: const Center(child: CircularProgressIndicator(color: Colors.white24)),
                          ),

                        // Overlay Layers
                        if (_isCameraReady) ...[
                          // Scanning Effect
                          if (_status == LivenessStatus.verifying)
                            _buildScanningEffect(),

                          // Guide Mask
                          _buildGuideMask(),

                          // Sensor Widget (Live)
                          Positioned(
                            bottom: 24,
                            left: 24,
                            right: 24,
                            child: GyroVisualizer(data: _gyroData),
                          ),

                          // Instructions Widget
                          if (_status == LivenessStatus.verifying)
                            Positioned(
                              top: 32,
                              left: 24,
                              right: 24,
                              child: _buildInstructionWidget(),
                            ),
                        ],

                        // State Overlays
                        if (_status == LivenessStatus.idle && !_isCameraReady)
                          _buildInitialOverlay(), // Wait for camera
                        if (_status == LivenessStatus.idle && _isCameraReady)
                          _buildInitialOverlay(),
                        if (_status == LivenessStatus.success)
                          _buildSuccessOverlay(),
                        if (_status == LivenessStatus.failed)
                          _buildFailedOverlay(),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 24),

                // Control Panel
                SizedBox(
                  width: double.infinity,
                  child: Column(
                    children: [
                      // Start Button
                      if (_status == LivenessStatus.idle && _isCameraReady)
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            onPressed: _startVerification,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF3B82F6),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
                            ),
                            child: const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.shield, size: 20),
                                SizedBox(width: 8),
                                Text(
                                  'Run Liveness Check',
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),

                      if (_error != null) ...[
                        const SizedBox(height: 16),
                        Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: const Color(0xFFF43F5E).withOpacity(0.1),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                              color: const Color(0xFFF43F5E).withOpacity(0.3),
                            ),
                          ),
                          child: Row(
                            children: [
                              const Icon(
                                Icons.error_outline,
                                color: Color(0xFFF43F5E),
                                size: 20,
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text(
                                  _error!,
                                  style: const TextStyle(
                                    color: Color(0xFFFCA5A5),
                                    fontSize: 12,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],

                      const SizedBox(height: 16),
                      // Info Chips
                      Row(
                        children: [
                          Expanded(
                            child: _buildInfoChip('DETECTION MODEL', 'GRU', const Color(0xFF60A5FA)),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildInfoChip(String label, String value, Color valueColor) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFF18181B),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: Colors.white.withOpacity(0.05),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 10,
              color: Colors.grey,
              letterSpacing: 1.5,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(
              fontSize: 12,
              color: valueColor,
              fontWeight: FontWeight.bold,
              fontFamily: 'monospace',
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScanningEffect() {
    return AnimatedBuilder(
      animation: _scanController,
      builder: (context, child) {
        return Positioned(
          top: -100 + (_scanController.value * 600),
          left: 0,
          right: 0,
          height: 80,
          child: Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Colors.transparent,
                  const Color(0xFF3B82F6).withOpacity(0.2),
                  Colors.transparent,
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildGuideMask() {
    return Center(
      child: Container(
        width: 256,
        height: 320,
        decoration: BoxDecoration(
          border: Border.all(
            color: Colors.white.withOpacity(0.2),
            width: 2,
            style: BorderStyle.solid,
          ),
          borderRadius: BorderRadius.circular(64),
        ),
      ),
    );
  }

  Widget _buildInstructionWidget() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF2563EB).withOpacity(0.9),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: const Color(0xFF3B82F6).withOpacity(0.3),
        ),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.videocam,
              color: Colors.white,
              size: 20,
            ),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'RECORDING',
                  style: TextStyle(
                    fontSize: 10,
                    color: Color(0xFFDBEAFE),
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.5,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  "Keep device steady for 4s...",
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInitialOverlay() {
    if (_isCameraReady) {
      return const SizedBox.shrink();
    }
    
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF18181B).withOpacity(0.9),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                color: const Color(0xFF3B82F6).withOpacity(0.2),
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.camera_alt,
                color: Color(0xFF3B82F6),
                size: 32,
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'Initializing Camera...',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSuccessOverlay() {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF10B981).withOpacity(0.9),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: const BoxDecoration(
                color: Colors.white,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.check,
                color: Color(0xFF10B981),
                size: 40,
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'Verified Successfully',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Liveness Confirmed',
              style: TextStyle(
                fontSize: 14,
                color: Color(0xFFD1FAE5),
              ),
            ),
            if (_result != null) ...[
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(
                          Icons.verified_user,
                          color: Colors.white,
                          size: 16,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'Confidence: ${(_result!.confidence).toStringAsFixed(1)}%',
                          style: const TextStyle(
                            fontSize: 14,
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    // Fixed: changed reasoning to reason
                    if (_result!.reason.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Text(
                        _result!.reason,
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          fontSize: 11,
                          color: Color(0xFFD1FAE5),
                          fontStyle: FontStyle.italic,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ],
            const SizedBox(height: 32),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _reset,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: const Color(0xFF10B981),
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text(
                  'Finish',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFailedOverlay() {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFFF43F5E).withOpacity(0.9),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Padding(
        padding: const EdgeInsets.all(32.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: const BoxDecoration(
                color: Colors.white,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.close,
                color: Color(0xFFF43F5E),
                size: 40,
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'Verification Failed',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              // Fixed: changed reasoning to reason
              _result != null ? _result!.reason : 'Verification failed. Please try again.',
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 14,
                color: Color(0xFFFECDD3),
              ),
            ),
            if (_result != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Confidence: ${(_result!.confidence).toStringAsFixed(1)}%',
                  style: const TextStyle(
                    fontSize: 12,
                    color: Color(0xFFFECDD3),
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ],
            const SizedBox(height: 32),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _reset,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: const Color(0xFFF43F5E),
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text(
                  'Try Again',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}