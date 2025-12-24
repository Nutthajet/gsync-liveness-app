import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'models/types.dart';
import 'services/camera_service.dart';
import 'services/sensor_service.dart';
import 'services/gemini_service.dart';
import 'widgets/gyro_visualizer.dart';

class App extends StatefulWidget {
  const App({super.key});

  @override
  State<App> createState() => _AppState();
}

class _AppState extends State<App> with TickerProviderStateMixin {
  final CameraService _cameraService = CameraService();
  final SensorService _sensorService = SensorService();
  late GeminiLivenessService _geminiService;

  LivenessStatus _status = LivenessStatus.idle;
  GyroData _gyroData = GyroData();
  Challenge? _currentChallenge;
  VerificationResult? _result;
  bool _isCameraReady = false;
  String? _error;

  Timer? _verificationTimer;

  @override
  void initState() {
    super.initState();
    // Initialize Gemini service - API key should be set via environment variable
    // Run with: flutter run --dart-define=GEMINI_API_KEY=your-key-here
    final apiKey = const String.fromEnvironment(
      'GEMINI_API_KEY',
      defaultValue: '',
    );
    if (apiKey.isEmpty) {
      debugPrint('Warning: GEMINI_API_KEY not set. Liveness verification will fail.');
    }
    _geminiService = GeminiLivenessService(apiKey: apiKey);
  }

  @override
  void dispose() {
    _verificationTimer?.cancel();
    _cameraService.dispose();
    _sensorService.dispose();
    super.dispose();
  }

  Future<void> _startCamera() async {
    try {
      await _cameraService.initialize();
      _sensorService.startListening();
      
      // Update gyro data periodically
      Timer.periodic(const Duration(milliseconds: 100), (timer) {
        if (mounted) {
          setState(() {
            _gyroData = _sensorService.gyroData;
          });
        }
      });

      setState(() {
        _isCameraReady = true;
      });
    } catch (err) {
      setState(() {
        _error = "Camera access denied. Please allow camera permissions to continue.";
      });
    }
  }

  Future<void> _startVerification() async {
    setState(() {
      _result = null;
      _error = null;
      _status = LivenessStatus.initializing;
    });

    // Pick random challenge
    final challenges = Challenge.challenges;
    final challenge = challenges[math.Random().nextInt(challenges.length)];
    setState(() {
      _currentChallenge = challenge;
      _status = LivenessStatus.challengeRequested;
    });

    // Wait for user to perform action
    _verificationTimer = Timer(const Duration(seconds: 4), () async {
      if (!mounted) return;
      
      setState(() {
        _status = LivenessStatus.verifying;
      });

      try {
        final base64Image = await _cameraService.takePictureBase64();
        final verification = await _geminiService.verifyLiveness(
          base64Image,
          _gyroData,
          challenge,
        );

        if (mounted) {
          setState(() {
            _result = verification;
            _status = verification.isLive
                ? LivenessStatus.success
                : LivenessStatus.failed;
          });
        }
      } catch (err) {
        if (mounted) {
          setState(() {
            _error = "Verification service unavailable. Check your internet connection.";
            _status = LivenessStatus.idle;
          });
        }
      }
    });
  }

  void _reset() {
    _verificationTimer?.cancel();
    setState(() {
      _status = LivenessStatus.idle;
      _result = null;
      _currentChallenge = null;
      _error = null;
    });
  }

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
                const Text(
                  'AINU Liveness Pro',
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    foreground: Paint()
                      ..shader = const LinearGradient(
                        colors: [Color(0xFF60A5FA), Color(0xFF818CF8)],
                      ).createShader(const Rect.fromLTWH(0.0, 0.0, 200.0, 70.0)),
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  'AI-Powered Identity Verification System',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey,
                  ),
                ),
                const SizedBox(height: 32),

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
                        // Camera Preview
                        if (_isCameraReady && _cameraService.controller != null)
                          ClipRRect(
                            borderRadius: BorderRadius.circular(24),
                            child: CameraPreview(_cameraService.controller!),
                          )
                        else
                          Container(
                            color: const Color(0xFF18181B),
                          ),

                        // Overlay Layers
                        if (_isCameraReady) ...[
                          // Scanning Effect
                          if (_status == LivenessStatus.verifying)
                            _buildScanningEffect(),

                          // Guide Mask
                          _buildGuideMask(),

                          // Sensor Widget
                          Positioned(
                            bottom: 24,
                            left: 24,
                            right: 24,
                            child: GyroVisualizer(data: _gyroData),
                          ),

                          // Instructions Widget
                          if (_status != LivenessStatus.idle &&
                              _status != LivenessStatus.success &&
                              _status != LivenessStatus.failed)
                            Positioned(
                              top: 32,
                              left: 24,
                              right: 24,
                              child: _buildInstructionWidget(),
                            ),
                        ],

                        // State Overlays
                        if (_status == LivenessStatus.idle && !_isCameraReady)
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
                      if (_isCameraReady && _status == LivenessStatus.idle)
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            onPressed: _startVerification,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF3B82F6),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
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
                      Row(
                        children: [
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.all(16),
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
                                  const Text(
                                    'SESSION ID',
                                    style: TextStyle(
                                      fontSize: 10,
                                      color: Colors.grey,
                                      letterSpacing: 1.5,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  const Text(
                                    '#AX-992384',
                                    style: TextStyle(
                                      fontSize: 12,
                                      color: Colors.white70,
                                      fontFamily: 'monospace',
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.all(16),
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
                                  const Text(
                                    'AI ENGINE',
                                    style: TextStyle(
                                      fontSize: 10,
                                      color: Colors.grey,
                                      letterSpacing: 1.5,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  const Text(
                                    'Gemini 3 Pro',
                                    style: TextStyle(
                                      fontSize: 12,
                                      color: Color(0xFF60A5FA),
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ],
                              ),
                            ),
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

  Widget _buildScanningEffect() {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: const Duration(seconds: 2),
      onEnd: () {
        if (_status == LivenessStatus.verifying && mounted) {
          setState(() {}); // Restart animation
        }
      },
      builder: (context, value, child) {
        return Positioned(
          top: -100 + (value * 600),
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
              Icons.fingerprint,
              color: Colors.white,
              size: 20,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'ACTION REQUIRED',
                  style: TextStyle(
                    fontSize: 10,
                    color: Color(0xFFDBEAFE),
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.5,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  _currentChallenge?.instruction ?? 'Initializing...',
                  style: const TextStyle(
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
              'Secure Identity Check',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              'We need access to your camera and motion sensors to verify that you are a real person.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey,
              ),
            ),
            const SizedBox(height: 32),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _startCamera,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF2563EB),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                ),
                child: const Text(
                  'Start Liveness Test',
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
                child: Text(
                  '"${_result!.reasoning}"',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 12,
                    color: Color(0xFFD1FAE5),
                    fontStyle: FontStyle.italic,
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
            const Text(
              'Potential spoofing detected or movement mismatch.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                color: Color(0xFFFECDD3),
              ),
            ),
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

