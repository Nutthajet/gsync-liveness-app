# AINU Liveness Pro - Flutter App

AI-Powered Identity Verification System using Flutter. This app uses camera frames and motion sensor data to verify user liveness and prevent spoofing, powered by Google's Gemini AI.

## Features

- 📸 Real-time camera preview
- 📱 Motion sensor integration (gyroscope, accelerometer, magnetometer)
- 🤖 AI-powered liveness detection using Gemini 3
- 🎯 Challenge-based verification (tilt/rotate phone)
- 📊 Real-time sensor visualization
- ✨ Beautiful, modern UI

## Prerequisites

- Flutter SDK (>=3.0.0)
- Dart SDK
- Android Studio / Xcode (for mobile development)
- Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

## Setup

1. **Install Flutter dependencies:**
   ```bash
   flutter pub get
   ```

2. **Configure API Key:**
   
   You have several options to set your Gemini API key:

   **Option 1: Environment variable (Recommended)**
   ```bash
   # For development
   export GEMINI_API_KEY="your-api-key-here"
   
   # Then run
   flutter run --dart-define=GEMINI_API_KEY=$GEMINI_API_KEY
   ```

   **Option 2: Modify code directly (Not recommended for production)**
   Edit `lib/app.dart` and replace the API key initialization:
   ```dart
   _geminiService = GeminiLivenessService(apiKey: 'your-api-key-here');
   ```

   **Option 3: Use flutter_dotenv package**
   Create a `.env` file in the root directory:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```
   Then load it in your code using the `flutter_dotenv` package.

3. **Platform-specific setup:**

   **Android:**
   - Permissions are already configured in `android/app/src/main/AndroidManifest.xml`
   - Minimum SDK version: 21

   **iOS:**
   - Permissions are already configured in `ios/Runner/Info.plist`
   - Minimum iOS version: 12.0

## Running the App

```bash
# Run on connected device/emulator
flutter run

# Run with API key from environment
flutter run --dart-define=GEMINI_API_KEY=your-api-key-here

# Build release APK (Android)
flutter build apk --release

# Build release IPA (iOS)
flutter build ios --release
```

## Project Structure

```
lib/
├── main.dart              # App entry point
├── app.dart               # Main app widget with state management
├── models/
│   └── types.dart        # Data models (GyroData, Challenge, etc.)
├── services/
│   ├── camera_service.dart    # Camera initialization and capture
│   ├── sensor_service.dart    # Motion sensor data collection
│   └── gemini_service.dart    # Gemini API integration
└── widgets/
    └── gyro_visualizer.dart   # Sensor data visualization widget
```

## How It Works

1. **Camera Initialization**: User grants camera permission and front camera starts
2. **Sensor Listening**: Motion sensors (accelerometer, gyroscope, magnetometer) start tracking device orientation
3. **Challenge Selection**: Random challenge is selected (tilt up/down, rotate left/right)
4. **Verification**: After 4 seconds, camera frame and sensor data are captured
5. **AI Analysis**: Gemini AI analyzes the image and sensor data to verify liveness
6. **Result**: User receives verification result (success/failed) with reasoning

## Technologies Used

- **Flutter**: Cross-platform UI framework
- **camera**: Camera access and preview
- **sensors_plus**: Motion sensor data
- **http**: API communication with Gemini
- **Google Gemini AI**: Liveness detection and analysis

## Notes

- The app requires camera and motion sensor permissions
- Internet connection is required for AI verification
- Sensor data is calculated from accelerometer, gyroscope, and magnetometer fusion
- The app is designed for portrait orientation

## Troubleshooting

- **Camera not working**: Ensure camera permissions are granted in device settings
- **Sensors not working**: Some emulators don't support sensors - test on a real device
- **API errors**: Verify your Gemini API key is correct and has proper permissions
- **Build errors**: Run `flutter clean` and `flutter pub get`

## License

This project is private and proprietary.
