# 🛡️ GSync Liveness Detection (Sensor & Video Fusion)

An advanced Anti-Spoofing and Liveness Detection system designed to verify human identity securely. This project leverages a multi-modal approach by fusing **Computer Vision** (Facial Analysis) with **Motion Sensor Data** (Gyroscope & Accelerometer) to distinguish between real humans and presentation attacks (Deepfakes, Replay attacks, Printed photos).

## 🚀 Features

* **Multi-Modal Fusion:** Combines facial features with device movement patterns for high-accuracy detection.
* **Deep Learning Engine:** Powered by a GRU/LSTM-based PyTorch model customized for sequential data.
* **Advanced Feature Extraction:**
    * **Vision:** Facial Landmarks (MediaPipe), Optical Flow, Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR).
    * **Sensors:** Synchronized Gyroscope & Accelerometer data interpolation.
* **Real-time Verification:** Fast processing via FastAPI with immediate feedback.
* **Cross-Platform App:** Mobile application built with Flutter for iOS and Android.

## 🛠️ Tech Stack

### Backend (Server)
* **Language:** Python 3.9+
* **Framework:** FastAPI, Uvicorn
* **AI/ML:** PyTorch, MediaPipe, Scikit-learn, OpenCV
* **Data Processing:** NumPy, Pandas, SciPy

### Frontend (Mobile App)
* **Framework:** Flutter (Dart)
* **Key Packages:** `camera`, `sensors_plus`, `http`

---

## 📂 Project Structure

```bash
GSync-Liveness-App/
├── android/                # Android native configuration
├── lib/                    # Flutter Source Code
│   ├── main.dart           # UI & App Logic
│   └── api_service.dart    # API Integration Logic
├── detection_model/        # Model Artifacts
│   ├── kyc_model_best.pth  # Trained PyTorch Model
│   └── scaler.pkl          # Data Scaler
├── server.py               # Main Python Server
├── requirements.txt        # Python Dependencies
└── README.md               # Documentation

## ⚙️ Installation & Setup

Follow these steps to get the system running locally.

### 1️⃣ Backend Setup (Python Server)

1.  **Navigate to the project directory:**
    ```bash
    cd path/to/project
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file (or use the command below) to install necessary libraries:
    ```bash
    pip install fastapi uvicorn torch numpy opencv-python pandas joblib mediapipe scipy python-multipart scikit-learn
    ```

4.  **Place Model Files:**
    Ensure your trained model files are in the `detection_model/` folder:
    * `detection_model/kyc_model_best.pth`
    * `detection_model/scaler.pkl`

5.  **Run the Server:**
    ```bash
    python server.py
    ```
    * The server will start at `http://0.0.0.0:8000`
    * You should see `🚀 SERVER READY!` in the terminal.

### 2️⃣ Frontend Setup (Flutter App)

1.  **Install Flutter Dependencies:**
    ```bash
    flutter pub get
    ```

2.  **⚠️ Configure API Endpoint (Crucial Step):**
    Open `lib/api_service.dart`. You **must** change `localhost` to your computer's local IP address so the mobile device can connect.
    * **Windows:** Run `ipconfig` in terminal.
    * **Mac/Linux:** Run `ifconfig` in terminal.
    
    ```dart
    // lib/api_service.dart
    
    // ❌ Don't use localhost or 127.0.0.1
    // ✅ Use your LAN IP (e.g., 192.168.1.105, 10.x.x.x)
    static const String baseUrl = '[http://192.168.1.](http://192.168.1.)XXX:8000'; 
    ```

3.  **Android Network Configuration:**
    To allow HTTP connections (local development), ensure `android/app/src/main/AndroidManifest.xml` has:
    ```xml
    <application
        android:label="GSync Liveness"
        android:usesCleartextTraffic="true"  ... >
    ```

4.  **Run the App:**
    ```bash
    flutter run
    ```

---

## 📡 API Usage

### `POST /verify`
The main endpoint to verify liveness.

* **URL:** `http://YOUR_IP:8000/verify`
* **Content-Type:** `multipart/form-data`
* **Body Parameters:**
    * `video`: The video file (`.mp4`, `.temp`).
    * `gyroscope`: CSV file (`x, y, z, seconds_elapsed`).
    * `accelerometer`: CSV file (`x, y, z, seconds_elapsed`).

* **Success Response (JSON):**
    ```json
    {
      "status": "success",
      "result": "REAL",
      "confidence": "MEDIUM",
      "pass_rate": 56.4,
      "score": 0.5817,
      "metrics": {
        "pass_rate": 56.4,
        "mean_score": 0.5817,
        "median_score": 0.9961
      }
    }
    ```

---

## 🔧 Troubleshooting / Common Issues

### ❌ 1. Connection Refused / Connection Timed Out
* **Symptoms:** App spins indefinitely or shows `SocketException`.
* **Fix:**
    1.  **Firewall:** Windows Firewall often blocks port 8000. Turn it off temporarily or add an Inbound Rule for Python.
    2.  **Network:** Phone and Computer must be on the **same Wi-Fi**.
    3.  **IP Address:** Double-check the IP in `api_service.dart`. If using Hotspot, the IP often changes.

### ❌ 2. App Shows "0.0%" Result
* **Symptoms:** Server logs show success, but the app displays 0% confidence.
* **Cause:** JSON key mismatch. The app expects `pass_rate` at the root level, but the server sends it inside `metrics`.
* **Fix:** Update `server.py` to return flat keys:
    ```python
    return {
        "status": "success",
        "pass_rate": pass_rate,  # Add this
        "score": mean_score,     # Add this
        "metrics": { ... }
    }
    ```

### ⚠️ 3. Feature Mismatch Warning (23 vs 25)
* **Symptoms:** Server log shows `⚠️ Feature mismatch: Data has 23, Model needs 25`.
* **Cause:** The model expects EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) features, but they are missing from extraction.
* **Fix:** The server currently auto-pads these with zeros to prevent crashing. for better accuracy, implement EAR/MAR calculation in `server.py`.

---

## 📜 License

This project is open-source and available for educational purposes.