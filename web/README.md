# GSync Liveness Detection Web Client 🎥🛡️

![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)

GSync Liveness is a secure, AI-powered biometric verification system running directly in the browser. It combines video analysis with real-time motion sensor data (Gyroscope & Accelerometer) to prevent spoofing attacks (such as holding up a photo or playing a video) during identity verification.

## ✨ Features

- **Real-time Camera Streaming:** Automatic camera initialization with user-facing mode priority.
- **Sensor Synchronization:** Captures Gyroscope and Accelerometer data synchronized with video recording.
- **Mock Data Fallback:** Smart fallback mechanism to generate dummy sensor data when testing on desktop environments (where sensors are absent).
- **Interactive UI:**
  - Modern, dark-themed UI with Tailwind CSS.
  - Real-time status indicators (Idle, Ready, Scanning, Processing).
  - Visual feedback for verification results (Success/Failed) with confidence scores.
- **Cross-Platform:** Optimized for mobile browsers (iOS/Android) with specific permission handling for iOS motion sensors.

## 🚀 Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- A backend server running the GSync verification API (Python/Flask)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MinSkoM/gsync-liveness-app.git
   cd gsync-liveness-web