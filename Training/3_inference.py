import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from train_model import AntiDeepfakeModelPro # ต้อง import class โมเดลมาจากไฟล์ train

# ================= ⚙️ CONFIGURATION =================
MODEL_PATH = 'liveness_model_final.pth'
SCALER_PATH = 'scaler_v2.pkl'
SEQ_LEN = 60
# ====================================================

# 1. Load Model & Scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AntiDeepfakeModelPro().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # สำคัญ! ปิด Dropout/BatchNorm เพื่อใช้งานจริง

scaler = joblib.load(SCALER_PATH)

print("✅ Model & Scaler loaded successfully!")

def pad_sequence(data, max_len):
    length = len(data)
    if length >= max_len:
        return data[:max_len]
    else:
        padding = np.zeros((max_len - length, data.shape[1]))
        return np.vstack((data, padding))

def predict_liveness(session_data):
    """
    session_data: DataFrame หรือ List of Dict ที่มี columns ครบ 25 ตัว
    return: Probability (0.0 - 1.0) --> ยิ่งใกล้ 1 ยิ่ง Real
    """
    # 1. Prepare Columns (ต้องเรียงให้เหมือนตอนเทรนเป๊ะๆ)
    feature_cols = [
        'd_nose_x', 'd_nose_y', 'd_nose_z',
        'd_leye_x', 'd_leye_y', 'd_leye_z',
        'd_reye_x', 'd_reye_y', 'd_reye_z',
        'd_lear_x', 'd_lear_y', 'd_lear_z',
        'd_rear_x', 'd_rear_y', 'd_rear_z',
        'bg_flow_x', 'bg_flow_y', 'fg_flow_x', 'fg_flow_y',
        'gyro_x', 'gyro_y', 'gyro_z',
        'accel_x', 'accel_y', 'accel_z'
    ]
    
    # แปลงเป็น DataFrame ถ้าจำเป็น
    if isinstance(session_data, list):
        df = pd.DataFrame(session_data)
    else:
        df = session_data
        
    # เช็คว่ามีข้อมูลพอไหม
    if len(df) < 3:
        return 0.0 # ข้อมูลน้อยเกินไป ตีเป็น Fake ไว้ก่อนเพื่อความปลอดภัย

    # 2. Extract & Pad
    data = df[feature_cols].values
    padded_data = pad_sequence(data, SEQ_LEN)
    
    # 3. Scale (สำคัญมาก! ต้อง Scale ด้วยตัวเดียวกับที่เทรน)
    N, F = padded_data.shape
    scaled_data = scaler.transform(padded_data) # Scale แบบ 2D
    
    # 4. Convert to Tensor
    inputs = torch.FloatTensor(scaled_data).unsqueeze(0).to(device) # เพิ่ม Batch Dimension (1, 60, 25)
    
    # 5. Predict
    with torch.no_grad():
        output = model(inputs)
        prob = torch.sigmoid(output).item()
        
    return prob

# ================= ทดสอบ (Mock Data) =================
if __name__ == "__main__":
    # ลองโหลดข้อมูลจริงมาเทสสัก 1 session
    print("🧪 Testing with a sample from CSV...")
    df_all = pd.read_csv('output_dataset.csv') # หรือชื่อไฟล์ csv ของคุณ
    
    # สุ่ม Session มาเทส
    random_session = df_all['session'].sample(1).values[0]
    sample_data = df_all[df_all['session'] == random_session]
    true_label = sample_data['label'].iloc[0]
    
    score = predict_liveness(sample_data)
    
    print(f"\n🎯 Session: {random_session}")
    print(f"📝 True Label: {'✅ REAL' if true_label==1 else '❌ FAKE'}")
    print(f"🤖 Model Score: {score:.4f} ({score*100:.2f}%)")
    
    # 🔥 Logic ใหม่: เข้มงวดขึ้น (Strict Threshold)
    REAL_THRESHOLD = 0.80  # ต้องมั่นใจเกิน 80% ถึงให้ผ่าน
    FAKE_THRESHOLD = 0.30  # ถ้าต่ำกว่า 30% คือปลอมแน่ๆ
    
    if score >= REAL_THRESHOLD:
        print("💡 Result: ✅ PASS (Real Person - High Confidence)")
    elif score <= FAKE_THRESHOLD:
        print("💡 Result: ❌ REJECT (Spoofing Detected)")
    else:
        # ช่วง 0.31 - 0.79 (ค่าก้ำกึ่งแบบ 0.58 ของคุณจะตกช่องนี้)
        print("💡 Result: ⚠️ UNSURE (Please try again & move phone more)")
        print("   -> Reason: Motion not clear enough.")