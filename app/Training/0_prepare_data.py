import os
import shutil
import zipfile
import cv2
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm

# ================= ⚙️ CONFIGURATION =================
RAW_DATA_DIR = 'Raw_Download'    
OUTPUT_DIR = 'Dataset'           
DELETE_RAW_AFTER_PROCESS = False 

# 🔥 CROP SETTINGS (V18: Hybrid)
MICRO_TRIM = 2           # ตัดขอบกินเนื้อเข้ามา 2px (ลบเส้นดำ)
MIN_AREA_RATIO = 0.2     # พื้นที่จอต้องใหญ่กว่า 20%
FORCE_CROP_RATIO = 0.15  # ⚠️ ถ้าหาไม่เจอจริงๆ บังคับตัดขอบทิ้ง 15% (แก้ปัญหาภาพ Full Screen)
# ======================================================

def read_image_safe(path):
    try:
        if not os.path.exists(path): return None
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"      ❌ Error reading file: {e}")
        return None

def save_image_safe(path, img):
    try:
        ext = os.path.splitext(path)[1]
        is_success, buffer = cv2.imencode(ext, img)
        if is_success:
            buffer.tofile(path)
            return True
    except Exception as e:
        print(f"      ❌ Error saving file: {e}")
    return False

def find_screen_box(img, aggressive=False):
    """
    ฟังก์ชันหาขอบจอ
    aggressive=True : จะเร่ง Contrast + เชื่อมเส้นแรงขึ้น (สำหรับภาพที่ขอบจาง)
    """
    h_img, w_img = img.shape[:2]
    img_area = h_img * w_img
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if aggressive:
        # 🔥 Boost Contrast: ทำให้ส่วนสว่างสว่างขึ้น ส่วนมืดมืดลง (เพื่อให้เห็นขอบชัด)
        gray = cv2.equalizeHist(gray)
        blur_val = (7, 7)
        canny_thresh = (10, 200) # กวาดทุกเส้นที่เป็นไปได้
        morph_iter = 4           # เชื่อมเส้นที่ขาดหนักๆ
    else:
        # Normal Mode
        blur_val = (5, 5)
        canny_thresh = (30, 100)
        morph_iter = 2

    blurred = cv2.GaussianBlur(gray, blur_val, 0)
    edges = cv2.Canny(blurred, canny_thresh[0], canny_thresh[1])

    # เชื่อมเส้น
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        area = cv2.contourArea(c)
        
        # กรองขนาด
        if area < (img_area * MIN_AREA_RATIO): continue
        if area > (img_area * 0.99): continue 

        # Check Shape
        x, y, w, h = cv2.boundingRect(c)
        aspect = float(w)/h
        if aspect < 0.4 or aspect > 2.5: continue

        # เจอแล้ว!
        return (x, y, w, h)

    return None

def get_best_crop_v18(img):
    h, w = img.shape[:2]

    # 1. 🟢 พยายามครั้งที่ 1: แบบปกติ (แม่นยำ)
    box = find_screen_box(img, aggressive=False)
    if box:
        return box, "Exact Match"

    # 2. 🟡 พยายามครั้งที่ 2: แบบดุดัน (Aggressive)
    # เร่งแสง เร่งเส้น เพื่อหาขอบที่จางๆ
    box = find_screen_box(img, aggressive=True)
    if box:
        return box, "Aggressive Match"

    # 3. 🔴 ทางเลือกสุดท้าย: Force Crop (ตัดขอบทิ้งเลย 15%)
    # ดีกว่าปล่อยภาพ Full Screen ที่มีขอบรกๆ หลุดไป
    margin_w = int(w * FORCE_CROP_RATIO)
    margin_h = int(h * FORCE_CROP_RATIO)
    box = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
    
    return box, "Force Fallback"

def process_session_folder(source_root, target_root, need_crop):
    src_cam = os.path.join(source_root, 'Camera')
    dst_cam = os.path.join(target_root, 'Camera')
    
    if not os.path.exists(src_cam): return False
    os.makedirs(dst_cam, exist_ok=True)
    
    image_files = sorted(glob.glob(os.path.join(src_cam, '*.*')))
    if not image_files: return False

    crop_box = None
    method_name = ""
    
    if need_crop:
        # เช็คภาพกลางๆ
        check_idx = 5 if len(image_files) > 10 else 0
        img = read_image_safe(image_files[check_idx])
        
        if img is not None:
            raw_box, method_name = get_best_crop_v18(img)
            
            # Apply Micro Trim (เฉพาะถ้าไม่ใช่ Force Fallback)
            if "Force" not in method_name:
                x, y, w, h = raw_box
                x += MICRO_TRIM
                y += MICRO_TRIM
                w -= (MICRO_TRIM * 2)
                h -= (MICRO_TRIM * 2)
                crop_box = (x, y, w, h)
            else:
                crop_box = raw_box # Force Crop คำนวณมาดีแล้ว ไม่ต้อง Trim

            print(f"      ✂️ Crop Strategy: {method_name} -> {crop_box}")

    for img_path in image_files:
        fname = os.path.basename(img_path)
        dst_path = os.path.join(dst_cam, fname)

        if crop_box:
            img = read_image_safe(img_path)
            if img is not None:
                x, y, w, h = crop_box
                h_img, w_img = img.shape[:2]
                x = max(0, x); y = max(0, y)
                w = min(w, w_img - x); h = min(h, h_img - y)
                
                cropped_img = img[y:y+h, x:x+w]
                save_image_safe(dst_path, cropped_img)
        else:
            shutil.copy2(img_path, dst_path)
            
    # Copy Sensors
    needed_files = ['Gyroscope.csv', 'Accelerometer.csv']
    sensor_ok = True
    for filename in needed_files:
        src_file = os.path.join(source_root, filename)
        dst_file = os.path.join(target_root, filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            sensor_ok = False 

    if not sensor_ok:
        shutil.rmtree(target_root)
        return False
        
    return True

def organize_dataset():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ ไม่พบโฟลเดอร์ '{RAW_DATA_DIR}'"); return

    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print("📦 Unzipping files...")
    for zip_path in tqdm(list(Path(RAW_DATA_DIR).rglob("*.zip")), desc="Unzipping"):
        extract_to = zip_path.with_suffix('')
        if not extract_to.exists():
            with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(extract_to)

    print("\n🚀 Processing (V18: Hybrid Backup System)...")
    count_processed = 0
    
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        if 'Gyroscope.csv' in files and 'Accelerometer.csv' in files and 'Camera' in dirs:
            
            rel_path = os.path.relpath(root, RAW_DATA_DIR)
            safe_name = rel_path.replace(os.sep, '_').replace(' ', '')
            
            target_path = os.path.join(OUTPUT_DIR, safe_name)
            is_fake_path = 'fake' in root.lower()
            
            print(f"   📂 Saving to: {safe_name}")
            
            success = process_session_folder(root, target_path, need_crop=is_fake_path)
            if success:
                count_processed += 1

    if DELETE_RAW_AFTER_PROCESS:
        shutil.rmtree(RAW_DATA_DIR)
        os.makedirs(RAW_DATA_DIR)

    print(f"\n🎉 Process Complete! Valid Sessions: {count_processed}")
    print(f"   Output: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    organize_dataset()