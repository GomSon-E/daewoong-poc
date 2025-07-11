from fastapi import FastAPI, File
import cv2
import numpy as np
import uvicorn
import os
from datetime import datetime
from ultralytics import YOLO
import json
import torch
import urllib.request
import zipfile

app = FastAPI()

# 디렉토리 설정
UPLOAD_DIR = "../images"
RESULTS_DIR = "../results"
MODELS_DIR = "../models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 모델 전역 변수
model = None

def download_yolo11_model():
    """YOLOv11 모델을 직접 다운로드하는 함수"""
    model_path = os.path.join(MODELS_DIR, "yolo11n.pt")
    
    if os.path.exists(model_path):
        print(f"모델 파일이 이미 존재합니다: {model_path}")
        return model_path
    
    try:
        print("YOLOv11n 모델을 다운로드하는 중...")
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
        urllib.request.urlretrieve(url, model_path)
        print(f"모델 다운로드 완료: {model_path}")
        return model_path
    except Exception as e:
        print(f"모델 다운로드 실패: {e}")
        return None

def load_yolo11_model():
    """YOLOv11 모델을 로드하는 함수"""
    global model
    print("YOLOv11 모델 로딩 시작...")
    
    # 1. 먼저 ultralytics에서 자동 다운로드 시도
    try:
        print("방법 1: ultralytics 자동 다운로드 시도...")
        model = YOLO('yolo11n.pt')
        print("YOLOv11n 모델 로드 성공! (자동 다운로드)")
        return True
    except Exception as e:
        print(f"자동 다운로드 실패: {e}")
    
    # 2. 수동 다운로드 후 로드 시도
    try:
        print("방법 2: 수동 다운로드 시도...")
        model_path = download_yolo11_model()
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
            print("YOLOv11n 모델 로드 성공! (수동 다운로드)")
            return True
    except Exception as e:
        print(f"수동 다운로드 후 로드 실패: {e}")
    
    # 3. 로컬에서 다른 경로 시도
    try:
        print("방법 3: 다른 모델명 시도...")
        alternative_names = ['yolov11n.pt', 'yolo11n.yaml']
        for name in alternative_names:
            try:
                model = YOLO(name)
                print(f"{name} 모델 로드 성공!")
                return True
            except:
                continue
    except Exception as e:
        print(f"대체 모델명 시도 실패: {e}")
    
    print("모든 YOLOv11 로드 방법 실패")
    return False

def draw_detections(image, results):
    """검출된 객체에 바운딩 박스와 레이블을 그리는 함수"""
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 신뢰도와 클래스
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # 신뢰도가 0.5 이상인 경우만 표시
                if confidence >= 0.5:
                    # 바운딩 박스 그리기
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 레이블 텍스트
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # 텍스트 배경 그리기
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                    
                    # 텍스트 그리기
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image

def extract_detection_info(results):
    """검출 결과에서 정보를 추출하는 함수"""
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # NumPy 타입을 Python 기본 타입으로 변환
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Python int로 명시적 변환
                
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = str(model.names[class_id])  # 문자열로 명시적 변환
                
                # 신뢰도가 0.5 이상인 경우만 포함
                if confidence >= 0.5:
                    center_x = int((x1 + x2) // 2)
                    center_y = int((y1 + y2) // 2)
                    
                    detections.append({
                        "class_name": class_name,
                        "confidence": round(confidence, 3),  # 소수점 3자리로 제한
                        "bbox": [x1, y1, x2, y2],
                        "center": [center_x, center_y]
                    })
    
    return detections

# 서버 시작시 YOLOv11 모델 로드
print("=" * 50)
print("YOLOv11 Object Detection Server 시작")
print("=" * 50)

if not load_yolo11_model():
    print("❌ 경고: YOLOv11 모델을 로드할 수 없습니다!")
    print("다음 명령어를 시도해보세요:")
    print("1. pip install --upgrade ultralytics")
    print("2. pip install torch torchvision")
    print("3. python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\"")
    exit(1)
else:
    print("✅ YOLOv11 모델 준비 완료!")

@app.get("/")
async def root():
    return {"message": "YOLOv11 Object Detection Server is running", "model_loaded": model is not None}

@app.post("/upload")
async def upload_image(image: bytes = File()):
    try:
        if model is None:
            return {"error": "YOLOv11 model is not loaded"}
        
        # 이미지 읽기
        img_array = np.frombuffer(image, np.uint8)
        cv_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            return {"error": "Invalid image format"}
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        original_filename = f"original_{timestamp}.jpg"
        result_filename = f"result_{timestamp}.jpg"
        json_filename = f"detection_{timestamp}.json"
        
        original_filepath = os.path.join(UPLOAD_DIR, original_filename)
        result_filepath = os.path.join(RESULTS_DIR, result_filename)
        json_filepath = os.path.join(RESULTS_DIR, json_filename)
        
        # 원본 이미지 저장
        cv2.imwrite(original_filepath, cv_image)
        
        # YOLOv11로 객체 검출 수행
        print(f"YOLOv11 객체 검출 시작: {original_filename}")
        results = model(cv_image)
        
        # 검출 정보 추출
        detections = extract_detection_info(results)
        
        # 결과 이미지에 바운딩 박스 그리기
        result_image = cv_image.copy()
        result_image = draw_detections(result_image, results)
        
        # 결과 이미지 저장
        cv2.imwrite(result_filepath, result_image)
        
        # 검출 결과를 JSON으로 저장
        detection_data = {
            "timestamp": str(timestamp),
            "model": "YOLOv11n",
            "original_image": str(original_filename),
            "result_image": str(result_filename),
            "total_detections": int(len(detections)),
            "detections": detections
        }
        
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(detection_data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as json_error:
            print(f"JSON 저장 오류: {json_error}")
            # JSON 저장이 실패해도 계속 진행
        
        # 콘솔에 검출 결과 출력
        print(f"YOLOv11 검출 완료: {len(detections)}개 객체 발견")
        for detection in detections:
            print(f"  - {detection['class_name']}: {detection['confidence']:.2f}")
        
        # 응답 반환
        response_data = {
            "status": "success",
            "message": f"YOLOv11: Successfully processed image with {len(detections)} detections",
            "model": "YOLOv11n",
            "detections": detections,
            "files": {
                "original": str(original_filename),
                "result": str(result_filename),
                "json": str(json_filename)
            }
        }
        
        return response_data
        
    except Exception as e:
        print(f"YOLOv11 처리 중 오류: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model": "YOLOv11n",
        "model_loaded": model is not None,
        "torch_version": torch.__version__
    }

@app.get("/model-info")
async def model_info():
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        return {
            "model_type": "YOLOv11n",
            "classes": list(model.names.values()),
            "num_classes": len(model.names)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)