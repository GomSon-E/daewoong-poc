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

# ! YOLOv11 모델을 로드
def load_yolo11_model():
    global model

    try:
        model = YOLO('yolo11n-seg.pt')
        print("YOLOv11n 모델 로드 성공! (자동 다운로드)")
        return True
    except Exception as e:
        print(f"자동 다운로드 실패: {e}")
    
    return False

# ! 검출된 객체에 폴리곤 마스크 그리기
def draw_detections(image, results):
    for result in results:
        masks = result.masks
        boxes = result.boxes
        
        if masks is not None and boxes is not None:
            for mask, box in zip(masks.data, boxes):
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                
                if confidence >= 0.5:
                    # 마스크를 폴리곤 좌표로 변환
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
                    
                    # 컨투어 찾기
                    contours, _ = cv2.findContours(
                        (mask_resized > 0.5).astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # 폴리곤 그리기
                    for contour in contours:
                        if len(contour) >= 3:  # 최소 3개 점이 있어야 폴리곤
                            cv2.polylines(image, [contour], True, (0, 255, 0), 2)
                            # 또는 채워진 폴리곤: cv2.fillPoly(image, [contour], (0, 255, 0))
                    
                    # 레이블 추가
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

# ! 검출 결과에서 객체 정보 추출
def extract_detection_info(results):
    detections = []
    
    for result in results:
        masks = result.masks
        boxes = result.boxes
        
        if masks is not None and boxes is not None:
            for mask, box in zip(masks.data, boxes):
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = str(model.names[class_id])
                
                if confidence >= 0.5:
                    # 마스크를 폴리곤 좌표로 변환
                    mask_np = mask.cpu().numpy()
                    # 이미지 크기에 맞게 리사이즈 (실제 구현시 이미지 크기 필요)
                    
                    # 컨투어 찾기
                    contours, _ = cv2.findContours(
                        (mask_np > 0.5).astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    polygons = []
                    for contour in contours:
                        if len(contour) >= 3:
                            # 폴리곤 좌표를 리스트로 변환
                            polygon = contour.reshape(-1, 2).tolist()
                            polygons.append(polygon)
                    
                    detections.append({
                        "class_name": class_name,
                        "confidence": round(confidence, 3),
                        "polygons": polygons,
                        "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist()
                    })
    
    return detections

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

if __name__ == "__main__":
    load_yolo11_model()
    uvicorn.run(app, host="0.0.0.0", port=8001)