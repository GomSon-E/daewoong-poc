from fastapi import FastAPI, File
import cv2
import numpy as np
import uvicorn
import os
from datetime import datetime

app = FastAPI()

UPLOAD_DIR = "../images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_image(image: bytes = File()):
    # 이미지 읽기
    img_array = np.frombuffer(image, np.uint8)
    cv_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"image_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    # 이미지 저장
    cv2.imwrite(filepath, cv_image)
    
    return "success"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)