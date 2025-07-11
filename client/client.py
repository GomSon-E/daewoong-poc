import requests
from pypylon import pylon
import cv2
import time
import numpy as np

def server_test():
    try:
        response = requests.get('http://192.168.0.4:8001/', timeout=10)
        print(f"Server is running: {response.status_code}")
        return True
    except Exception as e:
        print(f"Server connection failed: {e}")
        return False

def auto_white_balance(image):
    """자동 화이트 밸런스 (Gray World 알고리즘)"""
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])
    
    gray_value = (avg_b + avg_g + avg_r) / 3
    
    image[:, :, 0] = image[:, :, 0] * (gray_value / avg_b)
    image[:, :, 1] = image[:, :, 1] * (gray_value / avg_g)
    image[:, :, 2] = image[:, :, 2] * (gray_value / avg_r)
    
    # 값이 255를 넘지 않도록 클리핑
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def capture_and_send(image_type):
    """이미지 촬영 및 전송"""
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    
    try:
        # 이미지 타입에 따라 픽셀 포맷 설정
        if image_type == 1:  # 흑백
            camera.PixelFormat.SetValue("Mono8")
            print("흑백 이미지로 처리합니다.")
        else:  # 컬러
            camera.PixelFormat.SetValue("BayerRG8")
            print("컬러 이미지로 처리합니다.")
        
        # 이미지 촬영
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
        if grabResult.GrabSucceeded():
            image = grabResult.Array
            print(f"원본 이미지 크기: {image.shape}")
            
            # 컬러 이미지인 경우 베이어 변환 및 화이트 밸런스 조정
            if image_type == 2:
                # BayerBG → BGR 변환
                image = cv2.cvtColor(image, cv2.COLOR_BayerBG2BGR)
                
                # 자동 화이트 밸런스 적용
                image = auto_white_balance(image)
            
            print(f"최종 이미지 크기: {image.shape}")
            
            # 이미지를 JPEG로 인코딩
            _, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 전송 시작 시간 기록 (밀리세컨드)
            start_time = time.time() * 1000
            
            # 서버로 전송
            files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
            response = requests.post('http://192.168.0.4:8001/upload', files=files, timeout=30)
            
            # 응답 받은 시간 기록 (밀리세컨드)
            end_time = time.time() * 1000
            transfer_time = end_time - start_time
            
            print(f"전송 시간: {transfer_time:.1f}ms")
            
            if response.status_code == 200:
                print(f"Upload successful: {response.text}")
            else:
                print(f"Upload failed: {response.status_code}")
                print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.Close()

def show_menu():
    print("\n=== 이미지 전송 메뉴 ===")
    print("1. 흑백 이미지 전송")
    print("2. 컬러 이미지 전송")
    print("3. 종료")
    print("====================")

def main():
    # 서버 연결 테스트
    print("서버 연결을 확인하는 중...")
    if not server_test():
        print("서버에 연결할 수 없습니다. 프로그램을 종료합니다.")
        return
    
    while True:
        show_menu()
        
        try:
            choice = int(input("선택하세요 (1-3): "))
            
            if choice == 1:
                print("\n흑백 이미지 전송을 시작합니다...")
                capture_and_send(1)
                
            elif choice == 2:
                print("\n컬러 이미지 전송을 시작합니다...")
                capture_and_send(2)
                
            elif choice == 3:
                print("프로그램을 종료합니다.")
                break
                
            else:
                print("잘못된 선택입니다. 1-3 사이의 숫자를 입력하세요.")
                
        except ValueError:
            print("숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()