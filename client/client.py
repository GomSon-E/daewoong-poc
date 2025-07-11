import requests
from pypylon import pylon
import cv2
import time
import numpy as np
import json

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

def print_detection_results(response_data):
    """검출 결과를 콘솔에 출력"""
    try:
        if response_data.get('status') == 'success':
            detections = response_data.get('detections', [])
            print(f"\n=== 객체 검출 결과 ===")
            print(f"총 검출된 객체: {len(detections)}개")
            
            if detections:
                print("\n검출된 객체 목록:")
                for i, detection in enumerate(detections, 1):
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    center = detection['center']
                    print(f"  {i}. {class_name} (신뢰도: {confidence:.2f}, 중심: {center})")
            else:
                print("검출된 객체가 없습니다.")
                
            files = response_data.get('files', {})
            print(f"\n저장된 파일:")
            print(f"  원본 이미지: {files.get('original', 'N/A')}")
            print(f"  결과 이미지: {files.get('result', 'N/A')}")
            print(f"  JSON 파일: {files.get('json', 'N/A')}")
            print("=" * 25)
        else:
            print(f"오류: {response_data.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"응답 처리 중 오류: {e}")

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
            
            print(f"전송 및 처리 시간: {transfer_time:.1f}ms")
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    print_detection_results(response_data)
                except json.JSONDecodeError:
                    print(f"Upload successful: {response.text}")
            else:
                print(f"Upload failed: {response.status_code}")
                print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.Close()

def show_menu():
    print("\n=== YOLOv11 객체 검출 메뉴 ===")
    print("1. 흑백 이미지 전송 및 검출")
    print("2. 컬러 이미지 전송 및 검출")
    print("3. 서버 상태 확인")
    print("4. 종료")
    print("=============================")

def check_server_health():
    """서버 상태 확인"""
    try:
        response = requests.get('http://192.168.0.4:8001/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"서버 상태: {data.get('status', 'unknown')}")
            print(f"모델: {data.get('model', 'unknown')}")
        else:
            print(f"서버 응답 오류: {response.status_code}")
    except Exception as e:
        print(f"서버 상태 확인 실패: {e}")

def main():
    # 서버 연결 테스트
    print("서버 연결을 확인하는 중...")
    if not server_test():
        print("서버에 연결할 수 없습니다. 프로그램을 종료합니다.")
        return
    
    while True:
        show_menu()
        
        try:
            choice = int(input("선택하세요 (1-4): "))
            
            if choice == 1:
                print("\n흑백 이미지 전송 및 객체 검출을 시작합니다...")
                capture_and_send(1)
                
            elif choice == 2:
                print("\n컬러 이미지 전송 및 객체 검출을 시작합니다...")
                capture_and_send(2)
                
            elif choice == 3:
                print("\n서버 상태를 확인합니다...")
                check_server_health()
                
            elif choice == 4:
                print("프로그램을 종료합니다.")
                break
                
            else:
                print("잘못된 선택입니다. 1-4 사이의 숫자를 입력하세요.")
                
        except ValueError:
            print("숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()