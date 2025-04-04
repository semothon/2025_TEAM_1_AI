import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt

# S3 URL
url = "https://demo-bucket-605134439665.s3.ap-northeast-2.amazonaws.com/wave.png"

def download_and_show_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 처리

        # 이미지 디코딩
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print("Error: Failed to decode image.")
            return

        # OpenCV는 BGR로 읽으므로 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 이미지 표시
        plt.imshow(image_rgb)
        plt.axis("off")  # 축 숨기기
        plt.show()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# 실행
download_and_show_image(url)