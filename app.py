from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from deepface import DeepFace

app = Flask(__name__)

def download_image_from_s3(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        
        if response.headers.get("Content-Type", "").startswith("image"):
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return None, "Failed to decode image"
            return image, None
        else:
            return None, f"Invalid content type: {response.headers.get('Content-Type')}"

    except requests.exceptions.RequestException as e:
        return None, str(e)

    
def resize_to_smallest(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    target_h, target_w = min(h1, h2), min(w1, w2)  # 작은 크기에 맞춤
    img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return img1_resized, img2_resized

def calculate_histogram_similarity(img1, img2, method):
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    hist1 = cv2.calcHist([img1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([img2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])

    hist1 = cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    hist2 = cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    similarity = cv2.compareHist(hist1, hist2, method)

    return similarity

@app.route('/compare_images', methods=['POST'])
def compare_images():
    data = request.get_json()

    if 'url1' not in data or 'url2' not in data:
        return jsonify({"error": "Both image URLs are required"}), 400

    img1, err1 = download_image_from_s3(data['url1'])
    img2, err2 = download_image_from_s3(data['url2'])

    if img1 is None or img2 is None:
        return jsonify({"error": "Invalid image data"}), 400
    
    img1, img2 = resize_to_smallest(img1, img2)
    
    bhattacharyya_dist = calculate_histogram_similarity(img1, img2, cv2.HISTCMP_BHATTACHARYYA)

    # correlation = calculate_histogram_similarity(img1, img2, cv2.HISTCMP_CORREL)

    score = round((1 - bhattacharyya_dist) * 100, 2)

    # "correlation_coefficient": correlation,
    # "bhattacharyya_distance": bhattacharyya_dist,
    return jsonify({
        "score" : score
    }), 200

@app.route('/happiness', methods=['POST'])
def happiness():
    data = request.get_json()

    if 'url' not in data:
        return jsonify({"error": "Image URL is required"}), 400

    img, err = download_image_from_s3(data['url'])

    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        happiness_score = analysis[0]['emotion']['happy']
        # score = 60 + 40 * (happiness_score / 100) ** 1.5
        if happiness_score < 90:
            score = 60 + 20 * (happiness_score / 90) ** 2
        else:
            score = 80 + 19 * ((happiness_score - 90) / 9) ** 3 
        
        score = min(100.0, round(float(score), 2))

        return jsonify({
            "score": score
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
