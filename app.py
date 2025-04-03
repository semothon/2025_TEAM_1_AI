from flask import Flask, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

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
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Both images are required"}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

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
    if 'file' not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

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
    app.run(debug=True)
