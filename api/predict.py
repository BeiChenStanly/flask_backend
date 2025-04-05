import base64
import io
import numpy as np
from flask import Blueprint, request, jsonify
from PIL import Image, ImageChops

predict_route = Blueprint('predict', __name__)

# 加载模型参数
model_weights = np.load('model/model_weights.npy', allow_pickle=True).item()
W, b = model_weights['W'], model_weights['b']

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

@predict_route.route('/predict', methods=['POST','OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'CORS preflight response'}), 200
    try:
        data = request.json
        # 图像预处理
        image_base64 = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((28, 28), Image.LANCZOS)
        
        if image.mode == 'RGBA':
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        
        image = image.convert("L")
        image = ImageChops.invert(image)
        pixels = np.array(image, dtype=np.float32).reshape(1, 784) / 255.0
        
        # 预测
        logits = np.dot(pixels, W) + b
        probabilities = softmax(logits)[0]
        
        return jsonify({
            "prediction": int(np.argmax(probabilities)),
            "probabilities": probabilities.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400