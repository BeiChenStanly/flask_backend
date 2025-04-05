import base64
import io
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image, ImageChops, ImageOps
from torch import nn

app = Flask(__name__)

# 定义与训练一致的模型结构
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 模型加载（带缓存）
@torch.no_grad()
def load_model():
    model = MNISTModel()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

@app.route("/api/predict", methods=["POST","OPTIONS"])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        # 解析请求
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing image data"}), 400
        
        # Base64解码
        header, image_base64 = data["image"].split(",", 1)
        image_bytes = base64.b64decode(image_base64)
        
        # 图像处理
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # 处理透明通道
            if image.mode == "RGBA":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg
                
            # 标准化处理
            image = image.convert("L")                      # 转灰度
            image = ImageOps.fit(image, (28, 28))          # 保持比例缩放
            image = ImageChops.invert(image)               # 颜色反转
            
            # 转换为张量
            pixels = np.array(image, dtype=np.float32) / 255.0
            tensor = torch.tensor(pixels).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
            tensor = (tensor - 0.1307) / 0.3081             # 与训练相同的归一化
        except Exception as e:
            return jsonify({"error": f"Image processing error: {str(e)}"}), 400
        
        # 模型预测
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        predicted = int(torch.argmax(probs))
        confidence = float(probs[predicted])
        
        return jsonify({
            "prediction": predicted,
            "confidence": round(confidence, 4)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel适配
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
else:
    from serverless_wsgi import handle_request
    def lambda_handler(event, context):
        return handle_request(app, event, context)