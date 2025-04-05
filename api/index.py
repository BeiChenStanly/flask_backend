import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image,ImageChops

# --------------- 模型定义（需与训练代码一致）---------------
class ImprovedMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --------------- Flask 初始化 ---------------
app = Flask(__name__)
cors = CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImprovedMNISTModel().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
# --------------- API 路由 ---------------
@app.route("/api/predict", methods=["POST","OPTIONS"])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    # 解析 Base64 图像
    data = request.json
    image_base64 = data["image"].split(",")[1]
    try:
        # 预处理
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((28, 28),Image.LANCZOS)  # 缩放至 28x28
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  #Alpha
        # 再转灰度
        image =background.convert("L")
        image =ImageChops.invert(image)
        pixels = np.array(image, dtype=np.float32) / 255.0
        pixels = (pixels - 0.1307) / 0.3081  # 使用 MNIST 的均值和标准差
        # 转换为 PyTorch 张量并推理
        input_tensor = torch.tensor(pixels, device=device).unsqueeze(0).unsqueeze(0)  # 格式 [1,1,28,28]
        with torch.no_grad():
            output = model(input_tensor)
            prediction = int(torch.argmax(output))
        
        return jsonify({"prediction": prediction,"potencial":output.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)