import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 超参数
LEARNING_RATE = 0.01
EPOCHS = 50
BATCH_SIZE = 256

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# 加载MNIST数据
print("Loading data...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X / 255.0
y = y.astype(np.int32)

# 转换为one-hot编码
y_onehot = np.zeros((y.size, 10))
y_onehot[np.arange(y.size), y] = 1

# 初始化参数
np.random.seed(42)
W = np.random.randn(784, 10) * 0.01
b = np.zeros(10)

# 训练循环
print("Training...")
for epoch in range(EPOCHS):
    for i in range(0, X.shape[0], BATCH_SIZE):
        X_batch = X[i:i+BATCH_SIZE]
        y_batch = y_onehot[i:i+BATCH_SIZE]
        
        # 前向传播
        logits = np.dot(X_batch, W) + b
        probs = softmax(logits)
        
        # 计算梯度
        grad = (probs - y_batch) / X_batch.shape[0]
        dW = np.dot(X_batch.T, grad)
        db = np.sum(grad, axis=0)
        
        # 更新参数
        W -= LEARNING_RATE * dW
        b -= LEARNING_RATE * db
    
    # 计算准确率
    logits = np.dot(X, W) + b
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y)
    print(f"Epoch {epoch+1}/{EPOCHS} - Accuracy: {acc:.4f}")

# 保存模型
np.save('model/model_weights.npy', {'W': W, 'b': b})
print("Model saved!")