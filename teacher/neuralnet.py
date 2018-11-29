import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    y = exp_a / np.sum(exp_a)
    return y

def softmax_batch(A):
    return np.apply_along_axis(arr=A, axis=1, func1d=softmax)

def cross_entroy_error(y_pred, y):
    """분류용 손실함수"""
    delta = 1e-7 # 아주 작은 값.     
    return -np.sum(y * np.log(y_pred + delta))

def cross_entropy_error_batch(y_pred, y):
    batch_size = len(y)
    cse = cross_entroy_error(y_pred, y) / batch_size
    return cse

def numerical_gradient(f, x):
    h = 1e-4
    경사 = np.zeros_like(x)
    
    # 각 x 축에 대해 경사 산출(미분 수행)
    for i, xi in enumerate(x):
        # f(x+h)
        x[i] = xi + h        
        fxh1 = f(x)
        # f(x-h)
        x[i] = xi - h
        fxh2 = f(x)
        
        경사[i] = (fxh1 - fxh2) / (2*h)
        # 원래 값 복원
        x[i] = xi
        
    return 경사

def numerical_gradient_2D(f, X):
    if X.ndim == 1:
        return numerical_gradient(f, X)
    
    # 2차원 행렬인 경우
    grad = np.zeros_like(X)
    for i,x in enumerate(X):
        grad[i] = numerical_gradient(f, x)
        
    return grad

class Layer:
    def __init__(self, 입력수, 출력수, 활성화):
        self.W = np.random.randn(입력수, 출력수)
        self.b = np.random.randn(출력수)
        self.활성화 = 활성화
        
    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        return self.활성화(z)
    
    
class FeedForwardNet:
    def __init__(self, loss_func):
        self.layers = []
        self.loss_func = loss_func
        
    def add(self, layer):
        self.layers.append(layer)
        
    def predict(self, X):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(layer_output)
        y = layer_output
        return y
    
    def compute_loss(self, X, y):
        y_pred = self.predict(X)
        loss = self.loss_func(y_pred, y)
        return loss
    
    def fit(self, X, y, 배치크기, 학습률, 학습횟수):
        loss_history = []
        for i in range(학습횟수):
            # 1. 미니배치
            샘플수 = len(X)
            배치색인 = np.random.choice(샘플수, 배치크기)
            X_batch = X[배치색인]
            y_batch = y[배치색인]
            # 2. 기울기 산출
            f = lambda W: self.compute_loss(X_batch, y_batch)
            기울기 = []
            for layer in self.layers:
                dW = numerical_gradient_2D(f, layer.W)
                db = numerical_gradient_2D(f, layer.b)
                기울기.append((dW, db))
            # 3. 매개변수 갱신 (경사하강법)
            for layer, (dW, db) in zip(self.layers, 기울기):
                layer.W -= dW * 학습률
                layer.b -= db * 학습률

            # 손실확인
            loss = self.compute_loss(X_batch, y_batch)
            loss_history.append(loss)
            print('[학습 {}] Loss: {}'.format(i, loss))
        
        return loss_history