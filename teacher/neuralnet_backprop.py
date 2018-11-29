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


class ReLu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = x > 0
        return np.where(self.mask, x, 0)
    
    def backward(self, dout):
        return np.where(self.mask, 1, 0)


class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


class Affine:
    def __init__(self, 입력수, 출력수):
        self.W = np.random.randn(입력수, 출력수)
        self.b = np.random.randn(출력수)
        self.X = None
        self.dW = None
        self.db = None
        
    def forward(self, X):
        self.X = X
        z = np.dot(X, self.W) + self.b
        return z
    
    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        return dX


class SoftmaxCrossEntropy:
    def __init__(self):
        self.Y = None
        self.Y_pred = None
        
    def forward(self, X, Y):
        self.Y = Y
        self.Y_pred = softmax_batch(X)
        loss = cross_entropy_error_batch(self.Y_pred, self.Y)
        return loss
    
    def backward(self, dout=1):
        batch_size = len(self.Y)
        dX = (self.Y_pred - self.Y) / batch_size
        return dX


class FeedForwadNet:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def predict(self, X):
        layer_output = X
        for layer in self.layers[:-1]:
            layer_output = layer.forward(layer_output)            
        return layer_output
    
    def compute_loss(self, X, Y):
        Y_pred = self.predict(X)
        loss = self.layers[-1].forward(Y_pred, Y)
        return loss
    
    def fit(self, X, y, 배치크기, 학습횟수, 학습률):
        loss_history = []
        for i in range(학습횟수):
            # 1. 미니배치
            샘플수 = len(X)
            배치색인 = np.random.choice(샘플수, 배치크기)
            X_batch = X[배치색인]
            y_batch = y[배치색인]
            # 2. 기울기 산출
            #  1) 순전파
            self.compute_loss(X_batch, y_batch)
            #  2) 역전파
            dout = 1
            for layer in reversed(self.layers):
                dout = layer.backward(dout)
            # 3. 갱신
            for layer in self.layers:
                if isinstance(layer, Affine):
                    layer.W -= layer.dW * 학습률
                    layer.b -= layer.db * 학습률
            
            # 손실 확인
            loss = self.compute_loss(X_batch, y_batch)
            loss_history.append(loss)
            print('[학습 {}] Loss: {}'.format(i+1, loss))
        
        return loss_history
