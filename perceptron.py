import numpy as np

def gathering(X, w, b):
    z = np.dot(X, w.T) + b
    return z

def activate(z):
    y = np.where(z>0, 1, -1)
    return y

class Neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
    def predict(self, x):
        z = gathering(x, self.w, self.b)
        y = activate(z)
        return y
    
    
class Perceptron(Neuron):
    def __init__(self):
        super().__init__(w=None, b=None)
        
    def fit(self, X, y, learning_count, learning_rate=0.01):
        # 가중치 초기화, 가중치의 갯수는 X 특징의 갯수
        sample_c, label_c = X.shape
        self.w = np.zeros(label_c)
        self.b = 0.0
        
        error_history = []
        for i in range(learning_count):
            total_error = 0
            for xi, yi in zip(X, y):
                yi_pred = self.predict(xi)
                error = yi - yi_pred
                # 오류를 부각시키기 위해서
                total_error += error**2 
                # 가중치 갱신
                update = error * learning_rate
                self.w += update * xi
                self.b += update * 1
            error_history.append(total_error)
            
        return error_history