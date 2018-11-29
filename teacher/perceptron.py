import numpy as np

def 취합(X, w, b):
    z = np.dot(X, w) + b
    return z

def 활성화(z):
    y = np.where(z > 0, 1, -1)
    return y

class 뉴런:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
    def predict(self, x):
        z = 취합(x, self.w, self.b)
        y = 활성화(z)
        return y
    
class 퍼셉트론(뉴런):
    def __init__(self):
        super().__init__(w=None, b=None)
    
    def fit(self, X, y, 학습횟수, 학습률=0.01):
        # 가중치 초기화
        샘플수, 특징수 = X.shape
        self.w = np.zeros(특징수)
        self.b = 0.0
        
        error_history = []
        for i in range(학습횟수):
            종합오류 = 0
            for xi, yi in zip(X, y):
                yi_pred = self.predict(xi)
                error = yi - yi_pred
                종합오류 += error**2
                # 가중치 갱신
                update = error * 학습률
                self.w += update * xi
                self.b += update
            error_history.append(종합오류)
            
        return error_history