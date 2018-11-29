class Layer:
    def __init__(self, input, output, activate):
        # 신경망에서는 초기값은 0으로 하면 안됨. perceptron은 초기값을 0으로 해도 됨.
        self.W = np.random.randn(input, output)
        self.b = np.random.randn(output)
        self.activate = activate
        
    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        a = self.activate(z)
        return a

class FeedForwardNet:
    def __init__(self):
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def predict(self, X):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward(layer_output)
        y = layer_output
        return y
    
    def fit(self, X, y):
        # TODO
        pass
    