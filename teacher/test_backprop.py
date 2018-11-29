import numpy as np
from deepy.dataset import mnist

from neuralnet_backprop import FeedForwadNet
from neuralnet_backprop import Affine, SoftmaxCrossEntropy, Sigmoid

(X_train, Y_train), (X_test, Y_test) = mnist.load_mnist(one_hot_label=True)

model = FeedForwadNet()
model.add(Affine(784, 100))
model.add(Sigmoid())
model.add(Affine(100, 200))
model.add(Sigmoid())
model.add(Affine(200, 100))
model.add(Sigmoid())
model.add(Affine(100, 10))
model.add(SoftmaxCrossEntropy())

loss_history = model.fit(X_train, Y_train, 100, 6000, 0.01)
