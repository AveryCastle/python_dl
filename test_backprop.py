import numpy as np
import matplotlib.pyplot as plt
from deepy.dataset import mnist

from neuralnet_backprop import FeedForwadNet
from neuralnet_backprop import Affine, SoftmaxCrossEntropy, Sigmoid

(X_train, Y_train), (X_test, Y_test) = mnist.load_mnist(one_hot_label=True)

model = FeedForwadNet()
model.add(Affine(784, 50))
model.add(Sigmoid())
model.add(Affine(50, 100))
model.add(Sigmoid())
model.add(Affine(100, 10))
model.add(SoftmaxCrossEntropy())

loss_history = model.fit(X_train, Y_train, 100, 600*10, 0.5)

plt.plot(loss_history)