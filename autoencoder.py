def plot_image(image):
  import matplotlib.pyplot as plt
  plt.matshow(1 - image, cmap = plt.cm.gray)
  plt.show()

# regular network
from theano import tensor
from blocks.bricks import Linear, Tanh
from blocks.initialization import IsotropicGaussian, Constant
x = tensor.matrix("input")
input_to_latent = Linear(input_dim=784, output_dim=100)
input_to_latent.weights_init = IsotropicGaussian()
input_to_latent.biases_init = Constant(0)
h = Tanh().apply(input_to_latent.apply(x))

latent_to_output = Linear(input_dim=100, output_dim=784)
latent_to_output.weights_init = IsotropicGaussian()
latent_to_output.biases_init = Constant(0)
output = Tanh().apply(latent_to_output.apply(h))

from blocks.bricks.cost import SquaredError
cost = SquaredError().apply(x, output)
#cg = ComputationGraph(cost)

# data
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten
mnist = MNIST(("train", ))
g = Flatten(DataStream.default_stream(mnist, iteration_scheme=\
    ShuffledScheme(examples=mnist.num_examples, batch_size=10)))

# to get an example
example = g.get_epoch_iterator().next()[0]

