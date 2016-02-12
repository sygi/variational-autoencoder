import matplotlib.pyplot as plt

def plot_images(orig, compressed, first):
  plt.subplot(121)
  plt.imshow(1 - orig.reshape(28, 28), cmap = plt.cm.gray)
  plt.subplot(122)
  plt.imshow(1 - compressed.reshape(28, 28), cmap = plt.cm.gray)
  if first:
    plt.show(block=False)
  else:
    plt.draw()

# regular network
from theano import tensor
from blocks.bricks import Linear, Tanh
from blocks.initialization import IsotropicGaussian, Constant
x = tensor.matrix("features")
input_to_latent = Linear(name="input_to_latent", input_dim=784, output_dim=30)
input_to_latent.weights_init = IsotropicGaussian(0.01)
input_to_latent.biases_init = Constant(0)
input_to_latent.initialize()
h = Tanh(name="transformation1").apply(input_to_latent.apply(x))

latent_to_output = Linear(name="latent_to_output", input_dim=30, output_dim=784)
latent_to_output.weights_init = IsotropicGaussian(0.01)
latent_to_output.biases_init = Constant(0)
latent_to_output.initialize()
output = Tanh(name="output").apply(latent_to_output.apply(h))

from blocks.bricks.cost import SquaredError, AbsoluteError
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Scale
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT

cost = AbsoluteError().apply(x, output)
cg = ComputationGraph(cost)
gd = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Scale(learning_rate=0.01))

# data
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten
mnist = MNIST(("train", ), sources=["features"])
ds = Flatten(DataStream.default_stream(mnist, iteration_scheme=\
    ShuffledScheme(examples=mnist.num_examples, batch_size=10)))

# to get an example
# example = ds.get_epoch_iterator().next()[0][0] # [sources][no-in-batch]

from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
monitor = TrainingDataMonitoring([cost])
main_loop = MainLoop(data_stream=ds, algorithm=gd, extensions=[monitor, FinishAfter(after_n_epochs=3),  ProgressBar(), Printing()])

main_loop.run()

mnist_test = MNIST(("test", ), sources=["features"])
test_ds = Flatten(DataStream.default_stream(mnist, iteration_scheme=\
    ShuffledScheme(examples=mnist.num_examples, batch_size=100)))

def showcase():
  import numpy
  import time
  first = True
  for image in next(test_ds.get_epoch_iterator())[0]:
    cg2 = cg.replace({x: numpy.asmatrix(image)})
    out, = VariableFilter(theano_name_regex="output_apply_output") (cg2.variables)
    plot_images(image, out.eval(), first)
    first = False
    time.sleep(1)
#    plt.close()

showcase()
