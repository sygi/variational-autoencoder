import matplotlib.pyplot as plt
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten
from theano import tensor
from blocks.bricks import Linear, Tanh
from blocks.initialization import IsotropicGaussian, Constant
from blocks.filter import VariableFilter

def plot_images(orig, compressed, first):
  plt.subplot(121)
  plt.imshow(1 - orig.reshape(28, 28), cmap = plt.cm.gray)
  plt.subplot(122)
  plt.imshow(1 - compressed.reshape(28, 28), cmap = plt.cm.gray)
  if first:
    plt.show(block=False)
  else:
    plt.draw()

def get_typical_layer(input_layer, input_dim, output_dim):
  layer = Linear(input_dim=input_dim, output_dim=output_dim)
  layer.weights_init = IsotropicGaussian(0.01)
  layer.biases_init = Constant(0)
  layer.initialize()
  return Tanh().apply(layer.apply(input_layer))

def get_data_stream(train=True, batch_size=100):
  mnist_id = "train" if train else "test"
  mnist = MNIST((mnist_id, ), sources=["features"])
  return Flatten(DataStream.default_stream(mnist, iteration_scheme=\
      ShuffledScheme(examples=mnist.num_examples, batch_size=batch_size)))
# to get an example
# example = ds.get_epoch_iterator().next()[0][0] # [sources][no-in-batch]

def showcase(cg, output_name="tanh_apply_output"):
  import numpy
  import time
  first = True
  test_ds = get_data_stream(False)
  for image in next(test_ds.get_epoch_iterator())[0]:
    cg2 = cg.replace({cg.inputs[0]: numpy.asmatrix(image)})
    out = (VariableFilter(theano_name_regex=output_name) (cg2.variables))[-1]
    plot_images(image, out.eval(), first)
    first = False
    time.sleep(1)
  plt.close()

def main():
  x = tensor.matrix("features")
  input_to_hidden1 = get_typical_layer(x, 784, 500)
  #hidden1_to_hidden2 = get_typical_layer(input_to_hidden1, 500, 300)
  hidden1_to_latent = get_typical_layer(input_to_hidden1, 500, 20)

  latent_to_hidden2 = get_typical_layer(hidden1_to_latent, 20, 500)
  #hidden3_to_hidden4 = get_typical_layer(latent_to_hidden3, 300, 500)
  hidden2_to_output = get_typical_layer(latent_to_hidden2, 500, 784)

  from blocks.bricks.cost import SquaredError, AbsoluteError
  from blocks.graph import ComputationGraph
  from blocks.algorithms import Adam, GradientDescent, Scale
  from blocks.roles import WEIGHT

  cost = AbsoluteError(name="error").apply(x, hidden2_to_output)
  cg = ComputationGraph(cost)
  gd = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Adam())

  from blocks.main_loop import MainLoop
  from blocks.extensions import FinishAfter, Printing, ProgressBar
  from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
  monitor = TrainingDataMonitoring([cost], after_epoch=True)
  main_loop = MainLoop(data_stream=get_data_stream(), algorithm=gd, extensions=[monitor, FinishAfter(after_n_epochs=4),  ProgressBar(), Printing()])

  main_loop.run()
  showcase(cg)

if __name__ == "__main__":
  main()
