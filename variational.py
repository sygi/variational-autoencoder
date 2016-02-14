import matplotlib.pyplot as plt
import time
from autoencoder import get_data_stream, get_typical_layer, plot_images
from theano import tensor
from blocks.bricks import Linear, Tanh, Logistic
from blocks.initialization import IsotropicGaussian, Constant
from theano.sandbox.rng_mrg import MRG_RandomStreams

L = 1
BATCH_SIZE = 20
J = 20 # latent dimension

def plot_batch(orig, compressed):
  for i in range(len(orig)):
    plt.subplot(121)
    plt.imshow(1 - orig[i].reshape(28, 28), cmap=plt.cm.gray)
    plt.subplot(122)
    plt.imshow(1 - compressed[i].reshape(28, 28), cmap=plt.cm.gray)
    if i == 0:
      plt.show(block=False)
    else:
      plt.draw()
    time.sleep(1)
  plt.close()

def showcase(cg, output_name="last_apply_output"):
  import numpy
  first = True
  test_ds = get_data_stream(False, BATCH_SIZE)
  for image in next(test_ds.get_epoch_iterator()):
    cg2 = cg.replace({cg.inputs[0]: numpy.asmatrix(image)})
    out = (VariableFilter(theano_name_regex=output_name) (cg2.variables))[-1]
    plot_batch(image, out.eval())
  plt.close()

def encoder_network(latent_dim=J):
  x = tensor.matrix("features")
  hidden1 = get_typical_layer(x, 784, 500)
  log_sigma_sq = get_typical_layer(hidden1, 500, latent_dim)
  mu = get_typical_layer(hidden1, 500, latent_dim)
  return (log_sigma_sq, mu, x)

def decoder_network(latent_sample, latent_dim=J):
  # bernoulli case
  hidden2 = get_typical_layer(latent_sample, latent_dim, 500)
  hidden2_to_output = Linear(name="last", input_dim=500, output_dim=784)
  hidden2_to_output.weights_init = IsotropicGaussian(0.02)
  hidden2_to_output.biases_init = Constant(0)
  hidden2_to_output.initialize()
  return Logistic().apply(hidden2_to_output.apply(hidden2))

rng = MRG_RandomStreams()

def get_cost(latent_dim=J):
  log_sigma_sq, mu, x = encoder_network(latent_dim)
  sigma_sq = tensor.exp(log_sigma_sq)
  eps = rng.normal((BATCH_SIZE, J))
  z = tensor.sqrt(sigma_sq) + mu * eps
  y = decoder_network(z) # TODO: L > 1
  log_p_x_y = tensor.sum(tensor.log(y) * x + (1 - x) * tensor.log(1 - y))
  KL = -0.5 * tensor.sum(1 + log_sigma_sq - mu ** 2 - sigma_sq)
  cost = -tensor.mean(-KL + log_p_x_y) # over batch
  cost.name = "cost"
  return cost * BATCH_SIZE # to have comparable costs for different batch sizes

from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Scale, Adam
from blocks.filter import VariableFilter
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring

cost = get_cost()
cg = ComputationGraph(cost)
gd = GradientDescent(cost=cost, parameters=cg.parameters,
    step_rule=Adam())
monitor = TrainingDataMonitoring([cost], after_epoch=True)
main_loop = MainLoop(data_stream = get_data_stream(True, BATCH_SIZE), algorithm=gd, extensions=[
  monitor, FinishAfter(after_n_epochs=5), ProgressBar(), Printing()])

main_loop.run()

showcase(cg, "last_apply_output")
