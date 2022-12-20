# ReSpikeProp

An implementation of "A gradient descent rule for spiking neurons emitting multiple spikes", Olaf Booij, Hieu tat Nguyen, Information Processing Letters, Volume 95, Issue 6, 30 September 2005, Pages 552-558.

Actually, there's multiple implementations:

A straightforward (non-optimized) implementation which is
easy to match with the formulas in the paper and therefore has some educational
value / serves as a good reference. See
[./respikeprop/respikeprop_reference_impl.hpp](./respikeprop/respikeprop_reference_impl.hpp).

An efficient event-based implementation that does not use discrete time-steps,
but computes exact spike-times, enabled by a smart choice of time-constants.
Gradients are stored in the forward pass to allow for efficient backpropagation. See
[./respikeprop/respikeprop_event_based.hpp](./respikeprop/respikeprop_event_based.hpp).

# Benchmarks

## XOR
The same network as used in the paper for the experiments is implemented, which
uses only feedforward connections.

<img src="./doc/xor_example_feedforward.svg">

See : [./test/xor_experiment.cpp](./test/xor_experiment.cpp).

However, as explained in the paper but left as future work,
the algorithm can also deal with recurrent connections. In
[./test/xor_experiment_recurrent.cpp](./test/xor_experiment_recurrent.cpp) we
show that the following network can also be trained to learn the XOR example.

<img src="./doc/xor_example_recurrent.svg">

Hence the re-branding of the algorithm as *Re*SpikeProp.

## N-MNIST

A more serious feed-forward example is given in
[./test/n-mnist.cpp](./test/n-mnist.cpp), which uses the Neuromorphic MNIST (N-MNIST) dataset (see
https://www.garrickorchard.com/datasets/n-mnist).


# Build

See : [./requirements.md](./requirements.md).

