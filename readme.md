# ReSpikeProp

An implementation of "A gradient descent rule for spiking neurons emitting multiple spikes", Olaf Booij, Hieu tat Nguyen, Information Processing Letters, Volume 95, Issue 6, 30 September 2005, Pages 552-558.

The implementation stores gradients in the forward pass to allow for efficient
backpropagation. Several efficiency improvements still need to be implemented though.

The repo also includes a straightforward (non-optimized) implementation which is
easy to match with the formulas in the paper and therefore has some educational
value / serves as a good reference. See
[./respikeprop/respikeprop_reference_impl.hpp](./respikeprop/respikeprop_reference_impl.hpp).

The same network as used in the paper for the experiments is implemented, which
uses only feedforward connections.

<img src="./doc/xor_example_feedforward.svg">

See : [./test/xor_experiment.cpp](./test/xor_experiment.cpp).

However, as explained in the paper but left as future work,
the algorithm can also deal with recurrent connections. In
[./test/xor_experiment_recurrent.cpp](./test/xor_experiment_recurrent.cpp) we
show that the following network can also be trained to learn the XOR example.

<img src="./doc/xor_example_recurrent.svg">


Hence the rebranding of the algorithm as *Re*SpikeProp.
