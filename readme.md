# ReSpikeProp

An implementation of "A gradient descent rule for spiking neurons emitting multiple spikes", Olaf Booij, Hieu tat Nguyen, Information Processing Letters, Volume 95, Issue 6, 30 September 2005, Pages 552-558.

The implementation stores gradients in the forward pass to allow for efficient
backpropagation. Several efficiency improvements still need to be implemented.

The repo includes a straight forward (not optimized) implementation which is
easy to match with the formulas of the paper and therefore has some educational
value / serves as a good reference. See
[./respikeprop/respikeprop_reference_impl.hpp](./respikeprop/respikeprop_reference_impl.hpp).

The same network as in the paper is implemented, which uses only feedforward
connections
<img src="./doc/xor_example_feedforward.svg">
See : [./test/xor_experiment.cpp](./test/xor_experiment.cpp).

However, as explained in the paper (but left as future work to investigate),
the algorithm can also deal with recurrent connections. In
[./test/xor_experiment_recurrent.cpp](./test/xor_experiment_recurrent.cpp) we
show that the following network can be trained to learn the XOR example.
<img src="./doc/xor_example_recurrent.svg">

This is also the reason for the name of this repository: *Re*SpikeProp.
