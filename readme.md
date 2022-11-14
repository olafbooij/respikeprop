# ReSpikeProp

An implementation of "A gradient descent rule for spiking neurons emitting multiple spikes", Olaf Booij, Hieu tat Nguyen, Information Processing Letters, Volume 95, Issue 6, 30 September 2005, Pages 552-558.

The implementation stores gradients in the forward pass to allow for efficient
backpropagation. Several efficiency improvements still need to be implemented.

The repo includes an implementation
([./respikeprop/respikeprop_reference_impl.hpp](./respikeprop/respikeprop_reference_impl.hpp)) which is easy to match with the
formulas of the paper and therefore has some educational value / serves as
a good reference. It is however very inefficient.

The name of this repository, ReSpikeProp, is to stress the fact that spiking
neurons can be connected in a recurrent manner, without any adjustment to the
backpropagation method.
