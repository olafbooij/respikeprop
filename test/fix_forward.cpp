#include<iostream>
#include<cassert>
#include<respikeprop/neuron.hpp>

int main()
{
  using namespace resp;
  auto in = make_neuron();
  auto out = make_neuron();
  auto s = make_synapse(out, in, 3.0, 1.0);
  for(auto time: {1.0, 4.0})
    fire(*in, time);
  for(double time = 0.; time < 10.; time += .001)
    forward_propagate(*out, time);
  assert(out->spike_times.size() == 1) ;
  assert(fabs(out->spike_times.at(0) - 5.949) < .01) ;

  return 0;
}

