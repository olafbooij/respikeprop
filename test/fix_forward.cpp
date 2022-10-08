#include<iostream>
#include<cassert>
#include<respikeprop/neuron.hpp>

int main()
{
  using namespace resp;
  double timestep = .001;
  auto in = make_neuron(timestep);
  auto out = make_neuron(timestep);
  auto s = make_synapse(out, in, 3.0, 1.0);
  for(auto time: {1.0, 4.0})
    fire(*in, time);
  for(double time = 0.; time < 10.; time += timestep)
    forward_propagate(*out, time);
  assert(out->spike_times.size() == 1) ;
  assert(fabs(out->spike_times.at(0) - 5.949) < .01) ;

  return 0;
}

