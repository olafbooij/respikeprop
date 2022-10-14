#include<iostream>
#include<cassert>
#include<respikeprop/forward_exhaust.hpp>

int main()
{
  using namespace resp;
  double timestep = .001;
  auto in = make_neuron();
  auto out = make_neuron();
  auto s = make_synapse(out, in, 3.0, 1.0);
  for(auto time: {1.0, 4.0})
    in->fire(time);
  for(double time = 0.; time < 10.; time += timestep)
    out->forward_propagate(time);
  assert(out->spikes.size() == 1) ;
  assert(fabs(out->spikes.at(0) - 5.414) < .01) ;

  return 0;
}

