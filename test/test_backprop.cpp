#include<iostream>
#include<random>
#include<array>
#include<vector>
#include<cassert>
#include<respikeprop/forward_backward_exhaust.hpp>

namespace resp
{}

int main()
{
  using namespace resp;

  Neuron input("input");
  Neuron output("output");
  auto& synapse = output.incoming_synapses.emplace_back(input, 7., 1.);
  input.post_neuron_ptrs.emplace_back(&output);

  input.fire(0.);
  const double timestep = .0001;
  for(double time = 0.; time < 40.; time += timestep)
    output.forward_propagate(time);
  //for(auto spike: output.spikes)
  //  std::cout << spike << std::endl;

  auto error_before = .5 * pow(output.spikes.at(0) - 3, 2);

  const double small = 0.03;
  synapse.weight += small;
  output.spikes.clear();
  for(double time = 0.; time < 40.; time += timestep)
    output.forward_propagate(time);
  auto error_after = .5 * pow(output.spikes.at(0) - 3, 2);

  //for(auto spike: output.spikes)
  //  std::cout << spike << std::endl;

  const double learning_rate = 1.;
  output.compute_delta_weights(learning_rate);

  //std::cout << error_after - error_before << std::endl;
  //std::cout << (error_after - error_before) /small << std::endl;
  //std::cout << - synapse.delta_weight / learning_rate << std::endl;
  assert(fabs((error_after - error_before) / small - (- synapse.delta_weight / learning_rate)) < (timestep / small));

  return 0;
}

