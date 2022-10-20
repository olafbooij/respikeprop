#include<iostream>
#include<random>
#include<array>
#include<vector>
#include<cassert>
#include<respikeprop/forward_backward_exhaust.hpp>

namespace resp
{}

void check_backprop(auto& network, auto& synapse)
{
  auto& output = network.back();
  const double timestep = .0001;

  for(double time = 0.; time < 40.; time += timestep)
    for(auto& neuron: network)
      neuron.forward_propagate(time);
  //std::cout << output.spikes.size() << " : " << std::endl;
  //for(auto spike: output.spikes)
  //  std::cout << spike << std::endl;

  auto error_before = .5 * pow(output.spikes.at(0) - 3, 2);

  const double small = 0.03;
  synapse.weight += small;
  for(auto& neuron: network)
    output.spikes.clear();
  for(double time = 0.; time < 40.; time += timestep)
    for(auto& neuron: network)
      neuron.forward_propagate(time);
  //std::cout << output.spikes.size() << " : " << std::endl;
  //for(auto spike: output.spikes)
  //  std::cout << spike << std::endl;
  auto error_after = .5 * pow(output.spikes.at(0) - 3, 2);

  const double learning_rate = 1.;
  output.compute_delta_weights(learning_rate);

  //std::cout << error_after - error_before << std::endl;
  //std::cout << (error_after - error_before) /small << std::endl;
  //std::cout << - synapse.delta_weight / learning_rate << std::endl;
  assert(fabs((error_after - error_before) / small - (- synapse.delta_weight / learning_rate)) < (timestep / small));
}

void check_one_input_one_output()
{
  using namespace resp;

  Neuron input("input");
  std::vector<Neuron> network;
  Neuron& output = network.emplace_back("output");
  auto& synapse = output.incoming_synapses.emplace_back(input, 7., 1.);
  input.post_neuron_ptrs.emplace_back(&output);

  input.fire(0.);
  check_backprop(network, synapse);
}

void check_two_input_one_output()
{
  using namespace resp;

  Neuron input_0("input");
  Neuron input_1("input");
  std::vector<Neuron> network;
  Neuron& output = network.emplace_back("output");
  output.incoming_synapses.emplace_back(input_0, 3., 1.);
  auto& synapse = output.incoming_synapses.emplace_back(input_1, 3., 1.);
  input_0.post_neuron_ptrs.emplace_back(&output);
  input_1.post_neuron_ptrs.emplace_back(&output);

  input_0.fire(0.);
  input_1.fire(0.);

  check_backprop(network, synapse);
}

void check_one_input_one_hidden_one_output()
{
  using namespace resp;

  Neuron input("input");
  std::vector<Neuron> network;
  network.reserve(2);
  Neuron& hidden = network.emplace_back("hidden");
  Neuron& output = network.emplace_back("output");
  hidden.incoming_synapses.emplace_back(input, 7., 1.);
  auto& synapse = output.incoming_synapses.emplace_back(hidden, 7., 1.);
  input.post_neuron_ptrs.emplace_back(&hidden);
  hidden.post_neuron_ptrs.emplace_back(&output);

  input.fire(0.);

  check_backprop(network, synapse);
}

int main()
{
  check_one_input_one_output();
  check_two_input_one_output();
  check_one_input_one_hidden_one_output();

  return 0;
}

