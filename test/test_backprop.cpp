#include<iostream>
#include<vector>
#include<cassert>
#include<respikeprop/respikeprop_inefficient.hpp>

// These tests check the gradient (d Error / d weight) used for gradient
// descent by comparing it against a numerically computed gradient.

// Especially for deeper, more elaborate spiking neural networks this check
// will not work because of small quantization errors combined with small
// gradients. So here just checking for some very simple networks.


void check_backprop(auto& network, auto& synapse)
{
  auto& output = network.back();
  const double timestep = .0001;

  for(auto& neuron: network)
    neuron.spikes.clear();
  for(double time = 0.; time < 40.; time += timestep)
    for(auto& neuron: network)
      neuron.forward_propagate(time);
  //for(auto& neuron: network)
  //{
  //  std::cout << neuron.key << " spikes before: " << neuron.spikes.size() << " : " << std::endl;
  //  for(auto spike: neuron.spikes)
  //    std::cout << spike << std::endl;
  //}

  auto error_before = .5 * pow(output.spikes.at(0) - output.clamped, 2);

  const double small = 0.03;
  synapse.weight += small;
  for(auto& neuron: network)
    neuron.spikes.clear();
  for(double time = 0.; time < 40.; time += timestep)
    for(auto& neuron: network)
      neuron.forward_propagate(time);
  synapse.weight -= small;
  //for(auto& neuron: network)
  //{
  //  std::cout << neuron.key << " spikes after : " << neuron.spikes.size() << " : " << std::endl;
  //  for(auto spike: neuron.spikes)
  //    std::cout << spike << std::endl;
  //}
  auto error_after = .5 * pow(output.spikes.at(0) - output.clamped, 2);

  const double learning_rate = 1.;
  for(auto& neuron: network)
  {
    for(auto& in_synapse: neuron.incoming_synapses)
      in_synapse.delta_weight = 0.;
    neuron.compute_delta_weights(learning_rate);
  }

  //std::cout << "difference         " << error_after - error_before << std::endl;
  //std::cout << "d_E / d_w          " << (error_after - error_before) / small << std::endl;
  //std::cout << "computed d_E / d_w " << - synapse.delta_weight / learning_rate << std::endl;
  assert(fabs((error_after - error_before) / small - (- synapse.delta_weight / learning_rate)) < (timestep / small * 2));
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
  output.clamped = 3.;
  check_backprop(network, synapse);
}

void check_two_inputs_one_output()
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
  output.clamped = 3.;

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
  auto& synapse_0 = hidden.incoming_synapses.emplace_back(input, 7., 1.);
  auto& synapse_1 = output.incoming_synapses.emplace_back(hidden, 7., 1.);
  input.post_neuron_ptrs.emplace_back(&hidden);
  hidden.post_neuron_ptrs.emplace_back(&output);

  input.fire(0.);
  output.clamped = 3.;

  check_backprop(network, synapse_0);
  check_backprop(network, synapse_1);
}

void check_two_inputs_two_hiddens_one_output()
{
  using namespace resp;

  Neuron input_0("input_0");
  Neuron input_1("input_1");
  std::vector<Neuron> network;
  network.reserve(3);
  Neuron& hidden_0 = network.emplace_back("hidden_0");
  Neuron& hidden_1 = network.emplace_back("hidden_1");
  Neuron& output = network.emplace_back("output");
  hidden_0.incoming_synapses.reserve(2);
  hidden_0.incoming_synapses.emplace_back(input_0, 3.3, 1.1);
  hidden_0.incoming_synapses.emplace_back(input_1, 2.7, 1.2);
  hidden_1.incoming_synapses.reserve(2);
  hidden_1.incoming_synapses.emplace_back(input_0, 3.0, 1.3);
  hidden_1.incoming_synapses.emplace_back(input_1, 3.7, 0.9);
  output.incoming_synapses.reserve(2);
  output.incoming_synapses.emplace_back(hidden_0, 3., 1.2);
  output.incoming_synapses.emplace_back(hidden_1, 3.2, 1.1);
  input_0.post_neuron_ptrs.emplace_back(&hidden_0);
  input_0.post_neuron_ptrs.emplace_back(&hidden_1);
  input_1.post_neuron_ptrs.emplace_back(&hidden_0);
  input_1.post_neuron_ptrs.emplace_back(&hidden_1);
  hidden_0.post_neuron_ptrs.emplace_back(&output);
  hidden_1.post_neuron_ptrs.emplace_back(&output);

  input_0.fire(0.2);
  input_1.fire(0.1);
  output.clamped = 3.;

  for(auto& neuron: network)
    for(auto& synapse: neuron.incoming_synapses)
      check_backprop(network, synapse);

  input_0.fire(0.3);
  input_1.fire(0.4);

  for(auto& neuron: network)
    for(auto& synapse: neuron.incoming_synapses)
      check_backprop(network, synapse);
}

void check_two_inputs_two_hiddens_one_output_multi_synapses()
{
  using namespace resp;

  Neuron input_0("input_0");
  Neuron input_1("input_1");
  std::vector<Neuron> network;
  network.reserve(3);
  Neuron& hidden_0 = network.emplace_back("hidden_0");
  Neuron& hidden_1 = network.emplace_back("hidden_1");
  Neuron& output = network.emplace_back("output");
  hidden_0.incoming_synapses.reserve(4);
  hidden_0.incoming_synapses.emplace_back(input_0, 1.6, 1.12);
  hidden_0.incoming_synapses.emplace_back(input_0, 1.8, 1.20);
  hidden_0.incoming_synapses.emplace_back(input_1, 1.0, 1.19);
  hidden_0.incoming_synapses.emplace_back(input_1, 1.7, 1.29);
  hidden_1.incoming_synapses.reserve(4);
  hidden_1.incoming_synapses.emplace_back(input_0, 1.0, 1.32);
  hidden_1.incoming_synapses.emplace_back(input_0, 4.0, 1.39);
  hidden_1.incoming_synapses.emplace_back(input_1,-3.7, 0.90);
  hidden_1.incoming_synapses.emplace_back(input_1, 6.7, 0.98);
  output.incoming_synapses.reserve(4);
  output.incoming_synapses.emplace_back(hidden_0, 5., 1.23);
  output.incoming_synapses.emplace_back(hidden_0, 4., 1.20);
  output.incoming_synapses.emplace_back(hidden_1,-1.2, 1.12);
  output.incoming_synapses.emplace_back(hidden_1,-0.2, 1.3);
  input_0.post_neuron_ptrs.emplace_back(&hidden_0);
  input_0.post_neuron_ptrs.emplace_back(&hidden_1);
  input_1.post_neuron_ptrs.emplace_back(&hidden_0);
  input_1.post_neuron_ptrs.emplace_back(&hidden_1);
  hidden_0.post_neuron_ptrs.emplace_back(&output);
  hidden_1.post_neuron_ptrs.emplace_back(&output);

  input_0.fire(0.2);
  input_1.fire(0.1);
  input_0.fire(0.3);
  input_1.fire(0.4);
  output.clamped = 3.;

  for(auto& neuron: network)
    for(auto& synapse: neuron.incoming_synapses)
      check_backprop(network, synapse);
}

int main()
{
  check_one_input_one_output();
  check_two_inputs_one_output();
  check_one_input_one_hidden_one_output();
  check_two_inputs_two_hiddens_one_output();
  check_two_inputs_two_hiddens_one_output_multi_synapses();

  return 0;
}
