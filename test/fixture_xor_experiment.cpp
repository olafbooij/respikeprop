#include<iostream>
#include<array>
#include<vector>
#include<random>
#include<respikeprop/respikeprop_event_based.hpp>
#include<respikeprop/create_network.hpp>
#include<respikeprop/xor_experiment.hpp>

// Fixture. Delta weights were taken from run at commit 14b78563a3fa

int main()
{
  std::mt19937 random_gen(2);
  using namespace resp;

  // Create network architecture
  std::array network{create_layer({"input 1", "input 2", "bias"}),
                     create_layer({"hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5"}),
                     create_layer({"output"})};
  init_xor_network(network, random_gen);
  connect_outgoing(network);
  auto& output_neuron = network.back().at(0);

  auto sample = get_xor_dataset().at(0);
  Events events;
  load_sample(network, events, sample);

  while(output_neuron.spikes.empty() && events.active()) // does not work with recurency, then should check on time
    events.process_event();

  const double learning_rate = 1e-2;
  backprop(events.actual_spikes, learning_rate);

  auto& [_, hidden_layer, output_layer] = network;
  //std::cout << fabs(output_layer.at(0).incoming_connections.at(2).synapses.at(14).delta_weight)   << std::endl;
  //std::cout << fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(15).delta_weight)   << std::endl;
  //std::cout << fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(11).delta_weight)   << std::endl;
  //std::cout << fabs(hidden_layer.at(0).incoming_connections.at(0).synapses.at(15).delta_weight)   << std::endl;
  //std::cout << fabs(hidden_layer.at(2).incoming_connections.at(1).synapses.at(10).delta_weight) << std::endl;
  assert(fabs(output_layer.at(0).incoming_connections.at(2).synapses.at(14).delta_weight - -0.0449549  ) < 1e-7); // output hidden 3 2
  assert(fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(15).delta_weight - -0.0989137  ) < 1e-7); // output hidden 2 1
  assert(fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(11).delta_weight -  0          ) < 1e-7); // output hidden 2 5
  assert(fabs(hidden_layer.at(0).incoming_connections.at(0).synapses.at(15).delta_weight - -0.0280662  ) < 1e-7); // hidden 1 input 1 1
  assert(fabs(hidden_layer.at(2).incoming_connections.at(1).synapses.at(10).delta_weight - -0.000938801) < 1e-7); // hidden 3 input 2 6
  return 0;
}
