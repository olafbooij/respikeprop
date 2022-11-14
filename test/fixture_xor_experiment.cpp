#include<iostream>
#include<array>
#include<vector>
#include<random>
#include<ctime>
#include<respikeprop/respikeprop_store_gradients.hpp>
#include<test/xor_experiment.hpp>

// Fixture. Delta weights were taken from run of reference implementation.

int main()
{
  std::mt19937 random_gen(2);
  using namespace resp;

  // Create network architecture
  std::array network{create_layer({"input 1", "input 2", "bias"}),
                     create_layer({"hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5"}),
                     create_layer({"output"})};
  init_network(network, random_gen);

  auto sample = get_xor_dataset().at(0);
  load_sample(network, sample);

  const double timestep = .1;
  propagate(network, 40., timestep);

  const double learning_rate = 1e-2;
  network.back().at(0).compute_delta_weights(learning_rate);

  auto& [_, hidden_layer, output_layer] = network;
  assert(fabs(output_layer.at(0).incoming_connections.at(2).synapses.at(14).delta_weight - -0.0319112) < 1e-5);
  assert(fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(15).delta_weight - -0.0575665) < 1e-5);
  assert(fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(11).delta_weight - -0.0092582) < 1e-5);
  assert(fabs(hidden_layer.at(0).incoming_connections.at(0).synapses.at(15).delta_weight - -0.014726) < 1e-5);
  assert(fabs(hidden_layer.at(2).incoming_connections.at(1).synapses.at(10).delta_weight - -0.000471913) < 1e-5);
  return 0;
}

