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
  //std::cout << fabs(output_layer.at(0).incoming_connections.at(2).synapses.at(14).delta_weight - -0.0681334 )   << std::endl;
  //std::cout << fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(15).delta_weight - -0.148263  )   << std::endl;
  //std::cout << fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(11).delta_weight -  0         )   << std::endl;
  //std::cout << fabs(hidden_layer.at(0).incoming_connections.at(0).synapses.at(15).delta_weight - -0.0182621 )   << std::endl;
  //std::cout << fabs(hidden_layer.at(2).incoming_connections.at(1).synapses.at(10).delta_weight - -0.00157232) << std::endl;
  assert(fabs(output_layer.at(0).incoming_connections.at(2).synapses.at(14).delta_weight - -0.0681334 ) < 1e-7); // output hidden 3 2
  assert(fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(15).delta_weight - -0.148263  ) < 1e-6); // output hidden 2 1
  assert(fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(11).delta_weight -  0         ) < 1e-7); // output hidden 2 5
  assert(fabs(hidden_layer.at(0).incoming_connections.at(0).synapses.at(15).delta_weight - -0.0182621 ) < 1e-7); // hidden 1 input 1 1
  assert(fabs(hidden_layer.at(2).incoming_connections.at(1).synapses.at(10).delta_weight - -0.00157232) < 1e-7); // hidden 3 input 2 6
  return 0;
}

