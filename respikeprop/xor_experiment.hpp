#pragma once
#include<array>
#include<random>
#include<ranges>
#include<algorithm>
#include<vector>

namespace resp
{
  auto get_xor_dataset()
  {
    struct sample
    {
      std::array<double, 3> input;
      double output;
    };
    std::array<sample, 4> dataset{{
      {{0., 0., 0.}, 16.},
      {{0., 6., 0.}, 10.},
      {{6., 0., 0.}, 10.},
      {{6., 6., 0.}, 16.}
    }};
    return dataset;
  };

  void init_xor_network(auto& network, auto& random_gen)
  {
    auto& [input_layer, hidden_layer, output_layer] = network;
    connect_layers(input_layer, hidden_layer);
    connect_layers(hidden_layer, output_layer);

    // Set random weights
    for(auto& n: hidden_layer)
      for(auto& incoming_connection: n.incoming_connections)
        for(auto& synapse: incoming_connection.synapses)
          synapse.weight = std::uniform_real_distribution<>(-.5, 1.0)(random_gen);

    for(auto& n: output_layer)
      for(auto& incoming_connection: n.incoming_connections)
      {
        if(incoming_connection.neuron->key == "hidden 5")
          for(auto& synapse: incoming_connection.synapses)
            synapse.weight = std::uniform_real_distribution<>(-.5, 0.)(random_gen);
        else
          for(auto& synapse: incoming_connection.synapses)
            synapse.weight = std::uniform_real_distribution<>(0., 1.)(random_gen);
      }
  }

}
