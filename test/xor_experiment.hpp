#pragma once
#include<array>
#include<random>
#include<ranges>
#include<algorithm>
#include<vector>

namespace resp
{

  void connect_neurons(auto& pre, auto& post)
  {
    auto& incoming_connection = post.incoming_connections.emplace_back(&pre);
    for(auto delay_i = 16; delay_i--;)
      incoming_connection.synapses.emplace_back(.0, delay_i + 1.0 + 1e-10, 0.);
  };

  void connect_layers(auto& pre_layer, auto& post_layer)
  {
    for(auto& pre: pre_layer)
      for(auto& post: post_layer)
        connect_neurons(pre, post);
  };

  auto create_layer(const std::vector<std::string>&& keys)
  {
    std::vector<Neuron> layer;
    for(const auto& key: keys)
      layer.emplace_back(key);
    return layer;
  };

  void propagate(auto& network, const double maxtime, const double timestep)
  {
    bool not_all_outputs_spiked = std::ranges::any_of(network.back(), [](auto& n){ return n.spikes.empty();});
    for(double time = 0.; time < maxtime && not_all_outputs_spiked; time += timestep)
      for(auto& layer: network)
        for(auto& n: layer)
          n.forward_propagate(time, timestep);
  }

  void clear(auto& network)
  {
    for(auto& layer: network)
      for(auto& n: layer)
        n.clear();
  }

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

  void load_sample(auto& network, const auto& sample)
  {
    auto& [input_layer, _, output_layer] = network;
    for(int input_i = 0; input_i < input_layer.size(); ++input_i)
      input_layer.at(input_i).fire(sample.input.at(input_i));
    output_layer.at(0).clamped = sample.output;
  }

  void init_network(auto& network, auto& random_gen)
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
