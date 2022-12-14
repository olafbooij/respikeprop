#pragma once

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

  void clear(auto& network)
  {
    for(auto& layer: network)
      for(auto& n: layer)
        n.clear();
  }
}

