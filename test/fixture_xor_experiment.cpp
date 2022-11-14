#include<iostream>
#include<array>
#include<vector>
#include<random>
#include<ctime>
#include<respikeprop/respikeprop_store_gradients.hpp>

// Fixture. Delta weights were taken from run of reference implementation.

// Some helpfull functions
namespace resp
{

  void connect_neurons(auto& pre, auto& post)
  {
    auto& incoming_connection = post.incoming_connections.emplace_back(&pre);
    for(auto delay_i = 16; delay_i--;)
      incoming_connection.synapses.emplace_back(.0, delay_i + 1.0, 0.);
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
    for(double time = 0.; time < maxtime; time += timestep)
      for(auto& layer: network)
        for(auto& n: layer)
          n.forward_propagate(time);
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

}

int main()
{
  std::mt19937 random_gen(2);
  using namespace resp;

  const double timestep = .1;
  const double learning_rate = 1e-2;

  // Create network architecture
  std::array network{create_layer({"input 1", "input 2", "bias"}),
                     create_layer({"hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5"}),
                     create_layer({"output"})};
  auto& [input_layer, hidden_layer, output_layer] = network;

  connect_layers(input_layer, hidden_layer);
  connect_layers(hidden_layer, output_layer);

  auto dataset = get_xor_dataset();

  // Set random weights
  for(auto& n: hidden_layer)
    for(auto& incoming_connection: n.incoming_connections)
      for(auto& synapse: incoming_connection.synapses)
        synapse.weight = std::uniform_real_distribution<>(-.5, 1.0)(random_gen);

  for(auto& incoming_connection: output_layer.front().incoming_connections)
  {
    if(incoming_connection.neuron->key == "hidden 5")
      for(auto& synapse: incoming_connection.synapses)
        synapse.weight = std::uniform_real_distribution<>(-.5, 0.)(random_gen);
    else
      for(auto& synapse: incoming_connection.synapses)
        synapse.weight = std::uniform_real_distribution<>(0., 1.)(random_gen);
  }

  auto sample = dataset.at(0);

  // Load data sample
  for(int input_i = 0; input_i < input_layer.size(); ++input_i)
    input_layer.at(input_i).fire(sample.input.at(input_i));
  // Forward propagation
  propagate(network, 40., timestep);
  // Backward propagation
  output_layer.at(0).clamped = sample.output;
  output_layer.at(0).compute_delta_weights(learning_rate);

  assert(fabs(output_layer.at(0).incoming_connections.at(2).synapses.at(14).delta_weight - -0.0319112) < 1e-5);
  assert(fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(15).delta_weight - -0.0575665) < 1e-5);
  assert(fabs(output_layer.at(0).incoming_connections.at(1).synapses.at(11).delta_weight - -0.0092582) < 1e-5);
  assert(fabs(hidden_layer.at(0).incoming_connections.at(0).synapses.at(15).delta_weight - -0.014726) < 1e-5);
  assert(fabs(hidden_layer.at(2).incoming_connections.at(1).synapses.at(10).delta_weight - -0.000471913) < 1e-5);
  return 0;
}

