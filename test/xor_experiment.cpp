#include<iostream>
#include<array>
#include<vector>
#include<random>
#include<ctime>
#include<respikeprop/respikeprop_store_gradients.hpp>

// Training a network to learn XOR as described in Section 4.1.

// Some helpfull functions
namespace resp
{

  void connect_neurons(auto& pre, auto& post)
  {
    auto& incoming_connection = post.incoming_connections.emplace_back(&pre);
    for(auto delay_i = 16; delay_i--;)
    {
      incoming_connection.synapses.emplace_back(.0, delay_i + 1.0);
    }
    pre.post_neuron_ptrs.emplace_back(&post);
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
  std::mt19937 random_gen(time(0));
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

  int avg_nr_of_epochs = 0;
  // Multiple trials for statistics
  for(int trial = 0; trial < 10; ++trial)
  {
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
    // Main training loop
    for(int epoch = 0; epoch < 1000; ++epoch)
    {
      double sum_squared_error = 0;
      for(auto sample: dataset)
      {
        clear(network);
        // Load data sample
        for(int input_i = 0; input_i < input_layer.size(); ++input_i)
          input_layer.at(input_i).fire(sample.input.at(input_i));
        output_layer.at(0).clamped = sample.output;
        // Forward propagation
        propagate(network, 40., timestep);
        // Backward propagation and changing weights (no batch-mode)
        sum_squared_error += .5 * pow(output_layer.at(0).spikes.at(0).time - output_layer.at(0).clamped, 2);
        for(auto& layer: network)
          for(auto& n: layer)
          {
            n.compute_delta_weights(learning_rate);
            for(auto& incoming_connection: n.incoming_connections)
              for(auto& synapse: incoming_connection.synapses)
              {
                synapse.weight += synapse.delta_weight;
                synapse.delta_weight = 0.;
              }
          }
      }
      std::cout << trial << " " << epoch << " " << sum_squared_error << std::endl;
      // Stopping criterion
      if(sum_squared_error < 1.0)
      {
        avg_nr_of_epochs = (avg_nr_of_epochs * trial + epoch) / (trial + 1);
        break;
      }
    }
  }
  std::cout << "Average nr of epochs = " << avg_nr_of_epochs << std::endl;

  return 0;
}

