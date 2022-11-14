#include<iostream>
#include<array>
#include<vector>
#include<random>
#include<ctime>
#include<respikeprop/respikeprop_reference_impl.hpp>
#include<test/xor_experiment.hpp>

// Training a network to learn XOR as described in Section 4.1.

// Some helpfull functions that differ from the ones in xor_experiment.hpp
namespace resp
{
namespace ref
{

  void connect_neurons(auto& pre, auto& post)
  {
    for(auto delay_i = 16; delay_i--;)
      post.incoming_synapses.emplace_back(pre, .0, delay_i + 1.0);
    pre.post_neuron_ptrs.emplace_back(&post);
  };

  void connect_layers(auto& pre_layer, auto& post_layer)
  {
    for(auto& pre: pre_layer)
      for(auto& post: post_layer)
        ref::connect_neurons(pre, post);
  };

  void clear(auto& network)
  {
    for(auto& layer: network)
      for(auto& n: layer)
        n.spikes.clear();
  }
  void init_network(auto& network, auto& random_gen)
  {
    auto& [input_layer, hidden_layer, output_layer] = network;
    ref::connect_layers(input_layer, hidden_layer);
    ref::connect_layers(hidden_layer, output_layer);

    // Set random weights
    for(auto& n: hidden_layer)
      for(auto& synapse: n.incoming_synapses)
        synapse.weight = std::uniform_real_distribution<>(-.5, 1.0)(random_gen);
    for(auto& synapse: output_layer.front().incoming_synapses)
      if(synapse.pre->key == "hidden 5")
        synapse.weight = std::uniform_real_distribution<>(-.5, 0.)(random_gen);
      else
        synapse.weight = std::uniform_real_distribution<>(0., 1.)(random_gen);
  }

}
}

int main()
{
  auto seed = time(0);
  std::cout << "random seed = " << seed << std::endl;
  std::mt19937 random_gen(seed);
  using namespace resp;

  const double timestep = .1;
  const double learning_rate = 1e-2;

  int avg_nr_of_epochs = 0;
  // Multiple trials for statistics
  for(int trial = 0; trial < 10; ++trial)
  {
    std::array network{create_layer({"input 1", "input 2", "bias"}),
                       create_layer({"hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5"}),
                       create_layer({"output"})};
    ref::init_network(network, random_gen);

    // Main training loop
    for(int epoch = 0; epoch < 1000; ++epoch)
    {
      double sum_squared_error = 0;
      for(auto sample: get_xor_dataset())
      {
        ref::clear(network);
        load_sample(network, sample);
        propagate(network, 40., timestep);
        sum_squared_error += .5 * pow(network.back().at(0).spikes.at(0) - network.back().at(0).clamped, 2);

        // Backward propagation and changing weights (no batch-mode)
        for(auto& layer: network)
          for(auto& n: layer)
          {
            n.compute_delta_weights(learning_rate);
            for(auto& synapse: n.incoming_synapses)
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

