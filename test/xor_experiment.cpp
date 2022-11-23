#include<iostream>
#include<array>
#include<random>
#include<ctime>
#include<respikeprop/respikeprop_store_gradients.hpp>
#include<test/xor_experiment.hpp>

// Training a network to learn XOR as described in Section 4.1.

int main()
{
  //auto seed = 2; //time(0);
  std::cout << "random seed = " << seed << std::endl;
  std::mt19937 random_gen(seed);
  using namespace resp;

  const double timestep = .1;
  const double learning_rate = 1e-2;

  double avg_nr_of_epochs = 0;
  // Multiple trials for statistics
  for(int trial = 0; trial < 10; ++trial)
  {
    // Create network architecture
    std::array network{create_layer({"input 1", "input 2", "bias"}),
                       create_layer({"hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5"}),
                       create_layer({"output"})};
    init_network(network, random_gen);
    auto& output_neuron = network.back().at(0);

    // Main training loop
    for(int epoch = 0; epoch < 10000; ++epoch)
    {
      double sum_squared_error = 0;
      for(auto sample: get_xor_dataset())
      {
        clear(network);
        load_sample(network, sample);
        propagate(network, 40., timestep);
        if(output_neuron.spikes.empty())
        {
          std::cout << "No output spikes! Replacing with different trial. " << std::endl;
          trial -= 1;
          sum_squared_error = epoch = 1e9; break;
        }
        sum_squared_error += .5 * pow(output_neuron.spikes.at(0) - output_neuron.clamped, 2);

        // Backward propagation and changing weights (no batch-mode)
        network.back().at(0).compute_delta_weights(learning_rate);
        for(auto& layer: network)
          for(auto& n: layer)
            for(auto& incoming_connection: n.incoming_connections)
              for(auto& synapse: incoming_connection.synapses)
              {
                synapse.weight += synapse.delta_weight;
                synapse.delta_weight = 0.;
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

