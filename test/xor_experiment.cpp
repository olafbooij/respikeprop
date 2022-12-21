#include<iostream>
#include<array>
#include<random>
#include<range/v3/view/zip.hpp>
#include<respikeprop/respikeprop_event_based.hpp>
#include<respikeprop/create_network.hpp>
#include<respikeprop/xor_experiment.hpp>

// Training a network to learn XOR as described in Section 4.1.

int main()
{
  auto seed = 3775610138; //std::random_device()();
  std::cout << "random seed = " << seed << std::endl;
  std::mt19937 random_gen(seed);
  using namespace resp;

  const double learning_rate = 1e-2;

  double avg_nr_of_epochs = 0;
  // Multiple trials for statistics
  for(int trial = 0; trial < 100; ++trial)
  {
    // Create network architecture
    std::array network{create_layer({"input 1", "input 2", "bias"}),
                       create_layer({"hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5"}),
                       create_layer({"output"})};
    init_xor_network(network, random_gen);
    connect_outgoing(network);
    auto& output_neuron = network.back().at(0);

    // Main training loop
    for(int epoch = 0; epoch < 10000; ++epoch)
    {
      double sum_squared_error = 0;
      for(auto sample: get_xor_dataset())
      {
        clear(network);
        Events events;
        load_sample(network, events, sample);
        while(output_neuron.spikes.empty() && events.active())
          events.process_event();
        if(output_neuron.spikes.empty())
          output_neuron.spike(100);  // bit of a hack to force backprop
        sum_squared_error += .5 * pow(output_neuron.spikes.at(0).time - output_neuron.clamped, 2);

        // Backward propagation and changing weights (no batch-mode)
        backprop(events.actual_spikes, learning_rate);
        for(auto& layer: network)
          for(auto& n: layer)
            for(auto& incoming_connection: n.incoming_connections)
              for(auto& synapse: incoming_connection.synapses)
              {
                synapse.weight += synapse.delta_weight;
                synapse.delta_weight = 0.;
              }
      }
      //std::cout << trial << " " << epoch << " " << sum_squared_error << std::endl;
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

