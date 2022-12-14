#include<iostream>
#include<array>
#include<random>
#include<range/v3/view/zip.hpp>
#include<respikeprop/respikeprop_event_based.hpp>
#include<respikeprop/create_network.hpp>
#include<respikeprop/xor_experiment.hpp>

// Training a recurrent network to learn XOR.
// Consisting of 8 neurons all connected with each other. Three are apointed as
// input and one as output. Training is still unstable, with commonly no output
// fire. A clear TODO...

// Some helpfull functions that differ from the ones in xor_experiment.hpp
namespace resp
{
namespace rec
{
  void init_xor_network(auto& network, auto& random_gen)
  {
    // using layers, so to easily reuse xor_experiment script.
    auto& [input_layer, hidden_layer, output_layer] = network;
    connect_layers(input_layer, input_layer);
    connect_layers(input_layer, hidden_layer);
    connect_layers(hidden_layer, input_layer);
    connect_layers(hidden_layer, hidden_layer);
    connect_layers(input_layer, output_layer);
    connect_layers(hidden_layer, output_layer);
    // not necessary to connect output back to others, because only its first
    // spike is important.

    connect_outgoing(network);

    // Set random weights
    for(auto& layer: network)
      for(auto& n: layer)
        for(auto& incoming_connection: n.incoming_connections)
          for(auto& synapse: incoming_connection.synapses)
            synapse.weight = std::uniform_real_distribution<>(-.5, 1.0)(random_gen);
  }
}
}

int main()
{
  auto seed = std::random_device()();
  std::cout << "random seed = " << seed << std::endl;
  std::mt19937 random_gen(seed);
  using namespace resp;

  const double learning_rate = 1e-2;

  double avg_nr_of_epochs = 0;
  // Multiple trials for statistics
  for(int trial = 0; trial < 10; ++trial)
  {
    // Create network architecture
    // with same number of synapses (4*(4+1)*16=320), as regular xor network
    // (16*(3*5+5*1)=320)
    std::array network{create_layer({"input 1", "input 2", "bias"}),
                       create_layer({"hidden 1"}),
                       create_layer({"output"})};

    rec::init_xor_network(network, random_gen);
    auto& output_neuron = network.back().at(0);

    // Main training loop
    for(int epoch = 0; epoch < 1000; ++epoch)
    {
      double sum_squared_error = 0;
      for(auto sample: get_xor_dataset())
      {
        clear(network);
        Events events;
        load_sample(network, events, sample);
        while(output_neuron.spikes.empty() && events.active() && (events.synapse_spikes.empty() || events.synapse_spikes.top().time < 20))
          events.process_event();
        if(output_neuron.spikes.empty())
        {
          std::cout << "No output spikes! Replacing with different trial. " << std::endl;
          trial -= 1;
          sum_squared_error = epoch = 1e9; break;
        }

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
      std::cout << trial << " " << epoch << " " << sum_squared_error << std::endl;
      // Stopping criterion
      if(sum_squared_error < 1.0)
      {
        std::cout << trial << " " << epoch << std::endl;
        avg_nr_of_epochs = (avg_nr_of_epochs * trial + epoch) / (trial + 1);
        break;
      }
    }
  }
  std::cout << "Average nr of epochs = " << avg_nr_of_epochs << std::endl;

  return 0;
}

