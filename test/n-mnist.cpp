#include<iostream>
#include<algorithm>
#include<respikeprop/load_n_mnist.hpp>
#include<respikeprop/respikeprop_store_gradients.hpp>
#include<test/xor_experiment.hpp>

namespace resp
{
  void load_n_mnist_sample(auto& network, const auto& pattern)
  {
    auto& [input_layer, _, output_layer] = network;
    for(auto& event: pattern.events)
      input_layer.at(event.x * event.y).fire(static_cast<double>(event.timestamp)/1e4); // making input within 30 timesteps

    // not doing anything with polarity.... should perhaps have 2 inputs per pixel...
    for(auto& neuron: output_layer)
      neuron.clamped = 60;
    output_layer.at(pattern.label).clamped = 40;
  }
}

// Dataset can be downloaded from https://www.garrickorchard.com/datasets/n-mnist
int main()
{
  auto seed = time(0);
  std::cout << "random seed = " << seed << std::endl;
  std::mt19937 random_gen(seed);
  using namespace resp;

  const double timestep = .1;
  const double learning_rate = 1e-2;

  // create network (convolutions????)
  //e.g.  28 * 28 x  10 x 10;
  std::array network{std::vector<Neuron>(28*28),
                     std::vector<Neuron>(10),
                     create_layer({"O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9"})};
  // and some delay lines...
  // that would be e.g. (28*28*10+10*10)*16 = 127040 synapses = 400x xor -> 400* .015 sec = 6 sec to train : - )
  // well.... times 60000 / 4 pat/pat= 15000 -> 24 hrs...
  // so somewhere between 6 secs and 24 hrs.
  init_network(network, random_gen);
  // make time delays a bit bigger {2, 4, 6, ..., 32}
  for(auto& layer: network)
    for(auto& n: layer)
      for(auto& incoming_connection: n.incoming_connections)
        for(auto& synapse: incoming_connection.synapses)
          synapse.delay *= 2;

  std::cout << "Loading spike patterns..." << std::endl;
  auto spike_patterns = load_n_mnist_training(.01, random_gen);
  std::cout << "Loaded " << spike_patterns.size() << " patterns" << std::endl;
  std::shuffle(spike_patterns.begin(), spike_patterns.end(), random_gen);

  // create training scheme
  for(int epoch = 0; epoch < 10; ++epoch)
  {
    //double sum_squared_error_epoch = 0;
    for(auto pattern: spike_patterns)
    {
      double sum_squared_error_pattern = 0;
      clear(network);
      load_n_mnist_sample(network, pattern);
        
      propagate(network, 100., timestep);
      //TODO change to all output neurons:
      //if(output_neuron.spikes.empty())
      //{
      //  std::cout << "No output spikes! Replacing with different trial. " << std::endl;
      //  trial -= 1;
      //  sum_squared_error = epoch = 1e9; break;
      //}
      for(auto& neuron: network.back())
        if(! neuron.spikes.empty())
        {
          sum_squared_error_pattern += .5 * pow(neuron.spikes.front() - neuron.clamped, 2);
          std::cout << neuron.key << " " << neuron.clamped << " " << neuron.spikes.front() << std::endl;
        }
      std::cout << sum_squared_error_pattern << std::endl;

      // Backward propagation and changing weights (no batch-mode)
      for(auto& neuron: network.back())
        neuron.compute_delta_weights(learning_rate); 
      // perhaps do the following in batch mode:
      for(auto& layer: network)
        for(auto& n: layer)
          for(auto& incoming_connection: n.incoming_connections)
            for(auto& synapse: incoming_connection.synapses)
            {
              synapse.weight += synapse.delta_weight;
              synapse.delta_weight = 0.;
            }
    }
    //std::cout << " " << epoch << " " << sum_squared_error << std::endl;
    // Stopping criterion
    //if(sum_squared_error < 1.0)
    //  break;
  }

  return 0;
}

