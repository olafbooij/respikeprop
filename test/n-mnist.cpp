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
      neuron.clamped = 40;
    output_layer.at(pattern.label).clamped = 30;
  }
  void init_network_n_mnist(auto& network, auto& random_gen)
  {
    auto& [input_layer, hidden_layer, output_layer] = network;
    connect_layers(input_layer, hidden_layer);
    connect_layers(hidden_layer, output_layer);

    // Set random weights
    for(auto& layer: network)
      for(auto& n: layer)
        for(auto& incoming_connection: n.incoming_connections)
          for(auto& synapse: incoming_connection.synapses)
            synapse.weight = std::uniform_real_distribution<>(-.5, 1.0)(random_gen);
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
  const double learning_rate = 1e-4;

  // create network
  std::array network{std::vector<Neuron>(28*28),
                     std::vector<Neuron>(10),
                     create_layer({"O0", "O1"})};
                     //create_layer({"O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9"})};
  init_network_n_mnist(network, random_gen);
  for(auto& layer: network)
    for(auto& n: layer)
      for(auto& incoming_connection: n.incoming_connections)
        for(auto& synapse: incoming_connection.synapses)
        {
          synapse.delay *= 2;     // time delays a bit bigger {2, 4, 6, ..., 32}
          synapse.weight *= .05;  // weights smaller
        }

  std::cout << "Loading spike patterns..." << std::endl;
  std::vector<Pattern> spike_patterns = load_n_mnist_training(.011, random_gen, std::array{0, 1});
  std::cout << "Loaded " << spike_patterns.size() << " patterns" << std::endl;
  std::ranges::shuffle(spike_patterns, random_gen);
  const size_t test_size = spike_patterns.size() / 10;
  std::vector<Pattern> spike_patterns_train(spike_patterns.begin(), spike_patterns.end() - test_size);
  std::vector<Pattern> spike_patterns_test(spike_patterns.end() - test_size, spike_patterns.end());
  auto spike_patterns_test_decimated = decimate_events(spike_patterns_test, 200, random_gen);

  // create training scheme
  for(int epoch = 0; epoch < 10; ++epoch)
  {
    auto spike_patterns_decimated = decimate_events(spike_patterns_train, 200, random_gen);
    //double sum_squared_error_epoch = 0;
    const int batch_size = 10;
    double sum_squared_error_batch = 0;
    double sum_squared_error_epoch = 0;
    for(const auto& [pattern_i, pattern]: ranges::views::enumerate(spike_patterns_decimated))
    {
      double sum_squared_error_pattern = 0;
      clear(network);
      load_n_mnist_sample(network, pattern);
      propagate(network, 100., timestep);

      for(auto& neuron: network.back())
        if(! neuron.spikes.empty())
        {
          sum_squared_error_pattern += .5 * pow(neuron.spikes.front() - neuron.clamped, 2);
          //std::cout << neuron.key << " " << neuron.clamped << " " << neuron.spikes.front() << std::endl;
        }
      //std::cout << sum_squared_error_pattern << std::endl;
      sum_squared_error_batch += sum_squared_error_pattern;
      sum_squared_error_epoch += sum_squared_error_pattern;

      // Backward propagation and changing weights (no batch-mode)
      for(auto& neuron: network.back())
        neuron.compute_delta_weights(learning_rate);
      if((pattern_i + 1) % batch_size == 0)
      {
        // perhaps do the following in batch mode:
        for(auto& layer: network)
          for(auto& n: layer)
            for(auto& incoming_connection: n.incoming_connections)
              for(auto& synapse: incoming_connection.synapses)
              {
                synapse.weight += synapse.delta_weight;
                synapse.delta_weight = 0.;
              }
        std::cout << pattern_i << " " << sum_squared_error_batch / batch_size << std::endl;
        sum_squared_error_batch = 0;
      }
    }
    std::cout << epoch  << " train error " << sum_squared_error_epoch / spike_patterns_train.size() << std::endl;
    // Stopping criterion
    //if(sum_squared_error < 1.0)
    //  break;
    {
      double sum_squared_error_test = 0;
      for(const auto& pattern: spike_patterns_test_decimated)
      {
        clear(network);
        load_n_mnist_sample(network, pattern);
        propagate(network, 100., timestep);
        for(auto& neuron: network.back())
          if(! neuron.spikes.empty())
            sum_squared_error_test += .5 * pow(neuron.spikes.front() - neuron.clamped, 2);
      }
      std::cout << epoch  << " test  error " << sum_squared_error_test / spike_patterns_test_decimated.size() << std::endl;
    }
  }

  return 0;
}

