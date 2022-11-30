#include<iostream>
#include<algorithm>
#include<respikeprop/load_n_mnist.hpp>
#include<respikeprop/respikeprop_store_gradients.hpp>
#include<test/xor_experiment.hpp>

// A script that applies the respikeprop to the N-MNIST dataset. Here we only
// learn 0's and 1's. The spiketrains per sample are decimated to 200 events
// (of the usual 3000~6000). Output uses time-to-first-spike. Polarity of
// events is not used. Training time is approx 3 minutes (on a simple laptop).
//
// The dataset should be downloaded from
// https://www.garrickorchard.com/datasets/n-mnist
// and put in "datasets/n-mnist/".

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

    for(auto& layer: network)
      for(auto& n: layer)
        for(auto& incoming_connection: n.incoming_connections)
          for(auto& synapse: incoming_connection.synapses)
            synapse.weight = std::uniform_real_distribution<>(-.5, 1.0)(random_gen);
  }
}

int main()
{
  auto seed = time(0);
  std::cout << "random seed = " << seed << std::endl;
  std::mt19937 random_gen(seed);
  using namespace resp;

  const double timestep = .1;
  const double learning_rate = 1e-4;
  const int batch_size = 10;

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

  // creating train and validation set
  std::cout << "Loading spike patterns..." << std::endl;
  std::vector<Pattern> spike_patterns = load_n_mnist_training(.012, random_gen, std::array{0, 1});
  std::ranges::shuffle(spike_patterns, random_gen);
  const size_t validation_size = spike_patterns.size() / 5;
  std::vector<Pattern> spike_patterns_train(spike_patterns.begin(), spike_patterns.end() - validation_size);
  std::vector<Pattern> spike_patterns_validation(spike_patterns.end() - validation_size, spike_patterns.end());
  auto spike_patterns_validation_decimated = decimate_events(spike_patterns_validation, 200, random_gen);
  std::cout << "Loaded " << spike_patterns_train.size() << " training patterns" << std::endl;
  std::cout << "and    " << spike_patterns_validation_decimated.size() << " validation patterns" << std::endl;

  for(int epoch = 0; epoch < 10; ++epoch)
  {
    auto spike_patterns_decimated = decimate_events(spike_patterns_train, 200, random_gen);
    double loss_batch = 0;
    double loss_epoch = 0;
    int error_epoch = 0;
    for(const auto& [pattern_i, pattern]: ranges::views::enumerate(spike_patterns_decimated))
    {
      clear(network);
      load_n_mnist_sample(network, pattern);
      propagate(network, 100., timestep);

      double loss_pattern = 0;
      for(auto& neuron: network.back())
        if(! neuron.spikes.empty())
          loss_pattern += .5 * pow(neuron.spikes.front() - neuron.clamped, 2);
      loss_batch += loss_pattern;
      loss_epoch += loss_pattern;
      auto output = std::ranges::min_element(network.back(), [](auto& a, auto& b){return a.spikes.front() < b.spikes.front();});
      if(ranges::distance(network.back().begin(), output) != pattern.label)
        ++error_epoch;

      // Backpropagation and changing weights
      for(auto& neuron: network.back())
        neuron.compute_delta_weights(learning_rate);
      if((pattern_i + 1) % batch_size == 0)
      {
        for(auto& layer: network)
          for(auto& n: layer)
            for(auto& incoming_connection: n.incoming_connections)
              for(auto& synapse: incoming_connection.synapses)
              {
                synapse.weight += synapse.delta_weight;
                synapse.delta_weight = 0.;
              }
        std::cout << "batch loss after pattern " << pattern_i + 1 << " " << loss_batch / batch_size << std::endl;
        loss_batch = 0;
      }
    }
    std::cout << "train loss after epoch  "<< epoch  << " " << loss_epoch / spike_patterns_train.size() << std::endl;
    std::cout << "train error after epoch "<< epoch  << " " << 100 * double(error_epoch) / spike_patterns_train.size() << " %" << std::endl;
    {
      double loss_validation = 0;
      int error_validation = 0;
      for(const auto& pattern: spike_patterns_validation_decimated)
      {
        clear(network);
        load_n_mnist_sample(network, pattern);
        propagate(network, 100., timestep);
        for(auto& neuron: network.back())
          if(! neuron.spikes.empty())
            loss_validation += .5 * pow(neuron.spikes.front() - neuron.clamped, 2);
        auto output = std::ranges::min_element(network.back(), [](auto& a, auto& b){return a.spikes.front() < b.spikes.front();});
        if(ranges::distance(network.back().begin(), output) != pattern.label)
          ++error_validation;
      }
      std::cout << "validation loss  after epoch "<< epoch  << " " << loss_validation / spike_patterns_validation.size() << std::endl;
      std::cout << "validation error after epoch "<< epoch  << " " << 100 * double(error_validation) / spike_patterns_validation.size() << " %" << std::endl;
    }
  }

  return 0;
}

