#include<iostream>
#include<algorithm>
#include<respikeprop/load_n_mnist.hpp>
#include<range/v3/view/zip.hpp>
#include<respikeprop/respikeprop_event_based.hpp>
#include<respikeprop/create_network.hpp>

// A script that applies the respikeprop to the Neuromorphic-MNIST dataset
// (N-MNIST). The spiketrains per sample are decimated to 200 events (of the
// usual 3000~6000). Output uses time-to-first-spike. Polarity of events is not
// used.
//
// The dataset should be downloaded from
// https://www.garrickorchard.com/datasets/n-mnist
// and put in "datasets/n-mnist/".

namespace resp
{
  void load_n_mnist_sample(auto& network, auto& events, const auto& pattern)
  {
    auto& [input_layer, _, output_layer] = network;
    for(auto& event: pattern.events)
      events.predicted_spikes.emplace_back(&input_layer.at(event.x * event.y), static_cast<double>(event.timestamp)/1e4); // making input within 30 timesteps
    // not doing anything with polarity.... should perhaps have 2 inputs per pixel...
    for(auto& neuron: output_layer)
      neuron.clamped = 40;
    output_layer.at(static_cast<std::size_t>(pattern.label)).clamped = 30;
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
            synapse.weight = std::uniform_real_distribution<>(-.025, .05)(random_gen);
  }
  auto first_spike_result(auto& network)
  {
    auto first_neuron = std::ranges::min_element(network.back(), [](auto& a, auto& b) noexcept {
      if(a.spikes.empty()) return false;
      if(b.spikes.empty()) return true;
      return a.spikes.front().time < b.spikes.front().time;
    });
    return ranges::distance(network.back().begin(), first_neuron);
  }

  auto compute_loss(const auto& network)
  {
    double loss_pattern = 0;
    for(auto& neuron: network.back())
      if(! neuron.spikes.empty())
        loss_pattern += .5 * pow(neuron.spikes.front().time - neuron.clamped, 2);
    return loss_pattern;
  }
}

int main()
{
  auto seed = std::random_device()();
  std::cout << "random seed = " << seed << std::endl;
  std::mt19937 random_gen(seed);
  using namespace resp;

  const double learning_rate = 1e-4;
  const int batch_size = 10;

  // create network
  std::array network{std::vector<Neuron>(28*28),
                     std::vector<Neuron>(10),
                     std::vector<Neuron>(10)};
  init_network_n_mnist(network, random_gen);
  connect_outgoing(network);
  // time delays a bit bigger {2, 4, 6, ..., 32}
  for(auto& layer: network)
    for(auto& n: layer)
      for(auto& incoming_connection: n.incoming_connections)
        for(auto& synapse: incoming_connection.synapses)
          synapse.delay *= 2;

  // creating train and validation set
  std::cout << "Loading spike patterns..." << std::endl;
  std::vector<Pattern> spike_patterns = load_n_mnist_training(.012, random_gen); //, std::array{0, 1});
  std::ranges::shuffle(spike_patterns, random_gen);
  const auto validation_split = spike_patterns.end() - spike_patterns.size() / 5;
  auto spike_patterns_train = std::ranges::subrange(spike_patterns.begin(), validation_split);
  auto spike_patterns_validation = std::ranges::subrange(validation_split, spike_patterns.end());
  auto spike_patterns_validation_decimated = decimate_events(spike_patterns_validation, 200, random_gen);
  std::cout << "Loaded " << spike_patterns_train.size() << " training patterns" << std::endl;
  std::cout << "and    " << spike_patterns_validation_decimated.size() << " validation patterns" << std::endl;

  auto forward_propagate = [&network](auto& pattern)
  {
    clear(network);
    Events events;
    load_n_mnist_sample(network, events, pattern);
    auto not_all_outputs_spiked = [&network]()
    {
      return std::ranges::any_of(network.back(), [](const auto& n) noexcept { return n.spikes.empty();});
    };
    while(not_all_outputs_spiked() && events.active())
      events.process_event();
    return events.actual_spikes;
  };

  for(int epoch = 0; epoch < 100; ++epoch)
  {
    auto spike_patterns_decimated = decimate_events(spike_patterns_train, 200, random_gen);
    double loss_batch = 0;
    double loss_epoch = 0;
    int error_epoch = 0;
    for(const auto& [pattern_i, pattern]: ranges::views::enumerate(spike_patterns_decimated))
    {
      // forward
      auto spikes = forward_propagate(pattern);

      // update logs
      loss_batch += compute_loss(network);
      if(first_spike_result(network) != pattern.label) error_epoch++;

      // backprop
      backprop(spikes, learning_rate);

      // per batch change weights and report logs
      if((pattern_i + 1) % batch_size == 0 || pattern_i + 1 == spike_patterns_decimated.size())
      {
        for(auto& layer: network)
          for(auto& n: layer)
            for(auto& incoming_connection: n.incoming_connections)
              for(auto& synapse: incoming_connection.synapses)
              {
                synapse.weight += synapse.delta_weight;
                synapse.delta_weight = 0.;
              }
        std::cout << "batch loss after pattern " << pattern_i + 1 << " " << loss_batch / (pattern_i % batch_size + 1) << std::endl;
        loss_epoch += loss_batch;
        loss_batch = 0;
      }
    }
    // report epoch logs
    std::cout << "train loss  after epoch "<< epoch  << " " << loss_epoch / spike_patterns_train.size() << std::endl;
    std::cout << "train error after epoch "<< epoch  << " " << 100 * double(error_epoch) / spike_patterns_train.size() << " %" << std::endl;
    {
      double loss_validation = 0;
      int error_validation = 0;
      for(const auto& pattern: spike_patterns_validation_decimated)
      {
        forward_propagate(pattern);
        loss_validation += compute_loss(network);
        if(first_spike_result(network) != pattern.label) ++error_validation;
      }
      std::cout << "validation loss  after epoch "<< epoch  << " " << loss_validation / spike_patterns_validation.size() << std::endl;
      std::cout << "validation error after epoch "<< epoch  << " " << 100 * double(error_validation) / spike_patterns_validation.size() << " %" << std::endl;
    }
  }

  return 0;
}

