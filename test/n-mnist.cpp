#include<iostream>
#include<respikeprop/load_n_mnist.hpp>
#include<respikeprop/respikeprop_store_gradients.hpp>
#include<test/xor_experiment.hpp>

namespace resp
{
  void load_n_mnist_sample(auto& network, const auto& pattern)
  {
    auto& [input_layer, _, output_layer] = network;
    for(auto& event: pattern.events)
      input_layer.at(event.x * event.y).fire(static_cast<double>(event.timestamp)/1000.0);
    // not doing anything with polarity.... should perhaps have 2 inputs per pixel...
    for(auto& neuron: output_layer)
      neuron.clamped = 70;
    output_layer.at(pattern.label) = 60;
  }
}

// Dataset can be downloaded from https://www.garrickorchard.com/datasets/n-mnist
int main()
{
  using namespace resp;
  auto seed = time(0);
  std::cout << "random seed = " << seed << std::endl;
  std::mt19937 random_gen(seed);

  //std::cout << "Loading spike patterns..." << std::endl;
  //auto spike_patterns = load_n_mnist_training();
  //std::cout << "Loaded " << spike_patterns.size() << " patterns" << std::endl;

  // create network (convolutions????)
  //e.g.  28 * 28 x  10 x 10;
  std::array network{std::vector<Neuron>(28*28),
                     std::vector<Neuron>(10),
                     std::vector<Neuron>(10)};
  // and some delay lines...
  init_network(network, random_gen);
  // make time delays a bit bigger {2, 4, 6, ..., 32}
  for(auto& layer: network)
    for(auto& n: layer)
      for(auto& incoming_connection: n.incoming_connections)
        for(auto& synapse: incoming_connection.synapses)
          synapse.delay *= 2;
  // that would be e.g. (28*28*10+10*10)*16 = 127040 synapses = 400x xor -> 400* .015 sec = 6 sec to train : - )
  // well.... times 60000 / 4 pat/pat= 15000 -> 24 hrs...
  // so somewhere between 6 secs and 24 hrs.

  // create training scheme

  return 0;
}

