#include<iostream>
#include<random>
#include<array>
#include<vector>
#include<cassert>
#include<respikeprop/forward_backward_exhaust.hpp>

namespace resp
{

  void connect_neurons(const auto& pre, auto& post, auto&& random_weight, auto& random_gen)
  {
    for(auto delay_i = 16; delay_i--;)
      post.incoming_synapses.emplace_back(pre, random_weight(random_gen), delay_i + 1.0);
  };

  void connect_layers(const auto& pre_layer, auto& post_layer, const double min_weight, const double max_weight, auto& random_gen)
  {
    std::uniform_real_distribution<> random_weight(min_weight, max_weight);
    for(auto& pre: pre_layer)
      for(auto& post: post_layer)
        connect_neurons(pre, post, random_weight, random_gen);
  };

  auto create_layer(const std::vector<std::string>&& keys)
  {
    std::vector<neuron> layer;
    for(const auto& key: keys)
      layer.emplace_back(key);
    return layer;
  };

}

void propagate(auto& network, const double timestep)
{
  for(double time = 0.; time < 40.; time += timestep)
    for(auto& layer: network)
      for(auto& n: layer)
        n.forward_propagate(time);
}
void clear(auto& network)
{
  for(auto& layer: network)
    for(auto& n: layer)
      n.spikes.clear();
}

int main()
{
  std::mt19937 random_gen(0);
  using namespace resp;
  std::array network{create_layer({"input 1", "input 2", "bias"}),
                     create_layer({"hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5"}),
                     create_layer({"output"})};
  auto& [input_layer, hidden_layer, output_layer] = network;

  connect_layers(input_layer, hidden_layer, -.5, 1., random_gen);

  // create synapses with only positive weights for 4 hidden neurons
  for(auto pre_it = hidden_layer.begin(); pre_it != hidden_layer.end() - 1; ++pre_it)
    connect_neurons(*pre_it, output_layer.at(0), std::uniform_real_distribution<>(0., 1.), random_gen);
  // and with only negative weights for the last hidden neuaron
  connect_neurons(hidden_layer.back(), output_layer.front(), std::uniform_real_distribution<>(-.5, 0.), random_gen);

  // first XOR pattern
  input_layer.at(0).fire(0.);
  input_layer.at(1).fire(0.);
  input_layer.at(2).fire(0.);

  const double timestep = .001;
  
  propagate(network, timestep);
  
  assert(! output_layer.at(0).spikes.empty());
  // fixture
  assert(fabs(output_layer.at(0).spikes.at(5) - 17.3) < 0.2);

  // check gradient:
  auto error_before = output_layer.at(0).spikes.at(5);

  // small weight change
  const double small = 0.03;
  auto& synapse_changed = hidden_layer.at(2).incoming_synapses.at(30);
  synapse_changed.weight += small;

  clear(network);
  input_layer.at(0).fire(0.);
  input_layer.at(1).fire(0.);
  input_layer.at(2).fire(0.);
  propagate(network, timestep);

  // check error change equals weight change
  auto error_after = output_layer.at(0).spikes.at(5);
  std::cout << error_before << std::endl;
  std::cout << error_after << std::endl;
  std::cout << error_after - error_before << std::endl;
  std::cout << (error_after - error_before) / small << std::endl;

  const double learning_rate = 1e-2;
  hidden_layer.at(2).compute_delta_weights(learning_rate);
  std::cout << synapse_changed.delta_weight << std::endl;
  //std::cout << timestep / small << std::endl;
  assert(fabs((error_after - error_before) / small - synapse_changed.delta_weight / learning_rate) < (timestep / small));

  //for(auto time: input_layer.at(0).spikes)
  //  std::cout << time << std::endl;
  //std::cout << std::endl;
  //for(auto time: hidden_layer.at(0).spikes)
  //  std::cout << time << std::endl;
  //std::cout << std::endl;
  //for(auto time: output_layer.at(0).spikes)
  //  std::cout << time << std::endl;

  return 0;
}

