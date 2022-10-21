#include<iostream>
#include<random>
#include<ctime>
#include<array>
#include<vector>
#include<cassert>
#include<respikeprop/forward_backward_exhaust.hpp>

namespace resp
{

  void connect_neurons(auto& pre, auto& post, auto&& random_weight, auto& random_gen)
  {
    for(auto delay_i = 16; delay_i--;)
    {
      post.incoming_synapses.emplace_back(pre, random_weight(random_gen), delay_i + 1.0);
      pre.post_neuron_ptrs.emplace_back(&post);
    }
  };

  void connect_layers(auto& pre_layer, auto& post_layer, const double min_weight, const double max_weight, auto& random_gen)
  {
    std::uniform_real_distribution<> random_weight(min_weight, max_weight);
    for(auto& pre: pre_layer)
      for(auto& post: post_layer)
        connect_neurons(pre, post, random_weight, random_gen);
  };

  auto create_layer(const std::vector<std::string>&& keys)
  {
    std::vector<Neuron> layer;
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
  std::mt19937 random_gen(time(0));
  using namespace resp;
  std::array network{create_layer({"input 1", "input 2", "bias"}),
                     create_layer({"hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5"}),
                     create_layer({"output"})};
  auto& [input_layer, hidden_layer, output_layer] = network;

  connect_layers(input_layer, hidden_layer, -.5, 1., random_gen);

  // create synapses with only positive weights for 4 hidden neurons
  for(auto pre_it = hidden_layer.begin(); pre_it != hidden_layer.end() - 1; ++pre_it)
    connect_neurons(*pre_it, output_layer.at(0), std::uniform_real_distribution<>(0., 1.), random_gen);
  // and with only negative weights for the last hidden neuron
  connect_neurons(hidden_layer.back(), output_layer.front(), std::uniform_real_distribution<>(-.5, 0.), random_gen);

  const double timestep = .01;
  const double learning_rate = 1e-2;
  
  struct sample
  {
    std::array<double, 3> input;
    double output;
  };
  std::array<sample, 4> dataset{{
    {{0., 0., 0.}, 16.},
    {{0., 6., 0.}, 10.},
    {{6., 0., 0.}, 10.},
    {{6., 6., 0.}, 16.}
  }};
  
  for(int trial = 0; trial < 100; ++trial)
  {
    for(auto& n: hidden_layer)
      for(auto& synapse: n.incoming_synapses) 
        synapse.weight = std::uniform_real_distribution<>(-.5, 1.0)(random_gen);
    for(auto& synapse: output_layer.front().incoming_synapses)
      if(&(synapse.pre) == &(hidden_layer.back()))
        synapse.weight = std::uniform_real_distribution<>(-.5, 0.)(random_gen);
      else
        synapse.weight = std::uniform_real_distribution<>(0., 1.)(random_gen);
  // and with only negative weights for the last hidden neuron
  for(int epoch = 0; epoch < 100; ++epoch)
  {
    double sum_squared_error = 0;
    for(auto sample: dataset)
    {
  // first XOR pattern
      clear(network);
      for(int input_i = 0; input_i < input_layer.size(); ++input_i)
        input_layer.at(input_i).fire(sample.input.at(input_i));
      output_layer.at(0).clamped = 16.;
      propagate(network, timestep);
      //std::cout << .5 * pow(output_layer.at(0).spikes.at(0) - output_layer.at(0).clamped, 2) << std::endl;
      sum_squared_error += .5 * pow(output_layer.at(0).spikes.at(0) - output_layer.at(0).clamped, 2);
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
    if(sum_squared_error < 1.0)
      break;
  }
  }


  return 0;
}

