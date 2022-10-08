#include<iostream>
#include<random>
#include<cassert>
#include<respikeprop/neuron.hpp>

namespace resp
{

  using synapse_ptr = std::shared_ptr<synapse>;
  using neuron_ptr = std::shared_ptr<neuron>;

  auto connect_neurons(auto pre, auto post, auto random_weight, auto random_gen)
  {
    std::vector<synapse_ptr> synapses;
    for(auto delay_i = 16; delay_i--;)
      synapses.emplace_back(make_synapse(post, pre, random_weight(random_gen), delay_i + 1.0));
    return synapses;
  };

  auto connect_layers(auto pre_layer, auto post_layer, double min_weight, double max_weight, auto random_gen)
  {
    std::vector<synapse_ptr> synapses;
    std::uniform_real_distribution<> random_weight(min_weight, max_weight);
    for(auto pre: pre_layer)
      for(auto post: post_layer)
        for(auto synapse: connect_neurons(pre, post, random_weight, random_gen)) 
          synapses.emplace_back(synapse);
    return synapses;
  };

}

int main()
{
  std::mt19937 random_gen(0);
  using namespace resp;
  double timestep = .1;
  auto input_layer = std::vector(3, make_neuron(timestep));
  auto hidden_layer = std::vector(5, make_neuron(timestep));
  auto output_layer = std::vector(1, make_neuron(timestep));
  std::vector<neuron_ptr> all_neurons;
  for(auto neuron: input_layer) all_neurons.emplace_back(neuron);
  for(auto neuron: hidden_layer) all_neurons.emplace_back(neuron);
  for(auto neuron: output_layer) all_neurons.emplace_back(neuron);

  std::vector<synapse_ptr> synapses_to_hidden = connect_layers(input_layer, hidden_layer, -.5, 1., random_gen);
  std::vector<synapse_ptr> synapses_to_output = connect_layers(hidden_layer, output_layer, 0., 1., random_gen);
  // hack to make the last hidden only inhibitory 
  for(auto synapse_weak: hidden_layer.at(4)->outgoing_synapses)
  {
    auto synapse = synapse_weak.lock();
    synapse->weight *= -.5;
  }

  // first XOR pattern
  fire(*(input_layer.at(0)), 0.);
  fire(*(input_layer.at(1)), 0.);
  fire(*(input_layer.at(2)), 0.);
  
  for(double time = 0.; time < 40.; time += timestep)
    for(auto neuron: all_neurons)
      forward_propagate(*neuron, time);

  
  for(auto time: input_layer.at(0)->spike_times)
    std::cout << time << std::endl;
  std::cout << std::endl;
  for(auto time: hidden_layer.at(0)->spike_times)
    std::cout << time << std::endl;
  std::cout << std::endl;
  for(auto time: output_layer.at(0)->spike_times)
    std::cout << time << std::endl;

  return 0;
}

