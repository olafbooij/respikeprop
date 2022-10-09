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

  auto create_layer(double timestep, std::vector<std::string>&& keys)
  {
    std::vector<neuron_ptr> layer;
    for(auto key: keys)
      layer.emplace_back(make_neuron(timestep, key));
    return layer;
  };

}

int main()
{
  std::mt19937 random_gen(0);
  using namespace resp;
  double timestep = .1;
  auto input_layer = create_layer(timestep, {"input 1", "input 2", "bias"});
  auto hidden_layer = create_layer(timestep, {"hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5"});
  auto output_layer = create_layer(timestep, {"output"});
  std::vector<neuron_ptr> all_neurons;
  for(auto n: input_layer) all_neurons.emplace_back(n);
  for(auto n: hidden_layer) all_neurons.emplace_back(n);
  for(auto n: output_layer) all_neurons.emplace_back(n);

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
    for(auto n: all_neurons)
      forward_propagate(*n, time);
  
  assert(! output_layer.at(0)->spike_times.empty());

  // fixture
  assert(fabs(output_layer.at(0)->spike_times.at(5) - 19.7) < 0.2);

  //for(auto time: input_layer.at(0)->spike_times)
  //  std::cout << time << std::endl;
  //std::cout << std::endl;
  //for(auto time: hidden_layer.at(0)->spike_times)
  //  std::cout << time << std::endl;
  //std::cout << std::endl;
  //for(auto time: output_layer.at(0)->spike_times)
  //  std::cout << time << std::endl;

  return 0;
}
