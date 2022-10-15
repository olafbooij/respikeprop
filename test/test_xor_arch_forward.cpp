#include<iostream>
#include<random>
#include<cassert>
#include<respikeprop/forward_exhaust.hpp>

namespace resp
{

  using neuron_ptr = std::shared_ptr<neuron>;

  auto connect_neurons(auto pre, auto post, auto random_weight, auto random_gen)
  {
    for(auto delay_i = 16; delay_i--;)
      post->incoming_synapses.emplace_back(pre, random_weight(random_gen), delay_i + 1.0);
  };

  auto connect_layers(auto pre_layer, auto post_layer, double min_weight, double max_weight, auto random_gen)
  {
    std::uniform_real_distribution<> random_weight(min_weight, max_weight);
    for(auto pre: pre_layer)
      for(auto post: post_layer)
        connect_neurons(pre, post, random_weight, random_gen);
  };

  auto create_layer(std::vector<std::string>&& keys)
  {
    std::vector<neuron_ptr> layer;
    for(auto key: keys)
      layer.emplace_back(make_neuron(key));
    return layer;
  };

}

int main()
{
  std::mt19937 random_gen(0);
  using namespace resp;
  double timestep = .1;
  auto input_layer = create_layer({"input 1", "input 2", "bias"});
  auto hidden_layer = create_layer({"hidden 1", "hidden 2", "hidden 3", "hidden 4", "hidden 5"});
  auto output_layer = create_layer({"output"});
  std::vector<neuron_ptr> all_neurons;
  for(auto n: input_layer) all_neurons.emplace_back(n);
  for(auto n: hidden_layer) all_neurons.emplace_back(n);
  for(auto n: output_layer) all_neurons.emplace_back(n);

  connect_layers(input_layer, hidden_layer, -.5, 1., random_gen);
  // create synapses with only positive weights for 4 hidden neurons
  connect_layers(std::vector<neuron_ptr>(hidden_layer.begin(), hidden_layer.end() - 1), output_layer, 0., 1., random_gen);
  // and with only negative weights for the last hidden neuaron
  std::uniform_real_distribution<> random_weight(-.5, 0.);
  connect_neurons(hidden_layer.back(), output_layer.front(), random_weight, random_gen);

  // first XOR pattern
  input_layer.at(0)->fire(0.);
  input_layer.at(1)->fire(0.);
  input_layer.at(2)->fire(0.);
  
  for(double time = 0.; time < 40.; time += timestep)
    for(auto n: all_neurons)
      n->forward_propagate(time);
  
  assert(! output_layer.at(0)->spikes.empty());

  // fixture
  assert(fabs(output_layer.at(0)->spikes.at(5) - 19.7) < 0.2);

  //for(auto time: input_layer.at(0)->spikes)
  //  std::cout << time << std::endl;
  //std::cout << std::endl;
  //for(auto time: hidden_layer.at(0)->spikes)
  //  std::cout << time << std::endl;
  //std::cout << std::endl;
  //for(auto time: output_layer.at(0)->spikes)
  //  std::cout << time << std::endl;

  return 0;
}

