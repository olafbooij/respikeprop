#include<iostream>
#include<vector>
#include<cassert>
#include<respikeprop/respikeprop_store_gradients.hpp>

int main()
{
  using namespace resp;

  Neuron bounce("bounce");
  Neuron output("output");
  bounce.tau_r = 4.0;
  output.tau_r = 4.0;

  auto add_synapse = [](auto& pre, auto& post, double weight, double delay)
  {
    auto& incoming_connection = post.incoming_connections.emplace_back(&pre);
    auto& synapse = incoming_connection.synapses.emplace_back(weight, delay);
    return synapse;
  };
  add_synapse(bounce, bounce, 6., 1.);
  add_synapse(bounce, output, 1., 1.);
  add_synapse(output, bounce, -10., .1);

  for(int epoch = 0; epoch < 100; ++epoch)
  {
    bounce.fire(0.);
    output.clamped = 7.;
    const double timestep = .01;
    for(double time = 0.; time < 30. && output.spikes.empty(); time += timestep)
    {
      bounce.forward_propagate(time, timestep);
      output.forward_propagate(time, timestep);
    }
    std::cout << output.spikes.front() << std::endl;

    const double learning_rate = 1e-2;
    output.compute_delta_weights(learning_rate);

    auto adjust_weights = [](auto& neuron)
    {
      for(auto& incoming_connection: neuron.incoming_connections)
        for(auto& synapse: incoming_connection.synapses)
        {
          synapse.weight += synapse.delta_weight;
          synapse.delta_weight = 0.;
        }
      neuron.clear();
    };
    adjust_weights(bounce);
    adjust_weights(output);
  }

  return 0;
}

