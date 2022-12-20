#include<iostream>
#include<vector>
#include<cassert>
#include<respikeprop/respikeprop_event_based.hpp>


void connect_outgoing(auto& neuron)
{
  for(auto& incoming_connection: neuron.incoming_connections)
  {
    incoming_connection.post_neuron = &neuron;
    incoming_connection.neuron->outgoing_connections.emplace_back(&incoming_connection);
  }
}

int main()
{
  using namespace resp;

  Neuron bounce("bounce");
  Neuron output("output");
  const double learning_rate = 1e-2;

  auto add_synapse = [](auto& pre, auto& post, double weight, double delay)
  {
    auto& incoming_connection = post.incoming_connections.emplace_back(&pre);
    incoming_connection.synapses.emplace_back(weight, delay);
  };
  add_synapse(bounce, bounce, 6., 1.);
  add_synapse(bounce, output, 1., 1.);
  add_synapse(output, bounce, -10., .1);
  connect_outgoing(bounce);
  connect_outgoing(output);

  for(int epoch = 0; epoch < 300; ++epoch)
  {
    output.clamped = 7.;
    Events events;
    events.neuron_spikes.emplace_back(&bounce, 0.);
    while(output.spikes.empty() && events.active())
      events.process_event();
    //std::cout << output.spikes.front() << std::endl;

    output.backprop(learning_rate);

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
  assert(fabs(output.spikes.front() - output.clamped) < 1e-7);

  return 0;
}

