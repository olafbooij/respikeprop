#include<iostream>
#include<vector>
#include<cassert>
#include<respikeprop/respikeprop_event_based.hpp>
#include<respikeprop/create_network.hpp>

// These tests check the gradient (d Error / d weight) used for gradient
// descent by comparing it against a numerically computed gradient.

// Given that simulation is event based, there's no quantization errors. Could
// check for more elaborate networks.

namespace resp {

void check_backprop(auto& network, auto& events, auto& synapse)
{
  auto& output = network.back();

  resp::Events events_before = events;
  for(auto& neuron: network)
    neuron.clear();
  while(output.spikes.empty() && events_before.active())
    events_before.process_event();
  //for(auto& neuron: network)
  //{
  //  std::cout << neuron.key << " spikes before: " << neuron.spikes.size() << " : " << std::endl;
  //  for(auto spike: neuron.spikes)
  //    std::cout << spike << std::endl;
  //}

  auto error_before = .5 * pow(output.spikes.at(0) - output.clamped, 2);

  const double small = 1e-6;
  synapse.weight += small;
  for(auto& neuron: network)
    neuron.clear();
  resp::Events events_after = events;
  while(output.spikes.empty() && events_after.active())
    events_after.process_event();
  //for(auto& neuron: network)
  //{
  //  std::cout << neuron.key << " spikes after : " << neuron.spikes.size() << " : " << std::endl;
  //  for(auto spike: neuron.spikes)
  //    std::cout << spike << std::endl;
  //}
  auto error_after = .5 * pow(output.spikes.at(0) - output.clamped, 2);

  const double learning_rate = 1.;
  for(auto& neuron: network)
    for(auto& incoming_connection: neuron.incoming_connections)
      for(auto& in_synapse: incoming_connection.synapses)
        in_synapse.delta_weight = 0.;
  output.compute_delta_weights(learning_rate);
  synapse.weight -= small;

  //std::cout << "difference         " << error_after - error_before << std::endl;
  //std::cout << "d_E / d_w          " << (error_after - error_before) / small << std::endl;
  //std::cout << "computed d_E / d_w " << - synapse.delta_weight / learning_rate << std::endl;
  //std::cout << (error_after - error_before) / small - (- synapse.delta_weight / learning_rate) << std::endl;
  assert(fabs((error_after - error_before) / small - (- synapse.delta_weight / learning_rate)) < small);
}

void check_backprop_all(auto& network, auto& events)
{
  for(auto& neuron: network)
    for(auto& incoming_connection: neuron.incoming_connections)
      for(auto& synapse: incoming_connection.synapses)
        check_backprop(network, events, synapse);
}

void add_synapse(auto& pre, auto& post, double weight, double delay)
{
  auto& incoming_connection = post.incoming_connections.emplace_back(&pre);
  incoming_connection.synapses.emplace_back(weight, delay);
}

void check_one_input_one_output()
{
  std::array<Neuron, 2> network{{{"input"}, {"output"}}};
  auto& [input, output] = network;
  add_synapse(input, output, 7., 1.);
  connect_outgoing_layer(network);

  Events events;
  events.neuron_spikes.emplace_back(&input, 0.);
  output.clamped = 3.;
  check_backprop_all(network, events);
}

void check_two_inputs_one_output()
{
  std::array<Neuron, 3> network{{{"input_0"}, {"input_1"}, {"output"}}};
  auto& [input_0, input_1, output] = network;
  add_synapse(input_0, output, 3., 1.);
  add_synapse(input_1, output, 3., 1.);
  connect_outgoing_layer(network);

  Events events;
  events.neuron_spikes.emplace_back(&input_0, 0.);
  events.neuron_spikes.emplace_back(&input_1, 0.);
  output.clamped = 3.;

  check_backprop_all(network, events);
}

void check_one_input_one_hidden_one_output()
{
  std::array<Neuron, 3> network{{{"input"}, {"hidden"}, {"output"}}};
  auto& [input, hidden, output] = network;
  add_synapse(input , hidden, 7., 1.);
  add_synapse(hidden, output, 7., 1.);
  connect_outgoing_layer(network);

  Events events;
  events.neuron_spikes.emplace_back(&input, 0.);
  output.clamped = 3.;

  check_backprop_all(network, events);
}

void check_two_inputs_two_hiddens_one_output()
{
  using namespace resp;

  std::array<Neuron, 5> network{{{"input_0"}, {"input_1"}, {"hidden_0"}, {"hidden_1"}, {"output"}}};
  auto& [input_0, input_1, hidden_0, hidden_1, output] = network;
  hidden_0.incoming_connections.reserve(2);
  add_synapse(input_0, hidden_0, 3.3, 1.1);
  add_synapse(input_1, hidden_0, 2.7, 1.2);
  hidden_1.incoming_connections.reserve(2);
  add_synapse(input_0, hidden_1, 3.0, 1.3);
  add_synapse(input_1, hidden_1, 3.7, 0.9);
  output.incoming_connections.reserve(2);
  add_synapse(hidden_0, output, 3., 1.2);
  add_synapse(hidden_1, output, 3.2, 1.1);
  connect_outgoing_layer(network);

  output.clamped = 3.;

  Events events;
  events.neuron_spikes.emplace_back(&input_0, .2);
  events.neuron_spikes.emplace_back(&input_1, .1);

  check_backprop_all(network, events);

  events.neuron_spikes.emplace_back(&input_0, .3);
  events.neuron_spikes.emplace_back(&input_1, .4);

  check_backprop_all(network, events);
}

void check_two_inputs_two_hiddens_one_output_multi_synapses()
{
  using namespace resp;

  std::array<Neuron, 5> network{{{"input_0"}, {"input_1"}, {"hidden_0"}, {"hidden_1"}, {"output"}}};
  auto& [input_0, input_1, hidden_0, hidden_1, output] = network;
  hidden_0.incoming_connections.emplace_back(&input_0, &hidden_0, std::vector<Connection::Synapse>{
    Connection::Synapse(1.6, 1.12),
    Connection::Synapse(1.8, 1.20)
  });
  hidden_0.incoming_connections.emplace_back(&input_1, &hidden_0, std::vector<Connection::Synapse>{
    Connection::Synapse(1.0, 1.19),
    Connection::Synapse(1.7, 1.29)
  });
  hidden_1.incoming_connections.emplace_back(&input_0, &hidden_1, std::vector<Connection::Synapse>{
    Connection::Synapse(1.0, 1.32),
    Connection::Synapse(4.0, 1.39)
  });
  hidden_1.incoming_connections.emplace_back(&input_1, &hidden_1, std::vector<Connection::Synapse>{
    Connection::Synapse(-3.7, 0.90),
    Connection::Synapse( 6.7, 0.98)
  });
  output.incoming_connections.emplace_back(&hidden_0, &output, std::vector<Connection::Synapse>{
    Connection::Synapse( 5., 1.23),
    Connection::Synapse( 4., 1.20)
  });
  output.incoming_connections.emplace_back(&hidden_1, &output, std::vector<Connection::Synapse>{
    Connection::Synapse(-1.2, 1.12),
    Connection::Synapse(-0.2, 1.3 )
  });
  connect_outgoing_layer(network);

  output.clamped = 3.;

  Events events;
  events.neuron_spikes.emplace_back(&input_0, .2);
  events.neuron_spikes.emplace_back(&input_1, .1);
  events.neuron_spikes.emplace_back(&input_0, .3);
  events.neuron_spikes.emplace_back(&input_1, .4);

  check_backprop_all(network, events);
}
}

int main()
{
  using namespace resp;
  check_one_input_one_output();
  check_two_inputs_one_output();
  check_one_input_one_hidden_one_output();
  check_two_inputs_two_hiddens_one_output();
  check_two_inputs_two_hiddens_one_output_multi_synapses();

  return 0;
}

