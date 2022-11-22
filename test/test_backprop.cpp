#include<iostream>
#include<vector>
#include<cassert>
#include<respikeprop/respikeprop_store_gradients.hpp>

// These tests check the gradient (d Error / d weight) used for gradient
// descent by comparing it against a numerically computed gradient.

// Especially for deeper, more elaborate spiking neural networks this check
// will not work because of small quantization errors combined with small
// gradients. So here just checking for some very simple networks.


void check_backprop(auto& network, auto& synapse)
{
  auto& output = network.back();
  const double timestep = .0001;

  for(auto& neuron: network)
    neuron.clear();
  for(double time = 0.; time < 40.; time += timestep)
    for(auto& neuron: network)
      neuron.forward_propagate(time, timestep);
  //for(auto& neuron: network)
  //{
  //  std::cout << neuron.key << " spikes before: " << neuron.spikes.size() << " : " << std::endl;
  //  for(auto spike: neuron.spikes)
  //    std::cout << spike << std::endl;
  //}

  auto error_before = .5 * pow(output.spikes.at(0) - output.clamped, 2);

  const double small = 0.03;
  synapse.weight += small;
  for(auto& neuron: network)
    neuron.clear();
  for(double time = 0.; time < 40.; time += timestep)
    for(auto& neuron: network)
      neuron.forward_propagate(time, timestep);
  //for(auto& neuron: network)
  //{
  //  std::cout << neuron.key << " spikes after : " << neuron.spikes.size() << " : " << std::endl;
  //  for(auto spike: neuron.spikes)
  //    std::cout << spike << std::endl;
  //}
  auto error_after = .5 * pow(output.spikes.at(0) - output.clamped, 2);

  const double learning_rate = 1.;
  for(auto& neuron: network)
  {
    for(auto& incoming_connection: neuron.incoming_connections)
      for(auto& in_synapse: incoming_connection.synapses)
        in_synapse.delta_weight = 0.;
    neuron.compute_delta_weights(learning_rate);
  }
  synapse.weight -= small;

  //std::cout << "difference         " << error_after - error_before << std::endl;
  //std::cout << "d_E / d_w          " << (error_after - error_before) / small << std::endl;
  //std::cout << "computed d_E / d_w " << - synapse.delta_weight / learning_rate << std::endl;
  assert(fabs((error_after - error_before) / small - (- synapse.delta_weight / learning_rate)) < (timestep / small * 2));
}

auto& add_synapse(auto& pre, auto& post, double weight, double delay)
{
  auto& incoming_connection = post.incoming_connections.emplace_back(&pre);
  auto& synapse = incoming_connection.synapses.emplace_back(weight, delay);
  return synapse;
}

void check_one_input_one_output()
{
  using namespace resp;

  Neuron input("input");
  std::vector<Neuron> network;
  Neuron& output = network.emplace_back("output");
  //auto& synapse = output.incoming_synapses.emplace_back(input, 7., 1.);
  //auto& incoming_connection = output.incoming_connections.emplace_back(&input);
  //auto& synapse = incoming_connection.synapses.emplace_back(7., 1.);
  auto& synapse = add_synapse(input, output, 7., 1.);

  input.fire(0.);
  output.clamped = 3.;
  check_backprop(network, synapse);
}

void check_two_inputs_one_output()
{
  using namespace resp;

  Neuron input_0("input");
  Neuron input_1("input");
  std::vector<Neuron> network;
  Neuron& output = network.emplace_back("output");
  //output.incoming_synapses.emplace_back(input_0, 3., 1.);
  add_synapse(input_0, output, 3., 1.);
  //auto& synapse = output.incoming_synapses.emplace_back(input_1, 3., 1.);
  auto& synapse = add_synapse(input_1, output, 3., 1.);

  input_0.fire(0.);
  input_1.fire(0.);
  output.clamped = 3.;

  check_backprop(network, synapse);
}

void check_one_input_one_hidden_one_output()
{
  using namespace resp;

  Neuron input("input");
  std::vector<Neuron> network;
  network.reserve(2);
  Neuron& hidden = network.emplace_back("hidden");
  Neuron& output = network.emplace_back("output");
  auto& synapse_0 = add_synapse(input , hidden, 7., 1.);
  auto& synapse_1 = add_synapse(hidden, output, 7., 1.);

  input.fire(0.);
  output.clamped = 3.;

  check_backprop(network, synapse_0);
  check_backprop(network, synapse_1);
}

void check_two_inputs_two_hiddens_one_output()
{
  using namespace resp;

  Neuron input_0("input_0");
  Neuron input_1("input_1");
  std::vector<Neuron> network;
  network.reserve(3);
  Neuron& hidden_0 = network.emplace_back("hidden_0");
  Neuron& hidden_1 = network.emplace_back("hidden_1");
  Neuron& output = network.emplace_back("output");
  hidden_0.incoming_connections.reserve(2);
  add_synapse(input_0, hidden_0, 3.3, 1.1);
  add_synapse(input_1, hidden_0, 2.7, 1.2);
  hidden_1.incoming_connections.reserve(2);
  add_synapse(input_0, hidden_1, 3.0, 1.3);
  add_synapse(input_1, hidden_1, 3.7, 0.9);
  output.incoming_connections.reserve(2);
  add_synapse(hidden_0, output, 3., 1.2);
  add_synapse(hidden_1, output, 3.2, 1.1);

  input_0.fire(0.2);
  input_1.fire(0.1);
  output.clamped = 3.;

  for(auto& neuron: network)
    for(auto& incoming_connection: neuron.incoming_connections)
      for(auto& synapse: incoming_connection.synapses)
        check_backprop(network, synapse);

  input_0.fire(0.3);
  input_1.fire(0.4);

  for(auto& neuron: network)
    for(auto& incoming_connection: neuron.incoming_connections)
      for(auto& synapse: incoming_connection.synapses)
        check_backprop(network, synapse);
}

void check_two_inputs_two_hiddens_one_output_multi_synapses()
{
  using namespace resp;

  Neuron input_0("input_0");
  Neuron input_1("input_1");
  std::vector<Neuron> network;
  network.reserve(3);
  Neuron& hidden_0 = network.emplace_back("hidden_0");
  Neuron& hidden_1 = network.emplace_back("hidden_1");
  Neuron& output = network.emplace_back("output");
  hidden_0.incoming_connections.emplace_back(&input_0, std::vector<Neuron::Connection::Synapse>{
    Neuron::Connection::Synapse(1.6, 1.12),
    Neuron::Connection::Synapse(1.8, 1.20)
  });
  hidden_0.incoming_connections.emplace_back(&input_1, std::vector<Neuron::Connection::Synapse>{
    Neuron::Connection::Synapse(1.0, 1.19),
    Neuron::Connection::Synapse(1.7, 1.29)
  });
  hidden_1.incoming_connections.emplace_back(&input_0, std::vector<Neuron::Connection::Synapse>{
    Neuron::Connection::Synapse(1.0, 1.32),
    Neuron::Connection::Synapse(4.0, 1.39)
  });
  hidden_1.incoming_connections.emplace_back(&input_1, std::vector<Neuron::Connection::Synapse>{
    Neuron::Connection::Synapse(-3.7, 0.90),
    Neuron::Connection::Synapse( 6.7, 0.98)
  });
  output.incoming_connections.emplace_back(&hidden_0, std::vector<Neuron::Connection::Synapse>{
    Neuron::Connection::Synapse( 5., 1.23),
    Neuron::Connection::Synapse( 4., 1.20)
  });
  output.incoming_connections.emplace_back(&hidden_1, std::vector<Neuron::Connection::Synapse>{
    Neuron::Connection::Synapse(-1.2, 1.12),
    Neuron::Connection::Synapse(-0.2, 1.3 )
  });

  input_0.fire(0.2);
  input_1.fire(0.1);
  input_0.fire(0.3);
  input_1.fire(0.4);
  output.clamped = 3.;

  for(auto& neuron: network)
    for(auto& incoming_connection: neuron.incoming_connections)
      for(auto& synapse: incoming_connection.synapses)
        check_backprop(network, synapse);
}

int main()
{
  check_one_input_one_output();
  check_two_inputs_one_output();
  check_one_input_one_hidden_one_output();
  check_two_inputs_two_hiddens_one_output();
  check_two_inputs_two_hiddens_one_output_multi_synapses();

  return 0;
}

