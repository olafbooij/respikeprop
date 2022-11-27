#pragma once

#include<vector>
#include<queue>
#include<cmath>
#include<range/v3/view/zip.hpp>
#include<range/v3/view/enumerate.hpp>

namespace resp {

  // Straightforward implementation of Spikeprop with multiple spikes per
  // neuron, with links to equations from the paper:
  // An implementation of "A gradient descent rule for spiking neurons emitting
  // multiple spikes", Olaf Booij, Hieu tat Nguyen, Information Processing
  // Letters, Volume 95, Issue 6, 30 September 2005, Pages 552-558.
  //
  // Forward propagation is not event-based and thus compute time in the order
  // of time-steps.
  // Backpropagation in this implementation is implemented quite efficiently,
  // keeping gradients in the forward pass.
  // Network connectivity is implemented using raw-pointers, leaving
  // responsibility of memory management with the user.
  struct Neuron;

  struct Connection
  {
    Neuron* neuron;  // putting a lot of responsibility on user...
    Neuron* post_neuron;  // putting a lot of responsibility on user...
    struct Synapse
    {
      double weight;
      double delay;
      double delta_weight;
      std::vector<double> dt_dws;  // same order as spikes
      // vector of future incoming spikes...
    };
    std::vector<Synapse> synapses;
    std::vector<std::vector<double>> dprets_dpostts; // per prespike per postspike
  };

  struct Neuron
  {
    Neuron(std::string key_ = "neuron") : key(key_) {}
    std::vector<Connection> incoming_connections;
    std::vector<Connection*> outgoing_connections;  // raw pointers again!
    std::vector<double> spikes;
    // The following settings are taken from the thesis "Temporal Pattern
    // Classification using Spiking Neural Networks" which differ from the
    // paper.
    double tau_m = 4.0;
    double tau_s = 2.0;
    double u_m;
    double u_s;
    double last_update = 0.;
    double clamped = 0.;
    std::string key;

    void clear()
    {
      spikes.clear();
      for(auto& incoming_connection: incoming_connections)
      {
        incoming_connection.dprets_dpostts.clear();
        for(auto& incoming_synapse: incoming_connection.synapses)
          incoming_synapse.dt_dws.clear();
      }
      u_m = u_s = 0;
    }

    void fire(double time)
    {
      spikes.emplace_back(time);
    }

    // Forward propagate and store gradients for backpropagation. For now
    // keeping this as one function. To be refactored.
    double incoming_spike(double time, double weight)
    {
      //std::cout << key << " recieves spike at " << time << std::endl;
      // update potentials till just before incoming spike
      u_m *= exp(- (time - last_update) / tau_m);  // could make this compile time by fixing timestep and tau's
      u_s *= exp(- (time - last_update) / tau_s);
      last_update = time;

      // adjust due to incoming spike
      u_m += weight;
      u_s -= weight;

      // compute exact future firing time
      const double threshold = 1.;
      double D = u_m * u_m - 4 * u_s * -threshold;
      double possible_spike = 0;
      if(D > 0)
      {
        double expdt = (- u_m - sqrt(D)) / (2 * u_s);
        if(expdt > 0)
        {
          double predict_spike = - log(expdt) * tau_m;
          if(predict_spike > 0)
            return time + predict_spike;
        }
      }
      return 0.;
    }

    double spike(double time)
    {
      //std::cout << key << " spikes at " << time << std::endl;
      const double threshold = 1.;
      // update potentials till just before incoming spike
      u_m *= exp(- (time - last_update) / tau_m);
      u_s *= exp(- (time - last_update) / tau_s);
      last_update = time;

      store_gradients(time);
      spikes.emplace_back(time);
      u_m -= threshold;

      // compute exact future firing time
      double D = u_m * u_m - 4 * u_s * -threshold;
      double possible_spike = 0;
      if(D > 0)
      {
        double expdt = (- u_m - sqrt(D)) / (2 * u_s);
        if(expdt > 0)
        {
          double predict_spike = - log(expdt) * tau_m;
          if(predict_spike > 0)
            return time + predict_spike;
        }
      }
      return 0.;
    }

    void store_gradients(double spike_time)
    {
      double du_dt = - u_m / tau_m - u_s / tau_s;
      if(du_dt < .1) // handling discontinuity circumstance 1 Sec 3.2
        du_dt = .1;

      for(auto& incoming_connection: incoming_connections)
      {
        incoming_connection.dprets_dpostts.resize(incoming_connection.neuron->spikes.size());  // make sure there's an entry for all pre spikes
        for(auto& dpret_dpostts: incoming_connection.dprets_dpostts)
          dpret_dpostts.resize(spikes.size() + 1, 0.);  // make sure there's an entry for all post spikes
        for(auto& synapse: incoming_connection.synapses)
          synapse.dt_dws.emplace_back(0.);

        for(auto& synapse: incoming_connection.synapses)
          for(const auto& [pre_spike, dpret_dpostts]: ranges::views::zip(incoming_connection.neuron->spikes, incoming_connection.dprets_dpostts))
          {
            double s = spike_time - pre_spike - synapse.delay;
            if(s >= 0)
            {
              auto u_m1 =   synapse.weight * exp(-s / tau_m);
              auto u_s1 = - synapse.weight * exp(-s / tau_s);
              synapse.dt_dws.back() += - (u_m1 + u_s1) / synapse.weight;
              dpret_dpostts.back() += - (u_m1 / tau_m + u_s1 / tau_s);
            }
          }

        for(const auto& [ref_spike_i, ref_spike]: ranges::views::enumerate(spikes))
        {
          double s = spike_time - ref_spike;
          if(s >= 0)
          {
            double u_r1 = exp(-s / tau_m) / tau_m;
            for(auto& synapse: incoming_connection.synapses)
              synapse.dt_dws.back() += u_r1 * synapse.dt_dws.at(ref_spike_i);
            for(auto& dpret_dpostts: incoming_connection.dprets_dpostts)
              dpret_dpostts.back()  += u_r1 * dpret_dpostts.at(ref_spike_i);
          }
        }
        for(auto& dpret_dpostts: incoming_connection.dprets_dpostts)
          dpret_dpostts.back() /= du_dt;
        for(auto& synapse: incoming_connection.synapses)
          synapse.dt_dws.back() /= du_dt;
      }
    }

    // Compute needed weight changes, and backpropagate to incoming
    // connections.
    // The implementation results in a bit of double work, because each dE_dt
    // change is pushed back separately. Could be more efficient if knowing for
    // each spike if all resulting post-spikes have been back-propagated.
    void add_dE_dt(int spike_i, double dE_dt, double learning_rate)
    {
      for(auto& incoming_connection: incoming_connections)
      {
        for(auto& synapse: incoming_connection.synapses)
          if(spike_i < synapse.dt_dws.size())
            synapse.delta_weight -= learning_rate * dE_dt * synapse.dt_dws.at(spike_i);
        for(int pre_spike_i = 0; pre_spike_i < incoming_connection.neuron->spikes.size(); ++pre_spike_i)
        if(spikes.at(spike_i) > incoming_connection.neuron->spikes.at(pre_spike_i))
          if(pre_spike_i < incoming_connection.dprets_dpostts.size())
            if(spike_i < incoming_connection.dprets_dpostts.at(pre_spike_i).size())
              incoming_connection.neuron->add_dE_dt(pre_spike_i, dE_dt * incoming_connection.dprets_dpostts.at(pre_spike_i).at(spike_i), learning_rate);
      }
    }
    void compute_delta_weights(const double learning_rate)  // missnomer, starts backprop
    {
      if(clamped > 0)  // to check that this is an output neuron
        if(! spikes.empty())
          add_dE_dt(0, spikes.front() - clamped, learning_rate);
    }
    void forward_propagate(double, double) {};
  };


  struct Events
  {
    struct SynapseSpike
    {
      Neuron* neuron;
      double weight;
      double time;
      friend bool operator<(auto a, auto b){return a.time > b.time;};  // earliest on top
    };
    std::priority_queue<SynapseSpike> synapse_spikes;
    struct NeuronSpike
    {
      Neuron* neuron;
      double time;
      //friend bool operator<(auto a, auto b){return a.time > b.time;};  // earliest on top
    };
    std::vector<NeuronSpike> neuron_spikes;
    bool active()
    {
      return ! (neuron_spikes.empty() && synapse_spikes.empty());
    }

    void process_event()
    {
      if(synapse_spikes.empty() && neuron_spikes.empty())
        return;
      // which one first
      //compute_earliest_neuron_spike
      auto neuron_spike = std::max_element(neuron_spikes.begin(), neuron_spikes.end(), [](auto a, auto b){return a.time > b.time;});
      // bit of ugly logic to determine what to process
      if(neuron_spikes.empty() || ((! synapse_spikes.empty()) && synapse_spikes.top().time < neuron_spike->time))
      { // process synapse
        auto& synapse_spike = synapse_spikes.top();
        // remove neuron's fire-time
        auto existing_spike =  std::find_if(neuron_spikes.begin(), neuron_spikes.end(), [synapse_spike](const auto& n){return synapse_spike.neuron == n.neuron;});
        if(existing_spike != neuron_spikes.end())
          neuron_spikes.erase(existing_spike);
        // compute possible next one
        auto future_spike = synapse_spike.neuron->incoming_spike(synapse_spike.time, synapse_spike.weight);
        // add it if it is
        if(future_spike > 0)
          neuron_spikes.emplace_back(synapse_spike.neuron, future_spike);
        synapse_spikes.pop();
      }
      else
      { // process neuron
        // update post_synapses
        auto spiking_neuron = neuron_spike->neuron;
        for(auto outgoing_connection: spiking_neuron->outgoing_connections)
          for(auto& post_synapse: outgoing_connection->synapses)
            synapse_spikes.emplace(outgoing_connection->post_neuron, post_synapse.weight, neuron_spike->time + post_synapse.delay);
        // update gradients and check for new spikes
        auto future_spike = spiking_neuron->spike(neuron_spike->time);
        neuron_spikes.erase(neuron_spike);
        if(future_spike > 0)
          neuron_spikes.emplace_back(spiking_neuron, future_spike);
      }
    }
  };

}

