#pragma once

#include<vector>
#include<queue>
#include<cmath>
#include<algorithm>
#include<ranges>
#include<range/v3/view/zip.hpp>
#include<range/v3/view/enumerate.hpp>

namespace resp {

  // An implementation of "A gradient descent rule for spiking neurons emitting
  // multiple spikes", Olaf Booij, Hieu tat Nguyen, Information Processing
  // Letters, Volume 95, Issue 6, 30 September 2005, Pages 552-558.
  //
  // Forward propagation is event-based, which means compution is efficient
  // (close to the order of spikes) and spike times are exact, i.e. no
  // quantification errors due to time-steps.
  // Backpropagation in this implementation is implemented quite efficiently,
  // keeping gradients in the forward pass.
  // Network connectivity is implemented using raw-pointers, leaving
  // responsibility of memory management with the user.
  struct Neuron;

  struct Connection
  {
    Neuron* neuron;  // putting a lot of responsibility on user...
    Neuron* post_neuron;  // again a lot of responsibility on user...
    struct Synapse
    {
      double weight;
      double delay;
      double delta_weight;
      std::vector<double> dt_dws;  // same order as spikes
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
    const double tau_m = 4.0;
    const double tau_s = tau_m / 2;
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

    void fire(double time)  // only used for input spikes
    {
      spikes.emplace_back(time);
    }

    void update_potentials(double time)
    {
      u_m *= exp(- (time - last_update) / tau_m);  // could make this compile time by fixing timestep and tau's
      u_s *= exp(- (time - last_update) / tau_s);
      last_update = time;
    }

    // compute exact future firing time (should document the derivation of this formula)
    double compute_future_spike()
    {
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
            return last_update + predict_spike;
        }
      }
      return 0.;  // should perhaps use std::optional
    }

    // Forward propagate and store gradients for backpropagation. For now
    // keeping this as one function. To be refactored.
    void incoming_spike(double time, double weight)
    {
      update_potentials(time);
      u_m += weight;
      u_s -= weight;
    }

    void spike(double time)
    {
      const double threshold = 1.;
      update_potentials(time);

      store_gradients(time);
      spikes.emplace_back(time);
      u_m -= threshold;
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
    void add_dE_dt(std::size_t spike_i, double dE_dt, double learning_rate)
    {
      for(auto& incoming_connection: incoming_connections)
      {
        for(auto& synapse: incoming_connection.synapses)
          if(spike_i < synapse.dt_dws.size())
            synapse.delta_weight -= learning_rate * dE_dt * synapse.dt_dws.at(spike_i);
        for(auto pre_spike_i: std::views::iota(0u, incoming_connection.neuron->spikes.size()))
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
    void forward_propagate(double, double) {};  // (not used in this implementation)
  };


  // The following class keeps track of all the event and handles the forward
  // propagation.
  struct NeuronSpike
  {
    Neuron* neuron;
    double time;
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
    std::priority_queue<SynapseSpike> synapse_spikes;  // newest on top, order is stable
    std::vector<NeuronSpike> neuron_spikes;  // priority_queue not an option here, because have to replace spikes. A bidirectional map might be faster.
    bool active()
    {
      return ! (neuron_spikes.empty() && synapse_spikes.empty());
    }

    // Process the next event in the queue. This might be either a spike
    // ariving at a neuron (named synapse_spike), or a neuron spiking.
    void process_event()
    {
      if(! active())
        return;
      // which one first
      // compute_earliest_neuron_spike
      auto neuron_spike = std::ranges::max_element(neuron_spikes, [](const auto& a, const auto& b) noexcept {return a.time > b.time;});
      Neuron* updated_neuron;
      // bit of ugly logic to determine which type of event is first 
      if(neuron_spikes.empty() || ((! synapse_spikes.empty()) && synapse_spikes.top().time < neuron_spike->time))
      { // process synapse
        auto& synapse_spike = synapse_spikes.top();
        updated_neuron = synapse_spike.neuron;
        // find neuron's existing fire-time
        neuron_spike = std::ranges::find_if(neuron_spikes, [updated_neuron](const auto& n) noexcept {return updated_neuron == n.neuron;});
        // update neuron
        updated_neuron->incoming_spike(synapse_spike.time, synapse_spike.weight);
        synapse_spikes.pop();
      }
      else
      { // process neuron
        // update post_synapses
        updated_neuron = neuron_spike->neuron;
        for(auto& outgoing_connection: updated_neuron->outgoing_connections)
          for(auto& post_synapse: outgoing_connection->synapses)
            synapse_spikes.emplace(outgoing_connection->post_neuron, post_synapse.weight, neuron_spike->time + post_synapse.delay);
        // update neuron itself, including gradients
        updated_neuron->spike(neuron_spike->time);
      }
      // remove affected neuron's spike
      if(neuron_spike != neuron_spikes.end())
        neuron_spikes.erase(neuron_spike);
      // check for new spike
      auto future_spike = updated_neuron->compute_future_spike(); 
      if(future_spike > 0)
        neuron_spikes.emplace_back(updated_neuron, future_spike);
    }
  };

}

