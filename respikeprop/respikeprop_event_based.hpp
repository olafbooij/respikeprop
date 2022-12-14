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
  // Backpropagation is implemented quite efficiently, keeping gradients in the
  // forward pass.
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
      double u_m;
      double u_s;
    };
    std::vector<Synapse> synapses;
    struct DPreSpike
    {
      std::vector<double> dpostts; // same order as spikes
      double u_m = 0;
      double u_s = 0;
    };
    std::vector<DPreSpike> dpre_spikes;
  };

  struct Neuron
  {
    Neuron(std::string key_ = "neuron") : key(key_) {}
    std::vector<Connection> incoming_connections;
    std::vector<Connection*> outgoing_connections;  // raw pointers again!
    struct Spike
    {
      double time;
      double dE_dt;
    };
    std::vector<Spike> spikes;
    static constexpr double tau = 4.0;
    double u_m;
    double u_s;
    double last_update = 0.;
    double last_spike = 0.;
    double clamped = 0.;
    std::string key;

    void clear()
    {
      spikes.clear();
      for(auto& incoming_connection: incoming_connections)
      {
        incoming_connection.dpre_spikes.clear();
        for(auto& incoming_synapse: incoming_connection.synapses)
        {
          incoming_synapse.dt_dws.clear();
          incoming_synapse.u_m = 0;
          incoming_synapse.u_s = 0;
        }
      }
      u_m = u_s = last_update = last_spike = 0;
    }

    void update_potentials(double time)
    {
      auto update = exp(- (time - last_update) / tau);
      u_m *= update;
      u_s *= update * update;
      last_update = time;
    }

    // compute exact future firing time (should document the derivation of this formula)
    double compute_future_spike()
    {
      double D = u_m * u_m + 4 * u_s;
      if(D > 0)
      {
        double expdt = (- u_m - sqrt(D)) / (2 * u_s);
        if(expdt > 0)
        {
          double predict_spike = - log(expdt) * tau;
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
      update_potentials(time);

      store_gradients(time);
      spikes.emplace_back(time, 0.);
      last_spike = time;
      u_m -= 1.;
    }

    void store_gradients(double spike_time)
    {
      double du_dt = - (u_m + u_s * 2) / tau;
      if(du_dt < .1) // handling discontinuity circumstance 1 Sec 3.2
        du_dt = .1;

      double spike_diff_exp_m = exp(- (spike_time - last_spike) / tau);
      double spike_diff_exp_s = spike_diff_exp_m * spike_diff_exp_m;

      for(auto& incoming_connection: incoming_connections)
      {
        incoming_connection.dpre_spikes.resize(incoming_connection.neuron->spikes.size());  // make sure there's an entry for all pre spikes
        for(auto& dpre_spike: incoming_connection.dpre_spikes)
        {
          dpre_spike.u_m *= spike_diff_exp_m;
          dpre_spike.u_s *= spike_diff_exp_s;
        }
        for(auto& synapse: incoming_connection.synapses)
        {
          // update_synapse_potentials
          synapse.u_m *= spike_diff_exp_m;
          synapse.u_s *= spike_diff_exp_s;
          for(const auto& [pre_spike, dpre_spike]: ranges::views::zip(incoming_connection.neuron->spikes, incoming_connection.dpre_spikes))
          {
            double s = spike_time - pre_spike.time - synapse.delay;
            if(pre_spike.time + synapse.delay > last_spike && s >= 0)  // pre-spike came between previous and this spike
            {
              auto u_m1 = exp(-s / tau);
              auto u_s1 = - u_m1 * u_m1;
              synapse.u_m += u_m1;
              synapse.u_s += u_s1;
              dpre_spike.u_m += synapse.weight * u_m1;
              dpre_spike.u_s += synapse.weight * u_s1;
            }
          }
          synapse.dt_dws.emplace_back(- (synapse.u_m + synapse.u_s) / du_dt);
          synapse.u_m -= spike_diff_exp_m / tau * synapse.dt_dws.back();
        }
        for(auto& dpre_spike: incoming_connection.dpre_spikes)
        {
          dpre_spike.dpostts.resize(spikes.size(), 0.); // needed because previous post spikes might be before pre spike
          dpre_spike.dpostts.emplace_back(- (dpre_spike.u_m + dpre_spike.u_s * 2) / tau / du_dt);
          dpre_spike.u_m -= spike_diff_exp_m / tau * dpre_spike.dpostts.back();
        }
      }
    }

    // Compute needed weight changes, and backpropagate to incoming
    // connections.
    void backprop_spike(std::size_t spike_i, double learning_rate)
    {
      auto& spike = spikes.at(spike_i);
      if(clamped > 0 && spike_i == 0)  // output neuron
        spike.dE_dt = spike.time - clamped;
      for(auto& incoming_connection: incoming_connections)
      {
        for(auto& synapse: incoming_connection.synapses)
          synapse.delta_weight -= learning_rate * spike.dE_dt * synapse.dt_dws.at(spike_i);
        for(const auto& [pre_spike, dpre_spike]: ranges::views::zip(incoming_connection.neuron->spikes, incoming_connection.dpre_spikes))
          pre_spike.dE_dt += spike.dE_dt * dpre_spike.dpostts.at(spike_i);
      }
    }
  };


  // The Events class keeps track of all the event and handles the forward
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
    std::vector<NeuronSpike> predicted_spikes;  // priority_queue not an option here, because have to replace spikes. A bidirectional map might be faster.
    struct SpikeRecord
    {
      Neuron* neuron;
      double time;
      std::size_t index;
    };
    std::vector<SpikeRecord> actual_spikes;  // recorded for backprop
    bool active()
    {
      return ! (predicted_spikes.empty() && synapse_spikes.empty());
    }

    // Process the next event in the queue. This might be either a spike
    // ariving at a neuron (named synapse_spike), or a neuron spiking.
    void process_event()
    {
      if(! active())
        return;
      // which one first
      // compute_earliest_neuron_spike
      auto neuron_spike = std::ranges::max_element(predicted_spikes, [](const auto& a, const auto& b) noexcept {return a.time > b.time;});
      Neuron* updated_neuron;
      // bit of ugly logic to determine which type of event is first
      if(predicted_spikes.empty() || ((! synapse_spikes.empty()) && synapse_spikes.top().time < neuron_spike->time))
      { // process synapse
        auto& synapse_spike = synapse_spikes.top();
        updated_neuron = synapse_spike.neuron;
        // find neuron's existing fire-time
        neuron_spike = std::ranges::find_if(predicted_spikes, [updated_neuron](const auto& n) noexcept {return updated_neuron == n.neuron;});
        // update neuron
        updated_neuron->incoming_spike(synapse_spike.time, synapse_spike.weight);
        synapse_spikes.pop();
      }
      else
      { // process neuron
        // update post_synapses
        updated_neuron = neuron_spike->neuron;
        // record
        actual_spikes.emplace_back(updated_neuron, neuron_spike->time, updated_neuron->spikes.size());
        for(auto& outgoing_connection: updated_neuron->outgoing_connections)
          for(auto& post_synapse: outgoing_connection->synapses)
            synapse_spikes.emplace(outgoing_connection->post_neuron, post_synapse.weight, neuron_spike->time + post_synapse.delay);
        // update neuron itself, including gradients
        updated_neuron->spike(neuron_spike->time);
      }
      // remove affected neuron's spike
      if(neuron_spike != predicted_spikes.end())
        predicted_spikes.erase(neuron_spike);
      // check for new spike
      auto future_spike = updated_neuron->compute_future_spike();
      if(future_spike > 0)
        predicted_spikes.emplace_back(updated_neuron, future_spike);
    }
  };
  void backprop(const auto& actual_spikes, const double learning_rate)
  {
    for(const auto& spike: std::ranges::reverse_view{actual_spikes})
      spike.neuron->backprop_spike(spike.index, learning_rate);
  }

}

