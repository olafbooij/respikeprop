#pragma once

#include<vector>
#include<cmath>
#include<range/v3/view/zip.hpp>

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

  struct Neuron
  {
    Neuron(std::string key_ = "neuron") : key(key_) {}
    struct Connection
    {
      Neuron* neuron;  // putting a lot of responsibility on user...
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
    std::vector<Connection> incoming_connections;
    std::vector<double> spikes;
    // The following settings are taken from the thesis "Temporal Pattern
    // Classification using Spiking Neural Networks" which differ from the
    // paper.
    double tau_m = 4.0;
    double tau_s = 2.0;
    double tau_r = 20.0;
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
    }
    auto epsilon(const auto s) const  // Eq (4)
    {
      if(s < 0.)
        return 0.;
      else
        return exp(-s / tau_m) - exp(-s / tau_s);
    };
    auto eta(const auto s) const  // Eq (5)
    {
      if(s < 0.)
        return 0.;
      else
        return - exp(-s / tau_r);
    };
    auto epsilond(const auto s)
    {
      if(s < 0.)
        return 0.;
      else
        return - exp(-s / tau_m) / tau_m + exp(-s / tau_s) / tau_s;
    };
    auto etad(const auto s)
    {
      if(s < 0.)
        return 0.;
      else
        return exp(-s / tau_r) / tau_r;
    };

    void fire(double time)
    {
      spikes.emplace_back(time);
    }

    // Forward propagate and store gradients for backpropagation. For now
    // keeping this as one function. To be refactored.
    void forward_propagate(double time)
    {
      const double threshold = 1.;
      double u = 0.;
      for(const auto& incoming_connection: incoming_connections)
        for(const auto& pre_spike: incoming_connection.neuron->spikes)
          for(const auto& synapse: incoming_connection.synapses)
            u += synapse.weight * epsilon(time - pre_spike - synapse.delay);
      for(const auto& ref_spike: spikes)
        u += eta(time - ref_spike);

      if(u > threshold) // fire
      {
        // computing du_dt, Eq (12)
        double du_dt = 0.;
        {
          for(const auto& incoming_connection: incoming_connections)
            for(const auto& pre_spike: incoming_connection.neuron->spikes)
              for(const auto& synapse: incoming_connection.synapses)
                du_dt += synapse.weight * epsilond(time - pre_spike - synapse.delay);
          for(const auto& ref_spike: spikes)
            du_dt += etad(time - ref_spike);
          if(du_dt < .1) // handling discontinuity circumstance 1 Sec 3.2
            du_dt = .1;
        }

        // computing and storing dt_dws, Eq (10)
        for(auto& incoming_connection: incoming_connections)
          for(auto& synapse: incoming_connection.synapses)
          {
            double du_dw = 0.;
            {
              for(const auto& pre_spike: incoming_connection.neuron->spikes)
                du_dw += epsilon(time - pre_spike - synapse.delay);
              for(const auto& [ref_spike, dt_dw]: ranges::views::zip(spikes, synapse.dt_dws))
                du_dw += - etad(time - ref_spike) * dt_dw;
            }
            double dt_dw = - du_dw / du_dt;
            synapse.dt_dws.emplace_back(dt_dw);
          }

        // computing and storing dt_dts, Eq (14)
        {
          for(auto& incoming_connection: incoming_connections)
          {
            incoming_connection.dprets_dpostts.resize(incoming_connection.neuron->spikes.size());  // make sure there's an entry for all pre spikes
            for(const auto& [pre_spike, dpret_dpostts]: ranges::views::zip(incoming_connection.neuron->spikes, incoming_connection.dprets_dpostts))
            {
              dpret_dpostts.resize(spikes.size(), 0.);  // make sure there's an entry for all post spikes
              double dpostu_dt = 0.;
              for(const auto& [ref_spike, ref_dpret_dpostt]: ranges::views::zip(spikes, dpret_dpostts))
                dpostu_dt -= etad(time - ref_spike) * ref_dpret_dpostt;
              for(const auto& synapse: incoming_connection.synapses)
                dpostu_dt -= synapse.weight * epsilond(time - pre_spike - synapse.delay);
              double dpret_dpostt = - dpostu_dt / du_dt;
              dpret_dpostts.emplace_back(dpret_dpostt);
            }
          }
        }
        spikes.emplace_back(time);
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
          synapse.delta_weight -= learning_rate * dE_dt * synapse.dt_dws.at(spike_i);
        for(int pre_spike_i = 0; pre_spike_i < incoming_connection.neuron->spikes.size(); ++pre_spike_i)
          if(spikes.at(spike_i) > incoming_connection.neuron->spikes.at(pre_spike_i))
            incoming_connection.neuron->add_dE_dt(pre_spike_i, dE_dt * incoming_connection.dprets_dpostts.at(pre_spike_i).at(spike_i), learning_rate);
      }
    }
    void compute_delta_weights(const double learning_rate)  // missnomer, starts backprop
    {
      if(clamped > 0)  // to check that this is an output neuron
        if(! spikes.empty())
          add_dE_dt(0, spikes.front() - clamped, learning_rate);
    }
  };
}

