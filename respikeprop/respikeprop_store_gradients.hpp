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
        //Synapse(double weight_, double delay_)
        //: weight(weight_)
        //, delay(delay_)
        //, delta_weight(0.) {}
        double weight;
        double delay;
        double delta_weight;
        std::vector<double> dt_dws;  // same order as spikes
      };
      std::vector<Synapse> synapses;
      std::vector<std::vector<double>> dprets_dpostts; // per prespike per postspike
    };
    std::vector<Connection> incoming_connections;
    struct Spike
    {
      double time;
      std::vector<std::vector<double>> dpostt_dts; // per spike per post-neuron
    };
    std::vector<Spike> spikes;  // Eq (1)
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
        for(auto& incoming_synapse: incoming_connection.synapses)
          incoming_synapse.dt_dws.clear();
        for(auto& pre_spike: incoming_connection.neuron->spikes)
          pre_spike.dpostt_dts.clear();
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

    void forward_propagate(double time)  // Eq (2)
    {
      const double threshold = 1.;
      double u = 0.;
      for(const auto& incoming_connection: incoming_connections)
        for(const auto& pre_spike: incoming_connection.neuron->spikes)
          for(const auto& synapse: incoming_connection.synapses)
            u += synapse.weight * epsilon(time - pre_spike.time - synapse.delay);
      for(const auto& ref_spike: spikes)
        u += eta(time - ref_spike.time);

      if(u > threshold)
      {
        double du_dt = 0.;
        {
          for(const auto& incoming_connection: incoming_connections)
            for(const auto& pre_spike: incoming_connection.neuron->spikes)
              for(const auto& synapse: incoming_connection.synapses)
                du_dt += synapse.weight * epsilond(time - pre_spike.time - synapse.delay);
          for(const auto& ref_spike: spikes)
            du_dt += etad(time - ref_spike.time);
          if(du_dt < .1) // handling discontinuity circumstance 1 Sec 3.2
            du_dt = .1;
        }
        for(auto& incoming_connection: incoming_connections)
          for(auto& synapse: incoming_connection.synapses)
          {
            double du_dw = 0.;
            {
              for(const auto& pre_spike: incoming_connection.neuron->spikes)
                du_dw += epsilon(time - pre_spike.time - synapse.delay);
              for(const auto& [ref_spike, dt_dw]: ranges::views::zip(spikes, synapse.dt_dws))
                du_dw += - etad(time - ref_spike.time) * dt_dw;
            }
            double dt_dw = - du_dw / du_dt;
            synapse.dt_dws.emplace_back(dt_dw);
          }

        {
          for(auto& incoming_connection: incoming_connections)
          {
            incoming_connection.dprets_dpostts.resize(incoming_connection.neuron->spikes.size());  // make sure there's an entry for all pre spikes
            for(const auto& [pre_spike, dpret_dpostts]: ranges::views::zip(incoming_connection.neuron->spikes, incoming_connection.dprets_dpostts))
            {
              dpret_dpostts.resize(spikes.size(), 0.);  // make sure there's an entry for all post spikes
              double dpostu_dt = 0.;
              for(const auto& [ref_spike, dpret_dpostt]: ranges::views::zip(spikes, dpret_dpostts))
                dpostu_dt -= etad(time - ref_spike.time) * dpret_dpostt;
              double dpostt_dt = - dpostu_dt / du_dt;
              for(const auto& synapse: incoming_connection.synapses)
                dpostt_dt += synapse.weight * epsilond(time - pre_spike.time - synapse.delay) / du_dt;
              dpret_dpostts.emplace_back(dpostt_dt);
            }
          }
        }
        spikes.emplace_back(time);
      }
    }

    // Results in a bit of double work, because each dE_dt change is pushed
    // back separately. Could do this more efficient spike could know all
    // resulting post-spikes have been back-propagated.
    // In addition, using indices here is not that nice perhaps.
    void add_dE_dt(int spike_i, double dE_dt, double learning_rate)
    {
      for(auto& incoming_connection: incoming_connections)
      {
        for(auto& synapse: incoming_connection.synapses)
          synapse.delta_weight -= learning_rate * dE_dt * synapse.dt_dws.at(spike_i);
        for(int pre_spike_i = 0; pre_spike_i < incoming_connection.neuron->spikes.size(); ++pre_spike_i)
          incoming_connection.neuron->add_dE_dt(pre_spike_i, dE_dt * incoming_connection.dprets_dpostts.at(pre_spike_i).at(spike_i), learning_rate);
      }
    }
    void compute_delta_weights(const double learning_rate)  // missnomer, starts backprop
    {
      if(clamped > 0)  // to check that this is an output neuron
        if(! spikes.empty())
          add_dE_dt(0, spikes.front().time - clamped, learning_rate);
    }
  };
}

