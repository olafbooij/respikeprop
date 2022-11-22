#pragma once

#include<vector>
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
        // vector of future incoming spikes...
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
    double u_m;
    double u_s;
    double u_r;
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
      u_m = u_s = u_r = 0;
    }

    void fire(double time)
    {
      spikes.emplace_back(time);
    }

    // Forward propagate and store gradients for backpropagation. For now
    // keeping this as one function. To be refactored.
    void forward_propagate(double time, double time_step)
    {
      //std::cout << key << " " << time << std::endl;
      const double threshold = 1.;
      // check recent incoming spikes
      for(const auto& incoming_connection: incoming_connections)
      {
        auto& pre_spikes = incoming_connection.neuron->spikes;
        for(const auto& synapse: incoming_connection.synapses)
          for(auto pre_spike = pre_spikes.rbegin(); pre_spike != pre_spikes.rend() && (time - *pre_spike - synapse.delay < time_step); pre_spike++)
            if(time - *pre_spike - synapse.delay >= 0)  // might result in some hair-trigger problems
            {
              u_m += synapse.weight;
              u_s -= synapse.weight;
            }
      }
      // update potentials
      u_m *= exp(- time_step / tau_m);
      u_s *= exp(- time_step / tau_s);
      u_r *= exp(- time_step / tau_r);
      double u = u_m + u_s + u_r;

      if(u > threshold) // fire
      {
        double du_dt = - u_m / tau_m - u_s / tau_s - u_r / tau_r;
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
              double s = time - pre_spike - synapse.delay;
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
            double s = time - ref_spike;
            if(s >= 0)
            {
              double u_r1 = exp(-s / tau_r) / tau_r;
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
        spikes.emplace_back(time);
        u_r -= threshold;
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
  };
}

