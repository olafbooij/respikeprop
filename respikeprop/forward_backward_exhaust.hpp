#pragma once

#include<vector>
#include<cmath>

namespace resp {

  struct neuron
  {
    neuron(std::string key = "neuron") : key(key) {}
    struct synapse
    {
      const neuron& pre;  // putting a lot of responsibility on user...
      double weight;
      double delay;
      double delta_weight;
    };
    std::vector<synapse> incoming_synapses;
    std::vector<double> spikes;  // Eq (1)
    double tau_m = 4.0;
    double tau_s = 2.0;
    double tau_r = 20.0;
    std::string key;

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
      if(compute_u > threshold)
        fire(time);
    }
    double compute_u(double time)  // Eq (3)
    {
      double u;
      for(auto incoming_synapse: incoming_synapses)
        for(auto pre_spike: incoming_synapse.pre.spikes)
          u += incoming_synapse.weight * epsilon(time - pre_spike - incoming_synapse.delay);
      for(auto ref_spike: spikes)
        u += eta(time - ref_spike);
      return u;
    }

    void compute_delta_weights(const double learning_rate)  // Eq (9)
    {
      for(auto& synapse: incoming_synapses)
        for(auto& spike: spikes)
          synapse.delta_weight += learning_rate * compute_dE_dt(spike) * compute_dt_dw(synapse, spike);
    }
    double compute_dt_dw(auto& synapse, const auto& spike)  // Eq (10)
    {
      return - compute_du_dw(synapse, spike) / compute_du_dt(spike);
    }
    double compute_du_dw(auto& synapse, const auto& spike)  // Eq (11)
    {
      double du_dw = 0.;
      for(auto& pre_spike: synapse.pre.spikes)
        du_dw += epsilon(spike - pre_spike - synapse.delay);
      for(auto& ref_spike: spikes)
        if(ref_spike < spike)
          du_dw += etad(spike - ref_spike) * compute_dt_dw(synapse, ref_spike);
      return du_dw;
    }
    double compute_du_dt(const auto& spike)  // Eq (12)
    {
      return 0.;
    }
    double compute_dE_dt(const auto& spike)  // Eq (13)
    {
      return 0.;
    }
    double compute_dt_dt(const auto& spike)  // Eq (14)
    {
      return 0.;
    }
  };

}

