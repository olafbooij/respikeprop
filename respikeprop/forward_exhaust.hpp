#pragma once

#include<vector>
#include<memory>
#include<cmath>

namespace resp {

  struct neuron
  {
    neuron(std::string key = "neuron")
      : key(key)
    {}
    struct synapse
    {
      const neuron& pre;  // putting a lot of responsibility on user...
      const double weight;
      const double delay;
    };
    std::vector<synapse> incoming_synapses;
    std::vector<double> spikes;
    const double tau_m = 4.0;
    const double tau_s = 2.0;
    const double tau_r = 20.0;
    const std::string key;

    auto epsilon(auto s)
    {
      if(s < 0.)
        return 0.;
      else
        return exp(-s / tau_m) - exp(-s / tau_s);
    };
    auto eta(auto s)
    {
      if(s < 0.)
        return 0.;
      else
        return - exp(-s / tau_r);
    };

    void fire(double time)
    {
      spikes.emplace_back(time);
    }

    void forward_propagate(double time)
    {
      double u;
      for(auto incoming_synapse: incoming_synapses)
        for(auto pre_spike: incoming_synapse.pre.spikes)
          u += incoming_synapse.weight * epsilon(time - pre_spike - incoming_synapse.delay);
      for(auto ref_spike: spikes)
        u += eta(time - ref_spike);
      const double threshold = 1.;
      if(u > threshold)
        fire(time);
    }
  };

}

