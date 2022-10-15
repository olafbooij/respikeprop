#pragma once

#include<vector>
#include<memory>
#include<cmath>

namespace resp {

  struct neuron;
  struct synapse
  {
    const std::shared_ptr<neuron> post;
    const std::shared_ptr<neuron> pre; // should swap these
    double weight;
    double delay;
  };
  
  auto make_synapse = [](auto post, auto pre, auto weight, auto delay)
  {
    auto s = std::make_shared<synapse>(post, pre, weight, delay);
    post->incoming_synapses.emplace_back(s);
    return s;
  };

  struct neuron
  {
    neuron(std::string key = "neuron")
      : key(key)
    {}
    std::vector<std::weak_ptr<synapse>> incoming_synapses;
    std::vector<double> spikes;
    const double tau_m = 4.0;
    const double tau_s = 2.0;
    const double tau_r = 20.0;
    std::string key;

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
      for(auto incoming_synapse_weak: incoming_synapses)
      {
        auto incoming_synapse = incoming_synapse_weak.lock();
        for(auto pre_spike: incoming_synapse->pre->spikes)
          u += incoming_synapse->weight * epsilon(time - pre_spike - incoming_synapse->delay);
      }
      for(auto ref_spike: spikes)
        u += eta(time - ref_spike);
      const double threshold = 1.;
      if(u > threshold)
        fire(time);
    }
  };
  template<typename... Args>
  auto make_neuron(Args&&... args)
  {
    return std::make_shared<neuron>(args...);
  };


}

