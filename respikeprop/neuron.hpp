#pragma once

#include<vector>
#include<queue>
#include<algorithm>
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
    pre->outgoing_synapses.emplace_back(s);
    return s;
  };

  struct neuron
  {
    neuron(double timestep, std::string key = "neuron")
      : uM(0)
      , uS(0)
      , uR(0)
      , timestep(timestep)
      , key(key)
    {}
    std::vector<std::weak_ptr<synapse>> outgoing_synapses;
    std::vector<std::weak_ptr<synapse>> incoming_synapses;
    std::priority_queue<std::pair<double, double>> incoming_spike_times;
    std::vector<double> spike_times;
    double uM;
    double uS;
    double uR;
    const double timestep;
    const double tauM = 4.0;
    const double tauS = 2.0;
    const double tauR = 20.0;
    const double tauM_step = 1 / exp(timestep / tauM);
    const double tauS_step = 1 / exp(timestep / tauS);
    const double tauR_step = 1 / exp(timestep / tauR);
    std::string key;
  };

  template<typename... Args>
  auto make_neuron(Args&&... args)
  {
    return std::make_shared<neuron>(args...);
  };

  void fire(neuron& n, double time)
  {
    n.spike_times.emplace_back(time);
    n.uR -= 1.;
    for(auto outgoing_synapse_weak: n.outgoing_synapses)
    {
      auto outgoing_synapse = outgoing_synapse_weak.lock();
      outgoing_synapse->post->incoming_spike_times.emplace(- time - outgoing_synapse->delay, outgoing_synapse->weight);
    }
  }

  void forward_propagate(neuron& n, double time)
  {
    while(!n.incoming_spike_times.empty() && - n.incoming_spike_times.top().first < time)
    {
      n.uM += n.incoming_spike_times.top().second;
      n.uS -= n.incoming_spike_times.top().second;
      n.incoming_spike_times.pop();
    }

    n.uM *= n.tauM_step;
    n.uS *= n.tauS_step;
    n.uR *= n.tauR_step;
    const double threshold = 1.;
    if(n.uM + n.uS + n.uR > threshold)
      fire(n, time);
  }

  void reset_neuron(neuron& n)
  {
    n.uM = n.uS = n.uR = 0;
    n.spike_times.clear();
    while(! n.incoming_spike_times.empty())
      n.incoming_spike_times.pop();
  }

}

