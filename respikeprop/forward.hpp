#pragma once

#include<cmath>
#include"neuron.hpp"

namespace resp {

  void fire(neuron& n, double time)
  {
    n.spike_times.emplace_back(time);
    n.uR -= 1.;
    for(auto outgoing_synapse_weak: n.outgoing_synapses)
    {
      auto outgoing_synapse = outgoing_synapse_weak.lock();
      outgoing_synapse->post->incoming_spike_times.emplace(time + outgoing_synapse->delay, outgoing_synapse->weight);
    }
  }

  void forward_propagate(neuron& n, double time)
  {
    while(!n.incoming_spike_times.empty() && n.incoming_spike_times.top().time < time)
    {
      n.uM += n.incoming_spike_times.top().weight;
      n.uS -= n.incoming_spike_times.top().weight;
      n.incoming_spike_times.pop();
    }

    n.uM *= n.tauM_step;
    n.uS *= n.tauS_step;
    n.uR *= n.tauR_step;
    const double threshold = 1.;
    if(n.uM + n.uS + n.uR > threshold)
      fire(n, time);
  }

}

