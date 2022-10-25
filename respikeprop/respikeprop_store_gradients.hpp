#pragma once

#include<vector>
#include<cmath>

namespace resp {

  // Straightforward implementation of Spikeprop with multiple spikes per
  // neuron, with links to equations from the paper:
  // An implementation of "A gradient descent rule for spiking neurons emitting
  // multiple spikes", Olaf Booij, Hieu tat Nguyen, Information Processing
  // Letters, Volume 95, Issue 6, 30 September 2005, Pages 552-558.
  //
  // Forward propagation is not event-based and thus compute time in the order
  // of time-steps.
  // Backpropagation in this implementation is very compute heavy.
  // Network connectivity is implemented using raw-pointers, leaving
  // responsibility of memory management with the user.

  struct Neuron
  {
    Neuron(std::string key_ = "neuron") : key(key_) {}
    struct Synapse
    {
      Synapse(const Neuron& pre_, double weight_, double delay_)
      : pre(&pre_)  // taking raw address
      , weight(weight_)
      , delay(delay_)
      , delta_weight(0.) {}
      const Neuron* pre;  // putting a lot of responsibility on user...
      double weight;
      double delay;
      double delta_weight;
    };
    std::vector<Synapse> incoming_synapses;
    std::vector<Neuron*> post_neuron_ptrs;
    struct Spike
    {
      double time;
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
      for(auto incoming_synapse: incoming_synapses)
        for(auto pre_spike: incoming_synapse.pre->spikes)
          u += incoming_synapse.weight * epsilon(time - pre_spike.time - incoming_synapse.delay);
      for(auto ref_spike: spikes)
        u += eta(time - ref_spike.time);
      if(u > threshold)
        spikes.emplace_back(time);
    }

    void compute_delta_weights(const double learning_rate)  // Eq (9)
    {
      for(auto& synapse: incoming_synapses)
        for(auto& spike: spikes)
          synapse.delta_weight -= learning_rate * compute_dE_dt(spike) * compute_dt_dw(synapse, spike);
    }
    double compute_dt_dw(auto& synapse, const auto& spike)  // Eq (10)
    {
      return - compute_du_dw(synapse, spike) / compute_du_dt(spike);
    }
    double compute_du_dw(auto& synapse, const auto& spike)  // Eq (11)
    {
      double du_dw = 0.;
      for(auto& pre_spike: synapse.pre->spikes)
        du_dw += epsilon(spike.time - pre_spike.time - synapse.delay);
      for(auto& ref_spike: spikes)
        if(ref_spike.time < spike.time)
          du_dw += - etad(spike.time - ref_spike.time) * compute_dt_dw(synapse, ref_spike);
      return du_dw;
    }
    double compute_du_dt(const auto& spike)  // Eq (12)
    {
      double du_dt = 0.;
      for(auto& synapse: incoming_synapses)
        for(auto& pre_spike: synapse.pre->spikes)
          du_dt += synapse.weight * epsilond(spike.time - pre_spike.time - synapse.delay);
      for(auto& ref_spike: spikes)
        if(ref_spike.time < spike.time)
          du_dt += etad(spike.time - ref_spike.time);
      if(du_dt < .1) // handling discontinuity circumstance 1 Sec 3.2
        du_dt = .1;
      return du_dt;
    }
    double compute_dE_dt(const auto& spike)  // Eq (13)
    {
      if(clamped > 0.)
        if(spike.time == spikes.front().time)
          return spike.time - clamped;

      double dE_dt = 0.;
      for(auto post_neuron_ptr: post_neuron_ptrs)
        for(auto& post_spike: post_neuron_ptr->spikes)
          if(post_spike.time > spike.time)
            dE_dt += post_neuron_ptr->compute_dE_dt(post_spike) * compute_dpostt_dt(spike, *post_neuron_ptr, post_spike);
      return dE_dt;
    }
    double compute_dpostt_dt(const auto& spike, auto& post_neuron, const auto& post_spike)  // Eq (14)
    {
      return - compute_dpostu_dt(spike, post_neuron, post_spike) / post_neuron.compute_du_dt(post_spike);
    }
    double compute_dpostu_dt(const auto& spike, auto& post_neuron, const auto& post_spike)  // Eq (15)
    {
      double dpostu_dt = 0.;
      for(auto& synapse: post_neuron.incoming_synapses)
        if(synapse.pre == this)
          dpostu_dt -= synapse.weight * epsilond(post_spike.time - spike.time - synapse.delay);
      for(auto& ref_post_spike: post_neuron.spikes)
        if(ref_post_spike.time < post_spike.time)
          dpostu_dt -= etad(post_spike.time - ref_post_spike.time) * compute_dpostt_dt(spike, post_neuron, ref_post_spike);
      return dpostu_dt;
    }
  };

}

