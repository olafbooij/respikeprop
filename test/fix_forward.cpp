#include<iostream>
#include<cassert>
#include<respikeprop/neuron.hpp>

int main()
{
  using namespace std;
  using namespace resp;
  auto in = std::make_shared<neuron>();
  auto out = std::make_shared<neuron>();
  auto s = make_synapse(out, in, 1.8, 1.0);
  in->outgoing_synapses.emplace_back(s);
  fire(*in, 1.0);
  fire(*in, 4.0);
  for(double time = 0.; time < 10.; time += .001)
  {
    forward_propagate(*out, time);
    //if(! out->spike_times.empty())
    //{
    //  cout << time << endl;
    //  out->spike_times.clear();
    //}
  }
  assert(out->spike_times.size() == 1) ;
  assert(fabs(out->spike_times.at(0) - 5.729) < .01) ;





  return 0;
}

