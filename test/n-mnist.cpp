#include<iostream>
#include<respikeprop/load_n_mnist.hpp>

// Dataset can be downloaded from https://www.garrickorchard.com/datasets/n-mnist
int main()
{
  using namespace resp;
  std::cout << "Loading spike patterns..." << std::endl;
  auto spike_patterns = load_n_mnist_training();
  std::cout << "Loaded " << spike_patterns.size() << " patterns" << std::endl;

  return 0;
}

