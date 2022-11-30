#pragma once

#include<vector>
#include<fstream>
#include<filesystem>
#include<random>
#include<ranges>

namespace resp
{

  struct Event
  {
    uint8_t x;
    uint8_t y;
    bool polarity;
    int timestamp;
  };

  auto load_events(auto&& file)
  {
    std::vector<Event> events;
    while(file.good())
    {
      Event event;
      event.x = file.get();
      event.y = file.get();
      uint8_t c = file.get();
      event.polarity = c >> 7;
      event.timestamp = (c & 127) << 16;
      event.timestamp += file.get() << 8;
      event.timestamp += file.get();
      if(file.good())
        if(event.x < 28 && event.y < 28)  // about 5% of the events seem to be outside the given 28x28 frame, dropping them
          events.emplace_back(event);
    }
    return events;
  }

  struct Pattern
  {
    std::vector<Event> events;
    int label;
  };
  auto load_spike_pattern(auto&& file, int label)
  {
    return Pattern(load_events(file), label);
  }

  auto load_n_mnist_training(double part, auto& random_gen)
  {
    std::vector<Pattern> spike_patterns;
    for(const auto label : std::views::iota(0, 10))
      for (auto const& file : std::filesystem::directory_iterator{"datasets/n-mnist/Train/" + std::to_string(label)}) 
        if(std::uniform_real_distribution<>()(random_gen) < part)
          spike_patterns.emplace_back(load_spike_pattern(std::ifstream(file.path(), std::ios::binary), label));
    return spike_patterns;
  }

}

