#pragma once

#include<algorithm>
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

  auto load_n_mnist_training(double part, auto& random_gen, auto&& labels)
  {
    std::vector<Pattern> spike_patterns;
    for(const auto label : labels)
      for (auto const& file : std::filesystem::directory_iterator{"datasets/n-mnist/Train/" + std::to_string(label)}) 
        if(std::uniform_real_distribution<>()(random_gen) < part)
          spike_patterns.emplace_back(load_spike_pattern(std::ifstream(file.path(), std::ios::binary), label));
    return spike_patterns;
  }
  auto load_n_mnist_training(double part, auto& random_gen)
  {
    return load_n_mnist_training(part, random_gen, std::views::iota(0, 10));
  }

  auto decimate_events(const auto& patterns, const std::size_t nr_of_events, auto& random_gen)
  {
    std::vector<Pattern> patterns_decimated;
    for(const auto& pattern: patterns)
    {
      std::vector<Event> subset;
      std::ranges::sample(pattern.events, std::back_inserter(subset), nr_of_events, random_gen);
      patterns_decimated.emplace_back(subset, pattern.label);
    }
    return patterns_decimated;
  }

}

