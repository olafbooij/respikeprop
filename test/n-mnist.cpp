// Dataset can be downloaded from https://www.garrickorchard.com/datasets/n-mnist
#include<iostream>
#include<vector>
#include<fstream>

namespace resp
{
  struct Event
  {
    uint8_t x;
    uint8_t y;
    bool polarity;
    int timestamp;
  };

  auto load_spike_pattern(auto&& file)
  {
    std::vector<Event> events;
    while(file.good())
    {
      Event event;
      event.x = file.get() + 1;
      event.y = file.get() + 1;
      uint8_t c = file.get();
      event.polarity = c >> 7;
      event.timestamp = (c & 127) << 16;
      event.timestamp += file.get() << 8;
      event.timestamp += file.get();
      if(file.good())
        if(event.x < 29 && event.y < 29)  // about 5% of the events seem to be outside the given 28x28 frame, dropping them
          events.emplace_back(event);
    }
    return events;
  }
}

int main()
{
  using namespace std;
  using namespace resp;
  auto events = load_spike_pattern(std::ifstream("datasets/n-mnist/Train/0/00002.bin", std::ios::binary));

  for(auto event: events)
    std::cout << static_cast<int>(event.x) << " " << static_cast<int>(event.y) << " " << event.polarity << " " << event.timestamp << std::endl;

  return 0;
}

