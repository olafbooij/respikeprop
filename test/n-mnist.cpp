// Dataset can be downloaded from https://www.garrickorchard.com/datasets/n-mnist
#include<iostream>
#include<vector>
#include<fstream>

int main()
{
  using namespace std;
  ifstream file("datasets/n-mnist/Train/0/00002.bin", std::ios::binary);
  struct Event
  {
    uint8_t x;
    uint8_t y;
    bool polarity;
    int timestamp;
  };

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
      events.emplace_back(event);
  }
  for(auto event: events)
    std::cout << static_cast<int>(event.x) << " " << static_cast<int>(event.y) << " " << event.polarity << " " << event.timestamp << std::endl;

  return 0;
}

