#include <iostream>
#include <chrono>
#include <string>

class Timer
{
  const std::chrono::time_point<std::chrono::system_clock> start;
  const std::string name;

public:
  Timer(const std::string& name_in): start(std::chrono::system_clock::now()), name(name_in) {};
  ~Timer()
  {
    auto end = std::chrono::system_clock::now();
    std::cout << "Function " << name
              << " takes " << std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start).count()
              << " us" << std::endl;
  }
};
