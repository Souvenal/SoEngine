#include "timer.h"

Timer::Timer():
    startTime(Clock::now()), lastTick(Clock::now())
{}

void Timer::start() {
    if (!running) {
        running = true;
        startTime = Clock::now();
        lapTime = startTime;
        lastTick = startTime;
    }
}

void Timer::lap() {
    if (running) {
        lapping = true;
        lapTime = Clock::now();
    }
}
