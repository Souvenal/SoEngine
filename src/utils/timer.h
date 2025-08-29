#pragma once

#include <chrono>

class Timer {
public:
    using Seconds = std::ratio<1>;
    using Milliseconds = std::ratio<1, 1000>;
    using Microseconds = std::ratio<1, 1000000>;
    using Nanoseconds = std::ratio<1, 1000000000>;

    using Clock = std::chrono::steady_clock;
    using DefaultResolution = Seconds;

    Timer();
    ~Timer() = default;

    /**
     * @brief Starts the timer
     */
    void start();

    void lap();

    /**
     * @brief Stops the timer
     * @return The duration between the start and stop time points (default in seconds)
     */
    template <typename T = DefaultResolution>
    double stop() {
        if (!running) {
            return 0;
        }

        running = false;
        lapping = false;
        auto duration = std::chrono::duration<double, T>(Clock::now() - startTime);
        startTime = Clock::now();
        lapTime = startTime;

        return duration.count();
    }

    /**
     * @brief Calculates the time between start and now, or between
     * last lap and now if `lap()` was called
     * @return The duration between the two time points (default in seconds)
     */
    template <typename T = DefaultResolution>
    [[nodiscard]] double elapsed() {
        if (!running) {
            return 0;
        }

        Clock::time_point start = lapping ? lapTime : startTime;

        auto duration = std::chrono::duration<double, T>(Clock::now() - start);
        return duration.count();
    }

    /**
     * @brief Calculates the time between the last tick and now,
     * @return The duration between the two ticks (default in seconds)
     */
    template <typename T = DefaultResolution>
    double tick() {
        auto now = Clock::now();
        auto duration = std::chrono::duration<double, T>(now - lastTick);
        lastTick = now;
        return duration.count();
    }

    [[nodiscard]] bool isRunning() const {
        return running;
    }

private:
    bool running    { false };
    bool lapping    { false };
    Clock::time_point startTime;
    Clock::time_point lapTime;
    Clock::time_point lastTick;
};
