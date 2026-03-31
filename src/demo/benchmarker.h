#ifndef DEMO_BENCHMARKER_H_
#define DEMO_BENCHMARKER_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <numeric>

class Benchmarker {
    using Clock = std::chrono::system_clock;
    static constexpr size_t SIZE = 100;

public:
    Benchmarker() { mCreated = Clock::now(); }
    void start() { mPrev = Clock::now(); }
    void end() {
        std::chrono::duration<float> dt = Clock::now() - mPrev;

        if (mCount >= SIZE) {
            mSum -= mTimes[mCount % SIZE];
        }

        float t = dt.count();
        mTimes[mCount++ % SIZE] = t;
        mSum += t;
        mRunningTime += t;
    }

    float getSec() const {
        if (mCount == 0) {
            return 0;
        }

        return mSum / (SIZE < mCount ? SIZE : mCount);
    }

    float getFPS() const {
        float avg_time = getSec();

        if (avg_time == 0) {
            return 0;
        }
        return 1 / avg_time;
    }

    float getRunningTime() const { return mRunningTime; }

    size_t getCount() const { return mCount; }

    float getTimeSinceCreated() const {
        std::chrono::duration<float> dt = Clock::now() - mCreated;
        return dt.count();
    }

private:
    std::array<float, SIZE> mTimes;
    float mSum = 0;
    size_t mCount = 0;
    Clock::time_point mPrev;
    Clock::time_point mCreated;
    float mRunningTime = 0;
};

#endif  // DEMO_BENCHMARKER_H_
