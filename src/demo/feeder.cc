#include "demo/feeder.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"

namespace {
std::string getYouTube(const std::string& youtube_url) {
#ifdef _MSC_VER
    std::cerr << "Youtube input is not implemented for MSVC.\n";
    return "";
#else
    char buf[128];
    std::string URL;
    std::string cmd = "yt-dlp -f \"best[height<=720][width<=1280]\" -g " + youtube_url;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return URL;
    }

    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        URL += buf;
    }
    pclose(pipe);

    if (!URL.empty()) {
        URL.erase(URL.find('\n'));
    }
    return URL;
#endif
}

bool openCaptureBySetting(const FeederSetting& feeder_setting, int index,
                          cv::VideoCapture& cap, bool& delay_on) {
    const std::string& src = feeder_setting.src_path[index];
    switch (feeder_setting.feeder_type) {
    case FeederType::CAMERA: {
        cap.open(stoi(src), cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
        cap.set(cv::CAP_PROP_FPS, 30);
        delay_on = false;
        break;
    }
    case FeederType::VIDEO: {
        cap.open(src, cv::CAP_FFMPEG);
        delay_on = true;
        break;
    }
    case FeederType::IPCAMERA: {
        cap.open(src);
        delay_on = false;
        break;
    }
    case FeederType::YOUTUBE: {
        cap.open(getYouTube(src));
        delay_on = true;
        break;
    }
    }
    return cap.isOpened();
}

bool rewindCapture(cv::VideoCapture& cap) {
    if (!cap.isOpened()) {
        return false;
    }
    if (!cap.set(cv::CAP_PROP_POS_FRAMES, 0)) {
        return false;
    }
    // Some backends report the first frame as 0 or 1 after seek.
    double pos = cap.get(cv::CAP_PROP_POS_FRAMES);
    return pos <= 1.0;
}
}  // namespace

Feeder::Feeder(const FeederSetting& feeder_setting) : mFeederSetting(feeder_setting) {
    for (int i = 0; i < mFeederSetting.src_path.size(); i++) {
        cv::VideoCapture cap;
        openCaptureBySetting(mFeederSetting, i, cap, mDelayOn);
        mCap.push_back(cap);
    }
}

void Feeder::feed(int index, ItemQueue& item_queue, cv::Size roi_size) {
    mFeederBuffer.open();
    while (mIsFeederRunning) {
        for (int i = 0; i < mCap.size(); i++) {
            const bool is_loop_source =
                (mFeederSetting.feeder_type == FeederType::VIDEO ||
                 mFeederSetting.feeder_type == FeederType::YOUTUBE);

            if (!mCap[i].isOpened()) {
                if (is_loop_source) {
                    openCaptureBySetting(mFeederSetting, i, mCap[i], mDelayOn);
                    if (!mCap[i].isOpened()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        continue;
                    }
                } else {
                    feedInternalDummy(index, item_queue, roi_size);
                    continue;
                }
            }

            feedInternal(index, item_queue, mCap[i], roi_size, mDelayOn);
            if (!mIsFeederRunning) {
                break;
            }

            if (is_loop_source) {
                bool rewind_ok = rewindCapture(mCap[i]);
                if (!rewind_ok || !mCap[i].isOpened()) {
                    mCap[i].release();
                    openCaptureBySetting(mFeederSetting, i, mCap[i], mDelayOn);
                }
            } else {
                rewindCapture(mCap[i]);
            }
        }
    }
    mFeederBuffer.close();
}

void Feeder::feedInternal(int index, ItemQueue& item_queue, cv::VideoCapture& cap,
                          cv::Size roi_size, bool delay_on) {
    Benchmarker benchmarker;
    while (true) {
        benchmarker.start();

        cv::Mat frame;
        cap >> frame;
        if (frame.empty() || !mIsFeederRunning) {
            break;
        }

        mFeederBuffer.put(frame);

        if (!roi_size.empty()) {
            cv::Mat resized_frame;
            resize(frame, resized_frame, roi_size);

            ItemQueue::StatusCode sc;
            sc = item_queue.push(
                {index, resized_frame, benchmarker.getFPS(), 0.0f, 0.0f, 0});
            if (sc != ItemQueue::OK) {
                break;
            }
        }

        if (delay_on) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        benchmarker.end();
    }
}

void Feeder::feedInternalDummy(int index, ItemQueue& item_queue, cv::Size roi_size) {
    Benchmarker benchmarker;
    while (true) {
        benchmarker.start();

        cv::Mat frame;
        frame = cv::Mat::zeros(360, 640, CV_8UC3);
        cv::putText(frame, "Dummy Feeder", cv::Point(140, 190), cv::FONT_HERSHEY_DUPLEX,
                    1.5, cv::Scalar(0, 255, 0), 2);
        if (frame.empty() || !mIsFeederRunning) {
            break;
        }

        mFeederBuffer.put(frame);

        if (!roi_size.empty()) {
            cv::Mat resized_frame;
            resize(frame, resized_frame, roi_size);

            ItemQueue::StatusCode sc;
            sc = item_queue.push(
                {index, resized_frame, benchmarker.getFPS(), 0.0f, 0.0f, 0});
            if (sc != ItemQueue::OK) {
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        benchmarker.end();
    }
}
