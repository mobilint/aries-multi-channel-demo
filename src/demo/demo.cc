#include "demo/demo.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "demo/benchmarker.h"
#include "demo/feeder.h"
#include "demo/model.h"
#include "opencv2/opencv.hpp"
#include "qbruntime/qbruntime.h"

using mobilint::Accelerator;
using mobilint::Cluster;
using mobilint::Core;
using mobilint::ModelConfig;
using mobilint::StatusCode;
using namespace std;

namespace {
void sleepForMS(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

constexpr int kTimeOverlayClipWidth = 210;
constexpr int kTimeOverlayClipHeight = 45;
constexpr int kTimeOverlayX = 10;
constexpr int kTimeOverlayY = 10;

cv::Rect getTimeOverlayRect(const cv::Size& display_size) {
    if (display_size.width <= kTimeOverlayX || display_size.height <= kTimeOverlayY) {
        return cv::Rect();
    }

    constexpr float kOverlayWidthRatio = 0.15f;
    int overlay_w =
        std::max(1, static_cast<int>(display_size.width * kOverlayWidthRatio));
    int overlay_h =
        std::max(1, overlay_w * kTimeOverlayClipHeight / kTimeOverlayClipWidth);

    overlay_w = std::min(overlay_w, display_size.width - kTimeOverlayX);
    overlay_h = std::min(overlay_h, display_size.height - kTimeOverlayY);
    return cv::Rect(kTimeOverlayX, kTimeOverlayY, overlay_w, overlay_h);
}

std::string fpsToString(float fps) {
    char buf[20];
    snprintf(buf, sizeof(buf), "%8.2f", fps);
    return std::string(buf);
}

std::string secToString(int sec) {
    int h = sec / 3600;
    int m = (sec % 3600) / 60;
    int s = sec % 60;

    char buf[20];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d", h, m, s);
    return std::string(buf);
}

std::string countToString(int count) {
    char buf[20];
    snprintf(buf, sizeof(buf), "%8d", count);
    return std::string(buf);
}

// ROI 사이즈 변화에 대해서 일정한 폰트를 유지할 수 있도록 한다.
// - 고정된 사이즈(w, h)에 대해 해당 폰트 사이즈와 굵기로 Benchmark 창을 만든다.
// - 필요한 만큼만 Benchmark 창을 Clip한다.
// - Clip한 창을 scale만큼 resize 후 frame에 띄운다.
void displayBenchmark(Item& item, bool is_fps_only = false) {
    float scale = 0.55;
    int w = 300;
    int h = 200;
    double font_scale = 1.0;
    int font_thickness = 1;

    cv::Mat board = cv::Mat::zeros(h, w, CV_8UC3);

    putText(board, "FPS", cv::Point(15, 40), cv::FONT_HERSHEY_DUPLEX, font_scale,
            cv::Scalar(255, 255, 255), font_thickness);
    putText(board, fpsToString(item.fps), cv::Point(112, 40), cv::FONT_HERSHEY_DUPLEX,
            font_scale, cv::Scalar(0, 255, 0), font_thickness);

    putText(board, "NPU FPS", cv::Point(15, 80), cv::FONT_HERSHEY_DUPLEX, font_scale,
            cv::Scalar(255, 255, 255), font_thickness);
    putText(board, fpsToString(item.npu_fps), cv::Point(112, 80), cv::FONT_HERSHEY_DUPLEX,
            font_scale, cv::Scalar(0, 255, 0), font_thickness);

    // if (!is_fps_only) {
    //     putText(board, "Time", cv::Point(15, 120), cv::FONT_HERSHEY_DUPLEX,
    //             font_scale, cv::Scalar(255, 255, 255), font_thickness);
    //     putText(board, secToString(item.time), cv::Point(110, 120),
    //             cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(0, 255, 0),
    //             font_thickness);
    // }

    int clip_w = 265;
    int clip_h = 94;

    cv::Mat clip = board({0, 0, clip_w, clip_h});

    float resize_scale = (float)item.img.size().width / w * scale;
    // 이미지가 커지만 Debug 창도 같이 커진다.
    // 일정이상 비율이라면 더 이상 커지지 않게끔 한다.
    if (resize_scale > 0.9) {
        resize_scale = 0.9;
    }
    cv::resize(clip, clip, {0, 0}, resize_scale, resize_scale);

    int offset = (int)(item.img.size().width * 0.03);
    cv::Mat roi = item.img({{offset, offset}, clip.size()});
    cv::addWeighted(clip, 1, roi, 0.5, 0, roi);
}

void displayTime(cv::Mat& display, bool validate, float time = 0.0f) {
    if (!validate) {
        return;
    }

    cv::Rect overlay_rect = getTimeOverlayRect(display.size());
    if (overlay_rect.width <= 0 || overlay_rect.height <= 0) {
        return;
    }

    cv::Mat board = cv::Mat::zeros(overlay_rect.height, overlay_rect.width, CV_8UC3);
    const float sx = static_cast<float>(overlay_rect.width) / kTimeOverlayClipWidth;
    const float sy = static_cast<float>(overlay_rect.height) / kTimeOverlayClipHeight;
    const float text_scale = std::min(sx, sy);
    const double font_scale = 0.8 * text_scale;
    const int font_thickness = std::max(1, static_cast<int>(text_scale));

    putText(
        board, "Time", cv::Point(static_cast<int>(10 * sx), static_cast<int>(30 * sy)),
        cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness);
    putText(board, secToString(time),
            cv::Point(static_cast<int>(75 * sx), static_cast<int>(30 * sy)),
            cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(0, 255, 0), font_thickness);

    cv::Mat roi = display(overlay_rect);
    board.copyTo(roi);
}
}  // namespace

Demo::Demo()
    : mDisplayFPSMode(false),
      mDisplayTimeMode(false),
      mModeIndex(-1),
      mThreadPool(30) {}

void Demo::startWorker(int index) {
    const WorkerLayout& wl = mLayoutSetting.worker_layout[index];

    if (wl.model_index >= mModels.size() || wl.feeder_index >= mFeeders.size()) {
        return;
    }

    auto it = mModelFutureMap.find(index);
    if (it == mModelFutureMap.end()) {
        mSizeState[index]->open();
        auto future = mThreadPool.enqueue(
            Model::work, mModels[wl.model_index].get(), index, mSizeState[index].get(),
            &mMainQueueForWorker, &mFeeders[wl.feeder_index]->getMatBuffer());
        mModelFutureMap.emplace(index, std::move(future));
    }
}

void Demo::stopWorker(int index) {
    auto it = mModelFutureMap.find(index);
    if (it != mModelFutureMap.end()) {
        mSizeState[index]->close();
        if (it->second.valid()) {
            it->second.get();
        }
        mModelFutureMap.erase(it);
    }
}

void Demo::startFeeder(int index) {
    // 해당 조건으로 인해 feeder가 feeder_layout보다 많을 경우 size는 (0,0)이 되며,
    // size가 (0,0)일 경우 feeder는 display하지 않는다.
    cv::Size size;
    if (index < mLayoutSetting.feeder_layout.size()) {
        size = mLayoutSetting.feeder_layout[index].roi.size();
    }

    auto it = mFeederThreadMap.find(index);
    if (it == mFeederThreadMap.end()) {
        mFeeders[index]->start();
        mFeederThreadMap.emplace(
            index, std::thread(
                       [=] { mFeeders[index]->feed(index, mMainQueueForFeeder, size); }));
    }
}

void Demo::stopFeeder(int index) {
    auto it = mFeederThreadMap.find(index);
    if (it != mFeederThreadMap.end()) {
        mFeeders[index]->stop();
        it->second.join();
        mFeederThreadMap.erase(it);
    }
}

void Demo::startWorkerAll() {
    for (int i = 0; i < mLayoutSetting.worker_layout.size(); i++) {
        startWorker(i);
    }
}

void Demo::stopWorkerAll() {
    for (int i = 0; i < mSizeState.size(); i++) {
        mSizeState[i]->close();
    }

    for (int i = 0; i < mLayoutSetting.worker_layout.size(); i++) {
        stopWorker(i);
    }
}

void Demo::startFeederAll() {
    for (int i = 0; i < mFeeders.size(); i++) {
        startFeeder(i);
    }
}

void Demo::stopFeederAll() {
    for (int i = 0; i < mFeeders.size(); i++) {
        stopFeeder(i);
    }
}

void Demo::startThreads() {
    mWorkerWatchdog = std::thread(&Demo::workerReceive, this);
    mFeederWatchdog = std::thread(&Demo::feederReceive, this);
}

void Demo::joinThreads() {
    mMainQueueForWorker.close();
    mMainQueueForFeeder.close();
    mWorkerWatchdog.join();
    mFeederWatchdog.join();
}

int Demo::getWorkerIndex(int x, int y) {
    for (int i = 0; i < mLayoutSetting.worker_layout.size(); i++) {
        if (mLayoutSetting.worker_layout[i].roi.contains(cv::Point(x, y))) {
            return i;
        }
    }
    return -1;
}

void Demo::onMouseEvent(int event, int x, int y, int flags, void* ctx) {
    if (event != cv::EVENT_RBUTTONDOWN && event != cv::EVENT_LBUTTONDOWN) {
        return;
    }

    Demo* demo = (Demo*)ctx;
    int worker_index = demo->getWorkerIndex(x, y);
    if (worker_index == -1) {
        return;
    }

    switch (event) {
    case cv::EVENT_RBUTTONDOWN:
        demo->stopWorker(worker_index);
        break;
    case cv::EVENT_LBUTTONDOWN:
        demo->startWorker(worker_index);
        break;
    }
}

void Demo::feederReceive() {
    ItemQueue::StatusCode sc;
    Item item;
    while (true) {
        sc = mMainQueueForFeeder.pop(item);
        if (sc != ItemQueue::StatusCode::OK) {
            break;
        }

        if (mDisplayFPSMode) {
            displayBenchmark(item, true);
        }

        unique_lock<mutex> lock(mDisplayMutex);
        item.img.copyTo(mDisplay(mLayoutSetting.feeder_layout[item.index].roi));
    }
}

void Demo::workerReceive() {
    ItemQueue::StatusCode sc;
    Item item;
    while (true) {
        sc = mMainQueueForWorker.pop(item);
        if (sc != ItemQueue::StatusCode::OK) {
            break;
        }

        cv::Mat roi = mDisplay(mLayoutSetting.worker_layout[item.index].roi);

        // worker는 종료되는 시점에서 Mat()을 push한다.
        if (item.img.empty()) {
            unique_lock<mutex> lock(mDisplayMutex);
            roi = cv::Scalar(255, 255, 255);  // clear
            continue;
        }

        // 다른 사이즈의 img는 스킵한다.
        if (roi.size() != item.img.size()) {
            continue;
        }

        if (mDisplayFPSMode) {
            displayBenchmark(item);
        }
        unique_lock<mutex> lock(mDisplayMutex);
        item.img.copyTo(roi);
    }
}

void Demo::initWindow() {
    cv::Size window_size(1920, 1080);

    // Init Window
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_GUI_NORMAL);
    cv::resizeWindow(WINDOW_NAME, window_size / 2);
    cv::moveWindow(WINDOW_NAME, 0, 0);
    cv::setMouseCallback(WINDOW_NAME, onMouseEvent, this);

    mDisplay = cv::Mat(window_size, CV_8UC3, {255, 255, 255});
    mBackground = mDisplay.clone();

    mSplashes.clear();
    for (string path : {"../rc/layout/splash_01.png", "../rc/layout/splash_02.png"}) {
        cv::Mat splash = cv::imread(path);
        cv::resize(splash, splash, cv::Size(1920, 1080));
        mSplashes.push_back(splash);
    }
}

void Demo::initLayout(std::string path) {
    mLayoutSetting = loadLayoutSettingYAML(path);

    cv::Mat background(mDisplay.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    for (const auto& il : mLayoutSetting.image_layout) {
        il.img.copyTo(background(il.roi));
    }

    {
        unique_lock<mutex> lock(mDisplayMutex);
        background.copyTo(mDisplay);
        background.copyTo(mBackground);
    }

    // SizeState
    mSizeState.clear();
    mSizeState.resize(mLayoutSetting.worker_layout.size());
    for (int i = 0; i < mLayoutSetting.worker_layout.size(); i++) {
        mSizeState[i] = make_unique<SizeState>();
        mSizeState[i]->update(mLayoutSetting.worker_layout[i].roi.size());
    }
}

void Demo::initModels(std::string path) {
    mModelSetting = loadModelSettingYAML(path);

    mModels.clear();
    mAccs.clear();
    mModels.resize(mModelSetting.size());
    for (int i = 0; i < mModelSetting.size(); i++) {
        int dev_no = mModelSetting[i].dev_no;
        auto it = mAccs.find(dev_no);
        if (it == mAccs.end()) {
            StatusCode sc;
            mAccs.emplace(dev_no, Accelerator::create(dev_no, sc));
        }
        mModels[i] = std::make_unique<Model>(mModelSetting[i], *mAccs[dev_no]);
    }
}

void Demo::initFeeders(std::string path) {
    mFeederSetting = loadFeederSettingYAML(path);

    mFeeders.resize(mFeederSetting.size());
    for (int i = 0; i < mFeederSetting.size(); i++) {
        mFeeders[i] = make_unique<Feeder>(mFeederSetting[i]);
    }
}

void Demo::display() {
    unique_lock<mutex> lock(mDisplayMutex);
    if (mDisplayTimeMode) {
        displayTime(mDisplay, true, mBenchmarker.getTimeSinceCreated());
    }
    cv::imshow(WINDOW_NAME, mDisplay);
}

void Demo::toggleDisplayFPSMode() { mDisplayFPSMode = !mDisplayFPSMode; }

void Demo::toggleDisplayTimeMode() {
    mDisplayTimeMode = !mDisplayTimeMode;
    if (!mDisplayTimeMode) {
        unique_lock<mutex> lock(mDisplayMutex);
        cv::Rect overlay_rect = getTimeOverlayRect(mDisplay.size());
        mBackground(overlay_rect).copyTo(mDisplay(overlay_rect));
        sleepForMS(50);
    }
}

void Demo::toggleScreenSize() {
    double cur = cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN);
    bool is_fullscreen = (cur == cv::WINDOW_FULLSCREEN);

    if (is_fullscreen) {
        cv::Size window_size(1920, 1080);
        cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN, cv::WINDOW_NORMAL);
        cv::resizeWindow(WINDOW_NAME, window_size / 2);
    } else {
        cv::setWindowProperty(WINDOW_NAME, cv::WND_PROP_FULLSCREEN,
                              cv::WINDOW_FULLSCREEN);
    }
}

bool Demo::keyHandler(int key) {
    // Window X 버튼 클릭
    if (cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_AUTOSIZE) != 0) {
        stopWorkerAll();
        stopFeederAll();
        return false;
    }

    if (key == -1) {
        return true;
    }

    if (key >= 128) {  // Numpad 반환값은 128을 빼서 사용
        key -= 128;
    }

    key = tolower(key);

    if (key == 'd') {  // 'D'ebug
        toggleDisplayFPSMode();
    } else if (key == 't') {  // 'T'ime
        toggleDisplayTimeMode();
    } else if (key == 'm') {  // 'M'aximize Screen
        toggleScreenSize();
    } else if (key == 'c') {  // 'C'lear
        stopWorkerAll();
    } else if (key == 'f') {  // 'F'ill Grid
        startWorkerAll();
    } else if (key == 'q' || key == 27) {  // 'Q'uit, esc
        stopWorkerAll();
        stopFeederAll();
        return false;
    } else if (key == '1' || key == '2' || key == '3') {
        setMode(key - '0');
    }

    return true;
}

void Demo::setMode(int mode_index) {
    // clang-format off
    switch (mode_index) {
        case 1: setMode1(); break;
        case 2: setMode2(); break;
        case 3: setMode3(); break;
    }
    // clang-format on
}

void Demo::setMode1() {
    if (mModeIndex == 1) {
        return;
    }

    stopWorkerAll();
    {
        unique_lock<mutex> lock(mDisplayMutex);
        mSplashes[0].copyTo(mDisplay);
        cv::imshow(WINDOW_NAME, mDisplay);
        cv::waitKey(100);
    }
    initLayout("../rc/LayoutSetting.yaml");
    if (mModeIndex == -1 || mModeIndex == 3) {
        initModels("../rc/ModelSetting.yaml");
    }
    startWorkerAll();
    mModeIndex = 1;
    sleepForMS(500);
}

void Demo::setMode2() {
    if (mModeIndex == 2) {
        return;
    }

    stopWorkerAll();
    {
        unique_lock<mutex> lock(mDisplayMutex);
        mSplashes[0].copyTo(mDisplay);
        cv::imshow(WINDOW_NAME, mDisplay);
        cv::waitKey(100);
    }
    initLayout("../rc/LayoutSetting2.yaml");
    if (mModeIndex == -1 || mModeIndex == 3) {
        initModels("../rc/ModelSetting.yaml");
    }
    startWorkerAll();
    mModeIndex = 2;
    sleepForMS(500);
}

void Demo::setMode3() {
    if (mModeIndex == 3) {
        return;
    }

    stopWorkerAll();
    {
        unique_lock<mutex> lock(mDisplayMutex);
        mSplashes[1].copyTo(mDisplay);
        cv::imshow(WINDOW_NAME, mDisplay);
        cv::waitKey(100);
    }
    initLayout("../rc/LayoutSetting3.yaml");
    initModels("../rc/ModelSetting.yaml");
    startWorkerAll();
    mModeIndex = 3;
    sleepForMS(500);
}

void Demo::run() {
    initWindow();
    initLayout("../rc/LayoutSetting.yaml");
    initModels("../rc/ModelSetting.yaml");
    initFeeders("../rc/FeederSetting.yaml");

    startFeederAll();

    startThreads();
    startWorkerAll();
    toggleScreenSize();
    toggleDisplayFPSMode();

    while (true) {
        display();
        if (!keyHandler(cv::waitKey(10))) {  // 1일 경우, 600 fps 이상이 나온다.
            break;
        }
    }

    joinThreads();

    cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
    Demo demo;
    demo.run();
    return 0;
}
