#ifndef DEMO_INCLUDE_DEMO_H_
#define DEMO_INCLUDE_DEMO_H_

#include <future>
#include <thread>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"
#include "qbruntime/qbruntime.h"

class Model;
class Feeder;

class Demo {
public:
    Demo();
    void run();

private:
    void startWorker(int index);
    void stopWorker(int index);
    void startFeeder(int index);
    void stopFeeder(int index);

    void startWorkerAll();
    void stopWorkerAll();
    void startFeederAll();
    void stopFeederAll();

    void startThreads();
    void joinThreads();

    int getWorkerIndex(int x, int y);
    static void onMouseEvent(int event, int x, int y, int flags, void* userdata);

    void feederReceive();
    void workerReceive();

    void initWindow();
    void initLayout(std::string path);
    void initModels(std::string path);
    void initFeeders(std::string path);
    void display();

    void toggleDisplayFPSMode();
    void toggleDisplayTimeMode();
    void toggleScreenSize();
    bool keyHandler(int key);

    void setMode(int mode_index);
    void setMode1();
    void setMode2();
    void setMode3();

    std::vector<FeederSetting> loadFeederSettingYAML(const std::string& path,
                                                     bool generate_default = false);
    std::vector<ModelSetting> loadModelSettingYAML(const std::string& path,
                                                   bool generate_default = false);
    LayoutSetting loadLayoutSettingYAML(const std::string& path,
                                        bool generate_default = false);

    const std::string WINDOW_NAME = "Mobilint Inference Demo";

    std::mutex mDisplayMutex;
    cv::Mat mDisplay;
    cv::Mat mBackground;
    Benchmarker mBenchmarker;

    bool mDisplayFPSMode;
    bool mDisplayTimeMode;

    std::vector<cv::Mat> mSplashes;
    int mModeIndex;

    std::vector<FeederSetting> mFeederSetting;  // FeederSetting.yaml에서 읽은 정보 저장
    std::vector<ModelSetting> mModelSetting;  // ModelSetting.yaml에서 읽은 정보 저장
    LayoutSetting mLayoutSetting;  // LayoutSetting.yaml에서 읽은 정보 저장

    std::map<int, std::unique_ptr<mobilint::Accelerator>> mAccs;

    // ModelSetting.yaml에 기재된 Model, default는 5개
    std::vector<std::unique_ptr<Model>> mModels;
    // FeederSetting.yaml에 기재된 Feeder, default는 6개
    std::vector<std::unique_ptr<Feeder>> mFeeders;
    // LayoutSetting.yaml에 기재된 worker_layout을 제어하기 위한 queue
    // default는 30개
    std::vector<std::unique_ptr<SizeState>> mSizeState;

    // 기존에는 worker thread를 들고 있었다.
    // thread pool을 도입하고 future를 들고있게 되는데,
    // future의 반환값이 유효하게 쓰이진 않고,
    // 의미상 thread join을 대신하는 정도로 사용한다.
    std::map<int, std::future<void>> mModelFutureMap;
    std::map<int, std::thread> mFeederThreadMap;

    ItemQueue mMainQueueForWorker;
    ItemQueue mMainQueueForFeeder;

    std::thread mFeederWatchdog;
    std::thread mWorkerWatchdog;

    ThreadPool mThreadPool;
};

#endif
