#ifndef DEMO_INCLUDE_MODEL_POST_SSD_H_
#define DEMO_INCLUDE_MODEL_POST_SSD_H_

#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "demo/post.h"

struct SSDPostItem {
    uint64_t id;
    std::vector<std::vector<float>>& result;
    std::vector<float>& boxes;
    std::vector<float>& classes;
    std::vector<float>& scores;
};

/**
 * Central Postprocessor for SSD-MobileNet.
 */
class SSDPostProcessor : public PostBase {
private:
    const int mOpenmpThreadCount = 2;

    std::thread mThread;
    uint8_t mPPType = 0;

    std::queue<SSDPostItem> mQueueIn;
    std::vector<uint64_t> mOut;
    uint64_t ticket = 0;

    std::mutex mMutexIn;
    std::mutex mMutexOut;
    std::condition_variable mCondIn;
    std::condition_variable mCondOut;

    bool destroyed = false;
    const int ch_depth_size = 64;

    float* prior_generation();
    float* priors_cpp = prior_generation();

    void worker();
    void reshape(int loop, int total_height, int total_channel, int8_t* dest,
                 int8_t* src);
    float area(float xmin, float ymin, float xmax, float ymax);
    float calculate_iou(std::array<float, 4> box1, std::array<float, 4> box2);
    float* decode(float* locations, float* priors);
    int filter_results(float* init_scores, float* init_boxes,
                       std::vector<float>& boxes_final, std::vector<float>& classes_final,
                       std::vector<float>& scores_final, float nms_threshold);
    void transpose_and_copy(float* boxes_float, float* clses_float,
                            std::vector<std::vector<float>>& result);
    int postprocessing(std::vector<std::vector<float>>& result, std::vector<float>& boxes,
                       std::vector<float>& classes, std::vector<float>& scores);

public:
    const int PP_SSD_MOBILENET = 1;
    const int PP_SSD_RESNET = 2;

    SSDPostProcessor();
    ~SSDPostProcessor();

    uint64_t enqueue(std::vector<std::vector<float>>& result, std::vector<float>& boxes,
                     std::vector<float>& classes, std::vector<float>& scores);

    void receive(uint64_t receipt_no);
};

#endif
