#ifndef DEMO_INCLUDE_POST_YOLO_BASE_H_
#define DEMO_INCLUDE_POST_YOLO_BASE_H_

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "post.h"

namespace mobilint::post {
enum class PostType { OBJECT, FACE, POSE, SEG };

struct YoloPostItem {
    uint64_t id;
    cv::Mat& im;
    std::vector<std::vector<float>>& npu_outs;
    std::vector<std::array<float, 4>>& boxes;
    std::vector<float>& scores;
    std::vector<int>& labels;
    std::vector<std::vector<float>>& extras;
};

class YoloPostBase : public PostBase {
public:
    YoloPostBase(int nc, int imh, int imw, float conf_thres, float iou_thres,
                 bool verbose, PostType type, bool start_worker = true);
    ~YoloPostBase() override;

    uint64_t enqueue(cv::Mat& im, std::vector<std::vector<float>>& npu_outs,
                     std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
                     std::vector<int>& labels,
                     std::vector<std::vector<float>>& extras) override;
    void receive(uint64_t receipt_no) override;
    PostType getType() const;

protected:
    void start_worker_thread();
    float area(float xmin, float ymin, float xmax, float ymax) const;
    float get_iou(std::array<float, 4> box1, std::array<float, 4> box2) const;
    double set_timer() const;
    std::vector<std::array<float, 4>>& get_result_box();
    std::vector<float>& get_result_score();
    std::vector<int>& get_result_label();
    std::vector<std::vector<float>>& get_result_extra();
    void compute_ratio_pads(const cv::Mat& im, float& ratio, float& xpad,
                            float& ypad) const;
    void print(const std::string& msg) const;
    virtual void worker();
    virtual void run_postprocess(const std::vector<std::vector<float>>& npu_outs) = 0;
    virtual void plot_boxes(cv::Mat& im, std::vector<std::array<float, 4>>& boxes,
                            std::vector<float>& scores, std::vector<int>& labels) = 0;
    virtual void plot_extras(cv::Mat& im, std::vector<std::vector<float>>& extras);

protected:
    const int mOpenmpThreadCount = 2;
    int m_nc;
    uint32_t m_imh;
    uint32_t m_imw;
    float m_conf_thres;
    float m_iou_thres;
    int m_max_det_num;
    bool m_verbose;
    PostType mType;

    std::vector<std::array<float, 4>> final_boxes;
    std::vector<float> final_scores;
    std::vector<int> final_labels;
    std::vector<std::vector<float>> final_extra;

    std::thread mThread;
    std::queue<YoloPostItem> mQueueIn;
    std::vector<uint64_t> mOut;
    uint64_t ticket;

    mutable std::mutex mPrintMutex;
    std::mutex mMutexIn;
    std::mutex mMutexOut;
    std::condition_variable mCondIn;
    std::condition_variable mCondOut;
    bool destroyed;
    bool mWorkerStarted;

    const std::vector<std::string> COCO_LABELS = {
        "person",        "bicycle",      "car",
        "motorcycle",    "airplane",     "bus",
        "train",         "truck",        "boat",
        "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench",        "bird",
        "cat",           "dog",          "horse",
        "sheep",         "cow",          "elephant",
        "bear",          "zebra",        "giraffe",
        "backpack",      "umbrella",     "handbag",
        "tie",           "suitcase",     "frisbee",
        "skis",          "snowboard",    "sports ball",
        "kite",          "baseball bat", "baseball glove",
        "skateboard",    "surfboard",    "tennis racket",
        "bottle",        "wine glass",   "cup",
        "fork",          "knife",        "spoon",
        "bowl",          "banana",       "apple",
        "sandwich",      "orange",       "broccoli",
        "carrot",        "hot dog",      "pizza",
        "donut",         "cake",         "chair",
        "couch",         "potted plant", "bed",
        "dining table",  "toilet",       "tv",
        "laptop",        "mouse",        "remote",
        "keyboard",      "cell phone",   "microwave",
        "oven",          "toaster",      "sink",
        "refrigerator",  "book",         "clock",
        "vase",          "scissors",     "teddy bear",
        "hair drier",    "toothbrush",
    };

    const std::vector<std::array<int, 3>> COLORS = {
        {56, 56, 255},  {151, 157, 255}, {31, 112, 255}, {29, 178, 255},  {49, 210, 207},
        {10, 249, 72},  {23, 204, 146},  {134, 219, 61}, {52, 147, 26},   {187, 212, 0},
        {168, 153, 44}, {255, 194, 0},   {147, 69, 52},  {255, 115, 100}, {236, 24, 0},
        {255, 56, 132}, {133, 0, 82},    {255, 56, 203}, {200, 149, 255}, {199, 55, 255}};
};
}  // namespace mobilint::post

#endif
