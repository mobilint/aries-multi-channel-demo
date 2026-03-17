#ifndef DEMO_INCLUDE_POST_H_
#define DEMO_INCLUDE_POST_H_

#include <array>
#include <cstdint>
#include <vector>

#include "opencv2/opencv.hpp"

class PostBase {
public:
    virtual ~PostBase() {}
    // SSD
    virtual uint64_t enqueue(std::vector<std::vector<float>>& result,
                             std::vector<float>& boxes, std::vector<float>& classes,
                             std::vector<float>& scores) {
        return 0;
    }

    // Yolov
    virtual uint64_t enqueue(cv::Mat& im, std::vector<std::vector<float>>& npu_outs,
                             std::vector<std::array<float, 4>>& boxes,
                             std::vector<float>& scores, std::vector<int>& labels,
                             std::vector<std::vector<float>>& extras) {
        return 0;
    }

    virtual void receive(uint64_t receipt_no) {}
};
#endif
