#ifndef DEMO_INCLUDE_POST_YOLO_DFLFREE_H_
#define DEMO_INCLUDE_POST_YOLO_DFLFREE_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "post_yolo_base.h"

namespace mobilint::post {
class YOLODFLFreePost : public YoloPostBase {
public:
    YOLODFLFreePost();
    YOLODFLFreePost(int nc, int imh, int imw, float conf_thres, float iou_thres,
                    bool verbose);
    ~YOLODFLFreePost() override = default;

public:
    virtual void run_postprocess(
        const std::vector<std::vector<float>>& npu_outs) override;

protected:
    YOLODFLFreePost(int nc, int imh, int imw, float conf_thres, float iou_thres,
                    bool verbose, bool start_worker);
    float sigmoid(float x) const;
    void decode_outputs(const std::vector<float>& box_out,
                        const std::vector<float>& cls_out,
                        std::vector<std::array<float, 4>>& pred_boxes,
                        std::vector<float>& pred_scores, std::vector<int>& pred_labels);
    void nms(const std::vector<std::array<float, 4>>& pred_boxes,
             const std::vector<float>& pred_scores, const std::vector<int>& pred_labels,
             std::vector<std::array<float, 4>>& final_boxes,
             std::vector<float>& final_scores, std::vector<int>& final_labels);
    virtual void plot_boxes(cv::Mat& im, std::vector<std::array<float, 4>>& boxes,
                            std::vector<float>& scores,
                            std::vector<int>& labels) override;
    virtual void plot_extras(cv::Mat& im,
                             std::vector<std::vector<float>>& extras) override;

protected:
    // extras for pose/seg
};
}  // namespace mobilint::post

#endif
