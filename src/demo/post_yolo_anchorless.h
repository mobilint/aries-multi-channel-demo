#ifndef DEMO_INCLUDE_POST_YOLO_ANCHORLESS_H_
#define DEMO_INCLUDE_POST_YOLO_ANCHORLESS_H_

#include <math.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "post_yolo_base.h"

namespace mobilint::post {
class YOLOAnchorlessPost : public YoloPostBase {
public:
    YOLOAnchorlessPost();
    YOLOAnchorlessPost(int nc, int imh, int imw, float conf_thres, float iou_thres,
                       bool verbose);
    ~YOLOAnchorlessPost() override = default;

public:
    std::vector<int> generate_strides(int nl);
    std::vector<std::vector<int>> generate_grids(int imh, int imw,
                                                 std::vector<int> strides);
    void run_postprocess(const std::vector<std::vector<float>>& npu_outs) override;

    int get_nl() const;
    int get_nc() const;
    float sigmoid(float num);
    std::vector<float> softmax(std::vector<float> vec);
    void xywh2xyxy(std::vector<std::array<float, 4>>& pred_boxes);
    virtual void decode_extra(const std::vector<float>& extra,
                              const std::vector<int>& grid, int stride, int idx,
                              std::vector<float>& pred_extra);
    void decode_boxes(const std::vector<float>& npu_out, const std::vector<int>& grid,
                      int stride, int idx, int det_stride,
                      std::array<float, 4>& pred_box);
    void decode_conf_thres(const std::vector<float>& npu_out,
                           const std::vector<int>& grid, int stride,
                           std::vector<std::array<float, 4>>& pred_boxes,
                           std::vector<float>& pred_conf, std::vector<int>& pred_label,
                           std::vector<std::pair<float, int>>& pred_scores,
                           const std::vector<float>& extra,
                           std::vector<std::vector<float>>& pred_extra,
                           const std::vector<float>* cls_out = nullptr);
    virtual void nms(std::vector<std::array<float, 4>> pred_boxes,
                     std::vector<float> pred_conf, std::vector<int> pred_label,
                     std::vector<std::pair<float, int>> scores,
                     std::vector<std::vector<float>> pred_extra,
                     std::vector<std::array<float, 4>>& final_boxes,
                     std::vector<float>& final_scores, std::vector<int>& final_labels,
                     std::vector<std::vector<float>>& final_extra);
    void plot_boxes(cv::Mat& im, std::vector<std::array<float, 4>>& boxes,
                    std::vector<float>& scores, std::vector<int>& labels) override;

protected:
    YOLOAnchorlessPost(int nc, int imh, int imw, float conf_thres, float iou_thres,
                       bool verbose, bool start_worker);
    int m_nextra;  // number of keypoints/landmarks/masks
    int m_nl;      // number of detection layers

    std::vector<int> m_strides;
    std::vector<std::vector<int>> m_grids;
};
}  // namespace mobilint::post

#endif
