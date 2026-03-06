#ifndef DEMO_INCLUDE_POST_YOLO_ANCHOR_H_
#define DEMO_INCLUDE_POST_YOLO_ANCHOR_H_

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

class YOLOAnchorPost : public mobilint::post::YoloPostBase {
public:
    YOLOAnchorPost(int nl, int nc, int no, int imh, int imw, float conf_thres,
                   float iou_thres, bool verbose);
    ~YOLOAnchorPost() override = default;

public:
    void generate_grids(int imh, int imw, std::vector<int> strides);
    void run_postprocess(const std::vector<std::vector<float>>& npu_outs) override;

    float sigmoid(float num);
    float inverse_sigmoid(float num);
    virtual std::vector<float> get_extra(const std::vector<float>& output,
                                         const std::vector<int>& grid,
                                         const std::vector<int>& anchor, int stride,
                                         int idx, int grid_idx, float conf_value = 0);
    virtual int get_cls_offset();
    void xywh2xyxy(std::vector<std::array<float, 4>>& pred_boxes);
    void decode_conf_thres(const std::vector<float>& npu_out,
                           const std::vector<int>& grid,
                           const std::vector<std::vector<int>>& anchor, int stride,
                           std::vector<std::array<float, 4>>& pred_boxes,
                           std::vector<float>& pred_conf, std::vector<int>& pred_label,
                           std::vector<std::pair<float, int>>& pred_scores,
                           std::vector<std::vector<float>>& pred_extra);
    void nms(const std::vector<std::array<float, 4>>& pred_boxes,
             const std::vector<float>& pred_conf, const std::vector<int>& pred_label,
             std::vector<std::pair<float, int>>& scores,
             const std::vector<std::vector<float>>& pred_extra,
             std::vector<std::array<float, 4>>& final_boxes,
             std::vector<float>& final_scores, std::vector<int>& final_labels,
             std::vector<std::vector<float>>& final_extra);
    void plot_boxes(cv::Mat& im, std::vector<std::array<float, 4>>& boxes,
                    std::vector<float>& scores, std::vector<int>& labels) override;
    virtual void plot_extras(cv::Mat& im,
                             std::vector<std::vector<float>>& extras) override;

protected:
    int m_no;      // number outputs per anchor (5 + nc + keypoints/landmarks/masks)
    int m_nextra;  // number of keypoints/landmarks/masks
    int m_nl;      // number of detection layers
    int m_na;      // number of anchors
    float m_inverse_conf_thres;
    bool m_only_person = false;

    std::vector<std::vector<int>> m_grids;
    std::vector<std::vector<std::vector<int>>> m_anchors;
    std::vector<int> m_strides;
};

#endif
