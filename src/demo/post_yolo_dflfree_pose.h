#ifndef DEMO_INCLUDE_POST_YOLO_DFLFREE_POSE_H_
#define DEMO_INCLUDE_POST_YOLO_DFLFREE_POSE_H_

#include "post_yolo_dflfree.h"

namespace mobilint::post {
class YOLODFLFreePosePost : public YOLODFLFreePost {
public:
    YOLODFLFreePosePost();
    YOLODFLFreePosePost(int nc, int imh, int imw, float conf_thres, float iou_thres,
                        bool verbose);

    void run_postprocess(const std::vector<std::vector<float>>& npu_outs) override;
    void plot_boxes(cv::Mat& im, std::vector<std::array<float, 4>>& boxes,
                    std::vector<float>& scores, std::vector<int>& labels);
    void plot_keypoints(cv::Mat& im, std::vector<std::vector<float>>& kpts);
    void plot_extras(cv::Mat& im, std::vector<std::vector<float>>& extras) override;

private:
    void decode_outputs(const std::vector<float>& box_out,
                        const std::vector<float>& cls_out,
                        const std::vector<float>& kpt_out,
                        std::vector<std::array<float, 4>>& pred_boxes,
                        std::vector<float>& pred_scores, std::vector<int>& pred_labels,
                        std::vector<std::vector<float>>& pred_extra);
    void nms(const std::vector<std::array<float, 4>>& pred_boxes,
             const std::vector<float>& pred_scores, const std::vector<int>& pred_labels,
             const std::vector<std::vector<float>>& pred_extra,
             std::vector<std::array<float, 4>>& final_boxes,
             std::vector<float>& final_scores, std::vector<int>& final_labels,
             std::vector<std::vector<float>>& final_extra);

    int m_nextra;

    const std::vector<std::array<int, 2>> m_skeleton = {
        {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
        {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
        {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

    const std::vector<std::array<int, 3>> m_pose_limb_color = {
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255},
        {255, 51, 255}, {255, 51, 255}, {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
        {255, 128, 0},  {255, 128, 0},  {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}};

    const std::vector<std::array<int, 3>> m_pose_kpt_color = {
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
        {255, 128, 0},  {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
        {51, 153, 255}, {51, 153, 255}};
};
}  // namespace mobilint::post

#endif
