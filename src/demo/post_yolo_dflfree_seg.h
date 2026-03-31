#ifndef DEMO_INCLUDE_POST_YOLO_DFLFREE_SEG_H_
#define DEMO_INCLUDE_POST_YOLO_DFLFREE_SEG_H_

#include "post_yolo_dflfree.h"

namespace mobilint::post {
class YOLODFLFreeSegPost : public YOLODFLFreePost {
public:
    YOLODFLFreeSegPost();
    YOLODFLFreeSegPost(int nc, int imh, int imw, float conf_thres, float iou_thres,
                       bool verbose);

    void run_postprocess(const std::vector<std::vector<float>>& npu_outs) override;
    cv::Mat& get_label_mask();
    cv::Mat& get_final_mask();
    void plot_boxes(cv::Mat& im, std::vector<std::array<float, 4>>& boxes,
                    std::vector<float>& scores, std::vector<int>& labels);
    void plot_masks(cv::Mat& im, cv::Mat& masks, cv::Mat& label_masks,
                    const std::vector<std::array<float, 4>>& boxes);
    void plot_extras(cv::Mat& im, std::vector<std::vector<float>>& extras) override;

private:
    void decode_outputs(const std::vector<float>& box_out,
                        const std::vector<float>& cls_out,
                        const std::vector<float>& mask_out,
                        std::vector<std::array<float, 4>>& pred_boxes,
                        std::vector<float>& pred_scores, std::vector<int>& pred_labels,
                        std::vector<std::vector<float>>& pred_extra);
    void nms(const std::vector<std::array<float, 4>>& pred_boxes,
             const std::vector<float>& pred_scores, const std::vector<int>& pred_labels,
             const std::vector<std::vector<float>>& pred_extra,
             std::vector<std::array<float, 4>>& final_boxes,
             std::vector<float>& final_scores, std::vector<int>& final_labels,
             std::vector<std::vector<float>>& final_extra);
    std::vector<std::array<float, 4>> downsample_boxes(
        std::vector<std::array<float, 4>> boxes);
    void process_mask(const std::vector<float>& proto,
                      const std::vector<std::vector<float>>& masks,
                      const std::vector<std::array<float, 4>>& boxes,
                      const std::vector<int>& labels);

    int m_nextra;
    int m_proto_stride;
    int m_proto_h;
    int m_proto_w;
    cv::Mat label_masks;
    cv::Mat final_masks;
    const int mOpenmpThreadCount = 2;
};
}  // namespace mobilint::post

#endif
