#ifndef DEMO_INCLUDE_POST_YOLO_ANCHORLESS_SEG_H_
#define DEMO_INCLUDE_POST_YOLO_ANCHORLESS_SEG_H_

#include "post_yolo_anchorless.h"

namespace mobilint::post {
class YOLOAnchorlessSegPost : public YOLOAnchorlessPost {
public:
    YOLOAnchorlessSegPost();
    YOLOAnchorlessSegPost(int nc, int imh, int imw, float conf_thres, float iou_thres,
                          bool verbose);

    void run_postprocess(const std::vector<std::vector<float>>& npu_outs) override;
    void decode_extra(const std::vector<float>& extra, const std::vector<int>& grid,
                      int stride, int idx, std::vector<float>& pred_extra) override;
    std::vector<std::array<float, 4>> downsample_boxes(
        std::vector<std::array<float, 4>> boxes);
    void process_mask(const std::vector<float>& proto,
                      const std::vector<std::vector<float>>& masks,
                      const std::vector<std::array<float, 4>>& boxes,
                      const std::vector<int>& labels);
    cv::Mat& get_label_mask();
    cv::Mat& get_final_mask();
    void plot_masks(cv::Mat& im, cv::Mat& masks, cv::Mat& label_masks,
                    const std::vector<std::array<float, 4>>& boxes);

    void worker() override;

protected:
    int m_proto_stride;
    int m_proto_h;
    int m_proto_w;
    cv::Mat label_masks;
    cv::Mat final_masks;
};
}  // namespace mobilint::post

#endif
