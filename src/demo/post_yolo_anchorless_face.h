#ifndef DEMO_INCLUDE_POST_YOLO_ANCHORLESS_FACE_H_
#define DEMO_INCLUDE_POST_YOLO_ANCHORLESS_FACE_H_

#include "post_yolo_anchorless.h"

namespace mobilint::post {
class YOLOAnchorlessFacePost : public YOLOAnchorlessPost {
public:
    YOLOAnchorlessFacePost();
    YOLOAnchorlessFacePost(int imh, int imw, float conf_thres, float iou_thres,
                           bool verbose);
    ~YOLOAnchorlessFacePost() override = default;

    void plot_boxes(cv::Mat& im, std::vector<std::array<float, 4>>& boxes,
                    std::vector<float>& scores, std::vector<int>& labels) override;
};
}  // namespace mobilint::post

#endif
