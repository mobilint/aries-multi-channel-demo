#ifndef DEMO_INCLUDE_POST_YOLO_NMSFREE_H_
#define DEMO_INCLUDE_POST_YOLO_NMSFREE_H_

#include "post_yolo_anchorless.h"

namespace mobilint::post {
class YOLONMSFreePost : public YOLOAnchorlessPost {
public:
    YOLONMSFreePost();
    YOLONMSFreePost(int nc, int imh, int imw, float conf_thres, float iou_thres,
                    bool verbose);

    void nms(std::vector<std::array<float, 4>> pred_boxes, std::vector<float> pred_conf,
             std::vector<int> pred_label, std::vector<std::pair<float, int>> scores,
             std::vector<std::vector<float>> pred_extra,
             std::vector<std::array<float, 4>>& final_boxes,
             std::vector<float>& final_scores, std::vector<int>& final_labels,
             std::vector<std::vector<float>>& final_extra) override;
};
}  // namespace mobilint::post

#endif
