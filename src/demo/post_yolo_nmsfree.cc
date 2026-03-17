#include "post_yolo_nmsfree.h"

namespace mobilint::post {
YOLONMSFreePost::YOLONMSFreePost()
    : YOLOAnchorlessPost(80, 640, 640, 0.5f, 0.45f, false) {}

YOLONMSFreePost::YOLONMSFreePost(int nc, int imh, int imw, float conf_thres,
                                 float iou_thres, bool verbose)
    : YOLOAnchorlessPost(nc, imh, imw, conf_thres, iou_thres, verbose) {}

void YOLONMSFreePost::nms(std::vector<std::array<float, 4>> pred_boxes,
                          std::vector<float> pred_conf, std::vector<int> pred_label,
                          std::vector<std::pair<float, int>> scores,
                          std::vector<std::vector<float>> pred_extra,
                          std::vector<std::array<float, 4>>& final_boxes,
                          std::vector<float>& final_scores,
                          std::vector<int>& final_labels,
                          std::vector<std::vector<float>>& final_extra) {
    (void)scores;
    final_boxes = std::move(pred_boxes);
    final_scores = std::move(pred_conf);
    final_labels = std::move(pred_label);
    final_extra = std::move(pred_extra);
}
}  // namespace mobilint::post
