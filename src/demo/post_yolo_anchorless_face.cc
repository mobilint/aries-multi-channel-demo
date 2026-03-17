#include "post_yolo_anchorless_face.h"

mobilint::post::YOLOAnchorlessFacePost::YOLOAnchorlessFacePost()
    : YOLOAnchorlessPost(1, 640, 640, 0.5f, 0.45f, false, false) {
    mType = PostType::FACE;
    start_worker_thread();
}

mobilint::post::YOLOAnchorlessFacePost::YOLOAnchorlessFacePost(int imh, int imw,
                                                               float conf_thres,
                                                               float iou_thres,
                                                               bool verbose)
    : YOLOAnchorlessPost(1, imh, imw, conf_thres, iou_thres, verbose, false) {
    mType = PostType::FACE;
    start_worker_thread();
}

void mobilint::post::YOLOAnchorlessFacePost::plot_boxes(
    cv::Mat& im, std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
    std::vector<int>& labels) {
    (void)scores;

    float ratio, xpad, ypad;
    compute_ratio_pads(im, ratio, xpad, ypad);

    cv::Rect rect;
    for (int i = 0; i < boxes.size(); i++) {
        if (labels[i] != 0) {
            continue;
        }
        int xmin = static_cast<int>(boxes[i][0] - xpad) / ratio;
        int ymin = static_cast<int>(boxes[i][1] - ypad) / ratio;
        int xmax = static_cast<int>(boxes[i][2] - xpad) / ratio;
        int ymax = static_cast<int>(boxes[i][3] - ypad) / ratio;

        xmin = std::max(xmin, 0);
        ymin = std::max(ymin, 0);
        xmax = std::min(xmax, im.cols);
        ymax = std::min(ymax, im.rows);

        rect.x = xmin;
        rect.y = ymin;
        rect.width = xmax - xmin;
        rect.height = ymax - ymin;

        cv::rectangle(im, rect, cv::Scalar(255, 255, 255), 1);
    }
}
