#include "demo/post_yolo_anchor_face.h"

YOLOAnchorFacePost::YOLOAnchorFacePost(int nl, int nc, int no, int imh, int imw,
                                       float conf_thres, float iou_thres, bool verbose)
    : YOLOAnchorPost(nl, nc, no, imh, imw, conf_thres, iou_thres, verbose) {
    mType = mobilint::post::PostType::FACE;
    if (m_nl == 3) {
        // face detection P5 anchors
        m_anchors = {
            {{4, 5}, {8, 10}, {13, 16}},          // P3/8
            {{23, 29}, {43, 55}, {73, 105}},      // P4/16
            {{146, 217}, {231, 300}, {335, 433}}  // P5/32
        };
    } else if (m_nl == 4) {
        // face detection P6 anchors
        m_anchors = {
            {{6, 7}, {9, 11}, {13, 16}},          // P3/8
            {{18, 23}, {26, 33}, {37, 47}},       // P4/16
            {{54, 67}, {77, 104}, {112, 154}},    // P5/32
            {{174, 238}, {258, 355}, {445, 568}}  // P6/64
        };
    } else {
        throw std::invalid_argument("Number of detection layers should 3 or 4");
    }
}

/*
        Get extra data out of model outputs. Simple object detection does not
        have any extra data. Face Detection has landmarks, Segmentatin has
        masks and Pose Estimation has keypoints

*/
std::vector<float> YOLOAnchorFacePost::get_extra(const std::vector<float>& output,
                                                 const std::vector<int>& grid,
                                                 const std::vector<int>& anchor,
                                                 int stride, int idx, int grid_idx,
                                                 float conf_value) {
    /* decode face landmarks */
    std::vector<float> landmarks;
    for (int i = 0; i < m_nextra; i++) {
        int j = i % 2;
        landmarks.push_back(output[idx + 5 + i] * anchor[j] +
                            grid[grid_idx + j] * stride);
    }

    return landmarks;
}

/*
        Get the offset for the class values. In Face Detection class values are
        located after landmarks.
*/
int YOLOAnchorFacePost::get_cls_offset() { return m_nextra; }

/*
        Mark landmarks(eyes, nose, mouth) on detected face
*/
void YOLOAnchorFacePost::plot_landmarks(cv::Mat& im,
                                        std::vector<std::vector<float>>& landmarks) {
    float ratio, xpad, ypad;
    compute_ratio_pads(im, ratio, xpad, ypad);

    for (int i = 0; i < landmarks.size(); i++) {
        cv::Point pt;
        for (int j = 0; j < m_nextra / 2; j++) {
            pt.x = (int)(landmarks[i][2 * j + 0] - xpad) / ratio;
            pt.y = (int)(landmarks[i][2 * j + 1] - ypad) / ratio;

            std::array<int, 3> bgr = LMARK_COLORS[j];
            cv::Scalar clr(bgr[0], bgr[1], bgr[2]);
            cv::circle(im, pt, 3, clr, -1);
        }
    }
}

/*
        Plot extras, in this case plot landmarks
*/
void YOLOAnchorFacePost::plot_extras(cv::Mat& im,
                                     std::vector<std::vector<float>>& extras) {
    plot_landmarks(im, extras);
}
