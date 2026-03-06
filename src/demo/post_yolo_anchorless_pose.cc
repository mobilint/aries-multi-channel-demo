#include "post_yolo_anchorless_pose.h"

mobilint::post::YOLOAnchorlessPosePost::YOLOAnchorlessPosePost()
    : YOLOAnchorlessPost(80, 640, 640, 0.5f, 0.45f, false, false) {
    m_nc = 1;       // number of classes
    m_nextra = 51;  // number of extra outputs (keypoints => 17 * 3)
    mType = PostType::POSE;

    m_strides = generate_strides(m_nl);
    m_grids = generate_grids(m_imh, m_imw, m_strides);
    start_worker_thread();
}

mobilint::post::YOLOAnchorlessPosePost::YOLOAnchorlessPosePost(int nc, int imh, int imw,
                                                               float conf_thres,
                                                               float iou_thres,
                                                               bool verbose)
    : YOLOAnchorlessPost(nc, imh, imw, conf_thres, iou_thres, verbose, false) {
    m_nextra = 51;  // number of extra outputs (keypoints => 17 * 3)
    mType = PostType::POSE;
    start_worker_thread();
}

/*
        Access elements in output related to keypoints and decode them
*/
void mobilint::post::YOLOAnchorlessPosePost::decode_extra(
    const std::vector<float>& extra, const std::vector<int>& grid, int stride, int idx,
    std::vector<float>& pred_extra) {
    pred_extra.clear();
    int num_kpts = m_nextra / 3;  // 51 / 3
    for (int i = 0; i < num_kpts; i++) {
        auto first = extra[idx * m_nextra + 3 * i + 0];
        auto second = extra[idx * m_nextra + 3 * i + 1];
        auto third = extra[idx * m_nextra + 3 * i + 2];

        first = (first * 2 + grid[idx * 2 + 0]) * stride;
        second = (second * 2 + grid[idx * 2 + 1]) * stride;
        third = sigmoid(third);

        pred_extra.push_back(first);
        pred_extra.push_back(second);
        pred_extra.push_back(third);
    }
}

void mobilint::post::YOLOAnchorlessPosePost::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();
    auto summarize_sizes = [&npu_outs]() {
        std::string msg = "[";
        for (int i = 0; i < static_cast<int>(npu_outs.size()); i++) {
            if (i > 0) msg += ", ";
            msg += std::to_string(npu_outs[i].size());
        }
        msg += "]";
        return msg;
    };

    if (npu_outs.size() < static_cast<size_t>(m_nl * 2)) {
        throw std::invalid_argument(
            "YOLOAnchorless Pose post-processing is "
            "expected to receive at least 6 NPU outputs, however received " +
            std::to_string(npu_outs.size()));
    }

    struct LayerOutputMap {
        int det_idx = -1;
        int cls_idx = -1;
        int kpt_idx = -1;
    };

    std::vector<LayerOutputMap> maps(m_nl);
    std::vector<bool> used(npu_outs.size(), false);
    for (int i = 0; i < m_nl; i++) {
        const int grid_size = static_cast<int>(m_grids[i].size() / 2);
        const size_t det_combined_size = static_cast<size_t>(grid_size * (m_nc + 64));
        const size_t det_cls_size = static_cast<size_t>(grid_size * m_nc);
        const size_t det_box_size = static_cast<size_t>(grid_size * 64);
        const size_t kpt_size = static_cast<size_t>(grid_size * m_nextra);

        for (int j = 0; j < static_cast<int>(npu_outs.size()); j++) {
            if (!used[j] && npu_outs[j].size() == kpt_size) {
                maps[i].kpt_idx = j;
                used[j] = true;
                break;
            }
        }

        for (int j = 0; j < static_cast<int>(npu_outs.size()); j++) {
            if (!used[j] && npu_outs[j].size() == det_combined_size) {
                maps[i].det_idx = j;
                used[j] = true;
                break;
            }
        }

        if (maps[i].det_idx < 0) {
            int cls_idx = -1;
            int box_idx = -1;
            for (int j = 0; j < static_cast<int>(npu_outs.size()); j++) {
                if (!used[j] && npu_outs[j].size() == det_cls_size) {
                    cls_idx = j;
                    break;
                }
            }
            for (int j = 0; j < static_cast<int>(npu_outs.size()); j++) {
                if (!used[j] && npu_outs[j].size() == det_box_size) {
                    box_idx = j;
                    break;
                }
            }
            if (cls_idx >= 0 && box_idx >= 0) {
                maps[i].cls_idx = cls_idx;
                maps[i].det_idx = box_idx;
                used[cls_idx] = true;
                used[box_idx] = true;
            }
        }

        if (maps[i].det_idx < 0 || maps[i].kpt_idx < 0) {
            throw std::invalid_argument(
                "Unable to infer output layout in YOLOAnchorless pose post-processing. "
                "Output sizes=" +
                summarize_sizes());
        }
    }
    for (int i = 0; i < static_cast<int>(used.size()); i++) {
        if (!used[i]) {
            throw std::invalid_argument(
                "Unexpected extra output detected in YOLOAnchorless pose "
                "post-processing.");
        }
    }

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<float> pred_conf;
    std::vector<int> pred_label;
    std::vector<std::pair<float, int>> pred_scores;
    std::vector<std::vector<float>> pred_extra;

    for (int i = 0; i < m_nl; i++) {
        const std::vector<float>* cls_out = nullptr;
        if (maps[i].cls_idx >= 0) {
            cls_out = &npu_outs[maps[i].cls_idx];
        }

        decode_conf_thres(npu_outs[maps[i].det_idx], m_grids[i], m_strides[i], pred_boxes,
                          pred_conf, pred_label, pred_scores, npu_outs[maps[i].kpt_idx],
                          pred_extra, cls_out);
    }

    xywh2xyxy(pred_boxes);

    nms(pred_boxes, pred_conf, pred_label, pred_scores, pred_extra, final_boxes,
        final_scores, final_labels, final_extra);

    double endd = set_timer();
    if (m_verbose) std::cout << "Real C++ Time        : " << endd - start << std::endl;
}

/*
        Draw human keypoints
*/
void mobilint::post::YOLOAnchorlessPosePost::plot_keypoints(
    cv::Mat& im, std::vector<std::vector<float>>& kpts) {
    int radius = 2;               // circle size (reduced)
    int steps = 3;                // (x, y, conf) * 17
    int num_kpts = m_nextra / 3;  // 51 / 3
    float kpts_conf_thres = 0.4;  // Do not draw low confidence skeleton

    float ratio, xpad, ypad;
    compute_ratio_pads(im, ratio, xpad, ypad);

    for (int i = 0; i < kpts.size(); i++) {
        for (int j = 0; j < num_kpts; j++) {
            kpts[i][3 * j + 0] = (kpts[i][3 * j + 0] - xpad) / ratio;
            kpts[i][3 * j + 1] = (kpts[i][3 * j + 1] - ypad) / ratio;
        }
    }

    for (const auto& kpt_t : kpts) {
        for (int j = 0; j < num_kpts; j++) {
            auto bgr = m_pose_kpt_color[j];
            cv::Scalar color(bgr[0], bgr[1], bgr[2]);
            int kpt_idx = steps * j;
            int x_coord = (int)kpt_t[kpt_idx];
            int y_coord = (int)kpt_t[kpt_idx + 1];
            float conf = kpt_t[kpt_idx + 2];

            if (conf < kpts_conf_thres) {
                continue;
            }

            if (x_coord % m_imw != 0 && y_coord % m_imh != 0) {
                cv::Point p(x_coord, y_coord);
                cv::circle(im, p, radius, color, -1);
            }
        }

        for (int j = 0; j < m_skeleton.size(); j++) {
            if (j == 17 || j == 18) {
                continue;
            }
            auto bgr = m_pose_limb_color[j];
            cv::Scalar color(bgr[0], bgr[1], bgr[2]);
            const auto& sk = m_skeleton[j];

            float conf1 = kpt_t[(sk[0] - 1) * steps + 2];
            float conf2 = kpt_t[(sk[1] - 1) * steps + 2];
            if (conf1 < 0.5 || conf2 < 0.5) {
                continue;
            }

            cv::Point p1((int)kpt_t[(sk[0] - 1) * steps],
                         (int)kpt_t[(sk[0] - 1) * steps + 1]);
            cv::Point p2((int)kpt_t[(sk[1] - 1) * steps],
                         (int)kpt_t[(sk[1] - 1) * steps + 1]);

            if (p1.x % m_imw == 0 || p1.y % m_imh == 0 || p1.x < 0 || p1.y < 0) {
                continue;
            }

            if (p2.x % m_imw == 0 || p2.y % m_imh == 0 || p2.x < 0 || p2.y < 0) {
                continue;
            }
            cv::line(im, p1, p2, color, 1);
        }
    }
}

/*
        Plot extras, in this case plot keypoints
*/
void mobilint::post::YOLOAnchorlessPosePost::plot_extras(
    cv::Mat& im, std::vector<std::vector<float>>& extras) {
    plot_keypoints(im, extras);
}
