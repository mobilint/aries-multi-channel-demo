#include "post_yolo_dflfree_pose.h"

#include <numeric>
#include <sstream>

namespace {
bool is_prob_score(float value) { return value >= 0.0f && value <= 1.0f; }

std::string summarize_sizes(const std::vector<std::vector<float>>& outs) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < static_cast<int>(outs.size()); i++) {
        if (i > 0) oss << ", ";
        oss << outs[i].size();
    }
    oss << "]";
    return oss.str();
}

bool infer_grid_and_stride(int imh, int imw, int num_cells, int& grid_h, int& grid_w,
                           int& stride) {
    for (int s = 1; s <= std::min(imh, imw); s++) {
        if (imh % s != 0 || imw % s != 0) continue;
        const int gh = imh / s;
        const int gw = imw / s;
        if (gh * gw == num_cells) {
            grid_h = gh;
            grid_w = gw;
            stride = s;
            return true;
        }
    }
    return false;
}
}  // namespace

mobilint::post::YOLODFLFreePosePost::YOLODFLFreePosePost()
    : YOLODFLFreePost(1, 640, 640, 0.25f, 0.65f, false, false) {
    m_nextra = 51;  // 17 keypoints * (x, y, conf)
    mType = PostType::POSE;
    start_worker_thread();
}

mobilint::post::YOLODFLFreePosePost::YOLODFLFreePosePost(int nc, int imh, int imw,
                                                         float conf_thres,
                                                         float iou_thres, bool verbose)
    : YOLODFLFreePost(nc, imh, imw, conf_thres, iou_thres, verbose, false) {
    m_nextra = 51;  // 17 keypoints * (x, y, conf)
    mType = PostType::POSE;
    start_worker_thread();
}

void mobilint::post::YOLODFLFreePosePost::decode_outputs(
    const std::vector<float>& box_out, const std::vector<float>& cls_out,
    const std::vector<float>& kpt_out, std::vector<std::array<float, 4>>& pred_boxes,
    std::vector<float>& pred_scores, std::vector<int>& pred_labels,
    std::vector<std::vector<float>>& pred_extra) {
    constexpr int kBoxElements = 4;
    constexpr int kKptDims = 3;  // (x, y, conf)
    const int num_detections = static_cast<int>(box_out.size() / kBoxElements);
    const int num_kpts = m_nextra / kKptDims;

    if (box_out.size() != static_cast<size_t>(num_detections * kBoxElements)) {
        throw std::invalid_argument(
            "Invalid box output shape in YOLODFLFree pose output.");
    }

    if (cls_out.size() != static_cast<size_t>(num_detections * m_nc)) {
        throw std::invalid_argument(
            "Invalid class output shape in YOLODFLFree pose output.");
    }

    if (kpt_out.size() != static_cast<size_t>(num_kpts * num_detections * kKptDims)) {
        throw std::invalid_argument(
            "Invalid keypoint output shape in YOLODFLFree pose output.");
    }

    pred_boxes.clear();
    pred_scores.clear();
    pred_labels.clear();
    pred_extra.clear();

    pred_boxes.reserve(num_detections);
    pred_scores.reserve(num_detections);
    pred_labels.reserve(num_detections);
    pred_extra.reserve(num_detections);

    for (int i = 0; i < num_detections; i++) {
        const int box_idx = i * kBoxElements;
        const int cls_idx = i * m_nc;
        float best_score = -1.0f;
        int best_label = -1;
        for (int c = 0; c < m_nc; c++) {
            float score = cls_out[cls_idx + c];
            if (!is_prob_score(score)) {
                score = sigmoid(score);
            }

            if (score > best_score) {
                best_score = score;
                best_label = c;
            }
        }

        if (best_score < m_conf_thres) {
            continue;
        }

        std::array<float, 4> box = {
            box_out[box_idx + 0],
            box_out[box_idx + 1],
            box_out[box_idx + 2],
            box_out[box_idx + 3],
        };

        if (box[2] <= box[0] || box[3] <= box[1]) {
            const float cx = box[0];
            const float cy = box[1];
            const float w = box[2];
            const float h = box[3];
            box[0] = cx - w * 0.5f;
            box[1] = cy - h * 0.5f;
            box[2] = cx + w * 0.5f;
            box[3] = cy + h * 0.5f;
        }

        std::vector<float> keypoints(m_nextra);
        // kpt_out is laid out as [num_kpts, num_detections, 3].
        for (int k = 0; k < num_kpts; k++) {
            const int src_base = (k * num_detections + i) * kKptDims;
            const int dst_base = k * kKptDims;
            keypoints[dst_base + 0] = kpt_out[src_base + 0];
            keypoints[dst_base + 1] = kpt_out[src_base + 1];
            keypoints[dst_base + 2] = kpt_out[src_base + 2];
        }
        for (int j = 2; j < m_nextra; j += 3) {
            if (!is_prob_score(keypoints[j])) {
                keypoints[j] = sigmoid(keypoints[j]);
            }
        }

        pred_boxes.push_back(box);
        pred_scores.push_back(best_score);
        pred_labels.push_back(best_label);
        pred_extra.push_back(std::move(keypoints));
    }
}

void mobilint::post::YOLODFLFreePosePost::nms(
    const std::vector<std::array<float, 4>>& pred_boxes,
    const std::vector<float>& pred_scores, const std::vector<int>& pred_labels,
    const std::vector<std::vector<float>>& pred_extra,
    std::vector<std::array<float, 4>>& final_boxes, std::vector<float>& final_scores,
    std::vector<int>& final_labels, std::vector<std::vector<float>>& final_extra) {
    if (pred_boxes.empty()) {
        return;
    }

    std::vector<int> order(pred_boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int lhs, int rhs) { return pred_scores[lhs] > pred_scores[rhs]; });

    std::vector<bool> suppressed(pred_boxes.size(), false);

    for (size_t i = 0; i < order.size(); i++) {
        const int idx = order[i];
        if (suppressed[idx]) {
            continue;
        }

        final_boxes.push_back(pred_boxes[idx]);
        final_scores.push_back(pred_scores[idx]);
        final_labels.push_back(pred_labels[idx]);
        final_extra.push_back(pred_extra[idx]);

        if (final_boxes.size() >= static_cast<size_t>(m_max_det_num)) {
            break;
        }

        for (size_t j = i + 1; j < order.size(); j++) {
            const int other_idx = order[j];
            if (suppressed[other_idx]) {
                continue;
            }
            if (pred_labels[idx] != pred_labels[other_idx]) {
                continue;
            }
            if (get_iou(pred_boxes[idx], pred_boxes[other_idx]) > m_iou_thres) {
                suppressed[other_idx] = true;
            }
        }
    }
}

void mobilint::post::YOLODFLFreePosePost::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    const double start = set_timer();

    if (npu_outs.size() < 3) {
        throw std::invalid_argument(
            "YOLODFLFree Pose post-processing expects at least 3 outputs, but received " +
            std::to_string(npu_outs.size()));
    }

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<float> pred_scores;
    std::vector<int> pred_labels;
    std::vector<std::vector<float>> pred_extra;

    if (npu_outs.size() == 3) {
        int box_idx = -1;
        int cls_idx = -1;
        int kpt_idx = -1;
        for (int i = 0; i < 3; i++) {
            if (npu_outs[i].size() % 4 != 0) continue;
            const int num_det = static_cast<int>(npu_outs[i].size() / 4);
            for (int j = 0; j < 3; j++) {
                if (j == i) continue;
                if (npu_outs[j].size() != static_cast<size_t>(num_det * m_nc)) continue;
                const int k = 3 - i - j;
                if (npu_outs[k].size() == static_cast<size_t>(num_det * m_nextra)) {
                    box_idx = i;
                    cls_idx = j;
                    kpt_idx = k;
                    break;
                }
            }
            if (box_idx >= 0) break;
        }
        if (box_idx < 0 || cls_idx < 0 || kpt_idx < 0) {
            throw std::invalid_argument(
                "Unable to infer decoded output layout for YOLODFLFree pose "
                "post-processing. "
                "Output sizes=" +
                summarize_sizes(npu_outs));
        }
        decode_outputs(npu_outs[box_idx], npu_outs[cls_idx], npu_outs[kpt_idx],
                       pred_boxes, pred_scores, pred_labels, pred_extra);
    } else {
        // Non-decoded multi-head outputs: [box4_h, cls_h, kpt_h] * nl (order-agnostic).
        struct LayerHead {
            int box_idx = -1;
            int cls_idx = -1;
            int kpt_idx = -1;
            int num_cells = 0;
            int grid_h = 0;
            int grid_w = 0;
            int stride = 0;
        };
        std::vector<LayerHead> layers;
        std::vector<bool> used(npu_outs.size(), false);

        for (int i = 0; i < static_cast<int>(npu_outs.size()); i++) {
            if (used[i] || npu_outs[i].size() % 4 != 0) continue;
            const int num_cells = static_cast<int>(npu_outs[i].size() / 4);
            int cls_idx = -1;
            int kpt_idx = -1;
            for (int j = 0; j < static_cast<int>(npu_outs.size()); j++) {
                if (used[j] || j == i) continue;
                if (npu_outs[j].size() == static_cast<size_t>(num_cells * m_nc)) {
                    cls_idx = j;
                    break;
                }
            }
            for (int j = 0; j < static_cast<int>(npu_outs.size()); j++) {
                if (used[j] || j == i || j == cls_idx) continue;
                if (npu_outs[j].size() == static_cast<size_t>(num_cells * m_nextra)) {
                    kpt_idx = j;
                    break;
                }
            }
            if (cls_idx < 0 || kpt_idx < 0) continue;

            LayerHead layer;
            layer.box_idx = i;
            layer.cls_idx = cls_idx;
            layer.kpt_idx = kpt_idx;
            layer.num_cells = num_cells;
            if (!infer_grid_and_stride(m_imh, m_imw, num_cells, layer.grid_h,
                                       layer.grid_w, layer.stride)) {
                throw std::invalid_argument(
                    "Failed to infer grid/stride for DFLFree pose head. num_cells=" +
                    std::to_string(num_cells));
            }
            used[i] = true;
            used[cls_idx] = true;
            used[kpt_idx] = true;
            layers.push_back(layer);
        }

        for (int i = 0; i < static_cast<int>(used.size()); i++) {
            if (!used[i]) {
                throw std::invalid_argument(
                    "Unable to infer output layout for YOLODFLFree pose post-processing. "
                    "Output sizes=" +
                    summarize_sizes(npu_outs));
            }
        }
        if (layers.empty()) {
            throw std::invalid_argument(
                "No valid detection heads found for YOLODFLFree pose post-processing. "
                "Output sizes=" +
                summarize_sizes(npu_outs));
        }

        const int num_kpts = m_nextra / 3;
        for (const auto& layer : layers) {
            const auto& box_out = npu_outs[layer.box_idx];
            const auto& cls_out = npu_outs[layer.cls_idx];
            const auto& kpt_out = npu_outs[layer.kpt_idx];

            for (int idx = 0; idx < layer.num_cells; idx++) {
                const int cls_base = idx * m_nc;
                float best_score = -1.0f;
                int best_label = -1;
                for (int c = 0; c < m_nc; c++) {
                    const float score = sigmoid(cls_out[cls_base + c]);
                    if (score > best_score) {
                        best_score = score;
                        best_label = c;
                    }
                }
                if (best_score < m_conf_thres) continue;

                const float ax = static_cast<float>(idx % layer.grid_w) + 0.5f;
                const float ay = static_cast<float>(idx / layer.grid_w) + 0.5f;
                const int box_base = idx * 4;
                const float l = box_out[box_base + 0];
                const float t = box_out[box_base + 1];
                const float r = box_out[box_base + 2];
                const float b = box_out[box_base + 3];

                std::vector<float> keypoints(m_nextra);
                const int kpt_base = idx * m_nextra;
                for (int k = 0; k < num_kpts; k++) {
                    const int base = kpt_base + k * 3;
                    keypoints[k * 3 + 0] = (kpt_out[base + 0] + ax) * layer.stride;
                    keypoints[k * 3 + 1] = (kpt_out[base + 1] + ay) * layer.stride;
                    keypoints[k * 3 + 2] = sigmoid(kpt_out[base + 2]);
                }

                pred_boxes.push_back({(ax - l) * layer.stride, (ay - t) * layer.stride,
                                      (ax + r) * layer.stride, (ay + b) * layer.stride});
                pred_scores.push_back(best_score);
                pred_labels.push_back(best_label);
                pred_extra.push_back(std::move(keypoints));
            }
        }
    }

    nms(pred_boxes, pred_scores, pred_labels, pred_extra, final_boxes, final_scores,
        final_labels, final_extra);

    const double end = set_timer();
    if (m_verbose) {
        std::cout << "Real C++ Time        : " << end - start << std::endl;
    }
}

void mobilint::post::YOLODFLFreePosePost::plot_boxes(
    cv::Mat& im, std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
    std::vector<int>& labels) {
    float ratio, xpad, ypad;
    compute_ratio_pads(im, ratio, xpad, ypad);

    cv::Rect rect;
    for (int i = 0; i < static_cast<int>(boxes.size()); i++) {
        if (labels[i] < 0 || labels[i] >= m_nc) {
            continue;
        }

        int xmin = static_cast<int>((boxes[i][0] - xpad) / ratio);
        int ymin = static_cast<int>((boxes[i][1] - ypad) / ratio);
        int xmax = static_cast<int>((boxes[i][2] - xpad) / ratio);
        int ymax = static_cast<int>((boxes[i][3] - ypad) / ratio);

        xmin = std::max(xmin, 0);
        ymin = std::max(ymin, 0);
        xmax = std::min(xmax, im.cols);
        ymax = std::min(ymax, im.rows);

        rect.x = xmin;
        rect.y = ymin;
        rect.width = xmax - xmin;
        rect.height = ymax - ymin;
        if (rect.width <= 0 || rect.height <= 0) {
            continue;
        }

        std::array<int, 3> bgr = COLORS[labels[i] % 20];
        cv::Scalar clr(bgr[0], bgr[1], bgr[2]);
        cv::rectangle(im, rect, clr, 1);
    }
}

void mobilint::post::YOLODFLFreePosePost::plot_keypoints(
    cv::Mat& im, std::vector<std::vector<float>>& kpts) {
    constexpr int radius = 2;
    constexpr int steps = 3;
    const int num_kpts = m_nextra / 3;
    constexpr float kpts_conf_thres = 0.4f;

    float ratio, xpad, ypad;
    compute_ratio_pads(im, ratio, xpad, ypad);

    for (auto& kpt_vec : kpts) {
        for (int j = 0; j < num_kpts; j++) {
            kpt_vec[3 * j + 0] = (kpt_vec[3 * j + 0] - xpad) / ratio;
            kpt_vec[3 * j + 1] = (kpt_vec[3 * j + 1] - ypad) / ratio;
        }
    }

    for (const auto& kpt_vec : kpts) {
        for (int j = 0; j < num_kpts; j++) {
            const auto bgr = m_pose_kpt_color[j];
            const cv::Scalar color(bgr[0], bgr[1], bgr[2]);
            const int kpt_idx = steps * j;
            const int x_coord = static_cast<int>(kpt_vec[kpt_idx]);
            const int y_coord = static_cast<int>(kpt_vec[kpt_idx + 1]);
            const float conf = kpt_vec[kpt_idx + 2];
            if (conf < kpts_conf_thres) {
                continue;
            }
            if (x_coord % m_imw != 0 && y_coord % m_imh != 0) {
                cv::circle(im, cv::Point(x_coord, y_coord), radius, color, -1);
            }
        }

        for (size_t j = 0; j < m_skeleton.size(); j++) {
            if (j == 17 || j == 18) {
                continue;
            }
            const auto bgr = m_pose_limb_color[j];
            const cv::Scalar color(bgr[0], bgr[1], bgr[2]);
            const auto& sk = m_skeleton[j];

            const float conf1 = kpt_vec[(sk[0] - 1) * steps + 2];
            const float conf2 = kpt_vec[(sk[1] - 1) * steps + 2];
            if (conf1 < 0.5f || conf2 < 0.5f) {
                continue;
            }

            const cv::Point p1(static_cast<int>(kpt_vec[(sk[0] - 1) * steps]),
                               static_cast<int>(kpt_vec[(sk[0] - 1) * steps + 1]));
            const cv::Point p2(static_cast<int>(kpt_vec[(sk[1] - 1) * steps]),
                               static_cast<int>(kpt_vec[(sk[1] - 1) * steps + 1]));

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

void mobilint::post::YOLODFLFreePosePost::plot_extras(
    cv::Mat& im, std::vector<std::vector<float>>& extras) {
    plot_keypoints(im, extras);
}
