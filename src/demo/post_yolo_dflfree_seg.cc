#include "post_yolo_dflfree_seg.h"

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

cv::Mat unpad_yolo_dflfree_seg(const cv::Mat& image, int xpad, int ypad) {
    const int rows = image.rows;
    const int cols = image.cols;
    const int width = cols - 2 * xpad;
    const int height = rows - 2 * ypad;
    const cv::Rect rect(xpad, ypad, width, height);
    cv::Mat roi = image(rect);
    cv::Mat cropped;
    roi.copyTo(cropped);
    return cropped;
}

template <typename T>
cv::Mat interpolate(const cv::Mat& input, const cv::Size& size, int mode) {
    cv::Mat output;
    cv::resize(input, output, size, 0, 0, mode);
    return output;
}
}  // namespace

mobilint::post::YOLODFLFreeSegPost::YOLODFLFreeSegPost()
    : YOLODFLFreePost(80, 640, 640, 0.2f, 0.55f, false, false) {
    m_nextra = 32;  // mask coefficients
    mType = PostType::SEG;
    m_proto_stride = 4;
    m_proto_h = m_imh / m_proto_stride;
    m_proto_w = m_imw / m_proto_stride;
    start_worker_thread();
}

mobilint::post::YOLODFLFreeSegPost::YOLODFLFreeSegPost(int nc, int imh, int imw,
                                                       float conf_thres, float iou_thres,
                                                       bool verbose)
    : YOLODFLFreePost(nc, imh, imw, conf_thres, iou_thres, verbose, false) {
    m_nextra = 32;  // mask coefficients
    mType = PostType::SEG;
    m_proto_stride = 4;
    m_proto_h = m_imh / m_proto_stride;
    m_proto_w = m_imw / m_proto_stride;
    start_worker_thread();
}

void mobilint::post::YOLODFLFreeSegPost::decode_outputs(
    const std::vector<float>& box_out, const std::vector<float>& cls_out,
    const std::vector<float>& mask_out, std::vector<std::array<float, 4>>& pred_boxes,
    std::vector<float>& pred_scores, std::vector<int>& pred_labels,
    std::vector<std::vector<float>>& pred_extra) {
    constexpr int kBoxElements = 4;
    const int num_detections = static_cast<int>(box_out.size() / kBoxElements);

    if (box_out.size() != static_cast<size_t>(num_detections * kBoxElements)) {
        throw std::invalid_argument(
            "Invalid box output shape in YOLODFLFree seg output.");
    }

    if (cls_out.size() != static_cast<size_t>(num_detections * m_nc)) {
        throw std::invalid_argument(
            "Invalid class output shape in YOLODFLFree seg output.");
    }

    if (mask_out.size() != static_cast<size_t>(num_detections * m_nextra)) {
        throw std::invalid_argument(
            "Invalid mask output shape in YOLODFLFree seg output.");
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
        const int mask_idx = i * m_nextra;

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

        std::vector<float> coeffs(m_nextra);
        for (int j = 0; j < m_nextra; j++) {
            coeffs[j] = mask_out[mask_idx + j];
        }

        pred_boxes.push_back(box);
        pred_scores.push_back(best_score);
        pred_labels.push_back(best_label);
        pred_extra.push_back(std::move(coeffs));
    }
}

void mobilint::post::YOLODFLFreeSegPost::nms(
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

cv::Mat& mobilint::post::YOLODFLFreeSegPost::get_label_mask() { return label_masks; }

cv::Mat& mobilint::post::YOLODFLFreeSegPost::get_final_mask() { return final_masks; }

void mobilint::post::YOLODFLFreeSegPost::plot_boxes(
    cv::Mat& im, std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
    std::vector<int>& labels) {}

void mobilint::post::YOLODFLFreeSegPost::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    const double start = set_timer();

    if (npu_outs.size() < 4) {
        throw std::invalid_argument(
            "YOLODFLFree Seg post-processing expects at least 4 outputs, but received " +
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
    std::vector<float> proto;

    if (npu_outs.size() == 4) {
        int proto_idx = -1;
        const size_t proto_expected =
            static_cast<size_t>(m_proto_h * m_proto_w * m_nextra);
        for (int i = 0; i < 4; i++) {
            if (npu_outs[i].size() == proto_expected) {
                proto_idx = i;
                break;
            }
        }
        if (proto_idx < 0) {
            throw std::invalid_argument(
                "Unable to infer proto output in decoded YOLODFLFree seg output.");
        }

        std::array<int, 3> det_indices = {-1, -1, -1};
        int det_count = 0;
        for (int i = 0; i < 4; i++) {
            if (i == proto_idx) continue;
            det_indices[det_count++] = i;
        }

        int box_idx = -1;
        int cls_idx = -1;
        int mask_idx = -1;
        for (int u = 0; u < 3; u++) {
            const int i = det_indices[u];
            if (npu_outs[i].size() % 4 != 0) continue;
            const int num_det = static_cast<int>(npu_outs[i].size() / 4);
            for (int v = 0; v < 3; v++) {
                if (v == u) continue;
                const int j = det_indices[v];
                if (npu_outs[j].size() != static_cast<size_t>(num_det * m_nc)) continue;
                int w = 0;
                while (w == u || w == v) w++;
                const int k = det_indices[w];
                if (npu_outs[k].size() == static_cast<size_t>(num_det * m_nextra)) {
                    box_idx = i;
                    cls_idx = j;
                    mask_idx = k;
                    break;
                }
            }
            if (box_idx >= 0) break;
        }
        if (box_idx < 0 || cls_idx < 0 || mask_idx < 0) {
            throw std::invalid_argument(
                "Unable to infer decoded output layout for YOLODFLFree seg "
                "post-processing. "
                "Output sizes=" +
                summarize_sizes(npu_outs));
        }

        decode_outputs(npu_outs[box_idx], npu_outs[cls_idx], npu_outs[mask_idx],
                       pred_boxes, pred_scores, pred_labels, pred_extra);
        proto = npu_outs[proto_idx];
    } else {
        // Non-decoded multi-head outputs: [box4_h, cls_h, mask_h] * nl + proto.
        int proto_idx = -1;
        const size_t proto_expected =
            static_cast<size_t>(m_proto_h * m_proto_w * m_nextra);
        for (int i = 0; i < static_cast<int>(npu_outs.size()); i++) {
            if (npu_outs[i].size() == proto_expected) {
                proto_idx = i;
                break;
            }
        }
        if (proto_idx < 0) {
            throw std::invalid_argument(
                "Unable to infer proto output for YOLODFLFree seg post-processing. "
                "Output sizes=" +
                summarize_sizes(npu_outs));
        }
        proto = npu_outs[proto_idx];

        struct LayerHead {
            int box_idx = -1;
            int cls_idx = -1;
            int mask_idx = -1;
            int num_cells = 0;
            int grid_h = 0;
            int grid_w = 0;
            int stride = 0;
        };
        std::vector<LayerHead> layers;
        std::vector<bool> used(npu_outs.size(), false);
        used[proto_idx] = true;

        for (int i = 0; i < static_cast<int>(npu_outs.size()); i++) {
            if (used[i] || npu_outs[i].size() % 4 != 0) continue;
            const int num_cells = static_cast<int>(npu_outs[i].size() / 4);

            int cls_idx = -1;
            int mask_idx = -1;
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
                    mask_idx = j;
                    break;
                }
            }
            if (cls_idx < 0 || mask_idx < 0) continue;

            LayerHead layer;
            layer.box_idx = i;
            layer.cls_idx = cls_idx;
            layer.mask_idx = mask_idx;
            layer.num_cells = num_cells;
            if (!infer_grid_and_stride(m_imh, m_imw, num_cells, layer.grid_h,
                                       layer.grid_w, layer.stride)) {
                throw std::invalid_argument(
                    "Failed to infer grid/stride for DFLFree seg head. num_cells=" +
                    std::to_string(num_cells));
            }
            used[i] = true;
            used[cls_idx] = true;
            used[mask_idx] = true;
            layers.push_back(layer);
        }

        for (int i = 0; i < static_cast<int>(used.size()); i++) {
            if (!used[i]) {
                throw std::invalid_argument(
                    "Unable to infer output layout for YOLODFLFree seg post-processing. "
                    "Output sizes=" +
                    summarize_sizes(npu_outs));
            }
        }
        if (layers.empty()) {
            throw std::invalid_argument(
                "No valid detection heads found for YOLODFLFree seg post-processing. "
                "Output sizes=" +
                summarize_sizes(npu_outs));
        }

        for (const auto& layer : layers) {
            const auto& box_out = npu_outs[layer.box_idx];
            const auto& cls_out = npu_outs[layer.cls_idx];
            const auto& mask_out = npu_outs[layer.mask_idx];

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

                std::vector<float> coeffs(m_nextra);
                const int mask_base = idx * m_nextra;
                for (int m = 0; m < m_nextra; m++) {
                    coeffs[m] = mask_out[mask_base + m];
                }

                pred_boxes.push_back({(ax - l) * layer.stride, (ay - t) * layer.stride,
                                      (ax + r) * layer.stride, (ay + b) * layer.stride});
                pred_scores.push_back(best_score);
                pred_labels.push_back(best_label);
                pred_extra.push_back(std::move(coeffs));
            }
        }
    }

    nms(pred_boxes, pred_scores, pred_labels, pred_extra, final_boxes, final_scores,
        final_labels, final_extra);
    process_mask(proto, final_extra, final_boxes, final_labels);

    const double end = set_timer();
    if (m_verbose) {
        std::cout << "Real C++ Time        : " << end - start << std::endl;
    }
}

std::vector<std::array<float, 4>> mobilint::post::YOLODFLFreeSegPost::downsample_boxes(
    std::vector<std::array<float, 4>> boxes) {
    for (auto& box : boxes) {
        for (auto& coord : box) {
            coord /= m_proto_stride;
        }
    }
    return boxes;
}

void mobilint::post::YOLODFLFreeSegPost::process_mask(
    const std::vector<float>& proto, const std::vector<std::vector<float>>& masks,
    const std::vector<std::array<float, 4>>& boxes, const std::vector<int>& labels) {
    const auto boxes_down = downsample_boxes(boxes);
    const int num_boxes = static_cast<int>(boxes.size());
    const int matmul_col = m_nextra;

    cv::Mat temp_label_masks = cv::Mat::zeros(m_proto_h, m_proto_w, CV_32F);
    cv::Mat temp_final_masks = cv::Mat::zeros(m_proto_h, m_proto_w, CV_32F);

    for (int i = 0; i < num_boxes; i++) {
        if (labels[i] != 0) {
            continue;
        }
        const int x_min = std::max(static_cast<int>(boxes_down[i][0]), 0);
        const int y_min = std::max(static_cast<int>(boxes_down[i][1]), 0);
        const int x_max = std::min(static_cast<int>(boxes_down[i][2]), m_proto_w - 1);
        const int y_max = std::min(static_cast<int>(boxes_down[i][3]), m_proto_h - 1);

#pragma omp parallel for num_threads(mOpenmpThreadCount)                 \
    shared(proto, masks, labels, x_min, y_min, x_max, y_max, matmul_col, \
           temp_label_masks, temp_final_masks)
        for (int h = y_min; h <= y_max; h++) {
            for (int w = x_min; w <= x_max; w++) {
                float temp = 0.0f;
                const int idx_proto = h * m_proto_w * matmul_col + w * matmul_col;
                for (int j = 0; j < matmul_col; j++) {
                    temp += masks[i][j] * proto[idx_proto + j];
                }
                const float temp_sig = sigmoid(temp);

                if (temp_final_masks.at<float>(h, w) < temp_sig) {
                    temp_label_masks.at<float>(h, w) = labels[i] + 1;  // 0 background
                    temp_final_masks.at<float>(h, w) = temp_sig;
                }
            }
        }
    }

    label_masks = interpolate<float>(temp_label_masks, cv::Size(m_imw, m_imh), 0);
    final_masks = interpolate<float>(temp_final_masks, cv::Size(m_imw, m_imh), 1);
}

void mobilint::post::YOLODFLFreeSegPost::plot_masks(
    cv::Mat& im, cv::Mat& masks, cv::Mat& lbl_masks,
    const std::vector<std::array<float, 4>>& boxes) {
    float ratio, xpad, ypad;
    compute_ratio_pads(im, ratio, xpad, ypad);

    cv::Mat colored_masks(m_imh, m_imw, CV_8UC3);
    colored_masks.setTo(cv::Scalar(0, 0, 0));

    for (size_t i = 0; i < boxes.size(); i++) {
        const int x_min = std::max(static_cast<int>(boxes[i][0]), 0);
        const int y_min = std::max(static_cast<int>(boxes[i][1]), 0);
        const int x_max =
            std::min(static_cast<int>(boxes[i][2]), static_cast<int>(m_imw) - 1);
        const int y_max =
            std::min(static_cast<int>(boxes[i][3]), static_cast<int>(m_imh) - 1);

        for (int h = y_min; h <= y_max; h++) {
            for (int w = x_min; w <= x_max; w++) {
                if (masks.at<float>(h, w) > 0.5f) {
                    const int idx = h * m_imw + w;
                    const int cls = static_cast<int>(lbl_masks.at<float>(h, w)) - 1;
                    if (cls < 0) {
                        continue;
                    }
                    const std::array<int, 3> bgr = COLORS[cls % 20];
                    colored_masks.data[3 * idx + 0] = static_cast<uint8_t>(bgr[0]);
                    colored_masks.data[3 * idx + 1] = static_cast<uint8_t>(bgr[1]);
                    colored_masks.data[3 * idx + 2] = static_cast<uint8_t>(bgr[2]);
                }
            }
        }
    }

    colored_masks = unpad_yolo_dflfree_seg(colored_masks, static_cast<int>(xpad),
                                           static_cast<int>(ypad));
    colored_masks = interpolate<float>(colored_masks, im.size(), 1);
    cv::addWeighted(im, 0.9, colored_masks, 0.7, 0.0, im);
}

void mobilint::post::YOLODFLFreeSegPost::plot_extras(
    cv::Mat& im, std::vector<std::vector<float>>& extras) {
    (void)extras;
    auto lbl_masks = get_label_mask();
    auto masks = get_final_mask();
    plot_masks(im, masks, lbl_masks, final_boxes);
}
