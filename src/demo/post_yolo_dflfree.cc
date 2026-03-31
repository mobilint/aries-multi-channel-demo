#include "post_yolo_dflfree.h"

#include <omp.h>

#include <cmath>
#include <numeric>
#include <sstream>

namespace {
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

int get_openmp_thread_count() {
    const int max_threads = omp_get_max_threads();
    if (max_threads <= 1) return 1;
    return std::min(4, max_threads);
}
}  // namespace

mobilint::post::YOLODFLFreePost::YOLODFLFreePost()
    : YOLODFLFreePost(80, 640, 640, 0.5f, 0.45f, false, true) {}

mobilint::post::YOLODFLFreePost::YOLODFLFreePost(int nc, int imh, int imw,
                                                 float conf_thres, float iou_thres,
                                                 bool verbose)
    : YOLODFLFreePost(nc, imh, imw, conf_thres, iou_thres, verbose, true) {}

mobilint::post::YOLODFLFreePost::YOLODFLFreePost(int nc, int imh, int imw,
                                                 float conf_thres, float iou_thres,
                                                 bool verbose, bool start_worker)
    : YoloPostBase(nc, imh, imw, conf_thres, iou_thres, verbose, PostType::OBJECT,
                   start_worker) {}

float mobilint::post::YOLODFLFreePost::sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}

void mobilint::post::YOLODFLFreePost::decode_outputs(
    const std::vector<float>& box_out, const std::vector<float>& cls_out,
    std::vector<std::array<float, 4>>& pred_boxes, std::vector<float>& pred_scores,
    std::vector<int>& pred_labels) {
    constexpr int kNumDetections = 8400;
    constexpr int kBoxElements = 4;

    if (box_out.size() != static_cast<size_t>(kNumDetections * kBoxElements)) {
        throw std::invalid_argument(
            "Size of model outputs does not match dimension of boxes, expected " +
            std::to_string(kNumDetections * kBoxElements) + " but received " +
            std::to_string(box_out.size()));
    }

    if (cls_out.size() != static_cast<size_t>(kNumDetections * m_nc)) {
        throw std::invalid_argument(
            "Size of model outputs does not match number of classes, expected " +
            std::to_string(kNumDetections * m_nc) + " but received " +
            std::to_string(cls_out.size()));
    }

    pred_boxes.clear();
    pred_scores.clear();
    pred_labels.clear();

    pred_boxes.reserve(kNumDetections);
    pred_scores.reserve(kNumDetections);
    pred_labels.reserve(kNumDetections);

    const int thread_count = get_openmp_thread_count();
    std::vector<std::vector<std::array<float, 4>>> thread_boxes(thread_count);
    std::vector<std::vector<float>> thread_scores(thread_count);
    std::vector<std::vector<int>> thread_labels(thread_count);

#pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < kNumDetections; i++) {
        const int tid = omp_get_thread_num();
        const int box_idx = i * kBoxElements;
        const int cls_idx = i * m_nc;

        float best_score = -1.0f;
        int best_label = -1;
        for (int c = 0; c < m_nc; c++) {
            float score = cls_out[cls_idx + c];
            if (score < 0.0f || score > 1.0f) {
                score = 1.0f / (1.0f + std::exp(-score));
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

        // If NPU output is cx, cy, w, h, convert to xyxy.
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

        thread_boxes[tid].push_back(box);
        thread_scores[tid].push_back(best_score);
        thread_labels[tid].push_back(best_label);
    }

    for (int t = 0; t < thread_count; t++) {
        pred_boxes.insert(pred_boxes.end(), thread_boxes[t].begin(),
                          thread_boxes[t].end());
        pred_scores.insert(pred_scores.end(), thread_scores[t].begin(),
                           thread_scores[t].end());
        pred_labels.insert(pred_labels.end(), thread_labels[t].begin(),
                           thread_labels[t].end());
    }
}

void mobilint::post::YOLODFLFreePost::nms(
    const std::vector<std::array<float, 4>>& pred_boxes,
    const std::vector<float>& pred_scores, const std::vector<int>& pred_labels,
    std::vector<std::array<float, 4>>& final_boxes, std::vector<float>& final_scores,
    std::vector<int>& final_labels) {
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

void mobilint::post::YOLODFLFreePost::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    const double start = set_timer();

    if (npu_outs.size() < 2) {
        throw std::invalid_argument(
            "Size of model outputs does not match number of outputs, expected at least " +
            std::to_string(2) + " but received " + std::to_string(npu_outs.size()));
    }

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<float> pred_scores;
    std::vector<int> pred_labels;

    if (npu_outs.size() == 2) {
        // Decoded NPU outputs: [box, cls]
        decode_outputs(npu_outs[0], npu_outs[1], pred_boxes, pred_scores, pred_labels);
    } else {
        // Non-decoded multi-head outputs: [box4_h1, cls_h1, box4_h2, cls_h2, ...]
        // (order-agnostic)
        struct LayerHead {
            int box_idx = -1;
            int cls_idx = -1;
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
            for (int j = 0; j < static_cast<int>(npu_outs.size()); j++) {
                if (used[j] || j == i) continue;
                if (npu_outs[j].size() == static_cast<size_t>(num_cells * m_nc)) {
                    cls_idx = j;
                    break;
                }
            }
            if (cls_idx < 0) continue;

            LayerHead layer;
            layer.box_idx = i;
            layer.cls_idx = cls_idx;
            layer.num_cells = num_cells;
            if (!infer_grid_and_stride(m_imh, m_imw, num_cells, layer.grid_h,
                                       layer.grid_w, layer.stride)) {
                throw std::invalid_argument(
                    "Failed to infer grid/stride for DFLFree object head. num_cells=" +
                    std::to_string(num_cells));
            }
            used[i] = true;
            used[cls_idx] = true;
            layers.push_back(layer);
        }

        for (int i = 0; i < static_cast<int>(used.size()); i++) {
            if (!used[i]) {
                throw std::invalid_argument(
                    "Unable to infer output layout for YOLODFLFree object "
                    "post-processing. "
                    "Output sizes=" +
                    summarize_sizes(npu_outs));
            }
        }
        if (layers.empty()) {
            throw std::invalid_argument(
                "No valid detection heads found for YOLODFLFree object post-processing. "
                "Output sizes=" +
                summarize_sizes(npu_outs));
        }

        const int thread_count = get_openmp_thread_count();
        for (const auto& layer : layers) {
            const auto& box_out = npu_outs[layer.box_idx];
            const auto& cls_out = npu_outs[layer.cls_idx];
            std::vector<std::vector<std::array<float, 4>>> thread_boxes(thread_count);
            std::vector<std::vector<float>> thread_scores(thread_count);
            std::vector<std::vector<int>> thread_labels(thread_count);

#pragma omp parallel for num_threads(thread_count)
            for (int idx = 0; idx < layer.num_cells; idx++) {
                const int tid = omp_get_thread_num();
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

                const int box_base = idx * 4;
                const float l = box_out[box_base + 0];
                const float t = box_out[box_base + 1];
                const float r = box_out[box_base + 2];
                const float b = box_out[box_base + 3];
                const float ax = static_cast<float>(idx % layer.grid_w) + 0.5f;
                const float ay = static_cast<float>(idx / layer.grid_w) + 0.5f;

                thread_boxes[tid].push_back(
                    {(ax - l) * layer.stride, (ay - t) * layer.stride,
                     (ax + r) * layer.stride, (ay + b) * layer.stride});
                thread_scores[tid].push_back(best_score);
                thread_labels[tid].push_back(best_label);
            }

            for (int t = 0; t < thread_count; t++) {
                pred_boxes.insert(pred_boxes.end(), thread_boxes[t].begin(),
                                  thread_boxes[t].end());
                pred_scores.insert(pred_scores.end(), thread_scores[t].begin(),
                                   thread_scores[t].end());
                pred_labels.insert(pred_labels.end(), thread_labels[t].begin(),
                                   thread_labels[t].end());
            }
        }
    }

    // DFLFree object is NMS-free in model-zoo reference implementation.
    // Keep top-k by confidence only to cap drawing/work.
    std::vector<int> order(pred_scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int lhs, int rhs) { return pred_scores[lhs] > pred_scores[rhs]; });
    const size_t keep = std::min(order.size(), static_cast<size_t>(m_max_det_num));
    final_boxes.reserve(keep);
    final_scores.reserve(keep);
    final_labels.reserve(keep);
    for (size_t i = 0; i < keep; i++) {
        const int idx = order[i];
        final_boxes.push_back(pred_boxes[idx]);
        final_scores.push_back(pred_scores[idx]);
        final_labels.push_back(pred_labels[idx]);
    }

    const double end = set_timer();
    if (m_verbose) {
        std::cout << "Real C++ Time        : " << end - start << std::endl;
    }
}

void mobilint::post::YOLODFLFreePost::plot_boxes(cv::Mat& im,
                                                 std::vector<std::array<float, 4>>& boxes,
                                                 std::vector<float>& scores,
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

        const std::string label_name = (labels[i] < static_cast<int>(COCO_LABELS.size()))
                                           ? COCO_LABELS[labels[i]]
                                           : ("class_" + std::to_string(labels[i]));
        const std::string desc =
            label_name + " " + std::to_string(static_cast<int>(scores[i] * 100)) + "%";
        const double font_scale = std::min(std::max(rect.width / 500.0, 0.35), 0.99);
        cv::putText(im, desc, cv::Point(xmin, std::max(ymin - 10, 0)),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, clr, 1, false);
    }
}

void mobilint::post::YOLODFLFreePost::plot_extras(
    cv::Mat& im, std::vector<std::vector<float>>& extras) {
    (void)im;
    (void)extras;
}
