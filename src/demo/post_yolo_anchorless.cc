#include "post_yolo_anchorless.h"

namespace {

/*
        Apply softmax function to the input vector in-place
*/
template <typename FloatContainer>
void softmax_inplace(FloatContainer& con) {
    float sum = 0;
    for (auto v : con) {
        sum += exp(v);
    }
    for (auto& v : con) {
        v = exp(v) / sum;
    }
}

}  // namespace

mobilint::post::YOLOAnchorlessPost::YOLOAnchorlessPost()
    : YOLOAnchorlessPost(80, 640, 640, 0.5f, 0.45f, false, true) {}

mobilint::post::YOLOAnchorlessPost::YOLOAnchorlessPost(int nc, int imh, int imw,
                                                       float conf_thres, float iou_thres,
                                                       bool verbose)
    : YOLOAnchorlessPost(nc, imh, imw, conf_thres, iou_thres, verbose, true) {}

mobilint::post::YOLOAnchorlessPost::YOLOAnchorlessPost(int nc, int imh, int imw,
                                                       float conf_thres, float iou_thres,
                                                       bool verbose, bool start_worker)
    : YoloPostBase(nc, imh, imw, conf_thres, iou_thres, verbose, PostType::OBJECT,
                   start_worker) {
    m_nl = 3;      // number of detection layers
    m_nextra = 0;  // number of extra outputs
    m_strides = generate_strides(m_nl);
    m_grids = generate_grids(m_imh, m_imw, m_strides);
}

/*
        Generate strides, needed for decoding
*/
std::vector<int> mobilint::post::YOLOAnchorlessPost::generate_strides(int nl) {
    std::vector<int> strides;
    for (int i = 0; i < nl; i++) {
        strides.push_back(pow(2, 3 + i));
    }
    return strides;
}

/*
        Generate grids, needed for decoding
*/
std::vector<std::vector<int>> mobilint::post::YOLOAnchorlessPost::generate_grids(
    int imh, int imw, std::vector<int> strides) {
    std::vector<std::vector<int>> all_grids;
    for (int i = 0; i < strides.size(); i++) {
        int grid_h = imh / strides[i];
        int grid_w = imw / strides[i];
        int grid_size = grid_h * grid_w * 2;

        std::vector<int> grids;
        for (int j = 0; j < grid_size; j++) {
            if (j % 2 == 0) {
                grids.push_back(((int)j / 2) % grid_w);
            } else {
                grids.push_back(((int)j / 2) / grid_w);
            }
        }

        all_grids.push_back(grids);
    }
    return all_grids;
}

int mobilint::post::YOLOAnchorlessPost::get_nl() const { return m_nl; }

int mobilint::post::YOLOAnchorlessPost::get_nc() const { return m_nc; }

/*
        Apply sigmoid function to the input float number
*/
float mobilint::post::YOLOAnchorlessPost::sigmoid(float num) {
    return 1 / (1 + exp(-(float)num));
}

/*
        Apply softmax function to the input vector
*/
std::vector<float> mobilint::post::YOLOAnchorlessPost::softmax(std::vector<float> vec) {
    std::vector<float> result;
    float sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += exp(vec[i]);
    }

    for (int i = 0; i < vec.size(); i++) {
        result.push_back(exp(vec[i]) / sum);
    }
    return result;
}

/*
        Convert boxes from Center Form(CxCyWiHe) to Corner Form(XminYminXmaxYmax)
*/
void mobilint::post::YOLOAnchorlessPost::xywh2xyxy(
    std::vector<std::array<float, 4>>& pred_boxes) {
    for (uint32_t i = 0; i < pred_boxes.size(); i++) {
        float cx = pred_boxes[i][0];
        float cy = pred_boxes[i][1];

        pred_boxes[i][0] = cx - pred_boxes[i][2] * 0.5;
        pred_boxes[i][1] = cy - pred_boxes[i][3] * 0.5;
        pred_boxes[i][2] = cx + pred_boxes[i][2] * 0.5;
        pred_boxes[i][3] = cy + pred_boxes[i][3] * 0.5;
    }
}

/*
        Access elements in output related to box coordinates and decode them
*/
void mobilint::post::YOLOAnchorlessPost::decode_boxes(const std::vector<float>& npu_out,
                                                      const std::vector<int>& grid,
                                                      int stride, int idx, int det_stride,
                                                      std::array<float, 4>& pred_box) {
    std::array<float, 4> box;
    std::array<float, 16> tmp;
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 16; k++) tmp[k] = npu_out[idx * det_stride + j * 16 + k];
        softmax_inplace(tmp);

        float box_value = 0;
        for (int k = 0; k < 16; k++) box_value += tmp[k] * k;
        box[j] = box_value;
    }

    float xmin = grid[idx * 2 + 0] - box[0] + 0.5;
    float ymin = grid[idx * 2 + 1] - box[1] + 0.5;
    float xmax = grid[idx * 2 + 0] + box[2] + 0.5;
    float ymax = grid[idx * 2 + 1] + box[3] + 0.5;

    float x = (xmin + xmax) / 2 * stride;
    float y = (ymin + ymax) / 2 * stride;
    float w = (xmax - xmin) * stride;
    float h = (ymax - ymin) * stride;

    pred_box = {x, y, w, h};
}

/*
        Access elements in output related to extra and decode them
*/
void mobilint::post::YOLOAnchorlessPost::decode_extra(const std::vector<float>& extra,
                                                      const std::vector<int>& grid,
                                                      int stride, int idx,
                                                      std::vector<float>& pred_extra) {
    // Do nothing for Object Detection
}

/*
        Decoding and masking with conf threshold
*/
void mobilint::post::YOLOAnchorlessPost::decode_conf_thres(
    const std::vector<float>& npu_out, const std::vector<int>& grid, int stride,
    std::vector<std::array<float, 4>>& pred_boxes, std::vector<float>& pred_conf,
    std::vector<int>& pred_label, std::vector<std::pair<float, int>>& pred_scores,
    const std::vector<float>& extra, std::vector<std::vector<float>>& pred_extra,
    const std::vector<float>* cls_out) {
    int grid_h = m_imh / stride;
    int grid_w = m_imw / stride;
    int grid_size = grid_h * grid_w;
    int det_stride = 0;
    int box_channels = 0;
    int tmp_no = 0;
    bool split_det_head = (cls_out != nullptr);

    if ((int)npu_out.size() % grid_size != 0) {
        throw std::invalid_argument(
            "Detection output size is not divisible by grid size.");
    }

    if (split_det_head) {
        box_channels = npu_out.size() / grid_size;
        det_stride = box_channels;
        if (box_channels != 64) {
            throw std::invalid_argument(
                "Split detection head expects 64 box channels per grid, received " +
                std::to_string(box_channels));
        }

        if ((int)cls_out->size() % grid_size != 0) {
            throw std::invalid_argument(
                "Class output size is not divisible by grid size.");
        }
        int cls_channels = cls_out->size() / grid_size;
        if (cls_channels != m_nc) {
            throw std::invalid_argument(
                "Split detection head expects class channels = m_nc (" +
                std::to_string(m_nc) + "), received " + std::to_string(cls_channels));
        }
    } else {
        tmp_no = npu_out.size() / grid_size;
        det_stride = tmp_no;
        if (tmp_no != m_nc + 64) {  // 64 = 16 * 4
            throw std::invalid_argument(
                "Number of outputs per grid should be " + std::to_string(m_nc + 64) +
                ", however post-processor received " + std::to_string(tmp_no));
        }
        box_channels = 64;
    }

#pragma omp parallel for num_threads(mOpenmpThreadCount) \
    shared(pred_boxes, pred_conf, pred_label, pred_scores, pred_extra)
    for (int i = 0; i < grid_size; i++) {
        std::array<float, 4> pred_box = {-999, -999, -999, -999};
        std::vector<float> pred_extra_values;
        for (int j = 0; j < m_nc; j++) {
            float conf = split_det_head ? sigmoid((*cls_out)[i * m_nc + j])
                                        : sigmoid(npu_out[i * tmp_no + box_channels + j]);
            if (conf > m_conf_thres) {
                if (pred_box[0] == -999) {  // decode box only once
                    decode_boxes(npu_out, grid, stride, i, det_stride, pred_box);
                    decode_extra(extra, grid, stride, i, pred_extra_values);
                }

#pragma omp critical
                {
                    pred_conf.push_back(conf);
                    pred_label.push_back(j);
                    pred_boxes.push_back(pred_box);
                    pred_scores.push_back(std::make_pair(conf, pred_scores.size()));
                    pred_extra.push_back(pred_extra_values);
                }
            }
        }
    }
}

/*
        Apply NMS
*/
void mobilint::post::YOLOAnchorlessPost::nms(
    std::vector<std::array<float, 4>> pred_boxes, std::vector<float> pred_conf,
    std::vector<int> pred_label, std::vector<std::pair<float, int>> scores,
    std::vector<std::vector<float>> pred_extra,
    std::vector<std::array<float, 4>>& final_boxes, std::vector<float>& final_scores,
    std::vector<int>& final_labels, std::vector<std::vector<float>>& final_extra) {
    // sort the scores(predicted confidence * predicted class score)
    sort(scores.begin(), scores.end(), std::greater<>());

    for (int i = 0; i < (int)scores.size(); i++) {
        float temp_score = scores[i].first;
        if (scores[i].first != -99) {  // check if the box valid or not
            int idx = scores[i].second;
            std::array<float, 4> max_box = pred_boxes[idx];

#pragma omp parallel for num_threads(mOpenmpThreadCount) \
    shared(pred_boxes, pred_conf, pred_label, scores, i, idx, max_box)
            for (int j = i; j < (int)scores.size(); j++) {
                int temp_idx = scores[j].second;
                std::array<float, 4> temp_box = pred_boxes[temp_idx];
                float iou = get_iou(max_box, temp_box);

                if (iou > m_iou_thres && pred_label[idx] == pred_label[temp_idx]) {
                    scores[j].first = -99;  // mark the invalid boxes
                }
            }

            final_boxes.push_back(max_box);
            final_scores.push_back(temp_score);
            final_labels.push_back(pred_label[idx]);
            final_extra.push_back(pred_extra[idx]);

            if (final_boxes.size() >= m_max_det_num) {
                break;
            }
        }
    }
}

void mobilint::post::YOLOAnchorlessPost::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();

    if (npu_outs.size() < m_nl)
        throw std::invalid_argument(
            "Size of model outputs does not match "
            "number of detection layers, expected at least " +
            std::to_string(m_nl) + " but received " + std::to_string(npu_outs.size()));

    struct LayerOutputMap {
        int det_idx = -1;
        int cls_idx = -1;
    };

    std::vector<LayerOutputMap> maps(m_nl);
    std::vector<bool> used(npu_outs.size(), false);
    for (int i = 0; i < m_nl; i++) {
        const int grid_size = static_cast<int>(m_grids[i].size() / 2);
        const size_t det_combined_size = static_cast<size_t>(grid_size * (m_nc + 64));
        const size_t cls_size = static_cast<size_t>(grid_size * m_nc);
        const size_t box_size = static_cast<size_t>(grid_size * 64);

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
                if (!used[j] && npu_outs[j].size() == cls_size) {
                    cls_idx = j;
                    break;
                }
            }
            for (int j = 0; j < static_cast<int>(npu_outs.size()); j++) {
                if (!used[j] && npu_outs[j].size() == box_size) {
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

        if (maps[i].det_idx < 0) {
            throw std::invalid_argument(
                "Unable to infer output layout in YOLOAnchorless post-processing.");
        }
    }

    for (int i = 0; i < static_cast<int>(used.size()); i++) {
        if (!used[i]) {
            throw std::invalid_argument(
                "Unexpected extra output detected in YOLOAnchorless post-processing.");
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
    std::vector<float> extra;

    for (int i = 0; i < m_nl; i++) {
        const std::vector<float>* cls_out = nullptr;
        if (maps[i].cls_idx >= 0) {
            cls_out = &npu_outs[maps[i].cls_idx];
        }

        decode_conf_thres(npu_outs[maps[i].det_idx], m_grids[i], m_strides[i], pred_boxes,
                          pred_conf, pred_label, pred_scores, extra, pred_extra, cls_out);
    }

    xywh2xyxy(pred_boxes);

    nms(pred_boxes, pred_conf, pred_label, pred_scores, pred_extra, final_boxes,
        final_scores, final_labels, final_extra);
    double end = set_timer();
    if (m_verbose) std::cout << "Real C++ Time        : " << end - start << std::endl;
}

/*
        Draw detected box and write the it's label & score
*/
void mobilint::post::YOLOAnchorlessPost::plot_boxes(
    cv::Mat& im, std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
    std::vector<int>& labels) {
    float ratio, xpad, ypad;
    compute_ratio_pads(im, ratio, xpad, ypad);

    cv::Rect rect;
    for (int i = 0; i < boxes.size(); i++) {
        if (labels[i] != 0) {
            continue;
        }
        int xmin = (int)(boxes[i][0] - xpad) / ratio;
        int ymin = (int)(boxes[i][1] - ypad) / ratio;
        int xmax = (int)(boxes[i][2] - xpad) / ratio;
        int ymax = (int)(boxes[i][3] - ypad) / ratio;

        // clip the box
        xmin = std::max(xmin, 0);
        ymin = std::max(ymin, 0);
        xmax = std::min(xmax, im.cols);
        ymax = std::min(ymax, im.rows);

        rect.x = xmin;
        rect.y = ymin;
        rect.width = xmax - xmin;
        rect.height = ymax - ymin;

        std::array<int, 3> bgr = COLORS[labels[i] % 20];
        cv::Scalar clr(bgr[0], bgr[1], bgr[2]);
        cv::rectangle(im, rect, clr, 1);

        double font_scale = std::min(std::max(rect.width / 500.0, 0.35), 0.99);
        std::string desc =
            COCO_LABELS[labels[i]] + " " + std::to_string((int)(scores[i] * 100)) + "%";
        cv::putText(im, desc, cv::Point(xmin, std::max(ymin - 10, 0)),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, clr, 1, false);
    }
}
