#include "post_yolo_base.h"

#include <algorithm>
#include <iostream>

namespace mobilint::post {
YoloPostBase::YoloPostBase(int nc, int imh, int imw, float conf_thres, float iou_thres,
                           bool verbose, PostType type, bool start_worker)
    : m_nc(nc),
      m_imh(imh),
      m_imw(imw),
      m_conf_thres(conf_thres),
      m_iou_thres(iou_thres),
      m_max_det_num(300),
      m_verbose(verbose),
      mType(type),
      ticket(0),
      destroyed(false),
      mWorkerStarted(false) {
    if (start_worker) {
        start_worker_thread();
    }
}

YoloPostBase::~YoloPostBase() {
    destroyed = true;
    mCondIn.notify_all();
    mCondOut.notify_all();
    if (mThread.joinable()) {
        mThread.join();
    }
}

void YoloPostBase::start_worker_thread() {
    if (mWorkerStarted) {
        return;
    }
    mThread = std::thread(&YoloPostBase::worker, this);
    mWorkerStarted = true;
}

uint64_t YoloPostBase::enqueue(cv::Mat& im, std::vector<std::vector<float>>& npu_outs,
                               std::vector<std::array<float, 4>>& boxes,
                               std::vector<float>& scores, std::vector<int>& labels,
                               std::vector<std::vector<float>>& extras) {
    auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto title = std::to_string(thres_id) + " | Postprocessor Enqueue | ";

    print(title + "Start");
    uint64_t ticket_save = 0;

    {
        std::lock_guard<std::mutex> lk(mMutexIn);
        mQueueIn.push({++ticket, im, npu_outs, boxes, scores, labels, extras});
        ticket_save = ticket;

        mCondIn.notify_all();
        print(title + "Input Queue size " + std::to_string(mQueueIn.size()));
    }

    print(title + "Finish");
    return ticket_save;
}

void YoloPostBase::receive(uint64_t receipt_no) {
    auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto title = std::to_string(thres_id) + " | Postprocessor Receive | ";

    while (!destroyed) {
        print(title + "Start | Receipt = " + std::to_string(receipt_no));
        std::unique_lock<std::mutex> lk(mMutexOut);

        if (mOut.empty()) {
            mCondOut.wait(lk, [this] { return !(mOut.empty()) || destroyed; });
        }

        print(title + "Received Output Queue of size " + std::to_string(mOut.size()));

        if (destroyed) {
            break;
        }

        for (int i = 0; i < static_cast<int>(mOut.size()); i++) {
            if (mOut[i] == receipt_no) {
                print(title + "Got my output | Receipt = " + std::to_string(mOut[i]));
                mOut.erase(mOut.begin() + i);
                return;
            }
        }

        lk.unlock();
        print(title + "Finish");
    }
}

PostType YoloPostBase::getType() const { return mType; }

float YoloPostBase::area(float xmin, float ymin, float xmax, float ymax) const {
    float width = xmax - xmin;
    float height = ymax - ymin;

    if (width < 0) return 0;
    if (height < 0) return 0;

    return width * height;
}

float YoloPostBase::get_iou(std::array<float, 4> box1, std::array<float, 4> box2) const {
    float epsilon = 1e-6f;

    float overlap_xmin = std::max(box1[0], box2[0]);
    float overlap_ymin = std::max(box1[1], box2[1]);
    float overlap_xmax = std::min(box1[2], box2[2]);
    float overlap_ymax = std::min(box1[3], box2[3]);

    float overlap_area = area(overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax);
    float area1 = area(box1[0], box1[1], box1[2], box1[3]);
    float area2 = area(box2[0], box2[1], box2[2], box2[3]);

    return overlap_area / (area1 + area2 - overlap_area + epsilon);
}

double YoloPostBase::set_timer() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::vector<std::array<float, 4>>& YoloPostBase::get_result_box() { return final_boxes; }

std::vector<float>& YoloPostBase::get_result_score() { return final_scores; }

std::vector<int>& YoloPostBase::get_result_label() { return final_labels; }

std::vector<std::vector<float>>& YoloPostBase::get_result_extra() { return final_extra; }

void YoloPostBase::compute_ratio_pads(const cv::Mat& im, float& ratio, float& xpad,
                                      float& ypad) const {
    cv::Size size = im.size();
    if (size.width > size.height) {
        ratio = static_cast<float>(m_imw) / size.width;
        xpad = 0;
        ypad = (m_imh - ratio * size.height) / 2;
    } else {
        ratio = static_cast<float>(m_imh) / size.height;
        xpad = (m_imw - ratio * size.width) / 2;
        ypad = 0;
    }
}

void YoloPostBase::print(const std::string& msg) const {
#ifdef DEBUG
    std::lock_guard<std::mutex> lk(mPrintMutex);
    std::cout << msg << std::endl;
#else
    (void)msg;
#endif
}

void YoloPostBase::plot_extras(cv::Mat& im, std::vector<std::vector<float>>& extras) {
    (void)im;
    (void)extras;
}

void YoloPostBase::worker() {
    auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto title = std::to_string(thres_id) + " | Postprocessor Worker | ";

    print(title + "Start");
    while (!destroyed) {
        std::unique_lock<std::mutex> lk(mMutexIn);
        if (mQueueIn.empty()) {
            mCondIn.wait(lk, [this] { return !mQueueIn.empty() || destroyed; });
        }

        if (destroyed) {
            break;
        }

        auto k = mQueueIn.front();
        mQueueIn.pop();
        lk.unlock();

        auto start = set_timer();

        run_postprocess(k.npu_outs);
        k.boxes = get_result_box();
        k.scores = get_result_score();
        k.labels = get_result_label();
        k.extras = get_result_extra();

        plot_boxes(k.im, k.boxes, k.scores, k.labels);
        plot_extras(k.im, k.extras);

        auto end = set_timer();
        auto elapsed = std::to_string(end - start);

        print(title + "Postprocessing time: " + elapsed);
        print(title + "Number of detections " + std::to_string(k.boxes.size()));

        std::unique_lock<std::mutex> lk2(mMutexOut);
        mOut.push_back(k.id);
        lk2.unlock();

        std::unique_lock<std::mutex> lk_(mMutexOut);
        mCondOut.notify_all();
        lk_.unlock();
    }
    print(title + "Finish");
}
}  // namespace mobilint::post
