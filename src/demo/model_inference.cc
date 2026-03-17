#include <algorithm>
#include <cmath>

#include "demo/model.h"
#include "opencv2/opencv.hpp"

namespace {
void prepare_yolo_input(const cv::Mat& frame, int w, int h, cv::Mat& resized_frame,
                        cv::Mat& padded_resized_frame) {
    const float scale =
        std::min(static_cast<float>(w) / frame.cols, static_cast<float>(h) / frame.rows);
    const int resized_w = std::max(1, static_cast<int>(std::round(frame.cols * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(frame.rows * scale)));

    const int x_pad = (w - resized_w) / 2;
    const int y_pad = (h - resized_h) / 2;

    cv::resize(frame, resized_frame, cv::Size(resized_w, resized_h));
    padded_resized_frame = cv::Mat::zeros(h, w, CV_8UC3);
    resized_frame.copyTo(
        padded_resized_frame(cv::Rect{x_pad, y_pad, resized_w, resized_h}));
}
}  // namespace

cv::Mat Model::inferenceSSD(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int c = mModel->getInputBufferInfo()[0].original_channel;

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(w, h));

    auto input_img = std::make_unique<float[]>(w * h * c);
    for (int i = 0; i < w * h * c; i++) {
        input_img.get()[i] = ((float)resized_frame.data[i] - 127.5) / 127.5;
    }

    npu_benchmarker.start();
    auto result = mModel->infer({input_img.get()}, sc);
    npu_benchmarker.end();
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

    std::vector<float> boxes, classes, scores;
    uint64_t ticket = mPost->enqueue(result, boxes, classes, scores);
    mPost->receive(ticket);

    cv::Mat result_frame;
    cv::resize(frame, result_frame, size);

    cv::Point pt1, pt2;
    for (int i = 0; i < scores.size(); i++) {
        if (classes[i] != 1) {
            continue;
        }
        pt1.x = boxes[i * 4 + 0] * size.width;
        pt1.y = boxes[i * 4 + 1] * size.height;
        pt2.x = boxes[i * 4 + 2] * size.width;
        pt2.y = boxes[i * 4 + 3] * size.height;

        cv::rectangle(result_frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }

    return result_frame;
}

cv::Mat Model::inferenceStyle(cv::Mat frame, cv::Size size,
                              Benchmarker& npu_benchmarker) {
    mobilint::StatusCode sc;
    int wi = mModel->getInputBufferInfo()[0].original_width;
    int hi = mModel->getInputBufferInfo()[0].original_height;
    int ci = mModel->getInputBufferInfo()[0].original_channel;

    int wo = mModel->getOutputBufferInfo()[0].original_width;
    int ho = mModel->getOutputBufferInfo()[0].original_height;
    int co = mModel->getOutputBufferInfo()[0].original_channel;

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(wi, hi));

    auto input_img = std::make_unique<float[]>(wi * hi * ci);
    constexpr float rev = 1.0f / 255.0f;
    float* ptr = input_img.get();
    for (int i = 0; i < wi * hi; i++) {
        // BGR -> RGB
        ptr[i * 3 + 0] = static_cast<float>(resized_frame.data[i * 3 + 2]) * rev;
        ptr[i * 3 + 1] = static_cast<float>(resized_frame.data[i * 3 + 1]) * rev;
        ptr[i * 3 + 2] = static_cast<float>(resized_frame.data[i * 3 + 0]) * rev;
    }

    npu_benchmarker.start();
    auto result = mModel->infer({input_img.get()}, sc);
    npu_benchmarker.end();
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

    for (int i = 0; i < wo * ho; i++) {
        // RGB -> BGR
        resized_frame.data[i * 3 + 0] =
            (uint8_t)std::max(0.0f, std::min(result[0][i * 3 + 2] * 255.0f, 255.0f));
        resized_frame.data[i * 3 + 1] =
            (uint8_t)std::max(0.0f, std::min(result[0][i * 3 + 1] * 255.0f, 255.0f));
        resized_frame.data[i * 3 + 2] =
            (uint8_t)std::max(0.0f, std::min(result[0][i * 3 + 0] * 255.0f, 255.0f));
    }

    int crop_x = 35;
    int crop_y = 20;
    int crop_w = wo - crop_x * 2;
    int crop_h = ho - crop_y * 2;
    cv::Mat cropped_frame = resized_frame(cv::Rect{crop_x, crop_y, crop_w, crop_h});

    cv::Mat result_frame;
    cv::resize(cropped_frame, result_frame, size);

    return result_frame;
}

cv::Mat Model::inferenceFace(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int c = mModel->getInputBufferInfo()[0].original_channel;

    cv::Mat resized_frame;
    cv::Mat padded_resized_frame;
    prepare_yolo_input(frame, w, h, resized_frame, padded_resized_frame);

    std::vector<std::vector<float>> result;
    if (mInputType == InputDataType::FLOAT32) {
        auto input_img = std::make_unique<float[]>(w * h * c);
        constexpr float rev = 1.0f / 255.0f;
        float* ptr = input_img.get();
        for (int i = 0; i < w * h * c; i++) {
            ptr[i] = static_cast<float>(padded_resized_frame.data[i]) * rev;
        }
        npu_benchmarker.start();
        result = mModel->infer({input_img.get()}, sc);
        npu_benchmarker.end();
    } else {
        npu_benchmarker.start();
        result = mModel->infer({padded_resized_frame.data}, sc);
        npu_benchmarker.end();
    }
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

    cv::Mat result_frame;
    cv::resize(frame, result_frame, size);

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> landmarks;
    uint64_t ticket =
        mPost->enqueue(result_frame, result, boxes, scores, labels, landmarks);
    mPost->receive(ticket);

    return result_frame;
}

cv::Mat Model::inferencePose(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int c = mModel->getInputBufferInfo()[0].original_channel;

    cv::Mat resized_frame;
    cv::Mat padded_resized_frame;
    prepare_yolo_input(frame, w, h, resized_frame, padded_resized_frame);

    std::vector<std::vector<float>> result;
    if (mInputType == InputDataType::FLOAT32) {
        auto input_img = std::make_unique<float[]>(w * h * c);
        constexpr float rev = 1.0f / 255.0f;
        float* ptr = input_img.get();
        for (int i = 0; i < w * h; i++) {
            int idx = i * 3;
            // BGR -> RGB 배열 변환
            ptr[idx + 0] = static_cast<float>(padded_resized_frame.data[idx + 2]) * rev;
            ptr[idx + 1] = static_cast<float>(padded_resized_frame.data[idx + 1]) * rev;
            ptr[idx + 2] = static_cast<float>(padded_resized_frame.data[idx + 0]) * rev;
        }
        npu_benchmarker.start();
        result = mModel->infer({input_img.get()}, sc);
        npu_benchmarker.end();
    } else {
        npu_benchmarker.start();
        result = mModel->infer({padded_resized_frame.data}, sc);
        npu_benchmarker.end();
    }
    if (!sc) {
        std::cout << "infer failed" << std::endl;
        return cv::Mat::zeros(size, CV_8UC3);
    }

    cv::Mat result_frame;
    cv::resize(frame, result_frame, size);

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> keypoints;
    uint64_t ticket =
        mPost->enqueue(result_frame, result, boxes, scores, labels, keypoints);
    mPost->receive(ticket);

    return result_frame;
}

cv::Mat Model::inferenceObject(cv::Mat frame, cv::Size size,
                               Benchmarker& npu_benchmarker) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int c = mModel->getInputBufferInfo()[0].original_channel;

    cv::Mat resized_frame;
    cv::Mat padded_resized_frame;
    prepare_yolo_input(frame, w, h, resized_frame, padded_resized_frame);

    std::vector<std::vector<float>> result;
    if (mInputType == InputDataType::FLOAT32) {
        auto input_img = std::make_unique<float[]>(w * h * c);
        constexpr float rev = 1.0f / 255.0f;
        float* ptr = input_img.get();
        for (int i = 0; i < w * h; i++) {
            int idx = i * 3;
            ptr[idx + 0] = static_cast<float>(padded_resized_frame.data[idx + 2]) * rev;
            ptr[idx + 1] = static_cast<float>(padded_resized_frame.data[idx + 1]) * rev;
            ptr[idx + 2] = static_cast<float>(padded_resized_frame.data[idx + 0]) * rev;
        }
        npu_benchmarker.start();
        result = mModel->infer({input_img.get()}, sc);
        npu_benchmarker.end();
    } else {
        npu_benchmarker.start();
        result = mModel->infer({padded_resized_frame.data}, sc);
        npu_benchmarker.end();
    }

    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

    cv::Mat result_frame;
    cv::resize(frame, result_frame, size);

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> extras;
    uint64_t ticket = mPost->enqueue(result_frame, result, boxes, scores, labels, extras);
    mPost->receive(ticket);

    return result_frame;
}

cv::Mat Model::inferenceSeg(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker) {
    mobilint::StatusCode sc;

    int w = mModel->getInputBufferInfo()[0].original_width;
    int h = mModel->getInputBufferInfo()[0].original_height;
    int c = mModel->getInputBufferInfo()[0].original_channel;

    cv::Mat resized_frame;
    cv::Mat padded_resized_frame;
    prepare_yolo_input(frame, w, h, resized_frame, padded_resized_frame);

    std::vector<std::vector<float>> result;
    if (mInputType == InputDataType::FLOAT32) {
        auto input_img = std::make_unique<float[]>(w * h * c);
        constexpr float rev = 1.0f / 255.0f;
        float* ptr = input_img.get();
        for (int i = 0; i < w * h * c; i++) {
            ptr[i] = static_cast<float>(padded_resized_frame.data[i]) * rev;
        }
        npu_benchmarker.start();
        result = mModel->infer({input_img.get()}, sc);
        npu_benchmarker.end();
    } else {
        npu_benchmarker.start();
        result = mModel->infer({padded_resized_frame.data}, sc);
        npu_benchmarker.end();
    }
    if (!sc) {
        return cv::Mat::zeros(size, CV_8UC3);
    }

    cv::Mat result_frame;
    cv::resize(frame, result_frame, size);

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> extras;
    uint64_t ticket = mPost->enqueue(result_frame, result, boxes, scores, labels, extras);
    mPost->receive(ticket);

    return result_frame;
}
