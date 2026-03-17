#include "demo/model.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "demo/post_ssd.h"
#include "demo/post_yolo_anchor_face.h"
#include "demo/post_yolo_anchorless.h"
#include "demo/post_yolo_anchorless_face.h"
#include "demo/post_yolo_anchorless_pose.h"
#include "demo/post_yolo_anchorless_seg.h"
#include "demo/post_yolo_dflfree.h"
#include "demo/post_yolo_dflfree_pose.h"
#include "demo/post_yolo_dflfree_seg.h"
#include "demo/post_yolo_nmsfree.h"
#include "opencv2/opencv.hpp"

Model::Model(const ModelSetting& model_setting, mobilint::Accelerator& acc)
    : mInputType(model_setting.input_type), mPostType(model_setting.post_type) {
    mobilint::StatusCode sc;
    mobilint::ModelConfig mc;

    if (model_setting.is_num_core) {
        mc.setSingleCoreMode(model_setting.num_core);
    } else {
        mc.setSingleCoreMode(model_setting.core_id);
    }

    mModel = mobilint::Model::create(model_setting.mxq_path, mc, sc);
    mModel->launch(acc);

    switch (model_setting.model_type) {
    case ModelType::SSD:
        initSSD();
        break;
    case ModelType::STYLENET:
        initStyle();
        break;
    case ModelType::FACE:
        initFace(mPostType);
        break;
    case ModelType::POSE:
        initPose(mPostType);
        break;
    case ModelType::OBJECT:
        initObject(mPostType);
        break;
    case ModelType::SEGMENTATION:
        initSeg(mPostType);
        break;
    }
}

Model::~Model() { mModel->dispose(); }

void Model::initSSD() {
    mPost = std::make_unique<SSDPostProcessor>();
    mInference = &Model::inferenceSSD;
}

void Model::initStyle() { mInference = &Model::inferenceStyle; }

void Model::initFace(PostProcessType post_type) {
    float face_conf_thres = 0.15;
    float face_iou_thres = 0.35;
    if (post_type == PostProcessType::ANCHOR) {
        int face_nl = 3;
        int face_nc = 1;
        int face_no = 16;
        int face_imh = 512;
        int face_imw = 640;
        mPost = std::make_unique<YOLOAnchorFacePost>(face_nl, face_nc, face_no, face_imh,
                                                     face_imw, face_conf_thres,
                                                     face_iou_thres, false);
        mInference = &Model::inferenceFace;
        return;
    }

    if (post_type == PostProcessType::ANCHORLESS) {
        int face_imh = mModel->getInputBufferInfo()[0].original_height;
        int face_imw = mModel->getInputBufferInfo()[0].original_width;
        mPost = std::make_unique<mobilint::post::YOLOAnchorlessFacePost>(
            face_imh, face_imw, face_conf_thres, face_iou_thres, false);
        mInference = &Model::inferenceFace;
        return;
    }

    throw std::invalid_argument(
        "FACE supports post=anchor or post=anchorless in this demo.");
}

void Model::initPose(PostProcessType post_type) {
    float pose_conf_thres = 0.25;
    float pose_iou_thres = 0.65;
    int pose_nc = 1;
    int pose_imh = mModel->getInputBufferInfo()[0].original_height;
    int pose_imw = mModel->getInputBufferInfo()[0].original_width;

    if (post_type == PostProcessType::ANCHORLESS) {
        mPost = std::make_unique<mobilint::post::YOLOAnchorlessPosePost>(
            pose_nc, pose_imh, pose_imw, pose_conf_thres, pose_iou_thres, false);
        mInference = &Model::inferencePose;
        return;
    }

    if (post_type == PostProcessType::DFLFREE) {
        int pose_imh = mModel->getInputBufferInfo()[0].original_height;
        int pose_imw = mModel->getInputBufferInfo()[0].original_width;
        mPost = std::make_unique<mobilint::post::YOLODFLFreePosePost>(
            pose_nc, pose_imh, pose_imw, pose_conf_thres, pose_iou_thres, false);
        mInference = &Model::inferencePose;
        return;
    }

    throw std::invalid_argument(
        "POSE supports post=anchorless or post=dflfree in this demo.");
}

void Model::initObject(PostProcessType post_type) {
    if (post_type == PostProcessType::DFLFREE) {
        mPost = std::make_unique<mobilint::post::YOLODFLFreePost>();
        mInference = &Model::inferenceObject;
        return;
    }

    float conf_thres = 0.25f;
    float iou_thres = 0.45f;
    int nc = 80;
    int imh = mModel->getInputBufferInfo()[0].original_height;
    int imw = mModel->getInputBufferInfo()[0].original_width;
    if (post_type == PostProcessType::ANCHORLESS) {
        mPost = std::make_unique<mobilint::post::YOLOAnchorlessPost>(
            nc, imh, imw, conf_thres, iou_thres, false);
        mInference = &Model::inferenceObject;
        return;
    }

    if (post_type == PostProcessType::NMSFREE) {
        mPost = std::make_unique<mobilint::post::YOLONMSFreePost>(
            nc, imh, imw, conf_thres, iou_thres, false);
        mInference = &Model::inferenceObject;
        return;
    }

    throw std::invalid_argument(
        "OBJECT supports post=anchorless, post=dflfree, or post=nmsfree in this demo.");
}

void Model::initSeg(PostProcessType post_type) {
    float seg_conf_thres = 0.20;
    float seg_iou_thres = 0.55;
    int seg_nc = 80;
    int seg_imh = mModel->getInputBufferInfo()[0].original_height;
    int seg_imw = mModel->getInputBufferInfo()[0].original_width;

    if (post_type == PostProcessType::ANCHORLESS) {
        mPost = std::make_unique<mobilint::post::YOLOAnchorlessSegPost>(
            seg_nc, seg_imh, seg_imw, seg_conf_thres, seg_iou_thres, false);
        mInference = &Model::inferenceSeg;
        return;
    }

    if (post_type == PostProcessType::DFLFREE) {
        int seg_imh = mModel->getInputBufferInfo()[0].original_height;
        int seg_imw = mModel->getInputBufferInfo()[0].original_width;
        mPost = std::make_unique<mobilint::post::YOLODFLFreeSegPost>(
            seg_nc, seg_imh, seg_imw, seg_conf_thres, seg_iou_thres, false);
        mInference = &Model::inferenceSeg;
        return;
    }

    throw std::invalid_argument(
        "SEGMENTATION supports post=anchorless or post=dflfree in this demo.");
}

void Model::work(Model* model, int worker_index, SizeState* size_state,
                 ItemQueue* item_queue, MatBuffer* feeder_buffer) {
    Benchmarker total_benchmarker;
    Benchmarker npu_benchmarker;

    cv::Mat frame, result;
    cv::Size result_size;

    int64_t frame_index = 0;

    // 첫 번째 프레임을 받고 시작해야한다.
    auto msc = feeder_buffer->get(frame, frame_index);
    if (msc != MatBuffer::StatusCode::OK) {
        item_queue->push({worker_index, cv::Mat()});
        return;
    }
    while (true) {
        total_benchmarker.start();
        // workerReceive 함수에서 Mat()를 받으면 worker가 죽은 것으로 간주하고 화면을
        // clear한다.
        auto ssc = size_state->checkUpdate(result_size);
        if (ssc != SizeState::StatusCode::OK) {
            item_queue->push({worker_index, cv::Mat()});
            break;
        }

        bool next_frame_exits = false;
        auto msc = feeder_buffer->peek(frame_index, next_frame_exits);
        if (msc != MatBuffer::StatusCode::OK) {
            item_queue->push({worker_index, cv::Mat()});
            break;
        }

        if (next_frame_exits) {
            auto msc = feeder_buffer->get(frame, frame_index);
            if (msc != MatBuffer::StatusCode::OK) {
                item_queue->push({worker_index, cv::Mat()});
                break;
            }
        }

#ifdef USE_SLEEP_DRIVER
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cv::resize(frame, result, result_size);
#else
        result = model->inference(frame, result_size, npu_benchmarker);
#endif

        total_benchmarker.end();
        if (next_frame_exits) {
            item_queue->push({worker_index, result, total_benchmarker.getFPS(),
                              npu_benchmarker.getFPS(),
                              total_benchmarker.getTimeSinceCreated(),
                              total_benchmarker.getCount()});
        }
    }
}

cv::Mat Model::inference(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker) {
    return (this->*mInference)(frame, size, npu_benchmarker);
}
