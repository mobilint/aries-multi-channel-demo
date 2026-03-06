#ifndef DEMO_INCLUDE_MODEL_H_
#define DEMO_INCLUDE_MODEL_H_

#include "demo/benchmarker.h"
#include "demo/define.h"
#include "opencv2/opencv.hpp"
#include "post.h"
#include "qbruntime/qbruntime.h"

class Model {
public:
    Model() = delete;
    Model(const ModelSetting& model_setting, mobilint::Accelerator& acc);
    ~Model();

    static void work(Model* model, int worker_index, SizeState* size_state,
                     ItemQueue* item_queue, MatBuffer* feeder_buffer);

    cv::Mat inference(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker);

private:
    cv::Mat (Model::*mInference)(cv::Mat, cv::Size, Benchmarker&);

    std::unique_ptr<mobilint::Model> mModel;
    std::unique_ptr<PostBase> mPost;
    InputDataType mInputType;
    PostProcessType mPostType;

    void initSSD();
    void initStyle();
    void initFace(PostProcessType post_type);
    void initPose(PostProcessType post_type);
    void initObject(PostProcessType post_type);
    void initSeg(PostProcessType post_type);

    cv::Mat inferenceSSD(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker);
    cv::Mat inferenceStyle(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker);
    cv::Mat inferenceFace(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker);
    cv::Mat inferencePose(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker);
    cv::Mat inferenceObject(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker);
    cv::Mat inferenceSeg(cv::Mat frame, cv::Size size, Benchmarker& npu_benchmarker);
};
#endif
