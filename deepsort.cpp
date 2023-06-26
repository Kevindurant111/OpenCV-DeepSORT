//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "deepsort.h"
DeepSort::DeepSort(std::shared_ptr<BMNNContext> context,const deepsort_params& params) {
    std::cout << "deepsort ctor .." << std::endl;
    featureExtractor = new FeatureExtractor(context);
    featureExtractor->Init();
    objTracker = new tracker(params.max_dist, 
                             params.nn_budget, 
                             featureExtractor->k_feature_dim,
                             params.max_iou_distance,
                             params.max_age,
                             params.n_init);
}

DeepSort::~DeepSort() {
    delete objTracker;
    delete featureExtractor;
}

void DeepSort::sort(bm_image& frame, vector<YoloV5Box>& dets, vector<TrackBox>& track_boxs, int frame_id) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    for (YoloV5Box i : dets) {
        /*very important, if not this code, you will suffer from segmentation fault.*/
        auto start_x = MIN(MAX(int(i.x), 0), frame.width - 16);
        auto start_y = MIN(MAX(int(i.y), 0), frame.height - 16);
        auto crop_w = MAX(MIN(int(i.width), frame.width - int(i.x)), 16); // vpp resize support width >= 16 
        auto crop_h = MAX(MIN(int(i.height), frame.height - int(i.y)), 16); // vpp resize support height >= 16 

        cv::Mat box(1, 4, CV_32F);
        box.at<float>(0) = start_x;
        box.at<float>(1) = start_y;
        box.at<float>(2) = crop_w;
        box.at<float>(3) = crop_h;
        
        DETECTION_ROW d;
        d.tlwh = box.clone();
        d.confidence = i.score;
        d.class_id = i.class_id;
        detections.push_back(d);
    }
    track_boxs.clear();
    if (detections.size() > 0) {
        LOG_TS(m_ts, "extractor time");
        bool flag = featureExtractor->getRectsFeature(frame, detections);
        LOG_TS(m_ts, "extractor time");
        LOG_TS(m_ts, "deepsort postprocess");
        if (flag) {
            objTracker->predict();
            objTracker->update(detections);
            int idx = 1;
            for (Track& track : objTracker->tracks) {
                if ((!track.is_confirmed() || track.time_since_update > 1) && frame_id > 2) { //when frame_id < 2, there is no track.
                    continue;
                }
                auto k = track.to_tlwh();
                if(frame_id == 276) {
                    std::cout << "结果检查：" << std::endl;
                    std::cout << "--------------------" << idx << std::endl;
                    std::cout << k.at<float>(0) << " " << k.at<float>(1)<< " "<< k.at<float>(2)<< " "<< k.at<float>(3)<< " "<< track.class_id<< " "<< track.track_id<<std::endl;
                    std::string cmp = std::to_string(int(k.at<float>(0)))+std::to_string(int(k.at<float>(1)))+std::to_string(int(k.at<float>(2)))+std::to_string(int(k.at<float>(3)))+std::to_string(track.class_id)+std::to_string(track.track_id);
                    if(idx == 1) {
                        if(cmp != "111433619748702") {
                            std::cout << "比对错误！";
                            exit(1);
                        }
                    }
                    else if(idx == 2) {
                        if(cmp != "15634288727604") {
                            std::cout << "比对错误！";
                            exit(1);
                        }
                    }
                    else if(idx == 3) {
                        if(cmp != "1444386110318012") {
                            std::cout << "比对错误！";
                            exit(1);
                        }
                    }
                    else if(idx == 4) {
                        if(cmp != "11853780210031") {
                            std::cout << "比对错误！";
                            exit(1);
                        }
                    }
                    else if(idx == 5) {
                        if(cmp != "174153674184045") {
                            std::cout << "比对错误！";
                            exit(1);
                        }
                    }
                    else if(idx == 6) {
                        if(cmp != "184641879215046") {
                            std::cout << "比对错误！";
                            exit(1);
                        }
                    }
                    else if(idx == 7) {
                        if(cmp != "1260316164461054") {
                            std::cout << "比对错误！";
                            exit(1);
                        }
                    }
                    else if(idx == 8) {
                        if(cmp != "137743382262056") {
                            std::cout << "比对错误！";
                            exit(1);
                        }
                    }
                    else if(idx == 9) {
                        if(cmp != "979361155440062") {
                            std::cout << "比对错误！";
                            exit(1);
                        }
                    }
                    idx++;
                }
                TrackBox tmp(k.at<float>(0), k.at<float>(1), k.at<float>(2), k.at<float>(3), 1., track.class_id, track.track_id);
                track_boxs.push_back(tmp);
            }
        }
        LOG_TS(m_ts, "deepsort postprocess");
    }

}
