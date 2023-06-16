#include "nn_matching.h"

using namespace Eigen;


NearNeighborDisMetric::NearNeighborDisMetric(NearNeighborDisMetric::METRIC_TYPE metric,
                                             float matching_threshold,
                                             int budget, int k_feature_dim) {
    // if (metric == euclidean) {
    //     _metric = &NearNeighborDisMetric::_nneuclidean_distance;
    // } else if (metric == cosine) {
    //     _metric = &NearNeighborDisMetric::_nncosine_distance;
    // }
    _metric = &NearNeighborDisMetric::_nncosine_distance;

    this->mating_threshold = matching_threshold;
    this->budget = budget;
    this->samples.clear();
    this->k_feature_dim = k_feature_dim;
}

DYNAMICM
NearNeighborDisMetric::distance(const FEATURESS& features, const std::vector<int>& targets) {
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(targets.size(), features.rows());
    int idx = 0;
    for (int target : targets) {
        cv::Mat x(this->samples[target].rows(), this->samples[target].cols(), CV_32F);
        cv::Mat y(features.rows(), features.cols(), CV_32F);
        for(int i = 0; i < x.rows; i++) {
            for(int j = 0; j < x.cols; j++) {
                x.at<float>(i, j) = this->samples[target](i, j);
            }
        }
        for(int i = 0; i < y.rows; i++) {
            for(int j = 0; j < y.cols; j++) {
                y.at<float>(i, j) = features(i, j);
            }
        }        
        cv::Mat vec = (this->*_metric)(x, y);
        for(int i = 0; i < cost_matrix.cols(); i++) {
            cost_matrix(idx, i) = vec.at<float>(i);
        }

        idx++;
    }
    return cost_matrix;
}

void NearNeighborDisMetric::partial_fit(std::vector<TRACKER_DATA>& tid_feats, std::vector<int>& active_targets) {
    /*python code:
     * let feature(target_id) append to samples;
     * && delete not comfirmed target_id from samples.
     * update samples;
     */

    for (TRACKER_DATA& data : tid_feats) {
        int track_id = data.first;
        FEATURESS newFeatOne = data.second;

        if (samples.find(track_id) != samples.end()) {  // append
            int oldSize = samples[track_id].rows();
            int addSize = newFeatOne.rows();
            int newSize = oldSize + addSize;

            if (newSize <= this->budget) {
                FEATURESS newSampleFeatures(newSize, k_feature_dim);
                newSampleFeatures.block(0, 0, oldSize, k_feature_dim) = samples[track_id];
                newSampleFeatures.block(oldSize, 0, addSize, k_feature_dim) = newFeatOne;
                samples[track_id] = newSampleFeatures;
            } else {
#if 1
                if (oldSize < this->budget) {  // original space is not enough;
                    FEATURESS newSampleFeatures(this->budget, k_feature_dim);
                    if (addSize >= this->budget) {
                        newSampleFeatures = newFeatOne.block(0, 0, this->budget, k_feature_dim);
                    } else {
                        newSampleFeatures.block(0, 0, this->budget - addSize, k_feature_dim) =
                            samples[track_id].block(addSize - 1, 0, this->budget - addSize, k_feature_dim).eval();
                        newSampleFeatures.block(this->budget - addSize, 0, addSize, k_feature_dim) = newFeatOne;
                    }
                    samples[track_id] = newSampleFeatures;
                } else {  // original space is ok;
                    if (addSize >= this->budget) {
                        samples[track_id] = newFeatOne.block(0, 0, this->budget, k_feature_dim);
                    } else {
                        samples[track_id].block(0, 0, this->budget - addSize, k_feature_dim) =
                            samples[track_id].block(addSize - 1, 0, this->budget - addSize, k_feature_dim).eval();
                        samples[track_id].block(this->budget - addSize, 0, addSize, k_feature_dim) = newFeatOne;
                    }
                }
#else
                if (oldSize < this->budget) {  // original space is not enough;
                    FEATURESS newSampleFeatures(this->budget, k_feature_dim);
                    if (addSize >= this->budget) {
                        newSampleFeatures = newFeatOne.block(0, 0, this->budget, k_feature_dim);
                    } else {
                        newSampleFeatures.block(0, 0, this->budget - addSize, k_feature_dim) =
                            samples[track_id]
                                .block(oldSize - (this->budget - addSize), 0, this->budget - addSize, k_feature_dim)
                                .eval();
                        newSampleFeatures.block(this->budget - addSize, 0, addSize, k_feature_dim) = newFeatOne;
                    }
                    samples[track_id] = newSampleFeatures;
                } else {  // original space is ok;
                    if (addSize >= this->budget) {
                        samples[track_id] = newFeatOne.block(0, 0, this->budget, k_feature_dim);
                    } else {
                        samples[track_id].block(0, 0, oldSize - addSize, k_feature_dim) =
                            samples[track_id].block(addSize, 0, oldSize - addSize, k_feature_dim).eval();
                        samples[track_id].block(oldSize - addSize, 0, addSize, k_feature_dim) = newFeatOne;
                    }
                }
#endif
            }
        } else {  // not exit, create new one;
            samples[track_id] = newFeatOne;
        }
    }  // add features;

    // erase the samples which not in active_targets;
    for (std::map<int, FEATURESS>::iterator i = samples.begin(); i != samples.end();) {
        bool flag = false;
        for (int j : active_targets)
            if (j == i->first) {
                flag = true;
                break;
            }
        if (flag == false)
            samples.erase(i++);
        else
            i++;
    }
}

cv::Mat NearNeighborDisMetric::_nncosine_distance(const cv::Mat& x, const cv::Mat& y) {
    cv::Mat distances = _cosine_distance(x, y);
    // 计算每一列的最小值
    cv::Mat minValues;
    cv::reduce(distances, minValues, 0, cv::REDUCE_MIN);

    cv::Mat res = minValues.t();
    return res;
}

cv::Mat NearNeighborDisMetric::_cosine_distance(const cv::Mat& x, const cv::Mat& y) {
    cv::Mat res;
    cv::gemm(x, y.t(), 1, cv::Mat(), 0, res);
    cv::subtract(cv::Scalar::all(1), res, res);
    return res;
}