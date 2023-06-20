#include "track.h"


Track::Track(KAL_MEAN& mean,
             KAL_COVA& covariance,
             int track_id,
             int class_id,
             int n_init,
             int max_age,
             const FEATURE& feature,
             int k_feature_dim) {
    this->mean = mean;
    this->covariance = covariance;
    this->track_id = track_id;
    this->class_id = class_id;
    this->hits = 1;
    this->age = 1;
    this->time_since_update = 0;
    this->state = TrackState::Tentative;
    features = FEATURESS(1, k_feature_dim);
    features.row(0).resize(k_feature_dim);
    features.row(0) = feature;  // features.rows() must = 0;
    this->k_feature_dim = k_feature_dim;
    this->_n_init = n_init;
    this->_max_age = max_age;
}

void Track::predit(KalmanFilter* kf) {
    /*Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        */
    cv::Mat _mean_(mean.rows(), mean.cols(), CV_32F);
    cv::Mat _covariance_(covariance.rows(), covariance.cols(), CV_32F);
     for(int i = 0; i < mean.rows(); i++) {
        for(int j = 0; j < mean.cols(); j++) {
            _mean_.at<float>(i, j) = mean(i, j);
        }
    }
    for(int i = 0; i < covariance.rows(); i++) {
        for(int j = 0; j < covariance.cols(); j++) {
            _covariance_.at<float>(i, j) = covariance(i, j);
        }
    }   
    auto pa = kf->predict(_mean_, _covariance_);

    cv::Mat _mean = pa.first;
    cv::Mat _covariance = pa.second;
    for(int i = 0; i < mean.rows(); i++) {
        for(int j = 0; j < mean.cols(); j++) {
            mean(i, j) = _mean.at<float>(i, j);
        }
    }
    for(int i = 0; i < covariance.rows(); i++) {
        for(int j = 0; j < covariance.cols(); j++) {
            covariance(i, j) = _covariance.at<float>(i, j);
        }
    }
    this->age += 1;
    this->time_since_update += 1;
}

void Track::update(KalmanFilter* const kf, const DETECTION_ROW& detection) {
    cv::Mat _mean_(mean.rows(), mean.cols(), CV_32F);
    cv::Mat _covariance_(covariance.rows(), covariance.cols(), CV_32F);
    cv::Mat box(detection.to_xyah().rows(), detection.to_xyah().cols(), CV_32F);
    for(int i = 0; i < mean.rows(); i++) {
        for(int j = 0; j < mean.cols(); j++) {
            _mean_.at<float>(i, j) = mean(i, j);
        }
    }
    for(int i = 0; i < covariance.rows(); i++) {
        for(int j = 0; j < covariance.cols(); j++) {
            _covariance_.at<float>(i, j) = covariance(i, j);
        }
    }
    for(int i = 0; i < detection.to_xyah().rows(); i++) {
        for(int j = 0; j < detection.to_xyah().cols(); j++) {
            box.at<float>(i, j) = detection.to_xyah()(i, j);
        }
    }

    auto pa = kf->update(_mean_, _covariance_, box);
    cv::Mat _mean = pa.first;
    cv::Mat _covariance = pa.second;
    for(int i = 0; i < mean.rows(); i++) {
        for(int j = 0; j < mean.cols(); j++) {
            mean(i, j) = _mean.at<float>(i, j);
        }
    }
    for(int i = 0; i < covariance.rows(); i++) {
        for(int j = 0; j < covariance.cols(); j++) {
            covariance(i, j) = _covariance.at<float>(i, j);
        }
    }
    

    featuresAppendOne(detection.feature);
    //    this->features.row(features.rows()) = detection.feature;
    this->hits += 1;
    this->time_since_update = 0;
    if (this->state == TrackState::Tentative && this->hits >= this->_n_init) {
        this->state = TrackState::Confirmed;
    }
}

void Track::mark_missed() {
    if (this->state == TrackState::Tentative) {
        this->state = TrackState::Deleted;
    } else if (this->time_since_update > this->_max_age) {
        this->state = TrackState::Deleted;
    }
}

bool Track::is_confirmed() {
    return this->state == TrackState::Confirmed;
}

bool Track::is_deleted() {
    return this->state == TrackState::Deleted;
}

bool Track::is_tentative() {
    return this->state == TrackState::Tentative;
}

DETECTBOX Track::to_tlwh() {
    DETECTBOX ret = mean.leftCols(4);
    ret(2) *= ret(3);
    ret.leftCols(2) -= (ret.rightCols(2) / 2);
    return ret;
}

void Track::featuresAppendOne(const FEATURE& f) {
    int size = this->features.rows();
    FEATURESS newfeatures = FEATURESS(size + 1, k_feature_dim);
    newfeatures.block(0, 0, size, k_feature_dim) = this->features;
    newfeatures.row(size) = f;
    features = newfeatures;
}
