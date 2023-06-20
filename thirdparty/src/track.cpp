#include "track.h"


Track::Track(const cv::Mat& mean,
             const cv::Mat& covariance,
             int track_id,
             int class_id,
             int n_init,
             int max_age,
             const FEATURE& feature,
             int k_feature_dim) {
    this->mean = mean.clone();
    this->covariance = covariance.clone();
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
    auto pa = kf->predict(mean, covariance);

    mean = pa.first.clone();
    covariance = pa.second.clone();

    this->age += 1;
    this->time_since_update += 1;
}

void Track::update(KalmanFilter* const kf, const DETECTION_ROW& detection) {
    // 改
    auto pa = kf->update(mean, covariance, detection.to_xyah());

    mean = pa.first.clone();
    covariance = pa.second.clone();    

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
    KAL_MEAN mean_;
    for(int i = 0; i < mean_.rows(); i++) {
        for(int j = 0; j < mean_.cols(); j++) {
            mean_(i, j) = mean.at<float>(i, j);
        }
    }
    DETECTBOX ret = mean_.leftCols(4);
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
