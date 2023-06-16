#include "kalmanfilter.h"
#include <Eigen/Cholesky>

void Cholesky(const cv::Mat& A, cv::Mat& S) {
    S = A.clone();
    cv::Cholesky ((float*)S.ptr(), S.step, S.rows, NULL, 0, 0);
    S = S.t();
    for (int i = 1; i < S.rows; i++) {
        for (int j = 0; j < i; j++) {
            S.at<float>(i, j)=0;
        }
    }
}

//sisyphus
const double KalmanFilter::chi2inv95[10] = {
    0,
    3.8415,
    5.9915,
    7.8147,
    9.4877,
    11.070,
    12.592,
    14.067,
    15.507,
    16.919};

KalmanFilter::KalmanFilter() {
    this->_std_weight_position = 1. / 20;
    this->_std_weight_velocity = 1. / 160;

    opencv_kf = new cv::KalmanFilter(8, 4);
    // 设置状态转移矩阵
    opencv_kf->transitionMatrix = (cv::Mat_<float>(8, 8) << 1, 0, 0, 0, 1, 0, 0, 0,
                                                            0, 1, 0, 0, 0, 1, 0, 0,
                                                            0, 0, 1, 0, 0, 0, 1, 0,
                                                            0, 0, 0, 1, 0, 0, 0, 1,
                                                            0, 0, 0, 0, 1, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 1, 0, 0,
                                                            0, 0, 0, 0, 0, 0, 1, 0,
                                                            0, 0, 0, 0, 0, 0, 0, 1);

    // 设置测量矩阵
    opencv_kf->measurementMatrix = (cv::Mat_<float>(4, 8) << 1, 0, 0, 0, 0, 0, 0, 0,
                                                            0, 1, 0, 0, 0, 0, 0, 0,
                                                            0, 0, 1, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 1, 0, 0, 0, 0);
}

KalmanFilter::~KalmanFilter() {
    delete opencv_kf;
}

KAL_DATA KalmanFilter::initiate(const DETECTBOX &measurement) {
    DETECTBOX mean_pos = measurement;
    DETECTBOX mean_vel;
    for (int i = 0; i < 4; i++)
        mean_vel(i) = 0;

    KAL_MEAN mean;
    for (int i = 0; i < 8; i++)
    {
        if (i < 4)
            mean(i) = mean_pos(i);
        else
            mean(i) = mean_vel(i - 4);
    }

    KAL_MEAN std;
    std(0) = 2 * _std_weight_position * measurement[3];
    std(1) = 2 * _std_weight_position * measurement[3];
    std(2) = 1e-2;
    std(3) = 2 * _std_weight_position * measurement[3];
    std(4) = 10 * _std_weight_velocity * measurement[3];
    std(5) = 10 * _std_weight_velocity * measurement[3];
    std(6) = 1e-5;
    std(7) = 10 * _std_weight_velocity * measurement[3];

    KAL_MEAN tmp = std.array().square();
    KAL_COVA var = tmp.asDiagonal();
    return std::make_pair(mean, var);
}

void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance) {
    float std_pos = _std_weight_position * mean(3) * _std_weight_position * mean(3);
    float std_vel = _std_weight_velocity * mean(3) * _std_weight_velocity * mean(3);
    opencv_kf->processNoiseCov = (cv::Mat_<float>(8, 8) << std_pos, 0, 0, 0, 0, 0, 0, 0,
                                                            0, std_pos, 0, 0, 0, 0, 0, 0,
                                                            0, 0, 1e-4, 0, 0, 0, 0, 0,
                                                            0, 0, 0, std_pos, 0, 0, 0, 0,
                                                            0, 0, 0, 0, std_vel, 0, 0, 0,
                                                            0, 0, 0, 0, 0, std_vel, 0, 0,
                                                            0, 0, 0, 0, 0, 0, 1e-10, 0,
                                                            0, 0, 0, 0, 0, 0, 0, std_vel);
    for (int i = 0; i < opencv_kf->statePost.rows; i++) {
        opencv_kf->statePost.at<float>(i, 0) = mean(0, i); //shape of statePost: (DP, 1);
    }
    for (int i = 0; i < opencv_kf->errorCovPost.rows; i++) {
        for(int j = 0; j < opencv_kf->errorCovPost.cols; j++) {
            opencv_kf->errorCovPost.at<float>(i, j) = covariance(i, j);
        }
    }

    opencv_kf->predict();

    for (int i = 0; i < opencv_kf->statePost.rows; i++) {
        mean(0, i) = opencv_kf->statePost.at<float>(i, 0); //shape of statePost: (DP, 1);
    }
    for (int i = 0; i < opencv_kf->errorCovPost.rows; i++) {
        for(int j = 0; j < opencv_kf->errorCovPost.cols; j++) {
            covariance(i, j) = opencv_kf->errorCovPost.at<float>(i, j);
        }
    }
}

KAL_DATA
KalmanFilter::update(
    const KAL_MEAN &mean,
    const KAL_COVA &covariance,
    const DETECTBOX &measurement) {
    cv::Mat measurement_(4, 1, CV_32F);
    // 将Eigen矩阵的数据复制到cv::Mat
    for (int i = 0; i < measurement_.rows; i++) {
        measurement_.at<float>(i, 0) = measurement(0, i);
    }
    for (int i = 0; i < opencv_kf->statePre.rows; i++) {
        opencv_kf->statePre.at<float>(i, 0) = mean(0, i);
    }
    for (int i = 0; i < opencv_kf->errorCovPre.rows; i++) {
        for(int j = 0; j < opencv_kf->errorCovPre.cols; j++) {
            opencv_kf->errorCovPre.at<float>(i, j) = covariance(i, j);
        }
    }
    float std_pos = _std_weight_position * mean(3) * _std_weight_position * mean(3);
    opencv_kf->measurementNoiseCov = (cv::Mat_<float>(4, 4) << std_pos, 0, 0, 0,
                                                            0, std_pos, 0, 0,
                                                            0, 0, 1e-2, 0,
                                                            0, 0, 0, std_pos);

    opencv_kf->correct(measurement_);

    KAL_MEAN new_mean;
    KAL_COVA new_covariance;
    for (int i = 0; i < opencv_kf->statePost.rows; i++) {
        new_mean(0, i) = opencv_kf->statePost.at<float>(i, 0);
    }
    for (int i = 0; i < opencv_kf->errorCovPost.rows; i++) {
        for(int j = 0; j < opencv_kf->errorCovPost.cols; j++) {
            new_covariance(i, j) = opencv_kf->errorCovPost.at<float>(i, j);
        }
    }

    return std::make_pair(new_mean, new_covariance);
}

Eigen::Matrix<float, 1, -1>
KalmanFilter::gating_distance(
    const KAL_MEAN &mean,
    const KAL_COVA &covariance,
    const std::vector<DETECTBOX> &measurements,
    bool only_position) {
    if (only_position) {
        printf("not implement!");
        exit(0);
    }

    cv::Mat std(1, 4, CV_32F);
    std.at<float>(0) = _std_weight_position * mean(3);
    std.at<float>(1) = _std_weight_position * mean(3);
    std.at<float>(2) = 1e-1;
    std.at<float>(3) = _std_weight_position * mean(3);
    cv::Mat _mean(mean.rows(), mean.cols(), CV_32F);
    for(int i = 0; i < mean.rows(); i++) {
        for(int j = 0; j < mean.cols(); j++) {
            _mean.at<float>(i, j) = mean(i, j);
        }
    }
    cv::Mat _covariance(covariance.rows(), covariance.cols(), CV_32F);
    for(int i = 0; i < covariance.rows(); i++) {
        for(int j = 0; j < covariance.cols(); j++) {
            _covariance.at<float>(i, j) = covariance(i, j);
        }
    }
    cv::Mat mean1 = opencv_kf->measurementMatrix * _mean.t();
    cv::Mat covariance1 = opencv_kf->measurementMatrix * _covariance * opencv_kf->measurementMatrix.t();

    cv::Mat diag = cv::Mat::zeros(4, 4, CV_32F);
    diag.at<float>(0, 0) = std.at<float>(0) * std.at<float>(0);
    diag.at<float>(1, 1) = std.at<float>(1) * std.at<float>(1);
    diag.at<float>(2, 2) = std.at<float>(2) * std.at<float>(2);
    diag.at<float>(3, 3) = std.at<float>(3) * std.at<float>(3);

    covariance1 += diag;

    cv::Mat d(measurements.size(), 4, CV_32F);
    int pos = 0;
    for (const auto& box : measurements)
    {
        cv::Mat _box(box.rows(), box.cols(), CV_32F);
        for(int i = 0; i < box.rows(); i++) {
            for(int j = 0; j < box.cols(); j++) {
                _box.at<float>(i, j) = box(i, j);
            }
        }
        cv::Mat diff = _box - mean1.t();
        diff.copyTo(d.row(pos++));
    }

    cv::Mat cvCovariance = covariance1;
    cv::Mat factor;
    Cholesky(cvCovariance, factor);

    cv::Mat cvD = d;
    cv::Mat cvZ = factor.inv(cv::DECOMP_CHOLESKY) * cvD.t();
    cv::Mat cvZZ = cvZ.mul(cvZ);
    cv::Mat cvSquareMaha = cv::Mat::zeros(1, cvZZ.cols, CV_32F);
    cv::reduce(cvZZ, cvSquareMaha, 0, cv::REDUCE_SUM);

    Eigen::Matrix<float, 1, -1> square_maha(1, cvSquareMaha.cols);
    for (int i = 0; i < cvSquareMaha.cols; ++i)
    {
        square_maha(0, i) = cvSquareMaha.at<float>(0, i);
    }

    return square_maha;
}
