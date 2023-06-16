#ifndef KALMANFILTER_H
#define KALMANFILTER_H
#include "dataType.h"

class KalmanFilter
{
public:
    static const double chi2inv95[10];
    KalmanFilter();
    ~KalmanFilter();
    KAL_DATA initiate(const DETECTBOX& measurement);
    void predict(KAL_MEAN& mean, KAL_COVA& covariance);
    KAL_DATA update(const KAL_MEAN& mean,
                    const KAL_COVA& covariance,
                    const DETECTBOX& measurement);
    Eigen::Matrix<float, 1, -1> gating_distance(
            const KAL_MEAN& mean,
            const KAL_COVA& covariance,
            const std::vector<DETECTBOX>& measurements,
            bool only_position = false);

private:
    cv::KalmanFilter* opencv_kf;
    float _std_weight_position;
    float _std_weight_velocity;
};

#endif // KALMANFILTER_H
