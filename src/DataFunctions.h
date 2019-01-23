#ifndef DATA_FUNCTIONS_H
#define DATA_FUNCTIONS_H

#include <RcppEigen.h>

template <typename matType>
void compute_moments(const matType & X,
                     const Eigen::Ref<const Eigen::VectorXd> & wgts_user,
                     Eigen::Ref<Eigen::VectorXd> xm,
                     Eigen::Ref<Eigen::VectorXd> xv,
                     Eigen::Ref<Eigen::VectorXd> xs,
                     const bool & center,
                     const bool & scale,
                     int idx) {
    if (center) {
        if (scale) {
            for (int j = 0; j < X.cols(); ++j, ++idx) {
                auto xj = X.col(j);
                xm[idx] = xj.cwiseProduct(wgts_user).sum();
                xs[idx] = 1 / std::sqrt(xj.cwiseProduct(xj.cwiseProduct(wgts_user)).sum() - xm[idx] * xm[idx]);
            }
        } else {
            for (int j = 0; j < X.cols(); ++j, ++idx) {
                auto xj = X.col(j);
                xm[idx] = xj.cwiseProduct(wgts_user).sum();
                xv[idx] = xj.cwiseProduct(xj.cwiseProduct(wgts_user)).sum() - xm[idx] * xm[idx];
            }
        }
    }
    else {
        if (scale) {
            for (int j = 0; j < X.cols(); ++j, ++idx) {
                auto xj = X.col(j);
                double xm_j = xj.cwiseProduct(wgts_user).sum();
                double vc = xj.cwiseProduct(xj.cwiseProduct(wgts_user)).sum() - xm_j * xm_j;
                xs[idx] = 1 / std::sqrt(vc);
                xv[idx] = 1.0 + xm_j * xm_j / vc;
            }
        } else {
            for (int j = 0; j < X.cols(); ++j, ++idx) {
                auto xj = X.col(j);
                xv[idx] = xj.cwiseProduct(xj.cwiseProduct(wgts_user)).sum();
            }
        }
    }
};

template <typename matA, typename matB>
Eigen::MatrixXd create_XZ(const matA & X,
                          const matB & Z,
                          const Eigen::Ref<const Eigen::VectorXd> & xm,
                          Eigen::Ref<Eigen::VectorXd> xv,
                          Eigen::Ref<Eigen::VectorXd> xs,
                          const bool & intr_ext,
                          const bool & scale_z,
                          int idx) {

    // initialize XZ matrix
    Eigen::MatrixXd XZ(0, 0);

    if (Z.size() == 0)
        return XZ;
    else
        XZ.resize(X.rows(), intr_ext + Z.cols());

    // map means and sds of X
    auto xm_X = xm.head(X.cols());
    auto xs_X = xs.head(X.cols());

    int col_xz = 0;

    // add intercept
    if (intr_ext) {
        XZ.col(col_xz) = (X * xs_X).array() - xs_X.cwiseProduct(xm_X).sum();
        xv[idx] = (XZ.col(col_xz).array() - XZ.col(col_xz).mean()).square().sum() / XZ.col(col_xz).size();
        ++idx;
        ++col_xz;
    }

    // fill in columns of XZ
    for (int j = 0; j < Z.cols(); ++j, ++col_xz, ++idx) {
        auto zj = Z.col(j);
        auto xzj = XZ.col(col_xz);
        if (scale_z) {
            double zm_j = zj.sum() / zj.size();
            xs[idx] = 1 / std::sqrt(zj.cwiseProduct(zj / zj.size()).sum() - zm_j * zm_j);
        }
        xzj = xs[idx] * ((X * zj.cwiseProduct(xs_X)) - xs_X.cwiseProduct(xm_X.cwiseProduct(zj)).sum() * Eigen::VectorXd::Ones(X.rows()));
        xv[idx] = (xzj.array() - xzj.mean()).square().sum() / xzj.size();
        xzj /= xs[idx];

    }
    return XZ;
};

#endif // DATA_FUNCTIONS_H
