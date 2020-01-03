#ifndef _IMAGE_DENOISING_H
#define _IMAGE_DENOISING_H

#include <armadillo>
#include <iostream>
#include <opencv2/opencv.hpp>

/***
@article{yin2019side,
  title={Side Window Filtering},
  author={Yin, Hui and Gong, Yuanhao and Qiu, Guoping},
  journal={arXiv: Computer Vision and Pattern Recognition},
  year={2019}
}

This is a reimplementation from https://github.com/YuanhaoGong/SideWindowFilter.
***/

void sideWindowBoxFilter(const cv::Mat& src, cv::Mat& dst, int radius, int iteration)
{
    int c = src.channels();
    int r = radius;

    int src_h = src.rows;
    int src_w = src.cols;
    int src_N = src_h * src_w;

    int pad_h = src_h + 2 * r;
    int pad_w = src_w + 2 * r;
    int pad_N = pad_h * pad_w;

    cv::Mat k = cv::Mat::ones(2 * r + 1, 1, CV_64F);
    cv::Mat k_L = k.clone();
    cv::Mat k_R = k.clone();

    k /= (2.0 * r + 1.0);
    k_L(cv::Range(r + 1, 2 * r + 1), cv::Range(0, 1)) = 0;
    k_L /= (r + 1.0);
    k_R(cv::Range(0, r), cv::Range(0, 1)) = 0;
    k_R /= (r + 1.0);

    cv::Mat I;
    src.convertTo(I, CV_64F, 1.0 / 255.0);
    std::vector<cv::Mat> I_channels;
    cv::split(I, I_channels);

    std::vector<cv::Mat> d(8);

    arma::mat D(pad_N, 8);

    for (int c_ = 0; c_ < c; c_++) {
        cv::Mat I_padded;
        cv::copyMakeBorder(I_channels[c_], I_padded, r, r, r, r, cv::BORDER_REPLICATE);

        for (int i = 0; i < iteration; i++) {
            cv::sepFilter2D(I_padded, d[0], CV_64F, k_L, k_L, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
            cv::sepFilter2D(I_padded, d[1], CV_64F, k_L, k_R, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
            cv::sepFilter2D(I_padded, d[2], CV_64F, k_R, k_L, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
            cv::sepFilter2D(I_padded, d[3], CV_64F, k_R, k_R, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
            cv::sepFilter2D(I_padded, d[4], CV_64F, k_L, k, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
            cv::sepFilter2D(I_padded, d[5], CV_64F, k_R, k, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
            cv::sepFilter2D(I_padded, d[6], CV_64F, k, k_L, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
            cv::sepFilter2D(I_padded, d[7], CV_64F, k, k_R, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);


            arma::mat U(reinterpret_cast<double*>(I_padded.data), pad_w, pad_h);
            U = arma::reshape(U, pad_N, 1);

            for (int j = 0; j < 8; j++) {
                arma::mat tmp(reinterpret_cast<double*>(d[j].data), pad_w, pad_h);
                D.col(j) = arma::reshape(tmp, pad_N, 1);
            }

            arma::ucolvec jj = arma::index_min(arma::abs(D.each_col() - U), 1);

            U = D.elem(jj * pad_N + arma::linspace<arma::ucolvec>(0, pad_N - 1, pad_N));

            I_padded = cv::Mat(pad_h, pad_w, CV_64F, U.memptr()).clone();
        }

        I_padded(cv::Range(r, src_h + r), cv::Range(r, src_w + r)).copyTo(I_channels[c_]);
    }

    cv::merge(I_channels, dst);
    dst.convertTo(dst, CV_8U, 255.0);

    return;
}

#endif