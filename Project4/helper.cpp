/*
  Sri Harsha Gollamudi
  Mar 2023

  This file contains the helper functions required to support calibration and augmented reality.
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "helper.h"

using namespace cv;

/*
* This function helps extract the corners from the checkerboard and project the corners on the image.
*/
Mat extractChessCorners(Mat src, std::vector<cv::Point2f>& corners) {
    Mat dst = src.clone();

    cv::Size patternSize(9, 6);
    bool found = cv::findChessboardCorners(dst, patternSize, corners);

    if (found) {
        cv::drawChessboardCorners(dst, patternSize, corners, found);
    }
    else {
        corners.clear();
    }

    return dst;
}