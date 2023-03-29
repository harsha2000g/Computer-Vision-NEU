/*
  Sri Harsha Gollamudi
  Mar 2023

  This header file contains the functions and their signatures to be used in the Main.cpp file to run the application.
*/

#pragma once
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat extractChessCorners(Mat src, std::vector<cv::Point2f>& corners);



