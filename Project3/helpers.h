/*
  Sri Harsha Gollamudi
  Feb 2023

  This header file contains the functions an their signatures to be used in the Source.cpp file to run the application.
*/

#pragma once

#include <dirent.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <numeric>

using namespace cv;

int blur5x5(cv::Mat& src, cv::Mat& dst);

double findMedian(Mat img);

int greyscale(cv::Mat& src, cv::Mat& dst);

Mat thresholdImage(Mat src);

Mat cleanUpImage(Mat src, int times);

Mat erosion(Mat src);

Mat dilation(Mat src);

Mat segmentImage(Mat src);

Mat regionFeaturesImage(Mat src, Mat srcOrginal);

std::vector<double> regionFeatureVector(Mat src);

std::vector<std::pair<std::vector<double>, char>> trainData(char dir[]);

std::vector<std::pair<std::vector<double>, char>> testData(char dir[]);

std::vector<std::pair<double, char>> squaredDifference(std::vector<std::pair<std::vector<double>, char>> features, std::vector<double> imgFeature);

void topNMatches(std::vector<std::pair<double, char>> distances, int n);

Mat classifyImage(Mat src, std::vector<std::pair<double, char>> distances);

std::vector<std::pair<double, char>> KNN(std::vector<std::pair<std::vector<double>, char>> features, std::vector<double> imgFeature, int k);

void printConfusionMatrix(std::vector<std::pair<std::vector<double>, char>> features, std::vector<std::pair<std::vector<double>, char>> testingData, int k);

std::vector<std::pair<double, char>> manhattan(std::vector<std::pair<std::vector<double>, char>> features, std::vector<double> imgFeature);
