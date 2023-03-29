/*
  Sri Harsha Gollamudi

  This header file contains the functions an their signatures to be used in the Source.cpp file to run the application.
*/

#pragma once
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>

using namespace cv;

std::vector<cv::Mat> readImages(char dir[]);

std::vector<double> baselineMatching(Mat img);

std::vector<std::pair<std::vector<double>, char*>> baselineMatchingDirectory(char dir[]);

std::vector<std::pair<double, char*>> squaredDifference(std::vector<std::pair<std::vector<double>, char*>> features, std::vector<double> imgFeature);

void topNMatches(std::vector<std::pair<double, char*>> distances, int n);

std::vector<double> generateHistogram(Mat img, int bins);

std::vector<std::pair<std::vector<double>, char*>> histogramMatchingDirectory(char dir[], int bins);

std::vector<std::pair<double, char*>> intersection(std::vector<std::pair<std::vector<double>, char*>> features, std::vector<double> imgFeature);

std::vector<std::pair<double, char*>> weightedIntersection(std::vector<std::pair<std::vector<std::vector<double>>, char*>> features, std::vector<std::vector<double>> imgFeature);

std::vector<std::pair<std::vector<std::vector<double>>, char*>> multiHistogramMatchingDirectory(char dir[], int bins);

std::vector<std::vector<double>> generateQuadrantsHistograms(Mat img, int bins);

std::vector<double> generateMagnitudeHistogram(Mat img, int bins);

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);

int sobelY3x3(cv::Mat& src, cv::Mat& dst);

int sobelX3x3(cv::Mat& src, cv::Mat& dst);

std::vector<double> generateOrientationHistogram(Mat img, int bins);

std::vector<std::pair<std::vector<std::vector<double>>, char*>> textureMatchingDirectory(char dir[], int bins);

std::vector<std::vector<double>> textureMatching(Mat img, int bins);

std::vector<std::pair<double, char*>> CBIR(Mat img, char dir[], int bins);

std::vector<double> generateLawsFilterHistogram(Mat img, int bins);

std::vector<std::vector<double>> textureMatching2(Mat img, int bins);

std::vector<std::pair<std::vector<std::vector<double>>, char*>> textureMatchingDirectory2(char dir[], int bins);

std::vector<std::pair<double, char*>> CBIR2(Mat img, char dir[], int bins);



