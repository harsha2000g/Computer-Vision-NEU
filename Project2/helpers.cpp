/*
  Sri Harsha Gollamudi

  This file contains the helper functions required to read images, extract feature vectors from them, 
  calculate the distances and list the top N matches for an image.
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "helpers.h"

using namespace cv;

/*
* Reads the images from a directory and returns a list of images in Mat format.
 */
std::vector<cv::Mat> readImages(char dir[]) {

    char dirname[256];
    char buffer[256];
    FILE* fp;
    DIR* dirp;
    struct dirent* dp;

    // get the directory path
    strcpy_s(dirname, dir);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        exit(-1);
    }

    std::vector<cv::Mat> imgs;

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // build the overall filename
            strcpy_s(buffer, dirname);
            strcat_s(buffer, "/");
            strcat_s(buffer, dp->d_name);

            //printf("full path name: %s\n", buffer);

            imgs.push_back(imread(buffer));

        }
    }

    return imgs;

}

/*
    This function extracts the 9x9 matrix from the center for all three color channels and converts it into a vector.
*/

std::vector<double> baselineMatching(Mat img) {

    int row = img.rows;
    int col = img.cols;
    std::vector<double> feature;
    int startRow = row / 2 - 4;
    int endRow = row / 2 + 5;
    int startCol = col / 2 - 4;
    int endCol = col / 2 + 5;
    
    for (int row = startRow; row < startRow + 9; row++) {
        for (int col = startCol; col < startCol + 9; col++) {
            feature.push_back(img.at<Vec3b>(row, col)[0]);
            feature.push_back(img.at<Vec3b>(row, col)[1]);
            feature.push_back(img.at<Vec3b>(row, col)[2]);
        }
    }

    return feature;

}

/*
    This function applies baselineMatching on all the images in a given directory.
*/

std::vector<std::pair<std::vector<double>, char*>> baselineMatchingDirectory(char dir[]) {

    char dirname[256];
    FILE* fp;
    DIR* dirp;
    struct dirent* dp;
    std::vector<double> feature;
    std::vector<std::pair<std::vector<double>, char*>> features;

    // get the directory path
    strcpy_s(dirname, dir);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // dynamically allocate memory for the buffer
            char* buffer = new char[256];
            memset(buffer, 0, 256);

            // build the overall filename
            strcpy_s(buffer, 256, dirname);
            strcat_s(buffer, 256, "/");
            strcat_s(buffer, 256, dp->d_name);

            //printf("full path name: %s\n", buffer);

            feature = baselineMatching(imread(buffer));

            features.push_back(std::make_pair(feature, buffer));

        }
    }

    return features;

}

/*
    This function calculates the sum of squared difference between the features of the target image and images in the directory.
*/

std::vector<std::pair<double, char*>> squaredDifference(std::vector<std::pair<std::vector<double>, char*>> features, std::vector<double> imgFeature) {

    std::vector<std::pair<double, char*>> distances;
    double distance;
    double difference;

    for (int i = 0; i < features.size(); i++) {

        distance = 0;

        for (int j = 0; j < imgFeature.size(); j++) {

            difference = abs(imgFeature[j] - features[i].first[j]);
            distance += (difference * difference);

        }

        distances.push_back(std::make_pair(distance, features[i].second));
    }

    std::sort(distances.begin(), distances.end());

    return distances;

}

/*
    This function prints the Top N matches for the target image by taking the distances as input.
*/

void topNMatches(std::vector<std::pair<double, char*>> distances, int n) {

    //std::vector<char*> topMatches;

    printf("The top %d matches are\n", n);

    for (int i = 1; i < n + 1; i++) {

        printf("%s\n", distances[i].second);

    }

}

/*
    This function generates the histogram of an image given the bins and also normalizes the histogram.
*/

std::vector<double> generateHistogram(Mat img, int bins) {
   
    int divisor = 256 / bins;

    int i, j, k;

    int dim[3] = { bins, bins, bins };

    Mat histogram = Mat::zeros(3, dim, CV_32S);

    for (int row = 0; row < img.rows; row++) {

        Vec3b* rowptr = img.ptr<Vec3b>(row);

        for (int col = 0; col < img.cols; col++) {
            
            int r = rowptr[col][2] / divisor;
            int g = rowptr[col][1] / divisor;
            int b = rowptr[col][0] / divisor;
            histogram.at<int>(r, b, g)++;
        }
    }

    int sum = 0;

    for (int b = 0; b < bins; b++) {
        for (int g = 0; g < bins; g++) {
            for (int r = 0; r < bins; r++) {
                sum += histogram.at<int>(b, g, r);
            }
        }
    }

    std::vector<double> flattenedHistogram;

    for (int b = 0; b < bins; b++) {
        for (int g = 0; g < bins; g++) {
            for (int r = 0; r < bins; r++) {
                flattenedHistogram.push_back((double)histogram.at<int>(b, g, r)/sum);
            }
        }
    }

    return flattenedHistogram;

}

/*
    This function applies generateHistogram on all the images in a given directory.
*/

std::vector<std::pair<std::vector<double>, char*>> histogramMatchingDirectory(char dir[], int bins) {

    char dirname[256];
    FILE* fp;
    DIR* dirp;
    struct dirent* dp;
    std::vector<double> feature;
    std::vector<std::pair<std::vector<double>, char*>> features;

    // get the directory path
    strcpy_s(dirname, dir);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // dynamically allocate memory for the buffer
            char* buffer = new char[256];
            memset(buffer, 0, 256);

            // build the overall filename
            strcpy_s(buffer, 256, dirname);
            strcat_s(buffer, 256, "/");
            strcat_s(buffer, 256, dp->d_name);

            //printf("full path name: %s\n", buffer);

            feature = generateHistogram(imread(buffer), bins);

            features.push_back(std::make_pair(feature, buffer));

        }
    }

    return features;

}

/*
    This function calculates the intersection between the histogram features of the target image and images in the directory.
*/

std::vector<std::pair<double, char*>> intersection(std::vector<std::pair<std::vector<double>, char*>> features, std::vector<double> imgFeature) {

    std::vector<std::pair<double, char*>> intersections;
    double intersection;

    for (int i = 0; i < features.size(); i++) {

        intersection = 0.0;

        for (int j = 0; j < imgFeature.size(); j++) {

            intersection += imgFeature[j] < features[i].first[j] ? imgFeature[j] : features[i].first[j];

        }

        intersection = 1 - intersection;

        intersections.push_back(std::make_pair(intersection, features[i].second));
    }

    std::sort(intersections.begin(), intersections.end());

    return intersections;

}

//Overlapping quadrants
std::vector<std::vector<double>> generateQuadrantsHistograms(Mat img, int bins) {

    std::vector<std::vector<double>> result;

    Mat topRight = img(cv::Range(0, 3 * img.rows / 4), cv::Range(img.cols / 4, img.cols));
    Mat bottomRight = img(cv::Range(img.rows / 4, img.rows), cv::Range(img.cols / 4, img.cols));
    Mat topLeft = img(cv::Range(0, 3 * img.rows / 4), cv::Range(0, 3 * img.cols / 4));
    Mat bottomLeft = img(cv::Range(img.rows / 4, img.rows), cv::Range(0, 3 * img.cols / 4));
    
    result.push_back(generateHistogram(topRight, 8));
    result.push_back(generateHistogram(bottomRight, 8));
    result.push_back(generateHistogram(topLeft, 8));
    result.push_back(generateHistogram(bottomLeft, 8));

    return result;

}

/*
    This function applies generateQuadrantsHistograms on all the images in a given directory.
*/

std::vector<std::pair<std::vector<std::vector<double>>, char*>> multiHistogramMatchingDirectory(char dir[], int bins) {

    char dirname[256];
    FILE* fp;
    DIR* dirp;
    struct dirent* dp;
    std::vector<std::vector<double>> feature;
    std::vector<std::pair<std::vector<std::vector<double>>, char*>> features;

    // get the directory path
    strcpy_s(dirname, dir);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // dynamically allocate memory for the buffer
            char* buffer = new char[256];
            memset(buffer, 0, 256);

            // build the overall filename
            strcpy_s(buffer, 256, dirname);
            strcat_s(buffer, 256, "/");
            strcat_s(buffer, 256, dp->d_name);

            //printf("full path name: %s\n", buffer);

            feature = generateQuadrantsHistograms(imread(buffer), bins);

            features.push_back(std::make_pair(feature, buffer));

        }
    }

    return features;

}

/*
    This function calculates the weighted intersection between the features of the target image and images in the directory.
    It handles multiple histograms as feature vector and takes average of all the histograms.
*/
std::vector<std::pair<double, char*>> weightedIntersection(std::vector<std::pair<std::vector<std::vector<double>>, char*>> features, std::vector<std::vector<double>> imgFeature) {

    std::vector<std::pair<double, char*>> intersections;
    double intersection;
    double weightedIntersection;

    for (int i = 0; i < features.size(); i++) {

        weightedIntersection = 0.0;

        for (int k = 0; k < features[i].first.size(); k++) {

            intersection = 0.0;

            for (int j = 0; j < imgFeature[k].size(); j++) {

                intersection += imgFeature[k][j] < features[i].first[k][j] ? imgFeature[k][j] : features[i].first[k][j];

            }

            weightedIntersection += (1 - intersection);

        }

        weightedIntersection = weightedIntersection / 4;

        intersections.push_back(std::make_pair(weightedIntersection, features[i].second));

    }

    std::sort(intersections.begin(), intersections.end());

    return intersections;

}

/*
    This function applies Sobel X filter on an image.
*/

int sobelX3x3(cv::Mat& src, cv::Mat& dst)
{
    // Initializing the temporary and destination image
    cv::Mat mid;
    mid.create(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    // 1x3 Vertical and Horizontal filters
    int verticalFilter[5] = { 1,2,1 };
    int horizontalFilter[5] = { 1,0,-1 };

    // Vertical filter
    for (int row = 0; row < src.rows; row++) {
        for (int col = 1; col < src.cols - 1; col++) {
            for (int clr = 0; clr < 3; clr++) {
                int res = 0;
                for (int i = -1; i <= 1; i++) {
                    // Dot product of the filter and pixel values 
                    res += src.at<cv::Vec3b>(row, col + i)[clr] * verticalFilter[i + 1];
                }
                // Normalizing the result and assigning it to the temporary pixel values.
                mid.at<cv::Vec3s>(row, col)[clr] = cv::saturate_cast<short>(res / 3);
            }
        }
    }

    // Horizontal filter
    for (int row = 1; row < src.rows - 1; row++) {
        for (int col = 0; col < src.cols; col++) {
            for (int clr = 0; clr < 3; clr++) {
                int res = 0;
                for (int i = -1; i <= 1; i++) {
                    // Dot product of the filter and pixel values 
                    res += mid.at<cv::Vec3s>(row + i, col)[clr] * horizontalFilter[i + 1];
                }
                // Normalizing the result and assigning it to the dst pixel values.
                dst.at<cv::Vec3s>(row, col)[clr] = cv::saturate_cast<short>(res / 3);
            }
        }
    }

    return 0;
}

/*
    This function applies Sobel Y filter on an image.
*/

int sobelY3x3(cv::Mat& src, cv::Mat& dst)
{
    // Initializing the temporary and destination image
    cv::Mat mid;
    mid.create(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    // 1x3 Vertical and Horizontal filters
    int horizontalFilter[5] = { 1,2,1 };
    int verticalFilter[5] = { 1,0,-1 };

    // Vertical filter
    for (int row = 0; row < src.rows; row++) {
        for (int col = 1; col < src.cols - 1; col++) {
            for (int clr = 0; clr < 3; clr++) {
                int res = 0;
                for (int i = -1; i <= 1; i++) {
                    // Dot product of the filter and pixel values 
                    res += src.at<cv::Vec3b>(row, col + i)[clr] * verticalFilter[i + 1];
                }
                // Normalizing the result and assigning it to the dst pixel values.
                mid.at<cv::Vec3s>(row, col)[clr] = cv::saturate_cast<short>(res / 2);
            }
        }
    }

    for (int row = 1; row < src.rows - 1; row++) {
        for (int col = 0; col < src.cols; col++) {
            for (int clr = 0; clr < 3; clr++) {
                int res = 0;
                for (int i = -1; i <= 1; i++) {
                    // Dot product of the filter and pixel values 
                    res += mid.at<cv::Vec3s>(row + i, col)[clr] * horizontalFilter[i + 1];
                }
                // Normalizing the result and assigning it to the dst pixel values.
                dst.at<cv::Vec3s>(row, col)[clr] = cv::saturate_cast<short>(res / 2);
            }
        }
    }

    return 0;
}

/*
    This function calculates the gradient magnitude of an image.
*/

int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst)
{
    // Initializing the destination image
    dst.create(sx.size(), CV_16SC3);

    int xVal, yVal;

    for (int row = 0; row < sx.rows; row++) {
        for (int col = 0; col < sx.cols; col++) {
            for (int clr = 0; clr < 3; clr++) {
                // Accessing the pixel values of the images after applying Sobel X and Y
                xVal = sx.at<cv::Vec3s>(row, col)[clr];
                yVal = sy.at<cv::Vec3s>(row, col)[clr];
                // Calculating the gradient based on the X and Y Sobel filters and normalizing it before assigning it to dst
                dst.at<cv::Vec3s>(row, col)[clr] = cv::saturate_cast<short>(std::sqrt(xVal * xVal + yVal * yVal));
            }
        }
    }
    return 0;

}

/*
    This function generates the histogram of an image's gradient magnitude given the bins and also normalizes the histogram.
*/

std::vector<double> generateMagnitudeHistogram(Mat img, int bins) {
    Mat sx, sy;

    sobelX3x3(img, sx);
    sobelY3x3(img, sy);

    cv::Mat gradientMagnitude;

    magnitude(sx, sy, gradientMagnitude);

    std::vector<double> flattenedHistogram;

    flattenedHistogram = generateHistogram(gradientMagnitude, bins);

    return flattenedHistogram;
}

/*
    This function generates the histogram of an image's gradient orientation given the bins and also normalizes the histogram.
*/

std::vector<double> generateOrientationHistogram(Mat img, int bins) {
    Mat sx, sy;

    sobelX3x3(img, sx);
    sobelY3x3(img, sy);

    std::vector<double> orientation;

    Mat temp = Mat::zeros(img.rows, img.cols, CV_32F);

    double sum = 0;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            for (int clr = 0; clr < 3; clr++) {
                temp.at<float>(i, j) = atan2(sy.at<float>(i, j), sx.at<float>(i, j));
            }           
        }
    }

    return generateHistogram(temp, bins);

}

/*
    This function generates feature vector for texture matching by calculating the histograms for an image and
    its gradient magnitude and orientation.
*/

std::vector<std::vector<double>> textureMatching(Mat img, int bins) {

    std::vector<std::vector<double>> result;

    result.push_back(generateHistogram(img, bins));
    result.push_back(generateMagnitudeHistogram(img, bins));
    result.push_back(generateOrientationHistogram(img, bins));

    return result;

}

/*
    This function applies textureMatching on all the images in a given directory.
*/

std::vector<std::pair<std::vector<std::vector<double>>, char*>> textureMatchingDirectory(char dir[], int bins) {

    char dirname[256];
    FILE* fp;
    DIR* dirp;
    struct dirent* dp;
    std::vector<std::vector<double>> feature;
    std::vector<std::pair<std::vector<std::vector<double>>, char*>> features;

    // get the directory path
    strcpy_s(dirname, dir);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // dynamically allocate memory for the buffer
            char* buffer = new char[256];
            memset(buffer, 0, 256);

            // build the overall filename
            strcpy_s(buffer, 256, dirname);
            strcat_s(buffer, 256, "/");
            strcat_s(buffer, 256, dp->d_name);

            //printf("full path name: %s\n", buffer);

            feature = textureMatching(imread(buffer), bins);

            features.push_back(std::make_pair(feature, buffer));

        }
    }

    return features;

}

/*
    This function applies content based image retrieval on an image. 
    It is specifically designed to identify objects based on intensity on the bottom part of the image along with slight weightage to texture 
    that mainly focuses on gradient magnitude and orientation to identify the edges and angles.
*/

std::vector<std::pair<double, char*>> CBIR(Mat img, char dir[], int bins) {

    std::vector<std::vector<double>> multiHistogramFeature = generateQuadrantsHistograms(img, bins);
    std::vector<std::pair<std::vector<std::vector<double>>, char*>> multiHistogramFeatures = multiHistogramMatchingDirectory(dir, bins);

    std::vector<std::vector<double>> textureFeature = textureMatching(img, bins);
    std::vector<std::pair<std::vector<std::vector<double>>, char*>> textureFeatures = textureMatchingDirectory(dir, bins);

    std::vector<std::pair<double, char*>> intersections1, intersections2;
    double intersection;
    double weightedIntersection;

    for (int i = 0; i < multiHistogramFeatures.size(); i++) {

        weightedIntersection = 0.0;

        for (int k = 0; k < multiHistogramFeatures[i].first.size(); k++) {

            intersection = 0.0;

            for (int j = 0; j < multiHistogramFeature[k].size(); j++) {

                intersection += multiHistogramFeature[k][j] < multiHistogramFeatures[i].first[k][j] ? multiHistogramFeature[k][j] : multiHistogramFeatures[i].first[k][j];

            }

            weightedIntersection += (1 - intersection);

        }

        weightedIntersection = weightedIntersection / 4;

        intersections1.push_back(std::make_pair(weightedIntersection, multiHistogramFeatures[i].second));

    }


    for (int i = 0; i < textureFeatures.size(); i++) {

        weightedIntersection = 0.0;

        for (int k = 0; k < textureFeatures[i].first.size(); k++) {

            intersection = 0.0;

            for (int j = 0; j < textureFeature[k].size(); j++) {

                intersection += textureFeature[k][j] < textureFeatures[i].first[k][j] ? textureFeature[k][j] : textureFeatures[i].first[k][j];

            }

            weightedIntersection += (1 - intersection);

        }

        weightedIntersection = weightedIntersection / 4;

        intersections2.push_back(std::make_pair(weightedIntersection, textureFeatures[i].second));

    }

    for (int i = 0; i < intersections1.size(); i++) {

        intersections1[i].first = intersections1[i].first * 0.75 + intersections2[i].first * 0.25;

    }

    std::sort(intersections1.begin(), intersections1.end());

    return intersections1;

}

/*
    This function applies laws filter on an image and returns the new output image.
*/

std::vector<double> generateLawsFilterHistogram(Mat img, int bins) {
    Mat result;
    Mat L5, E5, S5, W5, R5;

    Mat kernL5 = (Mat_<double>(1, 5) << 1, 4, 6, 4, 1);
    filter2D(img, L5, -1, kernL5);

    Mat kernE5 = (Mat_<double>(1, 5) << -1, -2, 0, 2, 1);
    filter2D(img, E5, -1, kernE5);

    Mat kernS5 = (Mat_<double>(1, 5) << -1, 0, 2, 0, -1);
    filter2D(img, S5, -1, kernS5);

    Mat kernW5 = (Mat_<double>(1, 5) << -1, 2, 0, -2, 1);
    filter2D(img, W5, -1, kernW5);

    Mat kernR5 = (Mat_<double>(1, 5) << 1, -4, 6, -4, 1);
    filter2D(img, R5, -1, kernR5);

    std::vector<Mat> lawsFilters = { L5, E5, S5, W5, R5 };
    merge(lawsFilters, result);

    std::vector<double> flattenedHistogram;

    flattenedHistogram = generateHistogram(result, bins);

    return flattenedHistogram;
}

/*
    This function generates feature vector for texture matching by calculating the histograms for an image and
    its law filter.
*/

std::vector<std::vector<double>> textureMatching2(Mat img, int bins) {

    std::vector<std::vector<double>> result;

    result.push_back(generateHistogram(img, bins));
    result.push_back(generateLawsFilterHistogram(img, bins));

    return result;

}

/*
    This function applies textureMatching2 on all the images in a given directory.
*/

std::vector<std::pair<std::vector<std::vector<double>>, char*>> textureMatchingDirectory2(char dir[], int bins) {

    char dirname[256];
    FILE* fp;
    DIR* dirp;
    struct dirent* dp;
    std::vector<std::vector<double>> feature;
    std::vector<std::pair<std::vector<std::vector<double>>, char*>> features;

    // get the directory path
    strcpy_s(dirname, dir);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {

            // dynamically allocate memory for the buffer
            char* buffer = new char[256];
            memset(buffer, 0, 256);

            // build the overall filename
            strcpy_s(buffer, 256, dirname);
            strcat_s(buffer, 256, "/");
            strcat_s(buffer, 256, dp->d_name);

            //printf("full path name: %s\n", buffer);

            feature = textureMatching2(imread(buffer), bins);

            features.push_back(std::make_pair(feature, buffer));

        }
    }

    return features;

}

/*
    This function applies content based image retrieval on an image.
    It is specifically designed to identify objects based on intensity on the bottom part of the image along with slight weightage to texture
    that mainly focuses on Laws Filter to identify textures.
*/

std::vector<std::pair<double, char*>> CBIR2(Mat img, char dir[], int bins) {

    std::vector<std::vector<double>> multiHistogramFeature = generateQuadrantsHistograms(img, bins);
    std::vector<std::pair<std::vector<std::vector<double>>, char*>> multiHistogramFeatures = multiHistogramMatchingDirectory(dir, bins);

    std::vector<std::vector<double>> textureFeature = textureMatching2(img, bins);
    std::vector<std::pair<std::vector<std::vector<double>>, char*>> textureFeatures = textureMatchingDirectory2(dir, bins);

    std::vector<std::pair<double, char*>> intersections1, intersections2;
    double intersection;
    double weightedIntersection;

    for (int i = 0; i < multiHistogramFeatures.size(); i++) {

        weightedIntersection = 0.0;

        for (int k = 0; k < multiHistogramFeatures[i].first.size(); k++) {

            intersection = 0.0;

            for (int j = 0; j < multiHistogramFeature[k].size(); j++) {

                intersection += multiHistogramFeature[k][j] < multiHistogramFeatures[i].first[k][j] ? multiHistogramFeature[k][j] : multiHistogramFeatures[i].first[k][j];

            }

            weightedIntersection += (1 - intersection);

        }

        weightedIntersection = weightedIntersection / 4;

        intersections1.push_back(std::make_pair(weightedIntersection, multiHistogramFeatures[i].second));

    }


    for (int i = 0; i < textureFeatures.size(); i++) {

        weightedIntersection = 0.0;

        for (int k = 0; k < textureFeatures[i].first.size(); k++) {

            intersection = 0.0;

            for (int j = 0; j < textureFeature[k].size(); j++) {

                intersection += textureFeature[k][j] < textureFeatures[i].first[k][j] ? textureFeature[k][j] : textureFeatures[i].first[k][j];

            }

            weightedIntersection += (1 - intersection);

        }

        weightedIntersection = weightedIntersection / 4;

        intersections2.push_back(std::make_pair(weightedIntersection, textureFeatures[i].second));

    }

    for (int i = 0; i < intersections1.size(); i++) {

        intersections1[i].first = intersections1[i].first * 0.5 + intersections2[i].first * 0.5;

    }

    std::sort(intersections1.begin(), intersections1.end());

    return intersections1;

}