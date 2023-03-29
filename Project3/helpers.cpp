/*
  Sri Harsha Gollamudi
  Feb 2023

  This file contains the helper functions required to perform thresholding, clean-up, 
  segmentation, region mapping, feature extraction, classification and confusion matrix.
*/

#include <dirent.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "helpers.h"
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <vector>
#include <iostream>


using namespace cv;

/*
* This function applies gaussian blur on the image.
 */
int blur5x5(cv::Mat& src, cv::Mat& dst)
{

    // 1x5 Gaussian Filter 
    int filter[5] = { 1, 2, 4, 2, 1 };

    // Initializing the destination image
    dst.create(src.size(), src.type());

    // Horizontal filter
    for (int row = 0; row < src.rows; row++) {
        for (int col = 2; col < src.cols - 2; col++) {
            for (int clr = 0; clr < 3; clr++) {
                int res = 0;
                for (int i = -2; i <= 2; i++) {
                    // Dot product of the filter and pixel values 
                    res += src.at<cv::Vec3b>(row, col + i)[clr] * filter[i + 2];
                }
                // Normalizing the result and assigning it to the dst pixel values.
                dst.at<cv::Vec3b>(row, col)[clr] = cv::saturate_cast<uchar>(res / 10);
            }
        }
    }

    // Vertical filter
    for (int row = 2; row < src.rows - 2; row++) {
        for (int col = 0; col < src.cols; col++) {
            for (int clr = 0; clr < 3; clr++) {
                int res = 0;
                for (int i = -2; i <= 2; i++) {
                    // Dot product of the filter and pixel values 
                    res += dst.at<cv::Vec3b>(row + i, col)[clr] * filter[i + 2];
                }
                // Normalizing the result and assigning it to the dst pixel values.
                dst.at<cv::Vec3b>(row, col)[clr] = cv::saturate_cast<uchar>(res / 10);
            }
        }
    }
    return 0;
}

/*
* This function converts an image into greyscale.
 */
int greyscale(cv::Mat& src, cv::Mat& dst) {

    // Initializing the destination image
    dst = cv::Mat(src.rows, src.cols, CV_8UC1);

    for (int i = 0; i < src.rows; i++) {
        // Get the row pointer in Vec3b format
        cv::Vec3b* row = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            // Computing the average color of blue, green and red colors for grey scale conversion
            dst.at<uchar>(i, j) = (row[j][0] + row[j][1] + row[j][2]) / 3;
        }
    }
    return 0;
}

/*
* This function thresholdes the image by using the threshold value ie 140 (tried median but performed worse)
 */
Mat thresholdImage(Mat src) {
    Mat dst, blurImg;

    blur5x5(src, blurImg);
    greyscale(blurImg, dst);

    // Calculate median value
    double median = findMedian(dst);
    median = 140;

    // Threshold using median value
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if (dst.at<uchar>(i, j) > median) {
                dst.at<uchar>(i, j) = 255;
            }
            else {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }

    return dst;
}

/*
* This function is used to find the median value of the pixels values in an image.
 */
double findMedian(Mat img) {
    std::vector<uchar> imgVec;
    imgVec.assign(img.data, img.data + img.total());

    std::sort(imgVec.begin(), imgVec.end());

    double median;
    int n = imgVec.size();
    if (n % 2 == 0) {
        median = (imgVec[n / 2 - 1] + imgVec[n / 2]) / 2.0;
    }
    else {
        median = imgVec[n / 2];
    }

    return median;
}

/*
* This function cleans the thresholeded image by using erosion and dilation.
 */
Mat cleanUpImage(Mat src, int times) {

    Mat dst = src.clone();

    for (int i = 0; i < times; i++) {

        dst = erosion(dst);

    }

    for (int i = 0; i < times; i++) {

        dst = dilation(dst);

    }

    return dst;

}

/*
* This function implements the erosion morphological operation from scratch.
 */
Mat erosion(Mat src)
{
    Mat dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    int fourConnected[3][3] = { {0, 1, 0}, {1, 1, 1}, {0, 1, 0} };
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {

            bool connected = true;

            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    if (fourConnected[x][y] == 1 && src.at<uchar>(i - 1 + x, j - 1 + y) == 0) {
                        connected = false;
                        break;
                    }
                }

                if (!connected) {
                    break;
                }
            }

            if (connected) {
                dst.at<uchar>(i, j) = 255;
            }
            else {
                dst.at<uchar>(i, j) = 0;
            }

        }
    }
    return dst;
}

/*
* This function implements the dilation morphological operation from scratch.
 */
Mat dilation(Mat src)
{
    Mat dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    int eightConnected[3][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {

            bool connected = false;

            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    if (eightConnected[x][y] == 1 && src.at<uchar>(i - 1 + x, j - 1 + y) == 255) {
                        connected = true;
                        break;
                    }
                }

                if (connected) {
                    break;
                }
            }

            if (connected) {
                dst.at<uchar>(i, j) = 255;
            }
            else {
                dst.at<uchar>(i, j) = 0;
            }

        }
    }
    return dst;
}

/*
* This function segments the cleaned image and identifies all the region maps in the image
  and also applies bounding box around them.
 */
Mat segmentImage(Mat src) {

    Mat dst = Mat::zeros(src.size(), CV_8UC3);
    Mat temp = src.clone();

    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {

            if (temp.at<uchar>(i, j) == 255) {
                temp.at<uchar>(i, j) = 0;
            }
            else
            {
                temp.at<uchar>(i, j) = 255;
            }

        }
    }

    Mat labels, stats, centroids;
    int numRegions = cv::connectedComponentsWithStats(temp, labels, stats, centroids, 4);

    std::vector<cv::Scalar> colorTable(numRegions);
    cv::RNG rng(12345);
    for (int i = 1; i < numRegions; i++) {
        colorTable[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    for (int i = 1; i < numRegions; i++) {
        cv::Scalar color = colorTable[i];
        cv::Mat mask = labels == i;
        dst.setTo(color, mask);
    }

    for (int i = 1; i < numRegions; i++) {
        cv::Rect rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
        cv::rectangle(dst, rect, colorTable[i], 2);
    }

    return dst;

}

/*
* This function normalizes the feature vector of the image.
 */
std::vector<double> normalize(std::vector<double> features) {

    double sum = 0;
    double squarredSum = 0;

    for (int i = 0; i < features.size(); i++) {
        sum += features[i];
        squarredSum += features[i] * features[i];
    }

    double mean = sum / features.size();
    double variance = squarredSum / features.size() - mean * mean;
    double stddev = std::sqrt(variance);

    std::vector<double> normalizedFeatures;
    for (int i = 0; i < features.size(); i++) {
        normalizedFeatures.push_back((features[i] - mean) / stddev);
    }

    return normalizedFeatures;
}

/*
* This function identifies the objects region map and applies an oriented bounding box around it.
 */
Mat regionFeaturesImage(Mat src, Mat srcOrginal) {

    Mat dst = Mat::zeros(src.size(), CV_8UC3);
    Mat temp = src.clone();
    Mat res = srcOrginal.clone();

    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {

            if (temp.at<uchar>(i, j) == 255) {
                temp.at<uchar>(i, j) = 0;
            }
            else
            {
                temp.at<uchar>(i, j) = 255;
            }

        }
    }

    Mat labels, stats, centroids;
    int numRegions = cv::connectedComponentsWithStats(temp, labels, stats, centroids, 4);

    std::vector<int> regionSizes(numRegions);
    for (int i = 1; i < numRegions; i++) {
        regionSizes[i] = stats.at<int>(i, cv::CC_STAT_AREA);
    }

    std::vector<int> sorted(regionSizes);
    sort(sorted.begin(), sorted.end());
    int n = sorted.size();
    int second_largest = sorted[n - 2];

    std::vector<cv::Scalar> colorTable(numRegions);
    cv::RNG rng(12345);
    for (int i = 1; i < numRegions; i++) {
        if (regionSizes[i - 1] == second_largest) {
            colorTable[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        }
    }

    for (int i = 1; i < numRegions; i++) {
        if (regionSizes[i - 1] == second_largest) {
            cv::Scalar color = colorTable[i];
            cv::Mat mask = labels == i;

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (size_t j = 0; j < contours.size(); j++) {
                cv::drawContours(dst, contours, j, color, 3);

                cv::RotatedRect rect = cv::minAreaRect(contours[j]);
                cv::Point2f points[4];
                rect.points(points);

                for (int k = 0; k < 4; k++) {
                    cv::line(res, points[k], points[(k + 1) % 4], color, 2);
                }
            }
        }
    }

    return res;
}

/*
* This function gives the feature vector of a region. It currently supports Hu moments and Aspect Ratio.
 */
std::vector<double> regionFeatureVector(Mat src) {

    Mat dst = Mat::zeros(src.size(), CV_8UC3);
    Mat temp1 = src.clone();

    Mat temp = cleanUpImage(thresholdImage(temp1),10);

    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {

            if (temp.at<uchar>(i, j) == 255) {
                temp.at<uchar>(i, j) = 0;
            }
            else
            {
                temp.at<uchar>(i, j) = 255;
            }

        }
    }

    Mat labels, stats, centroids;
    int numRegions = cv::connectedComponentsWithStats(temp, labels, stats, centroids, 4);

    std::vector<int> regionSizes(numRegions);
    for (int i = 1; i < numRegions; i++) {
        regionSizes[i] = stats.at<int>(i, cv::CC_STAT_AREA);
    }

    std::vector<int> sorted(regionSizes);
    sort(sorted.begin(), sorted.end());
    int n = sorted.size();
    int second_largest = sorted[n - 2];

    std::vector<cv::Scalar> colorTable(numRegions);
    cv::RNG rng(12345);
    for (int i = 1; i < numRegions; i++) {
        if (regionSizes[i - 1] == second_largest) {
            colorTable[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        }
    }

    std::vector<double> features;
    cv::RotatedRect rect;
    double aspectRatio = 0;
    double area = 0;

    for (int i = 1; i < numRegions; i++) {
        if (regionSizes[i - 1] == second_largest && features.size() != 7) {
            cv::Scalar color = colorTable[i];
            cv::Mat mask = labels == i;
            dst.setTo(color, mask);

            cv::Moments moments = cv::moments(mask, true);
            double hu[7];
            cv::HuMoments(moments, hu);

            for (int i = 0; i < 7; i++) {
                features.push_back(hu[i]);
            }


            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            for (size_t j = 0; j < contours.size(); j++) {
                cv::drawContours(dst, contours, j, color, 2);
                rect = cv::minAreaRect(contours[j]);
                cv::Point2f points[4];
                rect.points(points);
            }
            
            aspectRatio = rect.size.aspectRatio();
            area = rect.size.area();

        }
    }

    features.push_back(aspectRatio);
    std::vector<double> norm = normalize(features);

    return norm;

}

/*
* This function loads the training data and returns the feature vector with labels.
 */
std::vector<std::pair<std::vector<double>, char>> trainData(char dir[]) {

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

    std::vector<std::pair<std::vector<double>, char>> data;
    std::vector<double> feature;
    char object{};

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".jfif") ||
            strstr(dp->d_name, ".tif")) {

            // build the overall filename
            strcpy_s(buffer, dirname);
            strcat_s(buffer, "/");
            strcat_s(buffer, dp->d_name);

            Mat img = imread(buffer);

            feature = regionFeatureVector(img);

            switch (buffer[5]) {
            case 'a':
                object = 'a';
                break;
            case 'b':
                object = 'b';
                break;
            case 'c':
                object = 'c';
                break;
            case 'd':
                object = 'd';
                break;
            case 'e':
                object = 'e';
                break;
            case 'f':
                object = 'f';
                break;
            case 'g':
                object = 'g';
                break;
            case 'h':
                object = 'h';
                break;
            case 'i':
                object = 'i';
                break;
            case 'j':
                object = 'j';
                break;
            }

            data.push_back(std::make_pair(feature, object));

        }
    }

    return data;

}

/*
* This function finds the squared difference between the image's features and the training data. It gives the sorted distances.
 */
std::vector<std::pair<double, char>> squaredDifference(std::vector<std::pair<std::vector<double>, char>> features, std::vector<double> imgFeature) {

    std::vector<std::pair<double, char>> distances;
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
* This function finds the manhattan distance between the image's features and the training data. It gives the sorted distances.
 */
std::vector<std::pair<double, char>> manhattan(std::vector<std::pair<std::vector<double>, char>> features, std::vector<double> imgFeature) {

    std::vector<std::pair<double, char>> distances;
    double distance;

    for (int i = 0; i < features.size(); i++) {

        distance = 0;

        for (int j = 0; j < imgFeature.size(); j++) {

            distance += abs(imgFeature[j] - features[i].first[j]);

        }

        distances.push_back(std::make_pair(distance, features[i].second));
    }

    std::sort(distances.begin(), distances.end());

    return distances;

}

/*
* This function returns the top N matches for a given image along with its labels.
 */
void topNMatches(std::vector<std::pair<double, char>> distances, int n) {

    printf("The top %d matches are\n", n);

    for (int i = 0; i < n; i++) {

        printf("Dist: %f Class: %c\n", distances[i].first, distances[i].second);

    }

}

/*
* This function displays the given image along with a predicted label in the image.
 */
Mat classifyImage(Mat src, std::vector<std::pair<double, char>> distances) {

    Mat dst = Mat::zeros(src.size(), CV_8UC3);
    Mat temp1 = src.clone();
    Mat res = src.clone();

    Mat temp = cleanUpImage(thresholdImage(temp1), 10);

    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {

            if (temp.at<uchar>(i, j) == 255) {
                temp.at<uchar>(i, j) = 0;
            }
            else
            {
                temp.at<uchar>(i, j) = 255;
            }

        }
    }

    Mat labels, stats, centroids;
    int numRegions = cv::connectedComponentsWithStats(temp, labels, stats, centroids, 4);

    std::vector<int> regionSizes(numRegions);
    for (int i = 1; i < numRegions; i++) {
        regionSizes[i] = stats.at<int>(i, cv::CC_STAT_AREA);
    }

    std::vector<int> sorted(regionSizes);
    sort(sorted.begin(), sorted.end());
    int n = sorted.size();
    int second_largest = sorted[n - 2];

    std::vector<cv::Scalar> colorTable(numRegions);
    cv::RNG rng(12345);
    for (int i = 1; i < numRegions; i++) {
        if (regionSizes[i - 1] == second_largest) {
            colorTable[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        }
    }

    for (int i = 1; i < numRegions; i++) {
        if (regionSizes[i - 1] == second_largest) {
            cv::Scalar color = colorTable[i];
            cv::Mat mask = labels == i;
            dst.setTo(color, mask);

        }
    }

    cv::Point2f points[4];

    for (int i = 1; i < numRegions; i++) {
        if (regionSizes[i - 1] == second_largest) {
            cv::Scalar color = colorTable[i];
            cv::Mat mask = labels == i;

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (size_t j = 0; j < contours.size(); j++) {
                cv::drawContours(dst, contours, j, color, 3);

                cv::RotatedRect rect = cv::minAreaRect(contours[j]);
                rect.points(points);

                for (int k = 0; k < 4; k++) {
                    cv::line(res, points[k], points[(k + 1) % 4], color, 2);
                }
            }


            std::string object = "ABC";
            switch (distances[1].second) {
            case 'a':
                object = "Wallet";
                break;
            case 'b':
                object = "Knife Holder";
                break;
            case 'c':
                object = "Fork";
                break;
            case 'd':
                object = "Trimmer";
                break;
            case 'e':
                object = "Web Cam";
                break;
            case 'f':
                object = "Mouse";
                break;
            case 'g':
                object = "Remote";
                break;
            case 'h':
                object = "Perfume Spray";
                break;
            case 'i':
                object = "Earphones Holder";
                break;
            case 'j':
                object = "Watch";
                break;
            }
            int font = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 1.5;
            int thickness = 2;
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(object, font, fontScale, thickness, &baseline);
            cv::Point textOrg(50, 50 - textSize.height);
            cv::putText(res, object, cv::Point(points[2].x, points[2].y - 1), font, fontScale, cv::Scalar(0, 255, 0), thickness);


        }
    }

    return res;

}

/*
* This function finds the KNN centriods difference from the image using squared difference. It gives the sorted distances.
 */
std::vector<std::pair<double, char>> KNN(std::vector<std::pair<std::vector<double>, char>> features, std::vector<double> imgFeature, int k) {

    std::vector<std::pair<double, char>> distances;
    double distance;
    double difference;

    for (int i = 0; i < features.size(); i++) {

        distance = 0;

        for (int j = 0; j < imgFeature.size(); j++) {

            difference = abs(imgFeature[j] - features[i].first[j]);
            distance += (difference * difference);

        }

        distances.emplace_back(distance, features[i].second);
    }

    std::sort(distances.begin(), distances.end());

    std::vector<std::pair<double, char>> output;
    std::unordered_map<char, std::vector<double>> distanceMap;
    int dist, label;

    for (int i = 0; i < distances.size(); i++) {

        dist = distances[i].first;
        label = distances[i].second;

        if (distanceMap[label].size() < k) {
            distanceMap[label].push_back(dist);
        }

        else {
            auto& dists = distanceMap[label];
            auto it = std::max_element(dists.begin(), dists.end());
            if (dist < *it) {
                *it = dist;
            }
        }
    }

    for (auto& elem : distanceMap) {
        char label = elem.first;
        auto& dists = elem.second;
        double sum = std::accumulate(dists.begin(), dists.end(), 0.0);
        sum /= k;
        output.emplace_back(sum, label);
    }

    std::sort(output.begin(), output.end());

    return output;
}

/*
* This function loads the testing data and returns the feature vector with labels.
 */
std::vector<std::pair<std::vector<double>, char>> testData(char dir[]) {

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

    std::vector<std::pair<std::vector<double>, char>> data;
    std::vector<double> feature;
    char object{};

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {

        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".jfif") ||
            strstr(dp->d_name, ".tif")) {

            // build the overall filename
            strcpy_s(buffer, dirname);
            strcat_s(buffer, "/");
            strcat_s(buffer, dp->d_name);

            Mat img = imread(buffer);

            feature = regionFeatureVector(img);

            switch (buffer[5]) {
            case 'a':
                object = 'a';
                break;
            case 'b':
                object = 'b';
                break;
            case 'c':
                object = 'c';
                break;
            case 'd':
                object = 'd';
                break;
            case 'e':
                object = 'e';
                break;
            case 'f':
                object = 'f';
                break;
            case 'g':
                object = 'g';
                break;
            case 'h':
                object = 'h';
                break;
            case 'i':
                object = 'i';
                break;
            case 'j':
                object = 'j';
                break;
            }

            data.push_back(std::make_pair(feature, object));

        }
    }

    return data;

}


/*
* This function prints the confusion matrix after using KNN on the testing data.
 */
void printConfusionMatrix(std::vector<std::pair<std::vector<double>, char>> features, std::vector<std::pair<std::vector<double>, char>> testingData, int k) {

    int confusionMatrix[10][10] = { 0 };

    for (int i = 0; i < testingData.size(); i++) {
        std::vector<double> imgFeature = testingData[i].first;
        char trueLabel = testingData[i].second;

        std::vector<std::pair<double, char>> knnOutput = KNN(features, imgFeature, k);

        char predictedLabel = knnOutput[0].second;

        confusionMatrix[trueLabel - 'a'][predictedLabel - 'a']++;
    }

    printf("CONFUSION MATRIX\n  ");
    for (char label = 'a'; label <= 'j'; label++) {
        printf("%c ", label);
    }
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("%c ", static_cast<char>(i + 'a'));
        for (int j = 0; j < 10; j++) {
            printf("%d ", confusionMatrix[i][j]);
        }
        printf("\n");
    }

    printf("Where \n");

    String object;

    for (char label = 'a'; label <= 'j'; label++) {


        switch (label) {
        case 'a':
            object = "Wallet";
            break;
        case 'b':
            object = "Knife Holder";
            break;
        case 'c':
            object = "Fork";
            break;
        case 'd':
            object = "Trimmer";
            break;
        case 'e':
            object = "Web Cam";
            break;
        case 'f':
            object = "Mouse";
            break;
        case 'g':
            object = "Remote";
            break;
        case 'h':
            object = "Perfume Spray";
            break;
        case 'i':
            object = "Earphones";
            break;
        case 'j':
            object = "Watch";
            break;
        }

        printf("%c represents %s\n", label, object);
    }
}