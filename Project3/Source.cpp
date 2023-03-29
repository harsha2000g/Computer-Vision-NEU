/*
  Sri Harsha Gollamudi
  Feb 2023

  This file contains the main function for the code and it calls the functions from helpers.cpp to execute the tasks.
  It shows the results after applying multiple operations on the images for object detection.

*/

#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "helpers.h"
#include <unordered_map>
#include <numeric>

using namespace cv;

int main()
{

    cv::VideoCapture* capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("Video", 1); // identifies a window
    //source, intermediate and result frames
    Mat frame;
    Mat resultFrame;
    Mat thresholdedImg, cleanUpImg, segmentImg, regionFeaturesImg, labelFrame;

    char dir[] = "data";
    std::vector<std::pair<std::vector<double>, char>> features = trainData(dir);

    char dirTest[] = "test";
    std::vector<std::pair<std::vector<double>, char>> testFeatures = testData(dirTest);

    //cv::namedWindow("Video", cv::WINDOW_NORMAL);
    //cv::resizeWindow("Video", 640, 480);
    //cv::namedWindow("Thresholding Video", cv::WINDOW_NORMAL);
    //cv::resizeWindow("Thresholding Video", 640, 480);
    //cv::namedWindow("Clean Up Video", cv::WINDOW_NORMAL);
    //cv::resizeWindow("Clean Up Video", 640, 480);
    //cv::namedWindow("Segment", cv::WINDOW_NORMAL);
    //cv::resizeWindow("Segment", 640, 480);
    //cv::namedWindow("Region Features", cv::WINDOW_NORMAL);
    //cv::resizeWindow("Region Features", 640, 480);
    //cv::namedWindow("Classify New Images", cv::WINDOW_NORMAL);
    //cv::resizeWindow("Classify New Images", 640, 480);
    //cv::namedWindow("Classify KNN", cv::WINDOW_NORMAL);
    //cv::resizeWindow("Classify KNN", 640, 480);

    //frame = imread("test/aa.jpg");
    //thresholdedImg = thresholdImage(frame);
    //cleanUpImg = cleanUpImage(thresholdedImg, 5);
    //segmentImg = segmentImage(cleanUpImg);
    //regionFeaturesImg = regionFeaturesImage(cleanUpImg, frame);
    //Mat img = imread("test/jj.jpg");
    //std::vector<double> imgFeatures = regionFeatureVector(img);

    //std::vector<std::pair<double, char>> mdistances = manhattan(features, imgFeatures);

    //std::vector<std::pair<double, char>> distances = squaredDifference(features, imgFeatures);
    //Mat classifyNN = classifyImage(img, mdistances);

    //std::vector<std::pair<double, char>> kdistances = KNN(features, imgFeatures, 2);
    //Mat classifyKNN = classifyImage(img, kdistances);


    //cv::imshow("Video", frame);
    //cv::imshow("Thresholding Video", thresholdedImg);
    //cv::imshow("Clean Up Video", cleanUpImg);
    //cv::imshow("Segment", segmentImg);
    //cv::imshow("Region Features", regionFeaturesImg);
    //cv::imshow("Classify New Images", classifyNN);
    //cv::imshow("Classify KNN", classifyKNN);

    ////printConfusionMatrix(features, testFeatures, 2);

    //waitKey(50000);

    //variable for key press
    char key;

    while (true) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }
        cv::imshow("Video", frame);

        // see if there is a waiting keystroke
        key = cv::waitKey(1);
        switch (key) {
        case 'q':
            //Quit the program
            return 0;
        case 't':
            // Thresholding
            while (true) {
                *capdev >> frame;
                thresholdedImg = thresholdImage(frame);
                cv::imshow("Video", frame);
                cv::imshow("Thresholding Video", thresholdedImg);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;
        case 'c':
            // Clean-up
            while (true) {
                *capdev >> frame;
                thresholdedImg = thresholdImage(frame);
                cleanUpImg = cleanUpImage(thresholdedImg, 10);
                cv::imshow("Video", frame);
                cv::imshow("Clean Up Video", cleanUpImg);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;
        case 's':
            // Segment
            while (true) {
                *capdev >> frame;
                thresholdedImg = thresholdImage(frame);
                cleanUpImg = cleanUpImage(thresholdedImg, 10);
                segmentImg = segmentImage(cleanUpImg);
                cv::imshow("Video", frame);
                cv::imshow("Segment", segmentImg);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;
        case 'r':
            // Region Features
            while (true) {
                *capdev >> frame;
                thresholdedImg = thresholdImage(frame);
                cleanUpImg = cleanUpImage(thresholdedImg, 10);
                regionFeaturesImg = regionFeaturesImage(cleanUpImg, frame);
                cv::imshow("Region Feature", regionFeaturesImg);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;
        case 'l':
            // Label
            while (true) {
                *capdev >> frame;
                thresholdedImg = thresholdImage(frame);
                cleanUpImg = cleanUpImage(thresholdedImg, 10);
                segmentImg = segmentImage(cleanUpImg);
                std::vector<double> imgFeatures = regionFeatureVector(frame);
                std::vector<std::pair<double, char>> kdistances = KNN(features, imgFeatures, 2);
                labelFrame = classifyImage(frame, kdistances);
                cv::imshow("Label", labelFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;


        }
    }

}