/*
  Sri Harsha Gollamudi
  Mar 2023

  This file contains the main function for the code and it calls the functions from helper.cpp to execute the tasks.
  It shows the results after calibrating the camera and projecting various things onto it using checkerboard.

*/

#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "helper.h"
#include <unordered_map>
#include <numeric>


using namespace cv;

int main() {

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
    Mat extractCornersImg;
    cv::Mat cameraMatrix, distCoeffs;
    cv::FileStorage fsr("intrinsics.yml", cv::FileStorage::READ);


    char key;
    std::vector<std::vector<cv::Vec3f> > point_list;
    std::vector<std::vector<cv::Point2f> > corner_list;
    std::vector<cv::Vec3f> point_set;
    std::vector<cv::Point2f> corners;
    std::vector<cv::Point2f> customCorners;

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 9; j++) {
            point_set.push_back(cv::Vec3f(j, i, 0));
        }
    }

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
            // Extract Corners
            while (true) {
                *capdev >> frame;
                extractCornersImg = extractChessCorners(frame, corners);
                cv::imshow("Extract Corners Video", extractCornersImg);
                key = waitKey(1);
                if (key == 's') {
                    if (corners.size() > 0) {
                        corner_list.push_back(corners);
                        point_list.push_back(point_set);
                        printf("Image added to calibration.\n");
                    }
                    else {
                        printf("Chessboard corners not found.\n");
                    }
                }
                else if (key == 'q') {
                    printf("\nNo of Corners: %d. First Corner Co-ordinates: %f,%f\n", corners.size(), corners[0].x, corners[0].y);
                    destroyAllWindows();
                    break;
                }
            }

            break;

        case 'c':
            // Calibrate Camera
            if (corner_list.size() >= 5) {

                // Initializing the Camera Matrix and Distortion Coefficients
                cv::Size imageSize(frame.cols, frame.rows);
                cameraMatrix = cv::Mat::eye(3, 3, CV_64FC1);
                cameraMatrix.at<double>(0, 2) = imageSize.width / 2;
                cameraMatrix.at<double>(1, 2) = imageSize.height / 2;
                distCoeffs = cv::Mat::zeros(8, 1, CV_64FC1);
                printf("Camera Matrix Before Calibration:\n");
                std::cout << cameraMatrix << std::endl;
                printf("Distortion Coefficients Before Calibration:\n");
                std::cout << distCoeffs << std::endl;

                // Calibrating the Camera
                std::vector<cv::Mat> rvecs, tvecs;
                int flags = CALIB_FIX_ASPECT_RATIO;
                double error = cv::calibrateCamera(point_list, corner_list, imageSize,
                    cameraMatrix, distCoeffs, rvecs, tvecs, flags);
                printf("Camera Matrix After Calibration:\n");
                std::cout << cameraMatrix << std::endl;
                printf("Distortion Coefficients After Calibration:\n");
                std::cout << distCoeffs << std::endl;
                printf("Re-projection error: %f\n", error);

                // Storing the Intrinsic Parameters to a file
                cv::FileStorage fsw("intrinsics.yml", cv::FileStorage::WRITE);
                fsw << "camera_matrix" << cameraMatrix;
                fsw << "distortion_coefficients" << distCoeffs;
                fsw.release();
            }
            else {
                printf("Not enough calibration images.\n");
            }
            break;

        case 'r':
            // Read Intrinsic Parameters from a file
            fsr["camera_matrix"] >> cameraMatrix;
            fsr["distortion_coefficients"] >> distCoeffs;

            while (true) {
                if (!cameraMatrix.empty() && !distCoeffs.empty()) {
                    *capdev >> frame;
                    extractCornersImg = extractChessCorners(frame, customCorners);
                    cv::imshow("Extract Corners Video Using Custom Intrinsic Parameters", extractCornersImg);
                    if (!customCorners.empty()) {
                        // Finding the rotation and translation vectors
                        cv::Mat rvec, tvec;
                        cv::solvePnP(point_set, customCorners, cameraMatrix, distCoeffs, rvec, tvec);

                        std::cout << "Rotation Vector: " << rvec << std::endl;
                        std::cout << "Translation Vector: " << tvec << std::endl;
                    }
                }
                else {
                    printf("Intrinsic Parameters are empty.\n");
                    break;
                }
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'p':
            // Read Intrinsic Parameters from a file and Project Axes
            fsr["camera_matrix"] >> cameraMatrix;
            fsr["distortion_coefficients"] >> distCoeffs;

            while (true) {
                if (!cameraMatrix.empty() && !distCoeffs.empty()) {
                    *capdev >> frame;

                    extractCornersImg = extractChessCorners(frame, customCorners);
                    //cv::imshow("Extract Corners Video Using Custom Intrinsic Parameters", extractCornersImg);

                    if (!customCorners.empty()) {

                        // Finding the rotation and translation vectors
                        cv::Mat rvec, tvec;
                        cv::solvePnP(point_set, customCorners, cameraMatrix, distCoeffs, rvec, tvec);

                        std::cout << "Rotation Vector: " << rvec << std::endl;
                        std::cout << "Translation Vector: " << tvec << std::endl;

                        // Project 3D Axes
                        std::vector<cv::Point3f> objectPointsAxes;
                        objectPointsAxes.push_back(cv::Point3f(0, 0, 0));
                        objectPointsAxes.push_back(cv::Point3f(8, 0, 0));
                        objectPointsAxes.push_back(cv::Point3f(0, 5, 0));
                        objectPointsAxes.push_back(cv::Point3f(0, 0, 4));


                        std::vector<cv::Point2f> imagePoints;
                        cv::projectPoints(objectPointsAxes, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

                        cv::drawFrameAxes(extractCornersImg, cameraMatrix, distCoeffs, rvec, tvec, 5);

                        for (const auto& p : customCorners) {
                            cv::circle(extractCornersImg, p, 3, cv::Scalar(255, 0, 0), -1);
                        }

                        cv::imshow("Projected Axes and Corners", extractCornersImg);
                    }
                    else {
                        printf("Corners not Found.\n");
                    }
                }
                else {
                    printf("Intrinsic Parameters are empty.\n");
                    break;
                }
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'v':
            // Create Diamond/Octahedron and project it onto the world co-ordinates
            fsr["camera_matrix"] >> cameraMatrix;
            fsr["distortion_coefficients"] >> distCoeffs;

            while (true) {
                if (!cameraMatrix.empty() && !distCoeffs.empty()) {
                    *capdev >> frame;

                    extractCornersImg = extractChessCorners(frame, customCorners);

                    if (!customCorners.empty()) {
                        cv::Mat rvec, tvec;
                        cv::solvePnP(point_set, customCorners, cameraMatrix, distCoeffs, rvec, tvec);

                        // Setting the object coordinates
                        std::vector<cv::Point3f> objectPointsPyramid;
                        float scale_factor = 0.5;
                        cv::Point3f translation(4, 3, 0);

                        objectPointsPyramid.push_back(cv::Point3f(0, 0, 0) * scale_factor + translation);
                        objectPointsPyramid.push_back(cv::Point3f(4, 4, -4) * scale_factor + translation);
                        objectPointsPyramid.push_back(cv::Point3f(4, -4, -4) * scale_factor + translation);
                        objectPointsPyramid.push_back(cv::Point3f(-4, -4, -4) * scale_factor + translation);
                        objectPointsPyramid.push_back(cv::Point3f(-4, 4, -4) * scale_factor + translation);
                        objectPointsPyramid.push_back(cv::Point3f(0, 0, -8) * scale_factor + translation);


                        std::vector<cv::Point2f> imagePoints;
                        cv::projectPoints(objectPointsPyramid, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

                        // Drawing the the lines
                        cv::line(extractCornersImg, imagePoints[0], imagePoints[1], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[0], imagePoints[2], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[0], imagePoints[4], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[1], imagePoints[2], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[2], imagePoints[3], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[3], imagePoints[4], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[4], imagePoints[1], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[1], imagePoints[5], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[2], imagePoints[5], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[3], imagePoints[5], cv::Scalar(255, 0, 0), 2);
                        cv::line(extractCornersImg, imagePoints[4], imagePoints[5], cv::Scalar(255, 0, 0), 2);

                        cv::imshow("Virtual Object", extractCornersImg);
                    }
                    else {
                        printf("Corners not Found.\n");
                    }
                }
                else {
                    printf("Intrinsic Parameters are empty.\n");
                    break;
                }
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'u':
            // Create Hexagon and project it onto the world co-ordinates
            fsr["camera_matrix"] >> cameraMatrix;
            fsr["distortion_coefficients"] >> distCoeffs;

            while (true) {
                if (!cameraMatrix.empty() && !distCoeffs.empty()) {
                    *capdev >> frame;

                    extractCornersImg = extractChessCorners(frame, customCorners);

                    if (!customCorners.empty()) {
                        cv::Mat rvec, tvec;
                        cv::solvePnP(point_set, customCorners, cameraMatrix, distCoeffs, rvec, tvec);

                        // Setting the object coordinates
                        float scale_factor = 3;
                        cv::Point3f translation(4, 3, 0);

                        std::vector<cv::Point3f> hexagonPoints;
                        hexagonPoints.push_back(cv::Point3f(-0.5, 0.0, 0.5 * sqrt(6))* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(0.5, 0.0, 0.5 * sqrt(6))* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(1.0, 0.0, 0.0)* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(0.5, 0.0, -0.5 * sqrt(6))* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(-0.5, 0.0, -0.5 * sqrt(6))* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(-1.0, 0.0, 0.0)* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(-0.5, 1.0, 0.5 * sqrt(6))* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(0.5, 1.0, 0.5 * sqrt(6))* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(1.0, 1.0, 0.0)* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(0.5, 1.0, -0.5 * sqrt(6))* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(-0.5, 1.0, -0.5 * sqrt(6))* scale_factor + translation);
                        hexagonPoints.push_back(cv::Point3f(-1.0, 1.0, 0.0)* scale_factor + translation);

                        std::vector<int> connect = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 6, 0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11 };

                        std::vector<cv::Point2f> imagePoints;
                        cv::projectPoints(hexagonPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

                        // Drawing the the lines
                        for (int i = 0; i < connect.size(); i += 2) {
                            cv::line(extractCornersImg, imagePoints[connect[i]], imagePoints[connect[i + 1]], cv::Scalar(255, 0, 0), 2);
                        }

                        cv::imshow("Virtual Object 2", extractCornersImg);
                    }
                    else {
                        printf("Corners not Found.\n");
                    }
                }
                else {
                    printf("Intrinsic Parameters are empty.\n");
                    break;
                }
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'h':
            // Harris Corners
            while (true)
            {
                *capdev >> frame;

                // Converting the image to greyscale and thresholding it to find harris corners
                Mat gray;
                cvtColor(frame, gray, COLOR_BGR2GRAY);

                Mat dst;
                cornerHarris(gray, dst, 2, 9, 0.04);

                Mat dst_norm, dst_norm_scaled;
                normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
                convertScaleAbs(dst_norm, dst_norm_scaled);
                threshold(dst_norm_scaled, dst_norm_scaled, 150, 255, THRESH_BINARY);

                for (int i = 0; i < dst_norm.rows; i++)
                {
                    for (int j = 0; j < dst_norm.cols; j++)
                    {
                        if ((int)dst_norm_scaled.at<uchar>(i, j) > 0)
                        {
                            circle(frame, Point(j, i), 5, Scalar(0, 0, 255), 2, 8, 0);
                        }
                    }
                }

                imshow("Harris Corners Features", frame);

                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'e':
            // Change Object
            // Finding the checkerboard and replacing it by constantly tracing the image points
            fsr["camera_matrix"] >> cameraMatrix;
            fsr["distortion_coefficients"] >> distCoeffs;

            while (true) {
                if (!cameraMatrix.empty() && !distCoeffs.empty()) {
                    *capdev >> frame;

                    extractCornersImg = extractChessCorners(frame, customCorners);

                    if (!customCorners.empty()) {
                        cv::Mat rvec, tvec;
                        cv::solvePnP(point_set, customCorners, cameraMatrix, distCoeffs, rvec, tvec);

                        std::vector<cv::Point3f> objectPoints;

                        objectPoints.push_back(cv::Point3f(-1, -1, 0));
                        objectPoints.push_back(cv::Point3f(9, -1, 0));
                        objectPoints.push_back(cv::Point3f(-1, 6, 0));
                        objectPoints.push_back(cv::Point3f(9, 6, 0));

                        std::vector<cv::Point3f> rectangle1;
                        std::vector<cv::Point3f> rectangle2;

                        rectangle1.push_back(cv::Point3f(-1, -1, 0));
                        rectangle1.push_back(cv::Point3f(9, -1, 0));
                        rectangle1.push_back(cv::Point3f(-1, 3.5, 0));
                        rectangle1.push_back(cv::Point3f(9, 3.5, 0));

                        rectangle2.push_back(cv::Point3f(-1, 3.5, 0));
                        rectangle2.push_back(cv::Point3f(9, 3.5, 0));
                        rectangle2.push_back(cv::Point3f(-1, 6, 0));
                        rectangle2.push_back(cv::Point3f(9, 6, 0));

                        // Define the colors for the rectangles
                        cv::Scalar red(0, 0, 255);
                        cv::Scalar blue(255, 0, 0);

                        // Fill the first rectangle with red color
                        std::vector<std::vector<cv::Point>> fillContour1(1);
                        std::vector<cv::Point2f> imagePoints1;
                        cv::projectPoints(rectangle1, rvec, tvec, cameraMatrix, distCoeffs, imagePoints1);
                        fillContour1[0].push_back(imagePoints1[0]);
                        fillContour1[0].push_back(imagePoints1[1]);
                        fillContour1[0].push_back(imagePoints1[3]);
                        fillContour1[0].push_back(imagePoints1[2]);
                        cv::fillPoly(frame, fillContour1, red);

                        // Fill the second rectangle with blue color
                        std::vector<std::vector<cv::Point>> fillContour2(1);
                        std::vector<cv::Point2f> imagePoints2;
                        cv::projectPoints(rectangle2, rvec, tvec, cameraMatrix, distCoeffs, imagePoints2);
                        fillContour2[0].push_back(imagePoints2[0]);
                        fillContour2[0].push_back(imagePoints2[1]);
                        fillContour2[0].push_back(imagePoints2[3]);
                        fillContour2[0].push_back(imagePoints2[2]);
                        cv::fillPoly(frame, fillContour2, blue);
                        
                        cv::imshow("Virtual Object", frame);
                    }
                    else {
                        printf("Corners not Found.\n");
                    }
                }
                else {
                    printf("Intrinsic Parameters are empty.\n");
                    break;
                }
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'a':
            // Aruco integration
            fsr["camera_matrix"] >> cameraMatrix;
            fsr["distortion_coefficients"] >> distCoeffs;

            while (true) {
                if (!cameraMatrix.empty() && !distCoeffs.empty()) {
                    *capdev >> frame;

                    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
                    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
                    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
                    cv::Mat output;
                    frame.copyTo(output);
                    std::vector<int> ids;
                    std::vector<std::vector<cv::Point2f>> corners, rejected;
                    detector.detectMarkers(frame, corners, ids, rejected);
                    if (ids.size() > 0) {
                        cv::aruco::drawDetectedMarkers(output, corners, ids);
                    }
                    cv::imshow("Aruco", output);

                }

                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'f':
            // Change Object with any image
            // Finding the checkerboard and replacing it by constantly tracing the image points
            fsr["camera_matrix"] >> cameraMatrix;
            fsr["distortion_coefficients"] >> distCoeffs;
            Mat replacementImg = imread("C:/Users/PREDATOR08/Downloads/imgcv4.jpg");

            while (true) {
                if (!cameraMatrix.empty() && !distCoeffs.empty()) {
                    *capdev >> frame;

                    extractCornersImg = extractChessCorners(frame, customCorners);

                    if (!customCorners.empty()) {
                        cv::Mat rvec, tvec;
                        cv::solvePnP(point_set, customCorners, cameraMatrix, distCoeffs, rvec, tvec);

                        std::vector<cv::Point3f> objectPoints;

                        objectPoints.push_back(cv::Point3f(-1, -1, 0));
                        objectPoints.push_back(cv::Point3f(9, -1, 0));
                        objectPoints.push_back(cv::Point3f(-1, 6, 0));
                        objectPoints.push_back(cv::Point3f(9, 6, 0));

                        std::vector<cv::Point2f> imagePoints;
                        cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

                        cv::Rect boundingBox = cv::boundingRect(imagePoints);

                        cv::Rect roi(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height);
                        cv::Mat roiMat = frame(roi);
                        cv::resize(replacementImg, replacementImg, roiMat.size());
                        replacementImg.copyTo(roiMat);

                        // Setting the object coordinates
                        std::vector<cv::Point3f> objectPointsPyramid;
                        float scale_factor = 0.5;
                        cv::Point3f translation(4, 3, 0);

                        objectPointsPyramid.push_back(cv::Point3f(0, 0, 0) * scale_factor + translation);
                        objectPointsPyramid.push_back(cv::Point3f(4, 4, -4) * scale_factor + translation);
                        objectPointsPyramid.push_back(cv::Point3f(4, -4, -4) * scale_factor + translation);
                        objectPointsPyramid.push_back(cv::Point3f(-4, -4, -4) * scale_factor + translation);
                        objectPointsPyramid.push_back(cv::Point3f(-4, 4, -4) * scale_factor + translation);
                        objectPointsPyramid.push_back(cv::Point3f(0, 0, -8) * scale_factor + translation);

                        cv::projectPoints(objectPointsPyramid, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

                        // Drawing the the lines
                        cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[0], imagePoints[2], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[0], imagePoints[4], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[1], imagePoints[2], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[2], imagePoints[3], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[3], imagePoints[4], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[4], imagePoints[1], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[1], imagePoints[5], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[2], imagePoints[5], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[3], imagePoints[5], cv::Scalar(255, 0, 0), 2);
                        cv::line(frame, imagePoints[4], imagePoints[5], cv::Scalar(255, 0, 0), 2);

                        cv::imshow("Replace Target", frame);
                    }
                    else {
                        printf("Corners not Found.\n");
                    }
                }
                else {
                    printf("Intrinsic Parameters are empty.\n");
                    break;
                }
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