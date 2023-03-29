#include <opencv2/opencv.hpp>
#include <cstdio>
// header file for functions
#include "filter.h"

using namespace cv;

int main(int argc, char* argv[]) {
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
    cv::Mat frame;
    cv::Mat resultFrame;
    cv::Mat resultFrame1;
    cv::Mat resultFrame2;

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
        case 's':
            // Save current frame to a file
            imwrite("output/captured_image.jpg", frame);
            break;
        case 'g':
            // Display grayscale using cvtColor
            while (true) {
                *capdev >> frame;
                cvtColor(frame, resultFrame, COLOR_BGR2GRAY);
                cv::imshow("Video", frame);
                cv::imshow("Greyscale Video", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;
        case 'h':
            // Display grayscale using greyscale function
            while (true) {
                *capdev >> frame;
                greyscale(frame, resultFrame);
                cv::imshow("Video", frame);
                cv::imshow("Greyscale Video 2", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'b':
            // Apply gaussian filter of size 5x5 to blur image
            while (true) {
                *capdev >> frame;
                blur5x5(frame, resultFrame);
                cv::imshow("Video", frame);
                cv::imshow("Blur 5x5", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'x':
            // Apply 3x3 Sobel X filter
            while (true) {
                *capdev >> frame;
                sobelX3x3(frame, resultFrame);
                cv::imshow("Video", frame);
                cv::convertScaleAbs(resultFrame, resultFrame);
                cv::imshow("X Sobel", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'y':
            // Apply 3x3 Sobel Y filter
            while (true) {
                *capdev >> frame;
                sobelY3x3(frame, resultFrame);
                cv::imshow("Video", frame);
                cv::convertScaleAbs(resultFrame, resultFrame, 1, 0);
                cv::imshow("Y Sobel", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'm':
            // Get gradient magnitude of the image
            while (true) {
                *capdev >> frame;
                sobelX3x3(frame, resultFrame1);
                sobelY3x3(frame, resultFrame2);
                magnitude(resultFrame1, resultFrame2, resultFrame);
                cv::imshow("Video", frame);
                cv::convertScaleAbs(resultFrame, resultFrame, 1, 0);
                cv::imshow("Magnitude", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'l':
            // Blurs the image using gaussian filter and quantizes the image into a fixed number of levels
            while (true) {
                *capdev >> frame;
                blurQuantize(frame, resultFrame,15);
                cv::imshow("Video", frame);
                cv::imshow("Blur Quantize", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'c':
            // Cartoonization of image
            while (true) {
                *capdev >> frame;
                cartoon(frame, resultFrame, 15, 15);
                cv::imshow("Video", frame);
                cv::imshow("Cartoonization", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'n':
            // Get Negative of an image
            while (true) {
                *capdev >> frame;
                negative(frame, resultFrame);
                cv::imshow("Video", frame);
                cv::imshow("Negative", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'p':
            // Change Color Palette of the image
            while (true) {
                *capdev >> frame;
                changeColorPalette(frame, resultFrame);
                cv::imshow("Video", frame);
                cv::imshow("Color Palette", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        case 'd':
            // Apply Laplacian filter for edge detection
            while (true) {
                *capdev >> frame;
                laplacianFilter(frame, resultFrame);
                cv::imshow("Video", frame);
                cv::imshow("Laplacian filter", resultFrame);
                key = waitKey(1);
                if (key == 'q') {
                    destroyAllWindows();
                    break;
                }
            }
            break;

        }


    }

    // Closes the camera
    delete capdev;
    return(0);
}
