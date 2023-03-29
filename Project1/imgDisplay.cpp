#include <opencv2/opencv.hpp>
#include <cstdio>

using namespace cv;

int main(int argc, char* argv[]) {
    // Read image file
    Mat img = imread("imgs/1.jpg");

    // Create window to display image
    namedWindow("Image", WINDOW_NORMAL);

    // Show image in window
    imshow("Image", img);

    // Wait for user input
    char key;

    while (true) {
        key = waitKey(0);
        // Check for key presses
        switch (key) {
        case 'q':
            // Quit if q is pressed
            return 0;
            break;
        }
    }

    return 0;
}