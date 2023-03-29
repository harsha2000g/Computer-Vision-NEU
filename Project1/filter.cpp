#include <opencv2/opencv.hpp>
#include <cstdio>
#include <random>
#include <iostream>
#include "filter.h"

using namespace std;
using namespace cv;

int greyscale(cv::Mat &src, cv::Mat &dst) {

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
                for (int i = -2; i <= 2; i++){
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



int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {

    int quantize;
    // Size of bucket obtained by dividing it by levels
    int buckerSize = 255 / levels;

    // Initializing the destination image
    dst.create(src.size(), src.type());

    // Using Gaussian filter to blur the image
    blur5x5(src, src);

    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            for (int clr = 0; clr < 3; clr++) {
                // Quantizing the pixel value by dividing and then multiplying it by the bucket size
                quantize = (src.at<cv::Vec3b>(row, col)[clr] / buckerSize) * buckerSize;
                // Assigning the quantized values to the dst image
                dst.at<cv::Vec3b>(row, col)[clr] = quantize;

            }
        }
    }
    return 0;
}

int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold) {

    // Initializing the destination image
    dst.create(src.size(), src.type());

    // Temporary matrices to hold intermediate results
    cv::Mat frameX, frameY, temp;

    // Applying Sobel X and Y to the src image
    sobelX3x3(src, frameX);
    sobelY3x3(src, frameY);

    // Calculating the magnitude of the gradient using the Sobel X and Y values
    magnitude(frameX, frameY, temp);

    // Applying blur quantize on the src image
    blurQuantize(src, dst, levels);

    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            for (int clr = 0; clr < 3; clr++) {
                // Converting all pixels with value above threshold to black pixels to outline edges
                if (temp.at<Vec3s>(row, col)[clr] > magThreshold) {
                    dst.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
                }
            }
        }
    }

    return 0;
}


void negative(cv::Mat& src, cv::Mat& dst) {

    // Initializing the destination image
    dst.create(src.size(), src.type());

    for (int row = 0; row < src.rows; row++) {
        // Get the row pointer in Vec3b format for both src and dst 
        cv::Vec3b* srcptr = src.ptr<Vec3b>(row);
        cv::Vec3b* dstptr = dst.ptr<Vec3b>(row);


        for (int col = 0; col < src.cols; col++) {
            // Subtracting the each channel value of the src from 255 and storing them in the destination image.
            dstptr[col][0] = 255 - srcptr[col][0];
            dstptr[col][1] = 255 - srcptr[col][1];
            dstptr[col][2] = 255 - srcptr[col][2];

        }
    }


}

void changeColorPalette(cv::Mat& src, cv::Mat& dst) {

    // Initializing the destination image
    dst.create(src.size(), src.type());
    
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            // Changing the color palette by swapping the values of the Red, Blue and Green channels in dst image
            dst.at<Vec3b>(row, col) = Vec3b(src.at<Vec3b>(row, col)[2], src.at<Vec3b>(row, col)[1], src.at<Vec3b>(row, col)[0]);
        }
    }

}

void laplacianFilter(cv::Mat& src, cv::Mat& dst) {

    // Initializing the destination image
    dst.create(src.size(), src.type());

    for (int row = 1; row < src.rows - 1; row++) {
        for (int col = 1; col < src.cols - 1; col++) {
            // Getting the value of the centermost pixel in the 3x3 area
            cv::Vec3b centerPixel = src.at<cv::Vec3b>(row, col);

            // Loop to access all the values around the center pixel in 3x3 area
            int newColor[3] = {0,0,0};
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    // Pass the center pixel
                    if (i == 0 && j == 0) {
                        continue;
                    }
                    // Storing the sum of all surrounding pixels in newColor array
                    cv::Vec3b currentPixel = src.at<cv::Vec3b>(row + i, col + j);
                    for (int clr = 0; clr <= 2; clr++) {
                        newColor[clr] += currentPixel[clr];
                    }
                    
                }
            }

            // Subtracting the average value of the surrounding pixels from the center pixel
            for (int clr = 0; clr <= 2; clr++) {
                newColor[clr] = centerPixel[clr] - newColor[clr] / 8;
            }
            
            // Normalizing and assigning the values new values to the dst image
            dst.at<cv::Vec3b>(row, col) = cv::Vec3b(cv::saturate_cast<uchar>(newColor[0]), 
                cv::saturate_cast<uchar>(newColor[1]), cv::saturate_cast<uchar>(newColor[2]));
        }
    }
}


