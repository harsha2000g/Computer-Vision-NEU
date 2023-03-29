/*
  Sri Harsha Gollamudi

  This file contains the main function for the code and it calls the functions from helpers.cpp to execute the tasks.
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "helpers.h"

using namespace cv;

int main(int argc, char* argv[]) {

    char img[256], dir[256], featureType[256], distanceMetric[256];
    int nMatches;

    // If there are not enough or more than enough arguments then give a prompt about the arguments and exit the program
    if (argc != 6) {
        printf("The five arguments are: [target image address] [image directory address] [feature set] [distance metric] [images to match]\n");
        return(0);
    }

    strcpy_s(img, argv[1]);
    strcpy_s(dir, argv[2]);
    strcpy_s(featureType, argv[3]);
    strcpy_s(distanceMetric, argv[4]);
    nMatches = std::stoi(argv[5]);

    Mat image = imread(img);

    int bins = 8;

    // Baseline Matching Case
    if (strcmp(featureType, "Baseline") == 0) {

        std::vector<double> baselineFeature = baselineMatching(image);

        std::vector<std::pair<std::vector<double>, char*>> baselineFeatures = baselineMatchingDirectory(dir);

        std::vector<std::pair<double, char*>> distances;

        if (strcmp(distanceMetric, "SD") == 0) {
            printf("Baseline Matching with Squared Difference for %s\n", img);
            distances = squaredDifference(baselineFeatures, baselineFeature);
        }
        else if (strcmp(distanceMetric, "I") == 0) {
            printf("Baseline Matching with Intersection for %s\n", img);
            distances = intersection(baselineFeatures, baselineFeature);
        }
        

        topNMatches(distances, nMatches);

        return(0);

    }
    // Histogram Matching Case
    else if (strcmp(featureType, "Histogram") == 0) {

        std::vector<double> histogramFeature = generateHistogram(image, bins);

        std::vector<std::pair<std::vector<double>, char*>> histogramFeatures = histogramMatchingDirectory(dir, bins);

        std::vector<std::pair<double, char*>> distances;

        if (strcmp(distanceMetric, "SD") == 0) {
            printf("Histogram Matching with Squared Difference for %s\n", img);
            distances = squaredDifference(histogramFeatures, histogramFeature);
        }
        else if (strcmp(distanceMetric, "I") == 0) {
            printf("Histogram Matching with Intersection for %s\n", img);
            distances = intersection(histogramFeatures, histogramFeature);
        }

        topNMatches(distances, nMatches);

        return(0);

    }
    // Multi-Histogram Matching Case
    else if (strcmp(featureType, "Multi-Histogram") == 0) {

        std::vector<std::vector<double>> multiHistogramFeature = generateQuadrantsHistograms(image, bins);

        std::vector<std::pair<std::vector<std::vector<double>>, char*>> multiHistogramFeatures = multiHistogramMatchingDirectory(dir, bins);

        std::vector<std::pair<double, char*>> distances = weightedIntersection(multiHistogramFeatures, multiHistogramFeature);

        printf("Histogram Matching with Intersection for %s\n", img);

        topNMatches(distances, nMatches);

        return(0);

    }
    // Texture 1 Matching Case
    else if (strcmp(featureType, "Texture") == 0) {

        std::vector<std::vector<double>> textureFeature = textureMatching(image, bins);

        std::vector<std::pair<std::vector<std::vector<double>>, char*>> textureFeatures = textureMatchingDirectory(dir, bins);

        std::vector<std::pair<double, char*>> distances = weightedIntersection(textureFeatures, textureFeature);

        printf("Texture Matching with Intersection for %s\n", img);

        topNMatches(distances, nMatches);

        return(0);

    }
    // Texture 2 Matching Case
    else if (strcmp(featureType, "Texture2") == 0) {

        std::vector<std::vector<double>> textureFeature2 = textureMatching2(image, bins);

        std::vector<std::pair<std::vector<std::vector<double>>, char*>> textureFeatures2 = textureMatchingDirectory2(dir, bins);

        std::vector<std::pair<double, char*>> distances = weightedIntersection(textureFeatures2, textureFeature2);

        printf("Texture Matching 2 with Intersection for %s\n", img);

        topNMatches(distances, nMatches);

        return(0);

    }
    // CBIR 1 Case
    else if (strcmp(featureType, "CBIR") == 0) {

        std::vector<std::pair<double, char*>> CBIRdistances = CBIR(image, dir, bins);

        printf("CBIR for %s\n", img);

        topNMatches(CBIRdistances, nMatches);

        return(0);

    }
    // CBIR 2 Case
    else if (strcmp(featureType, "CBIR2") == 0) {

        std::vector<std::pair<double, char*>> CBIRdistances = CBIR2(image, dir, bins);

        printf("CBIR 2 for %s\n", img);

        topNMatches(CBIRdistances, nMatches);

        return(0);

    }

    return(0);
}


