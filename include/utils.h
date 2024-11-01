#pragma once
/// \file colors.h
/// \brief Header file defining color constants and utility functions.
///
/// This header file contains a set of predefined colors for the Cityscapes dataset,
/// a structure to hold clicked point data, and utility functions for overlaying masks
/// on images and handling mouse events.
///
/// \author Hamdi Boukamcha
/// \date 2024
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Global variables to store the selected point and image
Point selectedPoint;
bool pointSelected = false;
float resizeScale = 1.0f; // Scale factor for resizing

// Colors
const std::vector<cv::Scalar> CITYSCAPES_COLORS = {
    cv::Scalar(128, 64, 128),
    cv::Scalar(232, 35, 244),
    cv::Scalar(70, 70, 70),
    cv::Scalar(156, 102, 102),
    cv::Scalar(153, 153, 190),
    cv::Scalar(153, 153, 153),
    cv::Scalar(30, 170, 250),
    cv::Scalar(0, 220, 220),
    cv::Scalar(35, 142, 107),
    cv::Scalar(152, 251, 152),
    cv::Scalar(180, 130, 70),
    cv::Scalar(60, 20, 220),
    cv::Scalar(0, 0, 255),
    cv::Scalar(142, 0, 0),
    cv::Scalar(70, 0, 0),
    cv::Scalar(100, 60, 0),
    cv::Scalar(90, 0, 0),
    cv::Scalar(230, 0, 0),
    cv::Scalar(32, 11, 119),
    cv::Scalar(0, 74, 111),
    cv::Scalar(81, 0, 81)
};

/// \struct PointData
/// \brief Structure to hold clicked point coordinates.
/// 
/// This structure stores the coordinates of a clicked point
/// and a flag indicating whether the point has been clicked.
struct PointData {
    cv::Point point; ///< The coordinates of the clicked point.
    bool clicked;    ///< Flag indicating if the point was clicked.
};

// Mouse callback function to capture the clicked point
void mouseCallback(int event, int x, int y, int flags, void* param) {
    if (event == EVENT_LBUTTONDOWN) {
        selectedPoint = Point(x, y);
        pointSelected = true;
        cout << "Point selected (in resized image): " << selectedPoint << endl;
    }
}

// Example overlay function
void overlay(Mat& image, const Mat& mask) {
    // Placeholder for the overlay logic
    // This function should blend the mask with the original image
    addWeighted(image, 0.5, mask, 0.5, 0, image);
}

/// \brief Overlays a mask on the given image.
/// 
/// This function overlays a colored mask on the input image 
/// using a specified transparency (alpha) and optionally shows
/// the contour edges of the mask.
/// 
/// \param image The image on which to overlay the mask.
/// \param mask The mask to overlay on the image.
/// \param color The color of the overlay mask (default is CITYSCAPES_COLORS[0]).
/// \param alpha The transparency level of the overlay (default is 0.8).
/// \param showEdge Whether to show the contour edges of the mask (default is true).
void overlay(Mat& image, Mat& mask, cv::Scalar color = cv::Scalar(128, 64, 128), float alpha = 0.8f, bool showEdge = true)
{
    // Draw mask
    Mat ucharMask(image.rows, image.cols, CV_8UC3, color);
    image.copyTo(ucharMask, mask <= 0);
    addWeighted(ucharMask, alpha, image, 1.0 - alpha, 0.0f, image);

    // Draw contour edge
    if (showEdge)
    {
        vector<vector<cv::Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(mask <= 0, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
        drawContours(image, contours, -1, Scalar(255, 255, 255), 2);
    }
}

/// \brief Handles mouse events for clicking on the image.
/// 
/// This function processes mouse events, storing the clicked
/// point coordinates in the provided PointData structure.
/// 
/// \param event The type of mouse event.
/// \param x The x-coordinate of the mouse event.
/// \param y The y-coordinate of the mouse event.
/// \param flags Any relevant flags associated with the mouse event.
/// \param userdata User data pointer to store the PointData structure.
void onMouse(int event, int x, int y, int flags, void* userdata) {
    PointData* pd = (PointData*)userdata;
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Save the clicked coordinates
        pd->point = cv::Point(x, y);
        pd->clicked = true;
    }
}

/// \brief Segments the image based on a user-selected point.
/// 
/// This function allows the user to click on a point within the image, which is then 
/// used to perform segmentation using the NanoSam model. The segmented result is 
/// overlaid on the original image and displayed in the same window. The final image 
/// is saved to the specified output path.
/// 
/// \param nanosam Reference to the SpeedSam model used for segmentation.
/// \param imagePath Path to the input image.
/// \param outputPath Path to save the segmented output image.
void segmentWithPoint(SpeedSam& nanosam, const string& imagePath, const string& outputPath) {
    // Load the image from the specified path
    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "Error: Unable to load image from " << imagePath << endl;
        return;
    }

    // Resize the image for easier viewing
    Mat resizedImage;
    const int maxWidth = 800; // Maximum width for display
    if (image.cols > maxWidth) {
        resizeScale = static_cast<float>(maxWidth) / image.cols;
        resize(image, resizedImage, Size(), resizeScale, resizeScale);
    }
    else {
        resizedImage = image;
    }

    // Display the image in a window
    namedWindow("Select Point", WINDOW_AUTOSIZE);
    setMouseCallback("Select Point", mouseCallback, nullptr);
    imshow("Select Point", resizedImage);

    // Wait indefinitely until the user clicks a point
    while (!pointSelected) {
        waitKey(10); // Small delay to prevent high CPU usage
    }

    // Scale the selected point back to the original image size
    Point originalPoint(static_cast<int>(selectedPoint.x / resizeScale),
        static_cast<int>(selectedPoint.y / resizeScale));
    cout << "Point mapped to original image: " << originalPoint << endl;

    // Label indicating that the prompt point corresponds to the foreground class
    vector<float> labels = { 1.0f };

    // Perform prediction using the NanoSam model at the specified prompt point
    auto mask = nanosam.predict(image, { originalPoint }, labels);

    // Overlay the segmentation mask on the original image
    overlay(image, mask);

    // Save the resulting image with the overlay to the specified output path
    imwrite(outputPath, image);

    if (image.cols > maxWidth) {
        resizeScale = static_cast<float>(maxWidth) / image.cols;
        resize(image, resizedImage, Size(), resizeScale, resizeScale);
    }
    else {
        resizedImage = image;
    }

    // Update the same window with the segmented image
    imshow("Select Point", resizedImage);
    waitKey(0); // Wait for another key press to close the window
    destroyAllWindows(); // Close all OpenCV windows
}

/// \brief Segments an image using bounding box information and saves the result.
/// 
/// This function loads an image from the specified path, 
/// performs segmentation using the provided bounding boxes, 
/// and saves the segmented image with an overlay to the specified output path.
/// 
/// \param nanosam The SpeedSam model used for segmentation.
/// \param imagePath The path to the input image.
/// \param outputPath The path where the output image will be saved.
/// \param bbox A vector of Points representing the top-left and bottom-right 
///             corners of the bounding box for segmentation.
void segmentBbox(SpeedSam& nanosam, string imagePath, string outputPath) {
    // Load the image from the specified path
    auto image = imread(imagePath);

    // Check if the image was loaded successfully
    if (image.empty()) {
        cerr << "Error: Unable to load image from " << imagePath << endl;
        return;
    }

    // Create a window for user interaction
    namedWindow("Select and View Result", cv::WINDOW_AUTOSIZE);

    // Let the user select the bounding box
    cv::Rect bbox = selectROI("Select and View Result", image, false, false);

    // Check if a valid bounding box was selected
    if (bbox.width == 0 || bbox.height == 0) {
        cerr << "No valid bounding box selected." << endl;
        return;
    }

    // Convert the selected bounding box to a vector of points
    vector<Point> bboxPoints = {
        Point(bbox.x, bbox.y),                                // Top-left point
        Point(bbox.x + bbox.width, bbox.y + bbox.height)      // Bottom-right point
    };

    // Labels corresponding to the bounding box classes
    // 2 : Bounding box top-left, 3 : Bounding box bottom-right
    vector<float> labels = { 2, 3 };

    // Perform prediction using the NanoSam model with the given bounding boxes and labels
    auto mask = nanosam.predict(image, bboxPoints, labels);

    // Overlay the segmentation mask on the original image
    overlay(image, mask);

    // Draw the bounding box on the image
    rectangle(image, bboxPoints[0], bboxPoints[1], cv::Scalar(255, 255, 0), 3);

    // Display the updated image in the same window
    imshow("Select and View Result", image);
    waitKey(0);

    // Save the resulting image to the specified output path
    imwrite(outputPath, image);
}
