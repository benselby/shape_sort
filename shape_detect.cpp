#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main( int argsc, char** argsv )
{
    cout<<"Segmenting the input image based on colour to resolve the blocks"<<endl;
    Mat img = imread("sample-imgs/basic_img.png");
//    Mat img = imread("sample-imgs/test-img1.png");
    // Convert the image into HSV format for convenient colour segmentation
    Mat img_hsv;
    cvtColor( img, img_hsv, CV_BGR2HSV );
   
    Mat square;
    Scalar lower_range = Scalar(17./2, 0.48*255, 0.55*255);
    Scalar upper_range = Scalar(27./2, 0.74*255, 0.7*255);
    inRange( img_hsv, lower_range, upper_range, square); 
    
    Mat triangle;
    lower_range = Scalar(88./2, 0.5*255, 0.54*255);
    upper_range = Scalar(106./2, 0.75*255, 0.62*255);
    inRange( img_hsv, lower_range, upper_range, triangle); 
    
    Mat octagon;
    lower_range = Scalar(210./2, 0.35*255, 0.55*255);
    upper_range = Scalar(220./2, 0.55*255, 0.71*255);
    inRange( img_hsv, lower_range, upper_range, octagon); 
    
    // Sum the images into one
    Mat sum;
    add(square, triangle, sum);
    add(sum, octagon, sum);

    // Find all the contours in the summed image:
    Mat filled_img = Mat::zeros(img.rows, img.cols, CV_8UC3);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(sum.clone(), contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);

        //if (area < le2 || le5 < area) continue;
        drawContours(filled_img, contours, i, CV_RGB(255, 255, 255), CV_FILLED, 8, hierarchy, 0);
//        getOrientation(contours[i], filled_img);   
    }

    imshow("Raw", img);
//    imshow("HSV", img_hsv);
//    imshow("Square - Color Segmented", square);
//    imshow("Triangle - Color Segmented", triangle);
//    imshow("Octagon - Color Segmented", octagon);
    imshow("Sum", sum);
    imshow("Filled Holes", filled_img);
    cv::waitKey(0);
    imwrite("segmented_shapes.png", sum); 
    return 0;
}

