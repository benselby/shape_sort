#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

double getOrientation(vector<Point> &pts, Mat &img);

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

    // Find all the contours in the summed image and then fill them
    Mat filled_img = Mat::zeros(img.rows, img.cols, CV_8UC3);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    double min_area = 10.0;
    
    findContours(sum.clone(), contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    cout<<"No. of identified contours: "<<contours.size()<<endl; 
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        // skip contours which are too small
        if (area < min_area) 
            continue;

        drawContours(filled_img, contours, i, CV_RGB(255, 255, 255), CV_FILLED, 8, hierarchy, 0);
    }
  
    cout<<"Finding edges of segmented shapes..."<<endl;
    vector<vector<Point> > filled_contours;
    vector<Vec4i> filled_hierarchy;
    vector<Point> poly_approx;
    Mat filled_grey;
    cvtColor(filled_img, filled_grey, CV_BGR2GRAY);
    findContours(filled_grey, filled_contours, filled_hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    cout<<"No. of identified contours: "<<filled_contours.size()<<endl; 
    for (size_t i = 0; i < filled_contours.size(); i++)
    {
        approxPolyDP( filled_contours[i], poly_approx, 1, true); 
        // draw lines connecting each of the vertices
        cout<<"Size of polygon: "<<poly_approx.size()<<endl;        
        
        Scalar colour( rand()%255, rand()%255, rand()%255 );

        for ( size_t j = 0; j<poly_approx.size()-1; j++ )
        {
            line( filled_img, poly_approx.at(j), poly_approx.at(j+1), colour, 2);
        }

        line( filled_img, poly_approx.at(poly_approx.size()-1), poly_approx.at(0), colour, 2);
        imshow("partial", filled_img);
        waitKey(0);
    }

    // Display the images:
    imshow("Raw", img);
//    imshow("HSV", img_hsv);
//    imshow("Square - Color Segmented", square);
//    imshow("Triangle - Color Segmented", triangle);
//    imshow("Octagon - Color Segmented", octagon);
    imshow("Sum", sum);
    imshow("Filled Holes", filled_img);
    waitKey(0);
    imwrite("segmented_shapes.png", sum); 
    return 0;
}

double getOrientation(vector<Point> &pts, Mat &img)
{
    //Construct a buffer used by the pca analysis
    Mat data_pts = Mat(pts.size(), 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
 
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
 
    //Store the position of the object
    Point pos = Point(pca_analysis.mean.at<double>(0, 0),
                      pca_analysis.mean.at<double>(0, 1));
 
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
 
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    }
 
    // Draw the principal components
    circle(img, pos, 3, CV_RGB(255, 0, 255), 2);
    line(img, pos, pos + 0.02 * Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]) , CV_RGB(255, 255, 0));
    line(img, pos, pos + 0.02 * Point(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]) , CV_RGB(0, 255, 255));
 
    return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
}
