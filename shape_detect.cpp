#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Point get_centroid( vector<Point> &pts, Mat &img );

int main( int argsc, char** argsv )
{
    cout<<"Segmenting the input image based on colour to resolve the blocks"<<endl;
//    Mat img = imread("sample-imgs/basic_img.png");
    Mat img = imread("sample-imgs/test-img1.png");
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
    
    Mat s_slot, t_slot, o_slot, tmp_slot;
    lower_range = Scalar(340./2, 0.30*255, 0.35*255);
    upper_range = Scalar(359./2, 0.74*255, 0.55*255);
    inRange( img_hsv, lower_range, upper_range, s_slot); 
    
    lower_range = Scalar(0./2, 0.30*255, 0.35*255);
    upper_range = Scalar(4./2, 0.74*255, 0.55*255);
    inRange( img_hsv, lower_range, upper_range, tmp_slot); 
    add(s_slot, tmp_slot, s_slot);
    
    lower_range = Scalar(140./2, 0.2*255, 0.10*255);
    upper_range = Scalar(190./2, 0.4*255, 0.3*255);
    inRange( img_hsv, lower_range, upper_range, t_slot); 
    
    lower_range = Scalar(230./2, 0.33*255, 0.30*255);
    upper_range = Scalar(260./2, 0.56*255, 0.5*255);
    inRange( img_hsv, lower_range, upper_range, o_slot); 
    
    Mat slot_sum;
    add(s_slot, t_slot, slot_sum);
    add(slot_sum, o_slot, slot_sum);
    imshow("Slots", slot_sum);
    
    // Sum the images into one
    Mat sum;
    add(square, triangle, sum);
    add(sum, octagon, sum);

    // Find all the contours in the summed image and then fill them
    Mat filled_img = Mat::zeros(img.rows, img.cols, CV_8UC3);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    double min_area = 10.0;
    
    cout<<"Filling holes in the segemented shapes..."<<endl;
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
    // Create a single channel version of the filled image to find contours
    Mat filled_grey;
    cvtColor(filled_img, filled_grey, CV_BGR2GRAY);
    findContours(filled_grey, filled_contours, filled_hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    cout<<"No. of identified contours: "<<filled_contours.size()<<endl; 
    for (size_t i = 0; i < filled_contours.size(); i++)
    {
        approxPolyDP( filled_contours[i], poly_approx, 2, true); 
        // draw lines connecting each of the vertices
        cout<<"Size of polygon: "<<poly_approx.size()<<endl;        
        
        Scalar colour( rand()%255, rand()%255, rand()%255 );

        for ( size_t j = 0; j<poly_approx.size()-1; j++ )
        {
            line( filled_img, poly_approx.at(j), poly_approx.at(j+1), colour, 2);
        }

        line( filled_img, poly_approx.at(poly_approx.size()-1), poly_approx.at(0), colour, 2);
        
        Point centroid = get_centroid( filled_contours[i], filled_img );
        
        // Find the midpoint of any one side and return the angle between that
        // point and the centroid - this is the grasping angle
        int mid_x = (poly_approx.at(0).x + poly_approx.at(1).x)/2;
        int mid_y = (poly_approx.at(0).y + poly_approx.at(1).y)/2;
        Point midpoint = Point(mid_x, mid_y);
        line(filled_img, centroid, midpoint, Scalar(0,255,0),2 );

        //imshow("partial", filled_img);
        //waitKey(0);
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

/*
*   Finds the centroid as a uniform sum of all the pixels in a given
*   blob. Draws the centre on the specified image.
*/
Point get_centroid(vector<Point> &pts, Mat &img)
{
    int x_sum = 0;
    int y_sum = 0;
    
    for (size_t i = 0; i < pts.size(); i++ )
    {
        x_sum += pts.at(i).x;
        y_sum += pts.at(i).y;
    }   

    Point centroid = Point( x_sum/pts.size(), y_sum/pts.size() );
                           
    // Draw the centroid on the image
    circle( img, centroid, 3, Scalar(255,0,255), 2 );
    return centroid;
}
