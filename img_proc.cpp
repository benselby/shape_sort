#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

Mat kmeanz(Mat src, char * type, int K) {
	Mat p = Mat::zeros(src.rows*src.cols, 5, CV_32F);
	Mat bestLabels, centers, clustered;
	vector<Mat> bgr;

	Mat src_seg;
	if (!strcmp(type, "l")) {
		cvtColor(src, src_seg, CV_BGR2Lab);
	}
	else if (!strcmp(type, "h")) {
		cvtColor(src, src_seg, CV_BGR2HSV);
	}

	cv::split(src_seg, bgr);
	// i think there is a better way to split pixel bgr color
	for (int i = 0; i<src.rows*src.cols; i++) {
		p.at<float>(i, 0) = (i / src.cols) / src.rows;
		p.at<float>(i, 1) = (i%src.cols) / src.cols;
		if (!strcmp(type, "h")) {
			//p.at<float>(i, 2) = bgr[0].data[i] / 255.0;
			p.at<float>(i, 2) = bgr[2].data[i] / 255.0;
			//p.at<float>(i, 3) = bgr[2].data[i] / 255.0;
		}
		else if (!strcmp(type, "l")) {
			p.at<float>(i, 2) = bgr[1].data[i] / 255.0;
			p.at<float>(i, 3) = bgr[2].data[i] / 255.0;
		}
	}

	cv::kmeans(p, K, bestLabels,
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 0.01),
		3, KMEANS_PP_CENTERS, centers);

	clustered = Mat(bestLabels).reshape(0, src.rows);

	return clustered;
}

Mat Segments(Mat crop, int K, Mat clustered, int k){
	Mat temp;

	temp = crop.clone();
	for (int i = 0; i < clustered.rows; i++) {
		for (int j = 0; j < clustered.cols; j++) {
			if (clustered.at<int>(i, j) == k)
			{
			}
			else
			{
				temp.at<Vec3b>(i, j)[0] = 0;
				temp.at<Vec3b>(i, j)[1] = 0;
				temp.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}

	return temp;
}

Point get_centroid(vector<Point> &pts, Mat &img)
{
	int x_sum = 0;
	int y_sum = 0;

	for (size_t i = 0; i < pts.size(); i++)
	{
		x_sum += pts.at(i).x;
		y_sum += pts.at(i).y;
	}

	Point centroid = Point(x_sum / pts.size(), y_sum / pts.size());

	// Draw the centroid on the image
	circle(img, centroid, 3, Scalar(255, 0, 255), 2);
	return centroid;
}

vector<Point> get_shape_points(int argc, char* argv) {
	Mat src;
	Mat clustered;
	Mat temp, temp2;
	vector<Point> result_points;
	//src = imread("D:/School/Yr4T2/ME547/Screenshot from 2014-03-31 15_29_18.png");
	src = imread(argv, 1);

	//imshow("original", src);

	int K = 4;
	Mat crop = src(Rect(400, 200, 500, 600));
	//sum = Mat::zeros(crop.rows, crop.cols, CV_32F);
	//imshow("crop", crop);
	Mat sum = Mat::zeros(crop.rows, crop.cols, CV_8UC3);

	clustered = kmeanz(crop, "l", K);

	for (int k = 0; k < K; k++) {
		temp = Segments(crop, K, clustered, k);
		int K2 = 4;
		Mat clustered2 = kmeanz(temp, "h", K2);

		for (int i = 0; i < K2; i++) {
			temp2 = Segments(temp, K2, clustered2, i);
			Mat temp3;
			cvtColor(temp2, temp3, CV_BGR2HSV);
			if (mean(temp3)[2] > 0.7 && mean(temp2)[1] > 0.5 && mean(temp2)[1] < 3) {
				inRange(temp2, Scalar(10, 10, 10), Scalar(255, 255, 255), temp2);

				Mat filled_img = Mat::zeros(temp2.rows, temp2.cols, CV_8UC3);
				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;
				double min_area = 1000.0;
				double area = 0, amax = 0, amax2 = 0;
				size_t q1 = 0, q2 = 0;

				cout << "Filling holes in the segemented shapes..." << endl;
				findContours(temp2.clone(), contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
				cout << "No. of identified contours: " << contours.size() << endl;
				for (size_t q = 0; q < contours.size(); q++)
				{
					area = contourArea(contours[q]);

					// skip contours which are too small
					if (area < min_area)
						continue;

					if (area > amax && area > amax2){
						amax2 = amax;
						q2 = q1;
						amax = area;
						q1 = q;
					}
					else if (area > amax2){
						amax2 = area;
						q2 = q;
					}
				}

				
				drawContours(filled_img, contours, q1, CV_RGB(255, 255, 255), CV_FILLED, 8, hierarchy, 0);
				if (amax > 2000)
					drawContours(filled_img, contours, q2, CV_RGB(0, 0, 0), CV_FILLED, 8, hierarchy, 0);
				sum += filled_img;
				//imshow("Seg", sum);

				//waitKey();
			}
		}
	}

	cout << "Finding edges of segmented shapes..." << endl;
	vector<vector<Point> > filled_contours;
	vector<Vec4i> filled_hierarchy;
	vector<Point> poly_approx;
	Mat filled_grey;
	double area2;
	cvtColor(sum, filled_grey, CV_BGR2GRAY);
	findContours(filled_grey, filled_contours, filled_hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cout << "No. of identified contours: " << filled_contours.size() << endl;
	for (size_t i = 0; i < filled_contours.size(); i++)
	{
		area2 = contourArea(filled_contours[i]);
		if (area2 > 3000)
			continue;

		approxPolyDP(filled_contours[i], poly_approx, 7, true);
		// draw lines connecting each of the vertices
		cout << "Size of polygon: " << poly_approx.size() << endl;

		Scalar colour(rand() % 255, rand() % 255, rand() % 255);

		for (size_t j = 0; j<poly_approx.size() - 1; j++)
		{
			line(sum, poly_approx.at(j), poly_approx.at(j + 1), colour, 2);
		}

		line(sum, poly_approx.at(poly_approx.size() - 1), poly_approx.at(0), colour, 2);

		Point centroid = get_centroid(filled_contours[i], sum);

		// Find the midpoint of any one side and return the angle between that
		// point and the centroid - this is the grasping angle        
	        int mid_x = (poly_approx.at(0).x + poly_approx.at(1).x) / 2;
		int mid_y = (poly_approx.at(0).y + poly_approx.at(1).y) / 2;
		Point midpoint = Point(mid_x, mid_y);
		line(sum, centroid, midpoint, Scalar(0, 255, 0), 2);
		result_points.at(2*i) = centroid;
		result_points.at(2*i+1) = midpoint;
		//imshow("partial", filled_img);
		//waitKey(0);
	}
	imshow("Filled Holes", sum);
	//imwrite("crop3.png", crop);
	//imwrite("sumcen3.png", sum);
	waitKey(0);

	return result_points;
}

//int main (int argc, char** argv)
//{
//	get_shape_points(argc, argv);
//	return 0;
//}
