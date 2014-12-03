#include <iostream>
using namespace std;
#include <cmath>
#include "util.h"
#include "opencv2/highgui/highgui.hpp"

// global variables
RNG	rng(12345);//rng(time(0));
const int imageWidth = 1440;
const int imageHeight = 1080;

// wrapper of the routines to show an image in the window
void showImage(const string& winname, Mat img, int autosize, int delay)
{
	namedWindow(winname, autosize);
	imshow(winname, img);
	waitKey(delay);
}

// wrapper of the routines to find contours in a grayscale image
vector<vector<Point> > extractContours(const Mat& img)
{
	vector<vector<Point> > contours;
    Mat tempImg = img.clone();
    findContours(tempImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point());
    tempImg.release();
	return contours;
}

// crop the image frame
void cropFrame(Mat img)
{
    for(int y = 0; y < img.rows; ++y){
        for(int x = 0; x < img.cols; ++x){
            if(x < 107 || x > 1338 || y > img.rows - 15)
                img.at<Vec3b>(y,x) = Vec3b(0,0,0);
        }
    }
}

// check if an object bounding box is fully inside the FOV
// isVerticalBelt: true = vertical
//                 false = horizontal
bool isInFOV(const Rect& rect, bool isVerticalBelt)
{
	Point tl = rect.tl();
	Point br = rect.br();
	bool left, right, top, bottom;
	if(isVerticalBelt){
		left = true;
		right = true;
		top = tl.y > 3;
		bottom = br.y < param.getFrameHeight() - 4;
	}
	else{
		left = tl.x > 5;
		right = br.x < param.getFrameWidth() - 5;
		top = true;
		bottom = true;
	}
	
	return (left && right && top && bottom);
}

// generate oriented bounding box
// find the rotation angle by principal component analysis (PCA)
RotatedRect orientedBoundingBox(const vector<Point>& contour)
{
	Mat mask8U = Mat::zeros(param.getFrameHeight(), param.getFrameWidth(), CV_8U);
	vector<vector<Point>> cs;
	cs.push_back(contour);
	drawContours(mask8U, cs, 0, Scalar(255), -1);
	//showImage("mask8U", mask8U, 1);

	vector<Point> fgPixels;
	fgPixels.reserve(int(contourArea(contour) + 0.5));
	for(int y = 0; y < mask8U.rows; ++y){
		for(int x = 0; x < mask8U.cols; ++x){
			if(mask8U.at<uchar>(y, x) == 255){
				fgPixels.push_back(Point(x, y));
			}
		}
	}
	
	RotatedRect orientedBox;
	if(contour.size() <= 2){
		if(contour.size() == 1){
			orientedBox.center = contour[0];
		}
		else{
			orientedBox.center.x = 0.5f*(contour[0].x + contour[1].x);
			orientedBox.center.y = 0.5f*(contour[0].x + contour[1].x);
			double dx = contour[1].x - contour[0].x;
			double dy = contour[1].y - contour[0].y;
			orientedBox.size.width = (float)sqrt(dx*dx + dy*dy);
			orientedBox.size.height = 0;
			orientedBox.angle = (float)atan2(dy, dx) * 180 / CV_PI;
		}
		return orientedBox;
	}
    
	Mat data = Mat::zeros(2, contour.size(), CV_32F);
	for(int j = 0; j < contour.size(); ++j){
		data.at<float>(0, j) = contour[j].x;
		data.at<float>(1, j) = contour[j].y;
	}

	PCA pcaObj = PCA(data, noArray(), CV_PCA_DATA_AS_COL);

	Mat result;
	pcaObj.project(data, result);

	// find two endpoints in principal component's direction      
	float maxU = 0, maxV = 0;
	float minU = 0, minV = 0;
	
	for(int j = 0; j < result.cols; ++j){
		float u = result.at<float>(0, j);
		float v = result.at<float>(1, j);
		if(u > 0 && u > maxU) 
			maxU = u;
		else if(u < 0 && u < minU)
			minU = u;
			
		if(v > 0 && v > maxV)
			maxV = v;  
		else if(v < 0 && v < minV)
			minV = v;
	}

	float cenU = 0.5*(maxU + minU);
	float cenV = 0.5*(maxV + minV);

	Mat cenUVMat = (Mat_<float>(2, 1) << cenU, cenV);
	Mat cenXYMat = pcaObj.backProject(cenUVMat);

	Point cen(cenXYMat.at<float>(0, 0), cenXYMat.at<float>(1, 0));

	float width = maxU - minU;
	float height = maxV - minV;

	Mat pc = pcaObj.eigenvectors;

	//cout << "V = " << pcaObj.eigenvectors << endl;
	//cout << "lambda = " << pcaObj.eigenvalues << endl;

	float pcx = pc.at<float>(0, 0);
	float pcy = pc.at<float>(0, 1);
	float theta = atan2(pcy, pcx) * 180 / 3.1415927;
		
	orientedBox.center = cen;
	orientedBox.size = Size2f(width, height);
	orientedBox.angle = theta;
	return orientedBox;
}