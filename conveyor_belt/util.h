#ifndef _UTIL_H_
#define _UTIL_H_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "parameters.h"

using namespace cv;

extern RNG rng;
extern const int imageWidth;
extern const int imageHeight;
extern Parameters param;

//******************************************************************************

const double PI = 3.1415927;

void showImage(const string& winname, Mat img, int autosize = 0, int delay = 0);
vector<vector<Point> > extractContours(const Mat& img);
void cropFrame(Mat img);

bool isInFOV(const Rect& rect, bool isHorizontalBelt);
RotatedRect orientedBoundingBox(const vector<Point>& contour);

//******************************************************************************

template <class T>
double distance2D(T p1, T p2)
{
	return sqrt(pow(double(p1.x - p2.x), 2) + pow(double(p1.y - p2.y), 2)); 
}

template <class T>
double distance3D(T p1, T p2)
{
	return sqrt(pow(double(p1.x - p2.x), 2) 
			  + pow(double(p1.y - p2.y), 2)
			  + pow(double(p1.z - p2.z), 2));
}

#endif