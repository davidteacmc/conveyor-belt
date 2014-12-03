#ifndef _FG_OBJECT_H_
#define _FG_OBJECT_H_

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "util.h"

using namespace std;
using namespace cv;

class FgObject
{
public:
	FgObject();
	FgObject(const FgObject& obj);
	FgObject(vector<Point> contour, Mat alphaMap = Mat());
	~FgObject();

	void setContour(const vector<Point>& contour) { _contour = contour; }
    void setRectColor(const Scalar& color) { _rectColor = color; }
	void setCSSCost0(const double& cost) { _cssCost0 = cost; }
	void setCSSCost1(const double& cost) { _cssCost1 = cost; }
	void setCSSCost2(const double& cost) { _cssCost2 = cost; }
	void setAlphaMap(const Mat& alphaMap) { _alphaMap = alphaMap.clone(); }

	void moveContour(const Point& vec);

	int				getFrameNum() const { return _frameNum; }
	vector<Point>	getContour() const { return _contour; }
	Scalar			getRectColor() const { return _rectColor; }
	double			getCSSCost0() const { return _cssCost0; }
	double			getCSSCost1() const { return _cssCost1; }
	double			getCSSCost2() const { return _cssCost2; }
	Mat				getAlphaMap() const	{ return _alphaMap;	}

	Rect			boundingBox() const { return boundingRect(_contour); }
	RotatedRect		orientedBox() const { return orientedBoundingBox(_contour); }
	Point2f			center() const { return orientedBoundingBox(_contour).center; }
    
	void update(const FgObject& obj);
	void draw(Mat img);
	void drawShifted(Mat img);

private:
	int				_frameNum;
	Scalar			_rectColor;
	vector<Point>	_contour;
	double			_cssCost0;
	double			_cssCost1;
	double			_cssCost2;
	Mat				_alphaMap;
	
public:
	bool			_updated;
};

double overlappingArea(const FgObject& obj1, const FgObject& obj2);

#endif