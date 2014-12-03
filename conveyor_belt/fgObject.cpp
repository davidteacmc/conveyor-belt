#include "fgObject.h"

FgObject::FgObject() :
	_frameNum(param.getFrameNum()),
	_rectColor(Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255))),
	_cssCost0(0),
	_cssCost1(0),
	_cssCost2(0),
	_alphaMap(Mat()),
	_updated(false)
{}

FgObject::FgObject(const FgObject& obj) :
	_frameNum(obj._frameNum),
	_contour(obj._contour),
	_rectColor(obj._rectColor),
	_cssCost0(obj._cssCost0),
	_cssCost1(obj._cssCost1),
	_cssCost2(obj._cssCost2),
	_alphaMap(obj._alphaMap),
	_updated(obj._updated)
{}

FgObject::FgObject(vector<Point> contour, Mat alphaMap) :
	_frameNum(param.getFrameNum()),
	_contour(contour),
	_rectColor(Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255))),
	_cssCost0(0),
	_cssCost1(0),
	_cssCost2(0),
	_alphaMap(alphaMap),
	_updated(false)
{}

FgObject::~FgObject()
{

}

void FgObject::update(const FgObject& obj)
{
	_contour = obj.getContour();
	_updated = true;
}

void FgObject::moveContour(const Point& vec)
{
	for(int i = 0; i < _contour.size(); ++i){
		_contour[i] += Point(vec);
	}
}

void FgObject::draw(Mat img)
{
	Rect boundRect = cv::boundingRect(_contour);
	//rectangle(img, boundRect.tl(), boundRect.br(), _rectColor, 4);
	RotatedRect orientedRect = orientedBoundingBox(_contour);
	Point2f pts [4];
	orientedRect.points(pts);
	for(int i = 0; i < 4; ++i)
		line(img, pts[i], pts[(i+1)%4], _rectColor, 4);
}

void FgObject::drawShifted(Mat img)
{
	Rect boundRect = cv::boundingRect(_contour);
	Point tl = Point(boundRect.tl().x + 104, boundRect.tl().y);
	Point br = Point(boundRect.br().x + 104, boundRect.br().y);
	rectangle(img, tl, br, _rectColor, 4);
}

double overlappingArea(const FgObject& obj1, const FgObject& obj2)
{
	Mat mask1 = Mat::zeros(Size(imageWidth, imageHeight), CV_8U);
	Mat mask2 = Mat::zeros(Size(imageWidth, imageHeight), CV_8U);

	vector<vector<Point> > contours;
	contours.push_back(obj1.getContour());
	contours.push_back(obj2.getContour());
	drawContours(mask1, contours, 0, Scalar::all(255), -1);
	drawContours(mask2, contours, 1, Scalar::all(255), -1);

	Mat overlapMask;
	bitwise_and(mask1, mask2, overlapMask);

	vector<vector<Point> > overlapContours = extractContours(overlapMask);

	if(overlapContours.empty()) return 0.0;
	double overlap = contourArea(overlapContours[0]);
	return overlap;
}