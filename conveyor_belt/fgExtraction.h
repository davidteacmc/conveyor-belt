#ifndef _FG_EXTRACTION_H_
#define _FG_EXTRACTION_H_

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "fgObject.h"
#include "fgUpdate.h"

using namespace std;
using namespace cv;

//************************** Segmentation **************************************
class Segmentation
{
public:
	virtual void segmentObjects(InputArray src, OutputArray dst) = 0;
};

//*********************** ThresholdSegmentation ********************************
class ThresholdSegmentation : public Segmentation
{
public:
	ThresholdSegmentation() {}
	~ThresholdSegmentation() {}

	void segmentObjects(InputArray src, OutputArray dst);

private:
	int getOtsuThreshold(InputArray inImg, int lowerVal, int upperVal, int* u1Ptr, InputArray roiMask = Mat());
	int getThreshold(InputArray src);
};

//*********************** WatershedSegmentation ********************************
class WatershedSegmentation : public Segmentation
{
public:
	WatershedSegmentation() { _threshSegment = new ThresholdSegmentation(); }
	~WatershedSegmentation() { delete _threshSegment; }

	void segmentObjects(InputArray src, OutputArray dst);

private:
	int generateMarkers(InputArray src, OutputArray dst);
	int generateMarkers2(InputArray src, OutputArray dst);
	vector<Point> hMax(InputArray src, int h, InputArray mask);

	ThresholdSegmentation* _threshSegment;
};

//*********************** CannyEdgeSegmentation ********************************
class CannyEdgeSegmentation : public Segmentation
{
public:
	CannyEdgeSegmentation() { }
	~CannyEdgeSegmentation() { }

	void segmentObjects(InputArray src, OutputArray dst);
	void segmentObjects(InputArray src, OutputArray dst, bool threeChannels);
};

//************************ GradientSegmentation ********************************
class GradientSegmentation : public Segmentation
{
public:
	GradientSegmentation() { }
	~GradientSegmentation() { }

	void segmentObjects(InputArray src, OutputArray dst);
};

//***************************** FgExtraction ***********************************
class FgExtraction 
{
public:
	FgExtraction();
	~FgExtraction();
	
	void appearanceFiltering(InputArray src, OutputArray dst, vector<vector<Point> > contours, Mat inFrame);
	void localBinaryPatterns(InputArray src, OutputArray dst, int radius, Mat mask);
	
	void cssImage(vector<Point> contour, OutputArray cssImg);
	double cssMatchingCost(const vector<pair<double, int>>& cssMax, int modelNum);
	
	int extractForeground(InputArray src, OutputArray dst, vector<FgObject>& fgObjs);
	
private:
	Segmentation*	_segmentation;
	FgUpdate*		_fgUpdate;
};

#endif
