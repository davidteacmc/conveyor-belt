#ifndef _FG_UPDATE_H_
#define _FG_UPDATE_H_

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "fgObject.h"

using namespace std;
using namespace cv;

//**************************** FgObjectUpdate **********************************
class FgUpdate
{
public:
	FgUpdate();
	FgUpdate(Mat inImg);

	virtual void updateObject(InputArray, OutputArray) = 0;

	Mat			_inImg;
	vector<Mat>	_alphaMaps;
};

//************************* HistBackprojectUpdate ******************************
class HistBackprojectUpdate : public FgUpdate
{
public:
	HistBackprojectUpdate();
	~HistBackprojectUpdate();

	void updateObject(InputArray src, OutputArray dst);
	
private:
	Mat histBackProject(InputArray hist, int binWidth, Mat roiMask);

	Mat		_ratioHistBP;
};

//***************************** MattingUpdate **********************************
class MattingUpdate : public FgUpdate
{
public:
	MattingUpdate();
	~MattingUpdate();

	void updateObject(InputArray src, OutputArray dst);

private:
	void findOptimalAlphaMap(InputArray src, InputArray srcFg, OutputArray dst);
    double pixelAlpha();
};

#endif