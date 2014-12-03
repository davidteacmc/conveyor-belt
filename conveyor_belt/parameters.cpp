#include <iostream>
using namespace std;
#include "parameters.h"

Parameters::Parameters() :
	frameNum(0),
	frameWidth(720),
	frameHeight(540),
	outWidth(1280),
	outHeight(480),
	showMetaData(0),
	showOutput(0),
	threshAdjustCoeff(0.2),
	minThreshold(80),
	maxThreshold(100),
	minArea(32000),
	maxArea(frameWidth * frameHeight * 4),
	minAspRatio(1.8), 
	maxAspRatio(7.0),
	cssMaxCoeff(0.2),
	seSizeSeg(5),
	thetaHistBP(0.3),
	weightOverlapping(0.7)
{
}

Parameters::~Parameters()
{
}

void Parameters::setInputFilename(char* filename)
{
    sprintf_s(inputFilename, "%s", filename);
}

void Parameters::setOutputCSVFilename(char* filename)
{
    sprintf_s(outputCSVFilename, "%s", filename);
}

void Parameters::setOutputPath(char* path)
{
    sprintf_s(outputPath, "%s", path);
}

void Parameters::incrementFrameNum()
{
	++frameNum;
}