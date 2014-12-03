#include <iostream>
#include <sstream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "parameters.h"
#include "cameraCalibrator.h"
#include "fgObject.h"
#include "fgExtraction.h"
#include "tracker.h"
#include "util.h"

using namespace std;
using namespace cv;

// global variables
Parameters param;
ofstream csvOut;

int mainBeltSegment(int argc, char** argv)
{
    char    filename [256];
	Mat     rawFrame;
	Mat		rawFrameCropped;
    Mat     inFrame;
	Mat		lastFrame;
    Mat     edgeMask;
	Mat		fgMask;
	Mat		lastTargetImg (param.getOutHeight(), param.getOutWidth(), CV_8UC3, Scalar::all(128));
	
    // command line arguments:
    //     video source (0=GoPro, 1=Other)
    //     input video file name
    //     start frame number
    //     end frame number (-1 would mean last frame)
    //     output .csv file name
    //     output mask directory
	//     show meta-data on output mask images (0=off, 1=on)
	//     show output real-time (0=off, 1=on)
    // example:
    //     conveyor_belt.exe 1 video\BeltVideo.MTS 440 700 results\output\bbox.csv results\output 0 0
	//     conveyor_belt.exe 0 Haul92-short.mp4 0 -1 results\bbox.csv results\output 0 0

    if(argc != 9){
        cout << "\nUsage:\nconveyor_belt.exe [video source] [input file name] [start frame #] [end frame #] [output .csv file name] [output mask directory] [showing meta-data] [showing output]" << endl << endl;
        cout << "video source: 0=GoPro, 1=Other" << endl
             << "end frame #: -1 means last frame" << endl
			 << "showing meta-data: 0=off, 1=on" << endl
			 << "showing output: 0=off, 1=on" << endl;
        return 0;
    }
    else{
        stringstream ss;
        ss << argv[1] << ' ' << argv[3] << ' ' << argv[4] << ' ' << argv[7] << ' ' << argv[8];
        int type = 0, startNum = 0, endNum = 0, showMetaData = 0, showOutput = 0;
        ss >> type >> startNum >> endNum >> showMetaData >> showOutput;
        
        param.setVideoSourceType(type);
        param.setInputFilename(argv[2]);
        param.setStartFrameNum(startNum);
        param.setEndFrameNum(endNum);
        param.setOutputCSVFilename(argv[5]);
        param.setOutputPath(argv[6]);
		param.setShowMetaData(showMetaData);
		param.setShowOutput(showOutput);
    }

	// parameters for run-time tuning
	/*double threshAdjustCoeff = 0.2;
	int minThreshold = 80;
	int maxThreshold = 100;
	double minArea = 32000;
	double minAspRatio = 1.8;
	double maxAspRatio = 7.0;
	int cssMaxCoeff = 0.2;
	int seSize = 5;
	param.setThreshAdjustCoeff(threshAdjustCoeff);
	param.setMinThreshold(minThreshold);
	param.setMaxThreshold(maxThreshold);
	param.setMinArea(minArea);
	param.setMinAspRatio(minAspRatio);
	param.setMaxAspRatio(maxAspRatio);
	param.setCSSMaxCoeff(cssMaxCoeff);
	param.setSESize(seSize);*/
	if(param.getVideoSourceType() == 0){
		param.setMinArea(24000);
		param.setMinThreshold(90);
		param.setMaxThreshold(95);
	}


    // camera undistortion
	CameraCalibrator camCalibrator;
    if(param.getVideoSourceType() == 0){
	    if(!camCalibrator.calibrate())
			return 0;
    }
	
	sprintf_s(filename, "%s", param.getInputFilename());
	    
	VideoCapture videoCapture(filename);

#ifdef VIDEO_OUTPUT
    sprintf_s(filename, "%s\\result.avi", param.getOutputPath());
    VideoWriter videoWriter(filename, -1, 15, Size(param.getOutWidth(), param.getOutHeight()), true);
#endif

	FgExtraction* fgExtract = new FgExtraction;

	Tracker* tracker = new Tracker;

	// open CSV file
	sprintf_s(filename, "%s", param.getOutputCSVFilename());
	csvOut.open(filename, ios::out);
	csvOut << "frame #" << ',' << "object #" << ','
		   << "X1" << ',' << "Y1" << ',' << "X2" << ',' << "Y2" << ','
		   << "X3" << ',' << "Y3" << ',' << "X4" << ',' << "Y4" << ','
		   << "length" << ',' << "width" << ',' << "aspect ratio" << ','
		   << "area" << ',' << "occupancy rate" << ',' << "CSS matching cost" << ','
		   << "color R" << ',' << "color G" << ',' << "color B"
		   << endl;

	// start processing video!
	cout << endl << "Fast-forwarding to start frame (# " << param.getStartFrameNum() << ") ..." << endl << endl;
    while(videoCapture.read(rawFrame))
	{
		param.incrementFrameNum();
		if(param.getFrameNum() < param.getStartFrameNum()) continue;

        if(param.getFrameNum() % 10 == 0) cout << "Frame " << param.getFrameNum() << endl;
		
        if(param.getVideoSourceType() == 0){
		    Mat undistorted = camCalibrator.undistortImage(rawFrame);
		    undistorted.copyTo(rawFrameCropped);
        }
        else{
		    rawFrame.rowRange(0, rawFrame.rows-9).copyTo(rawFrameCropped);
		    //sprintf_s(filename, "%s\\rawFrame\\%d.jpg", param.getOutputPath(), param.getFrameNum());
		    //imwrite(filename, rawFrame);
		    //continue;
        }

		resize(rawFrameCropped, inFrame, Size(param.getFrameWidth(), param.getFrameHeight()));

#ifdef DEBUG
		//showImage("rawFrame", rawFrame, 1, 1);
		//showImage("rawFrameCropped", rawFrameCropped, 1, 1);
		//showImage("inFrame", inFrame, 1, 1 );
#endif

		//**********************************************************************
		// Perform image segmentation
		vector<FgObject> fgObjs;
		int nObj = fgExtract->extractForeground(inFrame, fgMask, fgObjs);	
#ifdef DEBUG
        //showImage("fgMaskFinal", fgMask, 0);
#endif

		//**********************************************************************
		// extract the final contours and track them
        tracker->trackTargets(fgObjs);
		
		Mat targetImg;
		bool hasOutput = tracker->makeTargetMaskOutputs(inFrame, targetImg);
		if(hasOutput) lastTargetImg = targetImg;

		inFrame.copyTo(lastFrame);

		//**********************************************************************
        // output the result

		tracker->drawTargets(inFrame);
		
        Mat resLeft;
		resize(inFrame, resLeft, Size(param.getOutWidth()/2, param.getOutHeight()), 0, 0, INTER_AREA);
		
		Mat resRight;
		//resize(fgMask, resRight, Size(param.getOutWidth()/2, param.getOutHeight()), 0, 0, INTER_AREA);
		resize(lastTargetImg, resRight, Size(param.getOutWidth()/2, param.getOutHeight()), 0, 0, INTER_AREA);

        Mat outFrame(param.getOutHeight(), param.getOutWidth(), resLeft.type());
		resLeft.copyTo(outFrame(Range(0, param.getOutHeight()), Range(0, param.getOutWidth()/2)));
		resRight.copyTo(outFrame(Range(0, param.getOutHeight()), Range(param.getOutWidth()/2, param.getOutWidth())));

#ifdef VIDEO_OUTPUT
        if(videoWriter.isOpened()) videoWriter << outFrame;
#else
		if(param.getShowOutput() == 1)
			showImage("output", outFrame, 1, 1);
		//sprintf_s(filename, "%s\\%05d.jpg", param.getOutputPath(), param.getFrameNum());
		//imwrite(filename, outFrame);
#endif
		
		if(param.getEndFrameNum() != -1 && param.getFrameNum() >= param.getEndFrameNum()) break;
    }

	delete fgExtract;
	delete tracker;
    return 0;
}

int main(int argc, char** argv)
{
    int ok = mainBeltSegment(argc, argv);
    return ok;
}