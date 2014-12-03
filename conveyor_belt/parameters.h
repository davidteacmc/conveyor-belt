#ifndef _PARAMETERS_
#define _PARAMETERS_

// preprocessors
//#define DEBUG
//#define VIDEO_OUTPUT

// all parameters used by this program are stored here
// access each parameter by the corresponding setter/getter
class Parameters
{
public:
	Parameters();
	~Parameters();

	// setters
    void setVideoSourceType(int type) { videoSourceType = type; }
    void setStartFrameNum(int n) { startFrameNum = n; }
    void setEndFrameNum(int n) { endFrameNum = n; }
    void setInputFilename(char* filename);
    void setOutputCSVFilename(char* filename);
    void setOutputPath(char* path);
	void setShowMetaData(int b) { showMetaData = b; }
	void setShowOutput(int b) { showOutput = b; }
	void setThreshAdjustCoeff(double p) { threshAdjustCoeff = p; }
	void setMinThreshold(int t) { minThreshold = t; }
	void setMaxThreshold(int t) { maxThreshold = t; }
	void setMinArea(double a) { minArea = a; }
	void setMaxArea(double a) { maxArea = a; }
	void setMinAspRatio(double r) { minAspRatio = r; }
	void setMaxAspRatio(double r) { maxAspRatio = r; }
	void setCSSMaxCoeff(double p) { cssMaxCoeff = p; }
	void setSESizeSeg(int s) { seSizeSeg = s; }
	void setThetaHistBP(double t) { thetaHistBP = t;}
	void setWeightOverlapping(double w) { weightOverlapping = w; }

	// getters
    int getVideoSourceType() const { return videoSourceType; }
	int getFrameNum() const { return frameNum; }
	int getFrameWidth() const { return frameWidth; }
	int getFrameHeight() const { return frameHeight; }
	int getStartFrameNum() const { return startFrameNum; }
	int getEndFrameNum() const { return endFrameNum; }
    int getOutWidth() const { return outWidth; }
	int getOutHeight() const { return outHeight; }
    char* getInputFilename() const { return (char*)inputFilename; }
    char* getOutputCSVFilename() const { return (char*)outputCSVFilename; }
    char* getOutputPath() const { return (char*)outputPath; }
	int getShowMetaData() const { return showMetaData; }
	int getShowOutput() const { return showOutput; }
	double getThreshAdjustCoeff() const { return threshAdjustCoeff; }
	int getMinThreshold() const { return minThreshold; }
	int getMaxThreshold() const { return maxThreshold; }
	double getMinArea() const { return minArea; }
	double getMaxArea() const { return maxArea; }
	double getMinAspRatio() const { return minAspRatio; }
	double getMaxAspRatio() const { return maxAspRatio; }
	double getCSSMaxCoeff() const { return cssMaxCoeff; }
	int getSESizeSeg() const { return seSizeSeg; }
	double getThetaHistBP() const { return thetaHistBP; }
	double getWeightOverlapping() const { return weightOverlapping; }

	// called every time a frame comes in
	void incrementFrameNum();

private:
    // video source type
    int videoSourceType;

	// frame number
	int frameNum;

	// frame width
	int frameWidth;

	// frame height
	int frameHeight;

	// start frame number
	int startFrameNum;
	
	// end frame number
	int endFrameNum;
	

	// output frame width
	int outWidth;
    
	// output frame height
	int outHeight;


	// input file name
    char inputFilename [256];

    // output CSV file name
    char outputCSVFilename [256];

	// path of output
	char outputPath [256];

	// showing meta-data on output images
	int showMetaData;

	// showing output real-time
	int showOutput;


	// threshold adjusting coefficient
	double		threshAdjustCoeff;
	
	// lower bound of threshold value
	int			minThreshold;

	// upper bound of threshold value
	int			maxThreshold;

	// lower bound of object area
	double		minArea;

	// upper bound of object area
	double		maxArea;

	// lower bound of object aspect ratio
	double		minAspRatio;

	// upper bound of object aspect ratio
	double		maxAspRatio;

	// coefficient for CSS maxima
	double		cssMaxCoeff;

	// size of structuring element in segmentation
	int			seSizeSeg;

	// threshold for histogram backprojection
	double		thetaHistBP;

	// weight of choosing the best frame of an object between overlapping rate (accuracy) and spatial distance from center
	double		weightOverlapping;
};

#endif