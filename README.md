conveyor-belt
=============

Aggregated segmentation of fish from conveyor belt videos

May/5/2013
-Parameters are able to be tuned at runtime by passing the desired value through corresponding setters of class Parameters with defaults if not given (see Section 4 for a list of setters/getters).
-Additional command-line argument as the toggle of showing output masks while running.
-Include frame number and object number in the output mask image and CSV file
-Object numbering are now always increment by 1 at a time.
Apr/30/2013
-Fixed the issue of same fish being segmented multiple times by setting a set of rules for track initiation and termination during multiple target tracking.
-Fixed the issue that CSV file is not produced.
-Fixed the issue of processing GoPro video. The calibration images (chessboard) should be placed in the same directory as the binary executable.
-Showing real-time the input frames and output mask images. In the input frame, each detected object are labeled by its oriented bounding box.
-Additional command-line argument as the toggle of object meta-data displayed in output masks.
-All object meta-data (length, aspect ratio, average color, etc.) are included in CSV file.
-Changes in oriented bounding box method for better accuracy. A custom-developed function orientedBoundingBox(), which performs principal component analysis (PCA) on the object contour, substitutes for OpenCV method minAreaRect().
1. Command-line interface with arguments
When calling the executable via command-line interface, arguments are required to be provided by the user along with the command. There are 6 arguments in total, which are listed below.
-video source (0=GoPro, 1=Other)
-input video file name
-start frame number
-end frame number (-1 means the last frame)
-output .csv file name for object parameters
-output mask directory
-whether the meta-data are displayed on output mask images (0=off, 1=on)
-whether the output mask images are displayed while running (0=off, 1=on)
Here is an example of calling the application along with arguments:
> conveyor_belt.exe 1 C:\data\my_video.mp4 400 -1 C:\output\out_data.csv C:\output\masks\ 0
If any of the arguments are not provided properly, the executable stops and an usage hint pops out:
Usage:
conveyor_belt.exe [video source] [input file name] [start frame #] [end frame #] [output .csv file name] [output mask directory] [showing meta-data] [showing output]
video source: 0=GoPro, 1=Other
end frame #: -1 means last frame
showing meta-data: 0=off, 1=on
showing output: 0=off, 1=on
2. Target Segmentation
int extractForeground(InputArray src, OutputArray dst, vector<FgObject>& fgObjs);
Parameters: src – input, the current frame
dst – output, binary image of foreground mask
fgObjs – extracted foreground objects
Return: number of extracted objects in this frame
3. Target Tracking
void trackTargets(const vector<FgObject>& fgObjs, bool isFirstFrame);
Parameters: fgObjs – extracted foreground objects from target segmentation
isFirstFrame – indicating if the current frame is the first frame being processed
bool makeTargetOutputs(InputArray src, OutputArray dst);
Parameters: src – input, the current frame
dst – output, an image of target segmentation generated from the frame that has the best segmentation within this target’s lifespan. The pixel values of foreground area is preserved, while the remaining region are all colored as gray. Statistical parameters of the target are put aside the foreground region.
Return: whether an output image is generated, i.e., a target exits the field of view
void drawTargets(Mat img);
Parameters: img – input, the current frame to draw bounding boxes for all tracked targets.
4. Compound data types
//************************** Parameters **************************************
// all the parameters used by this program are stored here.
// access each parameter by the corresponding setter/getter.
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
double threshAdjustCoeff;
// lower bound of threshold value
int minThreshold;
// upper bound of threshold value
int maxThreshold;
// lower bound of object area
double minArea;
// upper bound of object area
double maxArea;
// lower bound of object aspect ratio
double minAspRatio;
// upper bound of object aspect ratio
double maxAspRatio;
// coefficient for CSS maxima
double cssMaxCoeff;
// size of structuring element in segmentation
int seSizeSeg;
// threshold for histogram backprojection
double thetaHistBP;
// weight of choosing the best frame of an object between overlapping rate (accuracy) and spatial distance from center
double weightOverlapping;
};
//****************************** FgObject **************************************
class FgObject
{
public:
FgObject();
FgObject(const FgObject& obj);
FgObject(vector<Point> contour);
~FgObject() {}
void setContour(const vector<Point>& contour) { _contour = contour; }
void setRectColor(const Scalar& color) { _rectColor = color; }
void setCSSCost0(const double& cost) { _cssCost0 = cost; }
void setCSSCost1(const double& cost) { _cssCost1 = cost; }
void setCSSCost2(const double& cost) { _cssCost2 = cost; }
void setAlphaMap(const Mat& alphaMap) { _alphaMap = alphaMap.clone(); }
void moveContour(const Point& vec);
int getFrameNum() const { return _frameNum; }
vector<Point> getContour() const { return _contour; }
Scalar getRectColor() const { return _rectColor; }
double getCSSCost0() const { return _cssCost0; }
double getCSSCost1() const { return _cssCost1; }
double getCSSCost2() const { return _cssCost2; }
Rect boundingBox() const { return boundingRect(_contour); }
RotatedRect orientedBox() const { return orientedBoundingBox(_contour); }
Point2f center() const { return orientedBoundingBox(_contour).center; }
void update(const FgObject& obj);
void draw(Mat img);
void drawShifted(Mat img);
private:
int _frameNum;
Scalar _rectColor;
vector<Point> _contour;
double _cssCost0;
double _cssCost1;
double _cssCost2;
public:
bool _updated;
};