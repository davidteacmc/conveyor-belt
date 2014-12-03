conveyor-belt
=============

Aggregated segmentation of fish from conveyor belt videos

### Updates

May/5/2013
* Parameters are able to be tuned at runtime by passing the desired value through corresponding setters of class Parameters with defaults if not given (see Section 4 for a list of setters/getters).
* Additional command-line argument as the toggle of showing output masks while running.
* Include frame number and object number in the output mask image and CSV file
* Object numbering are now always increment by 1 at a time.

Apr/30/2013
* Fixed the issue of same fish being segmented multiple times by setting a set of rules for track initiation and termination during multiple target tracking.
* Fixed the issue that CSV file is not produced.
* Fixed the issue of processing GoPro video. The calibration images (chessboard) should be placed in the same directory as the binary executable.
* Showing real-time the input frames and output mask images. In the input frame, each detected object are labeled by its oriented bounding box.
* Additional command-line argument as the toggle of object meta-data displayed in output masks.
* All object meta-data (length, aspect ratio, average color, etc.) are included in CSV file.
* Changes in oriented bounding box method for better accuracy. A custom-developed function orientedBoundingBox(), which performs principal component analysis (PCA) on the object contour, substitutes for OpenCV method minAreaRect().

### 1. Command-line interface with arguments
When calling the executable via command-line interface, arguments are required to be provided by the user along with the command. There are 6 arguments in total, which are listed below.

* video source (0=GoPro, 1=Other)
* input video file name
* start frame number
* end frame number (-1 means the last frame)
* output .csv file name for object parameters
* output mask directory
* whether the meta-data are displayed on output mask images (0=off, 1=on)
* whether the output mask images are displayed while running (0=off, 1=on

Here is an example of calling the application along with arguments:
> `conveyor_belt.exe 1 C:\data\my_video.mp4 400 -1 C:\output\out_data.csv C:\output\masks\ 0`

If any of the arguments are not provided properly, the executable stops and an usage hint pops out:  
Usage:  
> `conveyor_belt.exe [video source] [input file name] [start frame #] [end frame #] [output .csv file name] [output mask directory] [showing meta-data] [showing output]`  
`video source: 0=GoPro, 1=Other`    
`end frame #: -1 means last frame`  
`showing meta-data: 0=off, 1=on`  
`showing output: 0=off, 1=on`  

### 2. Target Segmentation

`int extractForeground(InputArray src, OutputArray dst, vector<FgObject>& fgObjs)`  
Parameters: src – input, the current frame  
dst – output, binary image of foreground mask  
fgObjs – extracted foreground objects

Return: number of extracted objects in this frame  

### 3. Target Tracking
`void trackTargets(const vector<FgObject>& fgObjs, bool isFirstFrame)`  
Parameters: fgObjs – extracted foreground objects from target segmentation  
isFirstFrame – indicating if the current frame is the first frame being processed  

`bool makeTargetOutputs(InputArray src, OutputArray dst)`  
Parameters: src – input, the current frame  
dst – output, an image of target segmentation generated from the frame that has the best segmentation within this target’s lifespan. The pixel values of foreground area is preserved, while the remaining region are all colored as gray. Statistical parameters of the target are put aside the foreground region.  

Return: whether an output image is generated, i.e., a target exits the field of view

`void drawTargets(Mat img)`  
Parameters: img – input, the current frame to draw bounding boxes for all tracked targets.

