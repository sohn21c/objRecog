# Object Detection with RGB-D camera
#### _[James Sohn (Click to see the portfolio)](http://sohn21c.github.io)_
#### _JAN 2019 ~ Present_

## Objective  
To build a object detection system with RGB-D camera that can show both the object classification and the distance from the sensor. In process of completing the pipeline, a number of detection algorithms are researched and one is selected for implementation. 

## Video demonstration
1. Object detection demo with YOLO (MS-COCO dataset)  
[![YouTube](https://github.com/sohn21c/objRecog/blob/master/img/yolo1.jpg?raw=true)](https://youtu.be/YU-P0f_Sr7g)  
2. Object detection demo with SSD (MS-COCO dataset)  
[![YouTube](https://github.com/sohn21c/objRecog/blob/master/img/ssd1.jpg?raw=true)](https://youtu.be/vZw5j909vss)

## Algorithm  
For the implementation of the pipeline, YOLOv3 was used. In the process of landing the decision to use YOLO, I have researched many other detection CNN architectures like R-CNN, SSD and etc,,. One can have an access to my personal write-up of my algorithm research and should consider it as brief summary. One should refer to original publication for detailed information.  
[Link to the algorithm search write-up](https://github.com/sohn21c/research/blob/master/objDetection.md)  

## Software

#### Realtime object detection with YOLO on MS-COCO dataset
`yolo_depth.py`  

**1. High level overview:**  

This script initiates the Intel Realsense D435 camera, performss object detection with YOLO trained on MS-COCO data, capable of detecting 80 different classifications, and shows the distance of each detection of objects from the sensor up to 10m.  

**2. Dependency:**  

- python 3.6.8  
- pyrealsense2  
- numpy  
- opencv  
- imutils  
See the full list of dependencies [here](https://github.com/sohn21c/objRecog/blob/master/dependencies.txt)  

**3. Input arguments:**  

- -h, --help	
- -y, --yolo 		 	path to yolo directory that contains weight file and class labels as shown in the file structure below  
- -c, --confidence 	min. probability filter, default = 0.5  
- -t, --threshold 	detection threshold, dafault = 0.3  

**4. How to run:**  

Before running the program, one should have downloaded the object class names, weights and configuration files shown in the file structure below.  

`coco.data` and `yolov3.cfg` can be found in the `cfg/` directory. And one can download the `yolov3.weights` files with the command below.  
>$ wget http://pjreddie.com/media/files/yolov3.weights  

Finally, one can run the script with the following command.  

>$ python yolo_depth.py --yolo yolo-coco/  

**5. File structure:**  
yolo  
│   ├── yolo-coco  
│   │   ├── coco.names  
│   │   ├── yolov3.cfg  
│   │   └── yolov3.weights  
│   └── yolo_depth.py  

**6. Add-on functions:**  
_(update on MAR 20th, 2019)_  
`yolo_depth.py` [(ae134a4)](ae134a45ec1a2d4909cb422885e1266081e54574) now as add-on function to generate one's own training data **with depth** while performing detection. One can simply input 'k' on one's keyboard while on the detection window. Doing so will create a folder `/img` in current directory and save color img, depth img and corresponding text files with detection label annotation `darknet` can accept for training.  


## Hardware
[Intel Realsense D435](https://github.com/IntelRealSense/librealsense)  
One should download and install [Realsense SDK2.0](https://github.com/IntelRealSense/librealsense) provided in the link and the [Python Wrapper](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python)  
#### Spec  
1. 2 HD image sensors (Left and Right side cameras)  
	1080p  
	10 bit Raw  
2. IR projector  
	Improves the depth information retrieval  
3. Color sensor  
	1080p  
	10 bit Raw RGB  


## Reference/Credit
Object detection using opencv DNN module is mostly inspired by [Pyimagesearch](https://pyimagesearch.com)

## License  
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

