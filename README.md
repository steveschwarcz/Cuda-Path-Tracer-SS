# Path Tracer SS

<a href="http://www.youtube.com/watch?feature=player_embedded&v=0Z7Eo0BJaAg-7Z0
" target="_blank"><img src="http://img.youtube.com/vi/0Z7Eo0BJaAg/hqdefault.jpg" 
alt="Cuda Path Tracer" width="240" height="180" border="10" /></a>

[Youtube link](http://www.youtube.com/watch?feature=player_embedded&v=0Z7Eo0BJaAg-7Z0)

![alt text](https://raw.githubusercontent.com/steveschwarcz/Cuda-Path-Tracer-SS/master/CudaTracer/renders/render2.png "")

![alt text](https://raw.githubusercontent.com/steveschwarcz/Cuda-Path-Tracer-SS/master/CudaTracer/renders/render4.png "")

![alt text](https://raw.githubusercontent.com/steveschwarcz/Cuda-Path-Tracer-SS/master/CudaTracer/renders/render7.png "")

![alt text](https://raw.githubusercontent.com/steveschwarcz/Cuda-Path-Tracer-SS/master/CudaTracer/renders/render9.png "")

This is a simple Path Tracer written in CUDA C/C++

It currently features:

* Iterative Path tracing with next-event estimation
* Stream compation for rays
* Importance sampled Lambert, Phong, and Cook-Torrance shaders
* Antialiasing through jittering
* Area and point lights
* Reflection, refraction with absorbance

Vector math was handled through the glm library.  The project also made use of the GPUAnimBitmap class from the book "CUDA By Example", by Jason Sanders and Edward Kandrot.

There's still several more features slated, including obj suport, use of a spatial data structure (BVH or k/d tree), color/normal mapping, and possibly bidirectional Path Tracing, along with a better control scheme.

All of the above images were taken running with a GTX 760, with an average of 30-45 rays per second.  The images were taken at roughly 2000 rays per pixel each.

## Usage

Controls are as follows:

* Space: switch between path tracing and ray tracing mode (ray tracing mode is very minimal)
* W/A/S/D : Move camera Forward/Left/Backwards/Right
* T/F/G/H : Rotate camera Up/Left/Down/Right
* Q/E: Move camera up/down
* 0: Save image (saved in renders folder)
