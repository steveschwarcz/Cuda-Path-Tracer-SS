# Path Tracer SS

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Hrlhny0T6kw-7Z0
" target="_blank"><img src="http://img.youtube.com/vi/Hrlhny0T6kw/hqdefault.jpg" 
alt="Cuda Path Tracer" width="240" height="180" border="10" /></a>

[Youtube link](http://www.youtube.com/watch?feature=player_embedded&v=Hrlhny0T6kw-7Z0)

This is a simple Path Tracer written in CUDA C/C++

It currently features:

* Iterative Path tracing with next-event estimation
* Stream compation for rays
* Importance sampled Lambert and Phong shaders
* Antialiasing through jittering
* Area and point lights
* Reflection, refraction with absorbance

Vector math was handled through the glm library.  The project also made use of the GPUAnimBitmap class from the book "CUDA By Example", by Jason Sanders and Edward Kandrot.

There's still several more features slated, including obj suport, use of a spatial data structure (BVH or k/d tree), color/normal mapping, and possibly bidirectional Path Tracing.

## Usage

Controls are as follows:

* Space: switch between path tracing and ray tracing mode (ray tracing mode is very minimal)
* W/A/S/D : Move camera Forward/Left/Backwards/Right
* T/F/G/H : Rotate camera Up/Left/Down/Right
* Q/E: Move camera up/down
* 0: Save image (saved in renders folder)
