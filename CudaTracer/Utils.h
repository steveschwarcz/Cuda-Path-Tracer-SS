
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

#define DIM 512
//#define WIDTH 800
//#define HEIGHT 600


//runs a cuda function, and outputs any errors.  Taken from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define CUDA_ERROR_HANDLE(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

//TODO: Refactor into something more specific to this project
/// <summary>
/// Taken from CUDA By Example, by Jason Sanders and Edward Kandrot
/// </summary>
struct GPUAnimBitmap
{
	//same GPU buffer, different names
	GLuint bufferObj;
	cudaGraphicsResource *resource;


	int width, height;
	void *dataBlock;
	void(*fAnim)(uchar4*, void*, int);
	void(*animExit)(void*);
	void(*clickDrag)(void*, int, int, int, int);
	int dragStartX, dragStartY;

	GPUAnimBitmap(int w, int h, void *d)
	{
		width = w;
		height = h;
		dataBlock = d;
		clickDrag = NULL;

		//pick a device that is compute capable v1.0 or better
		cudaDeviceProp prop;
		int dev;
		memset(&prop, 0, sizeof(cudaDeviceProp));
		prop.major = 1;
		prop.minor = 0;
		CUDA_ERROR_HANDLE(cudaChooseDevice(&dev, &prop));

		//use this device for OpenGL
		CUDA_ERROR_HANDLE(cudaGLSetGLDevice(dev));

		//Init GLUT
		int c = 1;
		char *name = "name";
		glutInit(&c, &name);
		glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
		glutInitWindowSize(DIM, DIM);
		glutInitWindowPosition(300, 200);
		glutCreateWindow("bitmap");

		glewInit();

		//create the shared data buffer
		//create the buffer handle
		glGenBuffers(1, &bufferObj);
		//bind the handle to a buffer
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
		//allocate buffer - dynamic draw means the buffer will be repeatedly modified
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

		//register buffer with CUDA as a graphics resource
		CUDA_ERROR_HANDLE(cudaGraphicsGLRegisterBuffer(&resource,
			bufferObj,
			cudaGraphicsMapFlagsNone));
	}


	~GPUAnimBitmap() {
		free_resources();
	}

	void free_resources(void) {
		CUDA_ERROR_HANDLE(cudaGraphicsUnregisterResource(resource));

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
	}

	long image_size(void) const { return width * height * 4; }

	void click_drag(void(*f)(void*, int, int, int, int)) {
		clickDrag = f;
	}

	void anim_and_exit(void(*f)(uchar4*, void*, int), void(*e)(void*)) {
		GPUAnimBitmap**   bitmap = get_bitmap_ptr();
		*bitmap = this;
		fAnim = f;
		animExit = e;

		glutKeyboardFunc(Key);
		glutDisplayFunc(Draw); 
		glutReshapeFunc(resize);
		if (clickDrag != NULL)
			glutMouseFunc(mouse_func);
		glutIdleFunc(idle_func);
		glutMainLoop();
	}

	// static method used for glut callbacks
	static GPUAnimBitmap** get_bitmap_ptr(void) {
		static GPUAnimBitmap*   gBitmap;
		return &gBitmap;
	}

	// static method used for glut callbacks
	static void mouse_func(int button, int state,
		int mx, int my) {
		if (button == GLUT_LEFT_BUTTON) {
			GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
			if (state == GLUT_DOWN) {
				bitmap->dragStartX = mx;
				bitmap->dragStartY = my;
			}
			else if (state == GLUT_UP) {
				bitmap->clickDrag(bitmap->dataBlock,
					bitmap->dragStartX,
					bitmap->dragStartY,
					mx, my);
			}
		}
	}

	// static method used for glut callbacks
	static void idle_func(void) {
		static int ticks = 1;
		GPUAnimBitmap*  bitmap = *(get_bitmap_ptr());
		uchar4*         devPtr;
		size_t  size;

		CUDA_ERROR_HANDLE(cudaGraphicsMapResources(1, &(bitmap->resource), NULL));
		CUDA_ERROR_HANDLE(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, bitmap->resource));

		bitmap->fAnim(devPtr, bitmap->dataBlock, ticks++);

		CUDA_ERROR_HANDLE(cudaGraphicsUnmapResources(1, &(bitmap->resource), NULL));

		glutPostRedisplay();
	}

	// static method used for glut callbacks
	static void Key(unsigned char key, int x, int y) {
		switch (key) {
		case 27:
			GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
			if (bitmap->animExit)
				bitmap->animExit(bitmap->dataBlock);
			bitmap->free_resources();
			exit(0);
		}
	}

	// static method used for glut callbacks
	static void Draw(void) {
		GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(bitmap->width, bitmap->height, GL_RGBA,
			GL_UNSIGNED_BYTE, 0);

		glutSwapBuffers();
	}

	static void resize(int width, int height) {
		// we ignore the params and do:
		glutReshapeWindow(DIM, DIM);
	}
};
