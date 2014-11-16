#define GL_GLEXT_PROTOTYPES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <gl/freeglut.h>
#include "cuda_gl_interop.h"
#include "CudaUtils.h"
#include <math.h>
#include "Primitives.h"
#include "CudaTracer.h"
#pragma comment(lib, "glew32.lib")

//TODO: These ought to be done differently

__global__ void kernel(uchar4 *ptr, Sphere* spheres, int numSpheres, Camera *camera, PointLight* pointLight, Material* materials, Ray* rays)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	//create the ray
	Ray tempRay = computeEyeRay(x, y, DIM, DIM, camera);

	//loop through spheres, finding intersection
	float distance = INFINITY;
	SurfaceElement surfel;
	char intersection = 0;
	char3 radiance;

	for (int i = 0; i < numSpheres; i++)
	{
		if (spheres[i].intersectRay(tempRay, distance, surfel))
		{
			intersection = 1;
		}
	}

	if (intersection)
	{	
		//intersection found, calculate direct light
		radiance = shade(surfel, pointLight->position, pointLight->power, tempRay.direction, materials);
	}
	else
	{
		radiance.x = 0;
		radiance.y = 0;
		radiance.z = 0;
		tempRay.alive = 0;
	}

	//save the ray
	rays[offset] = tempRay;

	//access uchar4
	ptr[offset].x = radiance.x;
	ptr[offset].y = radiance.y;
	ptr[offset].z = radiance.z;
	ptr[offset].w = 255;
}

__device__ 
Ray computeEyeRay(int x, int y, int dimX, int dimY, Camera* camera)
{
	const float aspectRatio = float(dimY) / dimX;

	// Compute the side of a square at z = -1 (the far clipping plane) based on the 
	// horizontal left-edge-to-right-edge field of view

	//multiplying by negative 2 offsets the -.5 in the next step
	const float s = -2 * tan(camera->fieldOfView * 0.5f);

	// xPos / image.width() : map from 0 - 1 where the pixel is on the image
	const vec3 start = vec3(((float(x) / dimX) - 0.5f) * s,
		1 * ((float(y) / dimY) - 0.5f) * s * aspectRatio,
		1.0f)
		* camera->zNear;

	return Ray(camera->position, glm::normalize(camera->rotation * start));
}

__device__
char3 shade(const SurfaceElement& surfel, const vec3& lightPoint, const vec3& lightPower, const vec3& w_o, Material* materials)
{
	vec3 w_i;
	float distance2;
	char3 radiance;
	const vec3 offset = lightPoint - surfel.point;

	distance2 = dot(offset, offset);

	w_i = offset / sqrt(distance2);

//	if (lineOfSight(surfel.shader().position, lightPoint, w_i, distance2))
//	{
		const vec3& L_i = lightPower / float(4 * M_PI * distance2);

//		Radiance3 bsdfResult(0, 0, 0);

		//either shade as glossy or lambertian
//		if (allowGlossy && surfel.material()->glossyExponent() != INFINITY)
//		{
//			bsdfResult = surfel.material()->evaluateGlossyBSDF(w_i, w_o, surfel.shader().normal, fresnel);
//		}
//		else
//		{
//			bsdfResult = surfel.material()->evaluateLambertianBSDF(w_i, w_o);
//		}

		float cosI = fmax(0.0f, dot(surfel.normal, w_i));
		Material mat = materials[surfel.materialIdx];

		radiance.x = 255 * (cosI * L_i.x * mat.diffuseColor.r / M_PI);
		radiance.y = 255 * (cosI * L_i.y * mat.diffuseColor.g / M_PI);
		radiance.z = 255 * (cosI * L_i.z * mat.diffuseColor.b / M_PI);

		//scatter the light
		return radiance;


//	}
}

bool lineOfSight(const vec3& point0, const vec3& point1, vec3& w_i, float& distance2)
{
	
}

void generateFrame(uchar4 *pixels, void* dataBlock, int ticks)
{
	RendererData *data = (RendererData *)dataBlock;
	

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel <<< grids, threads >>>(pixels, data->spheres, data->numSpheres, data->camera, data->light, data->materials, data->rays);
}

int main(int argc, char *argv[])
{
	int numSpheres = 20;
	int numMaterials = 3;

	Camera *camera;
	PointLight *light;
	Ray* rays;
	Sphere *spheres;
	Triangle *triangles;
	Material *materials;

	//initialize bitmap and data
	RendererData *data = (RendererData*)malloc(sizeof(RendererData));
	GPUAnimBitmap bitmap(DIM, DIM, data);

	//allocate GPU pointers
	CUDA_ERROR_HANDLE(cudaMalloc((void**)&camera,
		sizeof(Camera)));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&light,
		sizeof(PointLight)));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&spheres,
		sizeof(Sphere)* numSpheres));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&rays,
		sizeof(Ray)* DIM * DIM));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&materials,
		sizeof(Material) * numMaterials));
	
	//initialize values
	Sphere *temp_s;
	Camera temp_c = Camera();
	PointLight temp_l = PointLight(vec3(0, 5, 0), vec3(400, 400, 400));
	Material *temp_m;

	temp_m = (Material*)malloc(sizeof(Material)* numMaterials);

	temp_m[0] = Material(vec3(0, 1.0f, 1.0f));
	temp_m[1] = Material(vec3(1.0f, 1.0f, 1.0f));
	temp_m[2] = Material(vec3(1.0f, 0.0f, 0.0f));

	temp_s = (Sphere*)malloc(sizeof(Sphere) * numSpheres);
	for (int i = 0; i < numSpheres; i++)
	{
		temp_s[i].position = vec3(rnd(6.0f) - 3.0f, rnd(6.0f) - 3.0f, rnd(5.0f) - 8.0f);
		temp_s[i].radius = rnd(0.5f) + 0.5f;
		temp_s[i].materialIdx = i % numMaterials;
	}


	CUDA_ERROR_HANDLE(cudaMemcpy(spheres, temp_s, sizeof(Sphere)* numSpheres, cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(camera, &temp_c, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(light, &temp_l, sizeof(PointLight), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(materials, temp_m, sizeof(Material)* numMaterials, cudaMemcpyHostToDevice));

	//put values in a data block
	data->camera = camera;
	data->light = light;
	data->spheres = spheres;
	data->rays = rays;
	data->numSpheres = numSpheres;
	data->materials = materials;

	bitmap.anim_and_exit((void(*)(uchar4*, void*, int))generateFrame, NULL, (void(*)(unsigned char, int, int))Key);

	//free
	CUDA_ERROR_HANDLE(cudaFree(camera));
	CUDA_ERROR_HANDLE(cudaFree(light));
	CUDA_ERROR_HANDLE(cudaFree(spheres));
	CUDA_ERROR_HANDLE(cudaFree(rays));
	CUDA_ERROR_HANDLE(cudaFree(materials));
	free(temp_s);
	free(data);

	return 0;
}

__global__ void moveCamera(Camera *camera, unsigned char key)
{
	switch (key) {
		case 119:
		{
					//forward (w)
					camera->position.z -= .1f;
					break;
		}
		case 97:
		{
				   //left (a)
				   camera->position.x -= .1f;
				   break;
		}
		case 115:
		{
					//backwards (s)
					camera->position.z += .1f;
					break;
		}
		case 100:
		{
					//right (d)
					camera->position.x += .1f;
					break;
		}
		case 32:
		{
				   //up (space)
				   camera->position.y += .1f;
				   break;
		}
		case 99:
		{
				   //down (c)
				   camera->position.y -= .1f;
				   break;
		}

	}
}


// static method used for glut callbacks
void Key(unsigned char key, int x, int y) {

	GPUAnimBitmap*   bitmap = *(GPUAnimBitmap::get_bitmap_ptr());

	switch (key) {
		case 27:
		{
			   if (bitmap->animExit)
				   bitmap->animExit(bitmap->dataBlock);
			   bitmap->free_resources();
			   exit(0);
		}
	}

	moveCamera <<< 1, 1 >>>(((RendererData*)bitmap->dataBlock)->camera, key);
}
