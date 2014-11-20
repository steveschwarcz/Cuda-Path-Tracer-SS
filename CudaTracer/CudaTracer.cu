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

__global__ void kernel(uchar4 *ptr, RendererData data)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	//create the ray
	Ray tempRay = computeEyeRay(x, y, DIM, DIM, data.camera);

	//loop through spheres, finding intersection
	float distance = INFINITY;
	SurfaceElement surfel;
	char intersection = 0;
	char3 radiance;

	for (unsigned int i = 0; i < data.numSpheres; i++)
	{
		if (data.spheres[i].intersectRay(tempRay, distance, surfel))
		{
			intersection = 1;
		}
	}

	for (unsigned int i = 0; i < data.numTriangles; i++)
	{
		if (data.triangles[i].intersectRay(tempRay, distance, surfel))
		{
			intersection = 1;
		}
	}

	if (intersection)
	{	
		//intersection found, calculate direct light
		radiance = shade(data, surfel, data.pLights->position, data.pLights->power, tempRay.direction, data.materials);
	}
	else
	{
		radiance.x = 100;
		radiance.y = 100;
		radiance.z = 100;
		tempRay.alive = 0;
	}

	//save the ray
	data.rays[offset] = tempRay;

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
char3 shade(const RendererData& data, const SurfaceElement& surfel, const vec3& lightPoint, const vec3& lightPower, const vec3& w_o, Material* materials)
{
	vec3 w_i;
	float distance2;
	char3 radiance;

	if (lineOfSight(data, surfel.point, lightPoint, w_i, distance2))
	{
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

		radiance.x = 255 * (cosI * L_i.r * mat.diffuseColor.r * mat.diffAvg / M_PI);
		radiance.y = 255 * (cosI * L_i.g * mat.diffuseColor.g * mat.diffAvg / M_PI);
		radiance.z = 255 * (cosI * L_i.b * mat.diffuseColor.b * mat.diffAvg / M_PI);

		//scatter the light
		return radiance;
	}

	radiance.x = 0;
	radiance.y = 0;
	radiance.z = 0;

	return radiance;
}

__device__
bool lineOfSight(const RendererData& data, const vec3& point0, const vec3& point1, vec3& w_i, float& distance2)
{
	const vec3 offset = point1 - point0;
	distance2 = dot(offset, offset);
	float distance = sqrt(distance2);

	w_i = offset / distance;

	const Ray losRay(point0 + (RAY_BUMP_EPSILON * w_i), w_i);

	//shorten distance.
	distance -= RAY_BUMP_EPSILON;

	//loop through all primitives, seeing if any intersections occur
	SurfaceElement surfel;

	//TODO: More robust implementation
	for (unsigned int i = 0; i < data.numSpheres; i++)
	{
		if (data.spheres[i].intersectRay(losRay, distance, surfel))
		{
			return false;
		}
	}

	for (unsigned int i = 0; i < data.numTriangles; i++)
	{
		if (data.triangles[i].intersectRay(losRay, distance, surfel))
		{
			return false;
		}
	}

	return true;
}

void generateFrame(uchar4 *pixels, void* dataBlock, int ticks)
{
	RendererData *data = (RendererData *)dataBlock;
	

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel <<< grids, threads >>>(pixels, *data);
}

int main(int argc, char *argv[])
{
	Scene scene;

	buildScene(scene);

	Camera *camera;
	PointLight *light;
	Ray* rays;
	Sphere *spheres;
	Triangle *triangles;
	Material *materials;

	//initialize bitmap and data
	RendererData *data = new RendererData();
	GPUAnimBitmap bitmap(DIM, DIM, data);

	//allocate GPU pointers
	CUDA_ERROR_HANDLE(cudaMalloc((void**)&camera,
		sizeof(Camera)));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&light,
		sizeof(PointLight)* scene.pointLightsVec.size()));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&spheres,
		sizeof(Sphere)* scene.spheresVec.size()));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&materials,
		sizeof(Material)* scene.materialsVec.size()));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&triangles,
		sizeof(Triangle)* scene.trianglesVec.size()));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&rays,
		sizeof(Ray)* DIM * DIM));
	
	//initialize values
	Camera temp_c = Camera();

	//copy data to GPU
	CUDA_ERROR_HANDLE(cudaMemcpy(camera, &temp_c, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(spheres, scene.spheresVec.data(), sizeof(Sphere)* scene.spheresVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(triangles, scene.trianglesVec.data(), sizeof(Triangle)* scene.trianglesVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(light, scene.pointLightsVec.data(), sizeof(PointLight)* scene.pointLightsVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(materials, scene.materialsVec.data(), sizeof(Material)* scene.materialsVec.size(), cudaMemcpyHostToDevice));

	//put values in a data block
	data->camera = camera;
	data->pLights = light;
	data->numPLights = scene.pointLightsVec.size();
	data->spheres = spheres;
	data->numSpheres = scene.spheresVec.size();
	data->triangles = triangles;
	data->numTriangles = scene.trianglesVec.size();
	data->rays = rays;
	data->materials = materials;

	bitmap.anim_and_exit((void(*)(uchar4*, void*, int))generateFrame, NULL, (void(*)(unsigned char, int, int))Key);

	//free
	CUDA_ERROR_HANDLE(cudaFree(camera));
	CUDA_ERROR_HANDLE(cudaFree(light));
	CUDA_ERROR_HANDLE(cudaFree(spheres));
	CUDA_ERROR_HANDLE(cudaFree(triangles));
	CUDA_ERROR_HANDLE(cudaFree(rays));
	CUDA_ERROR_HANDLE(cudaFree(materials));
	
	delete data;

	return 0;
}

void buildScene(Scene& scene) {
	float power = 800;

	//add point light
	scene.pointLightsVec.push_back(PointLight(vec3(-2, 4.0f, 0), vec3(power, power, power)));

	//add Spheres
	addRandomSpheres(scene, 20);

	//add cornell box
	addCornellBox(scene, 10);
}

void addRandomSpheres(Scene& scene, const size_t numSpheres)
{
	int matIdx = scene.materialsVec.size();

	//add matrials
	scene.materialsVec.push_back(Material(vec3(0, 1.0f, 1.0f), 0.9f));
	scene.materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.9f));
	scene.materialsVec.push_back(Material(vec3(1.0f, 0.0f, 0.0f), 0.9f));

	for (int i = 0; i < numSpheres; i++)
	{
		Sphere s;

		s.position = vec3(rnd(6.0f) - 3.0f, rnd(6.0f) - 3.0f, rnd(5.0f) - 8.0f);
		s.radius = rnd(0.5f) + 0.5f;
		s.materialIdx = matIdx + (i % 3);

		scene.spheresVec.push_back(s);
	}
}

void addCornellBox(Scene& scene, const float wallSize)
{
	using glm::translate;
	using glm::scale;
	using glm::rotate;
	
	int matIdx = scene.materialsVec.size();

	scene.materialsVec.push_back(Material(vec3(1.0f, 1.0f, 0.8f), 0.9f));	//white			(+0)
	scene.materialsVec.push_back(Material(vec3(1.0f, 0.0f, 0.0f), 0.9f));	//red			(+1)
	scene.materialsVec.push_back(Material(vec3(0.0f, 1.0f, 0.0f), 0.9f));	//green			(+2)
	scene.materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f)));			//white light	(+3)

	const float offset = wallSize / 2;

	const mat4 scaleToWall = scale(vec3(wallSize, wallSize, wallSize));

	//floor
	mat4 trans = translate(vec3(0, -offset, -offset)) *
		rotate(-(glm::mediump_float)90, vec3(1, 0, 0)) *
		scaleToWall;
	scene.addRectangularModel(trans, matIdx);

	//ceiling
	trans = translate(vec3(0, offset, -offset)) *
		rotate((glm::mediump_float)90, vec3(1, 0, 0)) *
		scaleToWall;
	scene.addRectangularModel(trans, matIdx);

	//left wall
	trans = translate(vec3(-offset, 0, -offset)) *
		rotate((glm::mediump_float)90, vec3(0, 1, 0)) *
		scaleToWall;
	scene.addRectangularModel(trans, matIdx + 1);

	//right wall
	trans = translate(vec3(offset, 0, -offset)) *
		rotate((glm::mediump_float)90, vec3(0, 1, 0)) *
		scaleToWall;
	scene.addRectangularModel(trans, matIdx + 2);

	//back wall
	trans = translate(vec3(0, 0, -wallSize)) *
//		rotate((glm::mediump_float)90, vec3(1, 0, 0)) *
		scaleToWall;
	scene.addRectangularModel(trans, matIdx);
}

//TODO: Clean - doesn't need to be a kernel
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
		case 113:
		{
				   //up (q)
				   camera->position.y += .1f;
				   break;
		}
		case 101:
		{
				   //down (e)
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
