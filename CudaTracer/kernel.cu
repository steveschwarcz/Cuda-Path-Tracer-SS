#define GL_GLEXT_PROTOTYPES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <gl/freeglut.h>
#include "cuda_gl_interop.h"
#include "utils.h"
#include <math.h>
#include "Renderer.h"
#pragma comment(lib, "glew32.lib")

//TODO: These ought to be done differently

#define numSpheres 20
#define rnd(x) (x * rand() / RAND_MAX)

#define inverse255 0.00392156862f

void generateFrame(uchar4 *pixels, void*, int ticks);

__device__
Ray computeEyeRay(int x, int y, int dimX, int dimY, Camera* camera);

__device__
char3 shade(const SurfaceElement& surfel, const vec3& lightPoint, const vec3& lightPower, const vec3& w_o);

Sphere *temp_s;

__global__ void kernel(uchar4 *ptr, Sphere* spheres, Camera *camera, PointLight* pointLight, Ray* rays)
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
	Sphere *intersectedSphere;
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

//		printf("surfel: (%f, %f, %f)\n", surfel.point.x, surfel.point.y, surfel.point.z);

		
		//intersection found, calculate direct light
		radiance = shade(surfel, pointLight->position, pointLight->power, tempRay.direction);
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


	//	AffineTransformation trans = Eigen::AngleAxisf(M_PI / 4, Vector3::UnitX()).matrix()

//	return Ray(position(), (m_rotation.matrix() * start).normalized());// -position);

	return Ray(vec3(0, 0, 0), glm::normalize(start));
}

__device__
char3 shade(const SurfaceElement& surfel, const vec3& lightPoint, const vec3& lightPower, const vec3& w_o)
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

		radiance.x = 255 * (cosI * L_i.x * 1.0 / M_PI);
		radiance.y = 255 * (cosI * L_i.y * 1.0 / M_PI);
		radiance.z = 255 * (cosI * L_i.z * 1.0 / M_PI);


//		radiance.x = 255;
//		radiance.y = 255;
//		radiance.z = 255;

		//scatter the light
		return radiance;


//	}
}

void generateFrame(uchar4 *pixels, void*, int ticks)
{
	Camera *c;
	PointLight *l;
	Ray* rays;
	Sphere *s;

	//allocate spheres
	CUDA_ERROR_HANDLE(cudaMalloc((void**)&c,
		sizeof(Camera)));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&l,
		sizeof(PointLight)));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&s,
		sizeof(Sphere) * numSpheres));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&rays,
		sizeof(Ray) * DIM * DIM));

	//20 random spheres
	
//	Camera *temp_c = (Camera*)malloc(sizeof(Camera));
//	PointLight *temp_l = (PointLight*)malloc(sizeof(PointLight));

	Camera temp_c = Camera();
	PointLight temp_l = PointLight(vec3(0, 5, 0), vec3(200, 200, 200));


	CUDA_ERROR_HANDLE(cudaMemcpy(s, temp_s, sizeof(Sphere)* numSpheres, cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(c, &temp_c, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(l, &temp_l, sizeof(PointLight), cudaMemcpyHostToDevice));

//	free(temp_s);
//	free(temp_c);
//	free(temp_l);

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel <<<grids, threads >>>(pixels, s, c, l, rays);

	CUDA_ERROR_HANDLE(cudaFree(c));
	CUDA_ERROR_HANDLE(cudaFree(l));
	CUDA_ERROR_HANDLE(cudaFree(s));
	CUDA_ERROR_HANDLE(cudaFree(rays));
}

int main(int argc, char *argv[])
{
	GPUAnimBitmap bitmap(DIM, DIM, NULL);

	temp_s = (Sphere*)malloc(sizeof(Sphere)* numSpheres);
	for (int i = 0; i < numSpheres; i++)
	{
		temp_s[i].position = vec3(rnd(6.0f) - 3.0f, rnd(6.0f) - 3.0f, rnd(5.0f) - 8.0f);
		temp_s[i].radius = rnd(0.5f) + 0.5f;
	}

	bitmap.anim_and_exit((void(*)(uchar4*, void*, int))generateFrame, NULL);

	free(temp_s);

	return 0;
}



