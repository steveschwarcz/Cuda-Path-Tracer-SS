#define GL_GLEXT_PROTOTYPES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <time.h>
#include <cmath>
#include <gl/glew.h>
#include <gl/GL.h>
#include <gl/freeglut.h>
#include "cuda_gl_interop.h"
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include "CudaUtils.h"
#include "Primitives.h"
#include "CudaTracer.h"
#pragma comment(lib, "glew32.lib")

//TODO: These ought to be done differently

//Initialize curand states
__global__ void curandSetupKernel(curandState *state)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	/* Each thread gets same seed, a different sequence number, no offset */ 
	curand_init((unsigned int)clock64(), offset, 0, &state[offset]);
}

__global__ void clearPixels(uchar4 *pixels, uint3 *totalPixelColors) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x; 
	
	uchar4 newPixel;
	newPixel.x = 0;
	newPixel.y = 0;
	newPixel.z = 0;
	newPixel.w = 0;

	uint3 newPixelTotal;
	newPixelTotal.x = 0;
	newPixelTotal.y = 0;
	newPixelTotal.z = 0;

	pixels[offset] = newPixel;
	totalPixelColors[offset] = newPixelTotal;
}

__global__ void computeEyeRaysKernel(Camera camera, Ray* rays, curandState* states) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	Ray ray = computeEyeRay(x, y, DIM, DIM, camera, states[offset]);

	ray.pixelOffset = offset;

	rays[offset] = ray;
}

__global__ void writeToPixelsKernel(uchar4 *pixels, uint3 *totalPixelColors, Ray* rays, int ticks) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	Ray ray = rays[offset];

	int pixelOffset = ray.pixelOffset;

	vec3 radiance = glm::clamp(ray.radiance0, 0.0f, 1.0f);

	//Gamma correction
	radiance.x = pow(radiance.x, GAMMA_CORRECTION);
	radiance.y = pow(radiance.y, GAMMA_CORRECTION);
	radiance.z = pow(radiance.z, GAMMA_CORRECTION);

	uchar3 newPixel;
	uint3 totalPixels = totalPixelColors[pixelOffset];		//the average of the current pixel, multiplied by number of samples (ticks)

	//update pixel
	newPixel.x = static_cast<unsigned int>(glm::clamp<float>(255 * radiance.x + 0.5f, 0.f, 255.f));
	newPixel.y = static_cast<unsigned int>(glm::clamp<float>(255 * radiance.y + 0.5f, 0.f, 255.f));
	newPixel.z = static_cast<unsigned int>(glm::clamp<float>(255 * radiance.z + 0.5f, 0.f, 255.f));

	//now average the pixels
	uchar4 currentPixel;

	totalPixels.x += newPixel.x;
	totalPixels.y += newPixel.y;
	totalPixels.z += newPixel.z;

	float inverseTicks = 1.f / (ticks + 1);
	currentPixel.x = static_cast<unsigned char>(totalPixels.x * inverseTicks + 0.5f);
	currentPixel.y = static_cast<unsigned char>(totalPixels.y * inverseTicks + 0.5f);
	currentPixel.z = static_cast<unsigned char>(totalPixels.z * inverseTicks + 0.5f);
	currentPixel.w = 255;

	totalPixelColors[pixelOffset] = totalPixels;
	pixels[pixelOffset] = currentPixel;

	return;
}

__global__ void pathTraceKernel(uchar4 *pixels, RendererData data, int iterations)
{
	int offset = threadIdx.x + blockIdx.x * blockDim.x;

	//get curand state
	curandState localState = data.curandStates[offset];

	Ray ray(true);

	//use the existing ray
	ray = data.rays[offset];

	//before continuing, be sure that the ray is active
	if (!ray.active) {
		return;
	}

	float distance = INFINITY;
	SurfaceElement surfel;
	char intersection = 0;

	//TODO: Refactor into reusable way to find an intersection
	//loop through spheres, finding intersection
	for (size_t i = 0; i < data.numSpheres; i++)
	{
		Sphere sphere = data.spheres[i];

		if (sphere.intersectRay(ray, distance, surfel))
		{
			intersection = 1;
		}
	}

	for (size_t i = 0; i < data.numTriangles; i++)
	{
		Triangle triangle = data.triangles[i];

		if (triangle.intersectRay(ray, distance, surfel))
		{
			intersection = 1;
		}
	}


	if (intersection)
	{
		//intersection occured

		//find cos of incidence

		float cosI = dot(-ray.direction, surfel.normal);



		//--------------------------
		//		Direct Light
		//--------------------------
			//get material
		Material material = data.materials[surfel.materialIdx];

		vec3 directRadiance(0, 0, 0);

		//emit if material is emitter
		directRadiance += material.emmitance;

		//calculate direct light, iff ray is not inside of primitive
		bool inside = cosI < 0.0f;
		if (!inside) {
			directRadiance += shade(data, surfel, material, localState);
		}


		//--------------------------
		//		Scattering
		//--------------------------
		vec3 indirectRadiance = computeIndirectRadianceAndScatter(ray, surfel, material, cosI, distance, inside, localState);

		//save radiance
		ray.radiance0 += ray.radiance1 * directRadiance;
		ray.radiance1 *= indirectRadiance;
	}
	else
	{
		//ray completely missed
		ray.radiance0 += data.defaultColor * ray.radiance1;
		ray.active = false;
	}


	//update curand state
	data.curandStates[offset] = localState;

	//save the ray
	data.rays[offset] = ray;
}

__device__
vec3 computeIndirectRadianceAndScatter(Ray& ray, const SurfaceElement& surfel, const Material& material, float cosI, const float distance, const bool inside, curandState& localState)
{
	float r = curand_uniform(&localState);

	//--------------------------
	//		Diffuse
	//--------------------------
	if (material.diffAvg > 0.0f)
	{
		r -= material.diffAvg;

		if (r < 0.0f)
		{
			ray.origin = surfel.point + RAY_BUMP_EPSILON * surfel.normal;
			ray.direction = randomDirectionLambert(surfel.normal, localState);

			return material.diffuseColor;
		}

	}
	//--------------------------
	//		Reflected
	//--------------------------


	//both indexes of refraction, n1 / n2, and sin T squared
	float n1, n2, n, sinT2;

	//compute values that will be needed later
	computeSinT2AndRefractiveIndexes(material.indexOfRefraction, cosI, sinT2, n1, n2, n);

	//compute fresnel
	const float fresnelReflective = computeFresnelForReflectance(cosI, sinT2, n1, n2, n);

	if (material.specAvg > 0.0f)
	{
		//glossy reflection
		if (material.flags & MAT_FLAG_PURE_REFLECTION) {
			//pure reflectance: do not compute fresnel
			r -= material.specAvg;
		}
		else {
			//include fresnel
			r -= material.specAvg * fresnelReflective;
		}

		if (r < 0.0f)
		{
			//check for cook torrance
			if (material.flags & MAT_FLAG_COOK_TORRANCE)
			{
				//get a new importance-sampled normal according to the Beckmann ditribution
				vec3 beckmannNormal = randomDirectionBeckmann(surfel.normal, material.roughness, localState);

				//store the old incident direction
				vec3 incident = ray.direction;

				//reflect the ray according to the new normal
				reflRay(ray, surfel.point, beckmannNormal);


				//calculate the geometric term of Cook-Torrance, to account for shadowing and masking
				vec3 half = normalize(ray.direction - incident);

				float nh = abs(dot(surfel.normal, half));
				float nl = abs(dot(surfel.normal, ray.direction));
				float vh = abs(dot(incident, half));
				float nv = abs(cosI);

				float geometric = glm::min<float>(glm::min<float>(1, 2 * nh * nl / vh), 2 * nh * nv / vh);

				//note that the nl term in the denominator cancels out
				return material.specularColor * geometric / nv;
			}

			reflRay(ray, surfel, cosI);

			//TODO: Reference paper
			//glossy scattering
			if (material.specularExponent != INFINITY) {
				//use an importance sampled ray to determine which way the ray ought to travel
				ray.direction = randomDirectionPhong(ray.direction, material.specularExponent, localState);
			}
			//mirror reflectance otherwise.  Nothing to do

			return material.specularColor;
		}
	}
	//--------------------------
	//		Refracted
	//--------------------------
	if (material.refrAvg > 0.0f)
	{
		const float fresnelRefractive = 1.0f - fresnelReflective;

		//refraction
		r -= material.refrAvg * fresnelRefractive;

		if (r < 0.0f)
		{
			refrRay(ray, surfel, cosI, sinT2, n);

			//Apply Beer's law
			if (inside)
			{
				return vec3(
					expf(-distance * material.absorption.x),
					expf(-distance * material.absorption.y),
					expf(-distance * material.absorption.z));
			}
			//not inside, no need for absorption 
			return vec3(1, 1, 1);
		}
	}

	//ray was absorbed
	ray.active = false;
	return vec3(0, 0, 0);
}


__device__ 
Ray computeEyeRay(int x, int y, int dimX, int dimY, const Camera& camera, curandState& state)
{
	const float aspectRatio = float(dimY) / dimX;

	float jitteredX = x + curand_uniform(&state);
	float jitteredY = y + curand_uniform(&state);

	// Compute the side of a square at z = -1 (the far clipping plane) based on the 
	// horizontal left-edge-to-right-edge field of view

	//multiplying by negative 2 offsets the -.5 in the next step
	const float s = -2 * tan(camera.fieldOfView * 0.5f);

	// xPos / image.width() : map from 0 - 1 where the pixel is on the image
	const vec3 start = vec3(((jitteredX / dimX) - 0.5f) * s,
		1 * ((jitteredY / dimY) - 0.5f) * s * aspectRatio,
		1.0f)
		* camera.zNear;

	return Ray(camera.position, glm::normalize(camera.rotation * start));
}

__device__
vec3 shade(const RendererData& data, const SurfaceElement& surfel, const Material& material, curandState& state)
{
	vec3 w_i;
	float distance2;
	vec3 radiance = vec3(0, 0, 0);


	//loop through all point lights
	for (size_t i = 0; i < data.numPointLights; i++)
	{
		PointLight light = data.pointLights[i];

		if (lineOfSight(data, surfel.normal, surfel.point, light.position, w_i, distance2))
		{
			const vec3 L_i = light.power / float(4 * M_PI * distance2);

			float cosI = fmax(0.0f, dot(surfel.normal, w_i));

			radiance.x += cosI * L_i.r * material.diffuseColor.r * material.diffAvg * INVERSE_PI;
			radiance.y += cosI * L_i.g * material.diffuseColor.g * material.diffAvg * INVERSE_PI;
			radiance.z += cosI * L_i.b * material.diffuseColor.b * material.diffAvg * INVERSE_PI;
		}
	}

	//loop through all area lights
	for (size_t i = 0; i < data.numAreaLights; i++)
	{
		AreaLight light = data.areaLights[i];

		vec3 point = getAreaLightPoint(light, data.triangles, state);

		if (lineOfSight(data, surfel.normal, surfel.point, point, w_i, distance2))
		{
			const vec3 L_i = light.power / float(4 * M_PI * distance2);

			float cosI = fmax(0.0f, dot(surfel.normal, w_i));

			radiance.x += cosI * L_i.r * material.diffuseColor.r * material.diffAvg * INVERSE_PI;
			radiance.y += cosI * L_i.g * material.diffuseColor.g * material.diffAvg * INVERSE_PI;
			radiance.z += cosI * L_i.b * material.diffuseColor.b * material.diffAvg * INVERSE_PI;
		}
	}

	return radiance;
}

__device__
vec3 getAreaLightPoint(const AreaLight& light, Triangle* triangles, curandState& state) {
	//get a random point on a triangle
//	float u1 = curand_uniform(&state);
//	float u2 = curand_uniform(&state) * u1;
//	float weight0 = u1 - u2, weight1 = u2, weight2 = 1 - u1;

	float u1 = curand_uniform(&state);
	float u2 = curand_uniform(&state);
	float u3 = curand_uniform(&state);
	float inverseTotal = 1 / (u1 + u2 + u3);
	float weight0 = u1 * inverseTotal, weight1 = u2 * inverseTotal, weight2 = u3 * inverseTotal;
	
	//FIXME: This only works because the light is known to be rectangular - This needs to be expanded for more complex lights
	//get a random point on the light's triangles
	if (curand_uniform(&state) > .5f) {
		return triangles[light.triangleIdx].vertex0 * weight0 +
			triangles[light.triangleIdx].vertex1 * weight1 +
			triangles[light.triangleIdx].vertex2* weight2;
	}
	else {
		return triangles[light.triangleIdx + 1].vertex0 * weight0 +
			triangles[light.triangleIdx + 1].vertex1 * weight1 +
			triangles[light.triangleIdx + 1].vertex2* weight2;
	}
}

__device__
bool lineOfSight(const RendererData& data, const vec3& normal, const vec3& point0, const vec3& point1, vec3& w_i, float& distance2)
{
	const vec3 offset = point1 - point0;
	distance2 = dot(offset, offset);
	float distance = sqrt(distance2);

	w_i = offset / distance;

	const Ray losRay(point0 + (RAY_BUMP_EPSILON * normal), w_i);

	//shorten distance.
	distance -= 2 * RAY_BUMP_EPSILON;

	//loop through all primitives, seeing if any intersections occur
	SurfaceElement surfel;

	//TODO: More robust implementation
	for (size_t i = 0; i < data.numSpheres; i++)
	{
		Sphere sphere = data.spheres[i];

		if (sphere.intersectRay(losRay, distance, surfel))
		{
			return false;
		}
	}

	for (size_t i = 0; i < data.numTriangles; i++)
	{
		Triangle triangle = data.triangles[i];

		if (triangle.intersectRay(losRay, distance, surfel))
		{
			return false;
		}
	}

	return true;
}

__device__
float computeFresnelForReflectance(const float cosI, const float sinT2, const float n1, const float n2, const float n)
{
	//check for TIR
	if (sinT2 > 1.0f)
	{
		return 1.0;
	}

	const float cosT = sqrt(1.0f - sinT2);

	const float r_s = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
	const float r_p = (n2 * cosI - n1 * cosT) / (n2 * cosI + n1 * cosT);

	return (r_s * r_s + r_p * r_p) * 0.5f;
}

__device__
void computeSinT2AndRefractiveIndexes(const float refrIndex, float& cosI, float& sinT2, float& n1, float& n2, float& n)
{
	if (cosI > 0)
	{
		n2 = refrIndex;
		n1 = 1.0f;
	}
	else
	{
		//make sure cos I positive
		cosI = -cosI;

		n1 = refrIndex;
		n2 = 1.0f;
	}

	n = n1 / n2;

	sinT2 = n * n * (1.0f - cosI * cosI);
}

__device__
void reflRay(Ray& ray, const SurfaceElement& surfel, const float cosI)
{
	vec3 w_o = ray.direction - 2 * (-cosI) * surfel.normal;

	ray.origin = surfel.point + (surfel.normal * RAY_BUMP_EPSILON);
	ray.direction = w_o;
}

__device__
void reflRay(Ray& ray, const vec3& point, const vec3& normal)
{
	float cosI = abs(dot(ray.direction, normal));

	vec3 w_o = ray.direction - 2 * (-cosI) * normal;

	ray.origin = point + (normal * RAY_BUMP_EPSILON);
	ray.direction = w_o;
}

__device__
void refrRay(Ray& ray, const SurfaceElement& surfel, const float cosI, const float sinT2, const float n)
{
	//check for TIR
	if (sinT2 > 1.0f)
	{
		ray.active = false;
	}

	const float cosT = sqrt(1.0f - sinT2);

	vec3 w_o = normalize(n * ray.direction + (n * cosI - cosT) * surfel.normal);

	ray.origin = surfel.point + (w_o * RAY_BUMP_EPSILON);
	ray.direction = w_o;
}

__device__
vec3 randomDirectionLambert(vec3 const& normal, curandState& state)
{
	float theta = curand_uniform(&state) * 2 * M_PI;
	float s = curand_uniform(&state);
	float y = sqrt(s);
	float r = sqrt(1 - y * y);

	vec3 sample(r * cos(theta), y, r * sin(theta));
	quat rot = rotateVectorToVector(vec3(0, 1, 0), normal);

	return rot * sample;
}

__device__
vec3 randomDirectionPhong(const vec3& w_o, float exponent, curandState& state)
{
	float theta = curand_uniform(&state) * 2 * M_PI;
	float s = curand_uniform(&state);
	float y = pow(s, 1 / (exponent + 1));
	float r = sqrt(1 - y * y);

	vec3 sample(r * cos(theta), y, r * sin(theta));
	quat rot = rotateVectorToVector(vec3(0, 1, 0), w_o);

	return rot * sample;
}

vec3 randomDirectionBeckmann(vec3 const& normal, float roughness, curandState& state)
{
	float theta = atan(-roughness * roughness * log(1.0f - curand_uniform(&state)));
	float phi = curand_uniform(&state) * 2 * M_PI;

	float cosPhi = cosf(phi);
	float sinPhi = sinf(phi);
	float cosTheta = cosf(theta);
	float sinTheta = sinf(theta);
	//convert to cartesian coordinants from spherical
	vec3 m = vec3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);

	quat rot = rotateVectorToVector(vec3(0, 1, 0), normal);

	return rot * m;
}

__device__
const quat rotateVectorToVector(const vec3& source, const vec3& target)
{
	vec3 axis = cross(source, target);
	quat rotation = quat(1.0f + dot(source, target), axis.x, axis.y, axis.z);
	return normalize(rotation);
}

void generateFrame(uchar4 *pixels, void* dataBlock, int ticks)
{
	ProgramData *data = (ProgramData *)dataBlock;

	//TODO: Implement a way to reset ticks when moving

	dim3 rayThreads(256);
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	//if necessary, clear the pixels and reset the number of ticks
	if (data->resetTicksThisFrame) {
		data->lastResetTick = ticks;

		clearPixels << <grids, threads >> >(pixels, data->totalPixelColors);

		data->resetTicksThisFrame = false;
	}

	//create the rays
	computeEyeRaysKernel << <grids, threads >> > (data->camera, data->renderData.rays, data->renderData.curandStates);

	int numRays = DIM * DIM;
	
	//fire n rays per pixel
	//number of iterations is only one if set to use ray tracing
	unsigned int numIterations = data->usePathTracer ? data->maxIterations : 1;
	for (unsigned int i = 0; i < numIterations && numRays != 0; i++) {
		dim3 rayGrids(numRays / 256);
		pathTraceKernel << < rayGrids, rayThreads >> >(pixels, data->renderData, i);

		if (i != numIterations - 1)
		{
			//Stream compaction
			thrust::device_ptr<Ray> dev_ray_ptr(data->renderData.rays);
			thrust::device_ptr<Ray> partitionRay = thrust::partition(dev_ray_ptr, dev_ray_ptr + numRays, ray_is_active());

			numRays = partitionRay - dev_ray_ptr;			
		}
	}

	//output number of rays cast thus far
	std::cout << "Rays per pixel: " << ticks - data->lastResetTick << "     \r";
	std::cout.flush();

	//write results to buffer
	writeToPixelsKernel << <grids, threads >> >(pixels, data->totalPixelColors, data->renderData.rays, ticks - data->lastResetTick);
}

int main(int argc, char *argv[])
{
	Scene scene;

	vec3 defaultColor(0, 0, 0);

	scene.build();

	PointLight *pointLights;
	AreaLight *areaLights;
	Ray* rays;
	Sphere *spheres;
	Triangle *triangles;
	Material *materials;
	curandState* curandStates;
	uint3* totalPixelColors;

	//initialize bitmap and data
	ProgramData *data = new ProgramData();
	GPUAnimBitmap bitmap(DIM, DIM, data);

	//allocate GPU pointers
	CUDA_ERROR_HANDLE(cudaMalloc((void**)&pointLights,
		sizeof(PointLight)* scene.pointLightsVec.size()));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&areaLights,
		sizeof(AreaLight)* scene.areaLightsVec.size()));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&spheres,
		sizeof(Sphere)* scene.spheresVec.size()));
	
	CUDA_ERROR_HANDLE(cudaMalloc((void**)&materials,
		sizeof(Material)* scene.materialsVec.size()));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&triangles,
		sizeof(Triangle)* scene.trianglesVec.size()));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&rays,
		sizeof(Ray)* DIM * DIM));

	CUDA_ERROR_HANDLE(cudaMalloc(&curandStates, 
		sizeof(curandState)* DIM *DIM));

	CUDA_ERROR_HANDLE(cudaMalloc((void**)&totalPixelColors,
		sizeof(uint3)* DIM * DIM));

	//copy data to GPU
	CUDA_ERROR_HANDLE(cudaMemcpy(spheres, scene.spheresVec.data(), sizeof(Sphere)* scene.spheresVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(triangles, scene.trianglesVec.data(), sizeof(Triangle)* scene.trianglesVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(pointLights, scene.pointLightsVec.data(), sizeof(PointLight)* scene.pointLightsVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(areaLights, scene.areaLightsVec.data(), sizeof(AreaLight)* scene.areaLightsVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(materials, scene.materialsVec.data(), sizeof(Material)* scene.materialsVec.size(), cudaMemcpyHostToDevice));

	//put values in a data block
	data->camera = Camera();
	data->renderData.pointLights = pointLights;
	data->renderData.numPointLights = scene.pointLightsVec.size();
	data->renderData.areaLights = areaLights;
	data->renderData.numAreaLights = scene.areaLightsVec.size();
	data->renderData.spheres = spheres;
	data->renderData.numSpheres = scene.spheresVec.size();
	data->renderData.triangles = triangles;
	data->renderData.numTriangles = scene.trianglesVec.size();
	data->renderData.rays = rays;
	data->renderData.materials = materials;
	data->renderData.defaultColor = defaultColor;
	data->renderData.curandStates = curandStates;
	data->totalPixelColors = totalPixelColors;

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	curandSetupKernel << < grids, threads >> > (curandStates);



	bitmap.anim_and_exit((void(*)(uchar4*, void*, int))generateFrame, NULL, (void(*)(unsigned char, int, int))Key);

	//free
	CUDA_ERROR_HANDLE(cudaFree(pointLights));
	CUDA_ERROR_HANDLE(cudaFree(areaLights));
	CUDA_ERROR_HANDLE(cudaFree(spheres));
	CUDA_ERROR_HANDLE(cudaFree(triangles));
	CUDA_ERROR_HANDLE(cudaFree(rays));
	CUDA_ERROR_HANDLE(cudaFree(materials));
	CUDA_ERROR_HANDLE(cudaFree(curandStates));
	CUDA_ERROR_HANDLE(cudaFree(totalPixelColors));
	
	delete data;

	return 0;
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
		case 32:
		{
				   //space bar pressed: switch between path tracer and rudimentary ray tracer
				   ((ProgramData*)bitmap->dataBlock)->usePathTracer = !((ProgramData*)bitmap->dataBlock)->usePathTracer;
				   ((ProgramData*)bitmap->dataBlock)->resetTicksThisFrame = true;
		}
		case 48:
		{
				   //0 key pressed.  save image to file, with filename taken from current time

				   //get time
				   time_t     now = time(0);
				   struct tm  tstruct;
				   char       buf[160];
				   tstruct = *localtime(&now);

				   //get filename from time
				   strftime(buf, sizeof(buf), "renders/render%Y-%m-%d-%H%M%S.tga", &tstruct);
				   saveScreenshot(buf, DIM, DIM);
		}
	}

	if (moveCamera(((ProgramData*)bitmap->dataBlock)->camera, key))
	{
		((ProgramData*)bitmap->dataBlock)->resetTicksThisFrame = true;
	}
}


/// <summary>
/// Saves a screenshot.  Taken from http://www.flashbang.se/archives/155
/// </summary>
/// <param name="filename">The filename.</param>
/// <param name="x">The x dimensions.</param>
/// <param name="y">The y dimensions.</param>
void saveScreenshot(char filename[160], int x, int y)
{
	// get the image data
	long imageSize = x * y * 3;
	unsigned char *data = new unsigned char[imageSize];
	glReadPixels(0, 0, x, y, GL_BGR, GL_UNSIGNED_BYTE, data);// split x and y sizes into bytes
	int xa = x % 256;
	int xb = (x - xa) / 256; int ya = y % 256;
	int yb = (y - ya) / 256;//assemble the header
	unsigned char header[18] = { 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, (char)xa, (char)xb, (char)ya, (char)yb, 24, 0 };
	// write header and data to file
	std::fstream File(filename, std::ios::out | std::ios::binary);
	File.write(reinterpret_cast<char *>(header), sizeof (char)* 18);
	File.write(reinterpret_cast<char *>(data), sizeof (char)*imageSize);
	File.close();

	delete[] data;
	data = NULL;
}


/// <summary>
/// Moves the camera.
/// </summary>
/// <param name="camera">The camera.</param>
/// <param name="key">The key that was pressed to move the camera.</param>
/// <returns>True if camera moved, false otherwise</returns>
bool moveCamera(Camera& camera, unsigned char key)
{
	float moveDist = .2f, rotateDist = 10.0f * M_PI / 180.0f;

	switch (key) {
	case 119:
		//forward (w)
		camera.position += camera.rotation * vec3(0, 0, -moveDist);
		return true;
	case 97:
		//left (a)
		camera.position += camera.rotation * vec3(-moveDist, 0, 0);
		return true;
	case 115:
		//backwards (s)
		camera.position += camera.rotation * vec3(0, 0, moveDist);
		return true;
	case 100:
		//right (d)
		camera.position += camera.rotation * vec3(moveDist, 0, 0);
		return true;
	case 113:
		//up (q)
		camera.position += camera.rotation * vec3(0, moveDist, 0);
		return true;
	case 101:
		//down (e)
		camera.position += camera.rotation * vec3(0, -moveDist, 0);
		return true;
	case 102:
		//rotate left (f)
		camera.rotation = glm::normalize(camera.rotation * quat(vec3(0, rotateDist, 0)));
		return true;
	case 104:
		//rotate right (h)
		camera.rotation = glm::normalize(camera.rotation * quat(vec3(0, -rotateDist, 0)));
		return true;
	case 103:
		//rotate down (g)
		camera.rotation = glm::normalize(camera.rotation * quat(vec3(-rotateDist, 0, 0)));
		return true;
	case 116:
		//rotate up (t)
		camera.rotation = glm::normalize(camera.rotation * quat(vec3(rotateDist, 0, 0)));
		return true;
	}

	return false;
}
