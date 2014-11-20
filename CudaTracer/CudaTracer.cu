#define GL_GLEXT_PROTOTYPES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <gl/glew.h>
#include <gl/GL.h>
#include <gl/freeglut.h>
#include "cuda_gl_interop.h"
#include <curand_kernel.h>
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

__global__ void pathTraceKernel(uchar4 *pixels, RendererData data, int ticks, int iterations)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	//get curand state
	curandState localState = data.curandStates[offset];

	Ray ray(true);

	if (iterations == 0)
	{
		//create the ray if this a new render
		ray = computeEyeRay(x, y, DIM, DIM, data.camera, localState);

		ray.pixelOffset = offset;
	}
	else
	{
		//use the existing ray
		ray = data.rays[offset];
	}

	//before continuing, be sure that the ray is still active
	if (!ray.active || iterations == 9)
	{
		vec3 radiance = ray.directLightColor + ray.indirectLightColor;

		uchar3 newPixel;
		int3 totalPixels;		//the average of the current pixel, multiplied by number of samples (ticks)

		//update pixel
		newPixel.x = static_cast<unsigned int>(glm::clamp<float>(255 * radiance.x, 0.f, 255.f));
		newPixel.y = static_cast<unsigned int>(glm::clamp<float>(255 * radiance.y, 0.f, 255.f));
		newPixel.z = static_cast<unsigned int>(glm::clamp<float>(255 * radiance.z, 0.f, 255.f));

		//now average the pixels
		uchar4 currentPixel = pixels[ray.pixelOffset];

		totalPixels.x = currentPixel.x * ticks;
		totalPixels.y = currentPixel.y * ticks;
		totalPixels.z = currentPixel.z * ticks;

		float inverseTicks = 1.f / (ticks + 1);
		currentPixel.x = static_cast<unsigned char>((totalPixels.x + newPixel.x) * inverseTicks);
		currentPixel.y = static_cast<unsigned char>((totalPixels.y + newPixel.y) * inverseTicks);
		currentPixel.z = static_cast<unsigned char>((totalPixels.z + newPixel.z) * inverseTicks);
		currentPixel.w = 255;


		pixels[ray.pixelOffset] = currentPixel;

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

		bool inside = false;

		//find cos of incidence
		float cosI = dot(-ray.direction, surfel.normal);

		//if cosI is positive, ray is inside of primitive 
		if (cosI < 0.0f)
		{
			inside = true;
		}



		//--------------------------
		//		Direct Light
		//--------------------------
			//get material
		Material material = data.materials[surfel.materialIdx];

		if (iterations == 0) {
			//emit if material is emitter
			ray.directLightColor += material.emmitance;

			//calculate direct light
			short3 directRadiance;
			if (!inside) {
				ray.directLightColor += shade(data, surfel, material);
			}
		}


		//--------------------------
		//		Scattering
		//--------------------------

		//both indexes of refraction, n1 / n2, and sin T squared
		float n1, n2, n, sinT2;

		//compute values that will be needed later
		computeSinT2AndRefractiveIndexes(material.indexOfRefraction, cosI, sinT2, n1, n2, n);

		//compute fresnel
		const float fresnelReflective = computeFresnelForReflectance(cosI, sinT2, n1, n2, n);

		float r = curand_uniform(&localState);

		//first check for diffuse indirect light
		if (material.diffAvg > 0.0f)
		{
			//diffuse reflection
			r -= material.diffAvg;

			if (r < 0.0f)
			{
				if (iterations == 0) {
					ray.indirectLightColor = vec3(1, 1, 1);
				}

				vec3 L(0, 0, 0);
				vec3 w_o = cosHemiRandom(surfel.normal, localState);

				ray.origin = surfel.point + RAY_BUMP_EPSILON * w_o;
				ray.direction = w_o;

				ray.indirectLightColor.x *= .1f * material.diffuseColor.x;
				ray.indirectLightColor.y *= .1f * material.diffuseColor.y;
				ray.indirectLightColor.z *= .1f * material.diffuseColor.z;
			}
			else {
				ray.active = false;
			}
		}
	}
	else
	{
		if (iterations == 0) {
			ray.directLightColor = data.defaultColor;
		}
		else {
			ray.indirectLightColor *= data.defaultColor;
		}

		ray.active = false;
	}


	//update curand state
	data.curandStates[offset] = localState;

	//save the ray
	data.rays[offset] = ray;
}

__device__ 
Ray computeEyeRay(int x, int y, int dimX, int dimY, Camera* camera, curandState& state)
{
	const float aspectRatio = float(dimY) / dimX;

	float jitteredX = x + curand_uniform(&state);
	float jitteredY = y + curand_uniform(&state);

	// Compute the side of a square at z = -1 (the far clipping plane) based on the 
	// horizontal left-edge-to-right-edge field of view

	//multiplying by negative 2 offsets the -.5 in the next step
	const float s = -2 * tan(camera->fieldOfView * 0.5f);

	// xPos / image.width() : map from 0 - 1 where the pixel is on the image
	const vec3 start = vec3(((jitteredX / dimX) - 0.5f) * s,
		1 * ((jitteredY / dimY) - 0.5f) * s * aspectRatio,
		1.0f)
		* camera->zNear;

	return Ray(camera->position, glm::normalize(camera->rotation * start));
}

__device__
vec3 shade(const RendererData& data, const SurfaceElement& surfel, const Material& material)
{
	vec3 w_i;
	float distance2;
	vec3 radiance = vec3(0, 0, 0);


	//TODO: loop through all lights
	for (size_t i = 0; i < data.numPointLights; i++)
	{
		PointLight light = data.pointLights[i];

		if (lineOfSight(data, surfel.point, light.position, w_i, distance2))
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

Ray reflRay(const vec3& incident, const SurfaceElement& surfel, const float cosI)
{
	vec3 w_o = incident - 2 * (-cosI) * surfel.normal;

	return Ray(surfel.point + (w_o * RAY_BUMP_EPSILON), w_o);
}

Ray refrRay(const vec3& incident, const SurfaceElement& surfel, const float cosI, const float sinT2, const float n)
{
	//check for TIR
	if (sinT2 > 1.0f)
	{
		return Ray(false);
	}

	const float cosT = sqrt(1.0f - sinT2);

	vec3 w_o = normalize(n * incident + (n * cosI - cosT) * surfel.normal);

	return Ray(surfel.point + (w_o * RAY_BUMP_EPSILON), w_o);
}

const vec3 cosHemiRandom(vec3 const& normal, curandState& state)
{
	float theta = curand_uniform(&state) * 2 * M_PI;
	float s = curand_uniform(&state);
	float y = sqrt(s);
	float r = sqrt(1 - y * y);

	vec3 sample(r * cos(theta), y, r * sin(theta));
	quat rot = rotateVectorToVector(vec3(0, 1, 0), normal);

	return rot * sample;
}

const vec3 cosHemiRandomPhong(const vec3& w_o, float exponent, curandState& state)
{
	float theta = curand_uniform(&state) * 2 * M_PI;
	float s = curand_uniform(&state);
	float y = pow(s, 1 / (exponent + 1));
	float r = sqrt(1 - y * y);

	vec3 sample(r * cos(theta), y, r * sin(theta));
	quat rot = rotateVectorToVector(vec3(0, 1, 0), w_o);

	return rot * sample;
}

const quat rotateVectorToVector(const vec3& source, const vec3& target)
{
	vec3 axis = cross(source, target);
	quat rotation = quat(1.0f + dot(source, target), axis.x, axis.y, axis.z);
	return normalize(rotation);
}

void generateFrame(uchar4 *pixels, void* dataBlock, int ticks)
{
	RendererData *data = (RendererData *)dataBlock;

	//TODO: Implement a way to reset ticks when moving

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	for (unsigned int i = 0; i < 10; i++) {
		pathTraceKernel <<< grids, threads >>>(pixels, *data, ticks, i);
	}
}

int main(int argc, char *argv[])
{
	Scene scene;

	vec3 defaultColor(100.f / 255.f, 100.f / 255.f, 100.f / 255.f);

	buildScene(scene);

	Camera *camera;
	PointLight *pointLights;
	AreaLight *areaLights;
	Ray* rays;
	Sphere *spheres;
	Triangle *triangles;
	Material *materials;
	curandState* curandStates;

	//initialize bitmap and data
	RendererData *data = new RendererData();
	GPUAnimBitmap bitmap(DIM, DIM, data);

	//allocate GPU pointers
	CUDA_ERROR_HANDLE(cudaMalloc((void**)&camera,
		sizeof(Camera)));

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
	
	//initialize values
	Camera temp_c = Camera();

	//copy data to GPU
	CUDA_ERROR_HANDLE(cudaMemcpy(camera, &temp_c, sizeof(Camera), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(spheres, scene.spheresVec.data(), sizeof(Sphere)* scene.spheresVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(triangles, scene.trianglesVec.data(), sizeof(Triangle)* scene.trianglesVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(pointLights, scene.pointLightsVec.data(), sizeof(PointLight)* scene.pointLightsVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(areaLights, scene.areaLightsVec.data(), sizeof(AreaLight)* scene.areaLightsVec.size(), cudaMemcpyHostToDevice));
	CUDA_ERROR_HANDLE(cudaMemcpy(materials, scene.materialsVec.data(), sizeof(Material)* scene.materialsVec.size(), cudaMemcpyHostToDevice));

	//put values in a data block
	data->camera = camera;
	data->pointLights = pointLights;
	data->numPointLights = scene.pointLightsVec.size();
	data->areaLights = areaLights;
	data->numAreaLights = scene.areaLightsVec.size();
	data->spheres = spheres;
	data->numSpheres = scene.spheresVec.size();
	data->triangles = triangles;
	data->numTriangles = scene.trianglesVec.size();
	data->rays = rays;
	data->materials = materials;
	data->defaultColor = defaultColor;
	data->curandStates = curandStates;

	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	curandSetupKernel << < grids, threads >> > (curandStates);



	bitmap.anim_and_exit((void(*)(uchar4*, void*, int))generateFrame, NULL, (void(*)(unsigned char, int, int))Key);

	//free
	CUDA_ERROR_HANDLE(cudaFree(camera));
	CUDA_ERROR_HANDLE(cudaFree(pointLights));
	CUDA_ERROR_HANDLE(cudaFree(areaLights));
	CUDA_ERROR_HANDLE(cudaFree(spheres));
	CUDA_ERROR_HANDLE(cudaFree(triangles));
	CUDA_ERROR_HANDLE(cudaFree(rays));
	CUDA_ERROR_HANDLE(cudaFree(materials));
	CUDA_ERROR_HANDLE(cudaFree(curandStates));
	
	delete data;

	return 0;
}

void buildScene(Scene& scene) {
	float power = 300;

	//add point light
	scene.pointLightsVec.push_back(PointLight(vec3(0, 0.0f, -2.5), vec3(power, power, power)));
//	scene.pointLightsVec.push_back(PointLight(vec3(2, 1.0f, 0), vec3(power, power, power)));

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
	scene.materialsVec.push_back(Material(vec3(0.0f, 0.0f, 1.0f), 0.9f));
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
		rotate((glm::mediump_float)-90, vec3(0, 1, 0)) *
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
