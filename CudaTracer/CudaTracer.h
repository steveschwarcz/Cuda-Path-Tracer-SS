#include "Scene.h"

#define INVERSE_255 0.00392156862f
#define INVERSE_PI 0.31830988618f

#define RAY_BUMP_EPSILON 1e-4f
#define GAMMA_CORRECTION (1/2.2f)


/// <summary>
/// A struct containing all the data to be passed to the rendering kernel
/// </summary>
struct RendererData
{
	vec3 defaultColor;
	PointLight *pointLights;
	size_t numPointLights;
	AreaLight *areaLights;
	size_t numAreaLights;
	Ray* rays;
	Sphere *spheres;
	size_t numSpheres;
	Triangle *triangles;
	size_t numTriangles;
	Material* materials;
	curandState* curandStates;
};

/// <summary>
/// A struct containing all the data to be used by the entire program.
/// </summary>
struct ProgramData
{
	RendererData renderData;			//all necessary data for main render loop
	Camera camera;						//The camera
	uint3 *totalPixelColors;			//A running total of the color in all pixels.  Used to average the results of ray tracing  
	int lastResetTick;					//The last tick since the tracing was "reset"
	unsigned int maxIterations = 5;		//The maximum number of bounces a ray can perform
	bool resetTicksThisFrame;			//If true, the tracing must be reset this frame
	bool usePathTracer = true;			//If true, then the path tracing algorithm is running.  If false, simple ray tracing is used instead
};

void generateFrame(uchar4 *pixels, void*, int ticks);
void Key(unsigned char key, int x, int y);
bool moveCamera(Camera& camera, unsigned char key);
void saveScreenshot(char filename[160], int x, int y);

__device__
Ray computeEyeRay(int x, int y, int dimX, int dimY, const Camera& camera, curandState& state);

__device__
vec3 shade(const RendererData& data, const SurfaceElement& surfel, const Material& material, curandState& state);

__device__
vec3 getAreaLightPoint(const AreaLight& light, Triangle* triangles, curandState& state);

__device__
bool lineOfSight(const RendererData& data, const vec3& normal, const vec3& point0, const vec3& point1, vec3& w_i, float& distance2);

__device__
vec3 computeIndirectRadianceAndScatter(Ray& ray, const SurfaceElement& surfel, const Material& material, float cosI, const float distance, const bool inside, curandState& localState);

__device__
float computeFresnelForReflectance(const float cosI, const float sinT2, const float n1, const float n2, const float n);

__device__
void computeSinT2AndRefractiveIndexes(const float refrIndex, float& cosI, float& sinT2, float& n1, float& n2, float& n);

__device__
void reflRay(Ray& ray, const vec3& point, const vec3& normal);

__device__
void reflRay(Ray& ray, const SurfaceElement& surfel, const float cosI);

__device__
void refrRay(Ray& ray, const SurfaceElement& surfel, const float cosI, const float sinT2, const float n);

__device__
vec3 randomDirectionLambert(vec3 const& normal, curandState& state);

__device__
vec3 randomDirectionPhong(const vec3& w_o, float exponent, curandState& state);

__device__
vec3 randomDirectionBeckmann(vec3 const& normal, float roughness, curandState& state);

__device__
const quat rotateVectorToVector(const vec3& source, const vec3& target);

struct ray_is_active
{
	__host__ __device__
	bool operator()(const Ray &ray)
	{
		return ray.active;
	}
};