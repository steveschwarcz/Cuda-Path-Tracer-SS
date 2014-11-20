#include "Scene.h"

#define rnd(x) (x * rand() / RAND_MAX)

#define inverse255 0.00392156862f

#define RAY_BUMP_EPSILON 1e-4f


/// <summary>
/// A struct containing all the data to be passed to the rendering kernel
/// </summary>
struct RendererData
{
	Camera *camera;
	PointLight *pLights;
	size_t numPLights;
	Ray* rays;
	Sphere *spheres;
	size_t numSpheres;
	Triangle *triangles;
	size_t numTriangles;
	Material* materials;
};

void buildScene(Scene& scene);
void addRandomSpheres(Scene& scene, const size_t numSpheres);
void addCornellBox(Scene& scene, const float wallSize);

void generateFrame(uchar4 *pixels, void*, int ticks);
void Key(unsigned char key, int x, int y);
__device__
Ray computeEyeRay(int x, int y, int dimX, int dimY, Camera* camera);

__device__
char3 shade(const RendererData& data, const SurfaceElement& surfel, const vec3& lightPoint, const vec3& lightPower, const vec3& w_o, Material* material);

__device__
bool lineOfSight(const RendererData& data, const vec3& point0, const vec3& point1, vec3& w_i, float& distance2);