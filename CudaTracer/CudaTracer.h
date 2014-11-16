
#define rnd(x) (x * rand() / RAND_MAX)

#define inverse255 0.00392156862f

void generateFrame(uchar4 *pixels, void*, int ticks);
void Key(unsigned char key, int x, int y);
__device__
Ray computeEyeRay(int x, int y, int dimX, int dimY, Camera* camera);

__device__
char3 shade(Sphere* spheres, const SurfaceElement& surfel, const vec3& lightPoint, const vec3& lightPower, const vec3& w_o, Material* material);

__device__
bool lineOfSight(Sphere* spheres, const vec3& point0, const vec3& point1, vec3& w_i, float& distance2);

struct RendererData
{
	Camera *camera;
	PointLight *light;
	Ray* rays;
	Sphere *spheres;
	int numSpheres;
	Material* materials;
};