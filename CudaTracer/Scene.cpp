#include "Scene.h"

#define rnd(x) (x * rand() / RAND_MAX)

Scene::Scene()
{
}


Scene::~Scene()
{
}

/// <summary>
/// Builds the scene by calling other scene building functions
/// </summary>
void Scene::build() {
	float power = 500;

	//add point light
	//scene.pointLightsVec.push_back(PointLight(vec3(0, 0.0f, 2.5), vec3(power, power, power)));
	//scene.pointLightsVec.push_back(PointLight(vec3(2, 9.0f, -5), vec3(power, power, power)));

	//add Spheres
	//addRandomSpheres(15);
	addDefinedSpheres(4);

	//add cornell box
	addCornellBox(8);
}

/// <summary>
/// Adds a rectangular area light.
/// </summary>
/// <param name="transformation">The transformation to use on the model as it is created.</param>
/// <param name="materialIdx">The index of the material to use.</param>
/// <param name="power">The power of the light to create.</param>
void Scene::addAreaLight(mat4 transformation, int materialIdx, vec3 power)
{
	size_t triangleIdx = trianglesVec.size();

	//create model for light
	addRectangularModel(transformation, materialIdx);

	//Get the area of the light.  Since hte light is 2 equally sized triangles, only the area of the first one needs to be calculated, then doubled
	vec3 edge1 = trianglesVec[triangleIdx].vertex1 - trianglesVec[triangleIdx].vertex2;
	vec3 edge2 = trianglesVec[triangleIdx].vertex2 - trianglesVec[triangleIdx].vertex0;

	float area = glm::length(cross(edge1, edge2));

	//add the light to the scene
	areaLightsVec.push_back(AreaLight(power, triangleIdx, 2, area));
}


/// <summary>
/// Initializes a rectangular model composed of 2 triangles.
/// </summary>
/// <param name="transformation">The transformation to use on the model as it is created.</param>
/// <param name="materialIdx">The index of the material to use.</param>
void Scene::addRectangularModel(mat4 transformation, int materialIdx)
{
	vec4 newVertex[4];

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			//transform the new point
			vec4 vert(i - 0.5f, j - 0.5f, 0.0f, 1);
			vert = transformation * vert;

			newVertex[2 * i + j] = vert;

			//TODO: add vertex to a vertexes array
		}
	}

	vec4 newNormal(0.0f, 0.0f, 1.0f, 0);
	//get the inverse-transpose of the transformation matrix, to properly transform the normals
	newNormal = glm::inverse(glm::transpose(transformation)) * newNormal;

	//normalize the new normal
	newNormal = normalize(newNormal);

	vec3 vertexes[4], normal = vec3(newNormal);

	//add normals and vertexes
	for (int i = 0; i < 4; i++)
	{
		vertexes[i] = vec3(newVertex[i]);
	}

	//push triangle into vector
	trianglesVec.push_back(Triangle(vertexes[0], vertexes[1], vertexes[2], normal, normal, normal, materialIdx));
	trianglesVec.push_back(Triangle(vertexes[3], vertexes[1], vertexes[2], normal, normal, normal, materialIdx));
}

void Scene::addDefinedSpheres(const float size) {
	int matIdx = materialsVec.size();

	materialsVec.push_back(Material(vec3(1.0f, 0.0f, 0.0f), 0.35f, vec3(1, 1, 1), 250, 0.6f, 1.75f));
//	materialsVec[matIdx].pureRefl = true;
	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.0f,
		vec3(1, 1, 1), 300, 0.9f, 1.55f,
		vec3(.15, .15, 0), .9f));

	spheresVec.push_back(Sphere(vec3(-2, -(size - 1.5f), -(size * 1.3f)), 1.5f, matIdx + 1));
	spheresVec.push_back(Sphere(vec3(1, -(size - 1.f), -(size * 1.4f)), 1.f, matIdx));
}

/// <summary>
/// Adds a set of random spheres.  Because the randomness is unseeded, the same spheres will appear every time
/// </summary>
/// <param name="numSpheres">The number spheres.</param>
void Scene::addRandomSpheres(const size_t numSpheres)
{
	int matIdx = materialsVec.size();

	//add matrials
	materialsVec.push_back(Material(vec3(0.0f, 0.0f, 1.0f), 0.9f));
	materialsVec.push_back(Material(vec3(1.0f, 0.0f, 0.0f), 0.9f));
	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.0f,
		vec3(1, 1, 1), INFINITY, 0.9f, 1.55f,
		vec3(1, 1, 1), .9f));
	/*scene.materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.0f,
	vec3(1, 1, 1), INFINITY, 0.9f, 1.6f));
	scene.materialsVec[matIdx + 3].pureRefl = true;*/


	for (size_t i = 0; i < numSpheres; i++)
	{
		Sphere s;

		s.position = vec3(rnd(5.0f) - 2.5f, rnd(5.0f) - 2.5f, rnd(5.0f) - 8.0f);
		s.radius = rnd(0.5f) + 0.5f;
		s.materialIdx = matIdx + (i % 3);

		spheresVec.push_back(s);
	}
}

/// <summary>
/// Adds a cornell box of a given size, centered at 0, with no back.
/// </summary>
/// <param name="wallSize">Size of the walls.</param>
void Scene::addCornellBox(const float wallSize)
{
	using glm::translate;
	using glm::scale;
	using glm::rotate;

	int matIdx = materialsVec.size();

	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 0.8f), 0.9f));	//white			(+0)
	materialsVec.push_back(Material(vec3(1.0f, 0.0f, 0.0f), 0.9f));	//red			(+1)
	//materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.1f, vec3(1, 1, 1), INFINITY, .9f, 1.5f));	//green			(+2)
	//materialsVec[matIdx + 2].pureRefl = true;
	materialsVec.push_back(Material(vec3(0.0f, 1.0f, 0.0f), 0.9f));	//green			(+2)


	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f)));			//white light	(+3)

	const float offset = wallSize / 2;

	const mat4 scaleToWall = scale(vec3(wallSize, wallSize, wallSize));

	//floor
	mat4 trans = translate(vec3(0, -offset, -offset)) *
		rotate(-(glm::mediump_float)90, vec3(1, 0, 0)) *
		scaleToWall;
	addRectangularModel(trans, matIdx);

	//ceiling
	trans = translate(vec3(0, offset, -offset)) *
		rotate((glm::mediump_float)90, vec3(1, 0, 0)) *
		scaleToWall;
	addRectangularModel(trans, matIdx);

	//left wall
	trans = translate(vec3(-offset, 0, -offset)) *
		rotate((glm::mediump_float)90, vec3(0, 1, 0)) *
		scaleToWall;
	addRectangularModel(trans, matIdx + 1);

	//right wall
	trans = translate(vec3(offset, 0, -offset)) *
		rotate((glm::mediump_float) - 90, vec3(0, 1, 0)) *
		scaleToWall;
	addRectangularModel(trans, matIdx + 2);

	//back wall
	trans = translate(vec3(0, 0, -wallSize)) *
		//		rotate((glm::mediump_float)90, vec3(1, 0, 0)) *
		scaleToWall;
	addRectangularModel(trans, matIdx);

	//light
	float power = 1200;
	trans = translate(vec3(0, offset - 0.001f, -offset)) *
		rotate((glm::mediump_float) 90, vec3(1, 0, 0)) *
		scale(vec3(2.5f, 2.5f, 2.5f));
	addAreaLight(trans, matIdx + 3, vec3(power, power, power));
}