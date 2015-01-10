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
	addRandomSpheres(5);
	addRandomGlassSpheres(15);
//	addDefinedSpheres(4);

	//add cornell box
	//addCornellBox(8);
	addMirrorBox(10);
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

	vec3 vertexes[4], normal = normalize(vec3(newNormal));

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

	materialsVec.push_back(Material(vec3(1.0f, 0.0f, 0.0f), 0.35f, vec3(1, 1, 1), 250, 0.6f, 2.5f));
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
void Scene::addRandomGlassSpheres(const size_t numSpheres)
{
	int matIdx = materialsVec.size();

	//add matrials

	////slightly blue glass
	//materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.0f,
	//	vec3(1, 1, 1), INFINITY, 0.7f, 1.55f,
	//	vec3(0.0f, 0.0f, 0), .7f));


	//red glass
	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.0f,
		vec3(1, 1, 1), INFINITY, 0.7f, 1.55f,
		vec3(0, 0.75f, 0.75f), .7f));

	//blue glass
	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.0f,
		vec3(1, 1, 1), INFINITY, 0.7f, 1.55f,
		vec3(0.75f, 0.75f, 0.f), .7f));

	//green glass
	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.0f,
		vec3(1, 1, 1), INFINITY, 0.7f, 1.55f,
		vec3(0.75f, 0.f, 0.75f), .7f));

	////green cook torrance glass
	//materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.0f,
	//	vec3(1, 1, 1), INFINITY, 0.4f, 1.55f,
	//	vec3(0.5f, 0, 0.5f), .7f));
	//materialsVec[matIdx + 2].flags |= MAT_FLAG_COOK_TORRANCE;
	//materialsVec[matIdx + 2].roughness = 0.2f;

	////cloudy glass
	//materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.0f,
	//	vec3(1, 1, 1), INFINITY, 0.7f, 1.55f,
	//	vec3(0.3f, 0.0f, 0.f), .7f));


	for (size_t i = 0; i < numSpheres; i++)
	{
		Sphere s;

		rnd(1); rnd(1); rnd(1);

		s.position = vec3(rnd(5.0f) - 2.5f, rnd(5.0f) - 2.5f, rnd(7.0f) - 9.0f);
		s.radius = rnd(1.0f) + 0.2f;
		s.materialIdx = matIdx + (i % 3);

		spheresVec.push_back(s);
	}
}

/// <summary>
/// Adds a set of random spheres.  Because the randomness is unseeded, the same spheres will appear every time
/// </summary>
/// <param name="numSpheres">The number spheres.</param>
void Scene::addRandomSpheres(const size_t numSpheres)
{
	int matIdx = materialsVec.size();

	//add matrials

//	//slightly blue glass
//	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.0f,
//		vec3(1, 1, 1), INFINITY, 0.7f, 1.55f,
//		vec3(0.25f, 0.25f, 0), .7f));
//
//	//diffuse teal
//	materialsVec.push_back(Material(vec3(0.0f, 1.0f, 1.0f), 0.7f));
//
//	//blue cook torrance
//	materialsVec.push_back(Material(vec3(0.4f, 0.1f, 1.0f), 0.1f, vec3(0.2f, 0.2f, 1.f), INFINITY, 0.6f, 2.7f));
//	materialsVec[matIdx + 2].flags |= MAT_FLAG_COOK_TORRANCE;
//	materialsVec[matIdx + 2].roughness = 0.7f;
//
//	//reflective
//	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f), 0.1f,
//	vec3(1, 1, 1), INFINITY, 0.6f, 4.0f));
////	materialsVec[matIdx + 3].pureRefl = true;

	//red cook torrance
	materialsVec.push_back(Material(vec3(1.0f, 0.1f, 0.1f), 0.1f, vec3(1.0f, 0.2f, 0.2f), INFINITY, 0.6f, 1.7f));
	materialsVec[matIdx].flags |= MAT_FLAG_COOK_TORRANCE;
	materialsVec[matIdx].roughness = 0.3f;

	//blue cook torrance
	materialsVec.push_back(Material(vec3(0.1f, 0.1f, 1.0f), 0.1f, vec3(0.2f, 0.2f, 1.0f), INFINITY, 0.6f, 1.7f));
	materialsVec[matIdx + 1].flags |= MAT_FLAG_COOK_TORRANCE;
	materialsVec[matIdx + 1].roughness = 0.1f;

	//green cook torrance
	materialsVec.push_back(Material(vec3(0.1f, 1.0f, 0.1f), 0.1f, vec3(0.2f, 1.f, 0.2f), INFINITY, 0.6f, 1.7f));
	materialsVec[matIdx + 2].flags |= MAT_FLAG_COOK_TORRANCE;
	materialsVec[matIdx + 2].roughness = 0.5f;

	
	for (size_t i = 0; i < numSpheres; i++)
	{
		Sphere s;

		rnd(1); rnd(1);

		s.position = vec3(rnd(5.0f) - 2.5f, rnd(5.0f) - 2.5f, rnd(7.0f) - 9.0f);
		s.radius = rnd(1.0f) + 0.2f;
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

	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 0.8f), 0.7f));	//white			(+0)
	materialsVec.push_back(Material(vec3(1.0f, 0.0f, 0.0f), 0.7f));	//red			(+1)
	materialsVec.push_back(Material(vec3(0.0f, 1.0f, 0.0f), 0.7f));	//green			(+2)


	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f)));			//white light	(+3)
	materialsVec.push_back(Material(vec3(0.0f, 0.0f, 0.0f), 0.0f, vec3(1, 1, 1), INFINITY, 0.8f, 5.8f));	//mirror		(+4)
	//materialsVec[matIdx + 4].flags |= MAT_FLAG_PURE_REFLECTION;

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

	//mirror
	trans = translate(vec3(offset - .02f, 0, -offset)) *
		rotate((glm::mediump_float) - 90, vec3(0, 1, 0)) *
		(scale(vec3(wallSize - 2, wallSize - 2, wallSize - 2)));
	addRectangularModel(trans, matIdx + 4);

	//back wall
	trans = translate(vec3(0, 0, -wallSize)) *
		//		rotate((glm::mediump_float)90, vec3(1, 0, 0)) *
		scaleToWall;
	addRectangularModel(trans, matIdx);

	//light
	float power = 400;
	trans = translate(vec3(0, offset - 0.01f, -offset)) *
		rotate((glm::mediump_float) 90, vec3(1, 0, 0)) *
		scale(vec3(2.5f, 2.5f, 2.5f));
	addAreaLight(trans, matIdx + 3, vec3(power, power, power));
}


/// <summary>
/// Adds a mirror box of a given size, centered at 0, with no back.
/// </summary>
/// <param name="wallSize">Size of the walls.</param>
void Scene::addMirrorBox(const float wallSize)
{
	using glm::translate;
	using glm::scale;
	using glm::rotate;

	int matIdx = materialsVec.size();

	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 0.8f), 0.7f));	//white			(+0)
	materialsVec.push_back(Material(vec3(1.0f, 0.0f, 0.0f), 0.7f));	//red			(+1)
	materialsVec.push_back(Material(vec3(0.0f, 1.0f, 0.0f), 0.7f));	//green			(+2)


	materialsVec.push_back(Material(vec3(1.0f, 1.0f, 1.0f)));			//white light	(+3)
	materialsVec.push_back(Material(vec3(0.0f, 0.0f, 0.0f), 0.0f, vec3(1, 1, 1), INFINITY, .9f, 5.8f));	//mirror		(+4)
	materialsVec[matIdx + 4].flags |= MAT_FLAG_PURE_REFLECTION;
	materialsVec.push_back(Material(vec3(1.0f, 0.6f, 1.0f)));			//violet light

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
	addRectangularModel(trans, matIdx + 4);

	//left wall
	trans = translate(vec3(-offset + .2 * offset, 0, -offset)) *
		rotate((glm::mediump_float)88, vec3(0, 1, 0)) *
		scaleToWall;
	addRectangularModel(trans, matIdx + 4);

	//right wall
	trans = translate(vec3(offset, 0, -offset)) *
		rotate((glm::mediump_float) - 90, vec3(0, 1, 0)) *
		scaleToWall;
	addRectangularModel(trans, matIdx + 4);

	//back wall
	trans = translate(vec3(0, 0, -wallSize)) *
		//		rotate((glm::mediump_float)90, vec3(1, 0, 0)) *
		scaleToWall;
	addRectangularModel(trans, matIdx);

	//front wall s
	trans = translate(vec3(0, 0, 0)) *
		rotate((glm::mediump_float)180, vec3(0, 1, 0)) *
		scaleToWall;
	addRectangularModel(trans, matIdx);

	//light
	float power = 400;
	trans = translate(vec3(0, offset - 0.01f, -offset)) *
		rotate((glm::mediump_float) 90, vec3(1, 0, 0)) *
		scale(vec3(2.5f, 2.5f, 2.5f));
	addAreaLight(trans, matIdx + 3, vec3(power / 4, power, power));


	trans = translate(vec3(0, -offset + 0.01f, -offset)) *
		rotate((glm::mediump_float) -90, vec3(1, 0, 0)) *
		scale(vec3(1.5f, 1.5f, 1.5f));
	addAreaLight(trans, matIdx + 5, vec3(power / 3, 0, power / 3));
}