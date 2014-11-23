#include "Scene.h"


Scene::Scene()
{
}


Scene::~Scene()
{
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