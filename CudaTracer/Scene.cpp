#include "Scene.h"


Scene::Scene()
{
}


Scene::~Scene()
{
}

void Scene::addRectangularModel(mat4 transformation, int materialIdx)
{
	vec4 newVertex[4];

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			//transform the new point
			newVertex[i] = vec4(i - 0.5f, j - 0.5f, 0.0f, 1);
			newVertex[i] = transformation * newVertex[i];

			//TODO: add vertex to a vertexes array
		}
	}

	vec4 newNormal(0.0f, 0.0f, 1.0f, 0);
	//get the inverse-transpose of the transformation matrix, to properly transform the normals
	newNormal = glm::transpose(glm::inverse(transformation)) * newNormal;

	//normalize the new normal
	normalize(newNormal);

	vec3 vertexes0[3], vertexes1[3], normals[3];

	//add normals and vertexes
	for (int i = 0; i < 3; i++)
	{
		vertexes0[i] = vec3(newVertex[i]);
		vertexes1[i] = vec3(newVertex[i == 0 ? 3 : i]);

		normals[i] = vec3(newNormal);
	}

	//push triangle into vector
	trianglesVec.push_back(Triangle(vertexes0, normals, materialIdx));
	trianglesVec.push_back(Triangle(vertexes1, normals, materialIdx));
}