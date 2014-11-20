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