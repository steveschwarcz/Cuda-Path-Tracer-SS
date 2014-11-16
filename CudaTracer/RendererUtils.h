#include <glm/glm.hpp>
#include "Primitives.h"

/// <summary>
/// Initializes a rectangular model composed of 2 triangles.
/// </summary>
/// <param name="transformation">The transformation to use on the model as it is created.</param>
/// <param name="triangle0">The triangle0.</param>
/// <param name="triangle1">The triangle1.</param>
static void initRectangularModel(mat4 transformation, Triangle* triangle0, Triangle* triangle1)
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

	//add all vertexes
	triangle0->vertexes[0] = vec3(newVertex[0]);
	triangle0->vertexes[0] = vec3(newVertex[1]);
	triangle0->vertexes[0] = vec3(newVertex[2]);
	triangle1->vertexes[0] = vec3(newVertex[3]);
	triangle1->vertexes[0] = vec3(newVertex[1]);
	triangle1->vertexes[0] = vec3(newVertex[2]);

	//add normals
	for (int i = 0; i < 3; i++)
	{
		triangle0->normals[i] = vec3(newNormal);
		triangle1->normals[i] = vec3(newNormal);
	}
}