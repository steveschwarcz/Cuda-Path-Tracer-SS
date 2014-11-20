#pragma once
#include "Primitives.h"
#include <vector>

class Scene
{
public:
	Scene();
	~Scene();

	std::vector<Sphere> spheresVec;
	std::vector<Triangle> trianglesVec;
	std::vector<Material> materialsVec;
	std::vector<PointLight> pointLightsVec;
	std::vector<int> areaLightIdxVec;		//stores the area lights as ID number of associated triangles

	/// <summary>
	/// Initializes a rectangular model composed of 2 triangles.
	/// </summary>
	/// <param name="transformation">The transformation to use on the model as it is created.</param>
	/// <param name="materialIdx">The index of the material to use.</param>
	void addRectangularModel(mat4 transformation, int materialIdx);


};

