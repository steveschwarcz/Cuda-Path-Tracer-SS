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
	std::vector<AreaLight> areaLightsVec;

	void addRectangularModel(mat4 transformation, int materialIdx);

	void addAreaLight(mat4 transformation, int materialIdx, vec3 power);

	void build();
	void addRandomSpheres(const size_t numSpheres);
	void addCornellBox(const float wallSize);
	void addDefinedSpheres(const float size);
};

