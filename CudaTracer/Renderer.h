#pragma once

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include "glm/glm.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


using glm::quat;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::dot;
using glm::cross;
using glm::normalize;



//ray
struct Ray
{
	vec3 origin;
	vec3 direction;
	char alive;

	__device__
	Ray(vec3 origin, vec3 direction) :
		direction(direction), origin(origin), alive(1) {};
};

//Camera
struct Camera
{
	float zNear;
	float zFar;
	float fieldOfView;
	vec3 position;
	quat rotation;

	__device__ __host__
	Camera() :
		zNear(-0.1f), zFar(-100.0f), fieldOfView(float(M_PI) / 2.0f), position(0,0,0) {};
};

//Point Light
struct PointLight
{
	vec3 position;
	vec3 power;

	__device__ __host__
	PointLight(vec3 position, vec3 power) :
		position(position), power(power) {}
};


struct Material
{
	vec3 diffuseColor;
	vec3 specularColor;
	float specularExponent;
	vec3 absorption;
	float indexOfRefraction;
	vec3 emmitance;

	__device__ __host__
	Material() {};

	__host__
	Material(vec3 diffuseColor) : diffuseColor(diffuseColor) {};

};

//Surface Element representing point of intersection
struct SurfaceElement {
	vec3 point;
	vec3 normal;
	int materialIdx;

	__device__
	SurfaceElement() {};

	__device__
	SurfaceElement(vec3 point, vec3 normal, int materialIdx) :
		point(point), normal(normal), materialIdx(materialIdx) {};
};