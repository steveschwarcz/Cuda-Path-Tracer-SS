#pragma once

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include "glm/glm.hpp"
#include "cuda_runtime.h"

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
	vec3 directLightColor;
	vec3 indirectLightColor;
	int pixelOffset;
	bool active;

	__device__
	Ray(vec3 origin, vec3 direction) :
	direction(direction), origin(origin), active(true), directLightColor(0, 0, 0), indirectLightColor(0, 0, 0) {};

	__device__
	explicit Ray(bool alive) :
		active(alive) {};
};

//Camera
struct Camera
{
	quat rotation;
	vec3 position;
	float zNear;
	float zFar;
	float fieldOfView;

	Camera() :
		zNear(-0.1f), zFar(-100.0f), fieldOfView(float(M_PI) / 2.0f), position(0,0,0) {};
};

//Point Light
struct PointLight
{
	vec3 position;
	vec3 power;

	PointLight(vec3 position, vec3 power) :
		position(position), power(power) {}
};

//Area Light - only 2 triangles for now
struct AreaLight
{
	vec3 power;
	int triangleIdx0;
	int triangleIdx1;

	AreaLight(int triangleIdx0, int triangleIdx1) :
		triangleIdx0(triangleIdx0), triangleIdx1(triangleIdx1), power(power) {}
};


struct Material
{
	vec3 diffuseColor;
	vec3 specularColor;
	vec3 absorption;
	vec3 emmitance;
	float specularExponent;
	float indexOfRefraction;
	float diffAvg;
	float specAvg;
	float refrAvg;


	Material() {};


	Material(vec3 diffuseColor, float diffAvg, vec3 specularColor = vec3(0, 0, 0), float specularExponent = 0, float specAvg = 0, 
		vec3 absorption = vec3(0, 0, 0), float indexOfRefraction = 1.0f, float refrAvg = 0, vec3 emmitance = vec3(0, 0, 0)) :
			diffuseColor(diffuseColor), diffAvg(diffAvg), specularColor(specularColor), specularExponent(specularExponent), specAvg(specAvg), 
			absorption(absorption), indexOfRefraction(indexOfRefraction), refrAvg(refrAvg), emmitance(emmitance) {};

	explicit Material(vec3 emmitance) :
		diffuseColor(vec3(0, 0, 0)), diffAvg(0), specularColor(vec3(0, 0, 0)), specularExponent(0), specAvg(0),
		absorption(vec3(0, 0, 0)), indexOfRefraction(1.0f), refrAvg(0), emmitance(emmitance) {};

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