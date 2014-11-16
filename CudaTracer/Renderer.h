#pragma once

#include "glm/glm.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using glm::vec3;
using glm::dot;

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


struct SurfaceElement {
	vec3 point;
	vec3 normal;

	__device__
	SurfaceElement() {};

	__device__
	SurfaceElement(vec3 point, vec3 normal) :
		point(point), normal(normal) {};
};

struct Sphere
{
	float radius;
	vec3 position;

	__device__ __host__
	Sphere(vec3 position, float radius) :
		radius(radius), position(position) {};

	__device__
	SurfaceElement getSurfaceElement(const Ray& ray, float distance) const
	{
		vec3 intersectionPoint = ray.origin + ray.direction * distance;
		vec3 normal = glm::normalize(intersectionPoint - position);

		return SurfaceElement(intersectionPoint, normal);
	}

	__device__
	bool intersectRay(const Ray& ray, float& distance, SurfaceElement& surfel) const
	{
		vec3 v = ray.origin - position;

		//a = 1, since all direction vectors are normal
		float b = dot(ray.direction,v) * 2;
		float c = dot(v, v) - (radius * radius);

		float discriminent = (b * b) - 4 * c;

		//negative discriminent = no real roots
		if (discriminent < 0)
			return false;

		discriminent = std::sqrt(discriminent);
		float t0 = (-b + discriminent) / 2;
		float t1 = (-b - discriminent) / 2;

		// make sure t0 is smaller than t1
		if (t0 > t1)
		{
			float temp = t0;
			t0 = t1;
			t1 = temp;
		}

		if (t1 < 0)
		{
			//misses sphere
			return false;
		}

		// intersects at t1, and ray is inside sphere
		if (t0 < 0)
		{
			//sphere too far away
			if (t1 > distance)
			{
				return false;
			}

			distance = t1;
			surfel = getSurfaceElement(ray, distance);
			return true;
		}
		// intersects at t0
		else
		{
			//sphere too far away
			if (t0 > distance)
			{
				return false;
			}

			distance = t0;
			surfel = getSurfaceElement(ray, distance);
			return true;
		}
	}
};
