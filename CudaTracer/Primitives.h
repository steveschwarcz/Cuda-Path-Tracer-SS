#pragma once
#include "cuda_runtime.h"

#include "RenderStructs.h"

class Triangle
{
public:
	//TODO: Refactor: find a better way to do this
	vec3 vertex0;
	vec3 vertex1;
	vec3 vertex2;
	vec3 normal0;
	vec3 normal1;
	vec3 normal2;
	int materialIdx;

	Triangle()
	{
	}

	Triangle(vec3 vertex0, vec3 vertex1, vec3 vertex2, vec3 normal0, vec3 normal1, vec3 normal2, int materialIdx) :
		vertex0(vertex0), vertex1(vertex1), vertex2(vertex2), normal0(normal0), normal1(normal1), normal2(normal2), materialIdx(materialIdx) {};

	__device__
	bool intersectRay(const Ray& ray, float& distance, SurfaceElement& surfel, bool updateSurfel = true) const
	{
		//intersect ray with triangle using the M�ller�Trumbore intersection algorithm

		//epsilon to check bounds of a if a is close to 0
		const float epsilon = 1e-7f;

		float weight[3];

		const vec3 e1 = vertex1 - vertex0;
		const vec3 e2 = vertex2 - vertex0;
		const vec3 q = cross(ray.direction, e2);

		const float det = dot(e1,q);

		if (fabs(det) <= epsilon) {
			return false;
		}

		const float inverseDet = 1 / det;

		const vec3 s = ray.origin - vertex0;
		const vec3 r = cross(s, e1);

		const float dist = dot(e2, r) * inverseDet;

		if ((dist <= 0.0f) || (dist > distance))
		{
			return false;
		}

		//calculate barycentric weights
		weight[1] = dot(s, q) * inverseDet;
		weight[2] = dot(ray.direction, r) * inverseDet;
		weight[0] = 1.0f - (weight[1] + weight[2]);

		if ((weight[0] < 0) || (weight[1] < 0)
			|| (weight[2] < 0))
		{
			//ray is parellel to triangle, or behind triangle, or misses triangle
			return false;
		}

		if (updateSurfel)
		{
			vec3 normalResult = normal0 * weight[0] +
				normal1 * weight[1] +
				normal2 * weight[2];

			vec3 intersectionPoint = ray.origin + ray.direction * dist;

			surfel = SurfaceElement(intersectionPoint, normalResult, materialIdx);
		}

		distance = dist;

		return true;
	}
};

class Sphere
{
public:
	vec3 position;
	float radius;
	int materialIdx;

	Sphere(){};

	Sphere(vec3 position, float radius, int materialIdx) :
		radius(radius), position(position), materialIdx(materialIdx){};

	__device__
	SurfaceElement getSurfaceElement(const Ray& ray, float distance) const
	{
		vec3 intersectionPoint = ray.origin + ray.direction * distance;
		vec3 normal = glm::normalize(intersectionPoint - position);

		return SurfaceElement(intersectionPoint, normal, materialIdx);
	}

	__device__
	bool intersectRay(const Ray& ray, float& distance, SurfaceElement& surfel, bool updateSurfel = true) const
	{
		vec3 v = ray.origin - position;

		//a = 1, since all direction vectors are normal
		float b = dot(ray.direction, v) * 2;
		float c = dot(v, v) - (radius * radius);

		float discriminent = (b * b) - 4 * c;

		//negative discriminent = no real roots
		if (discriminent < 0)
			return false;

		discriminent = std::sqrt(discriminent);
		float t0 = (-b + discriminent) * 0.5f;
		float t1 = (-b - discriminent) * 0.5f;

		if (t0 < 0 && t1 < 0)
		{
			//misses sphere
			return false;
		}

		////TODO: REMOVE
		//if (sqrt(dotVV) < radius) {
		//	
		//}

		// make sure t0 is smaller than t1
		if (t0 > t1)
		{
			float temp = t0;
			t0 = t1;
			t1 = temp;
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
			if (updateSurfel) {
				surfel = getSurfaceElement(ray, distance);
			}
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
			if (updateSurfel) {
				surfel = getSurfaceElement(ray, distance);
			}
			return true;
		}
	}
};
