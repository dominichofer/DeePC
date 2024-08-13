#include "vector.h"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

Vector operator+(Vector l, const Vector& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] += r[i];
	return l;
}

Vector operator+(const Vector& l, Vector&& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		r[i] += l[i];
	return r;
}

Vector operator-(Vector l, const Vector& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] -= r[i];
	return l;
}

Vector operator-(const Vector& l, Vector&& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		r[i] = l[i] - r[i];
	return r;
}

Vector operator*(Vector v, double factor)
{
	int64_t size = static_cast<int64_t>(v.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		v[i] *= factor;
	return v;
}

Vector operator*(double factor, Vector v)
{
	return v * factor;
}

Vector operator/(Vector v, double factor)
{
	int64_t size = static_cast<int64_t>(v.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		v[i] /= factor;
	return v;
}

double dot(const Vector& l, const Vector& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	double result = 0.0f;
	#pragma omp parallel for schedule(static) reduction(+:result)
	for (int64_t i = 0; i < size; i++)
		result += l[i] * r[i];
	return result;
}

double norm(const Vector& x)
{
	return std::sqrt(dot(x, x));
}

double sum(const Vector& x)
{
	int64_t size = static_cast<int64_t>(x.size());
	double result = 0.0f;
	#pragma omp parallel for schedule(static) reduction(+:result)
	for (int64_t i = 0; i < size; i++)
		result += x[i];
	return result;
}

Vector concat(Vector a, const Vector& b)
{
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

std::string to_string(const Vector& v)
{
	std::string str = "[";
	for (size_t i = 0; i < v.size(); i++)
	{
		str += std::to_string(v[i]);
		if (i < v.size() - 1)
			str += ", ";
	}
	str += "]";
	return str;
}
