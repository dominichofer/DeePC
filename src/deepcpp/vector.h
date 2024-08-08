#pragma once
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

using Vector = std::vector<double>;

inline Vector operator+(Vector l, const Vector& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] += r[i];
	return l;
}

inline Vector operator+(const Vector& l, Vector&& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		r[i] += l[i];
	return r;
}

inline Vector operator-(Vector l, const Vector& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		l[i] -= r[i];
	return l;
}

inline Vector operator-(const Vector& l, Vector&& r)
{
	if (l.size() != r.size())
		throw std::runtime_error("Size mismatch");

	int64_t size = static_cast<int64_t>(l.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		r[i] = l[i] - r[i];
	return r;
}

inline Vector operator*(Vector v, double factor)
{
	int64_t size = static_cast<int64_t>(v.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		v[i] *= factor;
	return v;
}

inline Vector operator*(double factor, Vector v)
{
	return v * factor;
}

inline Vector operator/(Vector v, double factor)
{
	int64_t size = static_cast<int64_t>(v.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		v[i] /= factor;
	return v;
}

inline Vector inv(Vector v)
{
	int64_t size = static_cast<int64_t>(v.size());
	#pragma omp parallel for schedule(static)
	for (int64_t i = 0; i < size; i++)
		v[i] = 1.0f / v[i];
	return v;
}

inline double dot(const Vector& l, const Vector& r)
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

inline double norm(const Vector& x)
{
	return std::sqrt(dot(x, x));
}

inline double sum(const Vector& x)
{
	int64_t size = static_cast<int64_t>(x.size());
	double result = 0.0f;
	#pragma omp parallel for schedule(static) reduction(+:result)
	for (int64_t i = 0; i < size; i++)
		result += x[i];
	return result;
}

inline Vector concat(Vector a, const Vector& b)
{
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
