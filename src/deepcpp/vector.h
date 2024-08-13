#pragma once
#include <string>
#include <vector>

using Vector = std::vector<double>;

Vector operator+(Vector, const Vector&);
Vector operator+(const Vector&, Vector&&);
Vector operator-(Vector, const Vector&);
Vector operator-(const Vector&, Vector&&);
Vector operator*(Vector, double);
Vector operator*(double, Vector);
Vector operator/(Vector, double);

double dot(const Vector&, const Vector&);
double norm(const Vector&);
double sum(const Vector&);

Vector concat(Vector, const Vector&);

std::string to_string(const Vector&);
