#pragma once
#include "vector.h"
#include <string>
#include <vector>

// Interface for matrix
class Matrix
{
public:
    virtual double operator()(int i, int j) const = 0;
    virtual int rows() const = 0;
    virtual int cols() const = 0;
    virtual ~Matrix() = default;

    bool operator==(const Matrix &) const;
};

class DiagonalMatrix : public Matrix
{
    std::vector<double> data;

public:
    DiagonalMatrix(std::vector<double> data) : data(std::move(data)) {}
    inline double operator()(int i, int j) const override { return i == j ? data[i] : 0; }
    inline int rows() const override { return data.size(); }
    inline int cols() const override { return data.size(); }
};

class TransposedMatrix : public Matrix
{
    const Matrix &matrix;

public:
    TransposedMatrix(const Matrix &matrix) noexcept : matrix(matrix) {}
    inline double operator()(int i, int j) const override { return matrix(j, i); }
    inline int rows() const override { return matrix.cols(); }
    inline int cols() const override { return matrix.rows(); }
};

class VStackMatrix : public Matrix
{
    const Matrix &a;
    const Matrix &b;

public:
    VStackMatrix(const Matrix &a, const Matrix &b) noexcept;
    double operator()(int i, int j) const override;
    inline int rows() const override { return a.rows() + b.rows(); }
    inline int cols() const override { return a.cols(); }
};

class HankelMatrix : public Matrix
{
    const std::vector<double> &data;
    int cols_;

public:
    HankelMatrix(const std::vector<double> &data, int rows) noexcept : data(data), cols_(data.size() - rows + 1) {}
    inline double operator()(int i, int j) const override { return data[i + j]; }
    inline int rows() const override { return data.size() - cols_ + 1; }
    inline int cols() const override { return cols_; }
};

class SubMatrix : public Matrix
{
    const Matrix &matrix;
    int row_begin, rows_, col_begin, cols_;

public:
    SubMatrix(const Matrix &, int row_begin, int row_end, int col_begin, int col_end) noexcept;
    inline double operator()(int i, int j) const override { return matrix(row_begin + i, col_begin + j); }
    inline int rows() const override { return rows_; }
    inline int cols() const override { return cols_; }
};

class DenseMatrix : public Matrix
{
    std::vector<double> data;
    int cols_;

public:
    DenseMatrix() noexcept = default;
    DenseMatrix(std::vector<double> data, int cols) noexcept;
    DenseMatrix(const Matrix &) noexcept;
    static DenseMatrix Zeros(int rows, int cols) noexcept;

    inline double operator()(int i, int j) const override { return data[i * cols_ + j]; }
    inline double &operator()(int i, int j) { return data[i * cols_ + j]; }
    inline int rows() const override { return data.size() / cols_; }
    inline int cols() const override { return cols_; }
};

std::string to_string(const Matrix &);

DiagonalMatrix IdentityMatrix(int size);
DiagonalMatrix ZeroMatrix(int size);

TransposedMatrix transposed(const Matrix &);

VStackMatrix vstack(const Matrix &, const Matrix &);
VStackMatrix vstack(const Matrix &, const Matrix &, const Matrix &);

std::vector<double> operator*(const Matrix &, const std::vector<double> &);
DenseMatrix operator*(const Matrix &, const Matrix &);
DenseMatrix operator+(const Matrix &, const Matrix &);
DenseMatrix operator-(const Matrix &, const Matrix &);
DenseMatrix pow(const Matrix &, int);
