#pragma once
#include "vector.h"
#include <string>
#include <vector>

// Interface for matrix
class IMatrix
{
public:
    virtual double operator()(int i, int j) const noexcept = 0;
    virtual int rows() const noexcept = 0;
    virtual int cols() const noexcept = 0;
    virtual ~IMatrix() = default;

    bool operator==(const IMatrix& o) const noexcept
	{
		if (rows() != o.rows() || cols() != o.cols())
			return false;
		for (int i = 0; i < rows(); ++i)
			for (int j = 0; j < cols(); ++j)
				if ((*this)(i, j) != o(i, j))
					return false;
		return true;
	}
};

inline std::string to_string(const IMatrix& m)
{
	std::string result = "[";
	for (int i = 0; i < m.rows(); ++i)
	{
		result += "[";
		for (int j = 0; j < m.cols(); ++j)
		{
			result += std::to_string(m(i, j));
			if (j < m.cols() - 1)
				result += ", ";
		}
		result += "]";
		if (i < m.rows() - 1)
			result += ", ";
	}
	result += "]";
	return result;
}

class DiagonalMatrix : public IMatrix
{
    double value;
    int size;
public:
    DiagonalMatrix(double value, int size) noexcept : value(value), size(size) {}
    double operator()(int i, int j) const noexcept override { return i == j ? value : 0; }
    int rows() const noexcept override { return size; }
    int cols() const noexcept override { return size; }
};

inline DiagonalMatrix IdentityMatrix(int size)
{
    return DiagonalMatrix(1, size);
}

inline DiagonalMatrix ZeroMatrix(int size)
{
    return DiagonalMatrix(0, size);
}

class TransposedMatrix : public IMatrix
{
    const IMatrix& matrix;
public:
    TransposedMatrix(const IMatrix& matrix) noexcept : matrix(matrix) {}
    double operator()(int i, int j) const noexcept override { return matrix(j, i); }
    int rows() const noexcept override { return matrix.cols(); }
    int cols() const noexcept override { return matrix.rows(); }
};

inline TransposedMatrix transposed(const IMatrix& matrix)
{
    return TransposedMatrix(matrix);
}

class HankelMatrix : public IMatrix
{
    const std::vector<double>& data;
    int cols_;
public:
    HankelMatrix(const std::vector<double>& data, int rows) noexcept : data(data), cols_(data.size() - rows + 1) {}
    double operator()(int i, int j) const noexcept override { return data[i + j]; }
    int rows() const noexcept override { return data.size() - cols_ + 1; }
    int cols() const noexcept override { return cols_; }
};

class SubMatrix : public IMatrix
{
    const IMatrix& matrix;
    int row_begin, rows_, col_begin, cols_;
public:
    SubMatrix(const IMatrix& matrix, int row_begin, int row_end, int col_begin, int col_end) noexcept
        : matrix(matrix), row_begin(row_begin), rows_(row_end - row_begin), col_begin(col_begin), cols_(col_end - col_begin)
    {
        assert(row_begin >= 0 && row_end <= matrix.rows());
        assert(col_begin >= 0 && col_end <= matrix.cols());
    }
    double operator()(int i, int j) const noexcept override { return matrix(row_begin + i, col_begin + j); }
    int rows() const noexcept override { return rows_; }
    int cols() const noexcept override { return cols_; }
};

class VStackMatrix : public IMatrix
{
    const IMatrix& a;
    const IMatrix& b;
public:
    VStackMatrix(const IMatrix& a, const IMatrix& b) noexcept : a(a), b(b)
    {
        assert(a.cols() == b.cols());
    }
    double operator()(int i, int j) const noexcept override
    {
        if (i < a.rows())
            return a(i, j);
        else
            return b(i - a.rows(), j);
    }
    int rows() const noexcept override { return a.rows() + b.rows(); }
    int cols() const noexcept override { return a.cols(); }
};

inline VStackMatrix vstack(const IMatrix& a, const IMatrix& b)
{
    return VStackMatrix(a, b);
}

inline VStackMatrix vstack(const IMatrix& a, const IMatrix& b, const IMatrix& c)
{
    return VStackMatrix(a, vstack(b, c));
}

class DenseMatrix : public IMatrix
{
    std::vector<double> data;
    int cols_;
public:
    DenseMatrix() noexcept = default;
    DenseMatrix(std::vector<double> data, int cols) noexcept : data(std::move(data)), cols_(cols)
    {
        assert(this->data.size() % cols_ == 0);
    }
    explicit DenseMatrix(const IMatrix& m) noexcept : data(m.rows()* m.cols()), cols_(m.cols())
    {
        for (int i = 0; i < m.rows(); ++i)
            for (int j = 0; j < m.cols(); ++j)
                data[i * cols_ + j] = m(i, j);
    }
    static DenseMatrix Zeros(int rows, int cols) noexcept { return DenseMatrix(std::vector<double>(rows * cols, 0), cols); }

    double operator()(int i, int j) const noexcept override { return data[i * cols_ + j]; }
    double& operator()(int i, int j) noexcept { return data[i * cols_ + j]; }
    int rows() const noexcept override { return data.size() / cols_; }
    int cols() const noexcept override { return cols_; }
};

// Matrix * Vector
inline std::vector<double> operator*(const IMatrix& a, const std::vector<double>& b)
{
    assert(a.cols() == b.size());
    int cols = a.cols();
    int rows = a.rows();
    std::vector<double> result(rows);
    for (int i = 0; i < rows; ++i)
    {
        double sum = 0; // Prevents cache thrashing
        for (int j = 0; j < cols; ++j)
            sum += a(i, j) * b[j];
        result[i] = sum;
    }
    return result;
}

// Matrix * Matrix
inline DenseMatrix operator*(const IMatrix& a, const IMatrix& b)
{
    assert(a.cols() == b.rows());
    int a_rows = a.rows();
    int a_cols = a.cols();
    int b_cols = b.cols();
    auto result = DenseMatrix::Zeros(a.rows(), b.cols());
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
        {
            double sum = 0; // Prevents cache thrashing
            for (int k = 0; k < a_cols; ++k)
                sum += a(i, k) * b(k, j);
            result(i, j) = sum;
        }
    return result;
}

// Matrix + Matrix
inline DenseMatrix operator+(const IMatrix& a, const IMatrix& b)
{
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    int a_rows = a.rows();
    int a_cols = a.cols();
    auto result = DenseMatrix::Zeros(a_rows, a_cols);
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < a_cols; ++j)
            result(i, j) = a(i, j) + b(i, j);
    return result;
}
