#include "matrix.h"
#include <cassert>
#include <cmath>

bool Matrix::operator==(const Matrix& o) const
{
    if (rows() != o.rows() || cols() != o.cols())
        return false;
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            if ((*this)(i, j) != o(i, j))
                return false;
    return true;
}

std::string to_string(const Matrix& m)
{
    using std::to_string;
	std::string str = "[";
	for (int i = 0; i < m.rows(); ++i)
	{
		str += "[";
		for (int j = 0; j < m.cols(); ++j)
		{
			str += to_string(m(i, j));
			if (j < m.cols() - 1)
				str += ", ";
		}
		str += "]";
		if (i < m.rows() - 1)
			str += ", ";
	}
	str += "]";
	return str;
}

DiagonalMatrix IdentityMatrix(int size)
{
    return DiagonalMatrix(std::vector<double>(size, 1));
}

DiagonalMatrix ZeroMatrix(int size)
{
    return DiagonalMatrix(std::vector<double>(size, 0));
}

TransposedMatrix transposed(const Matrix& matrix)
{
    return TransposedMatrix(matrix);
}

SubMatrix::SubMatrix(const Matrix& matrix, int row_begin, int row_end, int col_begin, int col_end) noexcept
    : matrix(matrix), row_begin(row_begin), rows_(row_end - row_begin), col_begin(col_begin), cols_(col_end - col_begin)
{
    assert(row_begin >= 0 && row_end <= matrix.rows());
    assert(col_begin >= 0 && col_end <= matrix.cols());
}

VStackMatrix::VStackMatrix(const Matrix& a, const Matrix& b) noexcept : a(a), b(b)
{
    assert(a.cols() == b.cols());
}

double VStackMatrix::operator()(int i, int j) const
{
    if (i < a.rows())
        return a(i, j);
    else
        return b(i - a.rows(), j);
}

VStackMatrix vstack(const Matrix& a, const Matrix& b)
{
    return VStackMatrix(a, b);
}

VStackMatrix vstack(const Matrix& a, const Matrix& b, const Matrix& c)
{
    return VStackMatrix(a, vstack(b, c));
}

DenseMatrix::DenseMatrix(std::vector<double> data, int cols) noexcept : data(std::move(data)), cols_(cols)
{
    assert(this->data.size() % cols_ == 0);
}

DenseMatrix::DenseMatrix(const Matrix& m) noexcept : data(m.rows()* m.cols()), cols_(m.cols())
{
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j)
            data[i * cols_ + j] = m(i, j);
}
DenseMatrix DenseMatrix::Zeros(int rows, int cols) noexcept
{
    return DenseMatrix(std::vector<double>(rows * cols, 0), cols);
}

std::vector<double> operator*(const Matrix& a, const std::vector<double>& b)
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

DenseMatrix operator*(const Matrix& a, const Matrix& b)
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

DenseMatrix operator+(const Matrix& a, const Matrix& b)
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

DenseMatrix operator-(const Matrix& a, const Matrix& b)
{
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    int a_rows = a.rows();
    int a_cols = a.cols();
    auto result = DenseMatrix::Zeros(a_rows, a_cols);
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < a_cols; ++j)
            result(i, j) = a(i, j) - b(i, j);
    return result;
}

DenseMatrix pow(const Matrix& a, int n)
{
    assert(a.rows() == a.cols());
    if (n == 0)
        return IdentityMatrix(a.rows());
    if (n == 1)
        return a;
    if (n % 2 == 0)
    {
        auto half = pow(a, n / 2);
        return half * half;
    }
    else
        return a * pow(a, n - 1);
}
