#include "matrix.h"
#include "vector.h"
#include <functional>
#include <tuple>
#include <vector>

DenseMatrix inv(const Matrix&);

double norm(const Matrix&);

// Returns matrix rank using SVD method
int matrix_rank(const Matrix&);

auto singular_value_decomposition(const Matrix&) -> std::tuple<DenseMatrix, DenseMatrix, DenseMatrix>;

DenseMatrix pseudoinverse(const Matrix&);
DenseMatrix left_pseudoinverse(const Matrix&);
DenseMatrix right_pseudoinverse(const Matrix&);

std::vector<double> solve(const Matrix&, const std::vector<double>&);

std::vector<double> projected_gradient_method(
    const Matrix& mat,
    const std::vector<double>& x_ini,
    const std::vector<double>& target,
    std::function<std::vector<double>(const std::vector<double>&)> constrain,
    int max_iterations = 300,
    double tolerance = 1e-6);
