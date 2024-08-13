#include "matrix.h"
#include "vector.h"
#include <functional>
#include <tuple>
#include <vector>

DenseMatrix inv(const IMatrix&);
double norm(const IMatrix&);
auto singular_value_decomposition(const IMatrix&) -> std::tuple<DenseMatrix, DenseMatrix, DenseMatrix>;

DenseMatrix pseudoinverse(const IMatrix&);
DenseMatrix left_pseudoinverse(const IMatrix&);
DenseMatrix right_pseudoinverse(const IMatrix&);

std::vector<double> solve(const IMatrix&, const std::vector<double>&);

std::vector<double> projected_gradient_method(
    const IMatrix& mat,
    const std::vector<double>& x_ini,
    const std::vector<double>& target,
    std::function<std::vector<double>(const std::vector<double>&)> constrain,
    int max_iterations = 300,
    double tolerance = 1e-6);
