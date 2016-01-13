#pragma once
#include <armadillo>
#include <algorithm>


namespace armapca {


template<typename T>
arma::Mat<T> covariance_matrix(const arma::Mat<T>& matrix)
{
    return (arma::trans(matrix) * matrix) * (1. / (matrix.n_rows - 1));
}


template<typename T, typename RandGen>
arma::Mat<T> shuffled_matrix(const arma::Mat<T>& matrix, RandGen gen)
{
    // ensure indices are unique
}


template<typename T>
void enforce_positive_sign_by_column(arma::Mat<T>& matrix)
{
    for (std::size_t i=0; i<matrix.n_cols; ++i) {
        const arma::uvec indices = arma::sort_index(matrix.col(i), 1);
        const auto min = matrix.col(i).at(indices(0));
        const auto max = matrix.col(i).at(indices(indices.n_elem - 1));
        bool change_sign = false;
        if (std::abs(max) >= std::abs(min)) {
            if (max < 0) change_sign = true;
        } else {
            if (min < 0) change_sign = true;
        }
        if (change_sign) matrix.col(i) *= static_cast<T>(-1);
    }
}


template<typename T, typename Functor>
arma::Row<T> column_apply(const arma::Mat<T>& matrix, Functor functor)
{
    arma::Row<T> result(matrix.n_cols);
    std::size_t index = 0;
    std::generate(result.begin(), result.end(), [&index]{
        return Functor(matrix.col(index++));
    });
    return result;
}


template<typename T>
arma::Row<T> column_mean(const arma::Mat<T>& matrix)
{
    return column_apply(matrix, arma::mean<T>);
}


template<typename T>
arma::Row<T> column_stddev(const arma::Mat<T>& matrix)
{
    return column_apply(matrix, arma::stddev<T>);
}


template<typename T, typename Vector>
void remove_by_column(arma::Mat<T>& matrix, Vector vector)
{
    for (std::size_t i=0; i<matrix.n_cols; ++i)
        matrix.col(i) -= vector(i);
}


template<typename T, typename Vector>
void divide_by_column(arma::Mat<T>& matrix, Vector vector)
{
    for (std::size_t i=0; i<matrix.n_cols; ++i)
        matrix.col(i) *= static_cast<T>(1) / vector(i);
}


template<typename T>
void remove_mean(arma::Mat<T>& matrix)
{
    const auto mean = armapca::column_mean(matrix);
    armapca::remove_by_column(matrix, mean);
}


template<typename T>
void normalize(arma::Mat<T>& matrix)
{
    const auto stddev = armapca::column_stddev(matrix);
    armapca::divide_by_column(matrix, stddev);
}


template<typename T>
struct pca_result {
    arma::Mat<T> eigenvectors;
    arma::Col<T> eigenvalues;
    T energy;
};


template<typename T>
armapca::pca_result<T> pca(const arma::Mat<T>& data, const std::string& solver="standard")
{
    const auto n_vars = data.n_cols;
    arma::Mat<T> eigvec(n_vars, n_vars);
    arma::Col<T> eigval(n_vars);

    const auto cov_mat = armapca::covariance_matrix(data);
    arma::eig_sym(eigvec, eigval, cov_mat, solver.c_str());
    const arma::uvec indices = arma::sort_index(eigval, 1);

    armapca::pca_result<T> result = {arma::Mat<T>(n_vars, n_vars), arma::Col<T>(n_vars), 0};
    for (std::size_t i=0; i<n_vars; ++i) {
        result.eigenvalues(i) = eigval(indices(i));
        result.eigenvectors(i) = eigvec.col(indices(i));
    }

    armapca::enforce_positive_sign_by_column(result.eigenvectors);
    result.energy = arma::sum(result.eigenvalues);
    result.eigenvalues *= static_cast<T>(1) / result.energy;

    return result;
}


template<typename T>
T check_eigenvectors_orthogonal(const armapca::pca_result<T>& result)
{
    return std::abs(arma::det(result.eigenvectors));
}


template<typename T>
T check_projection_accurate(const arma::Mat<T>& data, const armapca::pca_result<T>& result)
{
    const arma::Mat<double> diff = ((data * result.eigenvectors) * arma::trans(result.eigenvectors)) - data;
    return static_cast<T>(1) - arma::sum(arma::sum( arma::abs(diff) )) / diff.n_elem;
}


} // armapca
