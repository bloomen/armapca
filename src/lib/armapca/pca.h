#pragma once
#include <armadillo>
#include <algorithm>
#include <random>


namespace armapca {


template<typename T>
arma::Mat<T> covariance_matrix(const arma::Mat<T>& matrix) {
  return (arma::trans(matrix) * matrix) * (1. / (matrix.n_rows - 1));
}


template<typename T, typename RandGen>
arma::Mat<T> shuffled_matrix(const arma::Mat<T>& matrix, RandGen* gen) {
  arma::Mat<T> shuffled(matrix.n_rows, matrix.n_cols);
  std::vector<std::size_t> indices(matrix.n_rows);
  std::iota(indices.begin(), indices.end(), 0);
  for (std::size_t j=0; j < matrix.n_cols; ++j) {
    std::shuffle(indices.begin(), indices.end(), *gen);
    for (std::size_t i=0; i < matrix.n_rows; ++i) {
      shuffled(i, j) = matrix(indices[i], j);
    }
  }
  return shuffled;
}


template<typename T>
void enforce_positive_sign_by_column(arma::Mat<T>* matrix) {
  for (std::size_t i=0; i < matrix->n_cols; ++i) {
    const arma::uvec indices = arma::sort_index(matrix->col(i), 1);
    const auto min = matrix->col(i)(indices(0));
    const auto max = matrix->col(i)(indices(indices.n_elem - 1));
    bool change_sign = false;
    if (std::abs(max) >= std::abs(min)) {
      if (max < 0) change_sign = true;
    } else {
      if (min < 0) change_sign = true;
    }
    if (change_sign) matrix->col(i) *= T{-1};
  }
}


template<typename T, typename Functor>
arma::Row<T> column_apply(const arma::Mat<T>& matrix, Functor functor) {
  arma::Row<T> result(matrix.n_cols);
  std::size_t index = 0;
  std::generate(result.begin(), result.end(), [&matrix, &index] {
    return Functor(matrix.col(index++));
  });
  return result;
}


template<typename T>
arma::Row<T> column_mean(const arma::Mat<T>& matrix) {
  return column_apply(matrix, arma::mean<T>);
}


template<typename T>
arma::Row<T> column_stddev(const arma::Mat<T>& matrix) {
  return column_apply(matrix, arma::stddev<T>);
}


template<typename T, typename Vector>
void remove_by_column(arma::Mat<T>* matrix, const Vector& vector) {
  for (std::size_t i=0; i < matrix->n_cols; ++i)
    matrix->col(i) -= vector(i);
}


template<typename T, typename Vector>
void divide_by_column(arma::Mat<T>* matrix, const Vector& vector) {
  for (std::size_t i=0; i < matrix->n_cols; ++i)
    matrix->col(i) *= T {1} / vector(i);
}


template<typename T>
void remove_mean(arma::Mat<T>* matrix) {
  const auto mean = armapca::column_mean(matrix);
  armapca::remove_by_column(matrix, mean);
}


template<typename T>
void normalize(arma::Mat<T>* matrix) {
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
armapca::pca_result<T> pca(const arma::Mat<T>& data,
                           bool compute_eigenvectors = true,
                           const std::string& solver = "dc") {
  const auto n_vars = data.n_cols;
  arma::Mat<T> eigvec;
  if (compute_eigenvectors)
    eigvec.set_size(n_vars, n_vars);
  arma::Col<T> eigval(n_vars);

  const auto cov_mat = armapca::covariance_matrix(data);
  if (compute_eigenvectors)
    arma::eig_sym(eigval, eigvec, cov_mat, solver.c_str());
  else
    arma::eig_sym(eigval, cov_mat, solver.c_str());
  const arma::uvec indices = arma::sort_index(eigval, 1);

  armapca::pca_result<T> result = {arma::Mat<T>(n_vars, n_vars),
      arma::Col<T>(n_vars), 0};
  for (std::size_t i=0; i < n_vars; ++i) {
    result.eigenvalues(i) = eigval(indices(i));
  }

  if (compute_eigenvectors) {
    for (std::size_t i=0; i < n_vars; ++i) {
      result.eigenvectors.col(i) = eigvec.col(indices(i));
    }
    armapca::enforce_positive_sign_by_column(&result.eigenvectors);
  }

  result.energy = arma::sum(result.eigenvalues);
  result.eigenvalues *= T {1} / result.energy;

  return result;
}


template<typename T>
std::vector<armapca::pca_result<T>> pca_bootstrap(
    const arma::Mat<T>& data,
    std::size_t n_bootstraps = 100,
    bool compute_eigenvectors = true,
    std::size_t random_seed = 1,
    const std::string& solver = "dc") {
  std::vector<armapca::pca_result<T>> result(n_bootstraps);
  std::mt19937 gen{random_seed};
  for (std::size_t i=0; i < n_bootstraps; ++i) {
    const auto shuffled = armapca::shuffled_matrix(data, &gen);
    result[i] = armapca::pca(shuffled, compute_eigenvectors, solver);
  }
  return result;
}


template<typename T>
T check_eigenvectors_orthogonal(const armapca::pca_result<T>& result) {
  return std::abs(arma::det(result.eigenvectors));
}


template<typename T>
T check_projection_accurate(const arma::Mat<T>& data,
                            const armapca::pca_result<T>& result) {
  const arma::Mat<double> diff = ((data * result.eigenvectors) *
      arma::trans(result.eigenvectors)) - data;
  return T {1} - arma::sum(arma::sum(arma::abs(diff) )) / diff.n_elem;
}


}  // namespace armapca
