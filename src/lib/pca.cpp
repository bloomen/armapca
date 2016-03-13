#include "armapca/pca.h"


namespace armapca {


#define EXPECTS(condition) \
if (!(condition)) throw std::logic_error( \
std::string(__func__) + "() expects: "#condition);


template<typename T>
arma::Mat<T> covariance_matrix(const arma::Mat<T>& matrix) {
  EXPECTS(matrix.n_rows >= 2);
  EXPECTS(matrix.n_cols >= 1);
  return (arma::trans(matrix) * matrix) * (T {1} / (matrix.n_rows - 1));
}
template arma::Mat<float> covariance_matrix(const arma::Mat<float>&);
template arma::Mat<double> covariance_matrix(const arma::Mat<double>&);


template<typename T>
arma::Mat<T> shuffled_matrix(const arma::Mat<T>& matrix, std::mt19937* gen) {
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
template arma::Mat<float> shuffled_matrix(
    const arma::Mat<float>&, std::mt19937*);
template arma::Mat<double> shuffled_matrix(
    const arma::Mat<double>&, std::mt19937*);


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
template void enforce_positive_sign_by_column(arma::Mat<float>*);
template void enforce_positive_sign_by_column(arma::Mat<double>*);


template<typename T, typename Functor>
arma::Row<T> column_apply(const arma::Mat<T>& matrix, Functor functor) {
  arma::Row<T> result(matrix.n_cols);
  std::size_t index = 0;
  std::generate(result.begin(), result.end(), [&matrix, &index, &functor] {
    return functor(matrix.col(index++));
  });
  return result;
}


template<typename T>
arma::Row<T> column_mean(const arma::Mat<T>& matrix) {
  return column_apply(matrix, [](const arma::subview_col<T>& v)
                              { return arma::mean(v); });
}
template arma::Row<float> column_mean(const arma::Mat<float>&);
template arma::Row<double> column_mean(const arma::Mat<double>&);


template<typename T>
arma::Row<T> column_stddev(const arma::Mat<T>& matrix) {
  return column_apply(matrix, [](const arma::subview_col<T>& v)
                              { return arma::stddev(v); });
}
template arma::Row<float> column_stddev(const arma::Mat<float>&);
template arma::Row<double> column_stddev(const arma::Mat<double>&);


template<typename T>
void remove_by_column(arma::Mat<T>* matrix, const arma::Row<T>& vector) {
  for (std::size_t i=0; i < matrix->n_cols; ++i)
    matrix->col(i) -= vector(i);
}
template void remove_by_column(arma::Mat<float>*, const arma::Row<float>&);
template void remove_by_column(arma::Mat<double>*, const arma::Row<double>&);


template<typename T>
void divide_by_column(arma::Mat<T>* matrix, const arma::Row<T>& vector) {
  for (std::size_t i=0; i < matrix->n_cols; ++i)
    matrix->col(i) *= T {1} / vector(i);
}
template void divide_by_column(arma::Mat<float>*, const arma::Row<float>&);
template void divide_by_column(arma::Mat<double>*, const arma::Row<double>&);


template<typename T>
void remove_mean(arma::Mat<T>* matrix) {
  const auto mean = armapca::column_mean(*matrix);
  armapca::remove_by_column(matrix, mean);
}
template void remove_mean(arma::Mat<float>* matrix);
template void remove_mean(arma::Mat<double>* matrix);


template<typename T>
void normalize(arma::Mat<T>* matrix) {
  const auto stddev = armapca::column_stddev(*matrix);
  armapca::divide_by_column(matrix, stddev);
}
template void normalize(arma::Mat<float>* matrix);
template void normalize(arma::Mat<double>* matrix);


template<typename T>
pca_result<T>::pca_result()
: eigenvectors{}, eigenvalues{}, energy{} {}


template<typename T>
armapca::pca_result<T> pca(const arma::Mat<T>& data,
                           bool compute_eigenvectors) {
  EXPECTS(data.n_rows >= 2);
  EXPECTS(data.n_cols >= 1);
  const auto n_vars = data.n_cols;
  armapca::pca_result<T> result;
  if (compute_eigenvectors)
    result.eigenvectors.set_size(n_vars, n_vars);
  result.eigenvalues.set_size(n_vars);

  const auto cov_mat = armapca::covariance_matrix(data);

  if (compute_eigenvectors) {
    arma::eig_sym(result.eigenvalues, result.eigenvectors, cov_mat);
  } else {
    arma::eig_sym(result.eigenvalues, cov_mat);
  }

  const arma::uvec indices = arma::sort_index(result.eigenvalues, 1);

  for (std::size_t i=0; i < n_vars; ++i) {
    result.eigenvalues(i) = result.eigenvalues(indices(i));
  }

  if (compute_eigenvectors) {
    for (std::size_t i=0; i < n_vars; ++i) {
      result.eigenvectors.col(i) = result.eigenvectors.col(indices(i));
    }
    armapca::enforce_positive_sign_by_column(&result.eigenvectors);
  }

  result.energy = arma::sum(result.eigenvalues);
  result.eigenvalues *= T {1} / result.energy;

  return result;
}
template armapca::pca_result<float> pca(const arma::Mat<float>&, bool);
template armapca::pca_result<double> pca(const arma::Mat<double>&, bool);


template<typename T>
std::vector<armapca::pca_result<T>> pca_bootstrap(const arma::Mat<T>& data,
                                                  bool compute_eigenvectors,
                                                  std::size_t n_bootstraps,
                                                  std::size_t random_seed) {
  EXPECTS(data.n_rows >= 2);
  EXPECTS(data.n_cols >= 1);
  std::vector<armapca::pca_result<T>> result(n_bootstraps);
  std::mt19937 gen{random_seed};
  for (std::size_t i=0; i < n_bootstraps; ++i) {
    const auto shuffled = armapca::shuffled_matrix(data, &gen);
    result[i] = armapca::pca(shuffled, compute_eigenvectors);
  }
  return result;
}
template std::vector<armapca::pca_result<float>> pca_bootstrap(
    const arma::Mat<float>&, bool, std::size_t, std::size_t);
template std::vector<armapca::pca_result<double>> pca_bootstrap(
    const arma::Mat<double>&, bool, std::size_t, std::size_t);


template<typename T>
T check_eigenvectors_orthogonal(const armapca::pca_result<T>& result) {
  return std::abs(arma::det(result.eigenvectors));
}
template float check_eigenvectors_orthogonal(
    const armapca::pca_result<float>&);
template double check_eigenvectors_orthogonal(
    const armapca::pca_result<double>&);


template<typename T>
T check_projection_accurate(const arma::Mat<T>& data,
                            const armapca::pca_result<T>& result) {
  EXPECTS(data.n_rows >= 2);
  EXPECTS(data.n_cols >= 1);
  EXPECTS(result.eigenvectors.n_cols >= 1);
  const arma::Mat<T> diff = ((data * result.eigenvectors) *
      arma::trans(result.eigenvectors)) - data;
  return T {1} - arma::sum(arma::sum(arma::abs(diff) )) / diff.n_elem;
}
template float check_projection_accurate(const arma::Mat<float>&,
                                         const armapca::pca_result<float>&);
template double check_projection_accurate(const arma::Mat<double>&,
                                          const armapca::pca_result<double>&);


}  // namespace armapca
