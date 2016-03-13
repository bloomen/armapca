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


}  // namespace armapca
