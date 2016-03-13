#pragma once
#include <armadillo>
#include <algorithm>
#include <random>
#include <vector>


namespace armapca {


template<typename T>
struct pca_result {
  pca_result();
  arma::Mat<T> eigenvectors;
  arma::Col<T> eigenvalues;
  T energy;
};


template<typename T>
armapca::pca_result<T> pca(const arma::Mat<T>& data,
                           bool compute_eigenvectors = true);


template<typename T>
std::vector<armapca::pca_result<T>> pca_bootstrap(const arma::Mat<T>& data,
                                              bool compute_eigenvectors = true,
                                              std::size_t n_bootstraps = 100,
                                              std::size_t random_seed = 1);


template<typename T>
T check_eigenvectors_orthogonal(const armapca::pca_result<T>& result);


template<typename T>
T check_projection_accurate(const arma::Mat<T>& data,
                            const armapca::pca_result<T>& result);


template<typename T>
void remove_mean(arma::Mat<T>* matrix);


template<typename T>
void normalize(arma::Mat<T>* matrix);


template<typename T>
arma::Mat<T> covariance_matrix(const arma::Mat<T>& matrix);


template<typename T>
arma::Mat<T> shuffled_matrix(const arma::Mat<T>& matrix, std::mt19937* gen);


template<typename T>
void enforce_positive_sign_by_column(arma::Mat<T>* matrix);


template<typename T>
arma::Row<T> column_mean(const arma::Mat<T>& matrix);


template<typename T>
arma::Row<T> column_stddev(const arma::Mat<T>& matrix);


template<typename T>
void remove_by_column(arma::Mat<T>* matrix, const arma::Row<T>& vector);


template<typename T>
void divide_by_column(arma::Mat<T>* matrix, const arma::Row<T>& vector);


}  // namespace armapca
