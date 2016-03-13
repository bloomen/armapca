#include <libunittest/all.hpp>
#include "../src/lib/armapca/pca.h"


COLLECTION(test_covariance_matrix) {

TEST_TPL(happy_path) {
  typedef Type1 T;
  arma::Mat<T> data(3, 4);
  data.row(0) = arma::Row<T>{1, 3, 4, 5};
  data.row(1) = arma::Row<T>{6, 7, 8, 9};
  data.row(2) = arma::Row<T>{10, 11, 12, 13};
  const auto cov_mat = armapca::covariance_matrix(data);
  arma::Mat<T> expected(4, 4);
  expected.row(0) = arma::Row<T>{68.5, 77.5, 86, 94.5};
  expected.row(1) = arma::Row<T>{77.5, 89.5, 100, 110.5};
  expected.row(2) = arma::Row<T>{86, 100, 112, 124};
  expected.row(3) = arma::Row<T>{94.5, 110.5, 124, 137.5};
  ASSERT_EQUAL_CONTAINERS(expected, cov_mat);
}
REGISTER(happy_path<float>)
REGISTER(happy_path<double>)

TEST_TPL(matrix_too_small) {
  typedef Type1 T;
  std::size_t rows = 0;
  auto lambda = [&rows] {
      arma::Mat<T> data(rows, 5);
      armapca::covariance_matrix(data);
  };
  ASSERT_THROW(std::logic_error, lambda);
  rows = 1;
  ASSERT_THROW(std::logic_error, lambda);
}
REGISTER(matrix_too_small<float>)
REGISTER(matrix_too_small<double>)

}
