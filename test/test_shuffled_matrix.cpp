#include <libunittest/all.hpp>
#include "../src/lib/armapca/pca.h"


COLLECTION(test_shuffled_matrix) {

TEST_TPL(happy_path) {
  typedef Type1 T;
  arma::Mat<T> data(4, 3);
  data.row(0) = arma::Row<T>{1, 3, 4};
  data.row(1) = arma::Row<T>{6, 7, 8};
  data.row(2) = arma::Row<T>{9, 10, 11};
  data.row(3) = arma::Row<T>{12, 13, 14};
  std::mt19937 gen(1);
  const auto result = armapca::shuffled_matrix(data, &gen);
  arma::Mat<T> expected(4, 3);
  expected.row(0) = arma::Row<T>{6, 10, 14};
  expected.row(1) = arma::Row<T>{1, 3, 11};
  expected.row(2) = arma::Row<T>{12, 7, 8};
  expected.row(3) = arma::Row<T>{9, 13, 4};
  ASSERT_EQUAL_CONTAINERS(expected, result);
}
REGISTER(happy_path<float>)
REGISTER(happy_path<double>)

TEST_TPL(matrix_empty) {
  typedef Type1 T;
  const arma::Mat<T> data = {};
  std::mt19937 gen(1);
  const auto result = armapca::shuffled_matrix(data, &gen);
  const arma::Mat<T> expected = {};
  ASSERT_EQUAL_CONTAINERS(expected, result);
}
REGISTER(matrix_empty<float>)
REGISTER(matrix_empty<double>)

}
