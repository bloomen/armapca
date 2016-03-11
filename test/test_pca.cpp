#include <libunittest/all.hpp>
#include "../src/lib/armapca/pca.h"


COLLECTION(test_pca) {

TEST_TPL(happy_path) {
  typedef Type1 T;
  arma::Mat<T> data(3, 4);
  data.row(0) = arma::Row<T>{1, 2.5, 42, 7};
  data.row(1) = arma::Row<T>{3, 4.2, 90, 7};
  data.row(2) = arma::Row<T>{456, 444, 0, 7};
  const auto result = armapca::pca(data);
  ASSERT_EQUAL(4, result.eigenvectors.n_rows);
  ASSERT_EQUAL(4, result.eigenvectors.n_cols);
  ASSERT_EQUAL(4, result.eigenvalues.n_elem);
  ASSERT_GREATER(result.energy, 0);
}
REGISTER(happy_path<float>)
REGISTER(happy_path<double>)

}
