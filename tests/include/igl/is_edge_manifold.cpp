#include <test_common.h>
#include <igl/is_edge_manifold.h>

// class is_edge_manifold : public ::testing::TestWithParam<std::string> {};

// TEST_P(is_edge_manifold, positive)
// {
//   Eigen::MatrixXd V;
//   Eigen::MatrixXi F;
//   test_common::load_mesh(GetParam(), V, F);
//   REQUIRE ( igl::is_edge_manifold(F) );
// }

TEST_CASE("is_edge_manifold: negative", "[igl]")
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  // Known non-manifold mesh
  test_common::load_mesh("truck.obj", V, F);
  REQUIRE (! igl::is_edge_manifold(F) );
}

// INSTANTIATE_TEST_CASE_P
// (
//  manifold_meshes,
//  is_edge_manifold,
//  ::testing::ValuesIn(test_common::manifold_meshes()),
//  test_common::string_test_name
// );
