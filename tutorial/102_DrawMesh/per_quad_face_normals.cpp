#include "per_quad_face_normals.h"
#include "igl/fit_plane.h"
#include <Eigen/Geometry>

template <typename DerivedV, typename DerivedF, typename DerivedN>
void igl::per_quad_face_normals(
  const Eigen::MatrixBase<DerivedV>& V,
  const Eigen::MatrixBase<DerivedF>& F,
  Eigen::PlainObjectBase<DerivedN>& N)
{
  N.resize(F.rows, 3);
  // loop over faces
  int Frows = F.rows();
#pragma omp parallel for if (Frows>10000)
  for (int i = 0; i < Frows; i++)
  {
    Eigen::RowVector3d N, C;
    Eigen::MatrixXd vV(4, 3);
    vV <<
      V.row(F(i, 0)),
      V.row(F(i, 1)),
      V.row(F(i, 2)),
      V.row(F(i, 3));
    fit_plane(vV, N, C);
  }
}