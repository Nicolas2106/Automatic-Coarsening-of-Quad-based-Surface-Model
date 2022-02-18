#ifndef IGL_PER_QUAD_FACE_NORMALS_H
#define IIGL_PER_QUAD_FACE_NORMALS_H

#include "igl/igl_inline.h"
#include <Eigen/Core>

namespace igl
{
  // Compute face normals via vertex position list, face list
  // Inputs:
  //   V  #V by 3 eigen Matrix of mesh vertex 3D positions
  //   F  #F by 4 eigen Matrix of face (quad) indices
  // Output:
  //   N  #F by 3 eigen Matrix of mesh face (quad) 3D normals
  template <typename DerivedV, typename DerivedF, typename DerivedN>
  IGL_INLINE void per_quad_face_normals(
    const Eigen::MatrixBase<DerivedV>& V,
    const Eigen::MatrixBase<DerivedF>& F,
    Eigen::PlainObjectBase<DerivedN>& N);
}

#ifndef IGL_STATIC_LIBRARY
#  include "per_quad_face_normals.cpp"
#endif

#endif