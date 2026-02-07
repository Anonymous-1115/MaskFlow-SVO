// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "edge_trans_norm.h"

#include <Eigen/Core>

#include "g2o/types/sba/vertex_se3_expmap.h"
#include "g2o/types/slam3d/se3_ops.h"
#include "g2o/types/slam3d/se3quat.h"

namespace g2o {

// e(T_cw) = (t^Tt - t0^Tt0)^2
void EdgeTransNorm::computeError() {
  const VertexSE3Expmap* v1 = vertexXnRaw<0>();  // T_1w
  const VertexSE3Expmap* v2 = vertexXnRaw<1>();  // T_2w

  SE3Quat T_1w(v1->estimate());
  SE3Quat T_2w(v2->estimate());
  SE3Quat T_12 = T_1w * T_2w.inverse();
  Vector3 t_12 = T_12.translation();

  double eps = 1e-6;
  error_[0] = t_12.dot(t_12) / (delta_t.dot(delta_t) + eps) - 1.0;
}

void EdgeTransNorm::linearizeOplus() {
  VertexSE3Expmap* v1 = vertexXnRaw<0>(); // T_1w
  VertexSE3Expmap* v2 = vertexXnRaw<1>(); // T_2w

  SE3Quat T_1w(v1->estimate());
  SE3Quat T_2w(v2->estimate());
  SE3Quat T_12 = T_1w * T_2w.inverse();
  Matrix3 R_12 = T_12.rotation().matrix();
  Vector3 t_12 = T_12.translation();

  double eps = 1e-6;
  Eigen::Matrix<double, 1, 3> de_dt = 2 * t_12.transpose() / (delta_t.dot(delta_t) + eps); // 1 x 3

  // 1 x 6: rotation first, translation second
  jacobianOplusXi_(0, 0) = 0.0;
  jacobianOplusXi_(0, 1) = 0.0;
  jacobianOplusXi_(0, 2) = 0.0;
  jacobianOplusXi_.block<1, 3>(0, 3) = de_dt;

  jacobianOplusXj_(0, 0) = 0.0;
  jacobianOplusXj_(0, 1) = 0.0;
  jacobianOplusXj_(0, 2) = 0.0;
  jacobianOplusXj_.block<1, 3>(0, 3) = -de_dt * R_12;

}

}  // namespace g2o
