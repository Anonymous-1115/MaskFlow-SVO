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

#include "edge_mine_onlypose.h"

#include <Eigen/Core>

#include "g2o/types/sba/vertex_se3_expmap.h"
#include "g2o/types/slam3d/se3_ops.h"
#include "g2o/types/slam3d/se3quat.h"

namespace g2o {

// e(T_cw) = p_c - T_cw * p_w
void EdgeMineOnlyPose::computeError() {
  const VertexSE3Expmap* v1 = vertexXnRaw<0>();  // T_cw
  Vector3 p_c(measurement_);
  error_ = p_c - v1->estimate().map(Xw);

  // re-define the information matrix
  const Matrix3 R = v1->estimate().rotation().matrix();
  information() = (cov_c + R * cov_w * R.transpose()).inverse();
}

bool EdgeMineOnlyPose::isDepthPositive() {
  const VertexSE3Expmap* v1 = vertexXnRaw<0>();
  return (v1->estimate().map(Xw))(2) > 0.0;
}

void EdgeMineOnlyPose::linearizeOplus() {
  VertexSE3Expmap* vi = vertexXnRaw<0>(); // T_cw
  SE3Quat T_cw(vi->estimate());
  Vector3 p_c = T_cw.map(Xw);

  double x = p_c[0];
  double y = p_c[1];
  double z = p_c[2];

  jacobianOplusXi_(0, 0) = 0.0;
  jacobianOplusXi_(0, 1) = -z;
  jacobianOplusXi_(0, 2) = y;
  jacobianOplusXi_(0, 3) = -1.0;
  jacobianOplusXi_(0, 4) = 0.0;
  jacobianOplusXi_(0, 5) = 0.0;

  jacobianOplusXi_(1, 0) = z;
  jacobianOplusXi_(1, 1) = 0.0;
  jacobianOplusXi_(1, 2) = -x;
  jacobianOplusXi_(1, 3) = 0.0;
  jacobianOplusXi_(1, 4) = -1.0;
  jacobianOplusXi_(1, 5) = 0.0;

  jacobianOplusXi_(2, 0) = -y;
  jacobianOplusXi_(2, 1) = x;
  jacobianOplusXi_(2, 2) = 0.0;
  jacobianOplusXi_(2, 3) = 0.0;
  jacobianOplusXi_(2, 4) = 0.0;
  jacobianOplusXi_(2, 5) = -1.0;
}

}  // namespace g2o
