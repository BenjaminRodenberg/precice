#include "math/Bspline.hpp"
#include <Eigen/Sparse>
#include "math/differences.hpp"
#include "utils/assertion.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cstdlib>
#include <unsupported/Eigen/Splines>
#include "math/differences.hpp"
#include "utils/assertion.hpp"

namespace precice::math {

Bspline::Bspline(Eigen::VectorXd ts, const Eigen::MatrixXd &xs, int splineDegree)
{

  PRECICE_ASSERT(ts.size() >= 2, "Interpolation requires at least 2 samples");
  PRECICE_ASSERT(std::is_sorted(ts.begin(), ts.end()), "Timestamps must be sorted");

  // organize data in columns. Each column represents one sample in time.
  PRECICE_ASSERT(xs.cols() == ts.size());
  _ndofs            = xs.rows(); // number of dofs. Each dof needs its own interpolant.
  _tsMin            = ts(0);
  _tsMax            = ts(ts.size() - 1);
  auto relativeTime = [tsMin = _tsMin, tsMax = _tsMax](double t) -> double { return (t - tsMin) / (tsMax - tsMin); };
  ts                = ts.unaryExpr(relativeTime);

  //The code for computing the knots and the control points is copied from Eigens bspline interpolation with minor modifications, https://gitlab.com/libeigen/eigen/-/blob/master/unsupported/Eigen/src/Splines/SplineFitting.h

  Eigen::KnotAveraging(ts, splineDegree, _knots);
  Eigen::DenseIndex           n = xs.cols();
  Eigen::SparseMatrix<double> A(n, n);

  std::vector<Eigen::Triplet<double>> matrixEntries;
  for (Eigen::DenseIndex i = 1; i < n - 1; ++i) {
    //Attempt at hack... the spline dimension is not used here explicitly
    const Eigen::DenseIndex span      = Eigen::Spline<double, 1>::Span(ts[i], splineDegree, _knots);
    auto                    basisFunc = Eigen::Spline<double, 1>::BasisFunctions(ts[i], splineDegree, _knots);

    for (Eigen::DenseIndex j = 0; j < splineDegree + 1; ++j) {
      matrixEntries.push_back(Eigen::Triplet<double>(i, span - splineDegree + j, basisFunc(j)));
    }
  }

  matrixEntries.push_back(Eigen::Triplet<double>(0, 0, 1.0));
  matrixEntries.push_back(Eigen::Triplet<double>(n - 1, n - 1, 1.0));
  A.setFromTriplets(matrixEntries.begin(), matrixEntries.end());
  A.makeCompressed();

  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
  qr.analyzePattern(A);
  qr.factorize(A);

  Eigen::MatrixXd controls = Eigen::MatrixXd::Zero(n, _ndofs);
  for (int i = 0; i < _ndofs; i++) {
    controls.col(i) = qr.solve(xs.row(i).transpose());
  }
  _ctrls = std::move(controls);
}

Eigen::VectorXd Bspline::interpolateAt(double t) const
{
  // transform t to the relative interval [0; 1]
  const double tRelative = std::clamp((t - _tsMin) / (_tsMax - _tsMin), 0.0, 1.0);

  Eigen::VectorXd interpolated(_ndofs);
  constexpr int   splineDimension = 1;

  for (int i = 0; i < _ndofs; i++) {
    interpolated[i] = Eigen::Spline<double, splineDimension>(_knots, _ctrls.col(i))(tRelative)[0];
  }

  return interpolated;
}
} // namespace precice::math
