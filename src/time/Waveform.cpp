#include "time/Waveform.hpp"
#include <algorithm>
#include <eigen3/unsupported/Eigen/Splines>
#include "cplscheme/CouplingScheme.hpp"
#include "logging/LogMacros.hpp"
#include "math/differences.hpp"
#include "time/Time.hpp"
#include "utils/EigenHelperFunctions.hpp"

namespace precice {
namespace time {

Waveform::Waveform(
    const int interpolationOrder)
    : _interpolationOrder(interpolationOrder)
{
  PRECICE_ASSERT(Time::MIN_INTERPOLATION_ORDER <= _interpolationOrder && _interpolationOrder <= Time::MAX_INTERPOLATION_ORDER);
}

int Waveform::getInterpolationOrder() const
{
  return _interpolationOrder;
}

void Waveform::initialize(const Eigen::VectorXd &values)
{
  int storageSize;
  PRECICE_ASSERT(_timeStepsStorage.nTimes() == 0);
  _timeStepsStorage.initialize(values);
  PRECICE_ASSERT(_interpolationOrder >= Time::MIN_INTERPOLATION_ORDER);
}

void Waveform::store(const Eigen::VectorXd &values, double normalizedDt)
{
  PRECICE_ASSERT(_timeStepsStorage.nTimes() > 0);
  // dt has to be in interval (0.0, 1.0]
  PRECICE_ASSERT(normalizedDt > 0.0); // cannot override value at beginning of window. It is locked!
  PRECICE_ASSERT(normalizedDt <= 1.0);

  if (math::equals(_timeStepsStorage.maxTime(), 1.0)) { // reached end of window and trying to write new data from next window. Clearing window first.
    bool keepZero = true;
    _timeStepsStorage.clear(keepZero);
  } else { // did not reach end of window yet, so dt has to strictly increase
    PRECICE_ASSERT(normalizedDt > _timeStepsStorage.maxTime(), normalizedDt, _timeStepsStorage.maxTime());
  }
  PRECICE_ASSERT(values.size() == _timeStepsStorage.nDofs());
  _timeStepsStorage.setValueAtTime(normalizedDt, values);
}

// helper function to compute x(t) from given data (x0,t0), (x1,t1), ..., (xn,tn) via B-spline interpolation (implemented using Eigen).
Eigen::VectorXd bSplineInterpolationAt(double t, Eigen::VectorXd ts, Eigen::MatrixXd xs, int splineDegree)
{
  // organize data in columns. Each column represents one sample in time.
  PRECICE_ASSERT(xs.cols() == ts.size());
  const int ndofs = xs.rows(); // number of dofs. Each dof needs it's own interpolant.

  Eigen::VectorXd interpolated(ndofs);

  const int splineDimension = 1;

  for (int i = 0; i < ndofs; i++) {
    auto spline     = Eigen::SplineFitting<Eigen::Spline<double, splineDimension>>::Interpolate(xs.row(i), splineDegree, ts);
    interpolated[i] = spline(t)[0]; // get component of spline associated with xs.row(i)
  }

  return interpolated;
}

Eigen::VectorXd Waveform::sample(double normalizedDt)
{
  PRECICE_ASSERT(_timeStepsStorage.nTimes() > 0);
  PRECICE_ASSERT(normalizedDt >= 0, "Sampling outside of valid range!");
  PRECICE_ASSERT(normalizedDt <= 1, "Sampling outside of valid range!");

  const int usedOrder = computeUsedOrder(_interpolationOrder, _timeStepsStorage.nTimes());

  PRECICE_ASSERT(math::equals(this->_timeStepsStorage.maxTime(), 1.0), this->_timeStepsStorage.maxTime()); // sampling is only allowed, if a window is complete.

  // @TODO: Improve efficiency: Check whether key = normalizedDt is in _timeStepsStorage. If yes, just get value and return. No need for interpolation.

  if (_interpolationOrder == 0) {
    // @TODO: Remove constant interpolation in preCICE v3.0? Usecase is unclear and does not generalize well with BSpline interpolation. It's also not 100% clear what to do at the jump.
    // constant interpolation = just use sample at the end of the window: x(dt) = x^t
    // At beginning of window use result from last window x(0) = x^(t-1)
    auto closestAfter = _timeStepsStorage.getClosestTimeAfter(normalizedDt);
    return this->_timeStepsStorage.getValueAtTime(closestAfter);
  }

  PRECICE_ASSERT(usedOrder >= 1);

  auto timesAscending = _timeStepsStorage.getTimes();
  auto nTimes         = _timeStepsStorage.nTimes();
  auto nDofs          = _timeStepsStorage.nDofs();
  PRECICE_ASSERT(math::equals(timesAscending[0], 0.0));
  PRECICE_ASSERT(math::equals(timesAscending[nTimes - 1], 1.0));
  Eigen::MatrixXd dataAscending(nDofs, nTimes);
  int             i = 0;
  for (int i = 0; i < nTimes; i++) {
    dataAscending.col(i) = this->_timeStepsStorage.getValueAtTime(timesAscending[i]);
  }
  return bSplineInterpolationAt(normalizedDt, timesAscending, dataAscending, usedOrder);
}

void Waveform::moveToNextWindow()
{
  _timeStepsStorage.move();
}

int Waveform::computeUsedOrder(int requestedOrder, int numberOfAvailableSamples)
{
  int usedOrder = -1;
  PRECICE_ASSERT(requestedOrder <= 3);
  if (requestedOrder == 0 || numberOfAvailableSamples < 2) {
    usedOrder = 0;
  } else if (requestedOrder == 1 || numberOfAvailableSamples < 3) {
    usedOrder = 1;
  } else if (requestedOrder == 2 || numberOfAvailableSamples < 4) {
    usedOrder = 2;
  } else if (requestedOrder == 3 || numberOfAvailableSamples < 5) {
    usedOrder = 3;
  } else {
    PRECICE_ASSERT(false); // not supported
  }
  return usedOrder;
}

} // namespace time
} // namespace precice
