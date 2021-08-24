#pragma once

#include <Eigen/Core>
#include "logging/Logger.hpp"

namespace precice {
namespace time {

class Waveform {
public:
  /**
   * @brief Waveform object which stores data of current and past time windows for performing extrapolation.
   * @param initializedNumberOfData defines how many pieces of data one sample in time consists of
   * @param extrapolatioOrder defines the maximum extrapolation order supported by this Waveform and reserves storage correspondingly
   * @param interpolationOrder defines the maximum interpolation order supported by this Waveform and reserves storage correspondingly
   */
  Waveform(int initializedNumberOfData,
           int extrapolationOrder = 0,
           int interpolationOrder = 0);

  /**
   * @brief resizes _timeWindows to store more data. Used for already created waveforms.
   * @param numberOfData defines how many pieces of data one sample in time consists of
   */
  void resizeData(int numberOfData);

  /**
   * @brief Updates entry in _timeWindows corresponding to this window with given data
   * @param data new sample for this time window
   */
  void store(const Eigen::VectorXd &data);

  /**
   * @brief Updates entry in _timeWindows corresponding to a given column ID with given data
   * @param data new sample for this time window
   * @param columnID ID of column to be updated
   */
  void storeAt(const Eigen::VectorXd data, int columnID);

  /**
   * @brief Updates entry in _timeWindows corresponding to a given column ID and data ID with given scalar value
   * @param value scalar value at data ID for this time window
   * @param columnID ID of column to be updated
   * @param valueIndex ID of value to be updated
   */
  void storeScalarAt(const double value, int columnID, int valueIndex);

  /**
   * @brief Called, when moving to the next time window. All entries in _timeWindows are shifted. The new entry is initialized as the value from the last window (= constant extrapolation)
   * @param timeWindows number of samples that are valid and may be used for extrapolation. Usually number of past time windows.
   */
  void moveToNextWindow(int timeWindows, int order = 0);

  /**
   * @brief sample Waveform
   * @param dt time where the sampling inside the window happens. dt = 0 refers to the beginning of the window and dt = 1 to the end.
   * @param timeWindows number of past samples that are valid and may be used for interpolation. Usually number of past time windows.
   * @param order interpolation order being used.
   */
  Eigen::VectorXd sample(double dt, int timeWindows, int order = 0);

  /**
   * @brief getter for Eigen::MatrixXd containing data of current and past time windows. Each column represents a sample in time, with col(0)
   * being the current time window.
   */
  const Eigen::MatrixXd &lastTimeWindows();

  /**
   * @brief returns number of samples in time stored by this waveform
   */
  int numberOfSamples();

  /**
   * @brief returns number of data per sample in time stored by this waveform
   */
  int numberOfData();

private:
  /// Data values of time windows.
  Eigen::MatrixXd _timeWindows;

  mutable logging::Logger _log{"time::Waveform"};

  /**
   * @brief Extrapolates data _timeWindows using an extrapolation scheme of given order. 
   * 
   * If the order condition cannot be satisfied, since there are not enough samples available, the order is automatically reduced.
   * If order two is required, but only two samples are available, the extrapolation order is automatically reduced to one.
   * 
   * @param order Order of the extrapolation scheme to be used.
   * @param timeWindows number of valid samples.
   */
  Eigen::VectorXd extrapolateData(int order, int timeWindows);

  /**
   * @brief Interpolates data inside current time time window using an interpolation scheme of given order.
   *
   * @param order Order of the interpolation scheme to be used.
   * @param timeWindows number of valid samples.
   * @param dt time where the sampling inside the window happens. dt = 0 refers to the beginning of the window and dt = 1 to the end.
   */
  Eigen::VectorXd interpolateData(int order, int timeWindows, double dt);
};

} // namespace time
} // namespace precice
