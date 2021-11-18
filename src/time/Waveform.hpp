#pragma once

#include <Eigen/Core>
#include "logging/Logger.hpp"

namespace precice {

namespace testing {
// Forward declaration to friend the boost test struct
class WaveformFixture;
} // namespace testing

namespace time {

class Waveform {
  friend class testing::WaveformFixture; // Make the fixture friend of this class
public:
  /// To be used, when the interpolation order is not defined for this Waveform.
  static const int UNDEFINED_INTERPOLATION_ORDER;

  /**
   * @brief Waveform object which stores values of current and past time windows for performing extrapolation. 
   *
   * Storage still needs to be initialized with Waveform::initialize, before the Waveform can be used.
   *
   * @param extrapolatioOrder defines the maximum extrapolation order supported by this Waveform and reserves storage correspondingly
   * @param interpolationOrder defines the maximum interpolation order supported by this Waveform and reserves storage correspondingly
   */
  Waveform(const int extrapolationOrder,
           const int interpolationOrder);

  /**
   * @brief Used to initialize _timeWindowsStorage according to required size.
   * @param valuesSize defines how many values one sample in time consists of
   */
  void initialize(const int valuesSize);

  /**
   * @brief resizes _timeWindows to store more data. Used for already created waveforms.
   * @param numberOfData defines how many pieces of data one sample in time consists of
   */
  void resizeData(int valuesSize);

  /**
   * @brief Updates entry in _timeWindows corresponding to this window with given values
   * @param values new sample for this time window
   */
  void store(const Eigen::VectorXd &values);

  /**
   * @brief Updates entry in _timeWindowsStorage corresponding to a given column ID with given values
   * @param values new sample for this time window
   * @param columnID ID of column to be updated
   */
  void storeAt(const Eigen::VectorXd values, int columnID);

  /**
   * @brief Called, when moving to the next time window. All entries in _timeWindows are shifted. The new entry is initialized as the value from the last window (= constant extrapolation)
   */
  void moveToNextWindow();

  /**
   * @brief sample Waveform. Uses interpolation with Waveform's interpolation order, if necessary
   * @param normalizedDt time where the sampling inside the window happens. 0 refers to the beginning of the window and 1 to the end.
   */
  Eigen::VectorXd sample(const double normalizedDt);

  /**
   * @brief getter for values at the current time window.
   */
  const Eigen::VectorXd getInitialGuess();

  // @todo try to make private!
  /**
   * @brief returns number of values per sample in time stored by this waveform
   */
  int valuesSize();

  // @todo try to make private!
  /**
   * @brief returns number samples in time this waveform can store
   */
  int sizeOfSampleStorage();

  // @todo try to make private!
  /**
   * @brief getter for a sample in _timeWindowsStorage
   */
  Eigen::VectorXd getSample(int sampleID);

  /// @todo try to get rid of this method
  void setExtrapolationOrder(int extrapolationOrder);

private:
  /// Set by initialize. Used for consistency checks.
  bool _storageIsInitialized = false;

  /// Stores values for several time windows.
  Eigen::MatrixXd _timeWindowsStorage;

  /**
   * Order of predictor of interface values for first participant.
   *
   * The first participant in the implicit coupling scheme has to take some
   * initial guess for the interface values computed by the second participant.
   * In order to improve this initial guess, an extrapolation from previous
   * time windows can be performed.
   *
   * The standard predictor is of order zero, i.e., simply the converged values
   * of the last time windows are taken as initial guess for the coupling iterations.
   * Currently, an order 1 predictor (linear extrapolation) and order 2 predictor
   * (see https://doi.org/10.1016/j.compstruc.2008.11.013, p.796, Algorithm line 1 )
   * is implement besides that.
   */
  int _extrapolationOrder; // @todo make const!

  /// interpolation order for this waveform
  const int _interpolationOrder;

  /// number of stored samples in _timeWindowsStorage
  int _numberOfStoredSamples;

  mutable logging::Logger _log{"time::Waveform"};

  /**
   * @brief Extrapolates values from _timeWindowsStorage using an extrapolation scheme of given order. 
   * 
   * If the order condition cannot be satisfied, since there are not enough samples available, the order is automatically reduced.
   * If order two is required, but only two samples are available, the extrapolation order is automatically reduced to one.
   */
  Eigen::VectorXd extrapolate();

  /**
   * @brief Interpolates values inside current time window using _timeWindowsStorage and an interpolation scheme of the order of this Waveform.
   *
   * @param normalizedDt time where the sampling inside the window happens. 0 refers to the beginning of the window and 1 to the end.
   */
  Eigen::VectorXd interpolate(const double normalizedDt);
};

} // namespace time
} // namespace precice
