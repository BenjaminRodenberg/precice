#pragma once

#include <Eigen/Core>
#include <vector>
#include "cplscheme/CouplingScheme.hpp"
#include "mesh/SharedPointer.hpp"
#include "time/Storage.hpp"
#include "utils/assertion.hpp"

namespace precice {
namespace cplscheme {

class CouplingData {
public:
  CouplingData(
      mesh::PtrData data,
      mesh::PtrMesh mesh,
      bool          requiresInitialization,
      int           extrapolationOrder = CouplingScheme::UNDEFINED_EXTRAPOLATION_ORDER);

  int getDimensions() const;

  int getSize() const;

  /// Returns a reference to the data values.
  Eigen::VectorXd &values();

  /// Returns a const reference to the data values.
  const Eigen::VectorXd &values() const;

  /// Returns a reference to the gradient data values.
  Eigen::MatrixXd &gradientValues();

  /// Returns a const reference to the gradient data values.
  const Eigen::MatrixXd &gradientValues() const;

  /// Returns if the data contains gradient data
  bool hasGradient() const;

  /// Returns the dimensions of the current mesh (2D or 3D)
  int meshDimensions() const;

  /// store _data->values() in read-only variable _previousIteration for convergence checks etc.
  void storeIteration();

  /// returns data value from previous iteration
  const Eigen::VectorXd previousIteration() const;

  /// returns gradient data from previous iteration
  const Eigen::MatrixXd &previousIterationGradients() const;

  /// returns size of previous iteration
  int getPreviousIterationSize() const;

  /// get ID of this CouplingData's mesh. See Mesh::getID().
  int getMeshID();

  /// get ID of this CouplingData's data. See Data::getID().
  int getDataID();

  /// get name of this CouplingData's data. See Data::getName().
  std::string getDataName();

  /// get vertex offsets of this CouplingData's mesh. See Mesh::getVertexOffsets().
  std::vector<int> getVertexOffsets();

  ///  True, if the data values of this CouplingData require to be initialized by this participant.
  const bool requiresInitialization;

  /// clears _timeStepsStorage. Called after data was written or before data is received.
  void clearTimeStepsStorage();

  /// moves _timeStepsStorage. Called after converged data was received.
  void moveTimeStepsStorage();

  /// stores data at key relativeDt in _timeStepsStorage for later use.
  void storeValuesAtTime(double relativeDt, Eigen::VectorXd data, bool mustOverwriteExisting = false);

  /// stores data for a given key into _data. Assumes that this data exists under the key.
  Eigen::VectorXd getValuesAtTime(double relativeDt);

private:
  /**
   * @brief Default constructor, not to be used!
   *
   * Necessary when compiler creates template code for std::map::operator[].
   */
  CouplingData()
      : requiresInitialization(false)
  {
    PRECICE_ASSERT(false);
  }

  /// Data values of previous iteration.
  Eigen::VectorXd _previousIteration;

  /// Gradient data of previous iteration.
  /// Lazy allocation: only used in case the corresponding data has gradients
  Eigen::MatrixXd _previousIterationGradients;

  /// Data associated with this CouplingData
  mesh::PtrData _data;

  /// Mesh associated with this CouplingData
  mesh::PtrMesh _mesh;

  /// Stores time steps in the current time window
  time::Storage _timeStepsStorage;
};

} // namespace cplscheme
} // namespace precice
