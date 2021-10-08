#ifndef PRECICE_NO_MPI

#include <Eigen/Core>
#include <algorithm>
#include <deque>
#include <fstream>
#include <istream>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "action/RecorderAction.hpp"
#include "logging/LogMacros.hpp"
#include "math/constants.hpp"
#include "math/geometry.hpp"
#include "mesh/Data.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/SharedPointer.hpp"
#include "mesh/Utils.hpp"
#include "mesh/Vertex.hpp"
#include "precice/SolverInterface.hpp"
#include "precice/impl/MeshContext.hpp"
#include "precice/impl/Participant.hpp"
#include "precice/impl/SharedPointer.hpp"
#include "precice/impl/SolverInterfaceImpl.hpp"
#include "precice/types.hpp"
#include "testing/TestContext.hpp"
#include "testing/Testing.hpp"

using namespace precice;
using precice::testing::TestContext;

struct SerialTestFixture : testing::WhiteboxAccessor {

  std::string _pathToTests;

  void reset()
  {
    mesh::Data::resetDataCount();
  }

  SerialTestFixture()
  {
    _pathToTests = testing::getPathToSources() + "/precice/tests/";
    reset();
  }
};

BOOST_AUTO_TEST_SUITE(PreciceTests)
BOOST_FIXTURE_TEST_SUITE(Serial, SerialTestFixture)
BOOST_AUTO_TEST_SUITE(Waveform)
BOOST_AUTO_TEST_SUITE(Explicit)

/// Test to run a simple coupling with subcycling.
/// Ensures that each time step provides its own data, but preCICE will only exchange data at the end of the window.
BOOST_AUTO_TEST_CASE(testExplicitReadWriteScalarDataWithSubcycling)
{
  PRECICE_TEST("SolverOne"_on(1_rank), "SolverTwo"_on(1_rank));

  SolverInterface precice(context.name, _pathToTests + "explicit-scalar-data-init.xml", 0, 1); // serial coupling, SolverOne first

  MeshID meshID;
  DataID writeDataID;
  DataID readDataID;

  typedef double (*DataFunction)(double, int);

  DataFunction dataOneFunction = [](double t, int idx) -> double {
    return (double) (2 + t + idx);
  };
  DataFunction dataTwoFunction = [](double t, int idx) -> double {
    return (double) (10 + t + idx);
  };
  DataFunction writeFunction;
  DataFunction readFunction;

  if (context.isNamed("SolverOne")) {
    meshID        = precice.getMeshID("MeshOne");
    writeDataID   = precice.getDataID("DataOne", meshID);
    writeFunction = dataOneFunction;
    readDataID    = precice.getDataID("DataTwo", meshID);
    readFunction  = dataTwoFunction;
  } else {
    BOOST_TEST(context.isNamed("SolverTwo"));
    meshID        = precice.getMeshID("MeshTwo");
    writeDataID   = precice.getDataID("DataTwo", meshID);
    writeFunction = dataTwoFunction;
    readDataID    = precice.getDataID("DataOne", meshID);
    readFunction  = dataOneFunction;
  }

  int n_vertices = 1;

  std::vector<VertexID> vertexIDs(n_vertices, 0);
  std::vector<double>   writeData(n_vertices, 0);
  std::vector<double>   readData(n_vertices, 0);
  double                oldWriteData, oldReadData;

  vertexIDs[0] = precice.setMeshVertex(meshID, Eigen::Vector3d(0.0, 0.0, 0.0).data());

  int    nSubsteps     = 4; // perform subcycling on solvers. 4 steps happen in each window.
  int    nWindows      = 5; // perform 5 windows.
  double maxDt         = precice.initialize();
  double windowDt      = maxDt;
  int    timestep      = 0;
  int    timewindow    = 0;
  double dt            = windowDt / (nSubsteps - 0.5); // Timestep length desired by solver. E.g. 4 steps with size 4/7. Fourth step will be restricted to 2/7 via preCICE steering to fit into the window.
  double expectedDts[] = {4.0 / 7.0, 4.0 / 7.0, 4.0 / 7.0, 2.0 / 7.0};
  double currentDt     = dt; // Timestep length used by solver
  double time          = timestep * dt;

  if (precice.isActionRequired(precice::constants::actionWriteInitialData())) {
    for (int i = 0; i < n_vertices; i++) {
      writeData[i] = writeFunction(time, i);
      precice.writeScalarData(writeDataID, vertexIDs[i], writeData[i]);
    }
    precice.markActionFulfilled(precice::constants::actionWriteInitialData());
  }

  precice.initializeData();

  while (precice.isCouplingOngoing()) {
    double readTime;
    if (context.isNamed("SolverOne")) {
      readTime = timewindow * windowDt; // SolverOne lags one window behind SolverTwo for serial-explicit coupling.
    } else {
      readTime = (timewindow + 1) * windowDt;
    }
    BOOST_TEST(readData.size() == n_vertices);
    for (int i = 0; i < n_vertices; i++) {
      oldReadData = readData[i];
      precice.readScalarData(readDataID, vertexIDs[i], readData[i]);
      if (precice.isTimeWindowComplete() ||
          (timestep == 0)) {                      // exception: First timestep will also have different data, even though formally no time window is completed.
        BOOST_TEST((readData[i] != oldReadData)); // ensure that read data changes from one step to the next, if a new window is entered
      } else {
        BOOST_TEST((readData[i] == oldReadData)); // ensure that read data stays the same from one step to the next, if not a new window is entered
      }
      BOOST_TEST(readData[i] == readFunction(readTime, i));
    }

    // solve usually goes here. Dummy solve: Just sampling the writeFunction.
    BOOST_TEST(currentDt == expectedDts[timestep % nSubsteps]);
    time += currentDt;

    BOOST_TEST(writeData.size() == n_vertices);
    for (int i = 0; i < n_vertices; i++) {
      oldWriteData = writeData[i];
      writeData[i] = writeFunction(time, i);
      BOOST_TEST(writeData[i] != oldWriteData); // ensure that write data differs from one step to the next
      precice.writeScalarData(writeDataID, vertexIDs[i], writeData[i]);
    }
    maxDt     = precice.advance(currentDt);
    currentDt = dt > maxDt ? maxDt : dt;
    timestep++;
    if (precice.isTimeWindowComplete()) {
      timewindow++;
    }
  }

  precice.finalize();
  BOOST_TEST(timestep == nWindows * nSubsteps);
}

/// Test to run a simple coupling with sampling from the waveform.
BOOST_AUTO_TEST_CASE(testExplicitReadWriteScalarDataWithWaveformSampling)
{
  PRECICE_TEST("SolverOne"_on(1_rank), "SolverTwo"_on(1_rank));

  SolverInterface precice(context.name, _pathToTests + "explicit-scalar-data-init.xml", 0, 1);

  MeshID meshID;
  DataID writeDataID;
  DataID readDataID;

  typedef double (*DataFunction)(double, int);

  DataFunction dataOneFunction = [](double t, int idx) -> double {
    return (double) (2 + t + idx);
  };
  DataFunction dataTwoFunction = [](double t, int idx) -> double {
    return (double) (10 + t + idx);
  };
  DataFunction writeFunction;
  DataFunction readFunction;

  if (context.isNamed("SolverOne")) {
    meshID        = precice.getMeshID("MeshOne");
    writeDataID   = precice.getDataID("DataOne", meshID);
    writeFunction = dataOneFunction;
    readDataID    = precice.getDataID("DataTwo", meshID);
    readFunction  = dataTwoFunction;
  } else {
    BOOST_TEST(context.isNamed("SolverTwo"));
    meshID        = precice.getMeshID("MeshTwo");
    writeDataID   = precice.getDataID("DataTwo", meshID);
    writeFunction = dataTwoFunction;
    readDataID    = precice.getDataID("DataOne", meshID);
    readFunction  = dataOneFunction;
  }

  int nVertices = 2;

  std::vector<VertexID> vertexIDs(nVertices, 0);
  std::vector<double>   writeData(nVertices, 0);
  std::vector<double>   readData(nVertices, 0);

  vertexIDs[0] = precice.setMeshVertex(meshID, Eigen::Vector3d(0.0, 0.0, 0.0).data());
  vertexIDs[1] = precice.setMeshVertex(meshID, Eigen::Vector3d(1.0, 0.0, 0.0).data());

  int    nWindows  = 5; // perform 5 windows.
  double maxDt     = precice.initialize();
  double windowDt  = maxDt;
  int    timestep  = 0;
  double dt        = maxDt; // Timestep length desired by solver
  double currentDt = dt;    // Timestep length used by solver
  double time      = timestep * dt;

  if (precice.isActionRequired(precice::constants::actionWriteInitialData())) {
    for (int i = 0; i < nVertices; i++) {
      writeData[i] = writeFunction(time, i);
      precice.writeScalarData(writeDataID, vertexIDs[i], writeData[i]);
    }
    precice.markActionFulfilled(precice::constants::actionWriteInitialData());
  }

  precice.initializeData();

  while (precice.isCouplingOngoing()) {
    BOOST_TEST(precice.isReadDataAvailable());
    double readTime;
    if (context.isNamed("SolverOne")) {
      readTime = time; // solver one lags one window behind solver two.
    } else {
      readTime = time + windowDt;
    }
    BOOST_TEST(readData.size() == nVertices);
    for (int i = 0; i < nVertices; i++) {
      if (timestep == 0) { // @todo should also work in first time window, not 100% clear for explicit coupling.
        precice.readScalarData(readDataID, vertexIDs[i], currentDt, readData[i]);
        BOOST_TEST(readData[i] == readFunction(readTime, i));
        precice.readScalarData(readDataID, vertexIDs[i], currentDt / 4, readData[i]);
        BOOST_TEST(readData[i] == readFunction(readTime, i));
        precice.readScalarData(readDataID, vertexIDs[i], currentDt / 2, readData[i]);
        BOOST_TEST(readData[i] == readFunction(readTime, i));
        precice.readScalarData(readDataID, vertexIDs[i], currentDt * 3 / 4, readData[i]);
        BOOST_TEST(readData[i] == readFunction(readTime, i));
      } else {
        precice.readScalarData(readDataID, vertexIDs[i], currentDt, readData[i]);
        BOOST_TEST(readData[i] == readFunction(readTime, i));
        precice.readScalarData(readDataID, vertexIDs[i], currentDt / 4, readData[i]);
        BOOST_TEST(readData[i] == readFunction(readTime - currentDt * 3 / 4, i));
        precice.readScalarData(readDataID, vertexIDs[i], currentDt / 2, readData[i]);
        BOOST_TEST(readData[i] == readFunction(readTime - currentDt / 2, i));
        precice.readScalarData(readDataID, vertexIDs[i], currentDt * 3 / 4, readData[i]);
        BOOST_TEST(readData[i] == readFunction(readTime - currentDt / 4, i));
      }
    }

    // solve usually goes here. Dummy solve: Just sampling the writeFunction.
    time += currentDt;
    for (int i = 0; i < nVertices; i++) {
      writeData[i] = writeFunction(time, i);
    }

    BOOST_TEST(writeData.size() == nVertices);
    for (int i = 0; i < nVertices; i++) {
      writeData[i] = writeFunction(time, i);
      precice.writeScalarData(writeDataID, vertexIDs[i], writeData[i]);
    }
    maxDt     = precice.advance(currentDt);
    currentDt = dt > maxDt ? maxDt : dt;
    timestep++;
  }

  precice.finalize();
  BOOST_TEST(timestep == nWindows);
}

/// Test to run a simple coupling with subcycling with sampling from the waveform.
BOOST_AUTO_TEST_CASE(testExplicitReadWriteScalarDataWithWaveformSubcycling)
{
  PRECICE_TEST("SolverOne"_on(1_rank), "SolverTwo"_on(1_rank));

  SolverInterface precice(context.name, _pathToTests + "explicit-scalar-data-init.xml", 0, 1);

  MeshID meshID;
  DataID writeDataID;
  DataID readDataID;

  typedef double (*DataFunction)(double, int);

  DataFunction dataOneFunction = [](double t, int idx) -> double {
    return (double) (2 + t + idx);
  };
  DataFunction dataTwoFunction = [](double t, int idx) -> double {
    return (double) (10 + t + idx);
  };
  DataFunction writeFunction;
  DataFunction readFunction;

  if (context.isNamed("SolverOne")) {
    meshID        = precice.getMeshID("MeshOne");
    writeDataID   = precice.getDataID("DataOne", meshID);
    writeFunction = dataOneFunction;
    readDataID    = precice.getDataID("DataTwo", meshID);
    readFunction  = dataTwoFunction;
  } else {
    BOOST_TEST(context.isNamed("SolverTwo"));
    meshID        = precice.getMeshID("MeshTwo");
    writeDataID   = precice.getDataID("DataTwo", meshID);
    writeFunction = dataTwoFunction;
    readDataID    = precice.getDataID("DataOne", meshID);
    readFunction  = dataOneFunction;
  }

  int nVertices = 1;

  std::vector<VertexID> vertexIDs(nVertices, 0);
  std::vector<double>   writeData(nVertices, 0);
  std::vector<double>   readData(nVertices, 0);

  vertexIDs[0] = precice.setMeshVertex(meshID, Eigen::Vector3d(0.0, 0.0, 0.0).data());

  int    nSubsteps = 4; // perform subcycling on solvers. 4 steps happen in each window.
  int    nWindows  = 5; // perform 5 windows.
  double maxDt     = precice.initialize();
  double windowDt  = maxDt;
  int    timestep  = 0;
  double dt        = windowDt / nSubsteps; // Timestep length desired by solver. E.g. 4 steps  with size 1/4
  dt += windowDt / nSubsteps / nSubsteps;  // increase timestep such that we get a non-matching subcycling. E.g. 3 step with size 5/16 and 1 step with size 1/16.
  double currentDt = dt;                   // Timestep length used by solver
  double time      = timestep * dt;

  if (precice.isActionRequired(precice::constants::actionWriteInitialData())) {
    for (int i = 0; i < nVertices; i++) {
      writeData[i] = writeFunction(time, i);
      precice.writeScalarData(writeDataID, vertexIDs[i], writeData[i]);
    }
    precice.markActionFulfilled(precice::constants::actionWriteInitialData());
  }

  precice.initializeData();

  while (precice.isCouplingOngoing()) {
    double readTime;
    if (context.isNamed("SolverOne")) {
      readTime = time - windowDt + currentDt; // solver one lags one window behind solver two.
    } else {
      readTime = time + currentDt;
    }

    BOOST_TEST(readData.size() == nVertices);
    for (int i = 0; i < nVertices; i++) {
      precice.readScalarData(readDataID, vertexIDs[i], currentDt, readData[i]);
      if (timestep < nSubsteps) { // in the first window, we only have one sample of data. Therefore constant interpolation
        if (context.isNamed("SolverOne")) {
          BOOST_TEST(readData[i] == readFunction(0, i));
        } else {
          BOOST_TEST(readData[i] == readFunction(windowDt, i));
        }
      } else { // in the following windows we have two samples of data. Therefore linear interpolation
        BOOST_TEST(readData[i] == readFunction(readTime, i));
      }
      precice.readScalarData(readDataID, vertexIDs[i], currentDt / 2, readData[i]);
      if (timestep < nSubsteps) { // in the first window, we only have one sample of data. Therefore constant interpolation
        if (context.isNamed("SolverOne")) {
          BOOST_TEST(readData[i] == readFunction(0, i));
        } else {
          BOOST_TEST(readData[i] == readFunction(windowDt, i));
        }
      } else { // in the following windows we have two samples of data. Therefore linear interpolation
        BOOST_TEST(readData[i] == readFunction(readTime - currentDt / 2, i));
      }
    }

    // solve usually goes here. Dummy solve: Just sampling the writeFunction.
    time += currentDt;
    for (int i = 0; i < nVertices; i++) {
      writeData[i] = writeFunction(time, i);
    }

    BOOST_TEST(writeData.size() == nVertices);
    for (int i = 0; i < nVertices; i++) {
      writeData[i] = writeFunction(time, i);
      precice.writeScalarData(writeDataID, vertexIDs[i], writeData[i]);
    }
    maxDt     = precice.advance(currentDt);
    currentDt = dt > maxDt ? maxDt : dt;
    timestep++;
  }

  precice.finalize();
  BOOST_TEST(timestep == nWindows * nSubsteps);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Implicit)

/// Test to run a simple coupling with subcycling.
/// Ensures that each time step provides its own data, but preCICE will only exchange data at the end of the window.
BOOST_AUTO_TEST_CASE(testImplicitReadWriteScalarDataWithSubcycling)
{
  PRECICE_TEST("SolverOne"_on(1_rank), "SolverTwo"_on(1_rank));

  SolverInterface precice(context.name, _pathToTests + "implicit-scalar-data-init.xml", 0, 1); // serial coupling, SolverOne first

  MeshID meshID;
  DataID writeDataID;
  DataID readDataID;

  typedef double (*DataFunction)(double, int);

  DataFunction dataOneFunction = [](double t, int idx) -> double {
    return (double) (2 + t + idx);
  };
  DataFunction dataTwoFunction = [](double t, int idx) -> double {
    return (double) (10 + t + idx);
  };
  DataFunction writeFunction;
  DataFunction readFunction;

  if (context.isNamed("SolverOne")) {
    meshID        = precice.getMeshID("MeshOne");
    writeDataID   = precice.getDataID("DataOne", meshID);
    writeFunction = dataOneFunction;
    readDataID    = precice.getDataID("DataTwo", meshID);
    readFunction  = dataTwoFunction;
  } else {
    BOOST_TEST(context.isNamed("SolverTwo"));
    meshID        = precice.getMeshID("MeshTwo");
    writeDataID   = precice.getDataID("DataTwo", meshID);
    writeFunction = dataTwoFunction;
    readDataID    = precice.getDataID("DataOne", meshID);
    readFunction  = dataOneFunction;
  }

  int n_vertices = 1;

  std::vector<VertexID> vertexIDs(n_vertices, 0);
  std::vector<double>   writeData(n_vertices, 0);
  std::vector<double>   readData(n_vertices, 0);
  double                oldWriteData, oldReadData;

  vertexIDs[0] = precice.setMeshVertex(meshID, Eigen::Vector3d(0.0, 0.0, 0.0).data());

  int    nSubsteps       = 4; // perform subcycling on solvers. 4 steps happen in each window.
  int    nWindows        = 5; // perform 5 windows.
  double maxDt           = precice.initialize();
  double windowDt        = maxDt;
  int    timestep        = 0;
  int    timewindow      = 0;
  double startTime       = 0;
  double windowStartTime = 0;
  int    windowStartStep = 0;
  int    iterations      = 0;
  double dt              = windowDt / (nSubsteps - 0.5); // Timestep length desired by solver. E.g. 4 steps with size 4/7. Fourth step will be restricted to 2/7 via preCICE steering to fit into the window.
  double expectedDts[]   = {4.0 / 7.0, 4.0 / 7.0, 4.0 / 7.0, 2.0 / 7.0};
  double currentDt       = dt; // Timestep length used by solver
  double time            = timestep * dt;

  if (precice.isActionRequired(precice::constants::actionWriteInitialData())) {
    for (int i = 0; i < n_vertices; i++) {
      writeData[i] = writeFunction(time, i);
      precice.writeScalarData(writeDataID, vertexIDs[i], writeData[i]);
    }
    precice.markActionFulfilled(precice::constants::actionWriteInitialData());
  }

  precice.initializeData();

  while (precice.isCouplingOngoing()) {
    if (precice.isActionRequired(precice::constants::actionWriteIterationCheckpoint())) {
      windowStartTime = time;
      windowStartStep = timestep;
      precice.markActionFulfilled(precice::constants::actionWriteIterationCheckpoint());
    }

    BOOST_TEST(readData.size() == n_vertices);
    for (int i = 0; i < n_vertices; i++) {
      oldReadData = readData[i];
      precice.readScalarData(readDataID, vertexIDs[i], readData[i]);
      //std::cout << context.name << " at time " << time << " reads " << readData[i] << " for time window = " << timewindow << ", time step "<< timestep << ", it = " << iterations << std::endl;
      if (iterations == 0 && timestep == 0 && context.isNamed("SolverOne")) {                      // special situation: SolverOne in its very first time window, first iteration, first time step
        BOOST_TEST(readData[i] != oldReadData);                                                    // update from uninitialized to initial data.
        BOOST_TEST(readData[i] == readFunction(startTime, i));                                     // use initial data only.
      } else if (iterations == 0 && context.isNamed("SolverOne")) {                                // special situation: SolverOne gets the old data its first iteration for all time windows.
        BOOST_TEST(readData[i] == oldReadData);                                                    // ensure that read data stays the same from one step to the next, if not a new window is entered
        BOOST_TEST(readData[i] == readFunction(startTime + (timewindow) *windowDt, i));            // data at end of window was written by other solver.
      } else if (iterations == 1 && timestep == windowStartStep && context.isNamed("SolverOne")) { // special situation: SolverOne in its second iteration, first timestep of window
        BOOST_TEST(readData[i] != oldReadData);                                                    // ensure that read data stays the same from one step to the next, if not a new window is entered
        BOOST_TEST(readData[i] == readFunction(startTime + (timewindow + 1) * windowDt, i));       // data at end of window was written by other solver.
      } else if (iterations == 0 && timestep == 0 && context.isNamed("SolverTwo")) {               // special situation: SolverTwo in its very first time window, first iteration, first time step
        BOOST_TEST(readData[i] != oldReadData);                                                    // update from uninitialized to initial data.
        BOOST_TEST(readData[i] == readFunction(startTime + (timewindow + 1) * windowDt, i));       // data at end of window was written by other solver.
      } else if (iterations == 0 && timestep > 0 && context.isNamed("SolverTwo")) {                // special situation: SolverTwo in its very first time window, first iteration, following time steps
        BOOST_TEST(readData[i] == oldReadData);                                                    // ensure that read data stays the same from one step to the next, if not a new window is entered
        BOOST_TEST(readData[i] == readFunction(startTime + (timewindow + 1) * windowDt, i));       // data at end of window was written by other solver.
      } else if (not precice.isTimeWindowComplete()) {                                             // still iterating in the same window
        BOOST_TEST(readData[i] == oldReadData);                                                    // ensure that read data stays the same from one step to the next, if not a new window is entered
        BOOST_TEST(readData[i] == readFunction(startTime + (timewindow + 1) * windowDt, i));       // data at end of window was written by other solver.
      } else if (precice.isTimeWindowComplete()) {
        BOOST_TEST(readData[i] != oldReadData);                                              // ensure that read data changes from one step to the next, if a new window is entered
        BOOST_TEST(readData[i] == readFunction(startTime + (timewindow + 1) * windowDt, i)); // data at end of window was written by other solver.
      }
    }

    // solve usually goes here. Dummy solve: Just sampling the writeFunction.
    BOOST_TEST(currentDt == expectedDts[timestep % nSubsteps]);
    time += currentDt;

    BOOST_TEST(writeData.size() == n_vertices);
    for (int i = 0; i < n_vertices; i++) {
      oldWriteData = writeData[i];
      writeData[i] = writeFunction(time, i);
      std::cout << context.name << " at time " << time << " writes " << writeData[i] << " for time window = " << timewindow << ", time step " << timestep << ", it = " << iterations << std::endl;
      BOOST_TEST(writeData[i] != oldWriteData); // ensure that write data differs from one step to the next
      precice.writeScalarData(writeDataID, vertexIDs[i], writeData[i]);
    }
    maxDt     = precice.advance(currentDt);
    currentDt = dt > maxDt ? maxDt : dt;
    timestep++;
    if (precice.isActionRequired(precice::constants::actionReadIterationCheckpoint())) { // at end of window and we have to repeat it.
      iterations++;
      timestep = windowStartStep;
      time     = windowStartTime;
      precice.markActionFulfilled(precice::constants::actionReadIterationCheckpoint()); // this test does not care about checkpointing, but we have to make the action
    }
    if (precice.isTimeWindowComplete()) {
      timewindow++;
      iterations = 0;
    }
  }

  precice.finalize();
  BOOST_TEST(timestep == nWindows * nSubsteps);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif // PRECICE_NO_MPI
