#ifndef PRECICE_NO_MPI

#include "math/differences.hpp"
#include "testing/Testing.hpp"

#include <precice/SolverInterface.hpp>
#include <vector>

using namespace precice;

BOOST_AUTO_TEST_SUITE(Integration)
BOOST_AUTO_TEST_SUITE(Serial)
BOOST_AUTO_TEST_SUITE(Time)
BOOST_AUTO_TEST_SUITE(Implicit)
BOOST_AUTO_TEST_SUITE(ParallelCoupling)

int    solverOneNSubsteps = 4;
int    solverTwoNSubsteps = 3;
double timeWindowSize     = 2.0;

double matchTimeFromOtherSolver(double thisTime, int windowCounter, int otherSubsteps, double *otherTimeGrid)
{
  double windowStartTime = (windowCounter - 1) * timeWindowSize;
  double returnedTime    = windowStartTime;

  if (math::equals(windowStartTime, thisTime)) { // we are at the beginning of the window
    return returnedTime;
  }

  for (int i = 0; i < otherSubsteps; i++) { // step through all times on the grid of the other solver
    double relativeDt = otherTimeGrid[i] * timeWindowSize;
    returnedTime      = windowStartTime + relativeDt;                  // point in time on grid of other solver
    if (math::greaterEquals(windowStartTime + relativeDt, thisTime)) { // thisTime lies between windowStartTime+otherTimeGrid[i-1] and windowStartTime+otherTimeGrid[i]
      return returnedTime;                                             // return windowStartTime+otherTimeGrid[i]
    }
  }
  BOOST_TEST(false); // unreachable
  return -1;
}

// helper to map a time on the time grid of solver two to the corresponding time on the time grid of solver one. Helps to determine the expected value in the constant interpolation
double solverOneTime(double time, int windowCounter)
{
  double relativeDts[solverOneNSubsteps] = {5.0 / 16.0, 10.0 / 16.0, 15.0 / 16.0, 16.0 / 16.0};
  return matchTimeFromOtherSolver(time, windowCounter, solverOneNSubsteps, relativeDts);
}

// helper to map a time on the time grid of solver one to the corresponding time on the time grid of solver two. Helps to determine the expected value in the constant interpolation
double solverTwoTime(double time, int windowCounter)
{
  double relativeDts[solverTwoNSubsteps] = {4.0 / 9.0, 8.0 / 9.0, 9.0 / 9.0};
  return matchTimeFromOtherSolver(time, windowCounter, solverTwoNSubsteps, relativeDts);
}

/**
 * @brief Test to run a simple coupling with zeroth order waveform subcycling. Uses different time step sizes for both solvers.
 */
BOOST_AUTO_TEST_CASE(ReadWriteScalarDataWithWaveformSubcyclingDifferentDts)
{
  PRECICE_TEST("SolverOne"_on(1_rank), "SolverTwo"_on(1_rank));

  SolverInterface precice(context.name, context.config(), 0, 1);

  MeshID meshID;
  DataID writeDataID;
  DataID readDataID;

  typedef double (*DataFunction)(double);

  DataFunction dataOneFunction = [](double t) -> double {
    return (double) (2 + t);
  };
  DataFunction dataTwoFunction = [](double t) -> double {
    return (double) (10 + t);
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

  double   writeData, readData;
  VertexID vertexID = precice.setMeshVertex(meshID, Eigen::Vector3d(0.0, 0.0, 0.0).data());

  int nSubsteps; // perform subcycling on solvers. nSubsteps steps happen in each window.
  if (context.isNamed("SolverOne")) {
    nSubsteps = solverOneNSubsteps;
  } else {
    nSubsteps = solverTwoNSubsteps;
  }
  int    nWindows = 5; // perform 5 windows.
  int    window   = 0; // counter for current window
  int    timestep = 0;
  double time     = 0;

  if (precice.isActionRequired(precice::constants::actionWriteInitialData())) {
    writeData = writeFunction(time);
    precice.writeScalarData(writeDataID, vertexID, writeData);
    precice.markActionFulfilled(precice::constants::actionWriteInitialData());
  }

  double maxDt    = precice.initialize();
  double windowDt = maxDt;
  BOOST_TEST(windowDt == timeWindowSize);
  int    timestepCheckpoint;
  double dt = windowDt / nSubsteps;       // Timestep length desired by solver. E.g. 4 steps  with size 1/4, 3 steps with size 1/3
  dt += windowDt / nSubsteps / nSubsteps; // increase timestep such that we get a non-matching subcycling. E.g. 3 steps with size 5/16 and 1 steps with size 1/16; 2 steps with size 4/9 and 1 step with size 1/9
  double currentDt = dt;                  // Timestep length used by solver
  double timeCheckpoint;
  int    iterations;

  while (precice.isCouplingOngoing()) {
    if (precice.isActionRequired(precice::constants::actionWriteIterationCheckpoint())) {
      timeCheckpoint     = time;
      timestepCheckpoint = timestep;
      iterations         = 0;
      window++;
      precice.markActionFulfilled(precice::constants::actionWriteIterationCheckpoint());
    }
    precice.readScalarData(readDataID, vertexID, currentDt, readData);

    if (iterations == 0) { // in the first iteration of each window, use data from previous window.
      BOOST_TEST(readData == readFunction(timeCheckpoint));
    } else { // in the following iterations, use data at the end of window.
      double readTime;
      if (context.isNamed("SolverOne")) {
        readTime = solverTwoTime(time + currentDt, window);
      } else {
        readTime = solverOneTime(time + currentDt, window);
      }
      BOOST_TEST(readData == readFunction(readTime));
    }

    precice.readScalarData(readDataID, vertexID, currentDt / 2, readData);

    if (iterations == 0) { // in the first iteration of each window, use data from previous window.
      BOOST_TEST(readData == readFunction(timeCheckpoint));
    } else { // in the following iterations, use data at the end of window.
      double readTime;
      if (context.isNamed("SolverOne")) {
        readTime = solverTwoTime(time + currentDt / 2, window);
      } else {
        readTime = solverOneTime(time + currentDt / 2, window);
      }
      BOOST_TEST(readData == readFunction(readTime));
    }

    precice.readScalarData(readDataID, vertexID, 0, readData);

    if (iterations == 0) { // in the first iteration of each window, use data from previous window.
      BOOST_TEST(readData == readFunction(timeCheckpoint));
    } else { // in the following iterations, use data at the end of window.
      double readTime;
      if (context.isNamed("SolverOne")) {
        readTime = solverTwoTime(time, window);
      } else {
        readTime = solverOneTime(time, window);
      }
      BOOST_TEST(readData == readFunction(readTime));
    }

    // solve usually goes here. Dummy solve: Just sampling the writeFunction.
    time += currentDt;
    timestep++;
    writeData = writeFunction(time);
    precice.writeScalarData(writeDataID, vertexID, writeData);
    maxDt = precice.advance(currentDt);
    if (precice.isActionRequired(precice::constants::actionReadIterationCheckpoint())) {
      time     = timeCheckpoint;
      timestep = timestepCheckpoint;
      iterations++;
      precice.markActionFulfilled(precice::constants::actionReadIterationCheckpoint());
    }
    currentDt = dt > maxDt ? maxDt : dt;
  }

  precice.finalize();
  BOOST_TEST(timestep == nWindows * nSubsteps);
}

BOOST_AUTO_TEST_SUITE_END() // Integration
BOOST_AUTO_TEST_SUITE_END() // Serial
BOOST_AUTO_TEST_SUITE_END() // Time
BOOST_AUTO_TEST_SUITE_END() // Explicit
BOOST_AUTO_TEST_SUITE_END() // SerialCoupling

#endif // PRECICE_NO_MPI
