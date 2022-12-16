#ifndef PRECICE_NO_MPI

#include "testing/Testing.hpp"

#include <precice/SolverInterface.hpp>
#include <vector>

using namespace precice;

BOOST_AUTO_TEST_SUITE(Integration)
BOOST_AUTO_TEST_SUITE(Serial)
BOOST_AUTO_TEST_SUITE(Time)
BOOST_AUTO_TEST_SUITE(Implicit)
BOOST_AUTO_TEST_SUITE(MultiCoupling)

int    solverOneNSubsteps   = 4;
int    solverTwoNSubsteps   = 3;
int    solverThreeNSubsteps = 2;
double timeWindowSize       = 2.0;

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

// helper to map a time to the corresponding time on the time grid of solver one. Helps to determine the expected value in the constant interpolation
double solverOneTime(double time, int windowCounter)
{
  BOOST_TEST(solverOneNSubsteps == 4);
  double relativeDts[4] = {0.5, 1.0, 1.5, 2.0};
  return matchTimeFromOtherSolver(time, windowCounter, solverOneNSubsteps, relativeDts);
}

// helper to map a time to the corresponding time on the time grid of solver two. Helps to determine the expected value in the constant interpolation
double solverTwoTime(double time, int windowCounter)
{
  BOOST_TEST(solverTwoNSubsteps == 3);
  double relativeDts[3] = {2.0 / 3, 4.0 / 3, 2.0};
  return matchTimeFromOtherSolver(time, windowCounter, solverTwoNSubsteps, relativeDts);
}

// helper to map a time to the corresponding time on the time grid of solver three. Helps to determine the expected value in the constant interpolation
double solverThreeTime(double time, int windowCounter)
{
  BOOST_TEST(solverThreeNSubsteps == 2);
  double relativeDts[2] = {1.0, 2.0};
  return matchTimeFromOtherSolver(time, windowCounter, solverThreeNSubsteps, relativeDts);
}

/**
 * @brief Test to run a multi coupling with zeroth order waveform subcycling. Uses different time step sizes for each solver.
 */
BOOST_AUTO_TEST_CASE(ReadWriteScalarDataWithWaveformSubcyclingDifferentDts)
{
  PRECICE_TEST("SolverOne"_on(1_rank), "SolverTwo"_on(1_rank), "SolverThree"_on(1_rank));

  SolverInterface precice(context.name, context.config(), 0, 1);

  MeshID meshID;
  DataID writeDataID;

  typedef double (*DataFunction)(double);
  std::vector<std::pair<DataID, DataFunction>> readDataPairs;

  DataFunction dataOneFunction = [](double t) -> double {
    return (double) (2 + t);
  };
  DataFunction dataTwoFunction = [](double t) -> double {
    return (double) (10 + t);
  };
  DataFunction dataThreeFunction = [](double t) -> double {
    return (double) (300 + t);
  };
  DataFunction writeFunction;

  int nSubsteps; // let three solvers use different time step sizes

  if (context.isNamed("SolverOne")) {
    meshID         = precice.getMeshID("MeshOne");
    writeDataID    = precice.getDataID("DataOne", meshID);
    writeFunction  = dataOneFunction;
    auto dataTwoId = precice.getDataID("DataTwo", meshID);
    readDataPairs.push_back(std::make_pair(dataTwoId, dataTwoFunction));
    auto dataThreeId = precice.getDataID("DataThree", meshID);
    readDataPairs.push_back(std::make_pair(dataThreeId, dataThreeFunction));
    nSubsteps = solverOneNSubsteps;
  } else if (context.isNamed("SolverTwo")) {
    meshID         = precice.getMeshID("MeshTwo");
    writeDataID    = precice.getDataID("DataTwo", meshID);
    writeFunction  = dataTwoFunction;
    auto dataOneId = precice.getDataID("DataOne", meshID);
    readDataPairs.push_back(std::make_pair(dataOneId, dataOneFunction));
    nSubsteps = solverTwoNSubsteps;
  } else {
    BOOST_TEST(context.isNamed("SolverThree"));
    meshID         = precice.getMeshID("MeshThree");
    writeDataID    = precice.getDataID("DataThree", meshID);
    writeFunction  = dataThreeFunction;
    auto dataOneId = precice.getDataID("DataOne", meshID);
    readDataPairs.push_back(std::make_pair(dataOneId, dataOneFunction));
    nSubsteps = solverThreeNSubsteps;
  }

  double   writeData, readData;
  VertexID vertexID = precice.setMeshVertex(meshID, Eigen::Vector3d(0.0, 0.0, 0.0).data());

  int    nWindows        = 5; // perform 5 windows.
  int    window          = 0; // counter for current window
  int    timestep        = 0;
  int    timewindow      = 0;
  double windowStartTime = 0;
  int    windowStartStep = 0;
  int    nSamples        = 4;
  int    iterations      = 0;
  double time            = 0;

  if (precice.requiresInitialData()) {
    writeData = writeFunction(time);
    precice.writeScalarData(writeDataID, vertexID, writeData);
  }

  double maxDt     = precice.initialize();
  double windowDt  = maxDt;
  double dt        = windowDt / nSubsteps; // Timestep length desired by solver. E.g. 2 steps with size 1/2
  double currentDt = dt;                   // Timestep length used by solver

  while (precice.isCouplingOngoing()) {
    if (precice.requiresWritingCheckpoint()) {
      windowStartTime = time;
      windowStartStep = timestep;
      iterations      = 0;
      window++;
    }

    for (auto readDataPair : readDataPairs) {
      auto readDataID   = readDataPair.first;
      auto readFunction = readDataPair.second;

      precice.readScalarData(readDataID, vertexID, currentDt, readData);

      if (iterations == 0) { // in the first iteration of each window, use data from previous window.
        BOOST_TEST(readData == readFunction(windowStartTime));
      } else { // in the following iterations, use data at the end of window.
        double readTime;
        if (readDataID == precice.getDataID("DataOne", meshID)) {
          readTime = solverOneTime(time + currentDt, window);
        } else if (precice.getDataID("DataTwo", meshID)) {
          readTime = solverTwoTime(time + currentDt, window);
        } else if (precice.getDataID("DataThree", meshID)) {
          readTime = solverThreeTime(time + currentDt, window);
        } else {
          BOOST_TEST(false);
        }
        BOOST_TEST(readData == readFunction(readTime));
      }

      precice.readScalarData(readDataID, vertexID, currentDt / 2, readData);

      if (iterations == 0) { // in the first iteration of each window, use data from previous window.
        BOOST_TEST(readData == readFunction(windowStartTime));
      } else { // in the following iterations, use data at the end of window.
        double readTime;
        if (readDataID == precice.getDataID("DataOne", meshID)) {
          readTime = solverOneTime(time + currentDt, window);
        } else if (precice.getDataID("DataTwo", meshID)) {
          readTime = solverTwoTime(time + currentDt, window);
        } else if (precice.getDataID("DataThree", meshID)) {
          readTime = solverThreeTime(time + currentDt, window);
        } else {
          BOOST_TEST(false);
        }
        BOOST_TEST(readData == readFunction(readTime));
      }

      precice.readScalarData(readDataID, vertexID, 0, readData);

      if (iterations == 0) { // in the first iteration of each window, use data from previous window.
        BOOST_TEST(readData == readFunction(windowStartTime));
      } else { // in the following iterations, use data at the end of window.
        double readTime;
        if (readDataID == precice.getDataID("DataOne", meshID)) {
          readTime = solverOneTime(time + currentDt, window);
        } else if (precice.getDataID("DataTwo", meshID)) {
          readTime = solverTwoTime(time + currentDt, window);
        } else if (precice.getDataID("DataThree", meshID)) {
          readTime = solverThreeTime(time + currentDt, window);
        } else {
          BOOST_TEST(false);
        }
        BOOST_TEST(readData == readFunction(readTime));
      }
    }

    // solve usually goes here. Dummy solve: Just sampling the writeFunction.
    time += currentDt;
    writeData = writeFunction(time);
    precice.writeScalarData(writeDataID, vertexID, writeData);
    maxDt     = precice.advance(currentDt);
    currentDt = dt > maxDt ? maxDt : dt;
    BOOST_CHECK(currentDt == windowDt / nSubsteps); // no subcycling.
    timestep++;
    if (precice.requiresReadingCheckpoint()) { // at end of window and we have to repeat it.
      iterations++;
      timestep = windowStartStep;
      time     = windowStartTime;
    }
    if (precice.isTimeWindowComplete()) {
      timewindow++;
      iterations = 0;
    }
  }

  precice.finalize();
  BOOST_TEST(timestep == nWindows * nSubsteps);
}

BOOST_AUTO_TEST_SUITE_END() // Integration
BOOST_AUTO_TEST_SUITE_END() // Serial
BOOST_AUTO_TEST_SUITE_END() // Time
BOOST_AUTO_TEST_SUITE_END() // Implicit
BOOST_AUTO_TEST_SUITE_END() // MultiCoupling

#endif // PRECICE_NO_MPI
