#include <Eigen/Core>
#include "testing/TestContext.hpp"
#include "testing/Testing.hpp"
#include "testing/WaveformFixture.hpp"
#include "time/Waveform.hpp"

using namespace precice;
using namespace precice::time;

BOOST_AUTO_TEST_SUITE(TimeTests)

BOOST_AUTO_TEST_SUITE(WaveformTests)

BOOST_AUTO_TEST_CASE(testExtrapolateData)
{
  PRECICE_TEST(1_rank);

  testing::WaveformFixture fixture;

  // Test first order extrapolation
  int      extrapolationOrder = 1;
  Waveform waveform(1, extrapolationOrder);
  BOOST_TEST(fixture.numberOfSamples(waveform) == 2);
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 1);
  BOOST_TEST(fixture.numberOfData(waveform) == 1);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 0.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 0.0));

  Eigen::VectorXd value(1);
  value(0) = 1.0;
  waveform.store(value);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 0.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 1);
  waveform.moveToNextWindow(); // only applies constant extrapolation in second window
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 1.0));

  value(0) = 4.0;
  waveform.store(value);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 4.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 1.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  waveform.moveToNextWindow(); // applies first order extrapolation in third window
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 7.0)); // 7.0 = 2 * 4 - 1
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 4.0));

  value(0) = 8.0;
  waveform.store(value);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 8.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 4.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  waveform.moveToNextWindow(); // linear extrapolation
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 12.0)); // 12.0 = 2 * 8 - 4
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 8.0));

  // Test second order extrapolation
  extrapolationOrder = 2;
  Waveform waveform2(1, extrapolationOrder);
  BOOST_TEST(fixture.numberOfSamples(waveform2) == 3);
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 1);
  BOOST_TEST(fixture.numberOfData(waveform2) == 1);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 0.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 0.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 0.0));

  value(0) = 1.0;
  waveform2.store(value);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 0.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 0.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 1);
  waveform2.moveToNextWindow(); // only applies constant extrapolation in second window
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 2);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 1.0)); // 1.0 = 1.0
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 1.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 0.0));

  value(0) = 4.0;
  waveform2.store(value);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 4.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 1.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 0.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 2);
  waveform2.moveToNextWindow(); // only applies first order extrapolation in third window
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 3);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 7.0)); // 7.0 = 2 * 4 - 1
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 4.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 1.0));

  value(0) = 8.0;
  waveform2.store(value);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 8.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 4.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 1.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 3);
  waveform2.moveToNextWindow(); // applies second order extrapolation in fourth window
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 3);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 12.5)); // 12.5 = 2.5 * 8 - 2 * 4 + 0.5 * 1
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 8.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 4.0));

  value(0) = 16.0;
  waveform2.store(value);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 16.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 8.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 4.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 3);
  waveform2.moveToNextWindow(); // applies second order extrapolation in fifth window
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 3);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 26.0)); // 24.5 = 2.5 * 16.0 - 2 * 8 + 0.5 * 4
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 16.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 8.0));
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
