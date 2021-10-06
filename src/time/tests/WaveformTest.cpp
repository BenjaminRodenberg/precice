#include <Eigen/Core>
#include "testing/TestContext.hpp"
#include "testing/Testing.hpp"
#include "testing/WaveformFixture.hpp"
#include "time/Waveform.hpp"

using namespace precice;
using namespace precice::time;

BOOST_AUTO_TEST_SUITE(TimeTests)

BOOST_AUTO_TEST_SUITE(WaveformTests)

BOOST_AUTO_TEST_SUITE(ExtrapolationTests)
BOOST_AUTO_TEST_CASE(testExtrapolateDataFirstOrder)
{
  PRECICE_TEST(1_rank);

  testing::WaveformFixture fixture;

  // Test first order extrapolation
  const int extrapolationOrder = 1;
  Waveform  waveform(1, extrapolationOrder, Waveform::UNDEFINED_INTERPOLATION_ORDER);
  BOOST_TEST(fixture.numberOfSamples(waveform) == 2);
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 1);
  BOOST_TEST(fixture.numberOfData(waveform) == 1);

  // use zero initial data
  waveform.moveToNextWindow();
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 0.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 0.0));

  Eigen::VectorXd value(1);
  value(0) = 1.0;
  waveform.store(value);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 0.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  waveform.moveToNextWindow(); // applies first order extrapolation in second window
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 2.0)); // = 2*1 - 0
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
  waveform.moveToNextWindow(); // applies first order extrapolation in forth window
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 12.0)); // 10.0 = 2 * 8 - 4
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 8.0));
}

BOOST_AUTO_TEST_CASE(testExtrapolateDataSecondOrder)
{
  PRECICE_TEST(1_rank);

  testing::WaveformFixture fixture;

  // Test second order extrapolation
  const int extrapolationOrder = 2;
  Waveform  waveform(1, extrapolationOrder, Waveform::UNDEFINED_INTERPOLATION_ORDER);
  BOOST_TEST(fixture.numberOfSamples(waveform) == 3);
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 1);
  BOOST_TEST(fixture.numberOfData(waveform) == 1);

  // use zero initial data
  waveform.moveToNextWindow();
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 0.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 0.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 2), 0.0));

  Eigen::VectorXd value(1);
  value(0) = 1.0;
  waveform.store(value);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 0.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 2), 0.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  waveform.moveToNextWindow(); // applies first order extrapolation in second window
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 3);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 2.0)); // = 2*1 - 0
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 1.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 2), 0.0));

  value(0) = 4.0;
  waveform.store(value);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 4.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 1.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 2), 0.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 3);
  waveform.moveToNextWindow(); // applies second order extrapolation in third window
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 3);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 8.0)); // = 2.5*4 - 2 * 1 + 0.5 * 0
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 4.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 2), 1.0));

  value(0) = 8.0;
  waveform.store(value);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 8.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 4.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 2), 1.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 3);
  waveform.moveToNextWindow(); // applies second order extrapolation in fourth window
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 3);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 12.5)); // = 2.5 * 8 - 2 * 4 + 0.5 * 1
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 8.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 2), 4.0));

  value(0) = 16.0;
  waveform.store(value);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 16.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 8.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 2), 4.0));
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 3);
  waveform.moveToNextWindow(); // applies second order extrapolation in fifth window
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 3);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 26.0)); // = 2.5 * 16.0 - 2 * 8 + 0.5 * 4
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 16.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 2), 8.0));
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(InterpolationTests)
BOOST_AUTO_TEST_CASE(testInterpolateDataZerothOrder)
{
  PRECICE_TEST(1_rank);

  testing::WaveformFixture fixture;

  // Test zeroth order interpolation
  const int extrapolationOrder = 0;
  const int interpolationOrder = 0;
  Waveform  waveform(1, extrapolationOrder, interpolationOrder);

  BOOST_TEST(waveform.lastTimeWindows().cols() == 2);
  BOOST_TEST(waveform.lastTimeWindows().rows() == 1);
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 1);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 0.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 0.0));

  Eigen::VectorXd value(1);
  value(0) = 1.0;
  waveform.store(value);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 0.0));

  BOOST_TEST(testing::equals(waveform.sample(0.0, interpolationOrder)(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform.sample(0.5, interpolationOrder)(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform.sample(1.0, interpolationOrder)(0, 0), 1.0));

  value(0) = 2.0;
  waveform.store(value);

  BOOST_TEST(testing::equals(waveform.sample(0.0, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.sample(0.5, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.sample(1.0, interpolationOrder)(0, 0), 2.0));

  waveform.moveToNextWindow();
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 2.0));

  BOOST_TEST(testing::equals(waveform.sample(0.0, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.sample(0.5, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.sample(1.0, interpolationOrder)(0, 0), 2.0));

  value(0) = 3.0;
  waveform.store(value);

  BOOST_TEST(testing::equals(waveform.sample(0.0, interpolationOrder)(0, 0), 3.0));
  BOOST_TEST(testing::equals(waveform.sample(0.5, interpolationOrder)(0, 0), 3.0));
  BOOST_TEST(testing::equals(waveform.sample(1.0, interpolationOrder)(0, 0), 3.0));
}

BOOST_AUTO_TEST_CASE(testInterpolateDataFirstOrder)
{
  PRECICE_TEST(1_rank);

  testing::WaveformFixture fixture;

  // Test first order interpolation
  const int extrapolationOrder = 0;
  const int interpolationOrder = 1;
  Waveform  waveform(1, extrapolationOrder, interpolationOrder);
  BOOST_TEST(waveform.lastTimeWindows().cols() == 2);
  BOOST_TEST(waveform.lastTimeWindows().rows() == 1);
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 1);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 0.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 0.0));

  Eigen::VectorXd value(1);
  value(0) = 1.0;
  waveform.store(value);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 0.0));

  BOOST_TEST(testing::equals(waveform.sample(0.0, interpolationOrder)(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform.sample(0.5, interpolationOrder)(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform.sample(1.0, interpolationOrder)(0, 0), 1.0));

  value(0) = 2.0;
  waveform.store(value);

  BOOST_TEST(testing::equals(waveform.sample(0.0, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.sample(0.5, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.sample(1.0, interpolationOrder)(0, 0), 2.0));

  waveform.moveToNextWindow();
  BOOST_TEST(fixture.numberOfValidSamples(waveform) == 2);
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 2.0));

  BOOST_TEST(testing::equals(waveform.sample(0.0, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.sample(0.5, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.sample(1.0, interpolationOrder)(0, 0), 2.0));

  value(0) = 3.0;
  waveform.store(value);

  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 0), 3.0));
  BOOST_TEST(testing::equals(waveform.lastTimeWindows()(0, 1), 2.0));

  BOOST_TEST(testing::equals(waveform.sample(0.0, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform.sample(0.5, interpolationOrder)(0, 0), 2.5));
  BOOST_TEST(testing::equals(waveform.sample(1.0, interpolationOrder)(0, 0), 3.0));
}

BOOST_AUTO_TEST_CASE(testInterpolateDataSecondOrder)
{
  PRECICE_TEST(1_rank);

  testing::WaveformFixture fixture;

  // Test second order interpolation
  const int extrapolationOrder = 0;
  const int interpolationOrder = 2;
  Waveform  waveform2(1, extrapolationOrder, interpolationOrder);
  BOOST_TEST(waveform2.lastTimeWindows().cols() == 3);
  BOOST_TEST(waveform2.lastTimeWindows().rows() == 1);
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 1);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 0.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 0.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 0.0));

  Eigen::VectorXd value(1);
  value(0) = 1.0;
  waveform2.store(value);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 0.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 0.0));

  BOOST_TEST(testing::equals(waveform2.sample(0.0, interpolationOrder)(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform2.sample(0.5, interpolationOrder)(0, 0), 1.0));
  BOOST_TEST(testing::equals(waveform2.sample(1.0, interpolationOrder)(0, 0), 1.0));

  value(0) = 2.0;
  waveform2.store(value);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 0.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 0.0));

  BOOST_TEST(testing::equals(waveform2.sample(0.0, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform2.sample(0.5, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform2.sample(1.0, interpolationOrder)(0, 0), 2.0));

  waveform2.moveToNextWindow();
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 2);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 2.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 0.0));

  BOOST_TEST(testing::equals(waveform2.sample(0.0, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform2.sample(0.5, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform2.sample(1.0, interpolationOrder)(0, 0), 2.0));

  value(0) = 8.0;
  waveform2.store(value);

  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 8.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 2.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 0.0));

  BOOST_TEST(testing::equals(waveform2.sample(0.0, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform2.sample(0.5, interpolationOrder)(0, 0), 5.0));
  BOOST_TEST(testing::equals(waveform2.sample(1.0, interpolationOrder)(0, 0), 8.0));

  value(0) = 4.0;
  waveform2.store(value);

  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 4.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 2.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 0.0));

  BOOST_TEST(testing::equals(waveform2.sample(0.0, interpolationOrder)(0, 0), 2.0));
  BOOST_TEST(testing::equals(waveform2.sample(0.5, interpolationOrder)(0, 0), 3.0));
  BOOST_TEST(testing::equals(waveform2.sample(1.0, interpolationOrder)(0, 0), 4.0));

  waveform2.moveToNextWindow();
  BOOST_TEST(fixture.numberOfValidSamples(waveform2) == 3);
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 4.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 4.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 2.0));

  BOOST_TEST(testing::equals(waveform2.sample(0.0, interpolationOrder)(0, 0), 4.0));
  BOOST_TEST(testing::equals(waveform2.sample(0.5, interpolationOrder)(0, 0), 4.25));
  BOOST_TEST(testing::equals(waveform2.sample(1.0, interpolationOrder)(0, 0), 4.0));

  value(0) = 8.0;
  waveform2.store(value);

  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 0), 8.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 1), 4.0));
  BOOST_TEST(testing::equals(waveform2.lastTimeWindows()(0, 2), 2.0));

  BOOST_TEST(testing::equals(waveform2.sample(0.0, interpolationOrder)(0, 0), 4.0));
  BOOST_TEST(testing::equals(waveform2.sample(0.5, interpolationOrder)(0, 0), 5.75));
  BOOST_TEST(testing::equals(waveform2.sample(1.0, interpolationOrder)(0, 0), 8.0));
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
