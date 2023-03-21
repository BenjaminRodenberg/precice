#include "WriteDataContext.hpp"

#include "utils/EigenHelperFunctions.hpp"

namespace precice::impl {

logging::Logger WriteDataContext::_log{"impl::WriteDataContext"};

WriteDataContext::WriteDataContext(
    mesh::PtrData data,
    mesh::PtrMesh mesh)
    : DataContext(data, mesh)
{
  _writeDataBuffer = WriteSample{Eigen::VectorXd()};
}

mesh::PtrData WriteDataContext::providedData()
{
  PRECICE_ASSERT(_providedData);
  return _providedData;
}

void WriteDataContext::writeIntoDataBuffer(int index, double value)
{
  PRECICE_DEBUG("Store value {} at id {}", value, index);
  if (index >= _writeDataBuffer.values.size()) {
    int expectedSize = index + 1;
    int actualSize   = _writeDataBuffer.values.size();
    PRECICE_DEBUG("Enlarge Buffer by {} ", expectedSize - actualSize);
    utils::append(_writeDataBuffer.values, Eigen::VectorXd(Eigen::VectorXd::Zero(expectedSize - actualSize)));
  }

  PRECICE_ASSERT(_writeDataBuffer.values.size() > index, _writeDataBuffer.values.size(), index);
  _writeDataBuffer.values[index] = value;
}

void WriteDataContext::storeWriteSampleAt(double relativeDt)
{
  _providedData->timeStepsStorage().setValuesAtTime(relativeDt, _writeDataBuffer.values);
}

void WriteDataContext::appendMappingConfiguration(MappingContext &mappingContext, const MeshContext &meshContext)
{
  PRECICE_ASSERT(meshContext.mesh->hasDataName(getDataName()));
  mesh::PtrData data = meshContext.mesh->data(getDataName());
  PRECICE_ASSERT(data != _providedData, "Data the write mapping is mapping to needs to be different from _providedData");
  mappingContext.fromData = _providedData;
  mappingContext.toData   = data;
  appendMapping(mappingContext);
  PRECICE_ASSERT(hasWriteMapping());
}

} // namespace precice::impl
