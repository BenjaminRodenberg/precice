#include "precice/SolverInterfaceFortran.hpp"
#include <iostream>
#include <memory>
#include <stddef.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include "logging/LogMacros.hpp"
#include "logging/Logger.hpp"
#include "precice/SolverInterface.hpp"
#include "precice/impl/versions.hpp"
#include "utils/assertion.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace std;

static std::unique_ptr<precice::SolverInterface> impl = nullptr;

static precice::logging::Logger _log("SolverInterfaceFortran");

static std::string errormsg = "preCICE has not been created properly. Be sure to call \"precicef_create\" before any other call to preCICE.";

namespace precice::impl {
/**
 * @brief Returns length of string without trailing whitespace.
 */
int strippedLength(const char *string, int length);

std::string_view strippedStringView(const char *string, int length);

} // namespace precice::impl

void precicef_create_(
    const char *participantName,
    const char *configFileName,
    const int * solverProcessIndex,
    const int * solverProcessSize,
    int         lengthAccessorName,
    int         lengthConfigFileName)
{
  // cout << "lengthAccessorName: " << lengthAccessorName << '\n';
  // cout << "lengthConfigFileName: " << lengthConfigFileName << '\n';
  // cout << "solverProcessIndex: " << *solverProcessIndex << '\n';
  // cout << "solverProcessSize: " << *solverProcessSize << '\n';
  int    strippedLength = precice::impl::strippedLength(participantName, lengthAccessorName);
  string stringAccessorName(participantName, strippedLength);
  strippedLength = precice::impl::strippedLength(configFileName, lengthConfigFileName);
  string stringConfigFileName(configFileName, strippedLength);
  // cout << "Accessor: " << stringAccessorName << "!" << '\n';
  // cout << "Config  : " << stringConfigFileName << "!" << '\n';
  impl.reset(new precice::SolverInterface(stringAccessorName,
                                          stringConfigFileName,
                                          *solverProcessIndex, *solverProcessSize));
}

void precicef_initialize_(
    double *timestepLengthLimit)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  *timestepLengthLimit = impl->initialize();
}

void precicef_advance_(
    double *timestepLengthLimit)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  *timestepLengthLimit = impl->advance(*timestepLengthLimit);
}

void precicef_finalize_()
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->finalize();
  impl.reset();
}

void precicef_get_dims_(
    int *dimensions)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  *dimensions = impl->getDimensions();
}

void precicef_is_coupling_ongoing_(
    int *isOngoing)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  if (impl->isCouplingOngoing()) {
    *isOngoing = 1;
  } else {
    *isOngoing = 0;
  }
}

void precicef_is_time_window_complete_(
    int *isComplete)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  if (impl->isTimeWindowComplete()) {
    *isComplete = 1;
  } else {
    *isComplete = 0;
  }
}

void precicef_requires_initial_data_(
    int *isRequired)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  *isRequired = impl->requiresInitialData() ? 1 : 0;
}

void precicef_requires_writing_checkpoint_(
    int *isRequired)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  *isRequired = impl->requiresWritingCheckpoint() ? 1 : 0;
}

void precicef_requires_reading_checkpoint_(
    int *isRequired)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  *isRequired = impl->requiresReadingCheckpoint() ? 1 : 0;
}

void precicef_has_mesh_(
    const char *meshName,
    int *       hasMesh,
    int         meshLengthName)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  int    strippedLength = precice::impl::strippedLength(meshName, meshLengthName);
  string stringMeshName(meshName, strippedLength);
  if (impl->hasMesh(meshName)) {
    *hasMesh = 1;
  } else {
    *hasMesh = 0;
  }
}

void precicef_has_data_(
    const char *mesh,
    const char *data,
    int *       hasData,
    int         meshLength,
    int         dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  if (impl->hasData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength))) {
    *hasData = 1;
  } else {
    *hasData = 0;
  }
}

void precicef_requires_mesh_connectivity_for_(
    const char *mesh,
    int *       required,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  if (impl->requiresMeshConnectivityFor(precice::impl::strippedStringView(mesh, meshLength))) {
    *required = 1;
  } else {
    *required = 0;
  }
}

void precicef_set_vertex_(
    const char *  mesh,
    const double *position,
    int *         vertexID,
    int           meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  *vertexID = impl->setMeshVertex(precice::impl::strippedStringView(mesh, meshLength), position);
}

void precicef_get_mesh_vertex_size_(
    const char *mesh,
    int *       meshSize,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  *meshSize = impl->getMeshVertexSize(precice::impl::strippedStringView(mesh, meshLength));
}

void precicef_set_vertices_(
    const char *mesh,
    const int * size,
    double *    positions,
    int *       positionIDs,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->setMeshVertices(precice::impl::strippedStringView(mesh, meshLength), *size, positions, positionIDs);
}

void precicef_set_edge_(
    const char *mesh,
    const int * firstVertexID,
    const int * secondVertexID,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->setMeshEdge(precice::impl::strippedStringView(mesh, meshLength), *firstVertexID, *secondVertexID);
}

void precicef_set_mesh_edges_(
    const char *mesh,
    const int * size,
    const int * vertices,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->setMeshEdges(precice::impl::strippedStringView(mesh, meshLength), *size, vertices);
}

void precicef_set_triangle_(
    const char *mesh,
    const int * firstVertexID,
    const int * secondVertexID,
    const int * thirdVertexID,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->setMeshTriangle(precice::impl::strippedStringView(mesh, meshLength), *firstVertexID, *secondVertexID, *thirdVertexID);
}

void precicef_set_mesh_triangles_(
    const char *mesh,
    const int * size,
    const int * vertices,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->setMeshTriangles(precice::impl::strippedStringView(mesh, meshLength), *size, vertices);
}

void precicef_set_quad_(
    const char *mesh,
    const int * firstVertexID,
    const int * secondVertexID,
    const int * thirdVertexID,
    const int * fourthVertexID,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->setMeshQuad(precice::impl::strippedStringView(mesh, meshLength), *firstVertexID, *secondVertexID, *thirdVertexID, *fourthVertexID);
}

void precicef_set_mesh_quads_(
    const char *mesh,
    const int * size,
    const int * vertices,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->setMeshQuads(precice::impl::strippedStringView(mesh, meshLength), *size, vertices);
}

void precicef_set_tetrahedron(
    const char *mesh,
    const int * firstVertexID,
    const int * secondVertexID,
    const int * thirdVertexID,
    const int * fourthVertexID,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->setMeshTetrahedron(precice::impl::strippedStringView(mesh, meshLength), *firstVertexID, *secondVertexID, *thirdVertexID, *fourthVertexID);
}

void precicef_set_mesh_tetrahedra_(
    const char *mesh,
    const int * size,
    const int * vertices,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->setMeshTetrahedra(precice::impl::strippedStringView(mesh, meshLength), *size, vertices);
}

void precicef_write_bvdata_(
    const char *mesh,
    const char *data,
    const int * size,
    int *       valueIndices,
    double *    values,
    int         meshLength,
    int         dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->writeBlockVectorData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *size, valueIndices, values);
}

void precicef_write_vdata_(
    const char *  mesh,
    const char *  data,
    const int *   valueIndex,
    const double *dataValue,
    int           meshLength,
    int           dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->writeVectorData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *valueIndex, dataValue);
}

void precicef_write_bsdata_(
    const char *mesh,
    const char *data,
    const int * size,
    int *       valueIndices,
    double *    values,
    int         meshLength,
    int         dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->writeBlockScalarData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *size, valueIndices, values);
}

void precicef_write_sdata_(
    const char *  mesh,
    const char *  data,
    const int *   valueIndex,
    const double *dataValue,
    int           meshLength,
    int           dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->writeScalarData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *valueIndex, *dataValue);
}

void precicef_read_bvdata_(
    const char *mesh,
    const char *data,
    const int * size,
    int *       valueIndices,
    double *    values,
    int         meshLength,
    int         dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->readBlockVectorData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *size, valueIndices, values);
}

void precicef_read_vdata_(
    const char *mesh,
    const char *data,
    const int * valueIndex,
    double *    dataValue,
    int         meshLength,
    int         dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->readVectorData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *valueIndex, dataValue);
}

void precicef_read_bsdata_(
    const char *mesh,
    const char *data,
    const int * size,
    int *       valueIndices,
    double *    values,
    int         meshLength,
    int         dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->readBlockScalarData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *size, valueIndices, values);
}

void precicef_read_sdata_(
    const char *mesh,
    const char *data,
    const int * valueIndex,
    double *    dataValue,
    int         meshLength,
    int         dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->readScalarData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *valueIndex, *dataValue);
}

int precice::impl::strippedLength(
    const char *string,
    int         length)
{
  int i = length - 1;
  while (((string[i] == ' ') || (string[i] == 0)) && (i >= 0)) {
    i--;
  }
  return i + 1;
}

std::string_view precice::impl::strippedStringView(const char *string, int length)
{
  return {string, static_cast<std::string_view::size_type>(strippedLength(string, length))};
}

void precicef_get_version_information_(
    char *versionInfo,
    int   lengthVersionInfo)
{
  const std::string &versionInformation = precice::versionInformation;
  PRECICE_ASSERT(versionInformation.size() < (size_t) lengthVersionInfo, versionInformation.size(), lengthVersionInfo);
  for (size_t i = 0; i < versionInformation.size(); i++) {
    versionInfo[i] = versionInformation[i];
  }
}

void precicef_requires_gradient_data_for_(
    const char *mesh,
    const char *data, int *required,
    int meshLength,
    int dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  if (impl->requiresGradientDataFor(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength))) {
    *required = 1;
  } else {
    *required = 0;
  }
}

void precicef_write_sgradient_data_(
    const char *  mesh,
    const char *  data,
    const int *   valueIndex,
    const double *gradientValues,
    int           meshLength,
    int           dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->writeScalarGradientData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *valueIndex, gradientValues);
}

void precicef_write_bsgradient_data_(
    const char *  mesh,
    const char *  data,
    const int *   size,
    const int *   valueIndices,
    const double *gradientValues,
    int           meshLength,
    int           dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->writeBlockScalarGradientData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *size, valueIndices, gradientValues);
}

void precicef_write_vgradient_data_(
    const char *  mesh,
    const char *  data,
    const int *   valueIndex,
    const double *gradientValues,
    int           meshLength,
    int           dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->writeVectorGradientData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *valueIndex, gradientValues);
}

void precicef_write_bvgradient_data_(
    const char *  mesh,
    const char *  data,
    const int *   size,
    const int *   valueIndices,
    const double *gradientValues,
    int           meshLength,
    int           dataLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->writeBlockVectorGradientData(precice::impl::strippedStringView(mesh, meshLength), precice::impl::strippedStringView(data, dataLength), *size, valueIndices, gradientValues);
}

void precicef_set_mesh_access_region_(
    const char *  mesh,
    const double *boundingBox,
    int           meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->setMeshAccessRegion(precice::impl::strippedStringView(mesh, meshLength), boundingBox);
}

void precicef_get_mesh_vertices_and_IDs_(
    const char *mesh,
    const int   size,
    int *       ids,
    double *    coordinates,
    int         meshLength)
{
  PRECICE_CHECK(impl != nullptr, errormsg);
  impl->getMeshVerticesAndIDs(precice::impl::strippedStringView(mesh, meshLength), size, ids, coordinates);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
