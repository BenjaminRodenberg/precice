#pragma once

#include "precice/Version.h"
#include "precice/export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C language bindings to main Application Programming Interface of preCICE
 *
 */

///@name Construction and Configuration
///@{

/**
 * @param[in] participantName Name of the participant using the interface. Has to
 *        match the name given for a participant in the xml configuration file.
 * @param[in] configurationFileName Name (with path) of the xml configuration file.
 * @param[in] solverProcessIndex If the solver code runs with several processes,
 *        each process using preCICE has to specify its index, which has to start
 *        from 0 and end with solverProcessSize - 1.
 * @param[in] solverProcessSize The number of solver processes using preCICE.
 * @param[in] communicator A pointer to an MPI_Comm to use as communicator.
 */
PRECICE_API void precicec_createSolverInterface_withCommunicator(
    const char *participantName,
    const char *configFileName,
    int         solverProcessIndex,
    int         solverProcessSize,
    void *      communicator);

/**
 * @brief Creates the coupling interface and configures it.
 *
 * Has to be called before any other method of this interface.
 *
 * @param[in] participantName Name of the participant accessing the interface. Has to
 *                          match one of the names specified in the
 *                          configuration xml file.
 * @param[in] configFileName (Path and) name of the xml configuration file
 *                            containing the precice configuration.
 * @param[in] solverProcessIndex If the solver code runs with several processes,
 *                               each process using preCICE has to specify its index, which has to start
 *                               from 0 and end with solverProcessSize - 1.
 * @param[in] solverProcessSize The number of solver processes using preCICE.
 */
PRECICE_API void precicec_createSolverInterface(
    const char *participantName,
    const char *configFileName,
    int         solverProcessIndex,
    int         solverProcessSize);

///@}

/// @name Steering Methods
///@{

/**
 * @brief Initiates the coupling to the coupling supervisor and initializes coupling data.
 *
 * @return Maximal length of first timestep to be computed by solver.
 */
PRECICE_API double precicec_initialize();

/**
 * @brief Exchanges data between solver and coupling supervisor.
 *
 * @param[in] computedTimestepLength Length of timestep computed by solver.
 * @return Maximal length of next timestep to be computed by solver.
 */
PRECICE_API double precicec_advance(double computedTimestepLength);

/**
 * @brief Finalizes the coupling to the coupling supervisor.
 */
PRECICE_API void precicec_finalize();

///@}

///@name Status Queries
///@{

/**
 * @brief Returns the number of spatial configurations for the coupling.
 */
PRECICE_API int precicec_getDimensions();

/**
 * @brief Returns true (->1), if the coupled simulation is ongoing
 */
PRECICE_API int precicec_isCouplingOngoing();

/**
 * @brief Returns true (->1), if the coupling time window is completed.
 */
PRECICE_API int precicec_isTimeWindowComplete();

///@}

///@name Action Methods
///@{

/// @copydoc precice::SolverInterface::requiresInitialData()
PRECICE_API int precicec_requiresInitialData();

/// @copydoc precice::SolverInterface::requiresWritingCheckpoint()
PRECICE_API int precicec_requiresWritingCheckpoint();

/// @copydoc precice::SolverInterface::requiresReadingCheckpoint()
PRECICE_API int precicec_requiresReadingCheckpoint();

///@name Mesh Access
///@anchor precice-mesh-access
///@{

/**
 * @brief Checks if the mesh with given name is used by a solver.
 *
 * @param[in] meshName the name of the mesh
 * @returns whether the mesh is used.
 */
PRECICE_API int precicec_hasMesh(const char *meshName);

/**
 * @brief Returns id belonging to the given mesh name
 */
PRECICE_API int precicec_getMeshID(const char *meshName);

/// @copydoc precice::SolverInterface::requiresMeshConnectivityFor()
PRECICE_API int precicec_requiresMeshConnectivityFor(int meshID);

/**
 * @brief Creates a mesh vertex
 *
 * @param[in] meshID the id of the mesh to add the vertex to.
 * @param[in] position a pointer to the coordinates of the vertex.
 * @returns the id of the created vertex
 */
PRECICE_API int precicec_setMeshVertex(
    int           meshID,
    const double *position);

/**
 * @brief Returns the number of vertices of a mesh.
 *
 * @param[in] meshID the id of the mesh
 * @returns the amount of the vertices of the mesh
 */
PRECICE_API int precicec_getMeshVertexSize(int meshID);

/**
 * @brief Creates multiple mesh vertices
 *
 * @param[in] meshID the id of the mesh to add the vertices to.
 * @param[in] size Number of vertices to create
 * @param[in] positions a pointer to the coordinates of the vertices
 *            The 2D-format is (d0x, d0y, d1x, d1y, ..., dnx, dny)
 *            The 3D-format is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz)
 *
 * @param[out] ids The ids of the created vertices
 */
PRECICE_API void precicec_setMeshVertices(
    int           meshID,
    int           size,
    const double *positions,
    int *         ids);

/**
 * @brief Get vertex positions for multiple vertex ids from a given mesh
 *
 * @param[in] meshID the id of the mesh to read the vertices from.
 * @param[in] size Number of vertices to lookup
 * @param[in] ids The ids of the vertices to lookup
 * @param[out] positions a pointer to memory to write the coordinates to
 *            The 2D-format is (d0x, d0y, d1x, d1y, ..., dnx, dny)
 *            The 3D-format is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz)
 */
PRECICE_API void precicec_getMeshVertices(
    int        meshID,
    int        size,
    const int *ids,
    double *   positions);

/**
 * @brief Gets mesh vertex IDs from positions.
 *
 * @param[in] meshID ID of the mesh to retrieve positions from
 * @param[in] size Number of vertices to lookup.
 * @param[in] positions Positions to find ids for.
 *            The 2D-format is (d0x, d0y, d1x, d1y, ..., dnx, dny)
 *            The 3D-format is (d0x, d0y, d0z, d1x, d1y, d1z, ..., dnx, dny, dnz)
 * @param[out] ids IDs corresponding to positions.
 */
PRECICE_API void precicec_getMeshVertexIDsFromPositions(
    int           meshID,
    int           size,
    const double *positions,
    int *         ids);

/**
 * @brief Sets mesh edge from vertex IDs, returns edge ID.
 *
 * @param[in] meshID ID of the mesh to add the edge to
 * @param[in] firstVertexID ID of the first vertex of the edge
 * @param[in] secondVertexID ID of the second vertex of the edge
 *
 * @return the ID of the edge
 */
PRECICE_API void precicec_setMeshEdge(
    int meshID,
    int firstVertexID,
    int secondVertexID);

/**
   * @brief Sets multiple mesh edge from vertex IDs
   *
   * @param[in] meshID ID of the mesh to add the edges to
   * @param[in] size the amount of edges to set
   * @param[in] vertices an array containing 2*size vertex IDs
   *
   * @pre vertices were added to the mesh with the ID meshID
   */
PRECICE_API void precicec_setMeshEdges(
    int        meshID,
    int        size,
    const int *vertices);

/**
 * @brief Sets a triangle from vertex IDs. Creates missing edges.
 */
PRECICE_API void precicec_setMeshTriangle(
    int meshID,
    int firstVertexID,
    int secondVertexID,
    int thirdVertexID);

/**
   * @brief Sets multiple mesh triangles from vertex IDs
   *
   * @param[in] meshID ID of the mesh to add the triangles to
   * @param[in] size the amount of triangles to set
   * @param[in] vertices an array containing 3*size vertex IDs
   *
   * @pre vertices were added to the mesh with the ID meshID
   */
PRECICE_API void precicec_setMeshTriangles(
    int        meshID,
    int        size,
    const int *vertices);

/**
  * @brief Sets surface mesh quadrangle from vertex IDs.
  *
  * @param[in] meshID ID of the mesh to add the Quad to
  * @param[in] firstVertexID ID of the first vertex of the Quad
  * @param[in] secondVertexID ID of the second vertex of the Quad
  * @param[in] thirdVertexID ID of the third vertex of the Quad
  * @param[in] fourthVertexID ID of the fourth vertex of the Quad
 */
PRECICE_API void precicec_setMeshQuad(
    int meshID,
    int firstVertexID,
    int secondVertexID,
    int thirdVertexID,
    int fourthVertexID);

/**
   * @brief Sets multiple mesh quads from vertex IDs
   *
   * @param[in] meshID ID of the mesh to add the quad to
   * @param[in] size the amount of quads to set
   * @param[in] vertices an array containing 4*size vertex IDs
   *
   * @pre vertices were added to the mesh with the ID meshID
   */
PRECICE_API void precicec_setMeshQuads(
    int        meshID,
    int        size,
    const int *vertices);

/**
  * @brief Sets mesh tetrahedron from vertex IDs.
  *
  * @param[in] meshID ID of the mesh to add the Tetra to
  * @param[in] firstVertexID ID of the first vertex of the Tetra
  * @param[in] secondVertexID ID of the second vertex of the Tetra
  * @param[in] thirdVertexID ID of the third vertex of the Tetra
  * @param[in] fourthVertexID ID of the fourth vertex of the Tetra
 */
PRECICE_API void precicec_setMeshTetrahedron(
    int meshID,
    int firstVertexID,
    int secondVertexID,
    int thirdVertexID,
    int fourthVertexID);

/**
   * @brief Sets multiple mesh tetrahedra from vertex IDs
   *
   * @param[in] meshID ID of the mesh to add the tetrahedra to
   * @param[in] size the amount of tetrahedra to set
   * @param[in] vertices an array containing 4*size vertex IDs
   *
   * @pre vertices were added to the mesh with the ID meshID
   */
PRECICE_API void precicec_setMeshTetrahedra(
    int        meshID,
    int        size,
    const int *vertices);

///@}

///@name Data Access
///@{

/**
 * @brief Returns true (!=0), if data with given name is available.
 */
PRECICE_API int precicec_hasData(const char *dataName, int meshID);

/**
 * @brief Returns the data id belonging to the given name.
 *
 * The given name (dataName) has to be one of the names specified in the
 * configuration file. The data id obtained can be used to read and write
 * data to and from the coupling mesh.
 */
PRECICE_API int precicec_getDataID(const char *dataName, int meshID);

/**
 * @brief Writes vector data values given as block.
 *
 * The block must contain the vector values in the following form:
 * values = (d0x, d0y, d0z, d1x, d1y, d1z, ...., dnx, dny, dnz), where n is
 * the number of vector values. In 2D, the z-components are removed.
 *
 * @param[in] dataID ID of the data to be written.
 * @param[in] size Number of indices, and number of values * dimensions.
 * @param[in] values Values of the data to be written.
 */
PRECICE_API void precicec_writeBlockVectorData(
    int           dataID,
    int           size,
    const int *   valueIndices,
    const double *values);

/**
 * @brief Writes vectorial foating point data to the coupling mesh.
 *
 * @param[in] dataID ID of the data to be written. Obtained by getDataID().
 * @param[in] dataPosition Spatial position of the data to be written.
 * @param[in] dataValue Vectorial data value to be written.
 */
PRECICE_API void precicec_writeVectorData(
    int           dataID,
    int           valueIndex,
    const double *dataValue);

/**
 * @brief See precice::SolverInterface::writeBlockScalarData().
 */
PRECICE_API void precicec_writeBlockScalarData(
    int           dataID,
    int           size,
    const int *   valueIndices,
    const double *values);

/**
 * @brief Writes scalar floating point data to the coupling mesh.
 *
 * @param[in] dataID ID of the data to be written. Obtained by getDataID().
 * @param[in] dataPosition Spatial position of the data to be written.
 * @param[in] dataValue Scalar data value to be written.
 */
PRECICE_API void precicec_writeScalarData(
    int    dataID,
    int    valueIndex,
    double dataValue);

/**
 * @brief Reads vector data values given as block.
 *
 * The block contains the vector values in the following form:
 * values = (d0x, d0y, d0z, d1x, d1y, d1z, ...., dnx, dny, dnz), where n is
 * the number of vector values. In 2D, the z-components are removed.
 *
 * @param[in] dataID ID of the data to be read.
 * @param[in] size  Number of indices, and number of values * dimensions.
 * @param[in] valueIndices Indices (from setReadPosition()) of data values.
 * @param[in] values Values of the data to be read.
 */
PRECICE_API void precicec_readBlockVectorData(
    int        dataID,
    int        size,
    const int *valueIndices,
    double *   values);

/**
 * @brief Reads vectorial foating point data from the coupling mesh.
 *
 * @param[in] dataID ID of the data to be read. Obtained by getDataID().
 * @param[in] dataPosition Position where the read data should be mapped to.
 * @param[out] dataValue Vectorial data value read.
 */
PRECICE_API void precicec_readVectorData(
    int     dataID,
    int     valueIndex,
    double *dataValue);

/**
 * @brief See precice::SolverInterface::readBlockScalarData().
 */
PRECICE_API void precicec_readBlockScalarData(
    int        dataID,
    int        size,
    const int *valueIndices,
    double *   values);

/**
 * @brief Reads scalar foating point data from the coupling mesh.
 *
 * @param[in] dataID ID of the data to be read. Obtained by getDataID().
 * @param[in] dataPosition Position where the read data should be mapped to.
 * @param[out] dataValue Scalar data value read.
 */
PRECICE_API void precicec_readScalarData(
    int     dataID,
    int     valueIndex,
    double *dataValue);

/**
 * @brief Returns information on the version of preCICE.
 *
 * Returns a semicolon-separated C-string containing:
 *
 * 1) the version of preCICE
 * 2) the revision information of preCICE
 * 3) the configuration of preCICE including MPI, PETSC, PYTHON
 */
PRECICE_API const char *precicec_getVersionInformation();

///@}

/** @name Experimental Data Access
 * These API functions are \b experimental and may change in future versions.
 */
///@{

/// @copydoc precice::SolverInterface::isGradientDataRequired
PRECICE_API int precicec_isGradientDataRequired(int dataID);

/// @copydoc precice::SolverInterface::writeScalarGradientData
PRECICE_API void precicec_writeScalarGradientData(
    int           dataID,
    int           valueIndex,
    const double *gradientValues);

/// @copydoc precice::SolverInterface::writeBlockScalarGradientData
PRECICE_API void precicec_writeBlockScalarGradientData(
    int           dataID,
    int           size,
    const int *   valueIndices,
    const double *gradientValues);

/// @copydoc precice::SolverInterface::writeVectorGradientData
PRECICE_API void precicec_writeVectorGradientData(
    int           dataID,
    int           valueIndex,
    const double *gradientValues);

/// @copydoc precice::SolverInterface::writeBlockVectorGradientData
PRECICE_API void precicec_writeBlockVectorGradientData(
    int           dataID,
    int           size,
    const int *   valueIndices,
    const double *gradientValues);

/**
 * @brief See precice::SolverInterface::setMeshAccessRegion().
 */
PRECICE_API void precicec_setMeshAccessRegion(
    const int     meshID,
    const double *boundingBox);

/**
 * @brief See precice::SolverInterface::getMeshVerticesAndIDs().
 */
PRECICE_API void precicec_getMeshVerticesAndIDs(
    const int meshID,
    const int size,
    int *     ids,
    double *  coordinates);

///@}

#ifdef __cplusplus
}
#endif
