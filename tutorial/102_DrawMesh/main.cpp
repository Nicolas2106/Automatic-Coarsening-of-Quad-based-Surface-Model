#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/fit_plane.h>
#include <igl/polygon_corners.h>
#include <igl/polygons_to_triangles.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_face_normals.h>
#include <igl/doublearea.h>
#include <queue>
#include "tutorial_shared_path.h"

struct Segment
{
  int v1, v2; // The two vertices at the ends of the segment
  double length;
  bool isDiagonal; // True if the segment is a diagonal, false if it's an edge

  bool operator==(const Segment& s2) const
  {
    return length == s2.length && isDiagonal == s2.isDiagonal &&
      std::min(v1, v2) == std::min(s2.v1, s2.v2) &&
      std::max(v1, v2) == std::max(s2.v1, s2.v2);
  }
};

struct CompareTwoSegments
{
  bool operator()(const Segment& s1, const Segment& s2) const
  {
    if (s1.length == s2.length)
    {
      int min1 = std::min(s1.v1, s1.v2), min2 = std::min(s2.v1, s2.v2);
      return min1 == min2 ? std::max(s1.v1, s1.v2) > std::max(s2.v1, s2.v2) : min1 > min2;
    } // TODO necessary?
    return s1.length > s2.length;
  }
};

double squared_distance(const Eigen::RowVector3d& vertex1,
  const Eigen::RowVector3d& vertex2)
{
  double deltaX = vertex1[0] - vertex2[0],
    deltaY = vertex1[1] - vertex2[1],
    deltaZ = vertex1[2] - vertex2[2];
  return deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;
}

void insertSegment(int vert1, int vert2, bool isDiagonal, Eigen::MatrixXd& V,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations)
{
  double baseLength = squared_distance(V.row(vert1), V.row(vert2));
  operations.emplace(Segment{ vert1, vert2, 
    isDiagonal ? baseLength : 2 * baseLength, isDiagonal });
}

void removeSegment(int vert1, int vert2, bool isDiagonal, Eigen::MatrixXd& V,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  double baseLength = squared_distance(V.row(vert1), V.row(vert2));
  delOperations.emplace(Segment{ vert1, vert2,
    isDiagonal ? baseLength : 2 * baseLength, isDiagonal });
}

void per_quad_faces_normals(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& N)
{
  int facesCount = F.rows();
  N.resize(facesCount, 3);

  // Loop over the faces
#pragma omp parallel for if (facesCount > 10000) // TODO correct number?
  for (int i = 0; i < facesCount; ++i)
  {
    Eigen::RowVector3d nN, C;
    Eigen::MatrixXd vV(4, 3);
    Eigen::RowVector3d row0 = V.row(F(i, 0)), row1 = V.row(F(i, 1)),
      row2 = V.row(F(i, 2)), row3 = V.row(F(i, 3));

    vV << row0, row1, row2, row3;
    igl::fit_plane(vV, nN, C);

    // Choose the correct normal direction
    Eigen::RowVector3d cp = (row0 - row2).cross(row1 - row3);
    N.row(i) = cp.dot(nN) >= 0 ? nN : -nN;
  }
}

void per_vertices_normals(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& N)
{
  N.setZero(V.rows(), 3);
  
  Eigen::MatrixXd FN;
  per_quad_faces_normals(V, F, FN);

  int numFaces = F.rows();
  for (int i = 0; i < numFaces; ++i) // Loop over the faces
  {
    for (int j = 0; j < 4; ++j)
    {
      N.row(F(i, j)) += FN.row(i);
    }
  }

  // Normalization
  N.rowwise().normalize();
}

void new_vertex_pos2(Eigen::MatrixXd& V, Eigen::MatrixXd& normals, std::vector<bool>& tombStonesV,
  std::set<int>& borderVerts, Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, 
  std::vector<std::set<int>>& adt, const int oldVert,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  if (!tombStonesV[oldVert]) { return; }

  if (borderVerts.count(oldVert))
  {
    return; // Vertex belongs to the mesh border. Avoid repositioning TODO sure?
  }
  
  Eigen::RowVector3d centroid{ 0.0, 0.0, 0.0 }, rawOldVert = V.row(oldVert);
  int size = 0;
  Eigen::RowVector4i f;
  std::vector<Eigen::RowVector4i> adjacentFaces;
  for (int face : adt[oldVert])
  {
    if (tombStonesF[face])
    {
      f = F.row(face);
      adjacentFaces.emplace_back(f);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == oldVert)
        {
          int nextVert = f[(i + 1) % 4], oppositeVert = f[(i + 2) % 4];
          
          centroid += V.row(nextVert);
          centroid += V.row(oppositeVert);
          size += 2;

          removeSegment(oldVert, nextVert, false, V, delOperations);
          removeSegment(oldVert, oppositeVert, true, V, delOperations);

          break;
        }
      }
    }
  }
  centroid /= size;

  /*Eigen::MatrixXi faces(adjacentFaces.size(), 4);
  for (int i = 0; i < adjacentFaces.size(); ++i)
  {
    Eigen::RowVector4i a = adjacentFaces[i];
    faces.row(i) << a[0], a[1], a[2], a[3];
  }
  Eigen::MatrixXd N;
  per_quad_faces_normals(V, faces, N);

  Eigen::RowVector3d normal = { 0.0,  0.0, 0.0 };
  for (int i = 0; i < N.rows(); ++i)
  {
    normal += N.row(i);
  }
  normal /= N.rows();
  normal.normalize();*/

  Eigen::RowVector3d normal = normals.row(oldVert);
  double proj = (centroid - rawOldVert).dot(normal);
  Eigen::RowVector3d rawNewVert = centroid - (proj * normal);
  V.row(oldVert) = rawNewVert;

  // Update the priority queue of potential operations according the new configuration
  for (Eigen::RowVector4i f : adjacentFaces)
  {
    for (int i = 0; i < 4; ++i)
    {
      if (f[i] == oldVert)
      {
        int nextVert = f[(i + 1) % 4], oppositeVert = f[(i + 2) % 4];

        insertSegment(oldVert, nextVert, false, V, operations);
        insertSegment(oldVert, oppositeVert, true, V, operations);

        break;
      }
    }
  }
}

void new_vertex_pos(Eigen::MatrixXd& V, Eigen::MatrixXd& normals, std::set<int>& borderVerts, Eigen::MatrixXi& F,
  std::vector<bool>& tombStonesF, std::vector<std::set<int>>& adt,
  const int oldVert, const Eigen::RowVector4i& baseFace, Eigen::RowVector3d vertOnFace,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations,
  bool collapse)
{
  //if (!tombStonesV[oldVert]) { return; } TODO
  
  if (borderVerts.count(oldVert))
  {
    return; // Vertex belongs to the mesh border. Avoid repositioning
  }
  
  Eigen::RowVector3d centroid{ 0.0, 0.0, 0.0 }, rawOldVert = V.row(oldVert);
  int size = 0;
  Eigen::RowVector4i f;
  std::vector<Eigen::RowVector4i> adjacentFaces;
  for (int face : adt[oldVert])
  {
    if (tombStonesF[face])
    {
      f = F.row(face);
      adjacentFaces.emplace_back(f);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == oldVert)
        {
          int nextVert = f[(i + 1) % 4], oppositeVert = f[(i + 2) % 4];
          Eigen::RowVector3d rawNextVert = V.row(nextVert),
            rawOppVert = V.row(oppositeVert);

          centroid += rawNextVert;
          centroid += rawOppVert;
          size += 2;

          if (!collapse)
          {
            removeSegment(oldVert, nextVert, false, V, delOperations);
            removeSegment(oldVert, oppositeVert, true, V, delOperations);
          }

          break;
        }
      }
    }
  }
  centroid /= size;

  Eigen::Matrix<double, 4, 3> pointCloud;
  pointCloud << V.row(baseFace[0]), V.row(baseFace[1]),
    V.row(baseFace[2]), V.row(baseFace[3]);
  Eigen::RowVector3d normal, pointOnPlane;
  igl::fit_plane(pointCloud, normal, pointOnPlane);

  double proj = (centroid - vertOnFace).dot(normal);
  Eigen::RowVector3d rawNewVert = centroid - (proj * normal);
  V.row(oldVert) = rawNewVert;
  
  Eigen::MatrixXd N;
  Eigen::MatrixXi faces(adjacentFaces.size(), 4);
  for (int i = 0; i < adjacentFaces.size(); ++i)
  {
    Eigen::RowVector4i a = adjacentFaces[i];
    faces.row(i) << a[0], a[1], a[2], a[3];
  }
  per_quad_faces_normals(V, faces, N);
  Eigen::RowVector3d n = { 0.0,  0.0, 0.0 };
  for (int i = 0; i < N.rows(); ++i)
  {
    n += N.row(i);
  }
  n /= N.rows();
  n.normalize();
  normals.row(oldVert) = n;

  // Update the priority queue of potential operations according the new configuration
  for (int face : adt[oldVert])
  {
    if (tombStonesF[face])
    {
      f = F.row(face);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == oldVert)
        {
          int nextVert = f[(i + 1) % 4], oppositeVert = f[(i + 2) % 4];

          insertSegment(oldVert, nextVert, false, V, operations);
          insertSegment(oldVert, oppositeVert, true, V, operations);

          break;
        }
      }
    }
  }
}

int adjacent_face(int face, int vert1, int vert2,
  std::vector<std::set<int>>& adt, std::vector<bool>& tombStonesF)
{
  for (int f1 : adt[vert1])
  {
    if (tombStonesF[f1] && f1 != face)
    {
      for (int f2 : adt[vert2])
      {
        if (tombStonesF[f2] && f2 != face)
        {
          if (f1 == f2) { return f1; }// Adjacent face found
        }
      }
    }
  }

  return -1; // No adjacent face
}

void remove_doublet(int vertexToBeRemoved, Eigen::MatrixXd& V, Eigen::MatrixXd& normals,
  std::vector<bool>& tombStonesV, std::set<int>& borderVerts, Eigen::MatrixXi& F,
  std::vector<bool>& tombStonesF, int& aliveFaces, std::vector<std::set<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations,
  bool force = false)
{
  if (aliveFaces < 2) { return; } // Too few faces to remove doublets
  if (!tombStonesV[vertexToBeRemoved]) { return; } // This vert has already been removed
  
  // Avoid removal if the vertex belongs to the mesh border
  if (borderVerts.count(vertexToBeRemoved) && !force)
  {
    return; // Doublet removal avoided
  }

  int adtSize = 0, faceToBeRemoved = -1, faceToBeMaintained = -1;
  for (int face : adt[vertexToBeRemoved])
  {
    if (tombStonesF[face])
    {
      ++adtSize;
      if (faceToBeRemoved == -1)
      {
        faceToBeRemoved = face;
      }
      else
      {
        faceToBeMaintained = face;
        break;
      }
    }
  }
  if (adtSize != 2) { return; } // Incorrect input (No doublet)
  if (faceToBeRemoved == -1 || faceToBeMaintained == -1) { return; }

  Eigen::RowVector4i rawFaceRemoved = F.row(faceToBeRemoved);
  int endVert1 = -1, oppositeVert, endVert2;
  for (int i = 0; i < 4; ++i)
  {
    if (rawFaceRemoved[i] == vertexToBeRemoved)
    {
      endVert1 = rawFaceRemoved[(i + 1) % 4];
      oppositeVert = rawFaceRemoved[(i + 2) % 4];
      endVert2 = rawFaceRemoved[(i + 3) % 4];
      break;
    }
  }
  if (endVert1 == -1) { return; }

  // Partially update the priority queue of operations according the doublet deletion
  Eigen::RowVector3d rawEndVert1 = V.row(endVert1);
  Eigen::RowVector3d rawEndVert2 = V.row(endVert2);
  Eigen::RowVector3d rawOppVert = V.row(oppositeVert);

  Eigen::RowVector3d rawVertToBeRemoved = V.row(vertexToBeRemoved);
  
  removeSegment(vertexToBeRemoved, endVert1, false, V, delOperations);
  removeSegment(vertexToBeRemoved, endVert2, false, V, delOperations);
  
  removeSegment(vertexToBeRemoved, oppositeVert, true, V, delOperations);
  removeSegment(endVert1, endVert2, true, V, delOperations);

  // Remove one of the two faces adjacent to the doublet
  tombStonesF[faceToBeRemoved] = false;
  --aliveFaces;

  // Modify the remaining face and complete the update of the priority queue
  Eigen::RowVector4i rawFaceMaintained = F.row(faceToBeMaintained);
  int v = -1;
  for (int i = 0; i < 4; ++i)
  {
    if (rawFaceMaintained[i] == vertexToBeRemoved)
    {
      F.row(faceToBeMaintained)[i] = oppositeVert;

      v = F.row(faceToBeMaintained)[(i + 2) % 4];
      removeSegment(v, vertexToBeRemoved, true, V, delOperations);
      insertSegment(v, oppositeVert, true, V, operations);

      break;
    }
  }
  if (v == -1) { return; }
  rawFaceMaintained = F.row(faceToBeMaintained);

  if (v != oppositeVert)
  {
    // Add the new face (the face mantained) to adt
    adt[oppositeVert].emplace(faceToBeMaintained);
  }

  // Remove the vertex at the center of the doublet
  tombStonesV[vertexToBeRemoved] = false;

  // Check if a singlet has been created
  if (v == oppositeVert)
  {
    // Singlet found. 
    // Remove the diagonals and the edges belong to the singlet
    int singletIndex = -1, valence = 0;
    for (int i = 0; i < 4; ++i, valence = 0)
    {
      for (int face : adt[rawFaceMaintained[i]])
      {
        if (tombStonesF[face]) { ++valence; }
      }

      if (valence == 1)
      {
        singletIndex = i; // Singlet vertex has valence equal to 1
      }

    }
    if (singletIndex == -1) { return; }

    int singletVert = rawFaceMaintained[singletIndex];
    int oppSingletVert = rawFaceMaintained[(singletIndex + 2) % 4];
    int doubleSingletVert = rawFaceMaintained[(singletIndex + 1) % 4];
    Eigen::RowVector3d rawOppSingVert = V.row(oppSingletVert);

    removeSegment(singletVert, oppSingletVert, true, V, delOperations);
    removeSegment(doubleSingletVert, doubleSingletVert, true, V, delOperations);

    removeSegment(doubleSingletVert, singletVert, false, V, delOperations);
    removeSegment(doubleSingletVert, oppSingletVert, false, V, delOperations);

    // Remove the face and the singlet vertex
    tombStonesF[faceToBeMaintained] = false;
    --aliveFaces;
    tombStonesV[singletVert] = false;
  }
  else
  {
    // Compute and set the new positions of vertices adjacent to the doublet
    new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, v,
      operations, delOperations);
    new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, oppositeVert,
      operations, delOperations);
    new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, endVert1,
      operations, delOperations);
    new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, endVert2,
      operations, delOperations);
  }

  // Check if other doublets are created in cascade at the two endpoints
  int v1Valence = 0, v2Valence = 0;
  for (int face : adt[endVert1])
  {
    if (tombStonesF[face]) { ++v1Valence; }
  }
  if (v1Valence == 2) // Doublet found
  {
    remove_doublet(endVert1, V, normals, tombStonesV, borderVerts, F, tombStonesF, 
      aliveFaces, adt, operations, delOperations);
  }

  for (int face : adt[endVert2])
  {
    if (tombStonesF[face]) { ++v2Valence; }
  }
  if (v2Valence == 2) // Doublet found
  {
    remove_doublet(endVert2, V, normals, tombStonesV, borderVerts, F, tombStonesF, 
      aliveFaces, adt, operations, delOperations);
  }
}

bool try_vertex_rotation(int rotationVert, Eigen::MatrixXd& V, Eigen::MatrixXd& normals,
  std::vector<bool>& tombStonesV, std::set<int>& borderVerts, Eigen::MatrixXi& F, 
  std::vector<bool>& tombStonesF,int& aliveFaces, std::vector<std::set<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations,
  int vertForNextCollapse = -1)
{
  if (!tombStonesV[rotationVert]) { return false; }
  
  // Avoid the rotation if the vertex belongs to the mesh border
  if (borderVerts.count(rotationVert))
  {
    return true; // Rotation avoided
  }
  
  Eigen::RowVector3d rawRotVert = V.row(rotationVert);
  std::vector<Segment> segments; // Edges and diagonals involved in the vertex rotation

  // The vertices to be checked (after the rotation) for the possible presence of doublets
  std::vector<int> vertsForDoublets;

  // Get the sum of the edges' and diagonals' lengths
  double edgesSum = 0.0, diagonalsSum = 0.0;
  Eigen::RowVector4i f;
  for (int face : adt[rotationVert])
  {
    if (tombStonesF[face])
    {
      f = F.row(face);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == rotationVert)
        {
          int nextVert = f[(i + 1) % 4], oppVert = f[(i + 2) % 4];
          vertsForDoublets.emplace_back(nextVert);

          double edgeLength = squared_distance(rawRotVert, V.row(nextVert));
          double diagLength = squared_distance(rawRotVert, V.row(oppVert));

          if (vertForNextCollapse == -1)
          {
            edgesSum += edgeLength;
            diagonalsSum += diagLength;
          }

          segments.emplace_back(Segment{ rotationVert, nextVert, 2*edgeLength, false });
          segments.emplace_back(Segment{ rotationVert, oppVert, diagLength, true });

          break;
        }
      }
    }
  }

  // Check if the sum of the edge lengths overcomes the sum of the diagonals lengths
  if (vertForNextCollapse != -1 || edgesSum > diagonalsSum) // Vertex rotation
  {
    // Sort the faces around the vertex to be rotated counterclockwise
    std::vector<int> facesCounterclockwise, vertsOnPerimeter;
    int firstFace = -1;
    for (int face : adt[rotationVert])
    {
      if (tombStonesF[face])
      {
        firstFace = face;
        break;
      }
    }
    if (firstFace == -1) { return false; }

    int currFace = firstFace;
    do
    {
      facesCounterclockwise.emplace_back(currFace);
      f = F.row(currFace);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == rotationVert)
        {
          int oppNextVert = f[(i + 3) % 4];

          vertsOnPerimeter.emplace_back(f[(i + 2) % 4]);
          vertsOnPerimeter.emplace_back(oppNextVert);

          currFace = adjacent_face(currFace, rotationVert, oppNextVert, adt, tombStonesF);

          break;
        }
      }
    } while (currFace != firstFace);

    // Rotate the faces involved in the vertex rotation
    int tmpCounter = 0, vertsOnPerimeterSize = vertsOnPerimeter.size();
    for (int face : facesCounterclockwise)
    {
      int v2 = vertsOnPerimeter[tmpCounter++],
        v3 = vertsOnPerimeter[tmpCounter++],
        v4 = vertsOnPerimeter[tmpCounter % vertsOnPerimeterSize];
      F.row(face) << rotationVert, v2, v3, v4;
    }

    // Update adt and partially update the priority queue of potential operations
    int facesCounterclockwiseSize = facesCounterclockwise.size();
    for (int i = 0; i < vertsOnPerimeterSize; )
    {
      int vert = vertsOnPerimeter[i];
      int faceIndex = i / 2 == 0 ? facesCounterclockwiseSize - 1 : (i / 2) - 1;
      adt[vert].emplace(facesCounterclockwise[faceIndex]);

      i += 2;
      int vert2 = vertsOnPerimeter[i % vertsOnPerimeterSize];
      insertSegment(vert, vert2, true, V, operations);
    }
    tmpCounter = 1;
    for (int i = 1; i < vertsOnPerimeterSize; ++tmpCounter)
    {
      int vert = vertsOnPerimeter[i];

      adt[vert].erase(facesCounterclockwise[tmpCounter % facesCounterclockwiseSize]);

      i += 2;
      int vert2 = vertsOnPerimeter[i % vertsOnPerimeterSize];
      removeSegment(vert, vert2, true, V, delOperations);
    }

    // Complete the update of the priority queue according the new configuration
    for (Segment segment : segments)
    {
      delOperations.emplace(segment);

      // Convert diagonals in edges and vice versa
      segment.length = segment.isDiagonal ? segment.length * 2 : segment.length / 2;
      segment.isDiagonal = !segment.isDiagonal;
      operations.emplace(segment);
    }

    // Searching for possible doublets
    bool result = true;
    for (int vertDoublet : vertsForDoublets)
    {
      int valence = 0;
      for (int face : adt[vertDoublet])
      {
        if (tombStonesF[face]) { ++valence; }
      }

      if (valence == 2) // Doublet found
      {
        if (borderVerts.count(vertDoublet)) { continue; }

        if (vertDoublet == vertForNextCollapse)
        {
          result = false; // Disable the next possible diagonal collapse
        }
        remove_doublet(vertDoublet, V, normals, tombStonesV, borderVerts, F, 
          tombStonesF, aliveFaces, adt, operations, delOperations);
      }
    }

    // Compute and set the new positions of vertices adjacent to the vertex rotated
    if (tombStonesV[rotationVert])
    {
      for (int face : adt[rotationVert])
      {
        if (tombStonesF[face])
        {
          f = F.row(face);
          for (int i = 0; i < 4; ++i)
          {
            if (f[i] == rotationVert)
            {
              // Next vertex
              new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, f[(i + 1) % 4],
                operations, delOperations);
              // Opposite vertex
              new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, f[(i + 2) % 4],
                operations, delOperations);
            }
          }
        }
      }
    }

    return result;
  }

  return true;
}

bool try_edge_rotation(int vert1, int vert2, Eigen::MatrixXd& V, Eigen::MatrixXd& normals,
  std::vector<bool>& tombStonesV, std::set<int>& borderVerts, Eigen::MatrixXi& F, 
  std::vector<bool>& tombStonesF, int& aliveFaces, std::vector<std::set<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations,
  bool forceRotation = false)
{
  int edgeFirst = vert1, edgeSecond = vert2;

  if (borderVerts.count(edgeFirst) && borderVerts.count(edgeSecond))
  {
    return true; // Both edge's endpoints belong to the mesh border
  }

  // Find the two faces involved in the edge rotation
  int face1 = -1, face2 = -1;
  for (int f1 : adt[edgeFirst])
  {
    if (tombStonesF[f1])
    {
      for (int f2 : adt[edgeSecond])
      {
        if (tombStonesF[f2])
        {
          if (f1 == f2)
          {
            if (face1 == -1)
            {
              face1 = f1;
              break;
            }

            face2 = f1;
            break;
          }
        }
      }
    }

    if (face2 != -1) { break; }
  }
  if (face1 == -1 || face2 == -1) { return false; }

  Eigen::RowVector4i f1 = F.row(face1), f2;
  for (int i = 0; i < 4; ++i)
  {
    if (f1[i] == edgeFirst)
    {
      if (f1[(i + 1) % 4] == edgeSecond)
      {
        f2 = F.row(face2);
        break;
      }

      f1 = F.row(face2);
      f2 = F.row(face1);

      int tmp = face1;
      face1 = face2;
      face2 = tmp;
      break;
    }
  }

  // Find the vertices belong to each of the two faces
  int oppositeVertF1 = -1, oppositeVertF2 = -1,
    oppositeNextVertF1 = -1, oppositeNextVertF2 = -1;
  for (int i = 0; i < 4; ++i)
  {
    if (f1[i] == edgeFirst)
    {
      oppositeVertF1 = f1[(i + 2) % 4];
      oppositeNextVertF1 = f1[(i + 3) % 4];
      break;
    }
  }
  for (int i = 0; i < 4; ++i)
  {
    if (f2[i] == edgeFirst)
    {
      oppositeVertF2 = f2[(i + 1) % 4];
      oppositeNextVertF2 = f2[(i + 2) % 4];
      break;
    }
  }
  if (oppositeNextVertF1 == -1 || oppositeNextVertF2 == -1) { return false; }

  // Edge rotation is profitable if it shortens the rotated edge and both the diagonals
  Eigen::RowVector3d rawEdgeFirst = V.row(edgeFirst), rawEdgeSecond = V.row(edgeSecond),
    oppVertF1 = V.row(oppositeVertF1), oppVertF2 = V.row(oppositeVertF2),
    oppNextVertF1 = V.row(oppositeNextVertF1), oppNextVertF2 = V.row(oppositeNextVertF2);

  double oldEdgeLength = squared_distance(rawEdgeFirst, rawEdgeSecond),
    oldDiag1F1Length = squared_distance(rawEdgeFirst, oppVertF1),
    oldDiag2F1Length = squared_distance(oppNextVertF1, rawEdgeSecond),
    oldDiag1F2Length = squared_distance(rawEdgeFirst, oppNextVertF2),
    oldDiag2F2Length = squared_distance(oppVertF2, rawEdgeSecond);

  double edgeClockLength = squared_distance(oppNextVertF1, oppNextVertF2),
    edgeCounterLength = squared_distance(oppVertF1, oppVertF2),
    newDiag1Length = squared_distance(oppVertF1, oppNextVertF2),
    newDiag2Length = squared_distance(oppNextVertF1, oppVertF2);

  bool rotation = false; // True if the edge rotation happens

  if (forceRotation || (edgeClockLength < oldEdgeLength
    && newDiag1Length < oldDiag1F1Length &&
    newDiag2Length < oldDiag2F2Length)) // Clockwise edge rotation
  {
    if (oppositeNextVertF1 == oppositeNextVertF2) { return true; }
    
    // Check if the edge rotation can produce a "bottleneck" in the quad mesh
    Eigen::RowVector4i f;
    for (int face : adt[oppositeNextVertF1])
    {
      if (tombStonesF[face])
      {
        f = F.row(face);
        for (int i = 0; i < 4; ++i)
        {
          if (f[i] == oppositeNextVertF1 && f[(i + 1) % 4] == oppositeNextVertF2)
          {
            return true; // Risck of bottleneck. Avoid edge rotation
          }
        }
      }
    }

    rotation = true;

    // Modify the edge
    removeSegment(edgeFirst, edgeSecond, false, V, delOperations);
    insertSegment(oppositeNextVertF1, oppositeNextVertF2, false, V, operations);

    // Modify the faces involved in the rotation
    F.row(face1) << oppositeNextVertF1, oppositeNextVertF2, edgeSecond, oppositeVertF1;
    F.row(face2) << oppositeNextVertF1, edgeFirst, oppositeVertF2, oppositeNextVertF2;

    // Modify the diagonals involved in the rotation
    removeSegment(oppositeVertF1, edgeFirst, true, V, delOperations);
    insertSegment(oppositeVertF1, oppositeNextVertF2, true, V, operations);

    removeSegment(oppositeVertF2, edgeSecond, true, V, delOperations);
    insertSegment(oppositeVertF2, oppositeNextVertF1, true, V, operations);

    // Update adt
    adt[edgeFirst].erase(face1);
    adt[edgeSecond].erase(face2);
    adt[oppositeNextVertF2].emplace(face1);
    adt[oppositeNextVertF1].emplace(face2);
  }
  else if (edgeCounterLength < oldEdgeLength && 
    newDiag1Length < oldDiag1F2Length &&
    newDiag2Length < oldDiag2F1Length) // Counterclockwise edge rotation
  {
    if (oppositeVertF1 == oppositeVertF2) { return true; }
    
    // Check if the edge rotation can produce a "bottleneck" in the quad mesh
    Eigen::RowVector4i f;
    for (int face : adt[oppositeVertF1])
    {
      if (tombStonesF[face])
      {
        f = F.row(face);
        for (int i = 0; i < 4; ++i)
        {
          if (f[i] == oppositeVertF1 && f[(i + 1) % 4] == oppositeVertF2)
          {
            return true; // Risck of bottleneck. Avoid edge rotation
          }
        }
      }
    }

    rotation = true;

    // Modify the edge
    removeSegment(edgeFirst, edgeSecond, false, V, delOperations);
    insertSegment(oppositeVertF1, oppositeVertF2, false, V, operations);

    // Modify the faces involved in the rotation
    F.row(face1) << oppositeNextVertF1, edgeFirst, oppositeVertF2, oppositeVertF1;
    F.row(face2) << oppositeVertF2, oppositeNextVertF2, edgeSecond, oppositeVertF1;

    // Modify the diagonals involved in the rotation
    removeSegment(oppositeNextVertF1, edgeSecond, true, V, delOperations);
    insertSegment(oppositeNextVertF1, oppositeVertF2, true, V, operations);

    removeSegment(oppositeNextVertF2, edgeFirst, true, V, delOperations);
    insertSegment(oppositeNextVertF2, oppositeVertF1, true, V, operations);

    // Update adt
    adt[edgeFirst].erase(face2);
    adt[edgeSecond].erase(face1);
    adt[oppositeVertF2].emplace(face1);
    adt[oppositeVertF1].emplace(face2);
  }

  if (rotation)
  {
    // Searching for the two possible doublets created after the edge rotation
    int v1Valence = 0, v2Valence = 0;
    for (int face : adt[edgeFirst])
    {
      if (tombStonesF[face]) { ++v1Valence; }
    }
    if (v1Valence == 2) // Doublet found
    {
      remove_doublet(edgeFirst, V, normals, tombStonesV, borderVerts, F, tombStonesF, 
        aliveFaces, adt, operations, delOperations);
    }

    for (int face : adt[edgeSecond])
    {
      if (tombStonesF[face]) { ++v2Valence; }
    }
    if (v2Valence == 2) // Doublet found
    {
      remove_doublet(edgeSecond, V, normals, tombStonesV, borderVerts, F, tombStonesF, 
        aliveFaces, adt, operations, delOperations);
    }

    // Compute and set the new positions of vertices adjacent to the edge rotated
    new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, edgeFirst,
      operations, delOperations);
    new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, edgeSecond,
      operations, delOperations);
    new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, oppositeVertF1,
      operations, delOperations);
    new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, oppositeVertF2,
      operations, delOperations);
    new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, oppositeNextVertF1,
      operations, delOperations);
    new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, oppositeNextVertF2,
      operations, delOperations);
  }

  return true;
}

// Edge rotation and vertex rotation
bool optimize_quad_mesh(std::vector<Segment>& involvedEdges, int involvedVertex,
  Eigen::MatrixXd& V, Eigen::MatrixXd& normals, std::vector<bool>& tombStonesV, std::set<int>& borderVerts,
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, int& aliveFaces, 
  std::vector<std::set<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  for (Segment& edge : involvedEdges)
  {
    if (tombStonesV[edge.v2] && tombStonesV[edge.v1])
    {
      if (!try_edge_rotation(edge.v1, edge.v2, V, normals, tombStonesV, borderVerts, 
        F, tombStonesF, aliveFaces, adt, operations, delOperations))
      {
        return false;
      }
    }
  }

  if (tombStonesV[involvedVertex])
  {
    try_vertex_rotation(involvedVertex, V, normals, tombStonesV, borderVerts, F, 
      tombStonesF, aliveFaces, adt, operations, delOperations);
  }

  return true;
}

bool diagonal_collapse(const Segment& diag, Eigen::MatrixXd& V, Eigen::MatrixXd& normals,
  std::vector<bool>& tombStonesV, std::set<int>& borderVerts, Eigen::MatrixXi& F, 
  std::vector<bool>& tombStonesF, int& aliveFaces, std::vector<std::set<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  int vertexToBeMaintained = diag.v1, vertexToBeRemoved = diag.v2;

  // Check if the diagonal's endpoints (both or only one) belong to the mesh border
  const bool isOnBorder = borderVerts.count(vertexToBeMaintained) && 
    borderVerts.count(vertexToBeRemoved);
  
  const bool isHalfOnBorder = isOnBorder ? false :
    borderVerts.count(vertexToBeMaintained) || borderVerts.count(vertexToBeRemoved);

  if (isHalfOnBorder)
  {
    if (borderVerts.count(vertexToBeRemoved))
    {
      // Choose the vertex to be maintained so that it's the one on the mesh border
      int tmp = vertexToBeMaintained;
      vertexToBeMaintained = vertexToBeRemoved;
      vertexToBeRemoved = tmp;
    }
  }

  Eigen::RowVector3d rawVertToBeMaintained = V.row(vertexToBeMaintained);
  Eigen::RowVector3d rawVertToBeRemoved = V.row(vertexToBeRemoved);
  
  Eigen::RowVector4i f;
  if (!isOnBorder && !isHalfOnBorder)
  {
    // Check if the diagonal collapse can produce a "bottleneck" in the quad mesh
    std::vector<int> verts1, verts2;
    for (int face : adt[vertexToBeMaintained])
    {
      if (tombStonesF[face])
      {
        f = F.row(face);
        for (int i = 0; i < 4; ++i)
        {
          if (f[i] == vertexToBeMaintained && f[(i + 2) % 4] != vertexToBeRemoved)
          {
            verts1.emplace_back(f[(i + 1) % 4]);
            break;
          }
        }
      }
    }
    for (int face : adt[vertexToBeRemoved])
    {
      if (tombStonesF[face])
      {
        f = F.row(face);
        for (int i = 0; i < 4; ++i)
        {
          if (f[i] == vertexToBeRemoved && f[(i + 2) % 4] != vertexToBeMaintained)
          {
            verts2.emplace_back(f[(i + 1) % 4]);
            break;
          }
        }
      }
    }
    for (int v1 : verts1)
    {
      for (int v2 : verts2)
      {
        if (v1 == v2)
        {
          // Risck of bottleneck
          Segment e = Segment{ vertexToBeMaintained, v1, 2 *
            squared_distance(rawVertToBeMaintained, V.row(v1)), false };

          try_edge_rotation(e.v1, e.v2, V, normals, tombStonesV, borderVerts, F, tombStonesF,
            aliveFaces, adt, operations, delOperations, true);

          if (tombStonesV[vertexToBeRemoved] && tombStonesV[v1])
          {
            e.v1 = vertexToBeRemoved;
            e.length = 2 * squared_distance(rawVertToBeRemoved, V.row(v1));

            try_edge_rotation(e.v1, e.v2, V, normals, tombStonesV, borderVerts, F, tombStonesF,
              aliveFaces, adt, operations, delOperations, true);
          }

          if (!tombStonesV[vertexToBeMaintained] ||
            !tombStonesV[vertexToBeRemoved])
          {
            return true;
          }
        }
      }
    }
  }

  // Check if there is a hole next to the diagonal collapse
  std::vector<int> holeVerts{ vertexToBeMaintained };
  if (isOnBorder)
  {
    for (int i = 0; i < 4; ++i)
    {
      bool endLoop = false;
      int size = holeVerts.size();
      int currVert = holeVerts.back();
      for (int face : adt[currVert])
      {        
        if (tombStonesF[face])
        {
          f = F.row(face);
          for (int j = 0; j < 4; ++j)
          {
            if (f[j] == currVert)
            {
              int nextVert = f[(j + 1) % 4];

              if (nextVert == vertexToBeMaintained)
              {
                endLoop = true;
                break;
              }

              if (borderVerts.count(nextVert))
              {
                holeVerts.emplace_back(nextVert);
              }
              break;
            }
          }
          if (endLoop) { break; }
          if (size != holeVerts.size()) { break; }
        }
      }

      if (endLoop && (holeVerts.size() == 4 || holeVerts.size() == 3)) { break; }
      if (size == holeVerts.size() || (i == 3 && !endLoop))
      {
        holeVerts.clear();
        break;
      }
    }
  }

  // Find the face to be collapsed
  int faceToBeCollapsed = -1;
  for (int face : adt[vertexToBeMaintained])
  {
    if (tombStonesF[face])
    {
      f = F.row(face);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == vertexToBeMaintained)
        {
          if (vertexToBeRemoved == f[(i + 2) % 4])
          {
            faceToBeCollapsed = face;

            if (borderVerts.count(f[0]) && borderVerts.count(f[1]) &&
              borderVerts.count(f[2]) && borderVerts.count(f[3]))
            {
              tombStonesF[faceToBeCollapsed] = false;
              --aliveFaces;
              removeSegment(f[0], f[2], true, V, delOperations);
              removeSegment(f[1], f[3], true, V, delOperations);

              return true;
            }
          }
          break;
        }
      }
      if (faceToBeCollapsed != -1) { break; }
    }
  }
  if (faceToBeCollapsed == -1) { return false; }
  
  int newVertOnBorder = -1, oppositeBorderVert = -1;
  if (isOnBorder)
  {
    // Find the other vertex on the mesh border
    for (int face : adt[vertexToBeRemoved])
    {
      if (tombStonesF[face])
      {
        f = F.row(face);
        for (int i = 0; i < 4; ++i)
        {
          if (f[i] == vertexToBeRemoved)
          {
            if (f[(i + 2) % 4] != vertexToBeMaintained)
            {
              break;
            }

            // We are inside the face to be collapsed
            int nextVert = f[(i + 1) % 4], oppNextVert = f[(i + 3) % 4];
            if (!borderVerts.count(oppNextVert))
            {
              newVertOnBorder = nextVert;
              oppositeBorderVert = oppNextVert;
              break;
            }
            if (!borderVerts.count(nextVert))
            {
              newVertOnBorder = oppNextVert;
              oppositeBorderVert = nextVert;
              break;
            }
            
            // Both verts on the mesh border
            int nfaces1 = 0, nfaces2 = 0;
            for (int face1 : adt[nextVert])
            {
              if (tombStonesF[face1]) { ++nfaces1; }
            }
            for (int face2 : adt[oppNextVert])
            {
              if (tombStonesF[face2]) { ++nfaces2; }
            }
            newVertOnBorder = nfaces1 < nfaces2 ? nextVert : oppNextVert;
            oppositeBorderVert = nfaces1 < nfaces2 ? oppNextVert : nextVert;

            break;
          }
        }
      }
      if (newVertOnBorder != -1) { break; }
    }
    if (newVertOnBorder == -1 || oppositeBorderVert == 1) { return false; }

    int holeValence = holeVerts.size();
    if (holeValence == 4) // Quadrilateral hole
    {
      removeSegment(newVertOnBorder, oppositeBorderVert, true, V, delOperations);

      int oppV = -1;
      for (int i = 0; i < 4; ++i)
      {
        if (holeVerts[i] == newVertOnBorder)
        {
          oppV = holeVerts[(i + 2) % 4];
        }
      }
      if (oppV == -1) { return false; }

      borderVerts.erase(vertexToBeMaintained);
      borderVerts.erase(vertexToBeRemoved);
      borderVerts.erase(oppV);

      insertSegment(oppositeBorderVert, oppV, true, V, operations);

      Eigen::RowVector4i rawFaceToBeCollapsed = F.row(faceToBeCollapsed);
      for (int i = 0; i < 4; ++i)
      {
        if (rawFaceToBeCollapsed[i] == newVertOnBorder)
        {
          F.row(faceToBeCollapsed)[i] = oppV;
          break;
        }
      }

      adt[oppV].emplace(faceToBeCollapsed);
      tombStonesV[newVertOnBorder] = false;

      insertSegment(vertexToBeMaintained, oppV, false, V, operations);
      insertSegment(vertexToBeRemoved, oppV, false, V, operations);

      new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, oppV,
        operations, delOperations);

      return true;
    }

    if (holeValence == 3) // Triangular hole
    {
      std::cout << "\nTriangular hole detected in the quad mesh. It is not allowed.\n";
      return false;
      /*if (borderVerts.count(oppositeBorderVert))
      {
        // All the vertices of the face to be collapsed belong to the border
        return false; // It is not allowed
      }
      else
      {
        for (int i = 0; i < 4; ++i)
        {
          if (f[i] == oppositeBorderVert)
          {
            try_edge_rotation(Segment{ oppositeBorderVert, f[(i + 3) % 4],
               2 * squared_distance(V.row(oppositeBorderVert), V.row(f[(i + 3) % 4])), false },
              V, tombStonesV, borderVerts, F, tombStonesF, aliveFaces, adt, operations, delOperations, true);

            if (tombStonesV[oppositeBorderVert])
            {
              try_vertex_rotation(oppositeBorderVert, V, tombStonesV, borderVerts, F,
                tombStonesF, aliveFaces, adt, operations, delOperations, -2);
            }

            break;
          }
        }

        
        try_vertex_rotation(oppositeBorderVert, V, tombStonesV, borderVerts, F,
          tombStonesF, aliveFaces, adt, operations, delOperations, -2); TODO

        return true;
      }*/
    }
  }

  if (!isHalfOnBorder)
  {
    // Modify the faces and remove the edges/diagonals involved in the collapse
    for (int face : adt[vertexToBeMaintained])
    {
      if (tombStonesF[face])
      {
        f = F.row(face);
        for (int i = 0; i < 4; ++i)
        {
          if (f[i] == vertexToBeMaintained)
          {
            int nextVert = f[(i + 1) % 4], oppositeVert = f[(i + 2) % 4];

            removeSegment(vertexToBeMaintained, oppositeVert, true, V, delOperations);

            if (isOnBorder)
            {
              if (oppositeVert != vertexToBeRemoved)
              {
                insertSegment(newVertOnBorder, oppositeVert, true, V, operations);

                if (nextVert != oppositeBorderVert && !borderVerts.count(nextVert))
                {
                  insertSegment(newVertOnBorder, nextVert, false, V, operations);
                }

                adt[newVertOnBorder].emplace(face);
                F.row(face)[i] = newVertOnBorder;
              }

              if (!borderVerts.count(nextVert))
              {
                removeSegment(vertexToBeMaintained, nextVert, false, V, delOperations);
              }
            }
            else
            {
              removeSegment(vertexToBeMaintained, nextVert, false, V, delOperations);
            }

            break;
          }
        }
      }
    }
  }

  // Check these verts (after the collapse) for the possible presence of doublets
  int vert1ForDoublet = -1, vert2ForDoublet = -1;

  for (int face : adt[vertexToBeRemoved])
  {
    if (tombStonesF[face])
    {
      f = F.row(face);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == vertexToBeRemoved)
        {
          int nextVert = f[(i + 1) % 4], oppositeVert = f[(i + 2) % 4];
          Eigen::RowVector3d rawNextVert = V.row(nextVert);

          if (isOnBorder)
          {
            if (oppositeVert != vertexToBeMaintained)
            {
              removeSegment(vertexToBeRemoved, oppositeVert, true, V, delOperations);

              insertSegment(newVertOnBorder, oppositeVert, true, V, operations);

              if (nextVert != oppositeBorderVert && !borderVerts.count(nextVert))
              {
                insertSegment(newVertOnBorder, nextVert, false, V, operations);
              }

              F.row(face)[i] = newVertOnBorder;
              adt[newVertOnBorder].emplace(face);
            }
            else
            {
              int oppositeNextVert = f[(i + 3) % 4];
              Eigen::RowVector3d rawOppNextVert = V.row(oppositeNextVert);
              
              removeSegment(oppositeNextVert, nextVert, true, V, delOperations);

              if (!borderVerts.count(oppositeNextVert) || !borderVerts.count(nextVert))
              {
                insertSegment(oppositeNextVert, nextVert, false, V, operations);
              }
            }

            if (!borderVerts.count(nextVert))
            {
              removeSegment(vertexToBeRemoved, nextVert, false, V, delOperations);
            }
          }
          else
          {
            removeSegment(vertexToBeRemoved, nextVert, false, V, delOperations);

            if (oppositeVert != vertexToBeMaintained)
            {
              removeSegment(vertexToBeRemoved, oppositeVert, true, V, delOperations);

              if (isHalfOnBorder)
              {
                insertSegment(vertexToBeMaintained, oppositeVert, true, V, operations);

                Eigen::RowVector4i rawFaceCollapsed = F.row(faceToBeCollapsed);
                for (int i = 0; i < 4; ++i)
                {
                  if (rawFaceCollapsed[i] == vertexToBeRemoved)
                  {
                    vert1ForDoublet = rawFaceCollapsed[(i + 1) % 4];
                    vert2ForDoublet = rawFaceCollapsed[(i + 3) % 4];
                    break;
                  }
                }

                if (nextVert != vert1ForDoublet && nextVert != vert2ForDoublet && 
                  !borderVerts.count(nextVert))
                {
                  insertSegment(vertexToBeMaintained, nextVert, false, V, operations);
                }
              }

              F.row(face)[i] = vertexToBeMaintained;
              adt[vertexToBeMaintained].emplace(face);
            }
            else
            {
              int oppositeNextVert = f[(i + 3) % 4];
              removeSegment(oppositeNextVert, nextVert, true, V, delOperations);

              if (isHalfOnBorder)
              {
                removeSegment(vertexToBeRemoved, oppositeVert, true, V, delOperations);
              }

              vert1ForDoublet = nextVert;
              vert2ForDoublet = oppositeNextVert;
            }
          }

          break;
        }
      }
    }
  }

  // Remove the vertices collapsed
  if (isOnBorder)
  {
    tombStonesV[vertexToBeMaintained] = false;
  }
  tombStonesV[vertexToBeRemoved] = false;

  // Remove the face collapsed
  tombStonesF[faceToBeCollapsed] = false;
  --aliveFaces;

  // Calculate and set the new position of the vertices adjacent to the vertex remained
  if (isOnBorder)
  {
    vertexToBeMaintained = newVertOnBorder;
  }
  else if (!isHalfOnBorder)
  {
    // Calculate and set the new vertex's position
    Eigen::RowVector3d midpoint = 0.5 * rawVertToBeMaintained + 0.5 * rawVertToBeRemoved;
    new_vertex_pos(V, normals, borderVerts, F, tombStonesF, adt, vertexToBeMaintained, 
      F.row(faceToBeCollapsed), midpoint, operations, delOperations, true);
  }

  for (int face : adt[vertexToBeMaintained])
  {
    if (tombStonesF[face])
    {
      f = F.row(face);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == vertexToBeMaintained)
        {
          int nextVert = f[(i + 1) % 4], oppositeVert = f[(i + 2) % 4];

          if (isOnBorder)
          { // TODO better with of without?
            /*if (!borderVerts.count(nextVert))
            {
              new_vertex_pos2(V, tombStonesV, borderVerts, F, tombStonesF, adt, nextVert,
                operations, delOperations);
            }
            if (!borderVerts.count(oppositeVert))
            {
              new_vertex_pos2(V, tombStonesV, borderVerts, F, tombStonesF, adt, oppositeVert,
                operations, delOperations);
            }*/
          }
          else
          {
            new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, nextVert,
              operations, delOperations);
            new_vertex_pos2(V, normals, tombStonesV, borderVerts, F, tombStonesF, adt, oppositeVert,
              operations, delOperations);
          }

          break;
        }
      }
    }
  }

  // Searching for possible doublets
  if (isOnBorder)
  {
    if (!borderVerts.count(oppositeBorderVert))
    {
      int valence = 0;
      for (int face : adt[oppositeBorderVert])
      {
        if (tombStonesF[face]) { ++valence; }
      }
      if (valence == 2) // Doublet found
      {
        remove_doublet(oppositeBorderVert, V, normals, tombStonesV, borderVerts, F,
          tombStonesF, aliveFaces, adt, operations, delOperations);
      }
    }
  }
  else
  {
    int v1Valence = 0, v2Valence = 0;
    for (int face : adt[vert1ForDoublet])
    {
      if (tombStonesF[face]) { ++v1Valence; }
    }
    if (v1Valence == 2) // Doublet found
    {
      remove_doublet(vert1ForDoublet, V, normals, tombStonesV, borderVerts, F,
        tombStonesF, aliveFaces, adt, operations, delOperations);
    }

    for (int face : adt[vert2ForDoublet])
    {
      if (tombStonesF[face]) { ++v2Valence; }
    }
    if (v2Valence == 2) // Doublet found
    {
      remove_doublet(vert2ForDoublet, V, normals, tombStonesV, borderVerts, F,
        tombStonesF, aliveFaces, adt, operations, delOperations);
    }
  }

  return true;
}

bool edge_collapse(Segment& edge, Eigen::MatrixXd& V, Eigen::MatrixXd& normals, std::vector<bool>& tombStonesV,
  std::set<int>& borderVerts, Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, 
  int& aliveFaces, std::vector<std::set<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  int vert1 = edge.v1, vert2 = edge.v2;
  
  if (borderVerts.count(vert1) && borderVerts.count(vert2))
  {
    return false; // Both edge's endpoints belong to the border. It shouldn't happen
  }
  
  if (borderVerts.count(vert1))
  {
    int tmp = vert1;
    vert1 = vert2;
    vert2 = tmp;
  }
  
  if (!try_vertex_rotation(vert1, V, normals, tombStonesV, borderVerts, F, tombStonesF,
    aliveFaces, adt, operations, delOperations, vert2)) // Force rotation
  {
    // The next diagonal collapse should not be done if the checking for doublets
    // during vertex rotation removed the vertex to be used by the next collapse
    return true;
  }

  if (tombStonesV[vert1])
  {
    Segment diag = edge;
    diag.length /= 2;
    diag.isDiagonal = true;
    if (!diagonal_collapse(diag, V, normals, tombStonesV, borderVerts, F, tombStonesF, 
      aliveFaces, adt, operations, delOperations))
    {
      return false;
    }
  }

  return true;
}

// Edge collapse or diagonal collapse
bool coarsen_quad_mesh(Eigen::MatrixXd& V, Eigen::MatrixXd& normals, std::vector<bool>& tombStonesV,
  std::set<int>& borderVerts, Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, 
  int& aliveFaces, std::vector<std::set<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  if (operations.empty()) { return false; }

  // Select the next operation to execute
  // Diagonal = diagonal collapse, Edge = edge collapse
  while (!delOperations.empty() && operations.top() == delOperations.top())
  {
    // Remove the operation
    operations.pop();
    delOperations.pop();
  }
  Segment nextOperation = operations.top(); // Next operation to perform

  // Quad mesh coarsening
  if (nextOperation.isDiagonal)
  {
    // Diagonal collapse
    if (!diagonal_collapse(nextOperation, V, normals, tombStonesV, borderVerts, F, tombStonesF, 
      aliveFaces, adt, operations, delOperations))
    {
      return false;
    }
  }
  else
  {
    // Edge collapse
    if (!edge_collapse(nextOperation, V, normals, tombStonesV, borderVerts, F, tombStonesF, 
      aliveFaces, adt, operations, delOperations))
    {
      return false;
    }
  }

  // Final local optimization
  int survivedVertex = nextOperation.v1;
  if (tombStonesV[survivedVertex])
  {
    Eigen::RowVector3d rawSurvivedVertex = V.row(survivedVertex);
    Eigen::RowVector4i f;
    std::vector<Segment> edgesFromVertex;
    for (int face : adt[survivedVertex])
    {
      if (tombStonesF[face])
      {
        f = F.row(face);
        for (int i = 0; i < 4; ++i)
        {
          if (f[i] == survivedVertex)
          {
            int nextVert = f[(i + 1) % 4];
            edgesFromVertex.emplace_back(Segment{ survivedVertex, nextVert,
              2 * squared_distance(rawSurvivedVertex, V.row(nextVert)), false });

            break;
          }
        }
      }
    }
    optimize_quad_mesh(edgesFromVertex, survivedVertex, V, normals, tombStonesV,
      borderVerts, F, tombStonesF, aliveFaces, adt, operations, delOperations);
  }

  return true;
}

bool start_simplification(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int finalNumberOfFaces)
{
  /*int minimumNumberOfFaces = 15;
  if (finalNumberOfFaces < minimumNumberOfFaces)
  {
    std::cout << "The minimum number of faces is " << minimumNumberOfFaces <<
      ". Use a larger number please.\n";
    return true;
  } TODO */

  // Used to navigate the mesh
  // For each vertex, store the indices of the adjacent faces
  int vSize = V.rows();
  std::vector<std::set<int>> adt(vSize);

  // Store a normal per vertex
  Eigen::MatrixXd normals;

  // Used to indicate the faces and vertices already deleted
  int aliveFaces = F.rows(); // Store the number of alive faces inside tombStonesF
  std::vector<bool> tombStonesF(aliveFaces, true),
    tombStonesV(vSize, true); // All are alive at the beginning

  // Retrieve the quad mesh's egdes and diagonals
  typedef CompareTwoSegments CompareTwoEdges;
  std::set<Segment, CompareTwoEdges> edgesTmp; // Temporary edge set to avoid duplicates
  std::vector<Segment> diagonals;
  std::vector<Segment> edges;

  Eigen::RowVector4i face;
  for (int i = 0; i < aliveFaces; ++i) // For each quad face
  {
    face = F.row(i);
    int v0 = face[0], v1 = face[1], v2 = face[2], v3 = face[3];
    Eigen::RowVector3d rawV0 = V.row(v0), rawV1 = V.row(v1),
      rawV2 = V.row(v2), rawV3 = V.row(v3);

    diagonals.emplace_back(Segment{ v0, v2, squared_distance(rawV0, rawV2), true });
    diagonals.emplace_back(Segment{ v1, v3, squared_distance(rawV1, rawV3), true });

    edgesTmp.emplace_hint(edgesTmp.end(),
      Segment{ v0, v1, 2 * squared_distance(rawV0, rawV1), false });
    edgesTmp.emplace_hint(edgesTmp.end(),
      Segment{ v1, v2, 2 * squared_distance(rawV1, rawV2), false });
    edgesTmp.emplace_hint(edgesTmp.end(),
      Segment{ v2, v3, 2 * squared_distance(rawV2, rawV3), false });
    edgesTmp.emplace_hint(edgesTmp.end(),
      Segment{ v3, v0, 2 * squared_distance(rawV3, rawV0), false });

    adt[v0].emplace_hint(adt[v0].end(), i);
    adt[v1].emplace_hint(adt[v1].end(), i);
    adt[v2].emplace_hint(adt[v2].end(), i);
    adt[v3].emplace_hint(adt[v3].end(), i);
  }

  per_vertices_normals(V, F, normals); // Compute the normals per vertex

  // Find the vertices that are on the mesh borders
  std::set<int> borderVertices;
  for (int vert = 0; vert < vSize; ++vert)
  {
    if (!borderVertices.count(vert))
    {
      for (int f : adt[vert])
      {
        face = F.row(f);
        for (int i = 0; i < 4; ++i)
        {
          if (face[i] == vert)
          {
            if (adjacent_face(f, vert, face[(i + 1) % 4], adt, tombStonesF) == -1)
            {
              // No adjacent face => two border vertices = one border edge
              borderVertices.emplace_hint(borderVertices.end(), vert);
              borderVertices.emplace_hint(borderVertices.end(), face[(i + 1) % 4]);
              
              break;
            }
          }
        }
      }
    }
  }

  // Remove the edges that belong to the quad mesh borders
  if (borderVertices.empty())
  {
    std::vector<Segment> eds(edgesTmp.begin(), edgesTmp.end());
    edges = eds;
  }
  else
  {
    std::set<Segment, CompareTwoEdges>::iterator it;
    for (it = edgesTmp.begin(); it != edgesTmp.end(); ++it)
    {
      Segment e = *it;
      if (!borderVertices.count(e.v1) || !borderVertices.count(e.v2))
      {
        // The edge is not on mesh border
        edges.emplace_back(e);
      }
    }
  }

  // Concatenate all the segments
  std::vector<Segment> segments(edges.begin(), edges.end());
  segments.insert(segments.end(), diagonals.begin(), diagonals.end());

  // A priority queue containing the next best potential operation on the top
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>
    operations(segments.begin(), segments.end());

  // A priority queue containing the deleted operations to be ignored
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments> delOperations;

  // Try to coarsen the quad mesh
  while (aliveFaces > finalNumberOfFaces)
  {
    if (aliveFaces == 30573)
    {
      if (!coarsen_quad_mesh(V, normals, tombStonesV, borderVertices, F, tombStonesF,
        aliveFaces, adt, operations, delOperations))
      {
        return false;
      }
      std::cout << aliveFaces << "\n"; // TODO delete
      continue;
    }
    
    if (!coarsen_quad_mesh(V, normals, tombStonesV, borderVertices, F, tombStonesF,
      aliveFaces, adt, operations, delOperations))
    {
      return false;
    }
    std::cout << aliveFaces << "\n"; // TODO delete
  }

  // Set the deleted faces invisible // TODO delete?
  for (int i = 0; i < tombStonesF.size(); ++i)
  {
    if (!tombStonesF[i])
    {
      F.row(i) << 0, 0, 0, 0;
    }
  }

  //std::cout << "\n\n" << F << "\n\n"; //TODO

  /*std::vector<Segment> segs;
  while (!operations.empty())
  {
    if (!delOperations.empty() && operations.top() == delOperations.top())
    {
      operations.pop();
      delOperations.pop();
    }
    else
    {
      segs.push_back(operations.top());
      operations.pop();
    }
  } TODO */

  return true;
}

void quad_corners(Eigen::MatrixXd& V, Eigen::MatrixXi& F,
  Eigen::VectorXi& I, Eigen::VectorXi& C)
{
  I.resize(F.size());
  int c = 0, rowsF = F.rows();
  C.resize(rowsF + 1);
  C(0) = 0;

  Eigen::MatrixXd v;
  Eigen::MatrixXi f;
  v.conservativeResize(4, 3);
  f.conservativeResize(4, 3);

  std::vector<int> verts;
  for (int p = 0; p < rowsF; ++p)
  {
    // Check if the quad (2 triangle) is convex or concave and choose
    // the best two triangles (representing the quad) according to that
    Eigen::VectorXd a;
    Eigen::RowVector4i face = F.row(p);
    for (int i = 0; i < 4; ++i)
    {
      v.row(i) = V.row(face[i]);
    }
    f.row(0) << 0, 1, 3;
    f.row(1) << 1, 2, 3;
    f.row(2) << 0, 1, 2;
    f.row(3) << 0, 2, 3;
    igl::doublearea(v, f, a);

    if (a[2] + a[3] < a[0] + a[1])
    {
      verts = { face[0], face[1], face[2], face[3] };
    }
    else
    {
      verts = { face[1], face[2], face[3], face[0] };
    }

    for (int i = 0; i < 4; ++i)
    {
      I(c++) = verts[i];
    }
    C(p + 1) = C(p) + 4;
  }
  I.conservativeResize(c);
}

void draw_quad_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
  Eigen::MatrixXi f;

  // 1) Triangulate the polygonal mesh
  Eigen::VectorXi I, C, J;
  quad_corners(V, F, I, C);
  igl::polygons_to_triangles(I, C, f, J);

  // 2) For every quad, fit a plane and get the normal from that plane
  Eigen::MatrixXd N;
  per_quad_faces_normals(V, F, N);
  int nRows = N.rows();
  Eigen::MatrixXd fN(2 * nRows, 3);
  Eigen::RowVectorXd nRow;
  for (int i = 0; i < nRows; ++i)
  {
    nRow = N.row(i);
    fN.row(i * 2) = nRow;
    fN.row(i * 2 + 1) = nRow;
  };

  // 3) Render the triangles, and assign to each one the normal 
  // of the corresponding polygon, and use per-face rendering
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, f);
  viewer.data().set_normals(fN);

  // 4) Add as a line overlay the set of all edges of the original polygonal mesh
  Eigen::MatrixXi E;
  E.resize(F.size(), 2);
  E <<
    F.col(0), F.col(1),
    F.col(1), F.col(2),
    F.col(2), F.col(3),
    F.col(3), F.col(0);
  Eigen::RowVector3d black(0.0, 0.0, 0.0);
  viewer.data().set_edges(V, E, black);
  viewer.data().show_lines = false;

  viewer.launch();
}

#include <chrono> // TODO delete

int main(int argc, char* argv[])
{
  const std::string MESHES_DIR =
    "F:\\Users\\Nicolas\\Desktop\\TESI\\Quadrilateral extension to libigl\\libigl\\tutorial\\102_DrawMesh\\quad\\";

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  // Load a mesh
  //igl::readOFF(MESHES_DIR + "edge_rotate_surface.off", V, F);
  igl::readOBJ(MESHES_DIR + "armadillo.obj", V, F);

  std::cout << "Quad mesh coarsening in progress...\n\n";
  auto start_time = std::chrono::high_resolution_clock::now(); // TODO delete

  /*// Convert V and F into vectors in order to improve the performances
  int verticesCount = origV.rows();
  int facesCount = origF.rows();
  std::vector<std::vector<double>> V(verticesCount);
  std::vector<std::vector<int>> F(facesCount);
  for (int i = 0; i < verticesCount; ++i)
  {
    auto vertex = origV.row(i); // TODO try to use explicit raw type, and put it outside the loop
    for (int j = 0; j < 3; ++j)
    {
      V[i].emplace_back(vertex[j]);
    }
  }
  for (int i = 0; i < facesCount; ++i)
  {
    auto face = origF.row(i); // TODO try to use explicit raw type, and put it outside the loop
    for (int j = 0; j < 4; ++j)
    {
      F[i].emplace_back(face[j]);
    }
  }*/

  if (start_simplification(V, F, 3000))
  {
    auto end_time = std::chrono::high_resolution_clock::now(); // TODO delete
    std::cout << "\n" << (end_time - start_time) / std::chrono::milliseconds(1) << " milliseconds\n\n"; // TODO delete
    std::cout << "Quad mesh drawing in progress...\n\n";
    draw_quad_mesh(V, F);
  }
  else
  {
    std::cout << "\n\n" << "ERROR occurred during the quad mesh simplification\n\n";
  }

}
