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

typedef Segment Diagonal;
typedef Segment Edge;

double squared_distance(const Eigen::RowVector3d& vertex1, 
  const Eigen::RowVector3d& vertex2)
{
  double deltaX = vertex1[0] - vertex2[0],
    deltaY = vertex1[1] - vertex2[1],
    deltaZ = vertex1[2] - vertex2[2];
  return deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;
}

Eigen::RowVector3d new_vertex_pos(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  std::vector<bool>& tombStonesF, std::vector<std::vector<int>>& adt, int oldVert,
  const std::vector<int>& faceVertices)
{
  Eigen::RowVector3d centroid{ 0.0, 0.0, 0.0 };
  int size = 0;
  for (int face : adt[oldVert])
  {
    if (tombStonesF[face])
    {
      Eigen::RowVector4i f = F.row(face);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == oldVert)
        {
          centroid += V.row(f[(i + 1) % 4]); // Next vertex
          centroid += V.row(f[(i + 2) % 4]); // Opposite vertex
          size += 2;
          break;
        }
      }
    }
  }
  centroid /= size;

  Eigen::RowVector3d normal, pointOnPlane;
  Eigen::Matrix<double, 4, 3> v;
  v <<
    V.row(faceVertices[0]),
    V.row(faceVertices[1]),
    V.row(faceVertices[2]),
    V.row(faceVertices[3]);
  igl::fit_plane(v, normal, pointOnPlane);
  
  double proj = (centroid - pointOnPlane).dot(normal);
  return centroid - (proj * normal);
}

int adjacent_face(int face, int vert1, int vert2, 
  std::vector<std::vector<int>>& adt, std::vector<bool>& tombStonesF)
{
  for (int f1 : adt[vert1])
  {
    if (tombStonesF[f1] && f1 != face)
    {
      for (int f2 : adt[vert2])
      {
        if (tombStonesF[f2] && f2 != face)
        {
          if (f1 == f2) { return f1; }
        }
      }
    }
  }

  return -1; // Error
}

void remove_doublet(int vertexToBeRemoved, Eigen::MatrixXd& V, 
  std::vector<bool>& tombStonesV, Eigen::MatrixXi& F, 
  std::vector<bool>& tombStonesF, int& aliveFaces, std::vector<std::vector<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  if (aliveFaces < 4) { return; } // Too few faces to remove doublets
  
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

  Eigen::RowVector4i f = F.row(faceToBeRemoved);
  int endVert1 = -1, oppositeVert, endVert2;
  for (int i = 0; i < 4; ++i)
  {
    if (f[i] == vertexToBeRemoved)
    {
      endVert1 = f[(i + 1) % 4];
      oppositeVert = f[(i + 2) % 4];
      endVert2 = f[(i + 3) % 4];
      break;
    }
  }
  if (endVert1 == -1) { return; }

  // Partially update the priority queue of operations according the doublet deletion
  Eigen::RowVector3d rawEndVert1 = V.row(endVert1);
  Eigen::RowVector3d rawEndVert2 = V.row(endVert2);
  Eigen::RowVector3d rawOppVert = V.row(oppositeVert);

  Eigen::RowVector3d rawVertToBeRemoved = V.row(vertexToBeRemoved);
  delOperations.emplace(Edge{ vertexToBeRemoved, endVert1, 
    2 * squared_distance(rawVertToBeRemoved, rawEndVert1), false });
  delOperations.emplace(Edge{ vertexToBeRemoved, endVert2,
    2 * squared_distance(rawVertToBeRemoved, rawEndVert2), false });
  delOperations.emplace(Diagonal{ vertexToBeRemoved, oppositeVert,
    squared_distance(rawVertToBeRemoved, rawOppVert), true });
  delOperations.emplace(Diagonal{ endVert1, endVert2,
    squared_distance(rawEndVert1, rawEndVert2), true });
  
  // Remove one of the two faces adjacent to the doublet
  tombStonesF[faceToBeRemoved] = false;
  --aliveFaces;

  // Modify the remaining face and complete the update of the priority queue
  f = F.row(faceToBeMaintained);
  for (int i = 0; i < 4; ++i)
  {
    if (f[i] == vertexToBeRemoved)
    {
      F.row(faceToBeMaintained)[i] = oppositeVert;

      int v = F.row(faceToBeMaintained)[(i + 2) % 4];
      Eigen::RowVector3d rawV = V.row(v);
      delOperations.emplace(Diagonal{ v, vertexToBeRemoved,
        squared_distance(rawV, rawVertToBeRemoved), true });
      operations.emplace(Diagonal{ v, oppositeVert,
        squared_distance(rawV, rawOppVert), true });

      break;
    }
  }

  // Add the new face (the face mantained) to adt
  adt[oppositeVert].emplace_back(faceToBeMaintained);

  // Remove the vertex at the center of the doublet
  tombStonesV[vertexToBeRemoved] = false;

  // Check if other doublets are created in cascade at the two endpoints
  int v1Valence = 0, v2Valence = 0;
  for (int face : adt[endVert1])
  {
    if (tombStonesF[face]) { ++v1Valence; }
  } 
  if (v1Valence == 2) // Doublet found
  {
    remove_doublet(endVert1, V, tombStonesV, F, tombStonesF, aliveFaces, adt, 
      operations, delOperations);
  }

  for (int face : adt[endVert2])
  {
    if (tombStonesF[face]) { ++v2Valence; }
  }
  if (v2Valence == 2) // Doublet found
  {
    remove_doublet(endVert2, V, tombStonesV, F, tombStonesF, aliveFaces, adt, 
      operations, delOperations);
  }
}

bool try_edge_rotation(const Edge& edge, Eigen::MatrixXd& V, 
  std::vector<bool>& tombStonesV, Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, 
  int& aliveFaces, std::vector<std::vector<int>>& adt, 
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  int edgeFirst = edge.v1, edgeSecond = edge.v2;

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

  if (edgeClockLength < oldEdgeLength && newDiag1Length < oldDiag1F1Length &&
    newDiag2Length < oldDiag2F2Length) // Clockwise edge rotation
  {
    rotation = true;

    // Modify the edge
    delOperations.emplace(edge);
    operations.emplace(Edge{ oppositeNextVertF1, oppositeNextVertF2,
      2 * squared_distance(oppNextVertF1, oppNextVertF2), false });

    // Modify the faces involved in the rotation
    F.row(face1) << oppositeNextVertF1, oppositeNextVertF2, edgeSecond, oppositeVertF1;
    F.row(face2) << oppositeNextVertF1, edgeFirst, oppositeVertF2, oppositeNextVertF2;

    // Modify the diagonals involved in the rotation
    delOperations.emplace(Diagonal{ oppositeVertF1, edgeFirst,
      squared_distance(oppVertF1, rawEdgeFirst), true });
    operations.emplace(Diagonal{ oppositeVertF1, oppositeNextVertF2,
      squared_distance(oppVertF1, oppNextVertF2), true });

    delOperations.emplace(Diagonal{ oppositeVertF2, edgeSecond,
      squared_distance(oppVertF2, rawEdgeSecond), true });
    operations.emplace(Diagonal{ oppositeVertF2, oppositeNextVertF1,
      squared_distance(oppVertF2, oppNextVertF1), true });

    // Update adt
    adt[edgeFirst].erase(std::remove(adt[edgeFirst].begin(),
      adt[edgeFirst].end(), face1), adt[edgeFirst].end());
    adt[edgeSecond].erase(std::remove(adt[edgeSecond].begin(),
      adt[edgeSecond].end(), face2), adt[edgeSecond].end());
    adt[oppositeNextVertF2].emplace_back(face1);
    adt[oppositeNextVertF1].emplace_back(face2);
  }
  else if (edgeCounterLength < oldEdgeLength && newDiag1Length < oldDiag1F2Length &&
    newDiag2Length < oldDiag2F1Length) // Counterclockwise edge rotation
  {
    rotation = true;

    // Modify the edge
    delOperations.emplace(edge);
    operations.emplace(Edge{ oppositeVertF1, oppositeVertF2,
      2 * squared_distance(oppVertF1, oppVertF2), false });

    // Modify the faces involved in the rotation
    F.row(face1) << oppositeNextVertF1, edgeFirst, oppositeVertF2, oppositeVertF1;
    F.row(face2) << oppositeVertF2, oppositeNextVertF2, edgeSecond, oppositeVertF1;

    // Modify the diagonals involved in the rotation
    delOperations.emplace(Diagonal{ oppositeNextVertF1, edgeSecond,
      squared_distance(oppNextVertF1, rawEdgeSecond), true });
    operations.emplace(Diagonal{ oppositeNextVertF1, oppositeVertF2,
      squared_distance(oppNextVertF1, oppVertF2), true });

    delOperations.emplace(Diagonal{ oppositeNextVertF2, edgeFirst,
      squared_distance(oppNextVertF2, rawEdgeFirst), true });
    operations.emplace(Diagonal{ oppositeNextVertF2, oppositeVertF1,
      squared_distance(oppNextVertF2, oppVertF1), true });

    // Update adt
    adt[edgeFirst].erase(std::remove(adt[edgeFirst].begin(),
      adt[edgeFirst].end(), face2), adt[edgeFirst].end());
    adt[edgeSecond].erase(std::remove(adt[edgeSecond].begin(),
      adt[edgeSecond].end(), face1), adt[edgeSecond].end());
    adt[oppositeVertF2].emplace_back(face1);
    adt[oppositeVertF1].emplace_back(face2);
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
      remove_doublet(edgeFirst, V, tombStonesV, F, tombStonesF, aliveFaces, adt,
        operations, delOperations);
    }

    for (int face : adt[edgeSecond])
    {
      if (tombStonesF[face]) { ++v2Valence; }
    }
    if (v2Valence == 2) // Doublet found
    {
      remove_doublet(edgeSecond, V, tombStonesV, F, tombStonesF, aliveFaces, adt,
        operations, delOperations);
    }
  }

  return true;
}

bool try_vertex_rotation(int rotationVert, Eigen::MatrixXd& V, 
  std::vector<bool>& tombStonesV, Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, 
  int& aliveFaces, std::vector<std::vector<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations,
  int vertForNextCollapse = -1)
{
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

          segments.emplace_back(Edge{ rotationVert, nextVert, 2 * edgeLength, false });
          segments.emplace_back(Diagonal{ rotationVert, oppVert, diagLength, true });

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
      adt[vert].emplace_back(facesCounterclockwise[faceIndex]);

      i += 2;
      int vert2 = vertsOnPerimeter[i % vertsOnPerimeterSize];
      operations.emplace(Diagonal{ vert, vert2,
        squared_distance(V.row(vert), V.row(vert2)), true });
    }
    tmpCounter = 1;
    for (int i = 1; i < vertsOnPerimeterSize; ++tmpCounter)
    {
      int vert = vertsOnPerimeter[i];

      adt[vert].erase(std::remove(adt[vert].begin(), adt[vert].end(),
        facesCounterclockwise[tmpCounter % facesCounterclockwiseSize]), adt[vert].end());

      i += 2;
      int vert2 = vertsOnPerimeter[i % vertsOnPerimeterSize];
      delOperations.emplace(Diagonal{ vert, vert2, 
        squared_distance(V.row(vert), V.row(vert2)), true });
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
        if (vertDoublet == vertForNextCollapse)
        {
          result = false; // Disable the next possible diagonal collapse
        }
        remove_doublet(vertDoublet, V, tombStonesV, F, tombStonesF, aliveFaces, adt,
          operations, delOperations);
      }
    }
     
    return result;
  }

  return true;
}

// Edge rotation and vertex rotation
bool optimize_quad_mesh(std::vector<Edge>& involvedEdges, int involvedVertex,
  Eigen::MatrixXd& V, std::vector<bool>& tombStonesV, Eigen::MatrixXi& F, 
  std::vector<bool>& tombStonesF, int& aliveFaces, std::vector<std::vector<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  for (Edge& edge : involvedEdges)
  {
    if (!try_edge_rotation(edge, V, tombStonesV, F, tombStonesF, aliveFaces, adt,
      operations, delOperations))
    {
      return false;
    }
  }

  try_vertex_rotation(involvedVertex, V, tombStonesV, F, tombStonesF, aliveFaces, adt, 
      operations, delOperations);

  return true;
}

bool diagonal_collapse(const Diagonal& diag, Eigen::MatrixXd& V, std::vector<bool>& tombStonesV,
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, int& aliveFaces,
  std::vector<std::vector<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  int vertexToBeMaintained = diag.v1, vertexToBeRemoved = diag.v2;
  Eigen::RowVector3d rawVertToBeMaintained = V.row(vertexToBeMaintained);
  Eigen::RowVector3d rawVertToBeRemoved = V.row(vertexToBeRemoved);
  
  // The vertices to be checked (after the collapse) for the possible presence of doublets
  int vert1ForDoublet, vert2ForDoublet = -1;
  
  Eigen::RowVector4i f;
  // Modify the faces and remove the edges and the diagonals involved in the collapse
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

          delOperations.emplace(Diagonal{ vertexToBeMaintained, oppositeVert,
              squared_distance(rawVertToBeMaintained, V.row(oppositeVert)), true });
          delOperations.emplace(Edge{ vertexToBeMaintained, nextVert,
            2 * squared_distance(rawVertToBeMaintained, V.row(nextVert)), false });

          break;
        }
      }
    }
  }

  int faceToBeCollapsed = -1;
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

          delOperations.emplace(Edge{ vertexToBeRemoved, nextVert, 
            2 * squared_distance(rawVertToBeRemoved, rawNextVert), false });
          
          if (oppositeVert != vertexToBeMaintained)
          {
            delOperations.emplace(Diagonal{ vertexToBeRemoved, oppositeVert,
              squared_distance(rawVertToBeRemoved, V.row(oppositeVert)), true });
            
            // Modify one of the faces around the collapse
            F.row(face)[i] = vertexToBeMaintained;

            adt[vertexToBeMaintained].emplace_back(face);
          }
          else
          {
            faceToBeCollapsed = face;
            int oppositeNextVert = f[(i + 3) % 4];
            delOperations.emplace(Diagonal{ oppositeNextVert, nextVert,
              squared_distance(V.row(oppositeNextVert), rawNextVert), true});
            vert1ForDoublet = nextVert;
            vert2ForDoublet = oppositeNextVert;
          }

          break;
        }
      }
    }
  }
  if (faceToBeCollapsed == -1 || vert2ForDoublet == -1) { return false; }

  // Remove one of the two vertices collapsed
  tombStonesV[vertexToBeRemoved] = false;

  // Remove the face collapsed
  tombStonesF[faceToBeCollapsed] = false;
  --aliveFaces;

  // Calculate and set the new vertex's position
  f = F.row(faceToBeCollapsed);
  rawVertToBeMaintained = new_vertex_pos(V, F, tombStonesF, adt, 
    vertexToBeMaintained, { f[0], f[1], f[2], f[3] });
  V.row(vertexToBeMaintained) = rawVertToBeMaintained;

  // Update the priority queue of potential operations according the new configuration
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
          
          operations.emplace(Edge{ vertexToBeMaintained, nextVert,
              2 * squared_distance(rawVertToBeMaintained, V.row(nextVert)), false });
          operations.emplace(Diagonal{ vertexToBeMaintained, oppositeVert,
              squared_distance(rawVertToBeMaintained, V.row(oppositeVert)), true });
          
          break;
        }
      }
    }
  }

  // Searching for possible doublets
  int v1Valence = 0, v2Valence = 0;
  for (int face : adt[vert1ForDoublet])
  {
    if (tombStonesF[face]) { ++v1Valence; }
  }
  if (v1Valence == 2) // Doublet found
  {
    remove_doublet(vert1ForDoublet, V, tombStonesV, F, tombStonesF, aliveFaces, adt,
      operations, delOperations);
  }

  for (int face : adt[vert2ForDoublet])
  {
    if (tombStonesF[face]) { ++v2Valence; }
  }
  if (v2Valence == 2) // Doublet found
  {
    remove_doublet(vert2ForDoublet, V, tombStonesV, F, tombStonesF, aliveFaces, adt,
      operations, delOperations);
  }

  return true;
}

bool edge_collapse(const Edge& edge, Eigen::MatrixXd& V, std::vector<bool>& tombStonesV,
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, int& aliveFaces, 
  std::vector<std::vector<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  if (!try_vertex_rotation(edge.v1, V, tombStonesV, F, tombStonesF, aliveFaces, adt,
    operations, delOperations, edge.v2)) // Force rotation
  {
    // The next diagonal collapse should not be done if the checking for doublets
    // during vertex rotation removed the vertex to be used by the collapse
    return true;
  }

  Diagonal diag = edge;
  diag.length /= 2;
  diag.isDiagonal = true;
  if (!diagonal_collapse(diag, V, tombStonesV, F, tombStonesF, aliveFaces, adt,
    operations, delOperations))
  {
    return false;
  }

  return true;
}

// Edge collapse or diagonal collapse
bool coarsen_quad_mesh(Eigen::MatrixXd& V, std::vector<bool>& tombStonesV, 
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, int& aliveFaces,
  std::vector<std::vector<int>>& adt,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& operations,
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments>& delOperations)
{
  if (operations.empty()) { return false; }

  // Select the next operation to execute
  // Diagonal = diagonal collapse, Edge = edge collapse
  if (!delOperations.empty())
  { 
    while (operations.top() == delOperations.top())
    {
      // Remove the operation
      operations.pop();
      delOperations.pop();
    }
  }
  Segment nextOperation = operations.top(); // Next operation to perform

  // Quad mesh coarsening
  if (nextOperation.isDiagonal)
  {
    // Diagonal collapse
    if (!diagonal_collapse(nextOperation, V, tombStonesV, F, tombStonesF, aliveFaces,
      adt, operations, delOperations))
    {
      return false;
    }
  }
  else
  {
    // Edge collapse
    if (!edge_collapse(nextOperation, V, tombStonesV, F, tombStonesF, aliveFaces,
      adt, operations, delOperations))
    {
      return false;
    }
  }
  
  // Final local optimization
  std::vector<Edge> edgesFromVertex;
  int survivedVertex = nextOperation.v1;
  Eigen::RowVector3d rawSurvivedVertex = V.row(survivedVertex);
  Eigen::RowVector4i f;
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
          edgesFromVertex.emplace_back(Edge{ survivedVertex, nextVert,
            2 * squared_distance(rawSurvivedVertex, V.row(nextVert)), false });

          break;
        }
      }
    }
  }
  if (!optimize_quad_mesh(edgesFromVertex, survivedVertex, V, tombStonesV,
    F, tombStonesF, aliveFaces, adt, operations, delOperations))
  {
    return false;
  }
  
  return true;
}

bool start_simplification(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int finalNumberOfFaces)
{
  int minimumNumberOfFaces = 15;
  if (finalNumberOfFaces < minimumNumberOfFaces)
  {
    std::cout << "The minimum number of faces is " << minimumNumberOfFaces << 
      ". Use a larger number please.\n";
    return true;
  }
  
  // Used to navigate the mesh
  // For each vertex, store the indices of the adjacent faces
  int vSize = V.rows();
  std::vector<std::vector<int>> adt(vSize);

  // Used to indicate the faces and vertices already deleted
  int aliveFaces = F.rows(); // Store the number of alive faces inside tombStonesF
  std::vector<bool> tombStonesF(aliveFaces, true),
    tombStonesV(vSize, true); // All are alive at the beginning

  // Retrieve the quad mesh's egdes and diagonals
  typedef CompareTwoSegments CompareTwoEdges;
  std::set<Edge, CompareTwoEdges> edgesTmp; // Temporary set of edges to avoid duplicates
  std::vector<Diagonal> diagonals;
  
  Eigen::RowVector4i face;
  for (int i = 0; i < aliveFaces; ++i) // For each quad face
  {
    face = F.row(i);
    int v0 = face[0], v1 = face[1], v2 = face[2], v3 = face[3];
    Eigen::RowVector3d rawV0 = V.row(v0), rawV1 = V.row(v1), 
      rawV2 = V.row(v2), rawV3 = V.row(v3);

    diagonals.emplace_back(Diagonal{ v0, v2, squared_distance(rawV0, rawV2), true });
    diagonals.emplace_back(Diagonal{ v1, v3, squared_distance(rawV1, rawV3), true });

    edgesTmp.emplace_hint(edgesTmp.end(), 
      Edge{ v0, v1, 2 * squared_distance(rawV0, rawV1), false });
    edgesTmp.emplace_hint(edgesTmp.end(), 
      Edge{ v1, v2, 2 * squared_distance(rawV1, rawV2), false });
    edgesTmp.emplace_hint(edgesTmp.end(), 
      Edge{ v2, v3, 2 * squared_distance(rawV2, rawV3), false });
    edgesTmp.emplace_hint(edgesTmp.end(), 
      Edge{ v3, v0, 2 * squared_distance(rawV3, rawV0), false });
    
    adt[v0].emplace_back(i); adt[v1].emplace_back(i);
    adt[v2].emplace_back(i); adt[v3].emplace_back(i);
  }

  // Concatenate all the segments
  std::vector<Segment> segments(edgesTmp.begin(), edgesTmp.end());
  segments.insert(segments.end(), diagonals.begin(), diagonals.end());

  // A priority queue containing the next best potential operation on the top
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments> 
    operations(segments.begin(), segments.end());
  
  // A priority queue containing the deleted operations to be ignored
  std::priority_queue<Segment, std::vector<Segment>, CompareTwoSegments> delOperations;

  // Try to coarsen the quad mesh
  while (aliveFaces > finalNumberOfFaces)
  {
    if (!coarsen_quad_mesh(V, tombStonesV, F, tombStonesF, aliveFaces, adt, 
      operations, delOperations))
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

  return true;
}

void per_quad_face_normals(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& N)
{
  int facesCount = F.rows();
  N.resize(facesCount, 3);
  
  // Loop over the faces
#pragma omp parallel for if (facesCount > 1000)
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
  per_quad_face_normals(V, F, N);
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
  //igl::readOFF(MESHES_DIR + "edge_rotate_doublet.off", V, F);
  igl::readOBJ(MESHES_DIR + "quad_cubespikes.obj", V, F);

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

  if (start_simplification(V, F, 300))
  {
    auto end_time = std::chrono::high_resolution_clock::now(); // TODO delete
    std::cout << "\n" << (end_time - start_time) / std::chrono::milliseconds(1) << " milliseconds\n\n"; // TODO delete
    std::cout << "Quad mesh drawing in progress...\n\n";
    draw_quad_mesh(V, F);
  }
  else
  {
    std::cout << "\n\n" << "ERROR occured during the quad mesh simplification\n\n";
  }
}
