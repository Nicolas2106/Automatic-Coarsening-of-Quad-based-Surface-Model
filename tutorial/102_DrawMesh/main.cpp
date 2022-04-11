#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/fit_plane.h>
#include <igl/polygon_corners.h>
#include <igl/polygons_to_triangles.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_face_normals.h>
#include <igl/doublearea.h>
#include "tutorial_shared_path.h"

typedef std::pair<int, int> Edge;

bool operator==(const Edge e1, const Edge e2)
{
  int e10 = e1.first, e11 = e1.second, e20 = e2.first;
  return e10 == e20 ? e11 == e2.second : (e11 == e20 ? e10 == e2.second : false);
}

struct compareTwoEdges {
  bool operator()(Edge e1, Edge e2) const {
    int min1 = std::min(e1.first, e1.second), min2 = std::min(e2.first, e2.second);
    return min1 == min2 ? 
      std::max(e1.first, e1.second) < std::max(e2.first, e2.second) : min1 < min2;
  }
};

double squared_distance(Eigen::MatrixXd V, int vertex1, int vertex2)
{
  Eigen::RowVector3d point1 = V.row(vertex1), point2 = V.row(vertex2);;
  double deltaX = point1[0] - point2[0],
    deltaY = point1[1] - point2[1],
    deltaZ = point1[2] - point2[2];
  return deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;
}

Eigen::RowVector3d new_vertex_pos(Eigen::MatrixXd V, Eigen::MatrixXi& F, 
  std::vector<bool> tombStonesF, std::vector<std::vector<int>> adt, int oldVert,
  std::vector<int> faceVertices)
{
  std::vector<int> verticesForCentroid;
  for (int face : adt[oldVert])
  {
    if (tombStonesF[face])
    {
      Eigen::RowVector4i f = F.row(face);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == oldVert)
        {
          verticesForCentroid.push_back(f[(i + 1) % 4]); // Next vertex
          verticesForCentroid.push_back(f[(i + 2) % 4]); // Opposite vertex
        }
      }
    }
  }
  
  Eigen::RowVector3d centroid{ 0.0, 0.0, 0.0 };
  for (int vertex : verticesForCentroid)
  {
    centroid += V.row(vertex);
  }
  centroid /= verticesForCentroid.size();

  Eigen::RowVector3d normal, pointOnPlane;
  Eigen::MatrixXd v(4, 3);
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
  std::vector<std::vector<int>> adt, std::vector<bool> tombStonesF)
{
  std::vector<int> facesVert1 = adt[vert1];
  std::vector<int> facesVert2 = adt[vert2];
  
  for (int f1 : facesVert1)
  {
    if (tombStonesF[f1] && f1 != face)
    {
      for (int f2 : facesVert2)
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

void remove_doublet(Eigen::MatrixXd& V, std::vector<bool>& tombStonesV, 
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF,
  std::vector<std::vector<int>>& adt, int vertexToBeRemoved, 
  std::set<Edge, compareTwoEdges>& edges, 
  std::set<Edge, compareTwoEdges>& diagonals)
{
  int adtSize = 0, faceToBeRemoved = -1, faceToBeMantained = -1;
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
        faceToBeMantained = face;
      }
    }
  }
  if (adtSize != 2) { return; } // Incorrect input (No doublet)
  if (faceToBeRemoved == -1 || faceToBeMantained == -1) { return; }
  
  int facesRemained = 0;
  for (bool t : tombStonesF)
  {
    if (t) { ++facesRemained; }
  }
  if (facesRemained < 4) { return; } // Too few faces to remove doublets

  Eigen::RowVector4i f = F.row(faceToBeRemoved);
  int endVert1, endVert2, oppositeVert;
  for (int i = 0; i < 4; ++i)
  {
    if (f[i] == vertexToBeRemoved)
    {
      endVert1 = f[(i + 1) % 4];
      oppositeVert = f[(i + 2) % 4];
      endVert2 = f[(i + 3) % 4];
    }
  }

  // Remove the two edges that make the doublet
  edges.erase(Edge{ vertexToBeRemoved, endVert1 });
  edges.erase(Edge{ vertexToBeRemoved, endVert2 });

  // Remove the diagonal belong only to the face to be removed
  diagonals.erase(Edge{ vertexToBeRemoved, oppositeVert });
  
  // Remove one of the two faces adjacent to the doublet
  tombStonesF[faceToBeRemoved] = false;

  // Modify the remaining face and one of the diagonals belong to it
  f = F.row(faceToBeMantained);
  for (int i = 0; i < 4; ++i)
  {
    if (f[i] == vertexToBeRemoved)
    {
      F.row(faceToBeMantained)[i] = oppositeVert;

      int v = (i + 2) % 4;
      diagonals.erase(Edge{ F.row(faceToBeMantained)[v], vertexToBeRemoved });
      diagonals.emplace(Edge{ F.row(faceToBeMantained)[v], oppositeVert });
      break;
    }
  }

  // Add the new face (the face mantained) to adt
  adt[oppositeVert].emplace_back(faceToBeMantained);

  /*// Set a better position for the vertex remained
  Eigen::RowVector4i face = F.row(faceToBeRemoved);
  std::vector<int> faceVertices{ face[0], face[1], face[2], face[3] };
  V.row(oppositeVert) = new_vertex_pos(V, F, tombStonesF, adt, oppositeVert, faceVertices);*/

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
    remove_doublet(V, tombStonesV, F, tombStonesF, adt, endVert1,
      edges, diagonals);
  }

  for (int face : adt[endVert2])
  {
    if (tombStonesF[face]) { ++v2Valence; }
  }
  if (v2Valence == 2) // Doublet found
  {
    remove_doublet(V, tombStonesV, F, tombStonesF, adt, endVert2,
      edges, diagonals);
  }
}

bool try_edge_rotation(Edge edge, Eigen::MatrixXd& V, std::vector<bool>& tombStonesV, 
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, std::vector<std::vector<int>>& adt, 
  std::set<Edge, compareTwoEdges>& edges, std::set<Edge, compareTwoEdges>& diagonals)
{
  // Find the two faces involved in the edge rotation
  int face1 = -1, face2 = -1;
  for (int f1 : adt[edge.first])
  {
    if (tombStonesF[f1])
    {
      for (int f2 : adt[edge.second])
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
    if (f1[i] == edge.first)
    {
      if (f1[(i + 1) % 4] == edge.second)
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
    if (f1[i] == edge.first)
    {
      oppositeVertF1 = f1[(i + 2) % 4];
      oppositeNextVertF1 = f1[(i + 3) % 4];
      break;
    }
  }
  for (int i = 0; i < 4; ++i)
  {
    if (f2[i] == edge.first)
    {
      oppositeVertF2 = f2[(i + 1) % 4];
      oppositeNextVertF2 = f2[(i + 2) % 4];
      break;
    }
  }
  if (oppositeNextVertF1 == -1 || oppositeNextVertF2 == -1) { return false; }

  // Edge rotation is profitable if it shortens the rotated edge and both the diagonals
  double oldEdgeLength = squared_distance(V, edge.first, edge.second),
    oldDiag1F1Length = squared_distance(V, edge.first, oppositeVertF1),
    oldDiag2F1Length = squared_distance(V, oppositeNextVertF1, edge.second),
    oldDiag1F2Length = squared_distance(V, edge.first, oppositeNextVertF2),
    oldDiag2F2Length = squared_distance(V, oppositeVertF2, edge.second);

  double edgeClockLength = squared_distance(V, oppositeNextVertF1, oppositeNextVertF2),
    edgeCounterLength = squared_distance(V, oppositeVertF1, oppositeVertF2),
    newDiag1Length = squared_distance(V, oppositeVertF1, oppositeNextVertF2),
    newDiag2Length = squared_distance(V, oppositeNextVertF1, oppositeVertF2);

  bool rotation = false; // True if the edge rotation happens

  if (edgeClockLength < oldEdgeLength &&
    newDiag1Length < oldDiag1F1Length &&
    newDiag2Length < oldDiag2F2Length) // Clockwise edge rotation
  {
    rotation = true;

    // Modify the edge
    edges.erase(edge);
    edges.emplace(Edge{ oppositeNextVertF1, oppositeNextVertF2 });

    // Modify the faces involved in the rotation
    F.row(face1) << oppositeNextVertF1, oppositeNextVertF2, edge.second, oppositeVertF1;
    F.row(face2) << oppositeNextVertF1, edge.first, oppositeVertF2, oppositeNextVertF2;

    // Modify the diagonals involved in the rotation
    diagonals.erase(Edge{ oppositeVertF1, edge.first });
    diagonals.emplace(Edge{ oppositeVertF1, oppositeNextVertF2 });

    diagonals.erase(Edge{ oppositeVertF2, edge.second });
    diagonals.emplace(Edge{ oppositeVertF2, oppositeNextVertF1 });

    // Update adt
    adt[edge.first].erase(std::remove(adt[edge.first].begin(),
      adt[edge.first].end(), face1), adt[edge.first].end());
    adt[edge.second].erase(std::remove(adt[edge.second].begin(), 
      adt[edge.second].end(), face2), adt[edge.second].end());
    adt[oppositeNextVertF2].push_back(face1); // TODO emplace_back
    adt[oppositeNextVertF1].push_back(face2);
  }
  else if (edgeCounterLength < oldEdgeLength &&
    newDiag1Length < oldDiag1F2Length &&
    newDiag2Length < oldDiag2F1Length) // Clockwise edge rotation
  {
    rotation = true;

    // Modify the edge
    edges.erase(edge);
    edges.emplace(Edge{ oppositeVertF1, oppositeVertF2 });

    // Modify the faces involved in the rotation
    F.row(face1) << oppositeNextVertF1, edge.first, oppositeVertF2, oppositeVertF1;
    F.row(face2) << oppositeVertF2, oppositeNextVertF2, edge.second, oppositeVertF1;

    // Modify the diagonals involved in the rotation
    diagonals.erase(Edge{ oppositeNextVertF1, edge.second });
    diagonals.emplace(Edge{ oppositeNextVertF1, oppositeVertF2 });

    diagonals.erase(Edge{ oppositeNextVertF2, edge.first });
    diagonals.emplace(Edge{ oppositeNextVertF2, oppositeVertF1 });

    // Update adt
    adt[edge.first].erase(std::remove(adt[edge.first].begin(),
      adt[edge.first].end(), face2), adt[edge.first].end());
    adt[edge.second].erase(std::remove(adt[edge.second].begin(),
      adt[edge.second].end(), face1), adt[edge.second].end());
    adt[oppositeVertF2].push_back(face1); // TODO emplace_back
    adt[oppositeVertF1].push_back(face2);
  }
  
  if (rotation)
  {
    // Searching for the two possible doublets created after the edge rotation
    int v1Valence = 0, v2Valence = 0;
    for (int face : adt[edge.first])
    {
      if (tombStonesF[face]) { ++v1Valence; }
    }
    if (v1Valence == 2) // Doublet found
    {
      remove_doublet(V, tombStonesV, F, tombStonesF, adt, edge.first,
        edges, diagonals);
    }

    for (int face : adt[edge.second])
    {
      if (tombStonesF[face]) { ++v2Valence; }
    }
    if (v2Valence == 2) // Doublet found
    {
      remove_doublet(V, tombStonesV, F, tombStonesF, adt, edge.second,
        edges, diagonals);
    }
  }

  return true;
}

bool try_vertex_rotation(int rotationVert, Eigen::MatrixXd& V, std::vector<bool>& tombStonesV,
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, std::vector<std::vector<int>>& adt,
  std::set<Edge, compareTwoEdges>& edges, std::set<Edge, compareTwoEdges>& diagonals,
  const int vertForNextCollapse = -1, const bool forceRotation = false)
{
  double edgesSum = 0.0, diagonalsSum = 0.0;

  if (!forceRotation)
  {
    for (int face : adt[rotationVert])
    {
      if (tombStonesF[face])
      {
        Eigen::RowVector4i f = F.row(face);
        for (int i = 0; i < 4; ++i)
        {
          if (f[i] == rotationVert)
          {
            edgesSum += squared_distance(V, rotationVert, f[(i + 1) % 4]);
            diagonalsSum += squared_distance(V, rotationVert, f[(i + 2) % 4]);
          }
        }
      }
    }
  }

  // The vertices to be checked (after the rotation) for the possible presence of doublets
  std::vector<int> vertsForDoublet;

  // Check if the sum of the edge lengths overcomes the sum of the diagonals lengths
  if (forceRotation || edgesSum > diagonalsSum) // Vertex rotation
  {
    // Sort the faces around the vertex to be rotated counterclockwise
    std::vector<int> facesCounterclockwise;
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
      facesCounterclockwise.push_back(currFace);
      Eigen::RowVector4i f = F.row(currFace);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == rotationVert)
        {
          currFace = adjacent_face(currFace, rotationVert, f[(i + 3) % 4],
            adt, tombStonesF);
          break;
        }
      }
    } while (currFace != firstFace);

    // Rotate the edges and remove the diagonals involved in the vertex rotation
    std::vector<int> vertsOnPerimeter;
    for (int face : facesCounterclockwise)
    {
      Eigen::RowVector4i f = F.row(face);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == rotationVert)
        {
          int nextVert = f[(i + 1) % 4];
          int oppositeVert = f[(i + 2) % 4];
          int oppositeNextVert = f[(i + 3) % 4];
          vertsOnPerimeter.push_back(oppositeVert);
          vertsOnPerimeter.push_back(oppositeNextVert);

          edges.erase(Edge{ rotationVert, nextVert });
          edges.emplace(Edge{ rotationVert, oppositeVert });

          diagonals.erase(Edge{ rotationVert, oppositeVert });
          diagonals.erase(Edge{ nextVert, oppositeNextVert });
        }
      }
    }

    // Rotate the faces involved in the vertex rotation and add the new diagonals
    int tmpCounter = 0;
    for (int i = 0; i < facesCounterclockwise.size(); ++i)
    {
      int v2 = vertsOnPerimeter[tmpCounter++],
        v3 = vertsOnPerimeter[tmpCounter++],
        v4 = vertsOnPerimeter[tmpCounter % vertsOnPerimeter.size()];
      F.row(facesCounterclockwise[i]) << rotationVert, v2, v3, v4;
      
      diagonals.emplace(Edge{ rotationVert, v3 });
      diagonals.emplace(Edge{ v2, v4 });
    }

    // Update adt
    for (int i = 0; i < vertsOnPerimeter.size(); i = i + 2)
    {
      int faceIndex = i / 2 == 0 ? facesCounterclockwise.size() - 1 : (i / 2) - 1;
      adt[vertsOnPerimeter[i]].push_back(facesCounterclockwise[faceIndex]); // TODO emplace_back
    }
    tmpCounter = 1;
    for (int i = 1; i < vertsOnPerimeter.size(); i = i + 2, ++tmpCounter)
    {
      std::vector<int>::iterator begin = adt[vertsOnPerimeter[i]].begin();
      std::vector<int>::iterator end = adt[vertsOnPerimeter[i]].end();
      adt[vertsOnPerimeter[i]].erase(std::remove(begin, end, 
        facesCounterclockwise[tmpCounter % facesCounterclockwise.size()]), 
        adt[vertsOnPerimeter[i]].end());
    }

    // Searching for possible doublets
    bool result = true;
    for (int face : facesCounterclockwise)
    {
      int valence = 0;
      Eigen::RowVector4i f = F.row(face);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == rotationVert)
        {
          int possibleDoubletVert = f[(i + 2) % 4];
          for (int fc : adt[possibleDoubletVert])
          {
            if (tombStonesF[fc]) { ++valence; }
          }

          if (valence == 2) // Doublet found
          {
            if (possibleDoubletVert == vertForNextCollapse)
            {
              result = false; // Disable the next possible diagonal collapse
            }
            remove_doublet(V, tombStonesV, F, tombStonesF, adt,
              possibleDoubletVert, edges, diagonals);
            break;
          }
        }
      }
    }

    return result;
  }

  return true;
}

// Edge rotation and vertex rotation
bool optimize_quad_mesh(Eigen::MatrixXd& V, std::vector<bool>& tombStonesV, 
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, std::vector<std::vector<int>>& adt,
  std::set<Edge, compareTwoEdges>& edges, std::set<Edge, compareTwoEdges>& diagonals,
  std::vector<Edge> involvedEdges, std::vector<int> involvedVertices)
{
  // Edge rotation
  for (Edge edge : involvedEdges)
  {
    if (!try_edge_rotation(edge, V, tombStonesV, F, tombStonesF, adt, edges, diagonals))
    {
      return false;
    }
  }

  // Vertex rotation
  for (int vert : involvedVertices)
  {
    try_vertex_rotation(vert, V, tombStonesV, F, tombStonesF, adt, edges, diagonals);
  }

  return true;
}

bool diagonal_collapse(Eigen::MatrixXd& V, std::vector<bool>& tombStonesV,
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, 
  std::vector<std::vector<int>>& adt, std::set<Edge, compareTwoEdges>& edges,
  std::set<Edge, compareTwoEdges>& diagonals,
  int vertexToBeMantained, int vertexToBeRemoved)
{
  // The vertices to be checked (after the collapse) for the possible presence of doublets
  int vert1ForDoublet, vert2ForDoublet = -1;
  
  // Modify the faces and modify/remove the edges and diagonals involved in the collapse
  int faceToBeCollapsed = -1;
  for (int face : adt[vertexToBeRemoved])
  {
    if (tombStonesF[face])
    {
      Eigen::RowVector4i f = F.row(face);
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == vertexToBeRemoved)
        {
          int nextVert = f[(i + 1) % 4];
          int oppositeVert = f[(i + 2) % 4];

          edges.erase(Edge{ vertexToBeRemoved, nextVert });
          diagonals.erase(Edge{ vertexToBeRemoved, oppositeVert });
          
          if (oppositeVert != vertexToBeMantained)
          {
            edges.emplace(Edge{ vertexToBeMantained, nextVert });
            diagonals.emplace(Edge{ vertexToBeMantained, oppositeVert });
            
            // Modify one of the faces around the collapse
            F.row(face)[i] = vertexToBeMantained;

            adt[vertexToBeMantained].push_back(face);
            break;
          }
          else
          {
            faceToBeCollapsed = face;
            int oppositeNextVert = f[(i + 3) % 4];
            diagonals.erase(Edge{ oppositeNextVert, nextVert });
            vert1ForDoublet = nextVert;
            vert2ForDoublet = oppositeNextVert;
          }
        }
      }
    }
  }
  if (faceToBeCollapsed == -1 || vert2ForDoublet == -1) { return false; }

  // Remove one of the two vertices collapsed
  tombStonesV[vertexToBeRemoved] = false;

  // Remove the face collapsed
  tombStonesF[faceToBeCollapsed] = false;

  // Calculate and set the new vertex's position
  Eigen::RowVector4i face = F.row(faceToBeCollapsed);
  std::vector<int> faceVertices = { face[0], face[1], face[2], face[3] };
  V.row(vertexToBeMantained) = 
    new_vertex_pos(V, F, tombStonesF, adt, vertexToBeMantained, faceVertices);

  // Searching for possible doublets
  int v1Valence = 0, v2Valence = 0;
  for (int face : adt[vert1ForDoublet])
  {
    if (tombStonesF[face]) { ++v1Valence; }
  }
  if (v1Valence == 2) // Doublet found
  {
    remove_doublet(V, tombStonesV, F, tombStonesF, adt, vert1ForDoublet, 
      edges, diagonals);
  }

  for (int face : adt[vert2ForDoublet])
  {
    if (tombStonesF[face]) { ++v2Valence; }
  }
  if (v2Valence == 2) // Doublet found
  {
    remove_doublet(V, tombStonesV, F, tombStonesF, adt, vert2ForDoublet, 
      edges, diagonals);
  }

  return true;
}

bool edge_collapse(Eigen::MatrixXd& V, std::vector<bool>& tombStonesV, 
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF, 
  std::vector<std::vector<int>>& adt, std::set<Edge, compareTwoEdges>& edges,
  std::set<Edge, compareTwoEdges>& diagonals, int vertex1, int vertex2)
{
  if (!try_vertex_rotation(vertex1, V, tombStonesV, F, tombStonesF, adt,
    edges, diagonals, vertex2, true)) // Force rotation
  {
    // The next diagonal collapse should not be done if the checking for doublets
    // during vertex rotation removed the vertex to be used by the collapse
    return true;
  }

  if (!diagonal_collapse(V, tombStonesV, F, tombStonesF, adt, edges, diagonals,
      vertex1, vertex2))
  {
    return false;
  }

  return true;
}

// Edge collapse or diagonal collapse
bool coarsen_quad_mesh(Eigen::MatrixXd& V, std::vector<bool>& tombStonesV, 
  Eigen::MatrixXi& F, std::vector<bool>& tombStonesF,
  std::vector<std::vector<int>>& adt, std::set<Edge, compareTwoEdges>& edges, 
  std::set<Edge, compareTwoEdges>& diagonals)
{
  std::vector<int> vertices;
  int verticesCount = V.rows();
  for (int i = 0; i < verticesCount; ++i)
  {
    vertices.push_back(i); // TODO Try to set a size to 'vertices' above, and change this line accordingly
  }
  
  // Search for the shortest edge
  Edge shortestEdge = *edges.begin();
  double currEdgeLength, shortestEdgeLength = 
    squared_distance(V, vertices[shortestEdge.first], vertices[shortestEdge.second]);
  for (Edge e : edges)
  {
    currEdgeLength = squared_distance(V, vertices[e.first], vertices[e.second]);
    if (currEdgeLength < shortestEdgeLength)
    {
      shortestEdgeLength = currEdgeLength;
      shortestEdge = e;
    }
  }

  // Search for the shortest diagonal
  std::pair<int, int> shortestDiag = *diagonals.begin();
  double currDiagonalLength, shortestDiagonalLength =
    squared_distance(V, vertices[shortestDiag.first], vertices[shortestDiag.second]);
  for (std::pair<int, int> d : diagonals)
  {
    currDiagonalLength = squared_distance(V, vertices[d.first], vertices[d.second]);
    if (currDiagonalLength < shortestDiagonalLength)
    {
      shortestDiagonalLength = currDiagonalLength;
      shortestDiag = d;
    }
  }

  // Quad mesh coarsening
  int vertexMantained, vertexRemoved;
  if (shortestDiagonalLength < shortestEdgeLength * 2)
  {
    // Diagonal collapse
    vertexMantained = shortestDiag.second;
    vertexRemoved = shortestDiag.first;
    if (!diagonal_collapse(V, tombStonesV, F, tombStonesF, adt, edges, diagonals,
      vertexMantained, vertexRemoved))
    {
      return false;
    }
  }
  else
  {
    // Edge collapse
    vertexMantained = shortestEdge.second;
    vertexRemoved = shortestEdge.first;
    if (!edge_collapse(V, tombStonesV, F, tombStonesF, adt, edges, diagonals,
      vertexMantained, vertexRemoved))
    {
      return false;
    }
  }

  // Final local optimization
  std::vector<Edge> edgesFromVertex;
  for (int face : adt[vertexMantained])
  {
    if (tombStonesF[face])
    {
      auto f = F.row(face); // TODO Try to explicit raw type outside the loop
      for (int i = 0; i < 4; ++i)
      {
        if (f[i] == vertexMantained)
        {
          edgesFromVertex.push_back(Edge{ vertexMantained, f[(i + 1) % 4] });
          break;
        }
      }
    }
  }
  if (!optimize_quad_mesh(V, tombStonesV, F, tombStonesF, adt, edges, diagonals,
    edgesFromVertex, { vertexMantained }))
  {
    return false;
  }

  return true;
}

int how_many_faces_alive(std::vector<bool> tombStonesF)
{
  int fCounter = 0;
  for (bool tomb : tombStonesF)
  {
    if (tomb) { ++fCounter; }
  }
  return fCounter; // TODO Try return std::count(tombStonesF.begin(), tombStonesF.end(), true);
}

bool start_simplification(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int finalNumberOfFaces)
{
  int minimumNumberOfFaces = 15;
  if (finalNumberOfFaces < minimumNumberOfFaces)
  {
    std::cout << "Minimun number of faces (" << minimumNumberOfFaces << ") reached.\n";
    return true;
  }
  
  // Used to navigate the mesh
  // For each vertex, store the indices of the adjacent faces
  int vSize = V.rows();
  std::vector<std::vector<int>> adt(vSize);

  // Used to indicate the faces and vertices already deleted
  int fSize = F.rows();
  std::vector<bool> tombStonesF(fSize, true),
    tombStonesV(vSize, true); // All are alive at the beginning

  // The set of the quad mesh's egdes and diagonals
  std::set<Edge, compareTwoEdges> edges, diagonals;
  
  for (int i = 0; i < fSize; ++i) // For each quad face
  {
    Eigen::RowVector4i face = F.row(i);

    diagonals.emplace_hint(diagonals.end(), std::make_pair(face[0], face[2]));
    diagonals.emplace_hint(diagonals.end(), std::make_pair(face[1], face[3]));

    edges.emplace_hint(edges.end(), Edge{ face[0], face[1] });
    edges.emplace_hint(edges.end(), Edge{ face[1], face[2] });
    edges.emplace_hint(edges.end(), Edge{ face[2], face[3] });
    edges.emplace_hint(edges.end(), Edge{ face[3], face[0] });
    
    for (int j = 0; j < 4; ++j)
    {
      adt[face[j]].push_back(i);
    }
  }

  // Try to coarsen the quad mesh
  while (how_many_faces_alive(tombStonesF) > finalNumberOfFaces)
  {
    if (!coarsen_quad_mesh(V, tombStonesV, F, tombStonesF, adt, edges, diagonals))
    {
      return false;
    }
    std::cout << how_many_faces_alive(tombStonesF) << "\n"; // // TODO delete
  }

  // Set invisible the deleted faces
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
  Eigen::MatrixXd fN(2 * N.rows(), 3);
  for (int i = 0; i < N.rows(); ++i)
  {
    fN.row(i * 2) = N.row(i);
    fN.row(i * 2 + 1) = N.row(i);
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
  const Eigen::RowVector3d black(0.0, 0.0, 0.0);
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
      V[i].push_back(vertex[j]);
    }
  }
  for (int i = 0; i < facesCount; ++i)
  {
    auto face = origF.row(i); // TODO try to use explicit raw type, and put it outside the loop
    for (int j = 0; j < 4; ++j)
    {
      F[i].push_back(face[j]);
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
