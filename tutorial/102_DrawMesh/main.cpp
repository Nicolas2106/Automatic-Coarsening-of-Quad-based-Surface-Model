#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/fit_plane.h>
#include <igl/polygon_corners.h>
#include <igl/polygons_to_triangles.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_face_normals.h>
#include "tutorial_shared_path.h"
#include <array>

struct HalfEdge {
  int vertex; // Index of the vertex from which the half-edge starts
  int face; // Index of the face
  int next; // Index of the next half-edge
};

typedef Eigen::RowVector2i Edge;

bool operator==(const Edge e1, const Edge e2)
{
  int e10 = e1[0], e11 = e1[1], e20 = e2[0], e21 = e2[1];
  return (e10 == e20 && e11 == e21) || (e11 == e20 && e10 == e21);
}

struct compareTwoEdges {
  bool operator()(Edge e1, Edge e2) const {
    int min1 = std::min(e1[0], e1[1]), min2 = std::min(e2[0], e2[1]);
    return min1 == min2 ? std::max(e1[0], e1[1]) < std::max(e2[0], e2[1]) : min1 < min2;
  }
};

double distance_two_points(Eigen::RowVector3d point1, Eigen::RowVector3d point2)
{
  double deltaX = point1[0] - point2[0],
    deltaY = point1[1] - point2[1],
    deltaZ = point1[2] - point2[2];
  return std::sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);
}

typedef struct {
  Eigen::MatrixXd V;
  
  // For diagonals
  bool operator()(const std::pair<int, int> d1, const std::pair<int, int> d2) const {
    return distance_two_points(V.row(d1.first), V.row(d1.second)) >
      distance_two_points(V.row(d2.first), V.row(d2.second));
  }

  // For edges
  bool operator()(const Edge e1, const Edge e2) const {
    return distance_two_points(V.row(e1[0]), V.row(e1[1])) >
      distance_two_points(V.row(e2[0]), V.row(e2[1]));
  }
} MinHeapCompare;

std::vector<Edge> find_edges(Eigen::RowVector4i face)
{
  std::vector<Edge> edges;
  for (int i = 0; i < 4; ++i)
  {
    edges.push_back(Edge{ face[i], face[(i + 1) % 4] });
  }
  return edges;
}

Edge find_edge(Eigen::RowVector4i face, int startingVertex)
{
  std::vector<Edge> edges = find_edges(face);
  for (Edge edge : edges)
  {
    if (edge[0] == startingVertex) return edge;
  }
  assert(false && "An error occured while searching for an edge.");
}

Eigen::RowVector3d new_vertex_pos(Eigen::MatrixXd& V, std::vector<int> vertices, 
  std::vector<int> faceVertices)
{
  Eigen::RowVector3d centroid{ 0.0, 0.0, 0.0 };
  for (int vertex : vertices)
  {
    centroid += V.row(vertex);
  }
  centroid /= vertices.size();

  Eigen::RowVector3d normal, pointOnPlane;
  Eigen::MatrixXd v(4, 3);
  v <<
    V.row(faceVertices[0]),
    V.row(faceVertices[1]),
    V.row(faceVertices[2]),
    V.row(faceVertices[3]);
  igl::fit_plane(v, normal, pointOnPlane);
  
  Eigen::RowVector3d vec = centroid - pointOnPlane;
  double proj = vec.dot(normal);
  return centroid - (proj * normal);
}

std::pair<int, int> half_edges_from_edge(std::vector<HalfEdge> halfEdges, Edge edge)
{
  int v1 = edge[0], v2 = edge[1];
  int firstHe = -1, secondHe = -1;

  for (int i = 0; i < halfEdges.size(); ++i)
  {
    if (firstHe != -1 && secondHe != -1) break;
    
    HalfEdge he = halfEdges[i];
    int vert1 = he.vertex;
    if (firstHe == -1 && vert1 == v1 && halfEdges[he.next].vertex == v2)
    {
      firstHe = i;
    }
    if (secondHe == -1 && vert1 == v2 && halfEdges[he.next].vertex == v1)
    {
      secondHe = i;
    }
  }

  return std::make_pair(firstHe, secondHe);
}

std::vector<Edge> edges_from_vertex(int vertex, std::vector<HalfEdge> halfEdges)
{
  std::vector<Edge> edges;
  for (HalfEdge e : halfEdges)
  {
    if (e.vertex == vertex)
    {
      edges.push_back(Edge{ e.vertex, halfEdges[e.next].vertex });
    }
  }
  return edges;
}

std::vector<int> half_edges_from_vertex(int vertex, std::vector<HalfEdge> halfEdges)
{
  std::vector<int> hes, hesTmp;
  
  for (int i = 0; i < halfEdges.size(); ++i)
  {
    if (halfEdges[i].vertex == vertex)
    {
      hesTmp.push_back(i);
    }
  }

  hes.push_back(hesTmp[0]);
  for (int i = 1; i < hesTmp.size(); ++i)
  {
    int v = halfEdges[halfEdges[halfEdges[halfEdges[hes.back()].next].next].next].vertex;
    for (int j = 1; j < hesTmp.size(); ++j)
    {
      if (halfEdges[halfEdges[hesTmp[j]].next].vertex == v)
      {
        hes.push_back(hesTmp[j]);
        break;
      }
    }
  }

  return hes;
}

void remove_face(Eigen::MatrixXi& F, int rowIndex, std::vector<HalfEdge>& halfEdges,
  std::vector<std::pair<int, int>>& diagonals)
{
  Eigen::RowVector4i face = F.row(rowIndex);
  
  Eigen::MatrixXi newF(F.rows() - 1, F.cols());
  newF << F.topRows(rowIndex), F.bottomRows(F.rows() - rowIndex - 1);
  F.conservativeResize(F.rows() - 1, Eigen::NoChange);
  F = newF;

  // Modify half-edges due to the face's deletion 
  for (HalfEdge& he : halfEdges)
  {
    if (he.face > rowIndex)
    {
      --he.face;
    }
  }

  // Remove the diagonals involved in the face's deletion
  Edge edge1 = Edge{ face[0], face[2] };
  Edge edge2 = Edge{ face[1], face[3] };
  for (int i = 0; i < diagonals.size(); ++i)
  {
    if (Edge{ diagonals[i].first, diagonals[i].second } == edge1)
    {
      diagonals.erase(diagonals.begin() + i);
      break;
    }
  }
  for (int i = 0; i < diagonals.size(); ++i)
  {
    if (Edge{ diagonals[i].first, diagonals[i].second } == edge2)
    {
      diagonals.erase(diagonals.begin() + i);
      break;
    }
  }
}

void remove_vertex(Eigen::MatrixXd& V, int rowIndex, 
  Eigen::MatrixXi& F, std::vector<HalfEdge>& halfEdges, 
  std::set<Edge, compareTwoEdges>& edges, 
  std::vector<std::pair<int, int>>& diagonals, int substituteVertex)
{
  // Modify the matrix V removing one row (= one vertex)
  Eigen::MatrixXd newV(V.rows() - 1, V.cols());
  newV << V.topRows(rowIndex), V.bottomRows(V.rows() - rowIndex - 1);
  V.conservativeResize(V.rows() - 1, Eigen::NoChange);
  V = newV;

  // Modify the vertices' indices inside the matrix F
  for (int i = 0; i < F.rows(); ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      if (F.row(i)[j] > rowIndex)
      {
        --F.row(i)[j];
      }
    }
  }

  // Modify the vertices' indices inside halfEdges vector
  for (HalfEdge& he : halfEdges)
  {
    if (he.vertex > rowIndex)
    {
      --he.vertex;
    }
  }

  std::vector<Edge> vec(edges.begin(), edges.end());
  // Clear edges set in a optimized way
  std::set<Edge, compareTwoEdges> emptySet;
  std::swap(edges, emptySet);
  std::set<Edge, compareTwoEdges>::iterator hint = edges.end();
  for (Edge& e : vec)
  {
    if (e[0] > rowIndex) --e[0];
    if (e[1] > rowIndex) --e[1];

    edges.emplace_hint(hint, e);
  }

  // Modify the diagonals involved in the vertex's deletion
  for (int i = 0; i < diagonals.size(); ++i)
  {
    if (diagonals[i].first == rowIndex)
    {
      diagonals[i].first = substituteVertex;
    }
    if (diagonals[i].second == rowIndex)
    {
      diagonals[i].second = substituteVertex;
    }
  }

  // Modify the vertices' indices for diagonals
  for (int i = 0; i < diagonals.size(); ++i)
  {
    if (diagonals[i].first > rowIndex)
    {
      --diagonals[i].first;
    }
    if (diagonals[i].second > rowIndex)
    {
      --diagonals[i].second;
    }
  }
}

void remove_half_edges(std::vector<int> hesToBeRemoved, std::vector<HalfEdge>& halfEdges)
{
  std::sort(hesToBeRemoved.begin(), hesToBeRemoved.end());

  for (int i = hesToBeRemoved.size() - 1; i >= 0; --i)
  {
    int heToBeRemoved = hesToBeRemoved[i];

    halfEdges.erase(halfEdges.begin() + heToBeRemoved);
    for (HalfEdge& he : halfEdges)
    {
      if (he.next > heToBeRemoved) --he.next;
    }
  }
}

void remove_doublet(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  int startingHalfEdge, std::vector<HalfEdge>& halfEdges, 
  std::set<Edge, compareTwoEdges>& edges, std::vector<std::pair<int, int>>& diagonals)
{
  int vertexToBeDeleted = halfEdges[startingHalfEdge].vertex;
  int otherStartingHalfEdge = -1;
  for (int i = 0; i < halfEdges.size(); ++i)
  {
    if (halfEdges[i].vertex == vertexToBeDeleted && i != startingHalfEdge)
    {
      otherStartingHalfEdge = i;
      break;
    }
  }
  assert(otherStartingHalfEdge != -1 && "An error occured a doublet removal.");
  
  int next1 = halfEdges[startingHalfEdge].next;
  int first2 = halfEdges[next1].next;
  int next2 = halfEdges[otherStartingHalfEdge].next;
  int first1 = halfEdges[next2].next;

  int vertexToBeMantained = halfEdges[first1].vertex;

  int secondHe = halfEdges[first1].next;
  int fouthHe = halfEdges[first2].next;

  halfEdges[first1].next = next1;
  halfEdges[first2].next = next2;

  int face = halfEdges[startingHalfEdge].face;
  halfEdges[first1].face = face;
  halfEdges[next2].face = face;

  for (int i = 0; i < 4; ++i)
  {
    if (F.row(face)[i] == vertexToBeDeleted)
    {
      F.row(face)[i] = halfEdges[first1].vertex;
    }
  }

  int faceToBeRemoved = halfEdges[secondHe].face;

  edges.erase(Edge{ vertexToBeDeleted, halfEdges[next1].vertex });
  edges.erase(Edge{ vertexToBeDeleted, halfEdges[next2].vertex });

  remove_half_edges({ startingHalfEdge, secondHe, otherStartingHalfEdge, fouthHe },
    halfEdges);
  
  remove_face(F, faceToBeRemoved, halfEdges, diagonals);

  remove_vertex(V, vertexToBeDeleted, F, halfEdges, edges, diagonals, 
    vertexToBeMantained);
}

bool try_edge_rotation(Edge edge, Eigen::MatrixXd& V, Eigen::MatrixXi& F,
  std::vector<HalfEdge>& halfEdges, std::set<Edge, compareTwoEdges>& edges,
  std::vector<std::pair<int, int>>& diagonals)
{
  std::pair<int, int> pair = half_edges_from_edge(halfEdges, edge);
  
  HalfEdge& firstHe1 = halfEdges[pair.second];
  HalfEdge& firstHe2 = halfEdges[firstHe1.next];
  HalfEdge& firstHe3 = halfEdges[firstHe2.next];
  HalfEdge& firstHe4 = halfEdges[firstHe3.next];

  HalfEdge& secondHe1 = halfEdges[pair.first];
  HalfEdge& secondHe2 = halfEdges[secondHe1.next];
  HalfEdge& secondHe3 = halfEdges[secondHe2.next];
  HalfEdge& secondHe4 = halfEdges[secondHe3.next];

  // Edge rotation is profitable if it shortens the rotated edge and both the diagonals
  double oldEdgeLength = distance_two_points(V.row(firstHe1.vertex), V.row(secondHe1.vertex));
  double oldDiag1Length = distance_two_points(V.row(firstHe4.vertex), V.row(firstHe2.vertex));
  double oldDiag2Length = distance_two_points(V.row(firstHe1.vertex), V.row(secondHe4.vertex));
  double oldDiag3Length = distance_two_points(V.row(firstHe1.vertex), V.row(firstHe3.vertex));
  double oldDiag4Length = distance_two_points(V.row(secondHe3.vertex), V.row(secondHe1.vertex));

  double edgeClockLength = distance_two_points(V.row(firstHe4.vertex), V.row(secondHe4.vertex));
  double edgeCounterLength = distance_two_points(V.row(firstHe3.vertex), V.row(secondHe3.vertex));
  double newDiag1Length = distance_two_points(V.row(firstHe3.vertex), V.row(secondHe4.vertex));
  double newDiag2Length = distance_two_points(V.row(firstHe4.vertex), V.row(secondHe3.vertex));

  int firstDoubletHe = firstHe1.next;
  std::pair<int, int> secondDoubletCheck = std::make_pair(secondHe2.vertex, secondHe3.vertex);

  bool rotation = false;

  if (edgeClockLength < oldEdgeLength &&
    newDiag1Length < oldDiag3Length &&
    newDiag2Length < oldDiag4Length) // Clockwise edge rotation
  {
    rotation = true;

    Edge oldDiag1 = Edge{ firstHe3.vertex, firstHe1.vertex };
    Edge oldDiag2 = Edge{ secondHe3.vertex, secondHe1.vertex };

    // Modify the half-edges map
    edges.emplace(Edge{ firstHe4.vertex, secondHe4.vertex });
    edges.erase(edge);

    // Modify the half-edges involved in the rotation
    firstHe1.vertex = firstHe4.vertex;
    int tmpFirstHe = firstHe1.next;
    firstHe1.next = secondHe3.next;
    secondHe1.vertex = secondHe4.vertex;
    int tmpSecondHe = secondHe1.next;
    secondHe1.next = firstHe3.next;
    firstHe3.next = firstHe4.next;
    secondHe4.face = firstHe1.face;
    secondHe3.next = secondHe4.next;
    secondHe4.next = tmpFirstHe;
    firstHe4.face = secondHe1.face;
    firstHe4.next = tmpSecondHe;

    // Modify the faces involved in the rotation
    for (int i = 0; i < 4; ++i)
    {
      if (F.row(firstHe1.face)[i] == secondHe2.vertex)
      {
        F.row(firstHe1.face)[i] = secondHe4.vertex;
      }
      if (F.row(secondHe1.face)[i] == firstHe2.vertex)
      {
        F.row(secondHe1.face)[i] = firstHe4.vertex;
      }
    }

    // Modify the diagonals involved in the rotation
    for (std::pair<int, int>& diag : diagonals)
    {
      if (Edge{ diag.first, diag.second } == oldDiag1)
      {
        if (diag.first == oldDiag1[1])
          diag.first = secondHe4.vertex;
        else
          diag.second = secondHe4.vertex;
      }
      else if (Edge{ diag.first, diag.second } == oldDiag2)
      {
        if (diag.first == oldDiag2[1])
          diag.first = firstHe4.vertex;
        else
          diag.second = firstHe4.vertex;
      }
    }
  }
  else if (edgeCounterLength < oldEdgeLength &&
    newDiag2Length < oldDiag3Length &&
    newDiag1Length < oldDiag4Length) // Counterclockwise edge rotation
  {
    rotation = true;

    Edge oldDiag1 = Edge{ firstHe4.vertex, firstHe2.vertex };
    Edge oldDiag2 = Edge{ secondHe4.vertex, secondHe2.vertex };

    // Modify the half-edges map
    edges.emplace(Edge{ firstHe3.vertex, secondHe3.vertex });
    edges.erase(edge);

    // Modify the half-edges involved in the rotation
    firstHe1.vertex = secondHe3.vertex;
    int tmpFirstHe = firstHe1.next;
    firstHe1.next = firstHe2.next;
    secondHe1.vertex = firstHe3.vertex;
    int tmpSecondHe = secondHe1.next;
    secondHe1.next = secondHe2.next;
    secondHe2.face = firstHe1.face;
    secondHe2.next = firstHe4.next;
    firstHe2.face = secondHe1.face;
    firstHe2.next = secondHe4.next;
    firstHe4.next = tmpSecondHe;
    secondHe4.next = tmpFirstHe;

    // Modify the faces involved in the rotation
    for (int i = 0; i < 4; ++i)
    {
      if (F.row(firstHe1.face)[i] == firstHe2.vertex)
      {
        F.row(firstHe1.face)[i] = secondHe3.vertex;
      }
      if (F.row(secondHe1.face)[i] == secondHe2.vertex)
      {
        F.row(secondHe1.face)[i] = firstHe3.vertex;
      }
    }

    // Modify the diagonals involved in the rotation
    for (std::pair<int, int>& diag : diagonals)
    {
      if (Edge{ diag.first, diag.second } == oldDiag1)
      {
        if (diag.first == oldDiag1[1])
          diag.first = secondHe3.vertex;
        else
          diag.second = secondHe3.vertex;
      }
      else if (Edge{ diag.first, diag.second } == oldDiag2)
      {
        if (diag.first == oldDiag2[1])
          diag.first = firstHe3.vertex;
        else
          diag.second = firstHe3.vertex;
      }
    }
  }
  
  if (rotation)
  {
    // Searching for the two possible doublets created after the rotation
    int countEdges = 0;
    int vert = halfEdges[firstDoubletHe].vertex;
    for (int i = 0; i < halfEdges.size(); ++i)
    {
      if (halfEdges[i].vertex == vert)
        ++countEdges;
    }
    if (countEdges == 2)
    {
      // First doublet found
      if (secondDoubletCheck.first > vert) --secondDoubletCheck.first;
      if (secondDoubletCheck.second > vert) --secondDoubletCheck.second;

      remove_doublet(V, F, firstDoubletHe, halfEdges, edges, diagonals);
    }

    std::pair<int, int> doubletHes = half_edges_from_edge(halfEdges, 
      Edge{ secondDoubletCheck.first, secondDoubletCheck.second });
    int secondDoubletHe = halfEdges[doubletHes.first].vertex == secondDoubletCheck.first ?
      doubletHes.first : doubletHes.second;
    vert = halfEdges[secondDoubletHe].vertex;
    countEdges = 0;
    for (int i = 0; i < halfEdges.size(); ++i)
    {
      if (halfEdges[i].vertex == vert)
        ++countEdges;
    }
    if (countEdges == 2)
    {
      // Second doublet found
      remove_doublet(V, F, secondDoubletHe, halfEdges, edges, diagonals);
    }
  }

  return true;
}

std::pair<bool, std::vector<int>> try_vertex_rotation(int vert,
  Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::vector<HalfEdge>& halfEdges,
  std::set<Edge, compareTwoEdges>& edges, std::vector<std::pair<int, int>>& diagonals, 
  const bool forceRotation, const int vertToBeMantained = -1)
{
  std::vector<int> verticesRemovedForDoublets;
  // The result will be used for cooperation during an edge collapse
  std::pair<bool, std::vector<int>> result = 
    std::make_pair(true, verticesRemovedForDoublets);
  
  Eigen::RowVector3d vertex = V.row(vert);
  std::vector<int> hes = half_edges_from_vertex(vert, halfEdges);
  std::vector<std::pair<int, int>*> diagonalsInvolved;
  double edgesSum = 0.0, diagonalsSum = 0.0;
  for (int he : hes)
  {
    HalfEdge h = halfEdges[he];
    edgesSum += distance_two_points(vertex, V.row(halfEdges[h.next].vertex));
    diagonalsSum += distance_two_points(vertex,
      V.row(halfEdges[halfEdges[h.next].next].vertex));

    Eigen::RowVector4i face = F.row(h.face);
    bool endLoop = false;
    for (std::pair<int, int>& diagonal : diagonals)
    {
      if ((diagonal.first == face[0] && diagonal.second == face[2]) || 
        (diagonal.second == face[0] && diagonal.first == face[2]) ||
        (diagonal.first == face[1] && diagonal.second == face[3]) || 
        (diagonal.second == face[1] && diagonal.first == face[3]))
      {
        diagonalsInvolved.push_back(&diagonal);
        if (endLoop) break;

        endLoop = true;
      }
    }
  }

  // Check if the sum of the edge lengths overcomes the sum of the diagonals lengths
  if (edgesSum > diagonalsSum || forceRotation) // Vertex rotation
  {
    // Modify the faces involved in the rotation
    for (int i = 0; i < hes.size(); ++i)
    {
      HalfEdge h = halfEdges[hes[i]];
      for (int j = 0; j < 4; ++j)
      {
        HalfEdge tmp = halfEdges[halfEdges[h.next].next];
        int v = F.row(h.face)[j];
        if (v == halfEdges[h.next].vertex)
        {
          F.row(h.face)[j] = tmp.vertex;
        }
        else if (v == tmp.vertex)
        {
          F.row(h.face)[j] = halfEdges[tmp.next].vertex;
        }
        else if (v == halfEdges[tmp.next].vertex)
        {
          F.row(h.face)[j] = halfEdges[halfEdges[halfEdges[
            hes[(i + 1) % hes.size()]].next].next].vertex;
        }
      }
    }

    // Modify the the half-edges involved in the rotation
    int tmp4 = 0, opposite2;
    for (int i = 0; i < hes.size(); ++i)
    {
      int he = hes[i];
      int tmp = halfEdges[he].next;
      int tmp1 = halfEdges[tmp].next;
      int tmp2 = halfEdges[tmp1].next;
      int opposite1 = hes[(i + 1) % hes.size()];
      if (i == 0)
      {
        opposite2 = halfEdges[halfEdges[halfEdges[hes[hes.size() - 1]].next].next].next;
      }

      if (i == hes.size() - 1)
      {
        halfEdges[tmp2].vertex = halfEdges[halfEdges[opposite1].next].vertex;
        halfEdges[tmp1].next = tmp4;
        halfEdges[tmp].face = halfEdges[opposite2].face;
        halfEdges[tmp].next = opposite2;
        halfEdges[he].next = tmp1;
        break;
      }

      halfEdges[tmp2].vertex = halfEdges[halfEdges[halfEdges[opposite1].next].next].vertex;
      halfEdges[tmp1].next = halfEdges[opposite1].next;
      halfEdges[tmp].face = halfEdges[opposite2].face;
      halfEdges[tmp].next = opposite2;
      halfEdges[he].next = tmp1;
      if (i == 0)
      {
        tmp4 = tmp;
      }
      opposite2 = tmp2;
    }

    // Modify the half-edges map involved in the rotation
    for (int he : hes)
    {
      HalfEdge h = halfEdges[halfEdges[halfEdges[he].next].next];
      Edge edgeDeleted = Edge{ vert, h.vertex };
      std::pair<int, int> tmp = half_edges_from_edge(halfEdges, edgeDeleted);
      edges.erase(edgeDeleted);
      edges.emplace(Edge{ vert, halfEdges[h.next].vertex });
    }

    // Modify the diagonals involved in the rotation
    for (int i = 0; i < hes.size(); ++i)
    {
      int he = hes[i];
      for (std::pair<int, int>* diag : diagonalsInvolved)
      {
        std::pair<int, int> diagonal = *diag;
        int v = halfEdges[halfEdges[halfEdges[hes[(hes.size() + i - 1) % hes.size()]]
          .next].next].vertex;
        HalfEdge heTmp = halfEdges[halfEdges[he].next];

        if (Edge{ diagonal.first, diagonal.second } == Edge{ halfEdges[he].vertex,
          heTmp.vertex })
        {
          heTmp = halfEdges[halfEdges[heTmp.next].next];
          if (diagonal.first == vert)
            (*diag).first = heTmp.vertex;
          else
            (*diag).second = heTmp.vertex;
        }
        else if (Edge{ diagonal.first, diagonal.second } ==
          Edge{ halfEdges[heTmp.next].vertex, v })
        {
          if (diagonal.first == halfEdges[heTmp.next].vertex)
            (*diag).second = vert;
          else
            (*diag).first = vert;
        }
      }
    }

    // Searching for doublets
    std::vector<std::pair<int, int>> doubletVerts;
    for (int he : hes)
    {
      int startingHeVertex = halfEdges[halfEdges[halfEdges[he].next].next].vertex;
      int countEdges = 0;
      for (int i = 0; i < halfEdges.size(); ++i)
      {
        if (halfEdges[i].vertex == startingHeVertex)
          ++countEdges;
      }
      if (countEdges == 2)
      {
        // Doublet found
        result.second.push_back(startingHeVertex);
        int tmp = halfEdges[halfEdges[he].next].next;
        doubletVerts.push_back(std::make_pair(startingHeVertex,
          halfEdges[halfEdges[tmp].next].vertex));
        if (halfEdges[tmp].vertex == vertToBeMantained)
        {
          result.first = false; // Disable the possible next diagonal collapse
        }
      }
    }

    int oldFirstVert = -1;
    for (int i = 0; i < doubletVerts.size(); ++i)
    {
      if (oldFirstVert != -1)
      {
        for (int j = i; j < doubletVerts.size(); ++j)
        {
          if (doubletVerts[j].first > oldFirstVert) --doubletVerts[j].first;
          if (doubletVerts[j].second > oldFirstVert) --doubletVerts[j].second;
        }
      }
      oldFirstVert = doubletVerts[i].first;
    }

    for (std::pair<int, int> verts : doubletVerts)
    {
      std::pair<int, int> hs = half_edges_from_edge(halfEdges, Edge{ verts.first, verts.second });
      int startingHe = halfEdges[hs.first].vertex == verts.first ? hs.first : hs.second;
      remove_doublet(V, F, startingHe, halfEdges, edges, diagonals);
    }
  }

  return result;
}

// Edge rotation and vertex rotation
bool optimize_quad_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  std::vector<HalfEdge>& halfEdges, std::set<Edge, compareTwoEdges>& edges,
  std::vector<std::pair<int, int>>& diagonals, 
  std::vector<Edge> involvedEdges, std::vector<int> involvedVertices)
{
  /* Edge rotation */
  for (Edge edge : involvedEdges)
  {
    bool success = try_edge_rotation(edge, V, F, halfEdges, edges, diagonals);
    if (!success) return false;
  }

  /* Vertex rotation */
  for (int vert : involvedVertices)
  {
    std::pair<bool, std::vector<int>> result = 
      try_vertex_rotation(vert, V, F, halfEdges, edges, diagonals, false);
  }

  return true;
}

bool diagonal_collapse(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  std::vector<HalfEdge>& halfEdges, std::set<Edge, compareTwoEdges>& edges,
  std::vector<std::pair<int, int>>& diagonals, 
  int vertexToBeMantained, int vertexToBeRemoved)
{
  int faceIndex = -1;
  Eigen::RowVector4i face;
  for (HalfEdge he : halfEdges)
  {
    if (he.vertex == vertexToBeMantained && 
      halfEdges[halfEdges[he.next].next].vertex == vertexToBeRemoved)
    {
      faceIndex = he.face;
      face = F.row(faceIndex);
    }
  }
  if (faceIndex == -1)
  {
    assert(false && 
      "An error occured while searching for a face during diag collapse.");
  }

  Edge edge = find_edge(face, vertexToBeRemoved);

  int initialHe, initialHeOpposite, finalHe, finalHeOpposite = -1;
  std::pair<int, int> initial = half_edges_from_edge(halfEdges, edge);
  if (halfEdges[initial.first].vertex == vertexToBeRemoved)
  {
    initialHe = initial.first;
    initialHeOpposite = initial.second;
  }
  else
  {
    initialHe = initial.second;
    initialHeOpposite = initial.first;
  }
  finalHe = halfEdges[halfEdges[halfEdges[initialHe].next].next].next;
  
  std::vector<int> heToBeModified = 
    half_edges_from_vertex(halfEdges[initialHe].vertex, halfEdges);
  heToBeModified.erase(std::remove(heToBeModified.begin(), 
    heToBeModified.end(), initialHe), heToBeModified.end());
  std::reverse(heToBeModified.begin(), heToBeModified.end());

  // Set new starting vertices for the half-edge to be modified
  int v1 = halfEdges[finalHe].vertex, v2 = halfEdges[initialHe].vertex;
  for (int he : heToBeModified)
  {
    if (halfEdges[he].vertex == v2 && halfEdges[halfEdges[he].next].vertex == v1)
    {
      finalHeOpposite = he;
      break;
    }
  }
  if (finalHeOpposite == -1) return false;

  for (int he : heToBeModified)
  {
    if (halfEdges[he].vertex != vertexToBeRemoved) return false;

    if (he != finalHeOpposite)
    {
      int v = halfEdges[halfEdges[he].next].vertex, heOpposite;
      for (int h : heToBeModified)
      {
        if (h != he &&
          v == halfEdges[halfEdges[halfEdges[halfEdges[h].next].next].next].vertex)
        {
          heOpposite = halfEdges[halfEdges[halfEdges[h].next].next].next;
          break;
        }
      }
      edges.emplace(Edge{ vertexToBeMantained, v });
      edges.erase(Edge{ vertexToBeRemoved, v });
    }

    halfEdges[he].vertex = vertexToBeMantained;

    for (int i = 0; i < 4; ++i)
    {
      if (F.row(halfEdges[he].face)[i] == vertexToBeRemoved)
      {
        F.row(halfEdges[he].face)[i] = vertexToBeMantained;
      }
    }
  }

  // Remove the half-edges belong to the face to be deleted
  edges.erase(edge);
  edges.erase(Edge{ vertexToBeRemoved, halfEdges[finalHe].vertex });

  std::vector<int> hesToBeRemoved{ initialHe, halfEdges[initialHe].next,
    halfEdges[halfEdges[initialHe].next].next , finalHe };

  remove_half_edges(hesToBeRemoved, halfEdges);

  // Store vertices involved in the collapse for a possible future cleaning
  std::vector<int> vertsToBeCleaned;
  vertsToBeCleaned.push_back(vertexToBeMantained);
  for (int i = 0; i < 4; ++i)
  {
    if (face[i] != vertexToBeRemoved && face[i] != vertexToBeMantained)
    {
      vertsToBeCleaned.push_back(face[i]);
    }
  }

  remove_face(F, faceIndex, halfEdges, diagonals);

  std::vector<int> verticesForCentroid;
  for (HalfEdge he : halfEdges)
  {
    if (he.vertex == vertexToBeMantained)
    {
      HalfEdge h = halfEdges[he.next];
      verticesForCentroid.push_back(h.vertex);
      verticesForCentroid.push_back(halfEdges[h.next].vertex);
    }
  }

  // Calculate the new vertex's position
  std::vector<int> faceVertices = { face[0], face[1], face[2], face[3] };
  V.row(vertexToBeMantained) = new_vertex_pos(V, verticesForCentroid, faceVertices);

  remove_vertex(V, vertexToBeRemoved, F, halfEdges, edges, 
    diagonals, vertexToBeMantained);

  for (int i = 0; i < vertsToBeCleaned.size(); ++i)
  {
    if (vertsToBeCleaned[i] > vertexToBeRemoved)
    {
      --vertsToBeCleaned[i];
    }
  }

  /* Cleaning operations */
  // Searching for doublets
  std::pair<int, int> p1 = half_edges_from_edge(halfEdges,
    Edge{ vertsToBeCleaned[0], vertsToBeCleaned[1] });
  std::pair<int, int> p2 = half_edges_from_edge(halfEdges, 
    Edge{ vertsToBeCleaned[0], vertsToBeCleaned[2] });
  int startingHe1 = halfEdges[p1.first].vertex == vertsToBeCleaned[1] ? p1.first : p1.second;
  int startingHe2 = halfEdges[p2.first].vertex == vertsToBeCleaned[2] ? p2.first : p2.second;

  int countEdges1 = 0, countEdges2 = 0;
  int vert1 = halfEdges[startingHe1].vertex;
  int vert2 = halfEdges[startingHe2].vertex;
  for (int i = 0; i < halfEdges.size(); ++i)
  {
    if (halfEdges[i].vertex == vert1)
      ++countEdges1;
    else if (halfEdges[i].vertex == vert2)
      ++countEdges2;
  }
 
  bool doublet1 = (countEdges1 == 2), doublet2 = (countEdges2 == 2);

  if (doublet1 && doublet2)
  {
    remove_doublet(V, F, startingHe1, halfEdges, edges, diagonals);

    int vertToBeCleaned = vertsToBeCleaned[2] > vertsToBeCleaned[1] ?
      vertsToBeCleaned[2] - 1 : vertsToBeCleaned[2];
    p2 = half_edges_from_edge(halfEdges, Edge{ vertsToBeCleaned[0] > vertsToBeCleaned[1] ?
      vertsToBeCleaned[0] - 1 : vertsToBeCleaned[0], vertToBeCleaned });
    startingHe2 = halfEdges[p2.first].vertex == vertToBeCleaned ? p2.first : p2.second;
    remove_doublet(V, F, startingHe2, halfEdges, edges, diagonals);
  }
  else if (doublet1)
  {
    remove_doublet(V, F, startingHe1, halfEdges, edges, diagonals);
  }
  else if (doublet2)
  {
    remove_doublet(V, F, startingHe2, halfEdges, edges, diagonals);
  }
}

bool edge_collapse(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  std::vector<HalfEdge>& halfEdges, std::set<Edge, compareTwoEdges>& edges,
  std::vector<std::pair<int, int>>& diagonals, 
  int vertexToBeMantained, int vertexToBeRemoved)
{
  std::pair<bool, std::vector<int>> result = 
    try_vertex_rotation(vertexToBeRemoved, V, F, halfEdges, edges, 
      diagonals, true, vertexToBeMantained); // Force rotation

  // Sort in descending order
  std::sort(result.second.begin(), result.second.end(), std::greater<int>());
  for (int vertexRemoved : result.second)
  {
    if (vertexToBeMantained > vertexRemoved) --vertexToBeMantained;
    if (vertexToBeRemoved > vertexRemoved) --vertexToBeRemoved;
  }

  if (result.first)
  {
    bool success = diagonal_collapse(V, F, halfEdges, 
      edges, diagonals, vertexToBeMantained, vertexToBeRemoved);
    if (!success) return false;
  }

  return true;
}

// Edge collapse or diagonal collapse
bool coarsen_quad_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  std::vector<HalfEdge>& halfEdges, std::set<Edge, compareTwoEdges>& edges,
  std::vector<std::pair<int, int>>& diagonals)
{
  std::vector<Eigen::RowVector3d> vertices;
  int size = V.rows();
  for (int i = 0; i < size; ++i)
  {
    vertices.push_back(V.row(i));
  }
  
  // Search for the shortest edge
  Edge shortestEdge = *edges.begin();
  for (Edge e : edges)
  {
    if (distance_two_points(vertices[e[0]], vertices[e[1]]) <
      distance_two_points(vertices[shortestEdge[0]], vertices[shortestEdge[1]]))
    {
      shortestEdge = e;
    }
  }

  // Search for the shortest diagonal
  int min = 0;
  std::pair<int, int> currDiag, currMinDiag;
  size = diagonals.size();
  for (int i = 0; i < size; ++i)
  {
    currDiag = diagonals[i];
    currMinDiag = diagonals[min];
    if (distance_two_points(vertices[currDiag.first], vertices[currDiag.second]) <
      distance_two_points(vertices[currMinDiag.first], vertices[currMinDiag.second]))
    {
      min = i;
    }
  }
  std::pair<int, int> shortestDiagonal = diagonals[min];

  double shortestEdgeLength = distance_two_points(
    V.row(shortestEdge[0]), V.row(shortestEdge[1]));
  
  double shortestDiagonalLength = distance_two_points(
    V.row(shortestDiagonal.first), V.row(shortestDiagonal.second));

  int vertexMantained, vertexRemoved;
  if (shortestDiagonalLength / sqrt(2) < shortestEdgeLength)
  {
    /* Diagonal collapse */
    vertexMantained = shortestDiagonal.second;
    vertexRemoved = shortestDiagonal.first;
    bool success = diagonal_collapse(V, F, halfEdges, edges, diagonals, 
      vertexMantained, vertexRemoved);
    if (!success) return false;
  }
  else
  {
    /* Edge collapse */
    vertexMantained = shortestEdge[1];
    vertexRemoved = shortestEdge[0];
    bool success = edge_collapse(V, F, halfEdges, edges, diagonals,
      vertexMantained, vertexRemoved);
    if (!success) return false;
  }

  // Final local optimization
  int v = vertexMantained > vertexRemoved ? vertexMantained - 1 : vertexMantained;
  optimize_quad_mesh(V, F, halfEdges, edges, diagonals,
    edges_from_vertex(v, halfEdges), { v });

  return true;
}

bool start_simplification(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int finalNumberOfFaces)
{
  // Used to navigate through half-edges
  std::vector<HalfEdge> halfEdges;

  std::set<Edge, compareTwoEdges> edges;

  std::vector<std::pair<int, int>> diagonals;
  
  for (int i = 0; i < F.rows(); ++i) // For each quad face
  {
    Eigen::RowVector4i face = F.row(i);

    diagonals.push_back(std::make_pair(face[0], face[2]));
    diagonals.push_back(std::make_pair(face[1], face[3]));

    std::vector<Edge> faceEdges = find_edges(face);
    
    // Retrieve edges and compute four half-edges per face
    for (int j = 0; j < 4; ++j)
    {
      halfEdges.push_back(HalfEdge{
          face[j], // vertex
          i, // face
          (i * 4) + (int(halfEdges.size() + 1) % 4) // next
        });
      
      edges.emplace_hint(edges.end(), faceEdges[j]);
    }
  }
  
  /* // Initial global optimization of the quad mesh
  std::vector<int> vertices;
  for (int i = 0; i < V.rows(); ++i)
  {
    vertices.push_back(i);
  }
  std::vector<Edge> eds(edges.begin(), edges.end());
  optimize_quad_mesh(V, F, halfEdges, edges, diagonals, eds, vertices); TODO Should I insert that? */
  

  while (F.rows() > finalNumberOfFaces)
  { 
    coarsen_quad_mesh(V, F, halfEdges, edges, diagonals);
    std::cout << F.rows() << "\n"; // TODO delete this line
  }

  return true;
}

void per_quad_face_normals(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& N)
{
  N.resize(F.rows(), 3);
  // Loop over the faces
#pragma omp parallel for if (F.rows() > 1000)
  for (int i = 0; i < F.rows(); ++i)
  {
    Eigen::RowVector3d nN, C;
    Eigen::MatrixXd vV(4, 3);
    vV <<
      V.row(F(i, 0)),
      V.row(F(i, 1)),
      V.row(F(i, 2)),
      V.row(F(i, 3));
    igl::fit_plane(vV, nN, C);

    // Choose the correct normal direction
    Eigen::Matrix<Eigen::MatrixXd::Scalar, 1, 3> v1 = V.row(F(i, 0)) - V.row(F(i, 2));
    Eigen::Matrix<Eigen::MatrixXd::Scalar, 1, 3> v2 = V.row(F(i, 1)) - V.row(F(i, 3));
    Eigen::RowVector3d cp = v1.cross(v2);

    N.row(i) = cp.dot(nN) >= 0 ? nN : -nN;
  }
}

void draw_quad_mesh(const Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
  Eigen::MatrixXi f;

  // 1) Triangulate the polygonal mesh
  std::vector<Edge> diagonals;
  for (int i = 0; i < F.rows(); ++i)
  {
    for (Edge diag : diagonals)
    {
      if (diag == Edge{ F.row(i)[0], F.row(i)[2] })
      {
        F.row(i) = Eigen::Vector4i{ F.row(i)[1], F.row(i)[2], F.row(i)[3], F.row(i)[0] };
      }
    }
    diagonals.push_back(Edge{F.row(i)[0], F.row(i)[2]});
  }
  Eigen::VectorXi I, C, J;
  igl::polygon_corners(F, I, C);
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
  E.resize(F.rows() * 4, 2);
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

int main(int argc, char *argv[])
{
  const std::string MESHES_DIR = "F:\\Users\\Nicolas\\Desktop\\TESI\\Quadrilateral extension to libigl\\libigl\\tutorial\\102_DrawMesh\\";
  
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  // Load a mesh
  //igl::readOFF(MESHES_DIR + "edge_rotate_doublet.off", V, F);
  igl::readOBJ(MESHES_DIR + "quad_cubespikes.obj", V, F);

  if (start_simplification(V, F, 1000)) 
  {
    draw_quad_mesh(V, F);
  }
  else
  {
    std::cout << "\n\n" << "ERROR occured during quad mesh simplification" << "\n\n";
  }
}
