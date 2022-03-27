#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/fit_plane.h>
#include <igl/polygon_corners.h>
#include <igl/polygons_to_triangles.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_face_normals.h>
#include "tutorial_shared_path.h"
#include <queue>
struct HalfEdge {
  int vertex; // Index of the vertex from which the half-edge starts
  int face; // Index of the face
  int next; // Index of the next half-edge
};

bool operator==(const HalfEdge& he1, const HalfEdge& he2)
{
  return he1.vertex == he2.vertex && he1.face == he2.face;
}

bool operator!=(const HalfEdge& he1, const HalfEdge& he2)
{
  return he1.vertex != he2.vertex || he1.face != he2.face;
}

typedef Eigen::RowVector2i Edge;
bool operator==(const Edge e1, const Edge e2)
{
  return (e1[0] == e2[0] && e1[1] == e2[1]) || 
    (e1[1] == e2[0] && e1[0] == e2[1]);
}

struct compareTwoEdges {
  bool operator()(const Edge e1, const Edge e2) const {
    std::string str1 = std::to_string(std::min(e1[0], e1[1])) +
      std::to_string(std::max(e1[0], e1[1]));
    std::string str2 = std::to_string(std::min(e2[0], e2[1])) +
      std::to_string(std::max(e2[0], e2[1]));
    return str1.compare(str2) < 0 ? true : false;
  }
};

int opposite_half_edge(int halfEdge, std::vector<HalfEdge> halfEdges,
  std::map<Edge, std::pair<int, int>, compareTwoEdges> halfEdgesMap)
{
  HalfEdge he = halfEdges[halfEdge];
  std::pair<int, int> pair = halfEdgesMap[Edge{ he.vertex, halfEdges[he.next].vertex}];
  return halfEdges[pair.first] == he ? pair.second : pair.first;
}

double distance_two_points(Eigen::RowVector3d point1, Eigen::RowVector3d point2)
{
  return std::sqrt(std::pow(point1[0] - point2[0], 2) +
    std::pow(point1[1] - point2[1], 2) + std::pow(point1[2] - point2[2], 2));
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
  for (int i = 0; i < 4; i++)
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

std::vector<Edge> edges_from_vertex(int vertex,
  std::map<Edge, std::pair<int, int>, compareTwoEdges> halfEdgesMap)
{
  std::vector<Edge> edges;
  for (const auto& kv : halfEdgesMap)
  {
    if (kv.first[0] == vertex || kv.first[1] == vertex)
    {
      edges.push_back(kv.first);
    }
  }
  return edges;
}

std::vector<int> half_edges_from_vertex(int vertex, std::vector<HalfEdge> halfEdges, 
  std::map<Edge, std::pair<int, int>, compareTwoEdges> halfEdgesMap)
{
  std::vector<int> hes;
  for (int i = 0; i < halfEdges.size(); i++)
  {
    if (halfEdges[i].vertex == vertex)
    {
      hes.push_back(i);
      break;
    }
  }

  int heTmp = opposite_half_edge(
    halfEdges[halfEdges[halfEdges[hes[0]].next].next].next, halfEdges, halfEdgesMap);
  while (heTmp != hes[0])
  {
    hes.push_back(heTmp);
    heTmp = opposite_half_edge(
      halfEdges[halfEdges[halfEdges[heTmp].next].next].next, halfEdges, halfEdgesMap);
  }
  
  return hes;
}

std::vector<Edge> get_edges_from_he_map(
  std::map<Edge, std::pair<int, int>, compareTwoEdges> halfEdgesMap)
{
  std::vector<Edge> keys;
  for (auto it = halfEdgesMap.begin(); it != halfEdgesMap.end(); it++)
  {
    keys.push_back(it->first);
  }
  return keys;
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
      he.face--;
    }
  }

  // Remove the diagonals involved in the face's deletion
  for (int i = 0; i < diagonals.size(); i++)
  {
    if (Edge{ diagonals[i].first, diagonals[i].second } == Edge{ face[0], face[2] })
    {
      diagonals.erase(diagonals.begin() + i);
      break;
    }
  }
  for (int i = 0; i < diagonals.size(); i++)
  {
    if (Edge{ diagonals[i].first, diagonals[i].second } == Edge{ face[1], face[3] })
    {
      diagonals.erase(diagonals.begin() + i);
      break;
    }
  }
}

void remove_vertex(Eigen::MatrixXd& V, int rowIndex, Eigen::MatrixXi& F, 
  std::vector<HalfEdge>& halfEdges, std::map<Edge, std::pair<int, int>, 
  compareTwoEdges>& halfEdgesMap, std::vector<std::pair<int, int>>& diagonals,
  int vertexToBeMantained)
{
  Eigen::MatrixXd newV(V.rows() - 1, V.cols());
  newV << V.topRows(rowIndex), V.bottomRows(V.rows() - rowIndex - 1);
  V.conservativeResize(V.rows() - 1, Eigen::NoChange);
  V = newV;

  for (int i = 0; i < F.rows(); i++)
  {
    for (int j = 0; j < 4; j++)
    {
      if (F.row(i)[j] > rowIndex)
      {
        F.row(i)[j]--;
      }
    }
  }

  // Modify half-edges involved in the vertex's deletion 
  for (HalfEdge& he : halfEdges)
  {
    if (he.vertex > rowIndex)
    {
      he.vertex--;
    }
  }

  std::map<Edge, std::pair<int, int>, compareTwoEdges> newHalfEdgesMap;
  for (auto& kv : halfEdgesMap)
  {
    if (kv.first[0] > rowIndex && kv.first[1] > rowIndex)
    {
      newHalfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
        Edge{ kv.first[0] - 1, kv.first[1] - 1 }, kv.second));
    }
    else if (kv.first[0] > rowIndex)
    {
      newHalfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
        Edge{ kv.first[0] - 1, kv.first[1] }, kv.second));
    }
    else if (kv.first[1] > rowIndex)
    {
      newHalfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
        Edge{ kv.first[0], kv.first[1] - 1 }, kv.second));
    }
    else
    {
      newHalfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
        kv.first, kv.second));
    }
  }

  halfEdgesMap = newHalfEdgesMap;

  // Modify the diagonals involved in the vertex's deletion
  for (int i = 0; i < diagonals.size(); i++)
  {
    if (diagonals[i].first == rowIndex)
    {
      diagonals[i].first = vertexToBeMantained;
    }
    if (diagonals[i].second == rowIndex)
    {
      diagonals[i].second = vertexToBeMantained;
    }
  }

  for (int i = 0; i < diagonals.size(); i++)
  {
    if (diagonals[i].first > rowIndex)
    {
      diagonals[i].first--;
    }
    if (diagonals[i].second > rowIndex)
    {
      diagonals[i].second--;
    }
  }
}

void remove_half_edges(std::vector<int> heToBeRemoved, std::vector<HalfEdge>& halfEdges,
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap)
{
  std::sort(heToBeRemoved.begin(), heToBeRemoved.end());

  for (int i = heToBeRemoved.size() - 1; i >= 0; i--)
  {
    halfEdges.erase(halfEdges.begin() + heToBeRemoved[i]);
    for (HalfEdge& he : halfEdges)
    {
      if (he.next > heToBeRemoved[i])
      {
        he.next--;
      }
    }

    for (auto& kv : halfEdgesMap)
    {
      if (kv.second.first > heToBeRemoved[i])
      {
        kv.second.first--;
      }
      if (kv.second.second > heToBeRemoved[i])
      {
        kv.second.second--;
      }
    }
  }
}

void remove_doublet(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  int startingHalfEdge, std::vector<HalfEdge>& halfEdges, 
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap,
  std::vector<std::pair<int, int>>& diagonals)
{
  int vertexToBeDeleted = halfEdges[startingHalfEdge].vertex;
  
  int next1 = halfEdges[startingHalfEdge].next;
  int first2 = halfEdges[next1].next;
  int next2 = halfEdges[halfEdges[opposite_half_edge(
    startingHalfEdge, halfEdges, halfEdgesMap)].next].next;
  int first1 = halfEdges[next2].next;

  int vertexToBeMantained = halfEdges[first1].vertex;

  halfEdges[first1].next = next1;
  halfEdges[first2].next = next2;

  halfEdges[first1].face = halfEdges[startingHalfEdge].face;
  halfEdges[next2].face = halfEdges[startingHalfEdge].face;

  for (int i = 0; i < 4; i++)
  {
    if (F.row(halfEdges[startingHalfEdge].face)[i] == vertexToBeDeleted)
    {
      F.row(halfEdges[startingHalfEdge].face)[i] = halfEdges[first1].vertex;
    }
  }

  int secondHe = opposite_half_edge(startingHalfEdge, halfEdges, halfEdgesMap);
  int faceToBeRemoved = halfEdges[secondHe].face;
  int thirdHe = halfEdges[secondHe].next;
  int fouthHe = opposite_half_edge(thirdHe, halfEdges, halfEdgesMap);

  halfEdgesMap.erase(Edge{ vertexToBeDeleted, halfEdges[next1].vertex });
  halfEdgesMap.erase(Edge{ vertexToBeDeleted, halfEdges[next2].vertex });

  remove_half_edges({ startingHalfEdge, secondHe, thirdHe, fouthHe }, halfEdges, halfEdgesMap);
  
  remove_face(F, faceToBeRemoved, halfEdges, diagonals);

  remove_vertex(V, vertexToBeDeleted, F, halfEdges, halfEdgesMap, diagonals, 
    vertexToBeMantained);
}

bool try_edge_rotation(Edge edge, Eigen::MatrixXd& V, Eigen::MatrixXi& F,
  std::vector<HalfEdge>& halfEdges,
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap,
  std::vector<std::pair<int, int>>& diagonals)
{
  HalfEdge& firstHe1 = halfEdges[halfEdgesMap[edge].second];
  HalfEdge& firstHe2 = halfEdges[firstHe1.next];
  HalfEdge& firstHe3 = halfEdges[firstHe2.next];
  HalfEdge& firstHe4 = halfEdges[firstHe3.next];

  HalfEdge& secondHe1 = halfEdges[halfEdgesMap[edge].first];
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

  if (edgeClockLength < oldEdgeLength &&
    newDiag1Length < oldDiag3Length &&
    newDiag2Length < oldDiag4Length)
  {
    /* Clockwise edge rotation */

    Edge oldDiag1 = Edge{ firstHe3.vertex, firstHe1.vertex };
    Edge oldDiag2 = Edge{ secondHe3.vertex, secondHe1.vertex };

    // Modify the half-edges map
    halfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
      Edge{ firstHe4.vertex, secondHe4.vertex }, halfEdgesMap[edge]));
    halfEdgesMap.erase(edge);

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
    for (int i = 0; i < 4; i++)
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
    newDiag1Length < oldDiag4Length)
  {
    /* Counterclockwise edge rotation */

    Edge oldDiag1 = Edge{ firstHe4.vertex, firstHe2.vertex };
    Edge oldDiag2 = Edge{ secondHe4.vertex, secondHe2.vertex };

    // Modify the half-edges map
    halfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
      Edge{ firstHe3.vertex, secondHe3.vertex }, halfEdgesMap[edge]));
    halfEdgesMap.erase(edge);

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
    for (int i = 0; i < 4; i++)
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
  
  // Searching for doublets
  if (halfEdges[opposite_half_edge(halfEdges[opposite_half_edge(firstDoubletHe,
    halfEdges, halfEdgesMap)].next, halfEdges, halfEdgesMap)].next == firstDoubletHe)
  {
    // Doublet found
    int vertexToBeRemoved = halfEdges[firstDoubletHe].vertex;
    if (secondDoubletCheck.first > vertexToBeRemoved) secondDoubletCheck.first--;
    if (secondDoubletCheck.second > vertexToBeRemoved) secondDoubletCheck.second--;

    remove_doublet(V, F, firstDoubletHe, halfEdges, halfEdgesMap, diagonals);
  }

  std::pair<int, int> doubletHes = halfEdgesMap[Edge{ secondDoubletCheck.first,
    secondDoubletCheck.second }];
  int secondDoubletHe = halfEdges[doubletHes.first].vertex == secondDoubletCheck.first ?
    doubletHes.first : doubletHes.second;
  if (halfEdges[opposite_half_edge(halfEdges[opposite_half_edge(secondDoubletHe,
    halfEdges, halfEdgesMap)].next, halfEdges, halfEdgesMap)].next == secondDoubletHe)
  {
    // Doublet found
    remove_doublet(V, F, secondDoubletHe, halfEdges, halfEdgesMap, diagonals);
  }

  return true;
}

std::pair<bool, std::vector<int>> try_vertex_rotation(const int vert,
  Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::vector<HalfEdge>& halfEdges,
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap,
  std::vector<std::pair<int, int>>& diagonals, const bool forceRotation, 
  const int vertToBeMantained = -1)
{
  // The result will be used for cooperation during an edge collapse
  std::vector<int> verticesRemovedForDoublets;
  std::pair<bool, std::vector<int>> result = 
    std::make_pair(true, verticesRemovedForDoublets);
  
  Eigen::RowVector3d vertex = V.row(vert);
  std::vector<int> hes = half_edges_from_vertex(vert, halfEdges, halfEdgesMap);
  std::vector<std::pair<int, int>*> diagonalsInvolved;
  double edgesSum = 0.0, diagonalsSum = 0.0;
  for (int he : hes)
  {
    HalfEdge h = halfEdges[he];
    edgesSum += distance_two_points(vertex, V.row(halfEdges[h.next].vertex));
    diagonalsSum += distance_two_points(vertex,
      V.row(halfEdges[halfEdges[h.next].next].vertex));

    Eigen::RowVector4i face = F.row(halfEdges[he].face);
    bool endLoop = false;
    for (std::pair<int, int>& diagonal : diagonals)
    {
      if (Edge{ diagonal.first, diagonal.second } == Edge{ face[0], face[2] } ||
        Edge{ diagonal.first, diagonal.second } == Edge{ face[1], face[3] })
      {
        diagonalsInvolved.push_back(&diagonal);
        if (endLoop) break;
        endLoop = !endLoop;
      }
    }
  }

  // Check if the sum of the edge lengths overcomes the sum of the diagonals lengths
  if (edgesSum > diagonalsSum || forceRotation) // Vertex rotation
  {
    // Modify the faces involved in the rotation
    for (int he : hes)
    {
      HalfEdge h = halfEdges[he];
      for (int i = 0; i < 4; i++)
      {
        int tmp = halfEdges[h.next].next;
        int v = F.row(h.face)[i];
        if (v == halfEdges[h.next].vertex)
        {
          F.row(h.face)[i] = halfEdges[tmp].vertex;
        }
        else if (v == halfEdges[tmp].vertex)
        {
          F.row(h.face)[i] = halfEdges[halfEdges[tmp].next].vertex;
        }
        else if (v == halfEdges[halfEdges[tmp].next].vertex)
        {
          F.row(h.face)[i] = halfEdges[halfEdges[halfEdges[opposite_half_edge(
            halfEdges[tmp].next, halfEdges, halfEdgesMap)].next].next].vertex;
        }
      }
    }

    // Modify the the half-edges involved in the rotation
    int tmp4 = 0;
    for (int he : hes)
    {
      HalfEdge h = halfEdges[he];

      int tmp1 = halfEdges[h.next].next;
      int tmp2 = halfEdges[tmp1].next;
      int tmp3 = halfEdges[tmp2].next;
      int opposite1 = opposite_half_edge(tmp2, halfEdges, halfEdgesMap);
      int opposite2 = opposite_half_edge(tmp3, halfEdges, halfEdgesMap);

      if (he == hes.back())
      {
        halfEdges[tmp2].vertex = halfEdges[halfEdges[opposite1].next].vertex;
        halfEdges[tmp1].next = tmp4;
        halfEdges[h.next].face = halfEdges[opposite2].face;
        halfEdges[h.next].next = opposite2;
        halfEdges[tmp3].next = tmp1;
        break;
      }

      halfEdges[tmp2].vertex = halfEdges[halfEdges[halfEdges[opposite1].next].next].vertex;
      halfEdges[tmp1].next = halfEdges[opposite1].next;
      halfEdges[h.next].face = halfEdges[opposite2].face;
      halfEdges[h.next].next = opposite2;
      halfEdges[tmp3].next = tmp1;
      if (he == hes.front())
      {
        tmp4 = h.next;
      }
    }

    // Modify the half-edges map involved in the rotation
    for (int he : hes)
    {
      HalfEdge h = halfEdges[he];

      Edge edgeDeleted = Edge{ vert, halfEdges[halfEdges[h.next].next].vertex };
      std::pair<int, int> tmp = halfEdgesMap[edgeDeleted];
      halfEdgesMap.erase(edgeDeleted);
      halfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
        Edge{ vert, halfEdges[halfEdges[halfEdges[h.next].next].next].vertex }, tmp));
    }

    // Modify the diagonals involved in the rotation
    for (int he : hes)
    {
      for (std::pair<int, int>* diag : diagonalsInvolved)
      {
        std::pair<int, int> diagonal = *diag;
        HalfEdge oppositeHe = halfEdges[opposite_half_edge(he, halfEdges, halfEdgesMap)];
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
          Edge{ halfEdges[heTmp.next].vertex,
          halfEdges[halfEdges[halfEdges[oppositeHe.next].next].next].vertex })
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
      int startingHe = halfEdges[halfEdges[he].next].next;
      if (halfEdges[opposite_half_edge(halfEdges[opposite_half_edge(startingHe,
        halfEdges, halfEdgesMap)].next, halfEdges, halfEdgesMap)].next == startingHe)
      {
        // Doublet found
        result.second.push_back(halfEdges[startingHe].vertex);
        doubletVerts.push_back(std::make_pair(halfEdges[startingHe].vertex, 
          halfEdges[halfEdges[startingHe].next].vertex));
        if (halfEdges[startingHe].vertex == vertToBeMantained)
        {
          result.first = false; // Disable the possible next diagonal collapse
        }
      }
    }

    int oldFirstVert = -1;
    for (int i = 0; i < doubletVerts.size(); i++)
    {
      if (oldFirstVert != -1)
      {
        for (int j = i; j < doubletVerts.size(); j++)
        {
          if (doubletVerts[j].first > oldFirstVert) doubletVerts[j].first--;
          if (doubletVerts[j].second > oldFirstVert) doubletVerts[j].second--;
        }
      }
      oldFirstVert = doubletVerts[i].first;
    }

    for (std::pair<int, int> verts : doubletVerts)
    {
      std::pair<int, int> hs = halfEdgesMap[Edge{ verts.first, verts.second }];
      int startingHe = halfEdges[hs.first].vertex == verts.first ? hs.first : hs.second;
      remove_doublet(V, F, startingHe, halfEdges, halfEdgesMap, diagonals);
    }
  }

  return result;
}

// Edge rotation and vertex rotation
bool optimize_quad_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  std::vector<HalfEdge>& halfEdges,
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap,
  std::vector<std::pair<int, int>>& diagonals, 
  std::vector<Edge> involvedEdges, std::vector<int> involvedVertices)
{
  /* Edge rotation */
  for (Edge edge : involvedEdges)
  {
    bool success = try_edge_rotation(edge, V, F, halfEdges, halfEdgesMap, diagonals);
    if (!success) return false;
  }

  /* Vertex rotation */
  for (int vert : involvedVertices)
  {
    std::pair<bool, std::vector<int>> result = 
      try_vertex_rotation(vert, V, F, halfEdges, halfEdgesMap, diagonals, false);
  }

  return true;
}

bool diagonal_collapse(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  std::vector<HalfEdge>& halfEdges,
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap,
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
    assert(false && "An error occured while searching for a face during diag collapse.");
  }

  Edge edge = find_edge(face, vertexToBeRemoved);

  std::vector<int> heToBeModified;
  int initialHe = halfEdges[halfEdgesMap[edge].first].vertex == vertexToBeRemoved ?
    halfEdgesMap[edge].first : halfEdgesMap[edge].second;
  int finalHe = halfEdges[halfEdges[halfEdges[initialHe].next].next].next;

  for (int tmp = opposite_half_edge(initialHe, halfEdges, halfEdgesMap); tmp != finalHe; )
  {
    tmp = halfEdges[tmp].next;
    heToBeModified.push_back(tmp);
    tmp = opposite_half_edge(tmp, halfEdges, halfEdgesMap);
  }

  // Set new starting vertices for the half-edge to be modified
  for (int he : heToBeModified)
  {
    if (halfEdges[he].vertex != vertexToBeRemoved) return false;

    if (he != opposite_half_edge(finalHe, halfEdges, halfEdgesMap))
    {
      halfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
        Edge{ vertexToBeMantained, halfEdges[halfEdges[he].next].vertex },
        std::make_pair(he, opposite_half_edge(he, halfEdges, halfEdgesMap))));
      halfEdgesMap.erase(Edge{ vertexToBeRemoved, halfEdges[halfEdges[he].next].vertex });
    }

    halfEdges[he].vertex = vertexToBeMantained;

    for (int i = 0; i < 4; i++)
    {
      if (F.row(halfEdges[he].face)[i] == vertexToBeRemoved)
      {
        F.row(halfEdges[he].face)[i] = vertexToBeMantained;
      }
    }
  }

  // Set new opposite half-edges
  std::pair<int, int>* firstPair = &halfEdgesMap[Edge{
    halfEdges[halfEdges[initialHe].next].vertex,
    halfEdges[halfEdges[halfEdges[initialHe].next].next].vertex }];

  std::pair<int, int>* secondPair = &halfEdgesMap[Edge{
    halfEdges[halfEdges[halfEdges[initialHe].next].next].vertex,
    halfEdges[finalHe].vertex }];

  int firstOpposite = opposite_half_edge(initialHe, halfEdges, halfEdgesMap);
  int secondOpposite = opposite_half_edge(finalHe, halfEdges, halfEdgesMap);

  if ((*firstPair).first == halfEdges[initialHe].next)
  {
    (*firstPair).first = firstOpposite;
  }
  else
  {
    (*firstPair).second = firstOpposite;
  }

  if ((*secondPair).first == halfEdges[halfEdges[initialHe].next].next)
  {
    (*secondPair).first = secondOpposite;
  }
  else
  {
    (*secondPair).second = secondOpposite;
  }

  // Remove the half-edges belong to the face to be deleted
  halfEdgesMap.erase(edge);
  halfEdgesMap.erase(Edge{ vertexToBeRemoved, halfEdges[finalHe].vertex });

  std::vector<int> heToBeRemoved{ initialHe, halfEdges[initialHe].next,
    halfEdges[halfEdges[initialHe].next].next , finalHe };

  remove_half_edges(heToBeRemoved, halfEdges, halfEdgesMap);

  // Store vertices involved in the collapse for a possible future cleaning
  std::vector<int> vertsToBeCleaned;
  vertsToBeCleaned.push_back(vertexToBeMantained);
  for (int i = 0; i < 4; i++)
  {
    if (face[i] != vertexToBeRemoved && face[i] != vertexToBeMantained)
    {
      vertsToBeCleaned.push_back(face[i]);
    }
  }

  // Remove a quad face
  remove_face(F, faceIndex, halfEdges, diagonals);

  std::vector<int> verticesForCentroid;
  for (HalfEdge he : halfEdges)
  {
    if (he.vertex == vertexToBeMantained)
    {
      verticesForCentroid.push_back(halfEdges[he.next].vertex);
      verticesForCentroid.push_back(halfEdges[halfEdges[he.next].next].vertex);
    }
  }

  // Calculate the new vertex's position
  std::vector<int> faceVertices = { face[0], face[1], face[2], face[3] };
  V.row(vertexToBeMantained) = new_vertex_pos(V, verticesForCentroid, faceVertices);

  // Remove a vertex
  remove_vertex(V, vertexToBeRemoved, F, halfEdges, halfEdgesMap, 
    diagonals, vertexToBeMantained);

  for (int i = 0; i < vertsToBeCleaned.size(); i++)
  {
    if (vertsToBeCleaned[i] > vertexToBeRemoved)
    {
      vertsToBeCleaned[i]--;
    }
  }

  /* Cleaning operations */
  // Searching for doublets
  std::pair<int, int> p1 = halfEdgesMap[Edge{ vertsToBeCleaned[0], vertsToBeCleaned[1] }];
  std::pair<int, int> p2 = halfEdgesMap[Edge{ vertsToBeCleaned[0], vertsToBeCleaned[2] }];
  int startingHe1 = halfEdges[p1.first].vertex == vertsToBeCleaned[1] ? p1.first : p1.second;
  int startingHe2 = halfEdges[p2.first].vertex == vertsToBeCleaned[2] ? p2.first : p2.second;

  bool doublet1 = halfEdges[opposite_half_edge(halfEdges[opposite_half_edge(startingHe1,
    halfEdges, halfEdgesMap)].next, halfEdges, halfEdgesMap)].next == startingHe1;
  bool doublet2 = halfEdges[opposite_half_edge(halfEdges[opposite_half_edge(startingHe2,
    halfEdges, halfEdgesMap)].next, halfEdges, halfEdgesMap)].next == startingHe2;

  if (doublet1 && doublet2)
  {
    remove_doublet(V, F, startingHe1, halfEdges, halfEdgesMap, diagonals);

    int vertToBeCleaned = vertsToBeCleaned[2] > vertsToBeCleaned[1] ?
      vertsToBeCleaned[2] - 1 : vertsToBeCleaned[2];
    p2 = halfEdgesMap[Edge{ vertsToBeCleaned[0] > vertsToBeCleaned[1] ?
      vertsToBeCleaned[0] - 1 : vertsToBeCleaned[0], vertToBeCleaned }];
    startingHe2 = halfEdges[p2.first].vertex == vertToBeCleaned ? p2.first : p2.second;
    remove_doublet(V, F, startingHe2, halfEdges, halfEdgesMap, diagonals);
  }
  else if (doublet1)
  {
    remove_doublet(V, F, startingHe1, halfEdges, halfEdgesMap, diagonals);
  }
  else if (doublet2)
  {
    remove_doublet(V, F, startingHe2, halfEdges, halfEdgesMap, diagonals);
  }
}

bool edge_collapse(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  std::vector<HalfEdge>& halfEdges,
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap,
  std::vector<std::pair<int, int>>& diagonals, 
  int vertexToBeMantained, int vertexToBeRemoved)
{
  std::pair<bool, std::vector<int>> result = try_vertex_rotation(vertexToBeRemoved, V, F, 
    halfEdges, halfEdgesMap, diagonals, true, vertexToBeMantained); // Force rotation

  // Sort in descending order
  std::sort(result.second.begin(), result.second.end(), std::greater<int>());
  for (int vertexRemoved : result.second)
  {
    if (vertexToBeMantained > vertexRemoved) vertexToBeMantained--;
    if (vertexToBeRemoved > vertexRemoved) vertexToBeRemoved--;
  }

  if (result.first)
  {
    bool success = diagonal_collapse(V, F, halfEdges, halfEdgesMap, diagonals,
      vertexToBeRemoved, vertexToBeMantained);
    if (!success) return false;
  }

  return true;
}

// Edge collapse and diagonal collapse
bool coarsen_quad_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  std::vector<HalfEdge>& halfEdges,
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap, 
  std::vector<std::pair<int, int>>& diagonals, 
  std::vector<Edge> edges)
{
  // Search for the shortest element (edge or diagonal)
  MinHeapCompare minHeap{ V };
  std::make_heap(edges.begin(), edges.end(), minHeap);
  std::make_heap(diagonals.begin(), diagonals.end(), minHeap);

  Edge shortestEdge = edges.front();
  double shortestEdgeLength = distance_two_points(V.row(shortestEdge[0]), V.row(shortestEdge[1]));
  std::pair<int, int> shortestDiagonal = diagonals.front();
  double shortestDiagonalLength = distance_two_points(V.row(shortestDiagonal.first), 
    V.row(shortestDiagonal.second));

  int vertexMantained, vertexRemoved;
  if (shortestDiagonalLength / sqrt(2) < shortestEdgeLength)
  {
    /* Diagonal collapse */
    vertexMantained = shortestDiagonal.second;
    vertexRemoved = shortestDiagonal.first;
    bool success = diagonal_collapse(V, F, halfEdges, halfEdgesMap, diagonals, 
      vertexMantained, vertexRemoved);
    if (!success) return false;
  }
  else
  {
    /* Edge collapse */
    vertexMantained = shortestEdge[1];
    vertexRemoved = shortestEdge[0];
    bool success = edge_collapse(V, F, halfEdges, halfEdgesMap, diagonals,
      vertexMantained, vertexRemoved);
    if (!success) return false;
  }

  // Final local optimization
  int v = vertexMantained > vertexRemoved ? vertexMantained - 1 : vertexMantained;
  optimize_quad_mesh(V, F, halfEdges, halfEdgesMap, diagonals,
    edges_from_vertex(v, halfEdgesMap), std::vector<int>{ v });

  return true;
}

bool start_simplification(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int finalNumberOfFaces)
{
  // Used to navigate through half-edges
  std::vector<HalfEdge> halfEdges;

  // Used for opposite half-edges
  std::map<Edge, std::pair<int, int>, compareTwoEdges> halfEdgesMap;

  std::vector<std::pair<int, int>> diagonals;
  
  for (int i = 0; i < F.rows(); i++) // For each quad face
  {
    Eigen::RowVector4i face = F.row(i);

    diagonals.push_back(std::make_pair(face[0], face[2]));
    diagonals.push_back(std::make_pair(face[1], face[3]));


    std::vector<Edge> faceEdges = find_edges(face);
    // Retrieve edges and compute four half-edges
    for (int j = 0; j < 4; j++)
    {
      HalfEdge halfEdge = HalfEdge{
        face[j], // vertex
        i, // face
        (i * 4) + (int(halfEdges.size() + 1) % 4) // next
      };

      halfEdges.push_back(halfEdge);

      if (halfEdgesMap.find(faceEdges[j]) == halfEdgesMap.end())
      {
        // If the half-edge doesn't exist yet
        halfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
          faceEdges[j], std::make_pair(halfEdges.size() - 1, halfEdges.size() - 1)));
      }
      else
      {
        // If the half-edge exists
        std::pair<int, int>* hes = &halfEdgesMap[faceEdges[j]];
        (*hes).second = halfEdges.size() - 1;
      }
    }
  }
  
  // Initial global optimization of the quad mesh
  std::vector<int> vertices;
  for (int i = 0; i < V.rows(); i++)
  {
    vertices.push_back(i);
  }
  optimize_quad_mesh(V, F, halfEdges, halfEdgesMap, diagonals, 
    get_edges_from_he_map(halfEdgesMap), vertices);
  
  std::cout << F.rows() << "\n"; // TODO delete this line
  while (F.rows() > finalNumberOfFaces)
  { 
    coarsen_quad_mesh(V, F, halfEdges, halfEdgesMap, diagonals, 
      get_edges_from_he_map(halfEdgesMap));

    //std::cout << F.rows() << "\n"; // TODO delete this line
  }

  return true;
}

void per_quad_face_normals(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& N)
{
  N.resize(F.rows(), 3);
  // Loop over the faces
#pragma omp parallel for if (F.rows() > 1000)
  for (int i = 0; i < F.rows(); i++)
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
  for (int i = 0; i < F.rows(); i++)
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
  //std::cout << "\n\n" << J << "\n\n";

  //std::cout << "\n\n" << f << "\n\n";

  // 2) For every quad, fit a plane and get the normal from that plane
  Eigen::MatrixXd N;
  per_quad_face_normals(V, F, N);
  Eigen::MatrixXd fN(2 * N.rows(), 3);
  for (int i = 0; i < N.rows(); i++)
  {
    fN.row(i * 2) = N.row(i);
    fN.row(i * 2 + 1) = N.row(i);
  };
  //std::cout << "\n\n" << f << "\n\n";

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
