#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/fit_plane.h>
#include <igl/polygon_corners.h>
#include <igl/polygons_to_triangles.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_face_normals.h>
#include <igl/doublearea.h>
#include "tutorial_shared_path.h"

struct HalfEdge {
  int vertex; // Index of the vertex from which the half-edge starts
  int face; // Index of the face
  int next; // Index of the next half-edge
};

typedef Eigen::RowVector2i Edge;

bool operator==(const Edge e1, const Edge e2)
{
  int e10 = e1[0], e11 = e1[1], e20 = e2[0];
  return e10 == e20 ? e11 == e2[1] : (e11 == e20 ? e10 == e2[1] : false);
}

struct compareTwoEdges {
  bool operator()(Edge e1, Edge e2) const {
    int min1 = std::min(e1[0], e1[1]), min2 = std::min(e2[0], e2[1]);
    return min1 == min2 ? std::max(e1[0], e1[1]) < std::max(e2[0], e2[1]) : min1 < min2;
  }
};

double squared_distance(Eigen::RowVector3d point1, Eigen::RowVector3d point2)
{
  double deltaX = point1[0] - point2[0],
    deltaY = point1[1] - point2[1],
    deltaZ = point1[2] - point2[2];
  return deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;
}

typedef struct {
  Eigen::MatrixXd V;
  
  // For diagonals
  bool operator()(const std::pair<int, int> d1, const std::pair<int, int> d2) const {
    return squared_distance(V.row(d1.first), V.row(d1.second)) >
      squared_distance(V.row(d2.first), V.row(d2.second));
  }

  // For edges
  bool operator()(const Edge e1, const Edge e2) const {
    return squared_distance(V.row(e1[0]), V.row(e1[1])) >
      squared_distance(V.row(e2[0]), V.row(e2[1]));
  }
} MinHeapCompare;

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
  
  double proj = (centroid - pointOnPlane).dot(normal);
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
  
  int size = F.rows() - 1;
  Eigen::MatrixXi newF(size, F.cols());
  newF << F.topRows(rowIndex), F.bottomRows(size - rowIndex);
  F.conservativeResize(size, Eigen::NoChange);
  F = newF;

  // Modify half-edges due to the face's deletion 
  for (HalfEdge& he : halfEdges)
  {
    if (he.face > rowIndex)
    {
      --he.face;
    }
  }

  // Remove the two diagonals involved in the face's deletion
  Edge edge1 = Edge{ face[0], face[2] };
  Edge edge2 = Edge{ face[1], face[3] };

  size = diagonals.size();
  for (int i = 0; i < size; ++i)
  {
    if (Edge{ diagonals[i].first, diagonals[i].second } == edge1)
    {
      diagonals.erase(diagonals.begin() + i);
      break;
    }
  }
  
  // One diagonal deleted. It's time to delete the second one
  --size;
  for (int i = 0; i < size; ++i)
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
  int size = V.rows() - 1;
  Eigen::MatrixXd newV(size, V.cols());
  newV << V.topRows(rowIndex), V.bottomRows(size - rowIndex);
  V.conservativeResize(size, Eigen::NoChange);
  V = newV;

  // Modify the vertices' indices inside the matrix F
  size = F.rows();
  for (int i = 0; i < size; ++i)
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
    if (he.vertex > rowIndex) --he.vertex;
  }

  std::vector<Edge> es(edges.begin(), edges.end());
  for (Edge& e : es)
  {
    if (e[0] > rowIndex) { --e[0]; }
    if (e[1] > rowIndex) { --e[1]; }
  }
  std::set<Edge, compareTwoEdges> newSet(es.begin(), es.end());
  std::swap(edges, newSet);

  // Modify the diagonals involved in the vertex's deletion
  for (std::pair<int, int>& diag : diagonals)
  {
    if (diag.first == rowIndex) { diag.first = substituteVertex; }

    if (diag.second == rowIndex) { diag.second = substituteVertex; }

    if (diag.first > rowIndex) { --diag.first; }

    if (diag.second > rowIndex) { --diag.second; }
  }
}

void remove_half_edges(std::vector<int> hesToBeRemoved, std::vector<HalfEdge>& halfEdges)
{
  std::sort(hesToBeRemoved.begin(), hesToBeRemoved.end(), std::greater<int>());

  int size = hesToBeRemoved.size();
  for (int i = 0; i < size; ++i)
  {
    int heToBeRemoved = hesToBeRemoved[i];

    halfEdges.erase(std::next(halfEdges.begin(), heToBeRemoved));
    for (HalfEdge& he : halfEdges)
    {
      if (he.next > heToBeRemoved) { --he.next; }
    }
  }
}

void remove_doublet(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
  int startingHalfEdge, std::vector<HalfEdge>& halfEdges, 
  std::set<Edge, compareTwoEdges>& edges, std::vector<std::pair<int, int>>& diagonals)
{
  int vertexToBeDeleted = halfEdges[startingHalfEdge].vertex;
  int otherStartingHalfEdge = -1;
  int size = halfEdges.size();
  for (int i = 0; i < size; ++i)
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

  int secondHe = halfEdges[first1].next;
  int fouthHe = halfEdges[first2].next;

  halfEdges[first1].next = next1;
  halfEdges[first2].next = next2;

  int face = halfEdges[startingHalfEdge].face;
  halfEdges[first1].face = face;
  halfEdges[next2].face = face;

  int vertexToBeMantained = halfEdges[first1].vertex;
  for (int i = 0; i < 4; ++i)
  {
    if (F.row(face)[i] == vertexToBeDeleted)
    {
      F.row(face)[i] = vertexToBeMantained;
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
  
  // Half-edges belong to the first face incident to the edge
  HalfEdge& firstHe1 = halfEdges[pair.second];
  HalfEdge& firstHe2 = halfEdges[firstHe1.next];
  HalfEdge& firstHe3 = halfEdges[firstHe2.next];
  HalfEdge& firstHe4 = halfEdges[firstHe3.next];

  // Half-edges belong to the second face incident to the edge
  HalfEdge& secondHe1 = halfEdges[pair.first];
  HalfEdge& secondHe2 = halfEdges[secondHe1.next];
  HalfEdge& secondHe3 = halfEdges[secondHe2.next];
  HalfEdge& secondHe4 = halfEdges[secondHe3.next];

  int vF1 = firstHe1.vertex, vS1 = secondHe1.vertex,
    vF3 = firstHe3.vertex, vS3 = secondHe3.vertex,
    vF4 = firstHe4.vertex, vS4 = secondHe4.vertex;
  
  Eigen::RowVector3d vertF1 = V.row(vF1), vertS1 = V.row(vS1),
    vertF3 = V.row(vF3), vertS3 = V.row(vS3),
    vertF4 = V.row(vF4), vertS4 = V.row(vS4);

  // Edge rotation is profitable if it shortens the rotated edge and both the diagonals
  double oldEdgeLength = squared_distance(vertF1, vertS1),
    oldDiag1Length = squared_distance(vertF4, vertS1),
    oldDiag2Length = squared_distance(vertF1, vertS4),
    oldDiag3Length = squared_distance(vertF1, vertF3),
    oldDiag4Length = squared_distance(vertS3, vertS1);

  double edgeClockLength = squared_distance(vertF4, vertS4),
    edgeCounterLength = squared_distance(vertF3, vertS3),
    newDiag1Length = squared_distance(vertF3, vertS4),
    newDiag2Length = squared_distance(vertF4, vertS3);

  int firstDoubletHe = firstHe1.next;
  std::pair<int, int> secondDoubletCheck = std::make_pair(secondHe2.vertex, vS3);

  bool rotation = false;

  if (edgeClockLength < oldEdgeLength &&
    newDiag1Length < oldDiag3Length &&
    newDiag2Length < oldDiag4Length) // Clockwise edge rotation
  {
    rotation = true;

    // Modify the edge
    edges.emplace(Edge{ vF4, vS4 });
    edges.erase(edge);

    // Modify the half-edges involved in the rotation
    firstHe1.vertex = vF4;
    int tmpFirstHe = firstHe1.next;
    firstHe1.next = secondHe3.next;
    secondHe1.vertex = vS4;
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
        F.row(firstHe1.face)[i] = vS4;
      }
      if (F.row(secondHe1.face)[i] == firstHe2.vertex)
      {
        F.row(secondHe1.face)[i] = vF4;
      }
    }

    // Modify the diagonals involved in the rotation
    Edge oldDiag1 = Edge{ vF3, vF1 },
      oldDiag2 = Edge{ vS3, vS1 };
    for (std::pair<int, int>& diag : diagonals)
    {
      if (Edge{ diag.first, diag.second } == oldDiag1)
      {
        if (diag.first == vF1)
          diag.first = vS4;
        else
          diag.second = vS4;
      }
      else if (Edge{ diag.first, diag.second } == oldDiag2)
      {
        if (diag.first == vS1)
          diag.first = vF4;
        else
          diag.second = vF4;
      }
    }
  }
  else if (edgeCounterLength < oldEdgeLength &&
    newDiag2Length < oldDiag3Length &&
    newDiag1Length < oldDiag4Length) // Counterclockwise edge rotation
  {
    rotation = true;
    int vF2 = firstHe2.vertex;
    int vS2 = secondHe2.vertex;

    // Modify the edge
    edges.emplace(Edge{ vF3, vS3 });
    edges.erase(edge);

    // Modify the half-edges involved in the rotation
    firstHe1.vertex = vS3;
    int tmpFirstHe = firstHe1.next;
    firstHe1.next = firstHe2.next;
    secondHe1.vertex = vF3;
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
      if (F.row(firstHe1.face)[i] == vF2)
      {
        F.row(firstHe1.face)[i] = vS3;
      }
      if (F.row(secondHe1.face)[i] == vS2)
      {
        F.row(secondHe1.face)[i] = vF3;
      }
    }

    // Modify the diagonals involved in the rotation
    Edge oldDiag1 = Edge{ vF4, vF2 },
      oldDiag2 = Edge{ vS4, vS2 };
    for (std::pair<int, int>& diag : diagonals)
    {
      if (Edge{ diag.first, diag.second } == oldDiag1)
      {
        if (diag.first == vF2)
          diag.first = vS3;
        else
          diag.second = vS3;
      }
      else if (Edge{ diag.first, diag.second } == oldDiag2)
      {
        if (diag.first == vS2)
          diag.first = vF3;
        else
          diag.second = vF3;
      }
    }
  }
  
  if (rotation)
  {
    // Searching for the two possible doublets created after the rotation
    int countEdges = 0, size = halfEdges.size();
    int vert = halfEdges[firstDoubletHe].vertex;
    for (int i = 0; i < size; ++i)
    {
      if (halfEdges[i].vertex == vert) ++countEdges;
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
    int secondDoubletHe = halfEdges[doubletHes.first].vertex == 
      secondDoubletCheck.first ? doubletHes.first : doubletHes.second;
    
    vert = halfEdges[secondDoubletHe].vertex;
    countEdges = 0;
    size = halfEdges.size();
    for (int i = 0; i < size; ++i)
    {
      if (halfEdges[i].vertex == vert) ++countEdges;
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
    edgesSum += squared_distance(vertex, V.row(halfEdges[h.next].vertex));
    diagonalsSum += squared_distance(vertex,
      V.row(halfEdges[halfEdges[h.next].next].vertex));

    Eigen::RowVector4i face = F.row(h.face);
    bool endLoop = false;
    for (std::pair<int, int>& diagonal : diagonals)
    {
      int first = diagonal.first, second = diagonal.second;
      if ((first == face[0] && second == face[2]) || 
        (second == face[0] && first == face[2]) ||
        (first == face[1] && second == face[3]) || 
        (second == face[1] && first == face[3]))
      {
        diagonalsInvolved.push_back(&diagonal);
        if (endLoop) break;

        endLoop = true;
      }
    }
  }

  // Check if the sum of the edge lengths overcomes the sum of the diagonals lengths
  if (forceRotation || edgesSum > diagonalsSum) // Vertex rotation
  {
    int size = hes.size();
    // Modify the faces involved in the rotation
    for (int i = 0; i < size; ++i)
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
            hes[(i + 1) % size]].next].next].vertex;
        }
      }
    }

    // Modify the the half-edges involved in the rotation
    int tmp4 = 0, opposite2;
    for (int i = 0; i < size; ++i)
    {
      int he = hes[i],
        tmp = halfEdges[he].next,
        tmp1 = halfEdges[tmp].next,
        tmp2 = halfEdges[tmp1].next,
        opposite1 = hes[(i + 1) % size];
      if (i == 0)
      {
        opposite2 = halfEdges[halfEdges[halfEdges[hes[size - 1]].next].next].next;
      }

      if (i == size - 1)
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

    // Modify the edges involved in the rotation
    for (int he : hes)
    {
      HalfEdge h = halfEdges[halfEdges[halfEdges[he].next].next];
      edges.erase(Edge{ vert, h.vertex });
      edges.emplace(Edge{ vert, halfEdges[h.next].vertex });
    }

    // Modify the diagonals involved in the rotation
    for (int i = 0; i < size; ++i)
    {
      int he = hes[i];
      for (std::pair<int, int>* diag : diagonalsInvolved)
      {
        std::pair<int, int> diagonal = *diag;
        int v = halfEdges[halfEdges[halfEdges[hes[(size + i - 1) % size]]
          .next].next].vertex;
        HalfEdge heTmp = halfEdges[halfEdges[he].next];

        if (Edge{ diagonal.first, diagonal.second } == 
          Edge{ halfEdges[he].vertex, heTmp.vertex })
        {
          if (diagonal.first == vert)
            (*diag).first = halfEdges[halfEdges[heTmp.next].next].vertex;
          else
            (*diag).second = halfEdges[halfEdges[heTmp.next].next].vertex;
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
      HalfEdge heTmp = halfEdges[halfEdges[halfEdges[he].next].next];
      int startingHeVertex = heTmp.vertex;
      int countEdges = 0, size = halfEdges.size();
      for (int i = 0; i < size; ++i)
      {
        if (halfEdges[i].vertex == startingHeVertex) ++countEdges;
      }
      if (countEdges == 2)
      {
        // Doublet found
        result.second.push_back(startingHeVertex);
        doubletVerts.push_back(std::make_pair(startingHeVertex, 
          halfEdges[heTmp.next].vertex));
        
        if (startingHeVertex == vertToBeMantained)
        {
          result.first = false; // Disable the possible next diagonal collapse
        }
      }
    }

    int oldFirstVert = -1;
    size = doubletVerts.size();
    for (int i = 0; i < size; ++i)
    {
      if (oldFirstVert != -1)
      {
        for (int j = i; j < size; ++j)
        {
          if (doubletVerts[j].first > oldFirstVert) --doubletVerts[j].first;
          if (doubletVerts[j].second > oldFirstVert) --doubletVerts[j].second;
        }
      }
      oldFirstVert = doubletVerts[i].first;
    }

    for (std::pair<int, int> verts : doubletVerts)
    {
      std::pair<int, int> hs = 
        half_edges_from_edge(halfEdges, Edge{ verts.first, verts.second });
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
    if (!try_edge_rotation(edge, V, F, halfEdges, edges, diagonals))
    {
      return false;
    }
  }

  /* Vertex rotation */
  for (int vert : involvedVertices)
  {
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

  Edge edge{ -1, -1 };
  for (int i = 0; i < 4; ++i)
  {
    if (face[i] == vertexToBeRemoved) 
    { 
      edge = { face[i], face[(i + 1) % 4] };
      break;
    }
  }
  if (edge[0] == -1) return false;

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
  
  int v1 = halfEdges[finalHe].vertex, v2 = halfEdges[initialHe].vertex;
  std::vector<int> heToBeModified = half_edges_from_vertex(v2, halfEdges);

  std::vector<int>::iterator end = heToBeModified.end();
  heToBeModified.erase(std::remove(heToBeModified.begin(), end, initialHe), end);
  std::reverse(heToBeModified.begin(), heToBeModified.end());

  // Set new starting vertices for the half-edge to be modified
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
        int tmp = halfEdges[halfEdges[halfEdges[h].next].next].next;
        if (h != he && v == halfEdges[tmp].vertex)
        {
          heOpposite = tmp;
          break;
        }
      }
      edges.emplace(Edge{ vertexToBeMantained, v });
      edges.erase(Edge{ vertexToBeRemoved, v });
    }

    halfEdges[he].vertex = vertexToBeMantained;

    for (int i = 0; i < 4; ++i)
    {
      int f = halfEdges[he].face;
      if (F.row(f)[i] == vertexToBeRemoved)
      {
        F.row(f)[i] = vertexToBeMantained;
      }
    }
  }

  // Remove the half-edges belong to the face to be deleted
  edges.erase(edge);
  edges.erase(Edge{ vertexToBeRemoved, v1 });

  std::vector<int> hesToBeRemoved{ initialHe, halfEdges[initialHe].next,
    halfEdges[halfEdges[initialHe].next].next , finalHe };
  remove_half_edges(hesToBeRemoved, halfEdges);

  // Store vertices involved in the collapse for a possible future cleaning
  std::vector<int> vertsToBeCleaned;
  vertsToBeCleaned.push_back(vertexToBeMantained);
  for (int i = 0; i < 4; ++i)
  {
    int f = face[i];
    if (f != vertexToBeRemoved && f != vertexToBeMantained)
    {
      vertsToBeCleaned.push_back(f);
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

  int size = vertsToBeCleaned.size();
  for (int i = 0; i < size; ++i)
  {
    if (vertsToBeCleaned[i] > vertexToBeRemoved) --vertsToBeCleaned[i];
  }

  /* Cleaning operations */
  // Searching for doublets
  int v0Cleaned = vertsToBeCleaned[0],
    v1Cleaned = vertsToBeCleaned[1],
    v2Cleaned = vertsToBeCleaned[2];

  std::pair<int, int> p1 = half_edges_from_edge(halfEdges, Edge{ v0Cleaned, v1Cleaned });
  std::pair<int, int> p2 = half_edges_from_edge(halfEdges, Edge{ v0Cleaned, v2Cleaned });
  int startingHe1 = halfEdges[p1.first].vertex == v1Cleaned ? p1.first : p1.second;
  int startingHe2 = halfEdges[p2.first].vertex == v2Cleaned ? p2.first : p2.second;

  int countEdges1 = 0, countEdges2 = 0;
  int vert1 = halfEdges[startingHe1].vertex,
    vert2 = halfEdges[startingHe2].vertex;
  size = halfEdges.size();
  for (int i = 0; i < size; ++i)
  {
    int v = halfEdges[i].vertex;

    if (v == vert1) ++countEdges1;
    else if (v == vert2) ++countEdges2;
  }
 
  bool doublet1 = (countEdges1 == 2), doublet2 = (countEdges2 == 2);

  if (doublet1 && doublet2)
  {
    // Remove first doublet
    remove_doublet(V, F, startingHe1, halfEdges, edges, diagonals);

    // Remove second doublet
    int vertToBeCleaned = v2Cleaned > v1Cleaned ? v2Cleaned - 1 : v2Cleaned;
    p2 = half_edges_from_edge(halfEdges, Edge{ v0Cleaned > v1Cleaned ?
      v0Cleaned - 1 : v0Cleaned, vertToBeCleaned });
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
  std::pair<bool, std::vector<int>> result = try_vertex_rotation(vertexToBeRemoved, 
    V, F, halfEdges, edges, diagonals, true, vertexToBeMantained); // Force rotation

  // The diagonal collapse should not be done if the previuos checking for doublets
  // (during vertex rotation) removed the vertex to be used by the collapse
  if (result.first)
  {
    // Possible vertices removed due to the doublets created during vertex rotation
    std::vector<int> verticesRemoved = result.second;
    if (!verticesRemoved.empty())
    {
      // Sort in descending order
      std::sort(verticesRemoved.begin(), verticesRemoved.end(), std::greater<int>());
      for (int vertexRemoved : verticesRemoved)
      {
        if (vertexToBeMantained > vertexRemoved) --vertexToBeMantained;
        if (vertexToBeRemoved > vertexRemoved) --vertexToBeRemoved;
      }
    }

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
  double currEdgeLength, shortestEdgeLength = 
    squared_distance(vertices[shortestEdge[0]], vertices[shortestEdge[1]]);
  for (Edge e : edges)
  {
    currEdgeLength = squared_distance(vertices[e[0]], vertices[e[1]]);
    if (currEdgeLength < shortestEdgeLength)
    {
      shortestEdgeLength = currEdgeLength;
      shortestEdge = e;
    }
  }

  // Search for the shortest diagonal
  std::pair<int, int> shortestDiag = diagonals[0];
  double currDiagonalLength, shortestDiagonalLength =
    squared_distance(vertices[shortestDiag.first], vertices[shortestDiag.second]);
  for (std::pair<int, int> d : diagonals)
  {
    currDiagonalLength = squared_distance(vertices[d.first], vertices[d.second]);
    if (currDiagonalLength < shortestDiagonalLength)
    {
      shortestDiagonalLength = currDiagonalLength;
      shortestDiag = d;
    }
  }

  int vertexMantained, vertexRemoved;
  if (shortestDiagonalLength < shortestEdgeLength * 2)
  {
    /* Diagonal collapse */
    vertexMantained = shortestDiag.second;
    vertexRemoved = shortestDiag.first;
    if (!diagonal_collapse(V, F, halfEdges, edges, diagonals, vertexMantained, vertexRemoved))
    {
      return false;
    }
  }
  else
  {
    /* Edge collapse */
    vertexMantained = shortestEdge[1];
    vertexRemoved = shortestEdge[0];
    if (!edge_collapse(V, F, halfEdges, edges, diagonals, vertexMantained, vertexRemoved))
    {
      return false;
    }
  }

  // Final local optimization
  if (vertexMantained > vertexRemoved) { --vertexMantained; }
  std::vector<Edge> edgesFromVertex;
  for (HalfEdge e : halfEdges)
  {
    if (e.vertex == vertexMantained)
    {
      edgesFromVertex.push_back(Edge{ e.vertex, halfEdges[e.next].vertex });
    }
  }
  if (!optimize_quad_mesh(V, F, halfEdges, edges, diagonals, 
    edgesFromVertex, { vertexMantained }))
  {
    return false;
  }

  return true;
}

bool start_simplification(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int finalNumberOfFaces)
{
  // Used to navigate through half-edges
  std::vector<HalfEdge> halfEdges;

  // The set of the quad mesh's egdes
  std::set<Edge, compareTwoEdges> edges;

  // The set of the quad mesh's diagonals
  std::vector<std::pair<int, int>> diagonals;
  
  int facesCount = F.rows();
  for (int i = 0; i < facesCount; ++i) // For each quad face
  {
    Eigen::RowVector4i face = F.row(i);

    diagonals.push_back(std::make_pair(face[0], face[2]));
    diagonals.push_back(std::make_pair(face[1], face[3]));

    std::vector<Edge> faceEdges = {
      Edge{ face[0], face[1] },
      Edge{ face[1], face[2] },
      Edge{ face[2], face[3] },
      Edge{ face[3], face[0] }
    };
    
    // Retrieve edges and compute four half-edges per face
    int hesSize = halfEdges.size();
    for (int j = 0; j < 4; ++j)
    {
      halfEdges.push_back(HalfEdge{
          face[j], // vertex
          i, // face
          (i * 4) + (++hesSize % 4) // next
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
    bool success = coarsen_quad_mesh(V, F, halfEdges, edges, diagonals);
    std::cout << F.rows() << "\n"; // // TODO delete
    if (!success) return false;
  }

  return true;
}

void per_quad_face_normals(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& N)
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

void quad_corners(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::VectorXi& I, Eigen::VectorXi& C)
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
    "F:\\Users\\Nicolas\\Desktop\\TESI\\Quadrilateral extension to libigl\\libigl\\tutorial\\102_DrawMesh\\";

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  // Load a mesh
  //igl::readOFF(MESHES_DIR + "quad_surface.off", V, F);
  igl::readOBJ(MESHES_DIR + "quad_cubespikes.obj", V, F);

  std::cout << "Quad mesh coarsening in progress...\n\n";
  auto start_time = std::chrono::high_resolution_clock::now(); // TODO delete
  if (start_simplification(V, F, 300))
  {
    auto end_time = std::chrono::high_resolution_clock::now(); // TODO delete
    std::cout << "\n" << (end_time - start_time) / std::chrono::milliseconds(1) << " milliseconds\n\n"; // TODO delete
    std::cout << "Quad mesh drawing in progress...\n\n";
    draw_quad_mesh(V, F);
  }
  else
  {
    std::cout << "\n\n" << "ERROR occured during quad mesh simplification\n\n";
  }
}
