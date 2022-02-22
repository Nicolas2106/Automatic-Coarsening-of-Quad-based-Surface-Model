#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/fit_plane.h>
#include <igl/polygon_corners.h>
#include <igl/polygons_to_triangles.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_face_normals.h>
#include "tutorial_shared_path.h"

typedef struct HalfEdge {
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

int opposite_half_edge(int halfEdge, std::vector<HalfEdge>& halfEdges,
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap)
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

std::vector<Edge> find_edges(Eigen::RowVector4i face)
{
  std::vector<Edge> edges;
  for (int i = 0; i < 4; i++)
  {
    edges.push_back(Edge{ face[i], face[(i + 1) % 4] });
  }
  return edges;
}

Edge find_edge(Eigen::RowVector4i face, int startingPointIndex)
{
  std::vector<Edge> edges = find_edges(face);
  for (Edge edge : edges)
  {
    if (edge[0] == startingPointIndex) return edge;
  }
  assert(false && "An error occured while searching for an edge.");
}

void remove_face(Eigen::MatrixXi& F, int rowIndex, std::vector<HalfEdge>& halfEdges)
{
  Eigen::MatrixXi newF(F.rows() - 1, F.cols());
  newF << F.topRows(rowIndex), F.bottomRows(F.rows() - rowIndex - 1);
  F.conservativeResize(F.rows() - 1, Eigen::NoChange);
  F = newF;

  for (HalfEdge& he : halfEdges)
  {
    if (he.face > rowIndex)
    {
      he.face--;
    }
  }
}

void remove_vertex(Eigen::MatrixXd& V, int rowIndex, Eigen::MatrixXi& F, 
  std::vector<HalfEdge>& halfEdges, std::map<Edge, std::pair<int, int>, 
  compareTwoEdges>& halfEdgesMap)
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

void remove_doublet(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int startingHalfEdge, 
  int vertexToBeDeleted, std::vector<HalfEdge>& halfEdges, 
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap)
{
  int next1 = halfEdges[startingHalfEdge].next;
  int first2 = halfEdges[next1].next;
  int next2 = halfEdges[halfEdges[opposite_half_edge(
    startingHalfEdge, halfEdges, halfEdgesMap)].next].next;
  int first1 = halfEdges[next2].next;

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
  
  remove_face(F, faceToBeRemoved, halfEdges);

  remove_vertex(V, vertexToBeDeleted, F, halfEdges, halfEdgesMap);
}

bool diag_collapse(Eigen::MatrixXd& V, Eigen::MatrixXi& F, std::vector<HalfEdge>& halfEdges,
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap)
{
  int fToBeRemoved = 42; //TODO remove this line
  
  Eigen::RowVector4i face = F.row(fToBeRemoved);
  double diag1 = distance_two_points(V.row(face[0]), V.row(face[2]));
  double diag2 = distance_two_points(V.row(face[3]), V.row(face[1]));

  int firstVertex = diag1 < diag2 ? face[0] : face[3]; // To be mantained
  int secondVertex = diag1 < diag2 ? face[2] : face[1]; // To be removed

  std::vector<int> heToBeModified;

  Edge edge = find_edge(face, secondVertex);
  
  int initialHe = halfEdges[halfEdgesMap[edge].first].vertex == secondVertex ?
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
    if (halfEdges[he].vertex != secondVertex) return false;

    if (he != opposite_half_edge(finalHe, halfEdges, halfEdgesMap))
    {
      halfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
        Edge{ firstVertex, halfEdges[halfEdges[he].next].vertex },
        std::make_pair(he, opposite_half_edge(he, halfEdges, halfEdgesMap))));
      halfEdgesMap.erase(Edge{ secondVertex, halfEdges[halfEdges[he].next].vertex });
    }

    halfEdges[he].vertex = firstVertex;

    for (int i = 0; i < 4; i++)
    {
      if (F.row(halfEdges[he].face)[i] == secondVertex)
      {
        F.row(halfEdges[he].face)[i] = firstVertex;
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
  halfEdgesMap.erase(Edge{ secondVertex, halfEdges[finalHe].vertex });
  
  std::vector<int> heToBeRemoved{ initialHe, halfEdges[initialHe].next,
    halfEdges[halfEdges[initialHe].next].next , finalHe };

  remove_half_edges(heToBeRemoved, halfEdges, halfEdgesMap);

  // Store vertices involved in the collapse for a possible future cleaning
  std::vector<int> vertsToBeCleaned;
  vertsToBeCleaned.push_back(firstVertex);
  for (int i = 0; i < 4; i++)
  {
    if (face[i] != secondVertex && face[i] != firstVertex)
    {
      vertsToBeCleaned.push_back(face[i]);
    }
  }

  // Remove one quad face
  remove_face(F, fToBeRemoved, halfEdges);

  // Move one vertex and remove the other
  Eigen::RowVector3d newPos = 0.5 * V.row(firstVertex) + 0.5 * V.row(secondVertex);
  V.row(firstVertex) = newPos;
  remove_vertex(V, secondVertex, F, halfEdges, halfEdgesMap);

  for (int i = 0; i < vertsToBeCleaned.size(); i++)
  {
    if (vertsToBeCleaned[i] > secondVertex)
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
    remove_doublet(V, F, startingHe1, vertsToBeCleaned[1], halfEdges, halfEdgesMap);
    
    int vertToBeCleaned = vertsToBeCleaned[2] > vertsToBeCleaned[1] ? 
      vertsToBeCleaned[2] - 1 : vertsToBeCleaned[2];
    p2 = halfEdgesMap[Edge{ vertsToBeCleaned[0] > vertsToBeCleaned[1] ? 
      vertsToBeCleaned[0] - 1 : vertsToBeCleaned[0], vertToBeCleaned }];
    startingHe2 = halfEdges[p2.first].vertex == vertToBeCleaned ? p2.first : p2.second;
    remove_doublet(V, F, startingHe2, vertToBeCleaned, halfEdges, halfEdgesMap);
  }
  else if (doublet1)
  {
    remove_doublet(V, F, startingHe1, vertsToBeCleaned[1], halfEdges, halfEdgesMap);
  }
  else if (doublet2)
  {
    remove_doublet(V, F, startingHe2, vertsToBeCleaned[2], halfEdges, halfEdgesMap);
  }
  
  /*for (auto& kv : halfEdgesMap)
  {
    std::cout << "\n" << kv.first << " --> " <<
      halfEdges[kv.second.first].vertex << " -- " << halfEdges[kv.second.first].face << " -- " << halfEdges[kv.second.first].next << " || " <<
      halfEdges[kv.second.second].vertex << " -- " << halfEdges[kv.second.second].face << " -- " << halfEdges[kv.second.second].next << "\n";
  }*/

  return true;
}

bool start_simplification(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int finalNumberOfFaces)
{
  // Used to navigate through half-edges
  std::vector<HalfEdge> halfEdges;

  // Used for opposite half-edges
  std::map<Edge, std::pair<int, int>, compareTwoEdges> halfEdgesMap;
  
  for (int i = 0; i < F.rows(); i++) // For each quad face
  {
    Eigen::RowVector4i face = F.row(i);

    std::vector<Edge> edges = find_edges(face);
    // Compute four half-edges
    for (int j = 0; j < 4; j++)
    {
      HalfEdge halfEdge = HalfEdge{
        face[j], // vertex
        i, // face
        (i * 4) + (int(halfEdges.size() + 1) % 4) // next
      };

      halfEdges.push_back(halfEdge);

      if (halfEdgesMap.find(edges[j]) == halfEdgesMap.end()) // The half-edge doesn't exist yet
      {
        halfEdgesMap.insert(std::pair<Edge, std::pair<int, int>>(
          edges[j], std::make_pair(halfEdges.size() - 1, halfEdges.size() - 1)));
      }
      else // The half-edge exists
      {
        std::pair<int, int>* hes = &halfEdgesMap[edges[j]];
        (*hes).second = halfEdges.size() - 1;
      }
    }
  }

  /*for (auto& kv : halfEdgesMap)
  {
    std::cout << "\n" << kv.first << " --> " << 
      halfEdges[kv.second.first].vertex << " -- " << halfEdges[kv.second.first].face << " -- " << halfEdges[kv.second.first].next << " || " <<
      halfEdges[kv.second.second].vertex << " -- " << halfEdges[kv.second.second].face << " -- " << halfEdges[kv.second.second].next << "\n";
  }*/

  for (int i = 0; i < 26; i++)
  {
    if (!diag_collapse(V, F, halfEdges, halfEdgesMap))
    {
      return false;
    }
    std::cout << i << "\n";
  }

  /*Eigen::MatrixXi newF(10, F.cols());
  newF << F.topRows(10);
  F.conservativeResize(10, Eigen::NoChange);
  F = newF;

  std::cout << "\n\n" << F << "\n\n";*/

  //remove_face(F, 46);

  return true;
}

void per_quad_face_normals(const Eigen::MatrixXd & V, const Eigen::MatrixXi & F, Eigen::MatrixXd & N)
{
  N.resize(F.rows(), 3);
  // loop over faces
  int Frows = F.rows();
#pragma omp parallel for if (Frows>10000)
  for (int i = 0; i < Frows; i++)
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
    Eigen::Matrix<Eigen::MatrixXd::Scalar, 1, 3> v1 = V.row(F(i, 1)) - V.row(F(i, 0));
    Eigen::Matrix<Eigen::MatrixXd::Scalar, 1, 3> v2 = V.row(F(i, 2)) - V.row(F(i, 0));
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
  //igl::readOFF(MESHES_DIR + "quad_surface.off", V, F);
  igl::readOBJ(MESHES_DIR + "quad_cubespikes.obj", V, F);

  if (start_simplification(V, F, 9))
  {
    draw_quad_mesh(V, F);
  }
  else
  {
    std::cout << "\n\n" << "ERROR occured during quad mesh simplification" << "\n\n";
  }
}
