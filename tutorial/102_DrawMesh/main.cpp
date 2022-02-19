#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/fit_plane.h>
#include <igl/polygon_corners.h>
#include <igl/polygons_to_triangles.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_face_normals.h>
#include "tutorial_shared_path.h"
#include <string>

Eigen::MatrixXd V;
Eigen::MatrixXi F;

typedef struct HalfEdge {
  int vertex; // Index of the vertex from which the half-edge starts
  int face; // Index of the face
  int next; // Index of the next half-edge
} HalfEdge;

bool operator==(const HalfEdge& he1, const HalfEdge& he2)
{
  return he1.vertex == he2.vertex && he1.face == he2.face;
}

bool operator!=(const HalfEdge& he1, const HalfEdge& he2)
{
  return he1.vertex != he2.vertex || he1.face != he2.face;
}

typedef Eigen::RowVector2i Edge;
/*bool operator==(const Edge e1, const Edge e2)
{
  return (e1[0] == e2[0] && e1[1] == e2[1]) || 
    (e1[1] == e2[0] && e1[0] == e2[1]);
}*/

struct compareTwoEdges {
  bool operator()(const Edge e1, const Edge e2) const {
    return std::stoi(std::to_string(std::min(e1[0], e1[1])) + 
      std::to_string(std::max(e1[0], e1[1]))) <
      std::stoi(std::to_string(std::min(e2[0], e2[1])) + 
      std::to_string(std::max(e2[0], e2[1])));
  }
};

// Used to navigate through half-edges
std::vector<HalfEdge> halfEdges;

// Used for opposite half-edges
std::map<Edge, std::pair<int, int>, compareTwoEdges> halfEdgesMap;

int opposite_half_edge(int halfEdge)
{
  HalfEdge he = halfEdges[halfEdge];
  std::pair<int, int> pair = halfEdgesMap[Edge{ he.vertex,
    halfEdges[he.next].vertex}];
  return halfEdges[pair.first] == he ? pair.second : pair.first;
}

double distance_two_points(Eigen::RowVector3d point1, Eigen::RowVector3d point2)
{
  return std::sqrt(std::pow(point1[0] - point2[0], 2) +
    std::pow(point1[1] - point2[1], 2) + std::pow(point1[2] - point2[2], 2));
}

void remove_vertex(std::vector<Eigen::RowVector3d>& vertices, int rowIndex)
{
  vertices[rowIndex] << -1, -1, -1;
}

void remove_face(std::vector<Eigen::RowVector4i>& faces, int rowIndex)
{
  faces[rowIndex] << -1, -1, -1, -1;
}

void remove_half_edge(std::vector<HalfEdge>& halfEdges, int index)
{
  halfEdges[index].vertex = -1;
  halfEdges[index].face = -1;
  halfEdges[index].next = -1;
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

bool diag_collapse(std::vector<Eigen::RowVector3d>& vertices, 
  std::vector<Eigen::RowVector4i> faces, std::vector<HalfEdge>& halfEdges, 
  std::map<Edge, std::pair<int, int>, compareTwoEdges>& halfEdgesMap)
{
  Eigen::RowVector4i face = faces[3];
  std::vector<Edge> edges = find_edges(face);
  double diag1 = distance_two_points(vertices[face[0]], vertices[face[2]]);
  double diag2 = distance_two_points(vertices[face[3]], vertices[face[1]]);

  if (diag1 < diag2)
  {

  }
  else
  {
    std::vector<int> halfEdgesToModify;

    Edge edge = find_edge(face, face[1]);

    int initialHe = halfEdges[halfEdgesMap[edge].first].vertex == face[1] ?
      halfEdgesMap[edge].first : halfEdgesMap[edge].second;
    int finalHe = halfEdges[halfEdges[halfEdges[initialHe].next].next].next;

    for (int tmp = opposite_half_edge(initialHe); tmp != finalHe; )
    {
      tmp = halfEdges[tmp].next;
      halfEdgesToModify.push_back(tmp);
      tmp = opposite_half_edge(tmp);
    }

    // Set new starting vertices for the half-edge to modify
    for (int he : halfEdgesToModify)
    {
      if (halfEdges[he].vertex != face[1]) return false;
      halfEdges[he].vertex = face[3];

      Eigen::RowVector4i f = faces[halfEdges[he].face];
      for (int i = 0; i < 4; i++)
      {
        if (f[i] == face[1])
        {
          faces[halfEdges[he].face][i] = face[3];
        }
      }
    }

    // Set new opposite half-edges for the half-edge to modify
    int firstEdge = std::find(edges.begin(), edges.end(), edge) - edges.begin();
    if (halfEdgesMap[edges[(firstEdge + 1) % 4]].first != halfEdges[initialHe].next)
    {
      halfEdgesMap[edges[(firstEdge + 1) % 4]].second = opposite_half_edge(initialHe);
    }
    else
    {
      halfEdgesMap[edges[(firstEdge + 1) % 4]].first = opposite_half_edge(initialHe);
    }

    if (halfEdgesMap[edges[(firstEdge + 2) % 4]].first !=
      halfEdges[halfEdges[initialHe].next].next)
    {
      halfEdgesMap[edges[(firstEdge + 2) % 4]].second = opposite_half_edge(finalHe);
    }
    else
    {
      halfEdgesMap[edges[(firstEdge + 2) % 4]].first = opposite_half_edge(finalHe);
    }

    // Remove the half-edges which belong to the face deleted
    remove_half_edge(halfEdges, finalHe);
    remove_half_edge(halfEdges, halfEdges[halfEdges[initialHe].next].next);
    remove_half_edge(halfEdges, halfEdges[initialHe].next);
    remove_half_edge(halfEdges, initialHe);

    halfEdgesMap.erase(edge);
    halfEdgesMap.erase(edges[(firstEdge + 3) % 4]);

    // Remove one quad face
    remove_face(faces, 3);

    // Move one vertex and remove the other
    Eigen::RowVector3d newPos = 0.5 * vertices[face[3]] + 0.5 * vertices[face[1]];
    vertices[face[3]] = newPos;
    remove_vertex(vertices, face[1]);
  }

  // Set the new vertices of the simplified mesh
  Eigen::MatrixXd newVertices;
  for (int i = 0; i < vertices.size(); i++)
  {
    Eigen::RowVector3d vertex = vertices[i];
    if (vertex[0] != -1)
    {
      newVertices.conservativeResize(newVertices.rows() + 1, 3);
      newVertices.row(newVertices.rows() - 1) = vertex;
    }
    else
    {
      V.conservativeResize(V.rows() - 1, Eigen::NoChange);
      for (int j = 0; j < faces.size(); j++)
      {
        Eigen::RowVector4i face = faces[j];
        for (int k = 0; k < 4; k++)
        {
          faces[j][k] = face[k] >= i ? face[k] - 1 : face[k];
        }
      }
    }
  }
  V = newVertices;

  // Set the new faces of the simplified mesh
  Eigen::MatrixXi newFaces;
  for (Eigen::RowVector4i face : faces)
  {
    if (face[0] != -1)
    {
      newFaces.conservativeResize(newFaces.rows() + 1, 4);
      newFaces.row(newFaces.rows() - 1) = face;
    }
    else
    {
      F.conservativeResize(F.rows() - 1, Eigen::NoChange);
    }
  }
  F = newFaces;

  return true;
}


bool start_simplification(int finalNumberOfFaces)
{
  std::vector<Eigen::RowVector3d> vertices;
  std::vector<Eigen::RowVector4i> faces;

  for (int i = 0; i < V.rows(); i++) // For each vertex
  {
    vertices.push_back(V.row(i));
  }
  
  for (int i = 0; i < F.rows(); i++) // For each quad face
  {
    Eigen::RowVector4i face = F.row(i);
    faces.push_back(face);

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

  return diag_collapse(vertices, faces, halfEdges, halfEdgesMap);
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

void draw_quad_mesh()
{
  Eigen::MatrixXi f;

  // 1) Triangulate the polygonal mesh
  Eigen::VectorXi I, C, J;
  igl::polygon_corners(F, I, C);
  //std::cout << "\n\n" << I << "\n\n";
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
  //std::cout << "\n\n" << fN << "\n\n";

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

  // Load a mesh in OFF format
  //igl::readOFF(MESHES_DIR + "quad_cube.off", V, F);
  igl::readOBJ(MESHES_DIR + "quad_cubespikes.obj", V, F);

  if (start_simplification(9))
  {
    draw_quad_mesh();
  }
  else
  {
    std::cout << "\n\n" << "ERROR occured during quad mesh simplification" << "\n\n";
  }
}
