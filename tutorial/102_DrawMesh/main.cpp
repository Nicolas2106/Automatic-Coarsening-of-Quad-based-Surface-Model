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

// Used to navigate through half-edges
std::vector<HalfEdge> halfEdges;

typedef Eigen::RowVector2i Edge;
bool operator==(const Edge e1, const Edge e2)
{
  return (e1[0] == e2[0] && e1[1] == e2[1]) || 
    (e1[1] == e2[0] && e1[0] == e2[1]);
}

struct comp {
  bool operator()(const Edge e1, const Edge e2) const {
    return std::stoi(std::to_string(e1[0]) + std::to_string(e1[1])) < 
      std::stoi(std::to_string(e2[0]) + std::to_string(e2[1]));
  }
};

// Used for opposite half-edges
std::map<Edge, std::pair<HalfEdge, HalfEdge>, comp> halfEdgesMap;

HalfEdge opposite_half_edge(HalfEdge& halfEdge)
{
  std::pair<HalfEdge, HalfEdge> pair = halfEdgesMap[Edge{
    std::min(halfEdge.vertex, halfEdges[halfEdge.next].vertex),
    std::max(halfEdge.vertex, halfEdges[halfEdge.next].vertex) }];
  return pair.first.vertex == halfEdge.vertex ? pair.second : pair.first;
}

double distance_two_points(Eigen::RowVector3d point1, Eigen::RowVector3d point2)
{
  return std::sqrt(std::pow(point1[0] - point2[0], 2) +
    std::pow(point1[1] - point2[1], 2) + std::pow(point1[2] - point2[2], 2));
}

void remove_face(std::vector<Eigen::RowVector4i>& faces, int rowIndex)
{
  faces[rowIndex] << -1, -1, -1, -1;
  
  /*Eigen::MatrixXi newMatrix(faces.rows() - 1, faces.cols());
  newMatrix << faces.block(0, 0, row_index, 4),
    faces.block(row_index + 1, 0, faces.rows() - row_index - 1, 4);
  return newMatrix;*/
}

void remove_vertex(std::vector<Eigen::RowVector3d>& vertices, int rowIndex)
{ 
  vertices[rowIndex] << -1, -1, -1;
  
  /*Eigen::MatrixXd newMatrix(vertices.rows() - 1, vertices.cols());
  newMatrix << vertices.block(0, 0, row_index, 4),
    vertices.block(row_index + 1, 0, vertices.rows() - row_index - 1, 4);
  return newMatrix;*/
}

std::vector<Edge> find_edges(Eigen::RowVector4i face)
{
  std::vector<Edge> edges;
  for (int i = 0; i < 4; i++)
  {
    edges.push_back(Edge{ std::min(face[i], face[(i + 1) % 4]),
      std::max(face[i], face[(i + 1) % 4]) });
  }
  assert(edges.size() == 4);
  return edges;
}

Edge find_edge(Eigen::RowVector4i face, int startingPointIndex)
{
  std::vector<Edge> edges = find_edges(face);
  for (Edge edge : edges)
  {
    if (edge[0] == startingPointIndex) return edge;
  }
  return edges[0];
}

bool diag_collapse(std::vector<Eigen::RowVector3d>& vertices, std::vector<Eigen::RowVector4i> faces,
  std::vector<HalfEdge>& halfEdges, std::map<Edge, std::pair<HalfEdge, HalfEdge>, comp>& halfEdgesMap)
{
  Eigen::RowVector4i face = faces[13];
  std::vector<Edge> edges = find_edges(face);
  double diag1 = distance_two_points(vertices[face[0]], vertices[face[2]]);
  double diag2 = distance_two_points(vertices[face[3]], vertices[face[1]]);

  if (diag1 < diag2)
  {

  }
  else
  {
    std::vector<HalfEdge> halfEdgesToModify;
    
    Edge edge = find_edge(face, face[1]);

    HalfEdge initialHe = halfEdgesMap[edge].first.vertex == face[1] ? 
      halfEdgesMap[edge].first : halfEdgesMap[edge].second;
    HalfEdge finalHe = halfEdges[halfEdges[halfEdges[initialHe.next].next].next];

    for (HalfEdge tmp = opposite_half_edge(initialHe); tmp != finalHe; )
    {
      tmp = halfEdges[tmp.next];
      halfEdgesToModify.push_back(tmp);
      tmp = opposite_half_edge(tmp);
    }

    // Set new starting vertices for the half-edge to modify
    for (HalfEdge he : halfEdgesToModify)
    {
      if (he.vertex != face[1]) return false;
      he.vertex = face[3];
    }

    // Set new opposite half-edges for the half-edge to modify
    int edgeIndex = std::find(edges.begin(), edges.end(), edge) - edges.begin();
    if (halfEdgesMap[edges[(edgeIndex + 1) % 4]].first != halfEdges[initialHe.next])
    {
      halfEdgesMap[edges[(edgeIndex + 1) % 4]].second = opposite_half_edge(initialHe);
    }
    else
    {
      halfEdgesMap[edges[(edgeIndex + 1) % 4]].first = opposite_half_edge(initialHe);
    }

    if (halfEdgesMap[edges[(edgeIndex + 2) % 4]].first != 
      halfEdges[halfEdges[initialHe.next].next])
    {
      halfEdgesMap[edges[(edgeIndex + 2) % 4]].second = opposite_half_edge(finalHe);
    }
    else
    {
      halfEdgesMap[edges[(edgeIndex + 2) % 4]].first = opposite_half_edge(finalHe);
    }

    // Remove the half-edges belong to the face deleted
    /* It's not essential to do this operation (it creates side effect!)
    halfEdges.erase(std::find(halfEdges.begin(), halfEdges.end(), finalHe));
    halfEdges.erase(std::find(halfEdges.begin(), halfEdges.end(),
      halfEdges[halfEdges[initialHe.next].next]));
    halfEdges.erase(std::find(halfEdges.begin(), halfEdges.end(),
      halfEdges[initialHe.next]));
    halfEdges.erase(std::find(halfEdges.begin(), halfEdges.end(), initialHe));*/
    halfEdgesMap.erase(edge);
    halfEdgesMap.erase(edges[(edgeIndex + 3) % 4]);

    // Remove one quad face
    remove_face(faces, 13);

    // Move one vertex and remove the other
    Eigen::RowVector3d newPos = 0.5 * vertices[face[3]] + 0.5 * vertices[face[1]];
    vertices[face[3]] = newPos;
    remove_vertex(vertices, face[1]);
  }

  return true;
}


bool start_simplification(int finalNumberOfFaces)
{
  std::vector<Eigen::RowVector3d> vertices;
  std::vector<Eigen::RowVector4i> faces;
  //std::list<Edge> edges;

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
        i * 4 + (int(halfEdges.size() + 1) % 4) // next
      };

      halfEdges.push_back(halfEdge);

      if (halfEdgesMap.find(edges[j]) == halfEdgesMap.end()) // The half-edge doesn't exist yet
      {
        halfEdgesMap.insert(std::pair<Edge, std::pair<HalfEdge, HalfEdge>>(
          edges[j], std::make_pair(halfEdge, halfEdge)));
      }
      else // The half-edge exists
      {
        std::pair<HalfEdge, HalfEdge>* hes = &halfEdgesMap[edges[j]];
        (*hes).second = halfEdge;
      }
    }
  }

  /*for (auto& kv : halfEdgesMap)
  {
    std::cout << "\n" << kv.first << " --> " << 
      kv.second.first.vertex << " -- " << kv.second.first.face << " -- " << kv.second.first.next << " || " <<
      kv.second.second.vertex << " -- " << kv.second.second.face << " -- " << kv.second.second.next << "\n";
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
  //viewer.data().show_overlay = true;
  //viewer.data().show_overlay_depth = true;
  /*for (unsigned i = 0; i < E.rows(); ++i)
  {
    viewer.data().add_edges
    (
      V.row(E(i, 0)),
      V.row(E(i, 1)),
      Eigen::RowVector3d(1, 0, 0)
    );
  }*/


  //viewer.data().lines.resize(0, 0); // clear lines

  /*Eigen::MatrixXd new_lines(E.rows(), 9);
  for (int i=0; i<E.rows(); i++)
  {
    Eigen::RowVector3d vert1 = V.row(E.row(i)[0]);
    Eigen::RowVector3d vert2 = V.row(E.row(i)[1]);

    new_lines.row(i) <<
      vert1[0], vert1[1], vert1[2], vert2[0], vert2[1], vert2[2], 0.0, 0.0, 0.0;
  }
  viewer.data().lines = new_lines;*/

  //std::cout << "\n\n" << viewer.data().dirty << "\n\n";
  //viewer.data().line_color << 1.0, 0.0, 0.0, 1.0;
  //std::cout << "\n\n" << viewer.data().line_color << "\n\n";

  viewer.launch();
}

int main(int argc, char *argv[])
{
  const std::string MESHES_DIR = "F:\\Users\\Nicolas\\Desktop\\TESI\\Quadrilateral extension to libigl\\libigl\\tutorial\\102_DrawMesh\\";

  // Load a mesh in OFF format
  igl::readOFF(MESHES_DIR + "quad_surface.off", V, F);
  //igl::readOBJ(MESHES_DIR + "quad_cubespikes.obj", V, F);

  if (start_simplification(9))
  {
    //draw_quad_mesh();
  }
  else
  {
    std::cout << "\n\n" << "ERROR occured during quad mesh simplification" << "\n\n";
  }
}
