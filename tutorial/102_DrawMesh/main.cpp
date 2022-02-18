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

bool operator!=(const HalfEdge& he1, const HalfEdge& he2)
{
  return he1.vertex != he2.vertex || he1.face != he2.face;
}

// Used to navigate through half-edges
std::vector<HalfEdge> halfEdges;

typedef Eigen::RowVector2i Edge;
/*bool operator==(const Edge e1, const Edge e2)
{
  return (e1[0] == e2[0] && e1[1] == e2[1]) || 
    (e1[1] == e2[0] && e1[0] == e2[1]);
}*/

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
  faces[rowIndex] << -1, -1, -1;
  
  /*Eigen::MatrixXi newMatrix(faces.rows() - 1, faces.cols());
  newMatrix << faces.block(0, 0, row_index, 4),
    faces.block(row_index + 1, 0, faces.rows() - row_index - 1, 4);
  return newMatrix;*/
}

void remove_vertex(std::vector<Eigen::RowVector3d>& vertices, int rowIndex)
{ 
  vertices[rowIndex] << -1, -1, -1, -1;
  
  /*Eigen::MatrixXd newMatrix(vertices.rows() - 1, vertices.cols());
  newMatrix << vertices.block(0, 0, row_index, 4),
    vertices.block(row_index + 1, 0, vertices.rows() - row_index - 1, 4);
  return newMatrix;*/
}

// It removes one quad face
/*bool diag_collapse(std::vector<Eigen::RowVector3d>& vertices, std::vector<Eigen::RowVector4i> faces,
  std::vector<HalfEdge>& halfEdges, std::map<Edge, std::pair<HalfEdge, HalfEdge>, comp>& halfEdgesMap)
{
  Eigen::RowVector4i face = faces[13];
  double diag1 = distance_two_points(V.row(face[0]), V.row(face[2]));
  double diag2 = distance_two_points(V.row(face[3]), V.row(face[1]));

  if (diag1 < diag2)
  {

  }
  else
  {
    std::vector<HalfEdge> halfEdgeToModify;
    
    Edge edge = Edge{ std::min(V.row(face[1]), V.row(face[(1 + 1) % 4])),
      std::max(V.row(face[1]), V.row(face[(1 + 1) % 4])) };

    HalfEdge initialHe = halfEdgesMap[edge].first.vertex == face[1] ? 
      halfEdgesMap[edge].first : halfEdgesMap[edge].second;
    HalfEdge finalHe = halfEdges[halfEdges[halfEdges[initialHe.next].next].next];

    for  (HalfEdge tmp = opposite_half_edge(initialHe); tmp != finalHe; )
    {
      tmp = halfEdges[tmp.next];
      halfEdgeToModify.push_back(tmp);
      tmp = opposite_half_edge(tmp);
    }


    for (HalfEdge he : halfEdgeToModify)
    {
      if (he.vertex != face[1]) return false;
      he.vertex = face[3];
    }

    // Remove one quad face
    remove_face(faces, 13);

    // Move one vertex and remove the other
    Eigen::RowVector3d newPos = 0.5 * V.row(face[3]) + 0.5 * V.row(face[1]);
    V.row(face[3]) = newPos;
    V = remove_vertex(V, face[1]);
  }

  return true;
}*/


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
    // Find edges
    /*edges.push_back(Edge(std::min(face[0], face[1]), std::max(face[0], face[1])));
    edges.push_back(Edge(std::min(face[1], face[2]), std::max(face[1], face[2])));
    edges.push_back(Edge(std::min(face[2], face[3]), std::max(face[2], face[3])));
    edges.push_back(Edge(std::min(face[3], face[0]), std::max(face[3], face[0])));*/
    
    Eigen::RowVector4i face = F.row(i);
    faces.push_back(face);

    // Compute four half-edges
    for (int j = 0; j < 4; j++)
    {
      HalfEdge halfEdge = HalfEdge{
        face[j], // vertex
        i, // face
        i * 4 + (int(halfEdges.size() + 1) % 4) // next
      };

      halfEdges.push_back(halfEdge);

      Edge edge = Edge{ std::min(face[j], face[(j + 1) % 4]), 
        std::max(face[j], face[(j + 1) % 4]) };
      if (halfEdgesMap.find(edge) == halfEdgesMap.end())
      {
        halfEdgesMap.insert(std::pair<Edge, std::pair<HalfEdge, HalfEdge>>(
          edge, std::make_pair(halfEdge, halfEdge)));
      }
      else
      {
        std::pair<HalfEdge, HalfEdge>* hes = &halfEdgesMap[edge];
        (*hes).second = halfEdge;
      }
    }
  }


  
  //return diag_collapse(vertices, faces, halfEdges, halfEdgesMap);

  /*edges.sort([](Edge e1, Edge e2) { return e1[0] < e2[0]; });
  edges.sort([](Edge e1, Edge e2) { return e1[1] < e2[1]; });

  edges.unique([](Edge e1, Edge e2) { return e1[0] == e2[0] && e1[1] == e2[1]; });*/


  for (auto& kv : halfEdgesMap)
  {
    std::cout << "\n" << kv.first << " --> " << 
      kv.second.first.vertex << " -- " << kv.second.first.face << " -- " << kv.second.first.next << " || " <<
      kv.second.second.vertex << " -- " << kv.second.second.face << " -- " << kv.second.second.next << "\n";
  }

  return true; // TODO togli
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

void draw_mesh()
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
    draw_mesh();
  }
  else
  {
    std::cout << "\n\n" << "ERROR occured during quad mesh simplification" << "\n\n";
  }
}
