#include "cut_tetrahedron.hpp"

#include <Eigen/Geometry>
#include <glfw/glfw3.h>
#include <igl/barycenter.h>
#include <igl/boundary_facets.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/trackball.h>
#include <iomanip>
#include <sstream>

int main(int argc, char** argv)
{
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    Eigen::MatrixXd V(4u, 3u);
    Eigen::MatrixXi T(1u, 4u);
    Eigen::MatrixXi F(4u, 3u);

    auto const reset_demo = [&]() {
        V = Eigen::MatrixXd(4u, 3u);
        T = Eigen::MatrixXi(1u, 4u);
        F = Eigen::MatrixXi(4u, 3u);

        // clang-format off
	    V.row(0u) = Eigen::RowVector3d{0., 0., -1.};
        V.row(1u) = Eigen::RowVector3d{1., 0.,  1.};
        V.row(2u) = Eigen::RowVector3d{2., 0., -1.};
        V.row(3u) = Eigen::RowVector3d{1., 1.5,  0.};

        T.row(0u) = Eigen::RowVector4i{0u, 1u, 2u, 3u};

        igl::boundary_facets(T, F);
        F = F.rowwise().reverse().eval();
        // clang-format on

        viewer.data().clear();
        viewer.data().show_labels = true;
        viewer.data().point_size  = 10.f;
        viewer.data().set_mesh(V, F);
        viewer.data().set_face_based(true);
    };

    Eigen::RowVector3d const red{1., 0., 0.};
    Eigen::RowVector3d const green{0., 1., 0.};
    Eigen::RowVector3d const blue{0., 0., 1.};

    auto const get_coordinate_string = [](Eigen::RowVector3d const& point) {
        std::ostringstream str{};
        str << std::fixed << std::setprecision(1);
        str << "(" << point.x() << ", " << point.y() << ", " << point.z() << ")";
        return str.str();
    };

    auto const display_tet_face_intersections = [&viewer, &red](
                                                    Eigen::MatrixXd const& V,
                                                    Eigen::MatrixXi const& T,
                                                    Eigen::Vector3d const& p,
                                                    Eigen::Vector3d const& q) {
        Eigen::MatrixXi TF(4u, 3u);
        TF.row(0u) = Eigen::RowVector3i{0u, 1u, 3u};
        TF.row(1u) = Eigen::RowVector3i{1u, 2u, 3u};
        TF.row(2u) = Eigen::RowVector3i{2u, 0u, 3u};
        TF.row(3u) = Eigen::RowVector3i{0u, 2u, 1u};

        auto const [f1_intersected, f1_intersection] = geometry::triangle_line_intersection_two_way(
            V.row(TF(0, 0)).transpose(),
            V.row(TF(0, 1)).transpose(),
            V.row(TF(0, 2)).transpose(),
            {p, q});

        auto const [f2_intersected, f2_intersection] = geometry::triangle_line_intersection_two_way(
            V.row(TF(1, 0)).transpose(),
            V.row(TF(1, 1)).transpose(),
            V.row(TF(1, 2)).transpose(),
            {p, q});

        auto const [f3_intersected, f3_intersection] = geometry::triangle_line_intersection_two_way(
            V.row(TF(2, 0)).transpose(),
            V.row(TF(2, 1)).transpose(),
            V.row(TF(2, 2)).transpose(),
            {p, q});

        auto const [f4_intersected, f4_intersection] = geometry::triangle_line_intersection_two_way(
            V.row(TF(3, 0)).transpose(),
            V.row(TF(3, 1)).transpose(),
            V.row(TF(3, 2)).transpose(),
            {p, q});

        if (f1_intersected)
        {
            viewer.data().add_points(f1_intersection.transpose(), red);
        }
        if (f2_intersected)
        {
            viewer.data().add_points(f2_intersection.transpose(), red);
        }
        if (f3_intersected)
        {
            viewer.data().add_points(f3_intersection.transpose(), red);
        }
        if (f4_intersected)
        {
            viewer.data().add_points(f4_intersection.transpose(), red);
        }
    };

    auto const diplay_tet_edge_intersections = [&viewer, &red](
                                                   Eigen::MatrixXd const& V,
                                                   Eigen::MatrixXi const& T,
                                                   Eigen::Vector3d const& a,
                                                   Eigen::Vector3d const& b,
                                                   Eigen::Vector3d const& c) {
        auto const& v1 = V.row(T(0, 0));
        auto const& v2 = V.row(T(0, 1));
        auto const& v3 = V.row(T(0, 2));
        auto const& v4 = V.row(T(0, 3));

        auto const [e1_intersected, e1_intersection] =
            geometry::triangle_line_intersection_two_way(a, b, c, {v1, v2});
        auto const [e2_intersected, e2_intersection] =
            geometry::triangle_line_intersection_two_way(a, b, c, {v2, v3});
        auto const [e3_intersected, e3_intersection] =
            geometry::triangle_line_intersection_two_way(a, b, c, {v3, v1});
        auto const [e4_intersected, e4_intersection] =
            geometry::triangle_line_intersection_two_way(a, b, c, {v1, v4});
        auto const [e5_intersected, e5_intersection] =
            geometry::triangle_line_intersection_two_way(a, b, c, {v2, v4});
        auto const [e6_intersected, e6_intersection] =
            geometry::triangle_line_intersection_two_way(a, b, c, {v3, v4});

        if (e1_intersected)
        {
            viewer.data().add_points(e1_intersection.transpose(), red);
        }
        if (e2_intersected)
        {
            viewer.data().add_points(e2_intersection.transpose(), red);
        }
        if (e3_intersected)
        {
            viewer.data().add_points(e3_intersection.transpose(), red);
        }
        if (e4_intersected)
        {
            viewer.data().add_points(e4_intersection.transpose(), red);
        }
        if (e5_intersected)
        {
            viewer.data().add_points(e5_intersection.transpose(), red);
        }
        if (e6_intersected)
        {
            viewer.data().add_points(e6_intersection.transpose(), red);
        }
    };

    reset_demo();

    menu.callback_draw_viewer_window = [&]() {
        viewer.data().clear_edges();
        viewer.data().clear_points();
        viewer.data().clear_labels();

        ImGui::SetNextWindowSize(ImVec2(150.0f, 480.0f), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Tetrahedral Subdivision");

        float const w = ImGui::GetContentRegionAvailWidth();
        float const p = ImGui::GetStyle().FramePadding.x;

        Eigen::Vector3d const origin{0., 0., 0.};
        static Eigen::Vector3d translation{0., 0., 0.};
        Eigen::Vector3d const z_unit_vector{0., 0., -1.};
        static Eigen::RowVector3d start_line_segment_p1, start_line_segment_p2;
        static Eigen::RowVector3d end_line_segment_p1, end_line_segment_p2;
        static Eigen::Vector3d start_roll_pitch_yaw{0., 0., 0.};
        static Eigen::Vector3d end_roll_pitch_yaw{0., 0., 0.};

        Eigen::MatrixXd CuttingTriangleV(3u, 3u);
        CuttingTriangleV.row(0u)      = Eigen::RowVector3d{0., 0., 0.5};
        CuttingTriangleV.row(1u)      = Eigen::RowVector3d{0.5, 0., -0.5};
        CuttingTriangleV.row(2u)      = Eigen::RowVector3d{-0.5, 0., -0.5};
        Eigen::RowVector3d const mean = CuttingTriangleV.colwise().mean().eval();
        CuttingTriangleV.rowwise() -= mean;

        Eigen::MatrixXi CuttingTriangleF(1u, 3u);
        CuttingTriangleF.row(0u) = Eigen::RowVector3i{0, 1, 2};

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::RowVector3d t{0., 0., 0.};
        Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
        static float roll = 0.f, pitch = 0.f, yaw = 0.f;
        static float tx = 0.f, ty = 0.f, tz = 0.f;
        static float sx = 1.f, sy = 1.f, sz = 1.f;

        if (ImGui::CollapsingHeader("Cutting Swept Surface", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("Roll Pitch Yaw of Cutting Triangle");
            ImGui::SliderFloat("Roll", &roll, 0.f, 2.f * 3.14159f, "%.1f");
            ImGui::SliderFloat("Pitch", &pitch, 0.f, 2.f * 3.14159f, "%.1f");
            ImGui::SliderFloat("Yaw", &yaw, 0.f, 2.f * 3.14159f, "%.1f");

            ImGui::Text("Translation of Cutting Triangle");
            ImGui::SliderFloat("tx", &tx, -5.f, 5.f, "%.1f");
            ImGui::SliderFloat("ty", &ty, -5.f, 5.f, "%.1f");
            ImGui::SliderFloat("tz", &tz, -5.f, 5.f, "%.1f");

            ImGui::Text("Scaling of Cutting Triangle");
            ImGui::SliderFloat("sx", &sx, -5.f, 5.f, "%.1f");
            ImGui::SliderFloat("sy", &sy, -5.f, 5.f, "%.1f");
            ImGui::SliderFloat("sz", &sz, -5.f, 5.f, "%.1f");
        }

        Eigen::AngleAxisd const roll_angle(static_cast<double>(roll), Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd const pitch_angle(static_cast<double>(pitch), Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd const yaw_angle(static_cast<double>(yaw), Eigen::Vector3d::UnitZ());
        t(0u)   = tx;
        t(1u)   = ty;
        t(2u)   = tz;
        S(0, 0) = sx;
        S(1, 1) = sy;
        S(2, 2) = sz;

        Eigen::Quaterniond const q = roll_angle * yaw_angle * pitch_angle;
        R                          = q.matrix();

        static bool is_cut{false};

        // coordinate frame visualization
        viewer.data().add_edges(
            0.5 * Eigen::RowVector3d{0., 0., 0.},
            0.5 * Eigen::RowVector3d{1., 0., 0.},
            red);
        viewer.data().add_edges(
            0.5 * Eigen::RowVector3d{0., 0., 0.},
            0.5 * Eigen::RowVector3d{0., 1., 0.},
            green);
        viewer.data().add_edges(
            0.5 * Eigen::RowVector3d{0., 0., 0.},
            0.5 * Eigen::RowVector3d{0., 0., 1.},
            blue);

        CuttingTriangleV = (R * S * CuttingTriangleV.transpose()).transpose();
        CuttingTriangleV.rowwise() += t;

        start_line_segment_p1 = CuttingTriangleV.row(0u);
        start_line_segment_p2 = CuttingTriangleV.row(2u);
        end_line_segment_p1   = CuttingTriangleV.row(0u);
        end_line_segment_p2   = CuttingTriangleV.row(1u);
        viewer.data().add_edges(start_line_segment_p1, start_line_segment_p2, red);
        viewer.data().add_edges(end_line_segment_p1, end_line_segment_p2, red);

        // draw tetrahedron edge intersections with cutting swept surface
        Eigen::Vector3d const swept_surface_triangle_a{start_line_segment_p1.transpose()};
        Eigen::Vector3d const swept_surface_triangle_b{start_line_segment_p2.transpose()};
        Eigen::Vector3d const swept_surface_triangle_c{end_line_segment_p2.transpose()};

        viewer.data().add_points(start_line_segment_p1, red);
        viewer.data().add_points(start_line_segment_p2, red);
        viewer.data().add_points(end_line_segment_p1, red);
        viewer.data().add_points(end_line_segment_p2, red);

        if (!is_cut)
        {
            display_tet_face_intersections(V, T, start_line_segment_p1, start_line_segment_p2);
            display_tet_face_intersections(V, T, end_line_segment_p1, end_line_segment_p2);

            diplay_tet_edge_intersections(
                V,
                T,
                swept_surface_triangle_a,
                swept_surface_triangle_b,
                swept_surface_triangle_c);

            viewer.data().add_label(
                start_line_segment_p1,
                get_coordinate_string(start_line_segment_p1));
            viewer.data().add_label(
                start_line_segment_p2,
                get_coordinate_string(start_line_segment_p2));
            viewer.data().add_label(
                end_line_segment_p1,
                get_coordinate_string(end_line_segment_p1));
            viewer.data().add_label(
                end_line_segment_p2,
                get_coordinate_string(end_line_segment_p2));
        }

        if (ImGui::Button("Cut", ImVec2((w - p) / 2.f, 0.f)))
        {
            geometry::cut_tetrahedron(
                V,
                T,
                0,
                {start_line_segment_p1, start_line_segment_p2},
                {end_line_segment_p1, end_line_segment_p2});

            std::cout << "Vertices:\n" << V << "\n";

            std::cout << "Subdivided tetrahedral mesh:\n" << T << "\n";

            igl::boundary_facets(T, F);
            F = F.rowwise().reverse().eval();

            std::cout << "Subdivided tetrahedral boundary facets:\n" << F << "\n";
            is_cut = true;

            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            viewer.data().set_face_based(true);
            viewer.data().show_lines = true;
        }
        if (ImGui::Button("Reset", ImVec2((w - p) / 2.f, 0.f)))
        {
            reset_demo();
            is_cut = false;
        }

        ImGui::End();
    };

    viewer.launch();
    return 0;
}