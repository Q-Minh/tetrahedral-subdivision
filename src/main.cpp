#include "cut_tetrahedron.hpp"

#include <Eigen/Geometry>
#include <igl/barycenter.h>
#include <igl/boundary_facets.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
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
        viewer.core().align_camera_center(V);
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
        double t = 1.;

        if (ImGui::CollapsingHeader("Cutting Swept Surface", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Text("Translation", "");
            static float tx = 0.f, ty = 0.f, tz = 0.f;
            ImGui::SliderFloat("tx##SweptSurface", &tx, -2.f, 2.f, "%.1f");
            ImGui::SliderFloat("ty##SweptSurface", &ty, -2.f, 2.f, "%.1f");
            ImGui::SliderFloat("tz##SweptSurface", &tz, -2.f, 2.f, "%.1f");

            translation(0) = tx;
            translation(1) = ty;
            translation(2) = tz;

            ImGui::Text("Line segment length", "");
            static float length = 1.f;
            ImGui::SliderFloat("t##SweptSurface", &length, 0.f, 10.f, "%.1f");

            t = length;

            ImGui::Text("Start Line 3d rotation", "");
            static float start_roll = 0.f, start_pitch = 0.f, start_yaw = 0.f;
            ImGui::SliderFloat(
                "roll##SweptSurfaceStartLine",
                &start_roll,
                0.f,
                2.f * 3.14159f,
                "%.1f");
            ImGui::SliderFloat(
                "pitch##SweptSurfaceStartLine",
                &start_pitch,
                0.f,
                2.f * 3.14159f,
                "%.1f");
            ImGui::SliderFloat(
                "yaw##SweptSurfaceStartLine",
                &start_yaw,
                0.f,
                2.f * 3.14159f,
                "%.1f");

            start_roll_pitch_yaw(0) = start_roll;
            start_roll_pitch_yaw(1) = start_pitch;
            start_roll_pitch_yaw(2) = start_yaw;

            ImGui::Text("End Line 3d rotation", "");
            static float end_roll = 0.f, end_pitch = 6.3f, end_yaw = 0.f;
            ImGui::SliderFloat("roll##SweptSurfaceEndLine", &end_roll, 0.f, 2.f * 3.14159f, "%.1f");
            ImGui::SliderFloat(
                "pitch##SweptSurfaceEndLine",
                &end_pitch,
                0.f,
                2.f * 3.14159f,
                "%.1f");
            ImGui::SliderFloat("yaw##SweptSurfaceEndLine", &end_yaw, 0.f, 2.f * 3.14159f, "%.1f");

            end_roll_pitch_yaw(0) = end_roll;
            end_roll_pitch_yaw(1) = end_pitch;
            end_roll_pitch_yaw(2) = end_yaw;
        }

        static bool is_cut{false};

        // draw the cutting "scalpel"
        Eigen::AngleAxisd const start_roll_angle(
            static_cast<double>(start_roll_pitch_yaw(0)),
            Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd const start_yaw_angle(
            static_cast<double>(start_roll_pitch_yaw(1)),
            Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd const start_pitch_angle(
            static_cast<double>(start_roll_pitch_yaw(2)),
            Eigen::Vector3d::UnitZ());

        Eigen::AngleAxisd const end_roll_angle(
            static_cast<double>(end_roll_pitch_yaw(0)),
            Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd const end_yaw_angle(
            static_cast<double>(end_roll_pitch_yaw(1)),
            Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd const end_pitch_angle(
            static_cast<double>(end_roll_pitch_yaw(2)),
            Eigen::Vector3d::UnitZ());

        Eigen::Quaterniond const start_q = start_roll_angle * start_yaw_angle * start_pitch_angle;
        Eigen::Matrix3d const start_swept_surface_rotation = start_q.matrix();

        start_line_segment_p1 = start_swept_surface_rotation * (origin + translation);
        start_line_segment_p2 =
            start_swept_surface_rotation * (origin + translation + t * z_unit_vector);

        Eigen::Quaterniond const end_q = end_roll_angle * end_yaw_angle * end_pitch_angle;
        Eigen::Matrix3d const end_swept_surface_rotation = end_q.matrix();

        end_line_segment_p1 = start_line_segment_p1;
        end_line_segment_p2 =
            (end_swept_surface_rotation * (start_line_segment_p2.transpose() - translation) +
                translation)
                .transpose();

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

        if (!is_cut)
        {
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
            viewer.data().add_label(end_line_segment_p1, get_coordinate_string(end_line_segment_p1));
            viewer.data().add_label(end_line_segment_p2, get_coordinate_string(end_line_segment_p2));
        }

        if (ImGui::Button("Cut", ImVec2((w - p) / 2.f, 0.f)))
        {
            geometry::cut_tetrahedron(
                V,
                T,
                0,
                {start_line_segment_p1, start_line_segment_p2},
                {end_line_segment_p1, end_line_segment_p2});

            std::cout << "Subdivided tetrahedral mesh:\n" << T << "\n";

            igl::boundary_facets(T, F);
            F = F.rowwise().reverse().eval();

            std::cout << "Subdivided tetrahedral boundary facets:\n" << F << "\n";

            viewer.data().clear();
            viewer.data().set_mesh(V, F);
            viewer.core().align_camera_center(V);
            viewer.data().show_lines = true;
        }
        if (ImGui::Button("Reset", ImVec2((w - p) / 2.f, 0.f)))
        {
            reset_demo();
        }

        ImGui::End();
    };

    viewer.launch();
    return 0;
}