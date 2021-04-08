#ifndef TET_CUT_CUT_TETRAHEDRON_HPP
#define TET_CUT_CUT_TETRAHEDRON_HPP

#include "intersection_tests.hpp"

#include <Eigen/Core>
#include <array>
#include <bitset>
#include <cassert>
#include <unordered_map>
#include <utility>

namespace geometry {

class tetrahedron_mesh_cutter_t
{
  public:
    bool subdivide_mesh(
        std::byte const& edge_intersection_mask,
        Eigen::MatrixXd& TV,
        Eigen::MatrixXi& TT,
        int tetrahedron,
        std::array<Eigen::Vector3d, 6u> const& edge_intersection_points,
        std::array<Eigen::Vector3d, 4u> const& face_intersection_points)
    {
        int constexpr v1{0};
        int constexpr v2{1};
        int constexpr v3{2};
        int constexpr v4{3};

        int constexpr e1{0};
        int constexpr e2{1};
        int constexpr e3{2};
        int constexpr e4{3};
        int constexpr e5{4};
        int constexpr e6{5};

        int constexpr f1{0};
        int constexpr f2{1};
        int constexpr f3{2};
        int constexpr f4{3};

        std::byte constexpr case_1_125{0b00010011};
        std::byte constexpr case_1_134{0b00001101};
        std::byte constexpr case_1_236{0b00100110};
        std::byte constexpr case_1_456{0b00111000};

        if (edge_intersection_mask == case_1_125)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e2];
            auto const& pe3 = edge_intersection_points[e5];
            subdivide_mesh_for_common_case_1(
                TV,
                TT,
                tetrahedron,
                {v1, v3, v4, v2},
                {pe1, pe2, pe3});
            return true;
        }
        if (edge_intersection_mask == case_1_134)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e4];
            auto const& pe3 = edge_intersection_points[e3];
            subdivide_mesh_for_common_case_1(
                TV,
                TT,
                tetrahedron,
                {v2, v4, v3, v1},
                {pe1, pe2, pe3});
            return true;
        }
        if (edge_intersection_mask == case_1_236)
        {
            auto const& pe1 = edge_intersection_points[e2];
            auto const& pe2 = edge_intersection_points[e3];
            auto const& pe3 = edge_intersection_points[e6];
            subdivide_mesh_for_common_case_1(
                TV,
                TT,
                tetrahedron,
                {v2, v1, v4, v3},
                {pe1, pe2, pe3});
            return true;
        }
        if (edge_intersection_mask == case_1_456)
        {
            auto const& pe1 = edge_intersection_points[e4];
            auto const& pe2 = edge_intersection_points[e5];
            auto const& pe3 = edge_intersection_points[e6];
            subdivide_mesh_for_common_case_1(
                TV,
                TT,
                tetrahedron,
                {v1, v2, v3, v4},
                {pe1, pe2, pe3});
            return true;
        }

        std::byte constexpr case_2_1246{0b00101011};
        std::byte constexpr case_2_1356{0b00110101};
        std::byte constexpr case_2_2345{0b00011110};

        if (edge_intersection_mask == case_2_1246)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e2];
            auto const& pe3 = edge_intersection_points[e4];
            auto const& pe4 = edge_intersection_points[e6];
            subdivide_mesh_for_common_case_2(
                TV,
                TT,
                tetrahedron,
                {v2, v4, v3, v1},
                {pe1, pe2, pe3, pe4});
            return true;
        }
        if (edge_intersection_mask == case_2_1356)
        {
            auto const& pe1 = edge_intersection_points[e5];
            auto const& pe2 = edge_intersection_points[e1];
            auto const& pe3 = edge_intersection_points[e6];
            auto const& pe4 = edge_intersection_points[e3];
            subdivide_mesh_for_common_case_2(
                TV,
                TT,
                tetrahedron,
                {v4, v1, v3, v2},
                {pe1, pe2, pe3, pe4});
            return true;
        }
        if (edge_intersection_mask == case_2_2345)
        {
            auto const& pe1 = edge_intersection_points[e4];
            auto const& pe2 = edge_intersection_points[e3];
            auto const& pe3 = edge_intersection_points[e5];
            auto const& pe4 = edge_intersection_points[e2];
            subdivide_mesh_for_common_case_2(
                TV,
                TT,
                tetrahedron,
                {v1, v2, v3, v4},
                {pe1, pe2, pe3, pe4});
            return true;
        }

        std::byte constexpr case_3_1{0b00000001};
        std::byte constexpr case_3_2{0b00000010};
        std::byte constexpr case_3_3{0b00000100};
        std::byte constexpr case_3_4{0b00001000};
        std::byte constexpr case_3_5{0b00010000};
        std::byte constexpr case_3_6{0b00100000};

        if (edge_intersection_mask == case_3_1)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pf1 = face_intersection_points[f4];
            auto const& pf2 = face_intersection_points[f1];
            subdivide_mesh_for_common_case_3(
                TV,
                TT,
                tetrahedron,
                {v1, v3, v4, v2},
                {pe1},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_3_2)
        {
            auto const& pe1 = edge_intersection_points[e2];
            auto const& pf1 = face_intersection_points[f2];
            auto const& pf2 = face_intersection_points[f4];
            subdivide_mesh_for_common_case_3(
                TV,
                TT,
                tetrahedron,
                {v3, v4, v1, v2},
                {pe1},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_3_3)
        {
            auto const& pe1 = edge_intersection_points[e3];
            auto const& pf1 = face_intersection_points[f3];
            auto const& pf2 = face_intersection_points[f4];
            subdivide_mesh_for_common_case_3(
                TV,
                TT,
                tetrahedron,
                {v1, v4, v2, v3},
                {pe1},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_3_4)
        {
            auto const& pe1 = edge_intersection_points[e4];
            auto const& pf1 = face_intersection_points[f1];
            auto const& pf2 = face_intersection_points[f3];
            subdivide_mesh_for_common_case_3(
                TV,
                TT,
                tetrahedron,
                {v1, v2, v3, v4},
                {pe1},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_3_5)
        {
            auto const& pe1 = edge_intersection_points[e5];
            auto const& pf1 = face_intersection_points[f2];
            auto const& pf2 = face_intersection_points[f1];
            subdivide_mesh_for_common_case_3(
                TV,
                TT,
                tetrahedron,
                {v2, v3, v1, v4},
                {pe1},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_3_6)
        {
            auto const& pe1 = edge_intersection_points[e6];
            auto const& pf1 = face_intersection_points[f3];
            auto const& pf2 = face_intersection_points[f2];
            subdivide_mesh_for_common_case_3(
                TV,
                TT,
                tetrahedron,
                {v3, v1, v2, v4},
                {pe1},
                {pf1, pf2});
            return true;
        }

        std::byte constexpr case_4_12{0b00000011};
        std::byte constexpr case_4_13{0b00000101};
        std::byte constexpr case_4_14{0b00001001};
        std::byte constexpr case_4_15{0b00010001};
        std::byte constexpr case_4_23{0b00000110};
        std::byte constexpr case_4_25{0b00010010};
        std::byte constexpr case_4_26{0b00100010};
        std::byte constexpr case_4_34{0b00001100};
        std::byte constexpr case_4_36{0b00100100};
        std::byte constexpr case_4_45{0b00011000};
        std::byte constexpr case_4_46{0b00101000};
        std::byte constexpr case_4_56{0b00110000};

        if (edge_intersection_mask == case_4_12)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e2];
            auto const& pf1 = face_intersection_points[f2];
            auto const& pf2 = face_intersection_points[f1];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v1, v3, v4, v2},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_13)
        {
            auto const& pe1 = edge_intersection_points[e3];
            auto const& pe2 = edge_intersection_points[e1];
            auto const& pf1 = face_intersection_points[f1];
            auto const& pf2 = face_intersection_points[f3];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v3, v2, v4, v1},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_14)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e4];
            auto const& pf1 = face_intersection_points[f3];
            auto const& pf2 = face_intersection_points[f4];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v2, v4, v3, v1},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_15)
        {
            auto const& pe1 = edge_intersection_points[e5];
            auto const& pe2 = edge_intersection_points[e1];
            auto const& pf1 = face_intersection_points[f4];
            auto const& pf2 = face_intersection_points[f2];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v4, v1, v3, v2},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_23)
        {
            auto const& pe1 = edge_intersection_points[e2];
            auto const& pe2 = edge_intersection_points[e3];
            auto const& pf1 = face_intersection_points[f3];
            auto const& pf2 = face_intersection_points[f2];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v2, v1, v4, v3},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_25)
        {
            auto const& pe1 = edge_intersection_points[e2];
            auto const& pe2 = edge_intersection_points[e5];
            auto const& pf1 = face_intersection_points[f1];
            auto const& pf2 = face_intersection_points[f4];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v3, v4, v1, v2},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_26)
        {
            auto const& pe1 = edge_intersection_points[e6];
            auto const& pe2 = edge_intersection_points[e2];
            auto const& pf1 = face_intersection_points[f4];
            auto const& pf2 = face_intersection_points[f3];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v4, v2, v1, v3},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_34)
        {
            auto const& pe1 = edge_intersection_points[e4];
            auto const& pe2 = edge_intersection_points[e3];
            auto const& pf1 = face_intersection_points[f4];
            auto const& pf2 = face_intersection_points[f1];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v4, v3, v2, v1},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_36)
        {
            auto const& pe1 = edge_intersection_points[e3];
            auto const& pe2 = edge_intersection_points[e6];
            auto const& pf1 = face_intersection_points[f2];
            auto const& pf2 = face_intersection_points[f4];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v1, v4, v2, v3},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_45)
        {
            auto const& pe1 = edge_intersection_points[e4];
            auto const& pe2 = edge_intersection_points[e5];
            auto const& pf1 = face_intersection_points[f2];
            auto const& pf2 = face_intersection_points[f3];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v1, v2, v3, v4},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_46)
        {
            auto const& pe1 = edge_intersection_points[e6];
            auto const& pe2 = edge_intersection_points[e4];
            auto const& pf1 = face_intersection_points[f1];
            auto const& pf2 = face_intersection_points[f2];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v3, v1, v2, v4},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_4_56)
        {
            auto const& pe1 = edge_intersection_points[e5];
            auto const& pe2 = edge_intersection_points[e6];
            auto const& pf1 = face_intersection_points[f3];
            auto const& pf2 = face_intersection_points[f1];
            subdivide_mesh_for_common_case_4(
                TV,
                TT,
                tetrahedron,
                {v2, v3, v1, v4},
                {pe1, pe2},
                {pf1, pf2});
            return true;
        }

        std::byte constexpr case_5_124{0b00001011};
        std::byte constexpr case_5_126{0b00100011};
        std::byte constexpr case_5_135{0b00010101};
        std::byte constexpr case_5_136{0b00100101};
        std::byte constexpr case_5_146{0b00101001};
        std::byte constexpr case_5_156{0b00110001};
        std::byte constexpr case_5_234{0b00001110};
        std::byte constexpr case_5_235{0b00010110};
        std::byte constexpr case_5_245{0b00011010};
        std::byte constexpr case_5_246{0b00101010};
        std::byte constexpr case_5_345{0b00011100};
        std::byte constexpr case_5_356{0b00110100};

        if (edge_intersection_mask == case_5_124)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e2];
            auto const& pe3 = edge_intersection_points[e4];
            auto const& pf1 = face_intersection_points[f3];
            auto const& pf2 = face_intersection_points[f2];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v3, v1, v4, v2},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_126)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e2];
            auto const& pe3 = edge_intersection_points[e6];
            auto const& pf1 = face_intersection_points[f3];
            auto const& pf2 = face_intersection_points[f1];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v1, v3, v4, v2},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_135)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e3];
            auto const& pe3 = edge_intersection_points[e5];
            auto const& pf1 = face_intersection_points[f3];
            auto const& pf2 = face_intersection_points[f2];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v4, v1, v3, v2},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_136)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e3];
            auto const& pe3 = edge_intersection_points[e6];
            auto const& pf1 = face_intersection_points[f1];
            auto const& pf2 = face_intersection_points[f2];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v4, v1, v2, v3},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_146)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e4];
            auto const& pe3 = edge_intersection_points[e6];
            auto const& pf1 = face_intersection_points[f2];
            auto const& pf2 = face_intersection_points[f4];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v2, v4, v3, v1},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_156)
        {
            auto const& pe1 = edge_intersection_points[e1];
            auto const& pe2 = edge_intersection_points[e5];
            auto const& pe3 = edge_intersection_points[e6];
            auto const& pf1 = face_intersection_points[f4];
            auto const& pf2 = face_intersection_points[f3];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v3, v2, v1, v4},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_234)
        {
            auto const& pe1 = edge_intersection_points[e2];
            auto const& pe2 = edge_intersection_points[e3];
            auto const& pe3 = edge_intersection_points[e4];
            auto const& pf1 = face_intersection_points[f1];
            auto const& pf2 = face_intersection_points[f2];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v2, v1, v4, v3},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_235)
        {
            auto const& pe1 = edge_intersection_points[e2];
            auto const& pe2 = edge_intersection_points[e3];
            auto const& pe3 = edge_intersection_points[e5];
            auto const& pf1 = face_intersection_points[f1];
            auto const& pf2 = face_intersection_points[f3];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v1, v2, v4, v3},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_245)
        {
            auto const& pe1 = edge_intersection_points[e2];
            auto const& pe2 = edge_intersection_points[e4];
            auto const& pe3 = edge_intersection_points[e5];
            auto const& pf1 = face_intersection_points[f4];
            auto const& pf2 = face_intersection_points[f3];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v1, v2, v3, v4},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_246)
        {
            auto const& pe1 = edge_intersection_points[e2];
            auto const& pe2 = edge_intersection_points[e4];
            auto const& pe3 = edge_intersection_points[e6];
            auto const& pf1 = face_intersection_points[f4];
            auto const& pf2 = face_intersection_points[f1];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v1, v3, v2, v4},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_345)
        {
            auto const& pe1 = edge_intersection_points[e3];
            auto const& pe2 = edge_intersection_points[e4];
            auto const& pe3 = edge_intersection_points[e5];
            auto const& pf1 = face_intersection_points[f4];
            auto const& pf2 = face_intersection_points[f2];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v2, v1, v3, v4},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }
        if (edge_intersection_mask == case_5_356)
        {
            auto const& pe1 = edge_intersection_points[e3];
            auto const& pe2 = edge_intersection_points[e5];
            auto const& pe3 = edge_intersection_points[e6];
            auto const& pf1 = face_intersection_points[f4];
            auto const& pf2 = face_intersection_points[f1];
            subdivide_mesh_for_common_case_5(
                TV,
                TT,
                tetrahedron,
                {v2, v3, v1, v4},
                {pe1, pe2, pe3},
                {pf1, pf2});
            return true;
        }

        return false;
    }

  private:
    void subdivide_mesh_for_common_case_1(
        Eigen::MatrixXd& TV,
        Eigen::MatrixXi& TT,
        int tetrahedron,
        std::array<int, 4u> const& vertex_ordering,
        std::array<Eigen::Vector3d, 3u> const& edge_intersection_points)
    {
        int const v1 = TT.row(tetrahedron)(vertex_ordering[0]);
        int const v2 = TT.row(tetrahedron)(vertex_ordering[1]);
        int const v3 = TT.row(tetrahedron)(vertex_ordering[2]);
        int const v4 = TT.row(tetrahedron)(vertex_ordering[3]);

        int const v5 = TV.rows();
        int const v6 = v5 + 1u;
        int const v7 = v5 + 2u;
        TV.conservativeResize(TV.rows() + 3u, Eigen::NoChange);

        TV.row(v5) = edge_intersection_points[0].transpose();
        TV.row(v6) = edge_intersection_points[1].transpose();
        TV.row(v7) = edge_intersection_points[2].transpose();

        int constexpr new_tetrahedron_count = 4u;
        int const t1                        = tetrahedron;
        int const t2                        = TT.rows();
        int const t3                        = TT.rows() + 1;
        int const t4                        = TT.rows() + 2;

        TT.conservativeResize(TT.rows() + (new_tetrahedron_count - 1), Eigen::NoChange);
        TT.row(t1) = Eigen::RowVector4i{v1, v2, v3, v7};
        TT.row(t2) = Eigen::RowVector4i{v1, v5, v2, v7};
        TT.row(t3) = Eigen::RowVector4i{v6, v5, v7, v2};
        TT.row(t4) = Eigen::RowVector4i{v5, v6, v7, v4};
    }

    void subdivide_mesh_for_common_case_2(
        Eigen::MatrixXd& TV,
        Eigen::MatrixXi& TT,
        int tetrahedron,
        std::array<int, 4u> const& vertex_ordering,
        std::array<Eigen::Vector3d, 4u> const& edge_intersection_points)
    {
        int const v1 = TT.row(tetrahedron)(vertex_ordering[0]);
        int const v2 = TT.row(tetrahedron)(vertex_ordering[1]);
        int const v3 = TT.row(tetrahedron)(vertex_ordering[2]);
        int const v4 = TT.row(tetrahedron)(vertex_ordering[3]);

        int constexpr new_vertex_count = 4u;
        int const v5                   = TV.rows();
        int const v6                   = v5 + 1u;
        int const v7                   = v5 + 2u;
        int const v8                   = v5 + 3u;
        TV.conservativeResize(TV.rows() + new_vertex_count, Eigen::NoChange);

        TV.row(v5) = edge_intersection_points[0].transpose();
        TV.row(v6) = edge_intersection_points[1].transpose();
        TV.row(v7) = edge_intersection_points[2].transpose();
        TV.row(v8) = edge_intersection_points[3].transpose();

        int constexpr new_tetrahedron_count = 6u;
        int const t1                        = tetrahedron;
        int const t2                        = TT.rows();
        int const t3                        = TT.rows() + 1;
        int const t4                        = TT.rows() + 2;
        int const t5                        = TT.rows() + 3;
        int const t6                        = TT.rows() + 4;

        TT.conservativeResize(TT.rows() + (new_tetrahedron_count - 1), Eigen::NoChange);
        TT.row(t1) = Eigen::RowVector4i{v5, v7, v6, v4};
        TT.row(t2) = Eigen::RowVector4i{v6, v7, v8, v4};
        TT.row(t3) = Eigen::RowVector4i{v6, v8, v3, v4};
        TT.row(t4) = Eigen::RowVector4i{v5, v6, v7, v1};
        TT.row(t5) = Eigen::RowVector4i{v1, v2, v6, v7};
        TT.row(t6) = Eigen::RowVector4i{v2, v8, v6, v7};
    }

    void subdivide_mesh_for_common_case_3(
        Eigen::MatrixXd& TV,
        Eigen::MatrixXi& TT,
        int tetrahedron,
        std::array<int, 4u> const& vertex_ordering,
        std::array<Eigen::Vector3d, 1u> const& edge_intersection_points,
        std::array<Eigen::Vector3d, 2u> const& face_intersection_points)
    {
        int const v1 = TT.row(tetrahedron)(vertex_ordering[0]);
        int const v2 = TT.row(tetrahedron)(vertex_ordering[1]);
        int const v3 = TT.row(tetrahedron)(vertex_ordering[2]);
        int const v4 = TT.row(tetrahedron)(vertex_ordering[3]);

        int constexpr new_vertex_count = 3u;
        int const v5                   = TV.rows();
        int const v6                   = v5 + 1u;
        int const v7                   = v5 + 2u;
        TV.conservativeResize(TV.rows() + new_vertex_count, Eigen::NoChange);

        TV.row(v5) = edge_intersection_points[0].transpose();
        TV.row(v6) = face_intersection_points[0].transpose();
        TV.row(v7) = face_intersection_points[1].transpose();

        int constexpr new_tetrahedron_count = 6u;
        int const t1                        = tetrahedron;
        int const t2                        = TT.rows();
        int const t3                        = TT.rows() + 1;
        int const t4                        = TT.rows() + 2;
        int const t5                        = TT.rows() + 3;
        int const t6                        = TT.rows() + 4;

        TT.conservativeResize(TT.rows() + (new_tetrahedron_count - 1), Eigen::NoChange);
        TT.row(t1) = Eigen::RowVector4i{v5, v6, v7, v4};
        TT.row(t2) = Eigen::RowVector4i{v5, v7, v6, v1};
        TT.row(t3) = Eigen::RowVector4i{v6, v3, v7, v4};
        TT.row(t4) = Eigen::RowVector4i{v2, v3, v6, v4};
        TT.row(t5) = Eigen::RowVector4i{v1, v7, v6, v3};
        TT.row(t6) = Eigen::RowVector4i{v1, v2, v3, v6};
    }

    void subdivide_mesh_for_common_case_4(
        Eigen::MatrixXd& TV,
        Eigen::MatrixXi& TT,
        int tetrahedron,
        std::array<int, 4u> const& vertex_ordering,
        std::array<Eigen::Vector3d, 2u> const& edge_intersection_points,
        std::array<Eigen::Vector3d, 2u> const& face_intersection_points)
    {
        int const v1 = TT.row(tetrahedron)(vertex_ordering[0]);
        int const v2 = TT.row(tetrahedron)(vertex_ordering[1]);
        int const v3 = TT.row(tetrahedron)(vertex_ordering[2]);
        int const v4 = TT.row(tetrahedron)(vertex_ordering[3]);

        int constexpr new_vertex_count = 4u;
        int const v5                   = TV.rows();
        int const v6                   = v5 + 1u;
        int const v7                   = v5 + 2u;
        int const v8                   = v5 + 3u;
        TV.conservativeResize(TV.rows() + new_vertex_count, Eigen::NoChange);

        TV.row(v5) = edge_intersection_points[0].transpose();
        TV.row(v6) = edge_intersection_points[1].transpose();
        TV.row(v7) = face_intersection_points[0].transpose();
        TV.row(v8) = face_intersection_points[1].transpose();

        int constexpr new_tetrahedron_count = 8u;
        int const t1                        = tetrahedron;
        int const t2                        = TT.rows();
        int const t3                        = TT.rows() + 1;
        int const t4                        = TT.rows() + 2;
        int const t5                        = TT.rows() + 3;
        int const t6                        = TT.rows() + 4;
        int const t7                        = TT.rows() + 5;
        int const t8                        = TT.rows() + 6;

        TT.conservativeResize(TT.rows() + (new_tetrahedron_count - 1), Eigen::NoChange);
        TT.row(t1) = Eigen::RowVector4i{v5, v6, v7, v4};
        TT.row(t2) = Eigen::RowVector4i{v5, v7, v8, v4};
        TT.row(t3) = Eigen::RowVector4i{v4, v7, v8, v3};
        TT.row(t4) = Eigen::RowVector4i{v7, v6, v5, v2};
        TT.row(t5) = Eigen::RowVector4i{v8, v7, v5, v2};
        TT.row(t6) = Eigen::RowVector4i{v8, v7, v2, v3};
        TT.row(t7) = Eigen::RowVector4i{v5, v8, v2, v1};
        TT.row(t8) = Eigen::RowVector4i{v1, v2, v3, v8};
    }

    void subdivide_mesh_for_common_case_5(
        Eigen::MatrixXd& TV,
        Eigen::MatrixXi& TT,
        int tetrahedron,
        std::array<int, 4u> const& vertex_ordering,
        std::array<Eigen::Vector3d, 3u> const& edge_intersection_points,
        std::array<Eigen::Vector3d, 2u> const& face_intersection_points)
    {
        int const v1 = TT.row(tetrahedron)(vertex_ordering[0]);
        int const v2 = TT.row(tetrahedron)(vertex_ordering[1]);
        int const v3 = TT.row(tetrahedron)(vertex_ordering[2]);
        int const v4 = TT.row(tetrahedron)(vertex_ordering[3]);

        int constexpr new_vertex_count = 5u;
        int const v5                   = TV.rows();
        int const v6                   = v5 + 1u;
        int const v7                   = v5 + 2u;
        int const v8                   = v5 + 3u;
        int const v9                   = v5 + 4u;
        TV.conservativeResize(TV.rows() + new_vertex_count, Eigen::NoChange);

        TV.row(v5) = edge_intersection_points[0].transpose();
        TV.row(v6) = edge_intersection_points[1].transpose();
        TV.row(v7) = edge_intersection_points[2].transpose();
        TV.row(v8) = face_intersection_points[0].transpose();
        TV.row(v9) = face_intersection_points[1].transpose();

        int constexpr new_tetrahedron_count = 9u;
        int const t1                        = tetrahedron;
        int const t2                        = TT.rows();
        int const t3                        = TT.rows() + 1;
        int const t4                        = TT.rows() + 2;
        int const t5                        = TT.rows() + 3;
        int const t6                        = TT.rows() + 4;
        int const t7                        = TT.rows() + 5;
        int const t8                        = TT.rows() + 6;
        int const t9                        = TT.rows() + 7;

        TT.conservativeResize(TT.rows() + (new_tetrahedron_count - 1), Eigen::NoChange);
        TT.row(t1) = Eigen::RowVector4i{v9, v6, v5, v1};
        TT.row(t2) = Eigen::RowVector4i{v9, v8, v6, v1};
        TT.row(t3) = Eigen::RowVector4i{v1, v8, v3, v9};
        TT.row(t4) = Eigen::RowVector4i{v8, v7, v6, v1};
        TT.row(t5) = Eigen::RowVector4i{v2, v6, v7, v1};
        TT.row(t6) = Eigen::RowVector4i{v5, v6, v9, v4};
        TT.row(t7) = Eigen::RowVector4i{v9, v6, v3, v4};
        TT.row(t8) = Eigen::RowVector4i{v9, v8, v3, v6};
        TT.row(t9) = Eigen::RowVector4i{v8, v7, v3, v6};
    }
};

std::pair<std::byte, std::array<Eigen::Vector3d, 6u>> get_edge_intersections(
    Eigen::Vector3d const& v1,
    Eigen::Vector3d const& v2,
    Eigen::Vector3d const& v3,
    Eigen::Vector3d const& v4,
    Eigen::Vector3d const& a,
    Eigen::Vector3d const& b,
    Eigen::Vector3d const& c)
{
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

    std::byte mask{0b00000000};
    std::array<Eigen::Vector3d, 6u> edge_intersections{};

    if (e1_intersected)
    {
        mask |= std::byte{0b00000001};
        edge_intersections[0] = e1_intersection;
    }
    if (e2_intersected)
    {
        mask |= std::byte{0b00000010};
        edge_intersections[1] = e2_intersection;
    }
    if (e3_intersected)
    {
        mask |= std::byte{0b00000100};
        edge_intersections[2] = e3_intersection;
    }
    if (e4_intersected)
    {
        mask |= std::byte{0b00001000};
        edge_intersections[3] = e4_intersection;
    }
    if (e5_intersected)
    {
        mask |= std::byte{0b00010000};
        edge_intersections[4] = e5_intersection;
    }
    if (e6_intersected)
    {
        mask |= std::byte{0b00100000};
        edge_intersections[5] = e6_intersection;
    }

    return {mask, edge_intersections};
}

std::pair<std::byte, std::array<Eigen::Vector3d, 4u>> get_face_intersections(
    Eigen::Vector3d const& pos1,
    Eigen::Vector3d const& pos2,
    Eigen::Vector3d const& pos3,
    Eigen::Vector3d const& pos4,
    std::pair<Eigen::Vector3d, Eigen::Vector3d> const& start_line,
    std::pair<Eigen::Vector3d, Eigen::Vector3d> const& end_line)
{
    auto const& p1 = start_line.first;
    auto const& q1 = start_line.second;
    auto const& p2 = end_line.first;
    auto const& q2 = end_line.second;

    int constexpr v1 = 0;
    int constexpr v2 = 1;
    int constexpr v3 = 2;
    int constexpr v4 = 3;

    Eigen::MatrixXi TF(4u, 3u);
    TF.row(0u) = Eigen::RowVector3i{v1, v2, v4};
    TF.row(1u) = Eigen::RowVector3i{v2, v3, v4};
    TF.row(2u) = Eigen::RowVector3i{v3, v1, v4};
    TF.row(3u) = Eigen::RowVector3i{v1, v3, v2};

    int constexpr f1 = 0;
    int constexpr f2 = 1;
    int constexpr f3 = 2;
    int constexpr f4 = 3;

    std::array<Eigen::Vector3d, 4u> vertices{pos1, pos2, pos3, pos4};

    std::array<Eigen::Vector3d, 4u> face_intersections{};

    std::byte start_line_mask{0b00000000};

    // intersect tet faces with line p1-q1
    {
        auto const [f1_intersected, f1_intersection] = geometry::triangle_line_intersection_two_way(
            vertices[TF(f1, 0)],
            vertices[TF(f1, 1)],
            vertices[TF(f1, 2)],
            {p1, q1});

        auto const [f2_intersected, f2_intersection] = geometry::triangle_line_intersection_two_way(
            vertices[TF(f2, 0)],
            vertices[TF(f2, 1)],
            vertices[TF(f2, 2)],
            {p1, q1});

        auto const [f3_intersected, f3_intersection] = geometry::triangle_line_intersection_two_way(
            vertices[TF(f3, 0)],
            vertices[TF(f3, 1)],
            vertices[TF(f3, 2)],
            {p1, q1});

        auto const [f4_intersected, f4_intersection] = geometry::triangle_line_intersection_two_way(
            vertices[TF(f4, 0)],
            vertices[TF(f4, 1)],
            vertices[TF(f4, 2)],
            {p1, q1});

        if (f1_intersected)
        {
            start_line_mask |= std::byte{0b00000001};
            face_intersections[0] = f1_intersection;
        }
        if (f2_intersected)
        {
            start_line_mask |= std::byte{0b00000010};
            face_intersections[1] = f2_intersection;
        }
        if (f3_intersected)
        {
            start_line_mask |= std::byte{0b00000100};
            face_intersections[2] = f3_intersection;
        }
        if (f4_intersected)
        {
            start_line_mask |= std::byte{0b00001000};
            face_intersections[3] = f4_intersection;
        }
    }

    std::byte end_line_mask{0b00000000};
    // intersect tet faces with line p2-q2
    {
        auto const [f1_intersected, f1_intersection] = geometry::triangle_line_intersection_two_way(
            vertices[TF(f1, 0)],
            vertices[TF(f1, 1)],
            vertices[TF(f1, 2)],
            {p2, q2});

        auto const [f2_intersected, f2_intersection] = geometry::triangle_line_intersection_two_way(
            vertices[TF(f2, 0)],
            vertices[TF(f2, 1)],
            vertices[TF(f2, 2)],
            {p2, q2});

        auto const [f3_intersected, f3_intersection] = geometry::triangle_line_intersection_two_way(
            vertices[TF(f3, 0)],
            vertices[TF(f3, 1)],
            vertices[TF(f3, 2)],
            {p2, q2});

        auto const [f4_intersected, f4_intersection] = geometry::triangle_line_intersection_two_way(
            vertices[TF(f4, 0)],
            vertices[TF(f4, 1)],
            vertices[TF(f4, 2)],
            {p2, q2});

        if (f1_intersected)
        {
            end_line_mask |= std::byte{0b00000001};
            face_intersections[0] = f1_intersection;
        }
        if (f2_intersected)
        {
            end_line_mask |= std::byte{0b00000010};
            face_intersections[1] = f2_intersection;
        }
        if (f3_intersected)
        {
            end_line_mask |= std::byte{0b00000100};
            face_intersections[2] = f3_intersection;
        }
        if (f4_intersected)
        {
            end_line_mask |= std::byte{0b00001000};
            face_intersections[3] = f4_intersection;
        }
    }

    if ((start_line_mask & end_line_mask) != std::byte{0b00000000})
    {
        return {std::byte{0b00000000}, face_intersections};
    }

    std::byte const mask = start_line_mask | end_line_mask;
    return {mask, face_intersections};
}

bool cut_tetrahedron(
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& T,
    int tetrahedron,
    std::pair<Eigen::Vector3d, Eigen::Vector3d> const& start_line,
    std::pair<Eigen::Vector3d, Eigen::Vector3d> const& end_line)
{
    auto const& p1 = start_line.first;
    auto const& q1 = start_line.second;
    auto const& p2 = end_line.first;
    auto const& q2 = end_line.second;

    // only triangle cutting surfaces are supported for now
    if (p1 != p2)
        return false;

    // parallel lines and p1 == p2 means lines are overlapping
    if ((q1 - p1).normalized() == (q2 - p2).normalized())
        return false;

    // get edge intersections
    Eigen::Vector3d const& a = p1;
    Eigen::Vector3d const& b = q1;
    Eigen::Vector3d const& c = q2;

    auto const& pos1 = V.row(T(tetrahedron, 0)).transpose();
    auto const& pos2 = V.row(T(tetrahedron, 1)).transpose();
    auto const& pos3 = V.row(T(tetrahedron, 2)).transpose();
    auto const& pos4 = V.row(T(tetrahedron, 3)).transpose();

    auto edge_pair = get_edge_intersections(pos1, pos2, pos3, pos4, a, b, c);
    std::byte const edge_intersection_mask                   = edge_pair.first;
    std::array<Eigen::Vector3d, 6u> const edge_intersections = edge_pair.second;

    auto face_pair = get_face_intersections(pos1, pos2, pos3, pos4, start_line, end_line);
    std::byte const face_intersection_mask                   = face_pair.first;
    std::array<Eigen::Vector3d, 4u> const face_intersections = face_pair.second;

    tetrahedron_mesh_cutter_t cutter{};
    bool const result = cutter.subdivide_mesh(
        edge_intersection_mask,
        V,
        T,
        tetrahedron,
        edge_intersections,
        face_intersections);

    return result;
}

} // namespace geometry

#endif // TET_CUT_CUT_TETRAHEDRON_HPP