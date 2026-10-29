//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#include "interface/update.hpp"

#include <memory>
#include <vector>

#include "config.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pack/sparse_pack/make_pack_descriptor.hpp"
#include "pack/sparse_pack/sparse_pack.hpp"
#include "sparse/sparse_management.hpp"

namespace parthenon {

namespace Update {

template <>
TaskStatus EmfCurl(
    MeshData<Real> *input, MeshData<Real> *rhs,
    const std::vector<std::string> &field_names) 
{
  const int ndim =
      input->GetBlockData(0)->GetBlockPointer()->pmy_mesh->ndim;

  const auto &input_pack =
      input->PackVariablesAndFluxes(field_names);

  auto rhs_pack = rhs->PackVariables(field_names);

  PARTHENON_REQUIRE(input_pack.GetDim(5) == rhs_pack.GetDim(5),
                    "EmfCurl: input and RHS packs have different block counts.");
  PARTHENON_REQUIRE(input_pack.GetDim(4) == rhs_pack.GetDim(4),
                    "EmfCurl: input and RHS packs have different component counts.");

    const auto ib1 = input->GetBoundsI(IndexDomain::interior, TE::F1);
    const auto jb1 = input->GetBoundsJ(IndexDomain::interior, TE::F1);
    const auto kb1 = input->GetBoundsK(IndexDomain::interior, TE::F1);

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "EmfCurl::X1", DevExecSpace(),
        0, input_pack.GetDim(5) - 1,
        0, input_pack.GetDim(4) - 1,
        kb1.s, kb1.e, jb1.s, jb1.e, ib1.s, ib1.e,
        KOKKOS_LAMBDA(const int b, const int l,
                    const int k, const int j, const int i) {
        if (!input_pack.IsAllocated(b, l) ||
            !rhs_pack.IsAllocated(b, l)) {
            return;
        }

        const auto &coords = input_pack.GetCoords(b);
        const auto &v = input_pack(b);
        auto &out = rhs_pack(b);

        out(TE::F1, l, k, j, i) =
            EmfCurlHelper<TE::F1>(
                l, k, j, i, ndim, coords, v);
        });
    if (ndim >= 2) {
    const auto ib2 = input->GetBoundsI(IndexDomain::interior, TE::F2);
    const auto jb2 = input->GetBoundsJ(IndexDomain::interior, TE::F2);
    const auto kb2 = input->GetBoundsK(IndexDomain::interior, TE::F2);

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "EmfCurl::X2", DevExecSpace(),
        0, input_pack.GetDim(5) - 1,
        0, input_pack.GetDim(4) - 1,
        kb2.s, kb2.e, jb2.s, jb2.e, ib2.s, ib2.e,
        KOKKOS_LAMBDA(const int b, const int l,
                    const int k, const int j, const int i) {
        if (!input_pack.IsAllocated(b, l) ||
            !rhs_pack.IsAllocated(b, l)) {
            return;
        }

        const auto &coords = input_pack.GetCoords(b);
        const auto &v = input_pack(b);
        auto &out = rhs_pack(b);

        out(TE::F2, l, k, j, i) =
            EmfCurlHelper<TE::F2>(
                l, k, j, i, ndim, coords, v);
        });
    }
    const auto ib3 = input->GetBoundsI(IndexDomain::interior, TE::F3);
    const auto jb3 = input->GetBoundsJ(IndexDomain::interior, TE::F3);
    const auto kb3 = input->GetBoundsK(IndexDomain::interior, TE::F3);

    if (ndim >= 3){
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "EmfCurl::X3", DevExecSpace(),
            0, input_pack.GetDim(5) - 1,
            0, input_pack.GetDim(4) - 1,
            kb3.s, kb3.e, jb3.s, jb3.e, ib3.s, ib3.e,
            KOKKOS_LAMBDA(const int b, const int l,
                        const int k, const int j, const int i) {
            if (!input_pack.IsAllocated(b, l) ||
                !rhs_pack.IsAllocated(b, l)) {
                return;
            }

            const auto &coords = input_pack.GetCoords(b);
            const auto &v = input_pack(b);
            auto &out = rhs_pack(b);

            out(TE::F3, l, k, j, i) =
                EmfCurlHelper<TE::F3>(
                    l, k, j, i, ndim, coords, v);
            });
    }
  return TaskStatus::complete;
}

template <>
TaskStatus UpdateWithEmfCurl(MeshData<Real> *mu0, MeshData<Real> *mu1, 
    const Real gam0, const Real gam1, 
    const Real beta_dt, const std::vector<std::string> &field_names) {
    auto pmb = mu0->GetBlockData(0)->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;

    auto B0_pack = mu0->PackVariablesAndFluxes(field_names);
    const auto &B1_pack = mu1->PackVariables(field_names);

    PARTHENON_REQUIRE(B0_pack.GetDim(5) == B1_pack.GetDim(5),
                      "UpdateWithEmfCurl: register packs have different block counts.");
    PARTHENON_REQUIRE(B0_pack.GetDim(4) == B1_pack.GetDim(4),
                      "UpdateWithEmfCurl: register packs have different component counts.");

    const auto ib1 = mu0->GetBoundsI(IndexDomain::interior, TE::F1);
    const auto jb1 = mu0->GetBoundsJ(IndexDomain::interior, TE::F1);
    const auto kb1 = mu0->GetBoundsK(IndexDomain::interior, TE::F1);

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ApplyFaceUpdate::X1", parthenon::DevExecSpace(), 0,
        B0_pack.GetDim(5) - 1, 0, B0_pack.GetDim(4) - 1, kb1.s, kb1.e, jb1.s, jb1.e, ib1.s, ib1.e,
        KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
            if (!B0_pack.IsAllocated(b, l) ||
                !B1_pack.IsAllocated(b, l)) {
              return;
            }

            auto &B0 = B0_pack(b);
            auto &B1 = B1_pack(b);

            Real &B0n = B0(TE::F1,  l, k, j, i);
            Real &B1n = B1(TE::F1,  l, k, j, i);

            const auto &coords = B0_pack.GetCoords(b);

            const Real rhs = EmfCurlHelper<TE::F1>(l, k, j, i, ndim, coords, B0);

            B0n = gam0 * B0n
                + gam1 * B1n
                + beta_dt * rhs;
            

        });
    if (ndim >= 2) {
    const auto ib2 = mu0->GetBoundsI(IndexDomain::interior, TE::F2);
    const auto jb2 = mu0->GetBoundsJ(IndexDomain::interior, TE::F2);
    const auto kb2 = mu0->GetBoundsK(IndexDomain::interior, TE::F2);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ApplyFaceUpdate::X2", parthenon::DevExecSpace(), 0,
        B0_pack.GetDim(5) - 1, 0, B0_pack.GetDim(4) - 1, kb2.s, kb2.e, jb2.s, jb2.e, ib2.s, ib2.e,
        KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
            if (!B0_pack.IsAllocated(b, l) ||
                !B1_pack.IsAllocated(b, l)) {
              return;
            }
            const auto &coords = B0_pack.GetCoords(b);

            auto &B0 = B0_pack(b);
            auto &B1 = B1_pack(b);

            Real &B0n = B0(TE::F2,  l, k, j, i);
            Real &B1n = B1(TE::F2,  l, k, j, i);

            const Real rhs = EmfCurlHelper<TE::F2>(l, k, j, i, ndim, coords, B0);

            B0n = gam0 * B0n
                + gam1 * B1n
                + beta_dt * rhs;
        });
    }
    const auto ib3 = mu0->GetBoundsI(IndexDomain::interior, TE::F3);
    const auto jb3 = mu0->GetBoundsJ(IndexDomain::interior, TE::F3);
    const auto kb3 = mu0->GetBoundsK(IndexDomain::interior, TE::F3);
    if (ndim >= 3){
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ApplyFaceUpdate::X3", parthenon::DevExecSpace(), 0,
        B0_pack.GetDim(5) - 1, 0, B0_pack.GetDim(4) - 1, kb3.s, kb3.e, jb3.s, jb3.e, ib3.s, ib3.e,
        KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
            if (!B0_pack.IsAllocated(b, l) ||
                !B1_pack.IsAllocated(b, l)) {
              return;
            }
            const auto &coords = B0_pack.GetCoords(b);

            auto &B0 = B0_pack(b);
            auto &B1 = B1_pack(b);

            Real &B0n = B0(TE::F3,  l, k, j, i);
            Real &B1n = B1(TE::F3,  l, k, j, i);

            const Real rhs = EmfCurlHelper<TE::F3>(l, k, j, i, ndim, coords, B0);

            B0n = gam0 * B0n
                + gam1 * B1n
                + beta_dt * rhs;
            
        });
    }
    return TaskStatus::complete;
}

} // namespace Update

} // namespace parthenon
