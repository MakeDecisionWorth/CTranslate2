#include "gemm_xmx.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include "ctranslate2/allocator.h"
#include "utils.h"

namespace ctranslate2 {
  namespace xpu {

    namespace jm = ::sycl::ext::oneapi::experimental::matrix;

    // GEMM written against the XMX engines directly, in place of oneMKL's.
    //
    // Why: oneMKL's float16 GEMM faults the GPU (GEN12_OCL_PAGEFAULT) whenever a desktop
    // compositor is working on the same card - an Arc A770 with a monitor on it, or
    // streamed over a remote desktop - at beam_size >= 4 on Whisper large-v2, through
    // Level Zero and OpenCL alike. These kernels run the same DPAS instructions and have
    // never faulted there. float32 goes through them too, as split bfloat16 (see the
    // header), so that no GEMM depends on oneMKL.

    namespace {

      using half = ::sycl::half;
      using bf16 = ::sycl::ext::oneapi::bfloat16;

      // XMX8, Alchemist's shape for 16-bit inputs into float: an 8x16 A tile times a 16x8
      // B tile, executed by a sub-group of 8 (static-query-use.hpp, intel_gpu_dg2_*).
      constexpr size_t TM = 8;
      constexpr size_t TN = 8;
      constexpr size_t TK = 16;
      constexpr size_t SG_SIZE = 8;

      // What this hardware can and cannot load into the XMX engines decides the whole
      // layout of this file:
      //
      // - The A operand loads natively from a row-major matrix. Column-major does not
      //   build for dg2 at all (__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_ColumnMajor).
      // - The B operand's native layout is VNNI-packed - rows k and k+1 of a column side
      //   by side. Row- and column-major B loads build, but have to be reshuffled in
      //   registers on every load.
      // - There are no bounds-checked loads ("SYCL Joint Matrix checked load API is not
      //   supported on targeted GPU device"), so a tile that crosses an edge has to be
      //   staged through local memory with zero padding.
      // - A load straight from global memory is a sub-group block read, and every row it
      //   reads must start 8-byte aligned. A misaligned start is not rejected: the
      //   hardware rounds the address down and the tile comes back as garbage. Measured:
      //   a row stride of 3000 bytes loads correctly, an odd number of halves does not.
      //
      // So the kernel computes P[R][Q] = L[R][K] * Rt[K][Q], with L row-major and Rt
      // always repacked into VNNI and zero-padded to whole tiles first. For a Dense layer,
      // x[m,k] * W[n,k]^T, that means computing C^T = W x^T: the weights - large - are L
      // and are read in place, and only x is repacked. (The large tilings copy L too,
      // when it is read often enough to pay for it: see TiledL in run().)
      constexpr size_t LOAD_ALIGN = 8;

      void check_device_has_xmx() {
        static const bool supported =
          get_queue().get_device().has(::sycl::aspect::ext_intel_matrix);
        if (!supported)
          THROW_RUNTIME_ERROR("This XPU device has no XMX matrix engines, which every "
                              "float16 and float32 GEMM on it runs on");
      }

      template <typename Tm>
      bool loadable(const Tm* hi, const Tm* lo, size_t ld, size_t stride, size_t batches) {
        auto aligned = [](size_t bytes) { return bytes % LOAD_ALIGN == 0; };
        auto aligned_ptr = [&](const Tm* ptr) {
          return !ptr || aligned(reinterpret_cast<uintptr_t>(ptr));
        };
        return aligned(ld * sizeof (Tm))
          && (batches == 1 || aligned(stride * sizeof (Tm)))
          && aligned_ptr(hi) && aligned_ptr(lo);
      }

      size_t round_up(size_t value, size_t multiple) {
        return (value + multiple - 1) / multiple * multiple;
      }

      struct Problem {
        size_t R, Q, K;
        size_t ld_l, stride_l;  // L, row-major, as the caller stored it
        size_t ld_r, stride_r;  // Rt, VNNI-packed: [Kp / 2][Qp * 2] per batch
        size_t ld_c, stride_c;
        size_t batches;
        float alpha, beta;
        // Tiles per row of the tile-major copies (see TiledL in run()): k steps in L,
        // TN-wide column tiles in Rt.
        size_t ktiles, qtiles;
      };

      template <typename Tm>
      inline ::sycl::multi_ptr<const Tm, ::sycl::access::address_space::global_space,
                               ::sycl::access::decorated::no>
      global(const Tm* ptr) {
        return ::sycl::address_space_cast<::sycl::access::address_space::global_space,
                                          ::sycl::access::decorated::no>(ptr);
      }

      // Copies the TM x TK tile of row-major L at (row0, col0) into the sub-group's own slice
      // of local memory, zero past the matrix's edges, so it loads like a whole tile.
      template <typename Tm, typename LocalPtr>
      inline void stage_edge_tile(const ::sycl::sub_group& sg, const Tm* src, size_t ld,
                                  size_t rows, size_t cols, size_t row0, size_t col0,
                                  LocalPtr scratch) {
        const size_t lane = sg.get_local_linear_id();
#pragma unroll
        for (size_t e = lane; e < TM * TK; e += SG_SIZE) {
          const size_t rr = e / TK;
          const size_t cc = e % TK;
          const size_t gr = row0 + rr;
          const size_t gc = col0 + cc;
          Tm v = Tm(0.0f);
          if (gr < rows && gc < cols)
            v = src[gr * ld + gc];
          scratch[rr * TK + cc] = v;
        }
      }

      // One sub-group owns an (ACC_R * TM) x (ACC_C * TN) block of P and a 1/SPLIT_K share
      // of the k steps. A work-group is SG_R x SG_Q such blocks, times SPLIT_K. The shares
      // are summed in local memory, which is also where alpha, beta, the edges and the
      // conversion to T are applied on the way out.
      //
      // T is the type of C. Tm is what the XMX engines are fed: half, or for float32 the
      // bfloat16 halves of each value, in which case Split adds the hi*lo and lo*hi terms.
      //
      // TiledL, TiledR: L or Rt has been copied tile-major and zero-padded to whole
      // work-group tiles, so each of its tiles is 256 contiguous bytes and none crosses an
      // edge. Read in place, a tile is 8 rows of 32 bytes, each a separate request - 48
      // requests for the 8 DPAS of a k step in the large tiling, which the load path
      // cannot keep up with.
      template <typename T, typename Tm, bool Split, bool TransposeOut, bool TiledL, bool TiledR,
                size_t ACC_R, size_t ACC_C, size_t SG_R, size_t SG_Q, size_t SPLIT_K>
      void run(const Problem& pb,
               const Tm* l_hi, const Tm* l_lo, const Tm* r_hi, const Tm* r_lo, T* c) {
        constexpr size_t ROWS = ACC_R * TM;
        constexpr size_t COLS = ACC_C * TN;
        constexpr size_t WG_ROWS = SG_R * ROWS;
        constexpr size_t WG_COLS = SG_Q * COLS;
        constexpr size_t NUM_SG = SG_R * SG_Q * SPLIT_K;
        constexpr size_t WG_SIZE = NUM_SG * SG_SIZE;
        constexpr size_t BLOCK = ROWS * COLS;
        constexpr size_t A_TILE = TM * TK;
        constexpr size_t SCRATCH = ACC_R * A_TILE * (Split ? 2 : 1);

        const Problem p = pb;
        const size_t tiles_r = (p.R + WG_ROWS - 1) / WG_ROWS;
        const size_t tiles_q = (p.Q + WG_COLS - 1) / WG_COLS;
        const size_t ksteps = (p.K + TK - 1) / TK;
        const size_t groups = p.batches * tiles_r * tiles_q;
        const bool l_loadable = loadable(l_hi, l_lo, p.ld_l, p.stride_l, p.batches);

        auto submit = [&](::sycl::handler& h) {
          // With split-K the sub-groups' blocks are summed here, so every block is kept.
          // Without it each sub-group only passes its accumulators through one tile at a
          // time on the way out - local memory is what limits how many work-groups an
          // Xe-core holds at once, and the whole blocks were 16 KB a work-group.
          ::sycl::local_accessor<float, 1> partial(
            ::sycl::range<1>(NUM_SG * (SPLIT_K > 1 ? BLOCK : TM * TN)), h);
          ::sycl::local_accessor<Tm, 1> scratch(::sycl::range<1>(NUM_SG * SCRATCH), h);

          h.parallel_for(
            ::sycl::nd_range<1>(groups * WG_SIZE, WG_SIZE),
            [=](::sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
              const size_t group = it.get_group(0);
              const size_t tile_q = group % tiles_q;
              const size_t rest = group / tiles_q;
              const size_t tile_r = rest % tiles_r;
              const size_t bi = rest / tiles_r;

              auto sg = it.get_sub_group();
              const size_t sg_id = sg.get_group_linear_id();
              const size_t share = sg_id % SPLIT_K;
              const size_t sq = (sg_id / SPLIT_K) % SG_Q;
              const size_t sr = sg_id / (SPLIT_K * SG_Q);
              const size_t r0 = tile_r * WG_ROWS + sr * ROWS;
              const size_t q0 = tile_q * WG_COLS + sq * COLS;

              const Tm* lh = l_hi + bi * p.stride_l;
              const Tm* rh = r_hi + bi * p.stride_r;
              const Tm* ll = Split ? l_lo + bi * p.stride_l : nullptr;
              const Tm* rl = Split ? r_lo + bi * p.stride_r : nullptr;

              auto mine = scratch.template get_multi_ptr<::sycl::access::decorated::no>()
                          + sg_id * SCRATCH;

              jm::joint_matrix<::sycl::sub_group, float, jm::use::accumulator, TM, TN>
                acc[ACC_R][ACC_C];
#pragma unroll
              for (size_t i = 0; i < ACC_R; ++i)
#pragma unroll
                for (size_t j = 0; j < ACC_C; ++j)
                  jm::joint_matrix_fill(sg, acc[i][j], 0.0f);

              // Rt is padded to whole work-group tiles, so only L can cross an edge.
              const bool rows_whole = l_loadable && r0 + ROWS <= p.R;

              for (size_t step = share; step < ksteps; step += SPLIT_K) {
                const size_t k0 = step * TK;
                const bool l_whole = rows_whole && k0 + TK <= p.K;

                jm::joint_matrix<::sycl::sub_group, Tm, jm::use::a, TM, TK,
                                 jm::layout::row_major> a_hi[ACC_R], a_lo[ACC_R];
#pragma unroll
                for (size_t i = 0; i < ACC_R; ++i) {
                  const size_t r = r0 + i * TM;
                  if constexpr (TiledL) {
                    const size_t a_off = ((r / TM) * p.ktiles + step) * A_TILE;
                    jm::joint_matrix_load(sg, a_hi[i], global(lh) + a_off, TK);
                    if constexpr (Split)
                      jm::joint_matrix_load(sg, a_lo[i], global(ll) + a_off, TK);
                  } else if (l_whole) {
                    jm::joint_matrix_load(sg, a_hi[i], global(lh) + r * p.ld_l + k0, p.ld_l);
                    if constexpr (Split)
                      jm::joint_matrix_load(sg, a_lo[i], global(ll) + r * p.ld_l + k0,
                                            p.ld_l);
                  } else {
                    auto slot = mine + i * A_TILE;
                    // The previous k step's tile may still be being read from this slot.
                    ::sycl::group_barrier(sg);
                    stage_edge_tile(sg, lh, p.ld_l, p.R, p.K, r, k0, slot);
                    if constexpr (Split)
                      stage_edge_tile(sg, ll, p.ld_l, p.R, p.K, r, k0,
                                      slot + ACC_R * A_TILE);
                    ::sycl::group_barrier(sg);
                    jm::joint_matrix_load(sg, a_hi[i], slot, TK);
                    if constexpr (Split)
                      jm::joint_matrix_load(sg, a_lo[i], slot + ACC_R * A_TILE, TK);
                  }
                }

#pragma unroll
                for (size_t j = 0; j < ACC_C; ++j) {
                  // Packed rows k0 / 2 onwards, columns q * 2 onwards - or, tiled, the
                  // tile at (step, q / TN), whose packed rows are 2 * TN apart.
                  const size_t b_off = TiledR
                    ? (step * p.qtiles + (q0 + j * TN) / TN) * (TK * TN)
                    : (k0 / 2) * p.ld_r + (q0 + j * TN) * 2;
                  const size_t b_ld = TiledR ? 2 * TN : p.ld_r;
                  jm::joint_matrix<::sycl::sub_group, Tm, jm::use::b, TK, TN,
                                   jm::layout::ext_intel_packed> b_hi, b_lo;
                  jm::joint_matrix_load(sg, b_hi, global(rh) + b_off, b_ld);
                  if constexpr (Split)
                    jm::joint_matrix_load(sg, b_lo, global(rl) + b_off, b_ld);
#pragma unroll
                  for (size_t i = 0; i < ACC_R; ++i) {
                    if constexpr (Split) {
                      // The small cross terms first, so they are not lost against the
                      // large one; lo*lo is below float32's resolution and is skipped.
                      jm::joint_matrix_mad(sg, acc[i][j], a_hi[i], b_lo, acc[i][j]);
                      jm::joint_matrix_mad(sg, acc[i][j], a_lo[i], b_hi, acc[i][j]);
                    }
                    jm::joint_matrix_mad(sg, acc[i][j], a_hi[i], b_hi, acc[i][j]);
                  }
                }
              }

              auto part = partial.template get_multi_ptr<::sycl::access::decorated::no>();

              if constexpr (SPLIT_K == 1) {
                // Each tile goes through the sub-group's own slot: stored as P[TM][TN], then
                // read back by the lanes along whichever P axis is contiguous in C.
                auto slot = part + sg_id * TM * TN;
                const size_t lane = sg.get_local_linear_id();
                T* cb = c + bi * p.stride_c;
#pragma unroll
                for (size_t i = 0; i < ACC_R; ++i)
#pragma unroll
                  for (size_t j = 0; j < ACC_C; ++j) {
                    jm::joint_matrix_store(sg, acc[i][j], slot, TN, jm::layout::row_major);
                    ::sycl::group_barrier(sg);
                    // The lanes walk C's contiguous axis together, so each store covers 8
                    // adjacent elements. Giving each lane its own line as an 8-wide vector
                    // instead was measured slower - twice as slow on attention scores.
#pragma unroll
                    for (size_t e = lane; e < TM * TN; e += SG_SIZE) {
                      const size_t rr = TransposeOut ? e % TM : e / TN;
                      const size_t qq = TransposeOut ? e / TM : e % TN;
                      const size_t pr = r0 + i * TM + rr;
                      const size_t pq = q0 + j * TN + qq;
                      if (pr < p.R && pq < p.Q) {
                        const size_t ci = TransposeOut ? pq * p.ld_c + pr : pr * p.ld_c + pq;
                        // beta == 0 must not read C: it is allowed to be uninitialised.
                        const float prev = (p.beta == 0.0f) ? 0.0f : static_cast<float>(cb[ci]);
                        cb[ci] = static_cast<T>(p.alpha * slot[rr * TN + qq] + p.beta * prev);
                      }
                    }
                    // The next tile overwrites the slot.
                    ::sycl::group_barrier(sg);
                  }
                return;
              }

#pragma unroll
              for (size_t i = 0; i < ACC_R; ++i)
#pragma unroll
                for (size_t j = 0; j < ACC_C; ++j)
                  jm::joint_matrix_store(sg, acc[i][j],
                                         part + sg_id * BLOCK + i * TM * COLS + j * TN,
                                         COLS, jm::layout::row_major);
              ::sycl::group_barrier(it.get_group());

              const size_t lid = it.get_local_id(0);
              for (size_t e = lid; e < SG_R * SG_Q * BLOCK; e += WG_SIZE) {
                const size_t block = e / BLOCK;  // which (sr, sq), as sr * SG_Q + sq
                const size_t within = e % BLOCK;
                // Walk along whichever P axis is contiguous in C, so the writes coalesce.
                const size_t rr = TransposeOut ? within % ROWS : within / COLS;
                const size_t qq = TransposeOut ? within / ROWS : within % COLS;
                const size_t pr = tile_r * WG_ROWS + (block / SG_Q) * ROWS + rr;
                const size_t pq = tile_q * WG_COLS + (block % SG_Q) * COLS + qq;
                if (pr >= p.R || pq >= p.Q)
                  continue;
                float sum = 0.0f;
#pragma unroll
                for (size_t s = 0; s < SPLIT_K; ++s)
                  sum += partial[(block * SPLIT_K + s) * BLOCK + rr * COLS + qq];
                T* cb = c + bi * p.stride_c;
                const size_t ci = TransposeOut ? pq * p.ld_c + pr : pr * p.ld_c + pq;
                // beta == 0 must not read C: it is allowed to be uninitialised.
                const float prev = (p.beta == 0.0f) ? 0.0f : static_cast<float>(cb[ci]);
                cb[ci] = static_cast<T>(p.alpha * sum + p.beta * prev);
              }
            });
        };
        SYCL_CHECK(get_queue().submit(submit));
      }

      // Tile shapes, chosen per problem. The skinny ones exist because one output tile per
      // sub-group leaves most of the card idle when P is thin - the decoder, where Q is
      // the beam, or attention's P * V, where R is one query - so the k steps are spread
      // over the sub-groups of a work-group instead.
      enum class Config {
        thin_q_deep,    // Q <= 8, long k
        thin_q,         // Q <= 8
        thin_q_short,   // Q <= 8, short k
        thin_r,         // R <= 8, long k
        thin_r_short,   // R <= 8, short k
        medium,         // too few output tiles for one per sub-group
        large,          // Rt tile-major, L read in place
        large_tiled,    // both tile-major, when each L tile is read often enough to pay
      };

      size_t wg_cols(Config config, bool split);

      Config choose(const Problem& p, bool split) {
        const size_t ksteps = (p.K + TK - 1) / TK;
        if (p.Q <= TN)
          return ksteps >= 32 ? Config::thin_q_deep
               : ksteps >= 8 ? Config::thin_q
               : Config::thin_q_short;
        if (p.R <= TM)
          return ksteps >= 16 ? Config::thin_r : Config::thin_r_short;
        // The large config has a sub-group per 16x32 block (16x16 when split). Below about
        // two thousand of them the card is under-occupied, and splitting k fills it.
        const size_t block_cols = split ? 16 : 32;
        const size_t blocks = p.batches * ((p.R + 15) / 16) * ((p.Q + block_cols - 1) / block_cols);
        if (blocks < 2048 && ksteps >= 16)
          return Config::medium;
        // Copying L costs a pass over it; it pays when its tiles are read by several
        // work-group columns. A Dense layer's weights are read by every one of them, an
        // attention matrix times V (Q = 64) by one.
        return p.Q >= 4 * wg_cols(Config::large, split) ? Config::large_tiled : Config::large;
      }

      // Rows of P one work-group of the large tiling covers, which its tile-major copy of
      // L is padded to. The other tilings read L in place.
      constexpr size_t LARGE_WG_ROWS = 4 * 2 * TM;

      // Columns of P one work-group covers, which Rt is padded to.
      size_t wg_cols(Config config, bool split) {
        switch (config) {
        case Config::thin_q_deep:
        case Config::thin_q:
        case Config::thin_q_short:
          return TN;
        case Config::thin_r:
          return 2 * 2 * TN;
        case Config::thin_r_short:
          return 8 * 2 * TN;
        case Config::medium:
          return 2 * 2 * TN;
        case Config::large:
        case Config::large_tiled:
          return 2 * (split ? 2 : 4) * TN;
        }
        return TN;
      }

      template <typename T, typename Tm, bool Split, bool TransposeOut>
      void dispatch(Config config, const Problem& p,
                    const Tm* l_hi, const Tm* l_lo, const Tm* r_hi, const Tm* r_lo, T* c) {
        //               ACC_R ACC_C SG_R SG_Q SPLIT_K
        switch (config) {
        case Config::thin_q_deep:
          run<T, Tm, Split, TransposeOut, false, false, 2, 1, 1, 1, 16>(p, l_hi, l_lo, r_hi, r_lo, c);
          break;
        case Config::thin_q:
          run<T, Tm, Split, TransposeOut, false, false, 2, 1, 4, 1, 4>(p, l_hi, l_lo, r_hi, r_lo, c);
          break;
        case Config::thin_q_short:
          run<T, Tm, Split, TransposeOut, false, false, 2, 1, 8, 1, 1>(p, l_hi, l_lo, r_hi, r_lo, c);
          break;
        case Config::thin_r:
          run<T, Tm, Split, TransposeOut, false, false, 1, 2, 1, 2, 8>(p, l_hi, l_lo, r_hi, r_lo, c);
          break;
        case Config::thin_r_short:
          run<T, Tm, Split, TransposeOut, false, false, 1, 2, 1, 8, 1>(p, l_hi, l_lo, r_hi, r_lo, c);
          break;
        case Config::medium:
          run<T, Tm, Split, TransposeOut, false, false, 2, 2, 2, 2, 4>(p, l_hi, l_lo, r_hi, r_lo, c);
          break;
        case Config::large:
          if constexpr (Split)  // twice the operand tiles in registers, so fewer accumulators
            run<T, Tm, Split, TransposeOut, false, true, 2, 2, 4, 2, 1>(p, l_hi, l_lo, r_hi, r_lo, c);
          else
            run<T, Tm, Split, TransposeOut, false, true, 2, 4, 4, 2, 1>(p, l_hi, l_lo, r_hi, r_lo, c);
          break;
        case Config::large_tiled:
          if constexpr (Split)
            run<T, Tm, Split, TransposeOut, true, true, 2, 2, 4, 2, 1>(p, l_hi, l_lo, r_hi, r_lo, c);
          else
            run<T, Tm, Split, TransposeOut, true, true, 2, 4, 4, 2, 1>(p, l_hi, l_lo, r_hi, r_lo, c);
          break;
        }
      }

      template <typename T, typename Tm, bool Split>
      void dispatch_problem(Config config, bool transposed, const Problem& p,
                            const Tm* l_hi, const Tm* l_lo, const Tm* r_hi, const Tm* r_lo,
                            T* c) {
        if (transposed)
          dispatch<T, Tm, Split, true>(config, p, l_hi, l_lo, r_hi, r_lo, c);
        else
          dispatch<T, Tm, Split, false>(config, p, l_hi, l_lo, r_hi, r_lo, c);
      }

      // Elements spanned by a strided batch of matrices, as the GEMM addresses them.
      size_t extent(size_t batches, size_t stride, size_t rows, size_t ld, size_t cols) {
        return (batches - 1) * stride + (rows - 1) * ld + cols;
      }

      // Owns a temporary buffer. Released as soon as the GEMM is enqueued, which the caching
      // allocator makes safe: the block is only reused by work queued after it on the same
      // in-order queue.
      template <typename Tm>
      class Temporary {
      public:
        explicit Temporary(size_t count)
          : _allocator(get_allocator<Device::XPU>())
          , _data(count > 0
                  ? static_cast<Tm*>(_allocator.allocate(count * sizeof (Tm)))
                  : nullptr) {
        }
        ~Temporary() {
          if (_data)
            _allocator.free(_data);
        }
        Temporary(const Temporary&) = delete;
        Temporary& operator=(const Temporary&) = delete;
        Tm* get() const { return _data; }
      private:
        Allocator& _allocator;
        Tm* _data;
      };

      // The Rt operand, repacked: for each batch, Rt[k][q] read from src at k * ld + q
      // (k_major) or q * ld + k, written VNNI-packed as [Kp / 2][Qp][2] and zero-padded to
      // Kp x Qp. With split, the bfloat16 low halves follow the high ones.
      //
      // Tiled, the same VNNI tiles are laid out one after another instead, each TK x TN
      // tile contiguous: [Kp / TK][Qp / TN][TK / 2][TN][2].
      template <typename Tm>
      class PackedRt {
      public:
        template <typename Ts>
        PackedRt(const Ts* src, size_t batches, size_t stride, size_t ld, bool k_major,
                 size_t K, size_t Q, size_t Kp, size_t Qp, bool split, bool tiled = false)
          : _count(batches * Kp * Qp)
          , _ld(tiled ? 2 * TN : Qp * 2)
          , _stride(Kp * Qp)
          , _split(split)
          , _buffer(_count * (split ? 2 : 1)) {
          if (_count == 0)
            return;
          Tm* hi = _buffer.get();
          Tm* lo = split ? hi + _count : nullptr;
          const size_t per_batch = Kp * Qp;
          const size_t row = Qp * 2;
          const size_t qtiles = Qp / TN;
          SYCL_CHECK(get_queue().parallel_for(
                       ::sycl::range<1>(_count),
                       [=](::sycl::id<1> id) {
                         const size_t i = id[0];
                         const size_t b = i / per_batch;
                         const size_t w = i % per_batch;
                         size_t q, k;
                         if (tiled) {
                           const size_t tile = w / (TK * TN);
                           const size_t within = w % (TK * TN);
                           q = (tile % qtiles) * TN + (within % (2 * TN)) / 2;
                           k = (tile / qtiles) * TK + (within / (2 * TN)) * 2 + within % 2;
                         } else {
                           q = (w % row) / 2;
                           k = (w / row) * 2 + w % 2;
                         }
                         float v = 0.0f;
                         if (k < K && q < Q)
                           v = static_cast<float>(src[b * stride + (k_major ? k * ld + q
                                                                            : q * ld + k)]);
                         const Tm high(v);
                         hi[i] = high;
                         if (lo)
                           lo[i] = Tm(v - static_cast<float>(high));
                       }));
        }
        const Tm* hi() const { return _buffer.get(); }
        const Tm* lo() const { return _split ? _buffer.get() + _count : nullptr; }
        size_t ld() const { return _ld; }
        size_t stride() const { return _stride; }
      private:
        size_t _count;
        size_t _ld;
        size_t _stride;
        bool _split;
        Temporary<Tm> _buffer;
      };

      // The L operand, when it cannot be read in place: a float32 one has to become
      // bfloat16 halves (split), and a transposed A has to become row-major (transpose,
      // packing each [rows][cols] source matrix into a [cols][rows] one). Untransposed,
      // the copy keeps the source's ld and stride.
      template <typename Tm>
      class StagedL {
      public:
        template <typename Ts>
        StagedL(const Ts* src, size_t batches, size_t stride,
                size_t rows, size_t cols, size_t ld, bool transpose, bool split)
          : _count(transpose ? batches * rows * cols : extent(batches, stride, rows, ld, cols))
          , _ld(transpose ? rows : ld)
          , _stride(transpose ? rows * cols : stride)
          , _split(split)
          , _buffer(_count * (split ? 2 : 1)) {
          Tm* hi = _buffer.get();
          Tm* lo = split ? hi + _count : nullptr;
          SYCL_CHECK(get_queue().parallel_for(
                       ::sycl::range<1>(_count),
                       [=](::sycl::id<1> id) {
                         const size_t i = id[0];
                         size_t from = i;
                         if (transpose) {
                           // i walks the destination, [batch][col][row], contiguously.
                           const size_t row = i % rows;
                           const size_t rest = i / rows;
                           from = (rest / cols) * stride + row * ld + rest % cols;
                         }
                         const float v = static_cast<float>(src[from]);
                         const Tm high(v);
                         hi[i] = high;
                         if (lo)
                           lo[i] = Tm(v - static_cast<float>(high));
                       }));
        }
        const Tm* hi() const { return _buffer.get(); }
        const Tm* lo() const { return _split ? _buffer.get() + _count : nullptr; }
        size_t ld() const { return _ld; }
        size_t stride() const { return _stride; }
      private:
        size_t _count;
        size_t _ld;
        size_t _stride;
        bool _split;
        Temporary<Tm> _buffer;
      };

      // The L operand copied tile-major for the large tiling: [Rp / TM][Kp / TK][TM][TK]
      // per batch, zero-padded to Rp x Kp, so every A tile the kernel loads is contiguous
      // and in range. L[r][k] is read from src at r * ld + k, or k * ld + r when transposed.
      template <typename Tm>
      class TiledL {
      public:
        // With src_lo, L arrives already split (Ts is Tm): src holds the high halves and
        // src_lo the low ones, and both are copied as they are.
        template <typename Ts>
        TiledL(const Ts* src, size_t batches, size_t stride, size_t ld, bool transpose,
               size_t R, size_t K, size_t Rp, size_t Kp, bool split,
               const Ts* src_lo = nullptr)
          : _count(batches * Rp * Kp)
          , _split(split)
          , _buffer(_count * (split ? 2 : 1)) {
          if (_count == 0)
            return;
          Tm* hi = _buffer.get();
          Tm* lo = split ? hi + _count : nullptr;
          const size_t per_batch = Rp * Kp;
          const size_t ktiles = Kp / TK;
          SYCL_CHECK(get_queue().parallel_for(
                       ::sycl::range<1>(_count),
                       [=](::sycl::id<1> id) {
                         const size_t i = id[0];
                         const size_t b = i / per_batch;
                         const size_t w = i % per_batch;
                         const size_t tile = w / (TM * TK);
                         const size_t within = w % (TM * TK);
                         const size_t r = (tile / ktiles) * TM + within / TK;
                         const size_t k = (tile % ktiles) * TK + within % TK;
                         const bool inside = r < R && k < K;
                         const size_t from = b * stride + (transpose ? k * ld + r : r * ld + k);
                         if constexpr (std::is_same_v<Ts, Tm>) {
                           if (src_lo) {
                             hi[i] = inside ? src[from] : Tm(0.0f);
                             lo[i] = inside ? src_lo[from] : Tm(0.0f);
                             return;
                           }
                         }
                         const float v = inside ? static_cast<float>(src[from]) : 0.0f;
                         const Tm high(v);
                         hi[i] = high;
                         if (lo)
                           lo[i] = Tm(v - static_cast<float>(high));
                       }));
        }
        const Tm* hi() const { return _buffer.get(); }
        const Tm* lo() const { return _split ? _buffer.get() + _count : nullptr; }
      private:
        size_t _count;
        bool _split;
        Temporary<Tm> _buffer;
      };

      // C = op(A) * op(B), row-major, mapped onto P = L * Rt. Ts is the type A and B are
      // stored in, T the type of C.
      template <typename T, typename Ts, typename Tm, bool Split>
      void gemm(bool transpose_a, bool transpose_b,
                dim_t m, dim_t n, dim_t k,
                float alpha,
                const Ts* a, dim_t lda, dim_t stridea,
                const Ts* b, dim_t ldb, dim_t strideb,
                float beta,
                T* c, dim_t ldc, dim_t stridec,
                dim_t batch_size,
                const Tm* pre_hi = nullptr, const Tm* pre_lo = nullptr) {
        // pre_hi, pre_lo: B already split into its bfloat16 halves (see xmx_split_float32),
        // laid out like B itself. Only used with B transposed, where B is L.
        const size_t M = static_cast<size_t>(m);
        const size_t N = static_cast<size_t>(n);
        Problem p;
        p.K = static_cast<size_t>(std::max<dim_t>(k, 0));
        p.ld_c = static_cast<size_t>(ldc);
        p.stride_c = static_cast<size_t>(stridec);
        p.batches = static_cast<size_t>(batch_size);
        p.alpha = alpha;
        p.beta = beta;

        // B transposed - every Dense layer, and attention's Q * K^T - computes P = C^T with
        // L = B as stored, [n][k], and Rt = A^T. Otherwise P = C, L = op(A), Rt = B.
        const bool transposed = transpose_b;
        p.R = transposed ? N : M;
        p.Q = transposed ? M : N;

        const Ts* l = transposed ? b : a;
        size_t ld_l = static_cast<size_t>(transposed ? ldb : lda);
        size_t stride_l = static_cast<size_t>(transposed ? strideb : stridea);
        // Only A can arrive column-major in the L position.
        const bool restage_transpose = !transposed && transpose_a;

        const Config config = choose(p, Split);
        const size_t Kp = round_up(p.K, TK);
        const size_t Qp = round_up(p.Q, wg_cols(config, Split));
        p.ktiles = Kp / TK;
        p.qtiles = Qp / TN;

        if (config == Config::large_tiled) {
          // Both operands copied tile-major, so that every load is a whole request (see
          // TiledL in run()). The copies cost one pass over each operand, a few percent of
          // a GEMM this size.
          const size_t Rp = round_up(p.R, LARGE_WG_ROWS);
          const PackedRt<Tm> rt = transposed
            ? PackedRt<Tm>(a, p.batches, static_cast<size_t>(stridea), static_cast<size_t>(lda),
                           transpose_a, p.K, p.Q, Kp, Qp, Split, true)
            : PackedRt<Tm>(b, p.batches, static_cast<size_t>(strideb), static_cast<size_t>(ldb),
                           true, p.K, p.Q, Kp, Qp, Split, true);
          const TiledL<Tm> tl = pre_hi
            ? TiledL<Tm>(pre_hi, p.batches, stride_l, ld_l, false, p.R, p.K, Rp, Kp, Split, pre_lo)
            : TiledL<Tm>(l, p.batches, stride_l, ld_l, restage_transpose, p.R, p.K, Rp, Kp, Split);
          p.ld_r = rt.ld();
          p.stride_r = rt.stride();
          p.ld_l = TK;
          p.stride_l = Rp * Kp;
          dispatch_problem<T, Tm, Split>(config, transposed, p, tl.hi(), tl.lo(), rt.hi(), rt.lo(), c);
          return;
        }

        // Rt, packed. Rt[k][q] is A[q][k] (or A[k][q] when A is transposed) for P = C^T,
        // and B[k][q] for P = C.
        const bool tiled_r = config == Config::large;
        const PackedRt<Tm> rt = transposed
          ? PackedRt<Tm>(a, p.batches, static_cast<size_t>(stridea), static_cast<size_t>(lda),
                         transpose_a, p.K, p.Q, Kp, Qp, Split, tiled_r)
          : PackedRt<Tm>(b, p.batches, static_cast<size_t>(strideb), static_cast<size_t>(ldb),
                         true, p.K, p.Q, Kp, Qp, Split, tiled_r);
        p.ld_r = rt.ld();
        p.stride_r = rt.stride();

        // Also when L is stored in another type than the engines take, split or not.
        const bool stage_l = Split || restage_transpose || !std::is_same_v<Ts, Tm>;
        if (pre_hi) {
          p.ld_l = ld_l;
          p.stride_l = stride_l;
          dispatch_problem<T, Tm, Split>(config, transposed, p, pre_hi, pre_lo, rt.hi(), rt.lo(), c);
        } else if (stage_l && p.K > 0) {
          const size_t rows = restage_transpose ? p.K : p.R;
          const size_t cols = restage_transpose ? p.R : p.K;
          const StagedL<Tm> sl(l, p.batches, stride_l, rows, cols, ld_l,
                               restage_transpose, Split);
          p.ld_l = sl.ld();
          p.stride_l = sl.stride();
          dispatch_problem<T, Tm, Split>(config, transposed, p, sl.hi(), sl.lo(), rt.hi(), rt.lo(), c);
        } else {
          p.ld_l = ld_l;
          p.stride_l = stride_l;
          // Only reached without a split, so Ts and Tm are the same 16-bit type, or K is
          // zero and nothing is loaded at all.
          const Tm* lh = reinterpret_cast<const Tm*>(l);
          dispatch_problem<T, Tm, Split>(config, transposed, p, lh, nullptr, rt.hi(), rt.lo(), c);
        }
      }

    }

    void xmx_gemm(bool transpose_a, bool transpose_b,
                  dim_t m, dim_t n, dim_t k,
                  float alpha,
                  const float16_t* a, dim_t lda, dim_t stridea,
                  const float16_t* b, dim_t ldb, dim_t strideb,
                  float beta,
                  float16_t* c, dim_t ldc, dim_t stridec,
                  dim_t batch_size) {
      if (m <= 0 || n <= 0 || batch_size <= 0)
        return;
      check_device_has_xmx();
      // float16_t and sycl::half are both IEEE binary16; see DeviceType in helpers.h.
      gemm<half, half, half, false>(transpose_a, transpose_b, m, n, k, alpha,
                                    reinterpret_cast<const half*>(a), lda, stridea,
                                    reinterpret_cast<const half*>(b), ldb, strideb,
                                    beta, reinterpret_cast<half*>(c), ldc, stridec,
                                    batch_size);
    }

    void xmx_split_float32(const float* src, float* dst, dim_t count) {
      if (count <= 0)
        return;
      if (static_cast<const void*>(src) == static_cast<const void*>(dst))
        throw std::invalid_argument("xmx_split_float32 cannot split in place");
      auto* hi = reinterpret_cast<bf16*>(dst);
      auto* lo = hi + count;
      SYCL_CHECK(get_queue().parallel_for(
                   ::sycl::range<1>(static_cast<size_t>(count)),
                   [=](::sycl::id<1> id) {
                     const size_t i = id[0];
                     const float v = src[i];
                     const bf16 high(v);
                     hi[i] = high;
                     lo[i] = bf16(v - static_cast<float>(high));
                   }));
    }

    void xmx_gemm_packed_b(bool transpose_a, bool transpose_b,
                           dim_t m, dim_t n, dim_t k,
                           float alpha,
                           const float* a, dim_t lda,
                           const float* b, dim_t ldb,
                           float beta,
                           float* c, dim_t ldc) {
      if (m <= 0 || n <= 0)
        return;
      if (!transpose_b)
        throw std::invalid_argument("XMX GEMM: a packed B must be transposed, as a Dense "
                                    "layer's weight is");
      check_device_has_xmx();
      const auto* hi = reinterpret_cast<const bf16*>(b);
      const auto* lo = hi + n * ldb;
      gemm<float, float, bf16, true>(transpose_a, transpose_b, m, n, k, alpha,
                                     a, lda, 0, b, ldb, 0,
                                     beta, c, ldc, 0, 1, hi, lo);
    }

    void xmx_gemm(bool transpose_a, bool transpose_b,
                  dim_t m, dim_t n, dim_t k,
                  float alpha,
                  const float* a, dim_t lda, dim_t stridea,
                  const float* b, dim_t ldb, dim_t strideb,
                  float beta,
                  float* c, dim_t ldc, dim_t stridec,
                  dim_t batch_size) {
      if (m <= 0 || n <= 0 || batch_size <= 0)
        return;
      check_device_has_xmx();
      gemm<float, float, bf16, true>(transpose_a, transpose_b, m, n, k, alpha,
                                     a, lda, stridea, b, ldb, strideb,
                                     beta, c, ldc, stridec, batch_size);
    }

  }
}
