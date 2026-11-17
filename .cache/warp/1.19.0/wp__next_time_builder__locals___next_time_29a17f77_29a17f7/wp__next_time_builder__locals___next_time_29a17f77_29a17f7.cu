#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"
#include "deterministic.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()

// CUDA Thread Block Cluster shape declaration. Expands to __cluster_dims__
// only on devices that support clusters (compute capability 9.0+); otherwise
// expands to nothing so the same source compiles cleanly for any target arch.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define WP_CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define WP_CLUSTER_DIMS(x, y, z)
#endif

// Maximum registers per thread. __maxnreg__ was added in CUDA Toolkit 12.4;
// older toolkits ignore the opt-in so the same source remains compilable.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4))
#define WP_MAXNREG(n) __maxnreg__(n)
#else
#define WP_MAXNREG(n)
#endif

// Allow PTXAS to spill registers into shared memory. CUDA Toolkit 13.0
// introduced the pragma, which is unavailable in device-debug compilation.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13) && !defined(_DEBUG)
#define WP_ENABLE_SMEM_SPILLING() asm volatile(".pragma \"enable_smem_spilling\";");
#else
#define WP_ENABLE_SMEM_SPILLING()
#endif


// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)



extern "C" __global__ void _next_time_builder__locals___next_time_7a0e2f92_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_time_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::int32 var_nworld_in,
    wp::int32 var_naconmax_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_ncollision_in,
    wp::array_t<wp::float32> var_time_out,
    wp::array_t<wp::int32> var_overflow_out)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::float32* var_1;
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        bool var_14;
        const bool var_15 = false;
        wp::int32* var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        wp::int32 var_19;
        bool var_20;
        const wp::int32 var_21 = 0;
        bool var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        bool var_31;
        const bool var_32 = false;
        wp::int32* var_33;
        const wp::int32 var_34 = 2;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        bool var_41;
        bool var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        const bool var_45 = false;
        const wp::str var_46 = "broadphase overflow - please increase nconmax beyond %u or naconmax beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.BROADPHASE (or = 0 for all)\n";
        wp::int32 var_47;
        wp::int32* var_48;
        const wp::int32 var_49 = 4;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        wp::int32* var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        bool var_56;
        bool var_57;
        const wp::int32 var_58 = 0;
        bool var_59;
        const bool var_60 = false;
        const wp::str var_61 = "narrowphase overflow - please increase nconmax beyond %u or naconmax beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NARROWPHASE (or = 0 for all)\n";
        wp::int32 var_62;
        wp::int32* var_63;
        const wp::int32 var_64 = 8;
        wp::int32 var_65;
        wp::int32 var_66;
        //---------
        // forward
        // def _next_time(                                                                        <L 326>
        // worldid = wp.tid()                                                                     <L 345>
        var_0 = builtin_tid1d();
        // time_out[worldid] = time_in[worldid] + opt_timestep[worldid % opt_timestep.shape[0]]       <L 346>
        var_1 = wp::address(var_time_in, var_0);
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_1);
        var_10 = wp::load(var_7);
        var_8 = wp::add(var_9, var_10);
        wp::array_store(var_time_out, var_0, var_8);
        // nefc = nefc_in[worldid]                                                                <L 347>
        var_11 = wp::address(var_nefc_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // if nefc > njmax_in:                                                                    <L 349>
        var_14 = (var_12 > var_njmax_in);
        if (var_14) {
            // if wp.static(bool(warn_overflow & OverflowType.NEFC)):                             <L 350>
            // overflow_out[worldid] = overflow_out[worldid] | OverflowType.NEFC                  <L 356>
            var_16 = wp::address(var_overflow_out, var_0);
            var_19 = wp::load(var_16);
            var_18 = wp::bit_or(var_19, var_17);
            wp::array_store(var_overflow_out, var_0, var_18);
        }
        if (!var_14) {
            // elif nefc > 0 and is_sparse:                                                       <L 357>
            var_22 = (var_12 > var_21);
            var_20 = var_22;
            if (var_20) {
                var_20 = var_20 && var_is_sparse;
            }
            if (var_20) {
                // efcid = wp.min(nefc, njmax_in) - 1                                             <L 358>
                var_23 = wp::min(var_12, var_njmax_in);
                var_25 = wp::sub(var_23, var_24);
                // efc_nnz = efc_J_rowadr_in[worldid, efcid] + efc_J_rownnz_in[worldid, efcid]       <L 359>
                var_26 = wp::address(var_efc_J_rowadr_in, var_0, var_25);
                var_27 = wp::address(var_efc_J_rownnz_in, var_0, var_25);
                var_29 = wp::load(var_26);
                var_30 = wp::load(var_27);
                var_28 = wp::add(var_29, var_30);
                // if efc_nnz > njmax_nnz_in:                                                     <L 360>
                var_31 = (var_28 > var_njmax_nnz_in);
                if (var_31) {
                    // if wp.static(bool(warn_overflow & OverflowType.NJMAX_NNZ)):                <L 361>
                    // overflow_out[worldid] = overflow_out[worldid] | OverflowType.NJMAX_NNZ       <L 367>
                    var_33 = wp::address(var_overflow_out, var_0);
                    var_36 = wp::load(var_33);
                    var_35 = wp::bit_or(var_36, var_34);
                    wp::array_store(var_overflow_out, var_0, var_35);
                }
            }
        }
        // ncollision = ncollision_in[0]                                                          <L 369>
        var_38 = wp::address(var_ncollision_in, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // if ncollision > naconmax_in:                                                           <L 370>
        var_41 = (var_39 > var_naconmax_in);
        if (var_41) {
            // if worldid == 0 and wp.static(bool(warn_overflow & OverflowType.BROADPHASE)):       <L 371>
            var_44 = (var_0 == var_43);
            var_42 = var_44;
            if (var_42) {
                var_42 = var_42 && var_45;
            }
            if (var_42) {
                // wp.printf(                                                                     <L 372>
                // "broadphase overflow - please increase nconmax beyond %u or naconmax beyond %u\n"       <L 373>
                // naconmax_in // nworld_in,                                                      <L 375>
                var_47 = wp::floordiv(var_naconmax_in, var_nworld_in);
                // naconmax_in,                                                                   <L 376>
                printf(var_46, var_47, var_naconmax_in);
            }
            // overflow_out[worldid] = overflow_out[worldid] | OverflowType.BROADPHASE            <L 378>
            var_48 = wp::address(var_overflow_out, var_0);
            var_51 = wp::load(var_48);
            var_50 = wp::bit_or(var_51, var_49);
            wp::array_store(var_overflow_out, var_0, var_50);
        }
        // nacon = nacon_in[0]                                                                    <L 380>
        var_53 = wp::address(var_nacon_in, var_52);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // if nacon > naconmax_in:                                                                <L 381>
        var_56 = (var_54 > var_naconmax_in);
        if (var_56) {
            // if worldid == 0 and wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)):       <L 382>
            var_59 = (var_0 == var_58);
            var_57 = var_59;
            if (var_57) {
                var_57 = var_57 && var_60;
            }
            if (var_57) {
                // wp.printf(                                                                     <L 383>
                // "narrowphase overflow - please increase nconmax beyond %u or naconmax beyond %u\n"       <L 384>
                // naconmax_in // nworld_in,                                                      <L 386>
                var_62 = wp::floordiv(var_naconmax_in, var_nworld_in);
                // naconmax_in,                                                                   <L 387>
                printf(var_61, var_62, var_naconmax_in);
            }
            // overflow_out[worldid] = overflow_out[worldid] | OverflowType.NARROWPHASE           <L 389>
            var_63 = wp::address(var_overflow_out, var_0);
            var_66 = wp::load(var_63);
            var_65 = wp::bit_or(var_66, var_64);
            wp::array_store(var_overflow_out, var_0, var_65);
        }
    }
}

