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



extern "C" __global__ void _next_time_builder__locals___next_time_6bcfac29_cuda_kernel_forward(
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
        const bool var_15 = true;
        const wp::str var_16 = "nefc overflow - please increase njmax beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NEFC (or = 0 for all)\n";
        wp::int32* var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::int32 var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        bool var_32;
        const bool var_33 = true;
        const wp::str var_34 = "njmax_nnz overflow - please increase njmax_nnz beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NJMAX_NNZ (or = 0 for all)\n";
        wp::int32* var_35;
        const wp::int32 var_36 = 2;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        bool var_43;
        bool var_44;
        const wp::int32 var_45 = 0;
        bool var_46;
        const bool var_47 = true;
        const wp::str var_48 = "broadphase overflow - please increase nconmax beyond %u or naconmax beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.BROADPHASE (or = 0 for all)\n";
        wp::int32 var_49;
        wp::int32* var_50;
        const wp::int32 var_51 = 4;
        wp::int32 var_52;
        wp::int32 var_53;
        const wp::int32 var_54 = 0;
        wp::int32* var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        bool var_58;
        bool var_59;
        const wp::int32 var_60 = 0;
        bool var_61;
        const bool var_62 = true;
        const wp::str var_63 = "narrowphase overflow - please increase nconmax beyond %u or naconmax beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NARROWPHASE (or = 0 for all)\n";
        wp::int32 var_64;
        wp::int32* var_65;
        const wp::int32 var_66 = 8;
        wp::int32 var_67;
        wp::int32 var_68;
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
            // wp.printf(                                                                         <L 351>
            // "nefc overflow - please increase njmax beyond %u\n"                                <L 352>
            // njmax_in,                                                                          <L 354>
            printf(var_16, var_njmax_in);
            // overflow_out[worldid] = overflow_out[worldid] | OverflowType.NEFC                  <L 356>
            var_17 = wp::address(var_overflow_out, var_0);
            var_20 = wp::load(var_17);
            var_19 = wp::bit_or(var_20, var_18);
            wp::array_store(var_overflow_out, var_0, var_19);
        }
        if (!var_14) {
            // elif nefc > 0 and is_sparse:                                                       <L 357>
            var_23 = (var_12 > var_22);
            var_21 = var_23;
            if (var_21) {
                var_21 = var_21 && var_is_sparse;
            }
            if (var_21) {
                // efcid = wp.min(nefc, njmax_in) - 1                                             <L 358>
                var_24 = wp::min(var_12, var_njmax_in);
                var_26 = wp::sub(var_24, var_25);
                // efc_nnz = efc_J_rowadr_in[worldid, efcid] + efc_J_rownnz_in[worldid, efcid]       <L 359>
                var_27 = wp::address(var_efc_J_rowadr_in, var_0, var_26);
                var_28 = wp::address(var_efc_J_rownnz_in, var_0, var_26);
                var_30 = wp::load(var_27);
                var_31 = wp::load(var_28);
                var_29 = wp::add(var_30, var_31);
                // if efc_nnz > njmax_nnz_in:                                                     <L 360>
                var_32 = (var_29 > var_njmax_nnz_in);
                if (var_32) {
                    // if wp.static(bool(warn_overflow & OverflowType.NJMAX_NNZ)):                <L 361>
                    // wp.printf(                                                                 <L 362>
                    // "njmax_nnz overflow - please increase njmax_nnz beyond %u\n"               <L 363>
                    // njmax_nnz_in,                                                              <L 365>
                    printf(var_34, var_njmax_nnz_in);
                    // overflow_out[worldid] = overflow_out[worldid] | OverflowType.NJMAX_NNZ       <L 367>
                    var_35 = wp::address(var_overflow_out, var_0);
                    var_38 = wp::load(var_35);
                    var_37 = wp::bit_or(var_38, var_36);
                    wp::array_store(var_overflow_out, var_0, var_37);
                }
            }
        }
        // ncollision = ncollision_in[0]                                                          <L 369>
        var_40 = wp::address(var_ncollision_in, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // if ncollision > naconmax_in:                                                           <L 370>
        var_43 = (var_41 > var_naconmax_in);
        if (var_43) {
            // if worldid == 0 and wp.static(bool(warn_overflow & OverflowType.BROADPHASE)):       <L 371>
            var_46 = (var_0 == var_45);
            var_44 = var_46;
            if (var_44) {
                var_44 = var_44 && var_47;
            }
            if (var_44) {
                // wp.printf(                                                                     <L 372>
                // "broadphase overflow - please increase nconmax beyond %u or naconmax beyond %u\n"       <L 373>
                // naconmax_in // nworld_in,                                                      <L 375>
                var_49 = wp::floordiv(var_naconmax_in, var_nworld_in);
                // naconmax_in,                                                                   <L 376>
                printf(var_48, var_49, var_naconmax_in);
            }
            // overflow_out[worldid] = overflow_out[worldid] | OverflowType.BROADPHASE            <L 378>
            var_50 = wp::address(var_overflow_out, var_0);
            var_53 = wp::load(var_50);
            var_52 = wp::bit_or(var_53, var_51);
            wp::array_store(var_overflow_out, var_0, var_52);
        }
        // nacon = nacon_in[0]                                                                    <L 380>
        var_55 = wp::address(var_nacon_in, var_54);
        var_57 = wp::load(var_55);
        var_56 = wp::copy(var_57);
        // if nacon > naconmax_in:                                                                <L 381>
        var_58 = (var_56 > var_naconmax_in);
        if (var_58) {
            // if worldid == 0 and wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)):       <L 382>
            var_61 = (var_0 == var_60);
            var_59 = var_61;
            if (var_59) {
                var_59 = var_59 && var_62;
            }
            if (var_59) {
                // wp.printf(                                                                     <L 383>
                // "narrowphase overflow - please increase nconmax beyond %u or naconmax beyond %u\n"       <L 384>
                // naconmax_in // nworld_in,                                                      <L 386>
                var_64 = wp::floordiv(var_naconmax_in, var_nworld_in);
                // naconmax_in,                                                                   <L 387>
                printf(var_63, var_64, var_naconmax_in);
            }
            // overflow_out[worldid] = overflow_out[worldid] | OverflowType.NARROWPHASE           <L 389>
            var_65 = wp::address(var_overflow_out, var_0);
            var_68 = wp::load(var_65);
            var_67 = wp::bit_or(var_68, var_66);
            wp::array_store(var_overflow_out, var_0, var_67);
        }
    }
}

