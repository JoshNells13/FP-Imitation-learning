#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 128
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:2513
static CUDA_CALLABLE wp::float32 _state_check_0(
    wp::float32 var_D,
    wp::int32 var_state)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    //---------
    // forward
    // def _state_check(D: float, state: int) -> float:                                       <L 2514>
    // if state == types.ConstraintState.QUADRATIC.value:                                     <L 2515>
    var_1 = (var_state == var_0);
    if (var_1) {
        // return D                                                                           <L 2516>
        return var_D;
    }
    if (!var_1) {
        // return 0.0                                                                         <L 2518>
        return var_2;
    }
    return {};
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:2521
static CUDA_CALLABLE wp::float32 _active_check_0(
    wp::int32 var_tid,
    wp::int32 var_threshold)
{
    //---------
    // primal vars
    bool var_0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 1.0;
    //---------
    // forward
    // def _active_check(tid: int, threshold: int) -> float:                                  <L 2522>
    // if tid >= threshold:                                                                   <L 2523>
    var_0 = (var_tid >= var_threshold);
    if (var_0) {
        // return 0.0                                                                         <L 2524>
        return var_1;
    }
    if (!var_0) {
        // return 1.0                                                                         <L 2526>
        return var_2;
    }
    return {};
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:2513
static CUDA_CALLABLE void adj__state_check_0(
    wp::float32 var_D,
    wp::int32 var_state,
    wp::float32 & adj_D,
    wp::int32 & adj_state,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:2521
static CUDA_CALLABLE void adj__active_check_0(
    wp::int32 var_tid,
    wp::int32 var_threshold,
    wp::int32 & adj_tid,
    wp::int32 & adj_threshold,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _update_gradient_JTDAJ_dense_tiled__locals__kernel_f17f0e8d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_colind,
    wp::array_t<wp::int32> var_M_hinit_i,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::int32> var_efc_state_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_h_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 784;
        wp::tuple_t<wp::int32> var_9;
        const wp::str var_10 = "shared";
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<784>, wp::tile_stride_t<1>>, true> var_11 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<784>,wp::tile_stride_t<1>,false>();
        wp::int32 var_12;
        const wp::int32 var_13 = 218;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::range_t var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        bool var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::int32* var_25;
        const wp::int32 var_26 = 28;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::float32* var_32;
        const wp::float32 var_33 = 0.0;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::tuple_t<wp::int32, wp::int32> var_36;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<28,28>, wp::tile_stride_t<28,1>>, false> var_37 = nullptr;
        const wp::int32 var_38 = 0;
        const wp::int32 var_39 = 64;
        const wp::int32 var_40 = 16;
        wp::range_t var_41;
        wp::int32 var_42;
        bool var_43;
        wp::array_t<wp::float32> var_44;
        const wp::int32 var_45 = 16;
        wp::tuple_t<wp::int32, wp::int32> var_46;
        const wp::int32 var_47 = 0;
        wp::tuple_t<wp::int32, wp::int32> var_48;
        const bool var_49 = false;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16,28>, wp::tile_stride_t<28,1>>, true> var_50 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16,28>,wp::tile_stride_t<28,1>,false>();
        const wp::int32 var_51 = 0;
        wp::array_t<wp::float32> var_52;
        const bool var_53 = false;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_54 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::array_t<wp::int32> var_55;
        const bool var_56 = false;
        wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_57 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_58 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_59 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        const wp::int32 var_60 = 0;
        const wp::int32 var_61 = 1;
        wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_62 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::int32 var_63;
        wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_64 = wp::tile_register_t<wp::int32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>> var_65 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<16>>>{};
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<16>, wp::tile_stride_t<1>>, true> var_66 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<16>,wp::tile_stride_t<1>,false>();
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<28,16>, wp::tile_stride_t<1,28>>, false> var_67 = nullptr;
        wp::tuple_t<wp::int32, wp::int32> var_68;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<28,16>, wp::tile_stride_t<0,1>>, false> var_69 = nullptr;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<28,16>, wp::tile_stride_t<16,1>>, true> var_70 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<28,16>,wp::tile_stride_t<16,1>,false>();
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<28,28>, wp::tile_stride_t<28,1>>, true> var_71 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<28,28>,wp::tile_stride_t<28,1>,false>();
        const wp::int32 var_72 = 0;
        const wp::int32 var_73 = 0;
        const wp::int32 var_74 = 0;
        const wp::float32 var_75 = 1.0;
        const wp::float32 var_76 = 0.0;
        wp::array_t<wp::float32> var_77;
        const bool var_78 = false;
        const wp::int32 var_79 = 0;
        const wp::int32 var_80 = 0;
        //---------
        // forward
        // def kernel(                                                                            <L 2600>
        // worldid, rank = wp.tid()                                                               <L 2615>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 2617>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2618>
            return;
        }
        var_4 = wp::load(var_2);
        // nefc = nefc_in[worldid]                                                                <L 2620>
        var_5 = wp::address(var_nefc_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // m_tile = wp.tile_zeros(shape=(wp.static(nv_pad * nv_pad),), dtype=float, storage="shared")       <L 2624>
        var_9 = wp::tuple(var_8);
        var_11 = wp::tile_zeros<float, 784>();
        // lanes = wp.block_dim()                                                                 <L 2628>
        var_12 = builtin_block_dim();
        // iters = (nC + lanes - 1) // lanes                                                      <L 2629>
        var_14 = wp::add(var_13, var_12);
        var_16 = wp::sub(var_14, var_15);
        var_17 = wp::floordiv(var_16, var_12);
        // for it in range(iters):                                                                <L 2630>
        var_18 = wp::range(var_17);
        start_for_1:;
            if (iter_cmp(var_18) == 0) goto end_for_1;
            var_19 = wp::iter_next(var_18);
            // e = it * lanes + rank                                                              <L 2631>
            var_20 = wp::mul(var_19, var_12);
            var_21 = wp::add(var_20, var_1);
            // enable = e < nC                                                                    <L 2632>
            var_22 = (var_21 < var_13);
            // ec = wp.where(enable, e, 0)                                                        <L 2633>
            var_24 = wp::where(var_22, var_21, var_23);
            // pos = M_colind[ec] * nv_pad + M_hinit_i[ec]                                        <L 2634>
            var_25 = wp::address(var_M_colind, var_24);
            var_28 = wp::load(var_25);
            var_27 = wp::mul(var_28, var_26);
            var_29 = wp::address(var_M_hinit_i, var_24);
            var_31 = wp::load(var_29);
            var_30 = wp::add(var_27, var_31);
            // wp.tile_scatter_add(m_tile, pos, wp.where(enable, M_in[worldid, ec], 0.0), enable)       <L 2635>
            var_32 = wp::address(var_M_in, var_0, var_24);
            var_35 = wp::load(var_32);
            var_34 = wp::where(var_22, var_35, var_33);
            wp::tile_scatter_add(var_11, var_30, var_34, var_22);
            goto start_for_1;
        end_for_1:;
        // sum_val = wp.tile_reshape(m_tile, (nv_pad, nv_pad))                                    <L 2636>
        var_36 = wp::tuple(var_26, var_26);
        var_37 = wp::tile_reshape<wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<28,28>, wp::tile_stride_t<28,1>>, false>>(var_11);
        // for k in range(0, njmax, TILE_SIZE_K):                                                 <L 2639>
        var_41 = wp::range(var_38, var_39, var_40);
        start_for_3:;
            if (iter_cmp(var_41) == 0) goto end_for_3;
            var_42 = wp::iter_next(var_41);
            // if k >= nefc:                                                                      <L 2640>
            var_43 = (var_42 >= var_6);
            if (var_43) {
                // break                                                                          <L 2641>
                goto end_for_3;
            }
            // J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE_K, nv_pad), offset=(k, 0), bounds_check=False)       <L 2646>
            var_44 = wp::view(var_efc_J_in, var_0);
            var_46 = wp::tuple(var_45, var_26);
            var_48 = wp::tuple(var_42, var_47);
            var_50 = wp::tile_load<wp::float32, false, false, 16, 28>(var_44, var_42, var_51);
            // D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)       <L 2649>
            var_52 = wp::view(var_efc_D_in, var_0);
            var_54 = wp::tile_load<wp::float32, false, false, 16>(var_52, var_42);
            // state = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)       <L 2650>
            var_55 = wp::view(var_efc_state_in, var_0);
            var_57 = wp::tile_load<wp::int32, false, false, 16>(var_55, var_42);
            // D_k = wp.tile_map(_state_check, D_k, state)                                        <L 2652>
            var_58 = wp::tile_binary_map(_state_check_0, var_54, var_57);
            // tid_tile = wp.tile_arange(TILE_SIZE_K, dtype=int)                                  <L 2655>
            var_59 = wp::tile_arange<wp::int32, 16>(var_60, var_45, var_61);
            // threshold_tile = wp.tile_ones(shape=TILE_SIZE_K, dtype=int) * (nefc - k)           <L 2656>
            var_62 = wp::tile_ones<int, 16>();
            var_63 = wp::sub(var_6, var_42);
            var_64 = wp::tile_mul(var_62, var_63);
            // active_tile = wp.tile_map(_active_check, tid_tile, threshold_tile)                 <L 2658>
            var_65 = wp::tile_binary_map(_active_check_0, var_59, var_64);
            // D_k = wp.tile_map(wp.mul, active_tile, D_k)                                        <L 2659>
            var_66 = wp::tile_binary_map(wp::mul, var_65, var_58);
            // J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_kj), wp.tile_broadcast(D_k, shape=(nv_pad, TILE_SIZE_K)))       <L 2661>
            var_67 = wp::tile_transpose(var_50);
            var_68 = wp::tuple(var_26, var_45);
            var_69 = wp::tile_broadcast<28, 16, 0, 1>(var_66);
            var_70 = wp::tile_binary_map(wp::mul, var_67, var_69);
            // sum_val += wp.tile_matmul(J_ki, J_kj)                                              <L 2663>
            var_71 = wp::tile_matmul(var_72, var_73, var_74, var_70, var_50, var_71, var_75, var_76);
            wp::tile_add_inplace(var_37, var_71);
            goto start_for_3;
        end_for_3:;
        // wp.tile_store(ctx_h_out[worldid], sum_val, bounds_check=False)                         <L 2665>
        var_77 = wp::view(var_ctx_h_out, var_0);
        wp::tile_store<wp::float32, false, false>(var_77, var_79, var_80, var_37);
}

