#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 64
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

#define WP_LTO_LINK_INPUTS 1
extern "C" {
void potrf_28_28_1_120_64_1_1_5_x_x_0(wp::float32*, int*);
void potrs_28_28_1_120_64_1_1_5_x_x_0(wp::float32*, wp::float32*);
}

// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\block_cholesky.py:21
static CUDA_CALLABLE wp::vec_t<2, wp::float32> solve_search_sums_0(
    wp::float32 var_grad,
    wp::float32 var_solution)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::vec_t<2, wp::float32> var_2;
    //---------
    // forward
    // def solve_search_sums(grad: float, solution: float):                                   <L 22>
    // return wp.vec2(solution * solution, grad * solution)                                   <L 23>
    var_0 = wp::mul(var_solution, var_solution);
    var_1 = wp::mul(var_grad, var_solution);
    var_2 = wp::vec_t<2, wp::float32>(var_0, var_1);
    return var_2;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\block_cholesky.py:21
static CUDA_CALLABLE void adj_solve_search_sums_0(
    wp::float32 var_grad,
    wp::float32 var_solution,
    wp::float32 & adj_grad,
    wp::float32 & adj_solution,
    wp::vec_t<2, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _update_gradient_cholesky__locals__kernel_cf3a6ef8_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_ctx_grad_in,
    wp::array_t<wp::float32> var_h_in,
    wp::array_t<wp::int32> var_state_changed_count_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_ctx_search_out,
    wp::array_t<wp::float32> var_ctx_search_dot_out,
    wp::array_t<wp::float32> var_ctx_newton_decrement_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 28;
        bool* var_2;
        bool var_3;
        bool var_4;
        const bool var_5 = false;
        wp::array_t<wp::float32> var_6;
        wp::tuple_t<wp::int32, wp::int32> var_7;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<28,28>, wp::tile_stride_t<28,1>>, true> var_8 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<28,28>,wp::tile_stride_t<28,1>,false>();
        const wp::int32 var_9 = 0;
        const wp::int32 var_10 = 0;
        const wp::str var_11 = "upper";
        wp::array_t<wp::float32> var_12;
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<28>, wp::tile_stride_t<1>>, true> var_13 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<28>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_14 = 0;
        const wp::str var_15 = "upper";
        wp::tile_shared_t<wp::float32,wp::tile_layout_strided_t<wp::tile_shape_t<28>, wp::tile_stride_t<1>>, true> var_16 = wp::tile_alloc_empty<wp::float32,wp::tile_shape_t<28>,wp::tile_stride_t<1>,false>();
        wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<28>>> var_17 = wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<28>>>{};
        wp::tile_shared_t<wp::vec_t<2, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_18 = wp::tile_alloc_empty<wp::vec_t<2, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_19 = 0;
        wp::vec_t<2, wp::float32> var_20;
        const wp::int32 var_21 = 0;
        wp::float32 var_22;
        const wp::int32 var_23 = 1;
        wp::float32 var_24;
        wp::array_t<wp::float32> var_25;
        const wp::float32 var_26 = -1.0;
        wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<28>>> var_27 = wp::tile_register_t<wp::float32,wp::tile_layout_register_t<wp::tile_shape_t<28>>>{};
        const wp::int32 var_28 = 0;
        //---------
        // forward
        // def kernel(                                                                            <L 2799>
        // worldid = wp.tid()                                                                     <L 2810>
        var_0 = builtin_tid1d();
        // TILE_SIZE = wp.static(tile_size)                                                       <L 2811>
        // if ctx_done_in[worldid]:                                                               <L 2813>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 2814>
            return;
        }
        var_4 = wp::load(var_2);
        // if wp.static(SKIP_NOFLIP):                                                             <L 2817>
        // mat_tile = wp.tile_load(h_in[worldid], shape=(TILE_SIZE, TILE_SIZE))                   <L 2821>
        var_6 = wp::view(var_h_in, var_0);
        var_7 = wp::tuple(var_1, var_1);
        var_8 = wp::tile_load<wp::float32, true, false, 28, 28>(var_6, var_9, var_10);
        // wp.tile_cholesky_inplace(mat_tile, fill_mode="upper")                                  <L 2822>
        wp::tile_cholesky_inplace<true>(potrf_28_28_1_120_64_1_1_5_x_x_0, var_8);
        // input_tile = wp.tile_load(ctx_grad_in[worldid], shape=TILE_SIZE)                       <L 2823>
        var_12 = wp::view(var_ctx_grad_in, var_0);
        var_13 = wp::tile_load<wp::float32, true, false, 28>(var_12, var_14);
        // output_tile = wp.tile_cholesky_solve(mat_tile, input_tile, fill_mode="upper")          <L 2824>
        var_16 = wp::tile_cholesky_solve<true>(potrs_28_28_1_120_64_1_1_5_x_x_0, var_8, var_13, var_16);
        // sums = wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, input_tile, output_tile))[0]       <L 2825>
        var_17 = wp::tile_binary_map(solve_search_sums_0, var_13, var_16);
        var_18 = wp::tile_reduce(wp::add, var_17);
        var_20 = wp::tile_extract(var_18, var_19);
        // ctx_search_dot_out[worldid] = sums[0]                                                  <L 2826>
        var_22 = wp::extract(var_20, var_21);
        wp::array_store(var_ctx_search_dot_out, var_0, var_22);
        // ctx_newton_decrement_out[worldid] = sums[1]                                            <L 2827>
        var_24 = wp::extract(var_20, var_23);
        wp::array_store(var_ctx_newton_decrement_out, var_0, var_24);
        // wp.tile_store(ctx_search_out[worldid], wp.tile_map(wp.mul, output_tile, -1.0))         <L 2828>
        var_25 = wp::view(var_ctx_search_out, var_0);
        var_27 = wp::tile_binary_map(wp::mul, var_16, var_26);
        wp::tile_store<wp::float32, true, false>(var_25, var_28, var_27);
}

