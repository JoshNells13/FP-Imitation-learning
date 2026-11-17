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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:138
static CUDA_CALLABLE wp::float32 _rescale_0(
    wp::int32 var_nv,
    wp::float32 var_meaninertia,
    wp::float32 var_value)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    //---------
    // forward
    // def _rescale(nv: int, meaninertia: float, value: float) -> float:                      <L 139>
    // return value / (meaninertia * float(nv))                                               <L 140>
    var_0 = wp::float(var_nv);
    var_1 = wp::mul(var_meaninertia, var_0);
    var_2 = wp::div(var_value, var_1);
    return var_2;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:138
static CUDA_CALLABLE void adj__rescale_0(
    wp::int32 var_nv,
    wp::float32 var_meaninertia,
    wp::float32 var_value,
    wp::int32 & adj_nv,
    wp::float32 & adj_meaninertia,
    wp::float32 & adj_value,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _solve_done__locals__kernel_11e78029_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_tolerance,
    wp::int32 var_opt_iterations,
    wp::array_t<wp::float32> var_stat_meaninertia,
    wp::array_t<wp::float32> var_ctx_grad_dot_in,
    wp::array_t<wp::float32> var_ctx_newton_decrement_in,
    wp::array_t<wp::float32> var_ctx_improvement_in,
    wp::array_t<wp::float32> var_ctx_alpha_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::int32> var_solver_niter_out,
    wp::array_t<wp::int32> var_overflow_out,
    wp::array_t<wp::int32> var_nsolving_out,
    wp::array_t<bool> var_ctx_done_out)
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
        bool* var_1;
        bool var_2;
        bool var_3;
        const wp::int32 var_4 = 1;
        wp::int32 var_5;
        wp::shape_t* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::shape_t var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::shape_t* var_14;
        const wp::int32 var_15 = 0;
        wp::int32 var_16;
        wp::shape_t var_17;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32* var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        wp::float32* var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.5;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        bool var_37;
        const wp::float32 var_38 = 0.0;
        bool var_39;
        bool var_40;
        const wp::float32 var_41 = 0.0;
        bool var_42;
        bool var_43;
        bool var_44;
        bool var_45;
        bool var_46;
        wp::int32* var_47;
        bool var_48;
        wp::int32 var_49;
        bool var_50;
        bool var_51;
        wp::int32* var_52;
        bool var_53;
        wp::int32 var_54;
        const bool var_55 = true;
        const wp::str var_56 = "solver iterations limit reached - please increase iterations beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n";
        wp::int32* var_57;
        const wp::int32 var_58 = 512;
        wp::int32 var_59;
        wp::int32 var_60;
        const bool var_61 = true;
        const wp::int32 var_62 = 0;
        const wp::int32 var_63 = -1;
        wp::int32 var_64;
        //---------
        // forward
        // def kernel(                                                                            <L 3713>
        // worldid = wp.tid()                                                                     <L 3732>
        var_0 = builtin_tid1d();
        // if ctx_done_in[worldid]:                                                               <L 3734>
        var_1 = wp::address(var_ctx_done_in, var_0);
        var_2 = wp::load(var_1);
        if (var_2) {
            // return                                                                             <L 3735>
            continue;
        }
        var_3 = wp::load(var_1);
        // solver_niter_out[worldid] += 1                                                         <L 3737>
        var_5 = wp::atomic_add(var_solver_niter_out, var_0, var_4);
        // tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]                            <L 3738>
        var_6 = &(var_opt_tolerance.shape);
        var_9 = wp::load(var_6);
        var_8 = wp::extract(var_9, var_7);
        var_10 = wp::mod(var_0, var_8);
        var_11 = wp::address(var_opt_tolerance, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]                    <L 3739>
        var_14 = &(var_stat_meaninertia.shape);
        var_17 = wp::load(var_14);
        var_16 = wp::extract(var_17, var_15);
        var_18 = wp::mod(var_0, var_16);
        var_19 = wp::address(var_stat_meaninertia, var_18);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // alpha = ctx_alpha_in[worldid]                                                          <L 3741>
        var_22 = wp::address(var_ctx_alpha_in, var_0);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // improvement = _rescale(nv, meaninertia, ctx_improvement_in[worldid])                   <L 3742>
        var_25 = wp::address(var_ctx_improvement_in, var_0);
        var_27 = wp::load(var_25);
        var_26 = _rescale_0(var_nv, var_20, var_27);
        // gradient = _rescale(nv, meaninertia, wp.sqrt(ctx_grad_dot_in[worldid]))                <L 3743>
        var_28 = wp::address(var_ctx_grad_dot_in, var_0);
        var_30 = wp::load(var_28);
        var_29 = wp::sqrt(var_30);
        var_31 = _rescale_0(var_nv, var_20, var_29);
        // model_improvement = _rescale(nv, meaninertia, 0.5 * ctx_newton_decrement_in[worldid])       <L 3744>
        var_33 = wp::address(var_ctx_newton_decrement_in, var_0);
        var_35 = wp::load(var_33);
        var_34 = wp::mul(var_32, var_35);
        var_36 = _rescale_0(var_nv, var_20, var_34);
        // done = (                                                                               <L 3745>
        // (alpha == 0.0)                                                                         <L 3746>
        var_39 = (var_23 == var_38);
        var_37 = var_39;
        if (!var_37) {
            // or (improvement > 0.0 and improvement < tolerance)                                 <L 3747>
            var_42 = (var_26 > var_41);
            var_40 = var_42;
            if (var_40) {
                var_43 = (var_26 < var_12);
                var_40 = var_40 && var_43;
            }
            var_37 = var_37 || var_40;
        }
        if (!var_37) {
            // or (gradient < tolerance)                                                          <L 3748>
            var_44 = (var_31 < var_12);
            var_37 = var_37 || var_44;
        }
        if (!var_37) {
            // or (model_improvement < tolerance)                                                 <L 3749>
            var_45 = (var_36 < var_12);
            var_37 = var_37 || var_45;
        }
        // if done or solver_niter_out[worldid] == opt_iterations:                                <L 3751>
        var_46 = var_37;
        if (!var_46) {
            var_47 = wp::address(var_solver_niter_out, var_0);
            var_49 = wp::load(var_47);
            var_48 = (var_49 == var_opt_iterations);
            var_46 = var_46 || var_48;
        }
        if (var_46) {
            // if not done and solver_niter_out[worldid] == opt_iterations:                       <L 3754>
            var_51 = wp::unot(var_37);
            var_50 = var_51;
            if (var_50) {
                var_52 = wp::address(var_solver_niter_out, var_0);
                var_54 = wp::load(var_52);
                var_53 = (var_54 == var_opt_iterations);
                var_50 = var_50 && var_53;
            }
            if (var_50) {
                // if wp.static(bool(warn_overflow & OverflowType.ITERATIONS)):                   <L 3755>
                // wp.printf(                                                                     <L 3756>
                // "solver iterations limit reached - please increase iterations beyond %u\n"       <L 3757>
                // opt_iterations,                                                                <L 3759>
                printf(var_56, var_opt_iterations);
                // overflow_out[worldid] = overflow_out[worldid] | OverflowType.ITERATIONS        <L 3761>
                var_57 = wp::address(var_overflow_out, var_0);
                var_60 = wp::load(var_57);
                var_59 = wp::bit_or(var_60, var_58);
                wp::array_store(var_overflow_out, var_0, var_59);
            }
            // ctx_done_out[worldid] = True                                                       <L 3762>
            wp::array_store(var_ctx_done_out, var_0, var_61);
            // wp.atomic_add(nsolving_out, 0, -1)                                                 <L 3763>
            var_64 = wp::atomic_add(var_nsolving_out, var_62, var_63);
        }
    }
}

