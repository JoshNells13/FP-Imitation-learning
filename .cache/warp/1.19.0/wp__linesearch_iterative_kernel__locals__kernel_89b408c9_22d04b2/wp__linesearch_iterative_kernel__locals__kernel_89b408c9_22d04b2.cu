#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 32
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:884
static CUDA_CALLABLE void _linesearch_iterative_kernel__locals___syncthreads_0(
    )
{
WP_TILE_SYNC();}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:148
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_direct_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.5;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _eval_pt_direct_alpha_zero(jaref: float, jv: float, d: float) -> wp.vec3:          <L 149>
    // jvD = jv * d                                                                           <L 151>
    var_0 = wp::mul(var_jv, var_d);
    // return wp.vec3(0.5 * d * jaref * jaref, jvD * jaref, jv * jvD)                         <L 152>
    var_2 = wp::mul(var_1, var_d);
    var_3 = wp::mul(var_2, var_jaref);
    var_4 = wp::mul(var_3, var_jaref);
    var_5 = wp::mul(var_0, var_jaref);
    var_6 = wp::mul(var_jv, var_0);
    var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    return var_7;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE wp::float32 safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::float32 var_3 = 1e-15;
    wp::float32 var_4;
    wp::float32 var_5;
    //---------
    // forward
    // def safe_div(x: Any, y: Any) -> Any:                                                   <L 1>
    // return x / wp.where(y != 0.0, y, types.MJ_MINVAL)                                      <L 2>
    var_1 = (var_y != var_0);
    var_4 = wp::where(var_1, var_y, var_3);
    var_5 = wp::div(var_x, var_4);
    return var_5;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:249
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_frictionloss_pt_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d)
{
    //---------
    // primal vars
    bool var_0;
    wp::float32 var_1;
    bool var_2;
    bool var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::float32 var_12;
    bool var_13;
    const wp::float32 var_14 = -0.5;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 0.0;
    wp::vec_t<3, wp::float32> var_21;
    const wp::float32 var_22 = -0.5;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    const wp::float32 var_27 = 0.0;
    wp::vec_t<3, wp::float32> var_28;
    //---------
    // forward
    // def _eval_frictionloss_pt(x: float, f: float, rf: float, jv: float, d: float) -> wp.vec3:       <L 250>
    // if (-rf < x) and (x < rf):                                                             <L 252>
    var_1 = wp::neg(var_rf);
    var_2 = (var_1 < var_x);
    var_0 = var_2;
    if (var_0) {
        var_3 = (var_x < var_rf);
        var_0 = var_0 && var_3;
    }
    if (var_0) {
        // jvD = jv * d                                                                       <L 253>
        var_4 = wp::mul(var_jv, var_d);
        // return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)                                 <L 254>
        var_6 = wp::mul(var_5, var_d);
        var_7 = wp::mul(var_6, var_x);
        var_8 = wp::mul(var_7, var_x);
        var_9 = wp::mul(var_4, var_x);
        var_10 = wp::mul(var_jv, var_4);
        var_11 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
        return var_11;
    }
    if (!var_0) {
        // elif x <= -rf:                                                                     <L 255>
        var_12 = wp::neg(var_rf);
        var_13 = (var_x <= var_12);
        if (var_13) {
            // return wp.vec3(f * (-0.5 * rf - x), -f * jv, 0.0)                              <L 256>
            var_15 = wp::mul(var_14, var_rf);
            var_16 = wp::sub(var_15, var_x);
            var_17 = wp::mul(var_f, var_16);
            var_18 = wp::neg(var_f);
            var_19 = wp::mul(var_18, var_jv);
            var_21 = wp::vec_t<3, wp::float32>(var_17, var_19, var_20);
            return var_21;
        }
        if (!var_13) {
            // return wp.vec3(f * (-0.5 * rf + x), f * jv, 0.0)                               <L 258>
            var_23 = wp::mul(var_22, var_rf);
            var_24 = wp::add(var_23, var_x);
            var_25 = wp::mul(var_f, var_24);
            var_26 = wp::mul(var_f, var_jv);
            var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
            return var_28;
        }
    }
    return {};
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:600
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _compute_efc_eval_pt_alpha_zero_pyramidal_0(
    wp::int32 var_efcid,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<3, wp::float32> var_6;
    bool var_7;
    wp::float32* var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def _compute_efc_eval_pt_alpha_zero_pyramidal(                                         <L 601>
    // if efcid >= ne + nf:                                                                   <L 613>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // if ctx_Jaref < 0.0:                                                                <L 614>
        var_3 = (var_ctx_Jaref < var_2);
        if (var_3) {
            // return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)                    <L 615>
            var_4 = _eval_pt_direct_alpha_zero_0(var_ctx_Jaref, var_ctx_jv, var_efc_D);
            return var_4;
        }
        // return wp.vec3(0.0)                                                                <L 616>
        var_6 = wp::vec_t<3, wp::float32>(var_5);
        return var_6;
    }
    // if efcid >= ne:                                                                        <L 619>
    var_7 = (var_efcid >= var_ne);
    if (var_7) {
        // f = efc_frictionloss[efcid]                                                        <L 620>
        var_8 = wp::address(var_efc_frictionloss, var_efcid);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // rf = math.safe_div(f, efc_D)                                                       <L 621>
        var_11 = safe_div_0(var_9, var_efc_D);
        // return _eval_frictionloss_pt(ctx_Jaref, f, rf, ctx_jv, efc_D)                      <L 622>
        var_12 = _eval_frictionloss_pt_0(var_ctx_Jaref, var_9, var_11, var_ctx_jv, var_efc_D);
        return var_12;
    }
    // return _eval_pt_direct_alpha_zero(ctx_Jaref, ctx_jv, efc_D)                            <L 625>
    var_13 = _eval_pt_direct_alpha_zero_0(var_ctx_Jaref, var_ctx_jv, var_efc_D);
    return var_13;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:163
static CUDA_CALLABLE wp::float32 _eval_pt_direct_cost_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_d)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    //---------
    // forward
    // def _eval_pt_direct_cost_alpha_zero(jaref: float, d: float) -> float:                  <L 164>
    // return 0.5 * d * jaref * jaref                                                         <L 165>
    var_1 = wp::mul(var_0, var_d);
    var_2 = wp::mul(var_1, var_jaref);
    var_3 = wp::mul(var_2, var_jaref);
    return var_3;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:168
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_direct_shifted_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_alpha,
    wp::float32 var_offset)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 0.5;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def _eval_pt_direct_shifted(jaref: float, jv: float, d: float, alpha: float, offset: float) -> wp.vec3:       <L 169>
    // jvD = jv * d                                                                           <L 171>
    var_0 = wp::mul(var_jv, var_d);
    // hessian = jv * jvD                                                                     <L 172>
    var_1 = wp::mul(var_jv, var_0);
    // alpha_h = alpha * hessian                                                              <L 173>
    var_2 = wp::mul(var_alpha, var_1);
    // return wp.vec3(alpha * (jvD * jaref + 0.5 * alpha_h) + offset, jvD * jaref + alpha_h, hessian)       <L 174>
    var_3 = wp::mul(var_0, var_jaref);
    var_5 = wp::mul(var_4, var_2);
    var_6 = wp::add(var_3, var_5);
    var_7 = wp::mul(var_alpha, var_6);
    var_8 = wp::add(var_7, var_offset);
    var_9 = wp::mul(var_0, var_jaref);
    var_10 = wp::add(var_9, var_2);
    var_11 = wp::vec_t<3, wp::float32>(var_8, var_10, var_1);
    return var_11;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:261
static CUDA_CALLABLE wp::float32 _eval_frictionloss_cost_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_d)
{
    //---------
    // primal vars
    bool var_0;
    wp::float32 var_1;
    bool var_2;
    bool var_3;
    const wp::float32 var_4 = 0.5;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    bool var_9;
    const wp::float32 var_10 = -0.5;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = -0.5;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def _eval_frictionloss_cost(x: float, f: float, rf: float, d: float) -> float:         <L 262>
    // if (-rf < x) and (x < rf):                                                             <L 263>
    var_1 = wp::neg(var_rf);
    var_2 = (var_1 < var_x);
    var_0 = var_2;
    if (var_0) {
        var_3 = (var_x < var_rf);
        var_0 = var_0 && var_3;
    }
    if (var_0) {
        // return 0.5 * d * x * x                                                             <L 264>
        var_5 = wp::mul(var_4, var_d);
        var_6 = wp::mul(var_5, var_x);
        var_7 = wp::mul(var_6, var_x);
        return var_7;
    }
    if (!var_0) {
        // elif x <= -rf:                                                                     <L 265>
        var_8 = wp::neg(var_rf);
        var_9 = (var_x <= var_8);
        if (var_9) {
            // return f * (-0.5 * rf - x)                                                     <L 266>
            var_11 = wp::mul(var_10, var_rf);
            var_12 = wp::sub(var_11, var_x);
            var_13 = wp::mul(var_f, var_12);
            return var_13;
        }
    }
    // return f * (-0.5 * rf + x)                                                             <L 267>
    var_15 = wp::mul(var_14, var_rf);
    var_16 = wp::add(var_15, var_x);
    var_17 = wp::mul(var_f, var_16);
    return var_17;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:244
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _shift_cost_0(
    wp::vec_t<3, wp::float32> var_pt,
    wp::float32 var_cost0)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::int32 var_3 = 1;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _shift_cost(pt: wp.vec3, cost0: float) -> wp.vec3:                                 <L 245>
    // return wp.vec3(pt[0] - cost0, pt[1], pt[2])                                            <L 246>
    var_1 = wp::extract(var_pt, var_0);
    var_2 = wp::sub(var_1, var_cost0);
    var_4 = wp::extract(var_pt, var_3);
    var_6 = wp::extract(var_pt, var_5);
    var_7 = wp::vec_t<3, wp::float32>(var_2, var_4, var_6);
    return var_7;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:504
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _compute_efc_eval_pt_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.0;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 0.0;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<3, wp::float32> var_16;
    bool var_17;
    wp::float32* var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    const wp::float32 var_28 = 0.0;
    wp::vec_t<3, wp::float32> var_29;
    //---------
    // forward
    // def _compute_efc_eval_pt_pyramidal(                                                    <L 505>
    // if efcid >= ne + nf:                                                                   <L 521>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // x = ctx_Jaref + alpha * ctx_jv                                                     <L 522>
        var_2 = wp::mul(var_alpha, var_ctx_jv);
        var_3 = wp::add(var_ctx_Jaref, var_2);
        // quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)                          <L 523>
        var_4 = _eval_pt_direct_cost_alpha_zero_0(var_ctx_Jaref, var_efc_D);
        // cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)                                      <L 524>
        var_6 = (var_ctx_Jaref < var_5);
        var_8 = wp::where(var_6, var_4, var_7);
        // offset = quad0 - cost0                                                             <L 527>
        var_9 = wp::sub(var_4, var_8);
        // if x < 0.0:                                                                        <L 528>
        var_11 = (var_3 < var_10);
        if (var_11) {
            // return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, offset)        <L 529>
            var_12 = _eval_pt_direct_shifted_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_alpha, var_9);
            return var_12;
        }
        // return wp.vec3(-cost0, 0.0, 0.0)                                                   <L 530>
        var_13 = wp::neg(var_8);
        var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
        return var_16;
    }
    // if efcid >= ne:                                                                        <L 533>
    var_17 = (var_efcid >= var_ne);
    if (var_17) {
        // f = efc_frictionloss[efcid]                                                        <L 534>
        var_18 = wp::address(var_efc_frictionloss, var_efcid);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // x = ctx_Jaref + alpha * ctx_jv                                                     <L 535>
        var_21 = wp::mul(var_alpha, var_ctx_jv);
        var_22 = wp::add(var_ctx_Jaref, var_21);
        // rf = math.safe_div(f, efc_D)                                                       <L 536>
        var_23 = safe_div_0(var_19, var_efc_D);
        // return _shift_cost(_eval_frictionloss_pt(x, f, rf, ctx_jv, efc_D), _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D))       <L 537>
        var_24 = _eval_frictionloss_pt_0(var_22, var_19, var_23, var_ctx_jv, var_efc_D);
        var_25 = _eval_frictionloss_cost_0(var_ctx_Jaref, var_19, var_23, var_efc_D);
        var_26 = _shift_cost_0(var_24, var_25);
        return var_26;
    }
    var_27 = wp::where(var_17, var_22, var_3);
    // return _eval_pt_direct_shifted(ctx_Jaref, ctx_jv, efc_D, alpha, 0.0)                   <L 540>
    var_29 = _eval_pt_direct_shifted_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_alpha, var_28);
    return var_29;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:218
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_pt_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_alpha)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 2.0;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 2.0;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    //---------
    // forward
    // def _eval_pt(quad: wp.vec3, alpha: float) -> wp.vec3:                                  <L 219>
    // aq2 = alpha * quad[2]                                                                  <L 221>
    var_1 = wp::extract(var_quad, var_0);
    var_2 = wp::mul(var_alpha, var_1);
    // return wp.vec3(                                                                        <L 222>
    // alpha * aq2 + alpha * quad[1] + quad[0],                                               <L 223>
    var_3 = wp::mul(var_alpha, var_2);
    var_5 = wp::extract(var_quad, var_4);
    var_6 = wp::mul(var_alpha, var_5);
    var_7 = wp::add(var_3, var_6);
    var_9 = wp::extract(var_quad, var_8);
    var_10 = wp::add(var_7, var_9);
    // 2.0 * aq2 + quad[1],                                                                   <L 224>
    var_12 = wp::mul(var_11, var_2);
    var_14 = wp::extract(var_quad, var_13);
    var_15 = wp::add(var_12, var_14);
    // 2.0 * quad[2],                                                                         <L 225>
    var_18 = wp::extract(var_quad, var_17);
    var_19 = wp::mul(var_16, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_10, var_15, var_19);
    return var_20;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:195
static CUDA_CALLABLE void _eval_pt_direct_shifted_3alphas_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::float32 var_offset,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.5;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::float32 var_13 = 0.5;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::float32 var_20 = 0.5;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    //---------
    // forward
    // def _eval_pt_direct_shifted_3alphas(                                                   <L 196>
    // jvD = jv * d                                                                           <L 200>
    var_0 = wp::mul(var_jv, var_d);
    // grad0 = jvD * jaref                                                                    <L 201>
    var_1 = wp::mul(var_0, var_jaref);
    // hessian = jv * jvD                                                                     <L 202>
    var_2 = wp::mul(var_jv, var_0);
    // lo_ah = lo_alpha * hessian                                                             <L 203>
    var_3 = wp::mul(var_lo_alpha, var_2);
    // hi_ah = hi_alpha * hessian                                                             <L 204>
    var_4 = wp::mul(var_hi_alpha, var_2);
    // mid_ah = mid_alpha * hessian                                                           <L 205>
    var_5 = wp::mul(var_mid_alpha, var_2);
    // return (                                                                               <L 206>
    // wp.vec3(lo_alpha * (grad0 + 0.5 * lo_ah) + offset, grad0 + lo_ah, hessian),            <L 207>
    var_7 = wp::mul(var_6, var_3);
    var_8 = wp::add(var_1, var_7);
    var_9 = wp::mul(var_lo_alpha, var_8);
    var_10 = wp::add(var_9, var_offset);
    var_11 = wp::add(var_1, var_3);
    var_12 = wp::vec_t<3, wp::float32>(var_10, var_11, var_2);
    // wp.vec3(hi_alpha * (grad0 + 0.5 * hi_ah) + offset, grad0 + hi_ah, hessian),            <L 208>
    var_14 = wp::mul(var_13, var_4);
    var_15 = wp::add(var_1, var_14);
    var_16 = wp::mul(var_hi_alpha, var_15);
    var_17 = wp::add(var_16, var_offset);
    var_18 = wp::add(var_1, var_4);
    var_19 = wp::vec_t<3, wp::float32>(var_17, var_18, var_2);
    // wp.vec3(mid_alpha * (grad0 + 0.5 * mid_ah) + offset, grad0 + mid_ah, hessian),         <L 209>
    var_21 = wp::mul(var_20, var_5);
    var_22 = wp::add(var_1, var_21);
    var_23 = wp::mul(var_mid_alpha, var_22);
    var_24 = wp::add(var_23, var_offset);
    var_25 = wp::add(var_1, var_5);
    var_26 = wp::vec_t<3, wp::float32>(var_24, var_25, var_2);
    ret_0 = var_12;
    ret_1 = var_19;
    ret_2 = var_26;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:270
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _eval_frictionloss_pt_one_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_half_d,
    wp::float32 var_jvD,
    wp::float32 var_hessian,
    wp::float32 var_f_jv)
{
    //---------
    // primal vars
    bool var_0;
    wp::float32 var_1;
    bool var_2;
    bool var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    bool var_9;
    const wp::float32 var_10 = -0.5;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<3, wp::float32> var_16;
    const wp::float32 var_17 = -0.5;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0;
    wp::vec_t<3, wp::float32> var_22;
    //---------
    // forward
    // def _eval_frictionloss_pt_one(x: float, f: float, rf: float, half_d: float, jvD: float, hessian: float, f_jv: float) -> wp.vec3:       <L 271>
    // if (-rf < x) and (x < rf):                                                             <L 273>
    var_1 = wp::neg(var_rf);
    var_2 = (var_1 < var_x);
    var_0 = var_2;
    if (var_0) {
        var_3 = (var_x < var_rf);
        var_0 = var_0 && var_3;
    }
    if (var_0) {
        // return wp.vec3(half_d * x * x, jvD * x, hessian)                                   <L 274>
        var_4 = wp::mul(var_half_d, var_x);
        var_5 = wp::mul(var_4, var_x);
        var_6 = wp::mul(var_jvD, var_x);
        var_7 = wp::vec_t<3, wp::float32>(var_5, var_6, var_hessian);
        return var_7;
    }
    if (!var_0) {
        // elif x <= -rf:                                                                     <L 275>
        var_8 = wp::neg(var_rf);
        var_9 = (var_x <= var_8);
        if (var_9) {
            // return wp.vec3(f * (-0.5 * rf - x), -f_jv, 0.0)                                <L 276>
            var_11 = wp::mul(var_10, var_rf);
            var_12 = wp::sub(var_11, var_x);
            var_13 = wp::mul(var_f, var_12);
            var_14 = wp::neg(var_f_jv);
            var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
            return var_16;
        }
        if (!var_9) {
            // return wp.vec3(f * (-0.5 * rf + x), f_jv, 0.0)                                 <L 278>
            var_18 = wp::mul(var_17, var_rf);
            var_19 = wp::add(var_18, var_x);
            var_20 = wp::mul(var_f, var_19);
            var_22 = wp::vec_t<3, wp::float32>(var_20, var_f_jv, var_21);
            return var_22;
        }
    }
    return {};
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:281
static CUDA_CALLABLE void _eval_frictionloss_pt_3alphas_0(
    wp::float32 var_x_lo,
    wp::float32 var_x_hi,
    wp::float32 var_x_mid,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.5;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def _eval_frictionloss_pt_3alphas(                                                     <L 282>
    // jvD = jv * d                                                                           <L 286>
    var_0 = wp::mul(var_jv, var_d);
    // half_d = 0.5 * d                                                                       <L 287>
    var_2 = wp::mul(var_1, var_d);
    // hessian = jv * jvD                                                                     <L 288>
    var_3 = wp::mul(var_jv, var_0);
    // f_jv = f * jv                                                                          <L 289>
    var_4 = wp::mul(var_f, var_jv);
    // return (                                                                               <L 290>
    // _eval_frictionloss_pt_one(x_lo, f, rf, half_d, jvD, hessian, f_jv),                    <L 291>
    var_5 = _eval_frictionloss_pt_one_0(var_x_lo, var_f, var_rf, var_2, var_0, var_3, var_4);
    // _eval_frictionloss_pt_one(x_hi, f, rf, half_d, jvD, hessian, f_jv),                    <L 292>
    var_6 = _eval_frictionloss_pt_one_0(var_x_hi, var_f, var_rf, var_2, var_0, var_3, var_4);
    // _eval_frictionloss_pt_one(x_mid, f, rf, half_d, jvD, hessian, f_jv),                   <L 293>
    var_7 = _eval_frictionloss_pt_one_0(var_x_mid, var_f, var_rf, var_2, var_0, var_3, var_4);
    ret_0 = var_5;
    ret_1 = var_6;
    ret_2 = var_7;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:672
static CUDA_CALLABLE void _compute_efc_eval_pt_3alphas_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    const wp::float32 var_11 = 0.0;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.0;
    const wp::float32 var_19 = 0.0;
    wp::vec_t<3, wp::float32> var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    wp::vec_t<3, wp::float32> var_23;
    const wp::float32 var_24 = 0.0;
    bool var_25;
    wp::vec_t<3, wp::float32> var_26;
    const wp::float32 var_27 = 0.0;
    bool var_28;
    wp::vec_t<3, wp::float32> var_29;
    bool var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32* var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 0.0;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<3, wp::float32> var_55;
    //---------
    // forward
    // def _compute_efc_eval_pt_3alphas_pyramidal(                                            <L 673>
    // if efcid >= ne + nf:                                                                   <L 692>
    var_0 = wp::add(var_ne, var_nf);
    var_1 = (var_efcid >= var_0);
    if (var_1) {
        // x_lo = ctx_Jaref + lo_alpha * ctx_jv                                               <L 693>
        var_2 = wp::mul(var_lo_alpha, var_ctx_jv);
        var_3 = wp::add(var_ctx_Jaref, var_2);
        // x_hi = ctx_Jaref + hi_alpha * ctx_jv                                               <L 694>
        var_4 = wp::mul(var_hi_alpha, var_ctx_jv);
        var_5 = wp::add(var_ctx_Jaref, var_4);
        // x_mid = ctx_Jaref + mid_alpha * ctx_jv                                             <L 695>
        var_6 = wp::mul(var_mid_alpha, var_ctx_jv);
        var_7 = wp::add(var_ctx_Jaref, var_6);
        // quad0 = _eval_pt_direct_cost_alpha_zero(ctx_Jaref, efc_D)                          <L 696>
        var_8 = _eval_pt_direct_cost_alpha_zero_0(var_ctx_Jaref, var_efc_D);
        // cost0 = wp.where(ctx_Jaref < 0.0, quad0, 0.0)                                      <L 697>
        var_10 = (var_ctx_Jaref < var_9);
        var_12 = wp::where(var_10, var_8, var_11);
        // offset = quad0 - cost0                                                             <L 699>
        var_13 = wp::sub(var_8, var_12);
        // pt_lo, pt_hi, pt_mid = _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, offset)       <L 700>
        _eval_pt_direct_shifted_3alphas_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_lo_alpha, var_hi_alpha, var_mid_alpha, var_13, var_14, var_15, var_16);
        // inactive = wp.vec3(-cost0, 0.0, 0.0)                                               <L 701>
        var_17 = wp::neg(var_12);
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // return (                                                                           <L 702>
        // wp.where(x_lo < 0.0, pt_lo, inactive),                                             <L 703>
        var_22 = (var_3 < var_21);
        var_23 = wp::where(var_22, var_14, var_20);
        // wp.where(x_hi < 0.0, pt_hi, inactive),                                             <L 704>
        var_25 = (var_5 < var_24);
        var_26 = wp::where(var_25, var_15, var_20);
        // wp.where(x_mid < 0.0, pt_mid, inactive),                                           <L 705>
        var_28 = (var_7 < var_27);
        var_29 = wp::where(var_28, var_16, var_20);
        ret_0 = var_23;
        ret_1 = var_26;
        ret_2 = var_29;
        return;
    }
    // if efcid >= ne:                                                                        <L 709>
    var_30 = (var_efcid >= var_ne);
    if (var_30) {
        // x_lo = ctx_Jaref + lo_alpha * ctx_jv                                               <L 710>
        var_31 = wp::mul(var_lo_alpha, var_ctx_jv);
        var_32 = wp::add(var_ctx_Jaref, var_31);
        // x_hi = ctx_Jaref + hi_alpha * ctx_jv                                               <L 711>
        var_33 = wp::mul(var_hi_alpha, var_ctx_jv);
        var_34 = wp::add(var_ctx_Jaref, var_33);
        // x_mid = ctx_Jaref + mid_alpha * ctx_jv                                             <L 712>
        var_35 = wp::mul(var_mid_alpha, var_ctx_jv);
        var_36 = wp::add(var_ctx_Jaref, var_35);
        // f = efc_frictionloss[efcid]                                                        <L 713>
        var_37 = wp::address(var_efc_frictionloss, var_efcid);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // rf = math.safe_div(f, efc_D)                                                       <L 714>
        var_40 = safe_div_0(var_38, var_efc_D);
        // cost0 = _eval_frictionloss_cost(ctx_Jaref, f, rf, efc_D)                           <L 715>
        var_41 = _eval_frictionloss_cost_0(var_ctx_Jaref, var_38, var_40, var_efc_D);
        // lo, hi, mid = _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, ctx_jv, efc_D)       <L 716>
        _eval_frictionloss_pt_3alphas_0(var_32, var_34, var_36, var_38, var_40, var_ctx_jv, var_efc_D, var_42, var_43, var_44);
        // return (_shift_cost(lo, cost0), _shift_cost(hi, cost0), _shift_cost(mid, cost0))       <L 717>
        var_45 = _shift_cost_0(var_42, var_41);
        var_46 = _shift_cost_0(var_43, var_41);
        var_47 = _shift_cost_0(var_44, var_41);
        ret_0 = var_45;
        ret_1 = var_46;
        ret_2 = var_47;
        return;
    }
    var_48 = wp::where(var_30, var_32, var_3);
    var_49 = wp::where(var_30, var_34, var_5);
    var_50 = wp::where(var_30, var_36, var_7);
    var_51 = wp::where(var_30, var_41, var_12);
    // return _eval_pt_direct_shifted_3alphas(ctx_Jaref, ctx_jv, efc_D, lo_alpha, hi_alpha, mid_alpha, 0.0)       <L 720>
    _eval_pt_direct_shifted_3alphas_0(var_ctx_Jaref, var_ctx_jv, var_efc_D, var_lo_alpha, var_hi_alpha, var_mid_alpha, var_52, var_53, var_54, var_55);
    ret_0 = var_53;
    ret_1 = var_54;
    ret_2 = var_55;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:229
static CUDA_CALLABLE void _eval_pt_3alphas_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::float32 var_6 = 2.0;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 2.0;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 2.0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::float32 var_31 = 2.0;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    //---------
    // forward
    // def _eval_pt_3alphas(quad: wp.vec3, lo_alpha: float, hi_alpha: float, mid_alpha: float) -> tuple[wp.vec3, wp.vec3, wp.vec3]:       <L 230>
    // q0, q1, q2 = quad[0], quad[1], quad[2]                                                 <L 232>
    var_1 = wp::extract(var_quad, var_0);
    var_3 = wp::extract(var_quad, var_2);
    var_5 = wp::extract(var_quad, var_4);
    // hessian = 2.0 * q2                                                                     <L 233>
    var_7 = wp::mul(var_6, var_5);
    // lo_aq2 = lo_alpha * q2                                                                 <L 234>
    var_8 = wp::mul(var_lo_alpha, var_5);
    // hi_aq2 = hi_alpha * q2                                                                 <L 235>
    var_9 = wp::mul(var_hi_alpha, var_5);
    // mid_aq2 = mid_alpha * q2                                                               <L 236>
    var_10 = wp::mul(var_mid_alpha, var_5);
    // return (                                                                               <L 237>
    // wp.vec3(lo_alpha * lo_aq2 + lo_alpha * q1 + q0, 2.0 * lo_aq2 + q1, hessian),           <L 238>
    var_11 = wp::mul(var_lo_alpha, var_8);
    var_12 = wp::mul(var_lo_alpha, var_3);
    var_13 = wp::add(var_11, var_12);
    var_14 = wp::add(var_13, var_1);
    var_16 = wp::mul(var_15, var_8);
    var_17 = wp::add(var_16, var_3);
    var_18 = wp::vec_t<3, wp::float32>(var_14, var_17, var_7);
    // wp.vec3(hi_alpha * hi_aq2 + hi_alpha * q1 + q0, 2.0 * hi_aq2 + q1, hessian),           <L 239>
    var_19 = wp::mul(var_hi_alpha, var_9);
    var_20 = wp::mul(var_hi_alpha, var_3);
    var_21 = wp::add(var_19, var_20);
    var_22 = wp::add(var_21, var_1);
    var_24 = wp::mul(var_23, var_9);
    var_25 = wp::add(var_24, var_3);
    var_26 = wp::vec_t<3, wp::float32>(var_22, var_25, var_7);
    // wp.vec3(mid_alpha * mid_aq2 + mid_alpha * q1 + q0, 2.0 * mid_aq2 + q1, hessian),       <L 240>
    var_27 = wp::mul(var_mid_alpha, var_10);
    var_28 = wp::mul(var_mid_alpha, var_3);
    var_29 = wp::add(var_27, var_28);
    var_30 = wp::add(var_29, var_1);
    var_32 = wp::mul(var_31, var_10);
    var_33 = wp::add(var_32, var_3);
    var_34 = wp::vec_t<3, wp::float32>(var_30, var_33, var_7);
    ret_0 = var_18;
    ret_1 = var_26;
    ret_2 = var_34;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:143
static CUDA_CALLABLE bool _in_bracket_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_y)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    bool var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    bool var_11;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    bool var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    const wp::float32 var_19 = 0.0;
    bool var_20;
    //---------
    // forward
    // def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:                                       <L 144>
    // return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)                    <L 145>
    var_3 = wp::extract(var_x, var_2);
    var_5 = wp::extract(var_y, var_4);
    var_6 = (var_3 < var_5);
    var_1 = var_6;
    if (var_1) {
        var_8 = wp::extract(var_y, var_7);
        var_10 = (var_8 < var_9);
        var_1 = var_1 && var_10;
    }
    var_0 = var_1;
    if (!var_0) {
        var_13 = wp::extract(var_x, var_12);
        var_15 = wp::extract(var_y, var_14);
        var_16 = (var_13 > var_15);
        var_11 = var_16;
        if (var_11) {
            var_18 = wp::extract(var_y, var_17);
            var_20 = (var_18 > var_19);
            var_11 = var_11 && var_20;
        }
        var_0 = var_0 || var_11;
    }
    return var_0;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:884
static CUDA_CALLABLE void adj__linesearch_iterative_kernel__locals___syncthreads_0(
    )
{
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:148
static CUDA_CALLABLE void adj__eval_pt_direct_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE void adj_safe_div_0(
    wp::float32 var_x,
    wp::float32 var_y,
    wp::float32 & adj_x,
    wp::float32 & adj_y,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:249
static CUDA_CALLABLE void adj__eval_frictionloss_pt_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:600
static CUDA_CALLABLE void adj__compute_efc_eval_pt_alpha_zero_pyramidal_0(
    wp::int32 var_efcid,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::int32 & adj_efcid,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_efc_D,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:163
static CUDA_CALLABLE void adj__eval_pt_direct_cost_alpha_zero_0(
    wp::float32 var_jaref,
    wp::float32 var_d,
    wp::float32 & adj_jaref,
    wp::float32 & adj_d,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:168
static CUDA_CALLABLE void adj__eval_pt_direct_shifted_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_alpha,
    wp::float32 var_offset,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::float32 & adj_alpha,
    wp::float32 & adj_offset,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:261
static CUDA_CALLABLE void adj__eval_frictionloss_cost_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_d,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_d,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:244
static CUDA_CALLABLE void adj__shift_cost_0(
    wp::vec_t<3, wp::float32> var_pt,
    wp::float32 var_cost0,
    wp::vec_t<3, wp::float32> & adj_pt,
    wp::float32 & adj_cost0,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:504
static CUDA_CALLABLE void adj__compute_efc_eval_pt_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::int32 & adj_efcid,
    wp::float32 & adj_alpha,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_efc_D,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:218
static CUDA_CALLABLE void adj__eval_pt_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_alpha,
    wp::vec_t<3, wp::float32> & adj_quad,
    wp::float32 & adj_alpha,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:195
static CUDA_CALLABLE void adj__eval_pt_direct_shifted_3alphas_0(
    wp::float32 var_jaref,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::float32 var_offset,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & adj_jaref,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::float32 & adj_offset,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:270
static CUDA_CALLABLE void adj__eval_frictionloss_pt_one_0(
    wp::float32 var_x,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_half_d,
    wp::float32 var_jvD,
    wp::float32 var_hessian,
    wp::float32 var_f_jv,
    wp::float32 & adj_x,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_half_d,
    wp::float32 & adj_jvD,
    wp::float32 & adj_hessian,
    wp::float32 & adj_f_jv,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:281
static CUDA_CALLABLE void adj__eval_frictionloss_pt_3alphas_0(
    wp::float32 var_x_lo,
    wp::float32 var_x_hi,
    wp::float32 var_x_mid,
    wp::float32 var_f,
    wp::float32 var_rf,
    wp::float32 var_jv,
    wp::float32 var_d,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::float32 & adj_x_lo,
    wp::float32 & adj_x_hi,
    wp::float32 & adj_x_mid,
    wp::float32 & adj_f,
    wp::float32 & adj_rf,
    wp::float32 & adj_jv,
    wp::float32 & adj_d,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:672
static CUDA_CALLABLE void adj__compute_efc_eval_pt_3alphas_pyramidal_0(
    wp::int32 var_efcid,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::int32 var_ne,
    wp::int32 var_nf,
    wp::float32 var_efc_D,
    wp::array_t<wp::float32> var_efc_frictionloss,
    wp::float32 var_ctx_Jaref,
    wp::float32 var_ctx_jv,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::int32 & adj_efcid,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::int32 & adj_ne,
    wp::int32 & adj_nf,
    wp::float32 & adj_efc_D,
    wp::array_t<wp::float32> & adj_efc_frictionloss,
    wp::float32 & adj_ctx_Jaref,
    wp::float32 & adj_ctx_jv,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:229
static CUDA_CALLABLE void adj__eval_pt_3alphas_0(
    wp::vec_t<3, wp::float32> var_quad,
    wp::float32 var_lo_alpha,
    wp::float32 var_hi_alpha,
    wp::float32 var_mid_alpha,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_quad,
    wp::float32 & adj_lo_alpha,
    wp::float32 & adj_hi_alpha,
    wp::float32 & adj_mid_alpha,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\solver.py:143
static CUDA_CALLABLE void adj__in_bracket_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> var_y,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_y,
    bool & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _linesearch_iterative_kernel__locals__kernel_d4570990_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_tolerance,
    wp::array_t<wp::float32> var_opt_ls_tolerance,
    wp::array_t<wp::float32> var_opt_impratio_invsqrt,
    wp::array_t<wp::float32> var_stat_meaninertia,
    wp::array_t<wp::int32> var_ne_in,
    wp::array_t<wp::int32> var_nf_in,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_in,
    wp::array_t<wp::int32> var_contact_dim_in,
    wp::array_t<wp::int32> var_contact_efc_address_in,
    wp::array_t<wp::int32> var_efc_type_in,
    wp::array_t<wp::int32> var_efc_id_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_efc_frictionloss_in,
    wp::int32 var_njmax_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<bool> var_ctx_search_unchanged_in,
    wp::array_t<wp::float32> var_ctx_Jaref_in,
    wp::array_t<wp::float32> var_ctx_search_in,
    wp::array_t<wp::float32> var_ctx_search_dot_in,
    wp::array_t<wp::float32> var_ctx_mv_in,
    wp::array_t<wp::float32> var_ctx_jv_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_in,
    wp::array_t<bool> var_ctx_done_in,
    wp::array_t<wp::float32> var_qacc_out,
    wp::array_t<wp::float32> var_efc_Ma_out,
    wp::array_t<wp::int32> var_overflow_out,
    wp::array_t<wp::float32> var_ctx_Jaref_out,
    wp::array_t<wp::float32> var_ctx_jv_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ctx_quad_out,
    wp::array_t<wp::float32> var_ctx_improvement_out,
    wp::array_t<wp::float32> var_ctx_alpha_out,
    wp::array_t<bool> var_ctx_ls_exhausted_out)
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
        wp::int32 var_1;
        bool* var_2;
        bool var_3;
        bool var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const bool var_14 = true;
        const bool var_15 = true;
        const bool var_16 = true;
        bool* var_17;
        bool var_18;
        bool var_19;
        wp::int32 var_20;
        wp::range_t var_21;
        wp::int32 var_22;
        const wp::float32 var_23 = 0.0;
        wp::float32 var_24;
        const bool var_25 = false;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::float32* var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const bool var_34 = false;
        wp::shape_t* var_35;
        const wp::int32 var_36 = 0;
        wp::int32 var_37;
        wp::shape_t var_38;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::shape_t* var_43;
        const wp::int32 var_44 = 0;
        wp::int32 var_45;
        wp::shape_t var_46;
        wp::int32 var_47;
        wp::float32* var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32* var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::shape_t* var_54;
        const wp::int32 var_55 = 0;
        wp::int32 var_56;
        wp::shape_t var_57;
        wp::int32 var_58;
        wp::float32* var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        const wp::float32 var_67 = 1e-06;
        wp::float32 var_68;
        const wp::float32 var_69 = 0.0;
        wp::vec_t<2, wp::float32> var_70;
        wp::int32 var_71;
        wp::range_t var_72;
        wp::int32 var_73;
        wp::float32* var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32* var_77;
        wp::float32* var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.5;
        wp::float32 var_84;
        wp::float32* var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::vec_t<2, wp::float32> var_88;
        wp::vec_t<2, wp::float32> var_89;
        wp::int32 var_90;
        wp::range_t var_91;
        wp::int32 var_92;
        wp::float32* var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32* var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32* var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        const wp::float32 var_102 = 0.5;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::vec_t<2, wp::float32> var_105;
        wp::vec_t<2, wp::float32> var_106;
        const bool var_107 = true;
        wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_108 = wp::tile_register_t<wp::vec_t<2, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<2, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_109 = wp::tile_alloc_empty<wp::vec_t<2, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_110 = 0;
        wp::vec_t<2, wp::float32> var_111;
        const wp::float32 var_112 = 0.0;
        const wp::int32 var_113 = 0;
        wp::float32 var_114;
        const wp::int32 var_115 = 1;
        wp::float32 var_116;
        wp::vec_t<3, wp::float32> var_117;
        const wp::float32 var_118 = 0.0;
        wp::vec_t<3, wp::float32> var_119;
        wp::int32 var_120;
        wp::int32 var_121;
        wp::range_t var_122;
        wp::int32 var_123;
        const bool var_124 = false;
        wp::float32* var_125;
        wp::array_t<wp::float32> var_126;
        wp::float32* var_127;
        wp::float32* var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::vec_t<3, wp::float32> var_133;
        const bool var_134 = true;
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_135 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_136 = wp::tile_alloc_empty<wp::vec_t<3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_137 = 0;
        wp::float32 var_138;
        const wp::int32 var_139 = 1;
        wp::float32 var_140;
        const wp::float32 var_141 = 2.0;
        const wp::int32 var_142 = 2;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::vec_t<3, wp::float32> var_145;
        const wp::int32 var_146 = 0;
        wp::vec_t<3, wp::float32> var_147;
        wp::vec_t<3, wp::float32> var_148;
        const wp::float32 var_149 = 0.0;
        const wp::int32 var_150 = 1;
        wp::float32 var_151;
        const wp::int32 var_152 = 2;
        wp::float32 var_153;
        wp::vec_t<3, wp::float32> var_154;
        const wp::int32 var_155 = 0;
        wp::vec_t<3, wp::float32> var_156;
        const wp::float32 var_157 = 2.0;
        const wp::int32 var_158 = 0;
        wp::float32 var_159;
        const wp::float32 var_160 = 0.0;
        wp::float32 var_161;
        wp::float32 var_162;
        const wp::int32 var_163 = 2;
        wp::float32 var_164;
        const wp::float32 var_165 = 0.0;
        wp::float32 var_166;
        wp::float32 var_167;
        wp::float32 var_168;
        const wp::int32 var_169 = 1;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        const wp::float32 var_173 = 9.5367432e-07;
        wp::float32 var_174;
        wp::float32 var_175;
        const wp::float32 var_176 = 0.0;
        wp::float32 var_177;
        const bool var_178 = true;
        const wp::float32 var_179 = 1.0;
        const wp::int32 var_180 = 2;
        wp::float32 var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        const wp::float32 var_185 = 0.0;
        wp::float32 var_186;
        const wp::float32 var_187 = 0.0;
        wp::float32 var_188;
        const wp::int32 var_189 = 1;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::float32 var_192;
        bool var_193;
        const wp::int32 var_194 = 1;
        wp::float32 var_195;
        const wp::int32 var_196 = 2;
        wp::float32 var_197;
        wp::float32 var_198;
        wp::float32 var_199;
        const wp::float32 var_200 = 0.0;
        wp::vec_t<3, wp::float32> var_201;
        bool var_202;
        const wp::float32 var_203 = 0.0;
        wp::vec_t<3, wp::float32> var_204;
        wp::int32 var_205;
        wp::int32 var_206;
        wp::range_t var_207;
        wp::int32 var_208;
        const bool var_209 = false;
        wp::float32* var_210;
        wp::array_t<wp::float32> var_211;
        wp::float32* var_212;
        wp::float32* var_213;
        wp::vec_t<3, wp::float32> var_214;
        wp::float32 var_215;
        wp::float32 var_216;
        wp::float32 var_217;
        wp::vec_t<3, wp::float32> var_218;
        const bool var_219 = true;
        wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_220 = wp::tile_register_t<wp::vec_t<3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::vec_t<3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_221 = wp::tile_alloc_empty<wp::vec_t<3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        wp::vec_t<3, wp::float32> var_222;
        const wp::int32 var_223 = 0;
        wp::vec_t<3, wp::float32> var_224;
        wp::vec_t<3, wp::float32> var_225;
        bool var_226;
        const wp::int32 var_227 = 1;
        wp::float32 var_228;
        wp::float32 var_229;
        bool var_230;
        const wp::int32 var_231 = 0;
        wp::float32 var_232;
        const wp::float32 var_233 = 0.0;
        bool var_234;
        const bool var_235 = true;
        wp::float32 var_236;
        const wp::int32 var_237 = 0;
        wp::float32 var_238;
        wp::float32 var_239;
        wp::float32 var_240;
        wp::float32 var_241;
        bool var_242;
        wp::int32 var_243;
        wp::float32 var_244;
        wp::float32 var_245;
        bool var_246;
        wp::vec_t<3, wp::float32> var_247;
        bool var_248;
        const wp::int32 var_249 = 1;
        wp::float32 var_250;
        const wp::int32 var_251 = 1;
        wp::float32 var_252;
        bool var_253;
        wp::vec_t<3, wp::float32> var_254;
        const wp::float32 var_255 = 0.0;
        wp::float32 var_256;
        wp::vec_t<3, wp::float32> var_257;
        const wp::float32 var_258 = 0.0;
        wp::float32 var_259;
        const wp::int32 var_260 = 20;
        wp::range_t var_261;
        wp::int32 var_262;
        const wp::int32 var_263 = 1;
        wp::float32 var_264;
        const wp::int32 var_265 = 2;
        wp::float32 var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        const wp::int32 var_269 = 1;
        wp::float32 var_270;
        const wp::int32 var_271 = 2;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        const wp::float32 var_275 = 0.5;
        wp::float32 var_276;
        wp::float32 var_277;
        const wp::float32 var_278 = 0.0;
        wp::vec_t<3, wp::float32> var_279;
        const wp::float32 var_280 = 0.0;
        wp::vec_t<3, wp::float32> var_281;
        const wp::float32 var_282 = 0.0;
        wp::vec_t<3, wp::float32> var_283;
        wp::int32 var_284;
        wp::int32 var_285;
        wp::range_t var_286;
        wp::int32 var_287;
        const bool var_288 = false;
        wp::float32* var_289;
        wp::array_t<wp::float32> var_290;
        wp::float32* var_291;
        wp::float32* var_292;
        wp::vec_t<3, wp::float32> var_293;
        wp::vec_t<3, wp::float32> var_294;
        wp::vec_t<3, wp::float32> var_295;
        wp::float32 var_296;
        wp::float32 var_297;
        wp::float32 var_298;
        wp::vec_t<3, wp::float32> var_299;
        wp::vec_t<3, wp::float32> var_300;
        wp::vec_t<3, wp::float32> var_301;
        const wp::int32 var_302 = 0;
        wp::float32 var_303;
        const wp::int32 var_304 = 0;
        wp::float32 var_305;
        const wp::int32 var_306 = 0;
        wp::float32 var_307;
        const wp::int32 var_308 = 1;
        wp::float32 var_309;
        const wp::int32 var_310 = 1;
        wp::float32 var_311;
        const wp::int32 var_312 = 1;
        wp::float32 var_313;
        const wp::int32 var_314 = 2;
        wp::float32 var_315;
        const wp::int32 var_316 = 2;
        wp::float32 var_317;
        const wp::int32 var_318 = 2;
        wp::float32 var_319;
        wp::mat_t<3, 3, wp::float32> var_320;
        const bool var_321 = true;
        wp::tile_register_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>> var_322 = wp::tile_register_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_register_t<wp::tile_shape_t<32>>>{};
        wp::tile_shared_t<wp::mat_t<3, 3, wp::float32>,wp::tile_layout_strided_t<wp::tile_shape_t<1>, wp::tile_stride_t<1>>, true> var_323 = wp::tile_alloc_empty<wp::mat_t<3, 3, wp::float32>,wp::tile_shape_t<1>,wp::tile_stride_t<1>,false>();
        const wp::int32 var_324 = 0;
        wp::mat_t<3, 3, wp::float32> var_325;
        wp::vec_t<3, wp::float32> var_326;
        wp::vec_t<3, wp::float32> var_327;
        wp::vec_t<3, wp::float32> var_328;
        const wp::int32 var_329 = 0;
        const wp::int32 var_330 = 0;
        wp::float32 var_331;
        const wp::int32 var_332 = 1;
        const wp::int32 var_333 = 0;
        wp::float32 var_334;
        const wp::int32 var_335 = 2;
        const wp::int32 var_336 = 0;
        wp::float32 var_337;
        wp::vec_t<3, wp::float32> var_338;
        wp::vec_t<3, wp::float32> var_339;
        const wp::int32 var_340 = 0;
        const wp::int32 var_341 = 1;
        wp::float32 var_342;
        const wp::int32 var_343 = 1;
        const wp::int32 var_344 = 1;
        wp::float32 var_345;
        const wp::int32 var_346 = 2;
        const wp::int32 var_347 = 1;
        wp::float32 var_348;
        wp::vec_t<3, wp::float32> var_349;
        wp::vec_t<3, wp::float32> var_350;
        const wp::int32 var_351 = 0;
        const wp::int32 var_352 = 2;
        wp::float32 var_353;
        const wp::int32 var_354 = 1;
        const wp::int32 var_355 = 2;
        wp::float32 var_356;
        const wp::int32 var_357 = 2;
        const wp::int32 var_358 = 2;
        wp::float32 var_359;
        wp::vec_t<3, wp::float32> var_360;
        wp::vec_t<3, wp::float32> var_361;
        bool var_362;
        const wp::int32 var_363 = 1;
        wp::float32 var_364;
        wp::float32 var_365;
        bool var_366;
        const wp::int32 var_367 = 0;
        wp::float32 var_368;
        const wp::float32 var_369 = 0.0;
        bool var_370;
        bool var_371;
        const wp::int32 var_372 = 1;
        wp::float32 var_373;
        wp::float32 var_374;
        bool var_375;
        const wp::int32 var_376 = 0;
        wp::float32 var_377;
        const wp::float32 var_378 = 0.0;
        bool var_379;
        bool var_380;
        const wp::int32 var_381 = 1;
        wp::float32 var_382;
        wp::float32 var_383;
        bool var_384;
        const wp::int32 var_385 = 0;
        wp::float32 var_386;
        const wp::float32 var_387 = 0.0;
        bool var_388;
        bool var_389;
        const bool var_390 = false;
        const bool var_391 = false;
        const wp::float32 var_392 = 10000000000.0;
        const wp::float32 var_393 = 10000000000.0;
        const wp::float32 var_394 = 0.0;
        const wp::float32 var_395 = 0.0;
        wp::vec_t<3, wp::float32> var_396;
        const wp::float32 var_397 = 0.0;
        wp::float32 var_398;
        bool var_399;
        const wp::int32 var_400 = 0;
        wp::float32 var_401;
        const wp::int32 var_402 = 0;
        wp::float32 var_403;
        bool var_404;
        wp::vec_t<3, wp::float32> var_405;
        wp::float32 var_406;
        wp::vec_t<3, wp::float32> var_407;
        wp::float32 var_408;
        bool var_409;
        const wp::int32 var_410 = 0;
        wp::float32 var_411;
        const wp::int32 var_412 = 0;
        wp::float32 var_413;
        bool var_414;
        wp::vec_t<3, wp::float32> var_415;
        wp::float32 var_416;
        wp::vec_t<3, wp::float32> var_417;
        wp::float32 var_418;
        bool var_419;
        const wp::int32 var_420 = 0;
        wp::float32 var_421;
        const wp::int32 var_422 = 0;
        wp::float32 var_423;
        bool var_424;
        wp::vec_t<3, wp::float32> var_425;
        wp::float32 var_426;
        wp::vec_t<3, wp::float32> var_427;
        wp::float32 var_428;
        wp::vec_t<3, wp::float32> var_429;
        wp::float32 var_430;
        wp::vec_t<3, wp::float32> var_431;
        wp::float32 var_432;
        bool var_433;
        wp::vec_t<3, wp::float32> var_434;
        wp::float32 var_435;
        bool var_436;
        wp::vec_t<3, wp::float32> var_437;
        wp::float32 var_438;
        bool var_439;
        wp::vec_t<3, wp::float32> var_440;
        wp::float32 var_441;
        bool var_442;
        bool var_443;
        bool var_444;
        bool var_445;
        const wp::int32 var_446 = 1;
        wp::float32 var_447;
        const wp::float32 var_448 = 0.0;
        bool var_449;
        const wp::int32 var_450 = 1;
        wp::float32 var_451;
        const wp::float32 var_452 = 0.0;
        bool var_453;
        wp::vec_t<3, wp::float32> var_454;
        wp::float32 var_455;
        bool var_456;
        wp::vec_t<3, wp::float32> var_457;
        wp::float32 var_458;
        bool var_459;
        wp::vec_t<3, wp::float32> var_460;
        wp::float32 var_461;
        bool var_462;
        wp::vec_t<3, wp::float32> var_463;
        wp::float32 var_464;
        wp::vec_t<3, wp::float32> var_465;
        wp::float32 var_466;
        bool var_467;
        bool var_468;
        bool var_469;
        bool var_470;
        bool var_471;
        bool var_472;
        bool var_473;
        const wp::int32 var_474 = 0;
        wp::float32 var_475;
        const wp::float32 var_476 = 0.0;
        bool var_477;
        const wp::int32 var_478 = 1;
        wp::float32 var_479;
        const wp::float32 var_480 = 0.0;
        bool var_481;
        const wp::int32 var_482 = 1;
        wp::float32 var_483;
        wp::float32 var_484;
        bool var_485;
        bool var_486;
        const wp::int32 var_487 = 0;
        wp::float32 var_488;
        const wp::float32 var_489 = 0.0;
        bool var_490;
        const wp::int32 var_491 = 1;
        wp::float32 var_492;
        const wp::float32 var_493 = 0.0;
        bool var_494;
        const wp::int32 var_495 = 1;
        wp::float32 var_496;
        bool var_497;
        bool var_498;
        const wp::int32 var_499 = 0;
        wp::float32 var_500;
        const wp::float32 var_501 = 0.0;
        bool var_502;
        const wp::int32 var_503 = 0;
        wp::float32 var_504;
        const wp::float32 var_505 = 0.0;
        bool var_506;
        const wp::int32 var_507 = 0;
        wp::float32 var_508;
        const wp::int32 var_509 = 0;
        wp::float32 var_510;
        bool var_511;
        wp::float32 var_512;
        const wp::int32 var_513 = 0;
        wp::float32 var_514;
        const wp::int32 var_515 = 0;
        wp::float32 var_516;
        wp::float32 var_517;
        wp::float32 var_518;
        wp::float32 var_519;
        wp::float32 var_520;
        const bool var_521 = true;
        wp::int32 var_522;
        wp::float32 var_523;
        wp::float32 var_524;
        wp::vec_t<3, wp::float32> var_525;
        wp::float32 var_526;
        wp::vec_t<3, wp::float32> var_527;
        wp::float32 var_528;
        wp::int32 var_529;
        wp::range_t var_530;
        wp::int32 var_531;
        wp::float32* var_532;
        wp::float32 var_533;
        wp::float32 var_534;
        wp::float32 var_535;
        wp::float32* var_536;
        wp::float32 var_537;
        wp::float32 var_538;
        wp::float32 var_539;
        wp::int32 var_540;
        wp::range_t var_541;
        wp::int32 var_542;
        wp::float32* var_543;
        wp::float32 var_544;
        wp::float32 var_545;
        wp::float32 var_546;
        const wp::int32 var_547 = 0;
        bool var_548;
        const bool var_549 = true;
        wp::float32 var_550;
        bool var_551;
        bool var_552;
        const bool var_553 = false;
        const wp::int32 var_554 = 1024;
        wp::int32 var_555;
        //---------
        // forward
        // def kernel(                                                                            <L 899>
        // worldid, tid = wp.tid()                                                                <L 945>
        builtin_tid2d(var_0, var_1);
        // if ctx_done_in[worldid]:                                                               <L 947>
        var_2 = wp::address(var_ctx_done_in, var_0);
        var_3 = wp::load(var_2);
        if (var_3) {
            // return                                                                             <L 948>
            continue;
        }
        var_4 = wp::load(var_2);
        // ne = ne_in[worldid]                                                                    <L 950>
        var_5 = wp::address(var_ne_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // nf = nf_in[worldid]                                                                    <L 951>
        var_8 = wp::address(var_nf_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // nefc = wp.min(njmax_in, nefc_in[worldid])                                              <L 952>
        var_11 = wp::address(var_nefc_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::min(var_njmax_in, var_13);
        // if wp.static(FUSE_JV):                                                                 <L 956>
        // recompute_jv = True                                                                    <L 957>
        // if wp.static(INCREMENTAL):                                                             <L 958>
        // recompute_jv = not ctx_search_unchanged_in[worldid]                                    <L 959>
        var_17 = wp::address(var_ctx_search_unchanged_in, var_0);
        var_19 = wp::load(var_17);
        var_18 = wp::unot(var_19);
        // if recompute_jv:                                                                       <L 960>
        if (var_18) {
            // for efcid in range(tid, nefc, wp.block_dim()):                                     <L 961>
            var_20 = builtin_block_dim();
            var_21 = wp::range(var_1, var_12, var_20);
            start_for_1:;
                if (iter_cmp(var_21) == 0) goto end_for_1;
                var_22 = wp::iter_next(var_21);
                // jv = float(0.0)                                                                <L 962>
                var_24 = wp::float(var_23);
                // if wp.static(IS_SPARSE):                                                       <L 963>
                // for i in range(nv):                                                            <L 971>
                var_26 = wp::range(var_nv);
                start_for_3:;
                    if (iter_cmp(var_26) == 0) goto end_for_3;
                    var_27 = wp::iter_next(var_26);
                    // jv += efc_J_in[worldid, efcid, i] * ctx_search_in[worldid, i]              <L 972>
                    var_28 = wp::address(var_efc_J_in, var_0, var_22, var_27);
                    var_29 = wp::address(var_ctx_search_in, var_0, var_27);
                    var_31 = wp::load(var_28);
                    var_32 = wp::load(var_29);
                    var_30 = wp::mul(var_31, var_32);
                    var_33 = wp::add(var_24, var_30);
                    wp::assign(var_24, var_33);
                    goto start_for_3;
                end_for_3:;
                // ctx_jv_out[worldid, efcid] = jv                                                <L 973>
                wp::array_store(var_ctx_jv_out, var_0, var_22, var_24);
                goto start_for_1;
            end_for_1:;
        }
        // _syncthreads()  # ensure all jv values are written before reading                      <L 975>
        _linesearch_iterative_kernel__locals___syncthreads_0();
        // if wp.static(IS_ELLIPTIC):                                                             <L 980>
        // tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]                            <L 1041>
        var_35 = &(var_opt_tolerance.shape);
        var_38 = wp::load(var_35);
        var_37 = wp::extract(var_38, var_36);
        var_39 = wp::mod(var_0, var_37);
        var_40 = wp::address(var_opt_tolerance, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]                   <L 1042>
        var_43 = &(var_opt_ls_tolerance.shape);
        var_46 = wp::load(var_43);
        var_45 = wp::extract(var_46, var_44);
        var_47 = wp::mod(var_0, var_45);
        var_48 = wp::address(var_opt_ls_tolerance, var_47);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // snorm = wp.sqrt(ctx_search_dot_in[worldid])                                            <L 1043>
        var_51 = wp::address(var_ctx_search_dot_in, var_0);
        var_53 = wp::load(var_51);
        var_52 = wp::sqrt(var_53);
        // meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]                    <L 1044>
        var_54 = &(var_stat_meaninertia.shape);
        var_57 = wp::load(var_54);
        var_56 = wp::extract(var_57, var_55);
        var_58 = wp::mod(var_0, var_56);
        var_59 = wp::address(var_stat_meaninertia, var_58);
        var_61 = wp::load(var_59);
        var_60 = wp::copy(var_61);
        // scale = meaninertia * wp.float(nv)                                                     <L 1045>
        var_62 = wp::float(var_nv);
        var_63 = wp::mul(var_60, var_62);
        // gtol = wp.max(tolerance * ls_tolerance * snorm * scale, 1e-6)                          <L 1046>
        var_64 = wp::mul(var_41, var_49);
        var_65 = wp::mul(var_64, var_52);
        var_66 = wp::mul(var_65, var_63);
        var_68 = wp::max(var_66, var_67);
        // local_gauss = wp.vec2(0.0)                                                             <L 1049>
        var_70 = wp::vec_t<2, wp::float32>(var_69);
        // for dofid in range(tid, nv, wp.block_dim()):                                           <L 1050>
        var_71 = builtin_block_dim();
        var_72 = wp::range(var_1, var_nv, var_71);
        start_for_5:;
            if (iter_cmp(var_72) == 0) goto end_for_5;
            var_73 = wp::iter_next(var_72);
            // search = ctx_search_in[worldid, dofid]                                             <L 1051>
            var_74 = wp::address(var_ctx_search_in, var_0, var_73);
            var_76 = wp::load(var_74);
            var_75 = wp::copy(var_76);
            // local_gauss += wp.vec2(                                                            <L 1052>
            // search * (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]),            <L 1053>
            var_77 = wp::address(var_efc_Ma_out, var_0, var_73);
            var_78 = wp::address(var_qfrc_smooth_in, var_0, var_73);
            var_80 = wp::load(var_77);
            var_81 = wp::load(var_78);
            var_79 = wp::sub(var_80, var_81);
            var_82 = wp::mul(var_75, var_79);
            // 0.5 * search * ctx_mv_in[worldid, dofid],                                          <L 1054>
            var_84 = wp::mul(var_83, var_75);
            var_85 = wp::address(var_ctx_mv_in, var_0, var_73);
            var_87 = wp::load(var_85);
            var_86 = wp::mul(var_84, var_87);
            var_88 = wp::vec_t<2, wp::float32>(var_82, var_86);
            // local_gauss += wp.vec2(                                                            <L 1052>
            var_89 = wp::add(var_70, var_88);
            wp::assign(var_70, var_89);
            goto start_for_5;
        end_for_5:;
        // for efcid in range(tid, ne, wp.block_dim()):                                           <L 1057>
        var_90 = builtin_block_dim();
        var_91 = wp::range(var_1, var_6, var_90);
        start_for_7:;
            if (iter_cmp(var_91) == 0) goto end_for_7;
            var_92 = wp::iter_next(var_91);
            // jv = ctx_jv_in[worldid, efcid]                                                     <L 1058>
            var_93 = wp::address(var_ctx_jv_in, var_0, var_92);
            var_95 = wp::load(var_93);
            var_94 = wp::copy(var_95);
            // jvD = jv * efc_D_in[worldid, efcid]                                                <L 1059>
            var_96 = wp::address(var_efc_D_in, var_0, var_92);
            var_98 = wp::load(var_96);
            var_97 = wp::mul(var_94, var_98);
            // local_gauss += wp.vec2(                                                            <L 1060>
            // jvD * ctx_Jaref_in[worldid, efcid],                                                <L 1061>
            var_99 = wp::address(var_ctx_Jaref_in, var_0, var_92);
            var_101 = wp::load(var_99);
            var_100 = wp::mul(var_97, var_101);
            // 0.5 * jv * jvD,                                                                    <L 1062>
            var_103 = wp::mul(var_102, var_94);
            var_104 = wp::mul(var_103, var_97);
            var_105 = wp::vec_t<2, wp::float32>(var_100, var_104);
            // local_gauss += wp.vec2(                                                            <L 1060>
            var_106 = wp::add(var_70, var_105);
            wp::assign(var_24, var_94);
            wp::assign(var_70, var_106);
            goto start_for_7;
        end_for_7:;
        // gauss_tile = wp.tile(local_gauss, preserve_type=True)                                  <L 1065>
        var_108 = wp::tile<wp::vec_t<2, wp::float32>>(var_70);
        // gauss_sum = wp.tile_reduce(wp.add, gauss_tile)                                         <L 1066>
        var_109 = wp::tile_reduce(wp::add, var_108);
        // gauss_reduced = gauss_sum[0]                                                           <L 1067>
        var_111 = wp::tile_extract(var_109, var_110);
        // ctx_quad_gauss = wp.vec3(0.0, gauss_reduced[0], gauss_reduced[1])                      <L 1068>
        var_114 = wp::extract(var_111, var_113);
        var_116 = wp::extract(var_111, var_115);
        var_117 = wp::vec_t<3, wp::float32>(var_112, var_114, var_116);
        // local_p0 = wp.vec3(0.0)                                                                <L 1071>
        var_119 = wp::vec_t<3, wp::float32>(var_118);
        // for efcid in range(ne + tid, nefc, wp.block_dim()):                                    <L 1072>
        var_120 = wp::add(var_6, var_1);
        var_121 = builtin_block_dim();
        var_122 = wp::range(var_120, var_12, var_121);
        start_for_9:;
            if (iter_cmp(var_122) == 0) goto end_for_9;
            var_123 = wp::iter_next(var_122);
            // if wp.static(IS_ELLIPTIC):                                                         <L 1073>
            // local_p0 += _compute_efc_eval_pt_alpha_zero(                                       <L 1110>
            // efcid,                                                                             <L 1111>
            // ne,                                                                                <L 1112>
            // nf,                                                                                <L 1113>
            // efc_D_in[worldid, efcid],                                                          <L 1114>
            var_125 = wp::address(var_efc_D_in, var_0, var_123);
            // efc_frictionloss_in[worldid],                                                      <L 1115>
            var_126 = wp::view(var_efc_frictionloss_in, var_0);
            // ctx_Jaref_in[worldid, efcid],                                                      <L 1116>
            var_127 = wp::address(var_ctx_Jaref_in, var_0, var_123);
            // ctx_jv_in[worldid, efcid],                                                         <L 1117>
            var_128 = wp::address(var_ctx_jv_in, var_0, var_123);
            var_130 = wp::load(var_125);
            var_131 = wp::load(var_127);
            var_132 = wp::load(var_128);
            var_129 = _compute_efc_eval_pt_alpha_zero_pyramidal_0(var_123, var_6, var_9, var_130, var_126, var_131, var_132);
            // local_p0 += _compute_efc_eval_pt_alpha_zero(                                       <L 1110>
            var_133 = wp::add(var_119, var_129);
            wp::assign(var_119, var_133);
            goto start_for_9;
        end_for_9:;
        // p0_tile = wp.tile(local_p0, preserve_type=True)                                        <L 1123>
        var_135 = wp::tile<wp::vec_t<3, wp::float32>>(var_119);
        // p0_sum = wp.tile_reduce(wp.add, p0_tile)                                               <L 1124>
        var_136 = wp::tile_reduce(wp::add, var_135);
        // p0 = wp.vec3(ctx_quad_gauss[0], ctx_quad_gauss[1], 2.0 * ctx_quad_gauss[2]) + p0_sum[0]       <L 1127>
        var_138 = wp::extract(var_117, var_137);
        var_140 = wp::extract(var_117, var_139);
        var_143 = wp::extract(var_117, var_142);
        var_144 = wp::mul(var_141, var_143);
        var_145 = wp::vec_t<3, wp::float32>(var_138, var_140, var_144);
        var_147 = wp::tile_extract(var_136, var_146);
        var_148 = wp::add(var_145, var_147);
        // p0_delta = wp.vec3(0.0, p0[1], p0[2])                                                  <L 1128>
        var_151 = wp::extract(var_148, var_150);
        var_153 = wp::extract(var_148, var_152);
        var_154 = wp::vec_t<3, wp::float32>(var_149, var_151, var_153);
        // rows = p0_sum[0]                                                                       <L 1138>
        var_156 = wp::tile_extract(var_136, var_155);
        // q1_abs = wp.sqrt(2.0 * wp.max(rows[0], 0.0) * wp.max(rows[2], 0.0)) + wp.abs(ctx_quad_gauss[1])       <L 1139>
        var_159 = wp::extract(var_156, var_158);
        var_161 = wp::max(var_159, var_160);
        var_162 = wp::mul(var_157, var_161);
        var_164 = wp::extract(var_156, var_163);
        var_166 = wp::max(var_164, var_165);
        var_167 = wp::mul(var_162, var_166);
        var_168 = wp::sqrt(var_167);
        var_170 = wp::extract(var_117, var_169);
        var_171 = wp::abs(var_170);
        var_172 = wp::add(var_168, var_171);
        // gtol_accept = wp.max(gtol, _ALPHA_NOISE_EPS * q1_abs)                                  <L 1142>
        var_174 = wp::mul(var_173, var_172);
        var_175 = wp::max(var_68, var_174);
        // noise_floor = float(0.0)                                                               <L 1144>
        var_177 = wp::float(var_176);
        // if wp.static(INCREMENTAL):                                                             <L 1145>
        // noise_floor = _ALPHA_NOISE_EPS * wp.max(1.0, math.safe_div(q1_abs, p0[2]))             <L 1146>
        var_181 = wp::extract(var_148, var_180);
        var_182 = safe_div_0(var_172, var_181);
        var_183 = wp::max(var_179, var_182);
        var_184 = wp::mul(var_173, var_183);
        // alpha = float(0.0)                                                                     <L 1148>
        var_186 = wp::float(var_185);
        // improvement = float(0.0)                                                               <L 1149>
        var_188 = wp::float(var_187);
        // ls_converged = wp.abs(p0[1]) < _ALPHA_NOISE_EPS * q1_abs                               <L 1152>
        var_190 = wp::extract(var_148, var_189);
        var_191 = wp::abs(var_190);
        var_192 = wp::mul(var_173, var_172);
        var_193 = (var_191 < var_192);
        // lo_alpha_in = -math.safe_div(p0[1], p0[2])                                             <L 1155>
        var_195 = wp::extract(var_148, var_194);
        var_197 = wp::extract(var_148, var_196);
        var_198 = safe_div_0(var_195, var_197);
        var_199 = wp::neg(var_198);
        // lo_in = wp.vec3(0.0)                                                                   <L 1156>
        var_201 = wp::vec_t<3, wp::float32>(var_200);
        // if not ls_converged:                                                                   <L 1158>
        var_202 = wp::unot(var_193);
        if (var_202) {
            // local_lo_in = wp.vec3(0.0)                                                         <L 1159>
            var_204 = wp::vec_t<3, wp::float32>(var_203);
            // for efcid in range(ne + tid, nefc, wp.block_dim()):                                <L 1160>
            var_205 = wp::add(var_6, var_1);
            var_206 = builtin_block_dim();
            var_207 = wp::range(var_205, var_12, var_206);
            start_for_11:;
                if (iter_cmp(var_207) == 0) goto end_for_11;
                var_208 = wp::iter_next(var_207);
                // if wp.static(IS_ELLIPTIC):                                                     <L 1161>
                // local_lo_in += _compute_efc_eval_pt(                                           <L 1199>
                // efcid,                                                                         <L 1200>
                // lo_alpha_in,                                                                   <L 1201>
                // ne,                                                                            <L 1202>
                // nf,                                                                            <L 1203>
                // efc_D_in[worldid, efcid],                                                      <L 1204>
                var_210 = wp::address(var_efc_D_in, var_0, var_208);
                // efc_frictionloss_in[worldid],                                                  <L 1205>
                var_211 = wp::view(var_efc_frictionloss_in, var_0);
                // ctx_Jaref_in[worldid, efcid],                                                  <L 1206>
                var_212 = wp::address(var_ctx_Jaref_in, var_0, var_208);
                // ctx_jv_in[worldid, efcid],                                                     <L 1207>
                var_213 = wp::address(var_ctx_jv_in, var_0, var_208);
                var_215 = wp::load(var_210);
                var_216 = wp::load(var_212);
                var_217 = wp::load(var_213);
                var_214 = _compute_efc_eval_pt_pyramidal_0(var_208, var_199, var_6, var_9, var_215, var_211, var_216, var_217);
                // local_lo_in += _compute_efc_eval_pt(                                           <L 1199>
                var_218 = wp::add(var_204, var_214);
                wp::assign(var_204, var_218);
                goto start_for_11;
            end_for_11:;
            // lo_in_tile = wp.tile(local_lo_in, preserve_type=True)                              <L 1210>
            var_220 = wp::tile<wp::vec_t<3, wp::float32>>(var_204);
            // lo_in_sum = wp.tile_reduce(wp.add, lo_in_tile)                                     <L 1211>
            var_221 = wp::tile_reduce(wp::add, var_220);
            // lo_in = _eval_pt(ctx_quad_gauss, lo_alpha_in) + lo_in_sum[0]                       <L 1212>
            var_222 = _eval_pt_0(var_117, var_199);
            var_224 = wp::tile_extract(var_221, var_223);
            var_225 = wp::add(var_222, var_224);
            // if wp.abs(lo_in[1]) < gtol_accept and lo_in[0] < 0.0:                              <L 1215>
            var_228 = wp::extract(var_225, var_227);
            var_229 = wp::abs(var_228);
            var_230 = (var_229 < var_175);
            var_226 = var_230;
            if (var_226) {
                var_232 = wp::extract(var_225, var_231);
                var_234 = (var_232 < var_233);
                var_226 = var_226 && var_234;
            }
            if (var_226) {
                // ls_converged = True                                                            <L 1216>
                // alpha = lo_alpha_in                                                            <L 1217>
                var_236 = wp::copy(var_199);
                // improvement = -lo_in[0]                                                        <L 1218>
                var_238 = wp::extract(var_225, var_237);
                var_239 = wp::neg(var_238);
            }
            var_240 = wp::where(var_226, var_236, var_186);
            var_241 = wp::where(var_226, var_239, var_188);
            var_242 = wp::where(var_226, var_235, var_193);
        }
        var_243 = wp::where(var_202, var_208, var_123);
        var_244 = wp::where(var_202, var_240, var_186);
        var_245 = wp::where(var_202, var_241, var_188);
        var_246 = wp::where(var_202, var_242, var_193);
        var_247 = wp::where(var_202, var_225, var_201);
        // if not ls_converged:                                                                   <L 1221>
        var_248 = wp::unot(var_246);
        if (var_248) {
            // lo_less = lo_in[1] < p0[1]                                                         <L 1223>
            var_250 = wp::extract(var_247, var_249);
            var_252 = wp::extract(var_148, var_251);
            var_253 = (var_250 < var_252);
            // lo = wp.where(lo_less, lo_in, p0_delta)                                            <L 1224>
            var_254 = wp::where(var_253, var_247, var_154);
            // lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)                                     <L 1225>
            var_256 = wp::where(var_253, var_199, var_255);
            // hi = wp.where(lo_less, p0_delta, lo_in)                                            <L 1226>
            var_257 = wp::where(var_253, var_154, var_247);
            // hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)                                     <L 1227>
            var_259 = wp::where(var_253, var_258, var_199);
            // for _ in range(LS_ITERATIONS):                                                     <L 1229>
            var_261 = wp::range(var_260);
            start_for_13:;
                if (iter_cmp(var_261) == 0) goto end_for_13;
                var_262 = wp::iter_next(var_261);
                // lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])                         <L 1230>
                var_264 = wp::extract(var_254, var_263);
                var_266 = wp::extract(var_254, var_265);
                var_267 = safe_div_0(var_264, var_266);
                var_268 = wp::sub(var_256, var_267);
                // hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])                         <L 1231>
                var_270 = wp::extract(var_257, var_269);
                var_272 = wp::extract(var_257, var_271);
                var_273 = safe_div_0(var_270, var_272);
                var_274 = wp::sub(var_259, var_273);
                // mid_alpha = 0.5 * (lo_alpha + hi_alpha)                                        <L 1232>
                var_276 = wp::add(var_256, var_259);
                var_277 = wp::mul(var_275, var_276);
                // local_lo = wp.vec3(0.0)                                                        <L 1234>
                var_279 = wp::vec_t<3, wp::float32>(var_278);
                // local_hi = wp.vec3(0.0)                                                        <L 1235>
                var_281 = wp::vec_t<3, wp::float32>(var_280);
                // local_mid = wp.vec3(0.0)                                                       <L 1236>
                var_283 = wp::vec_t<3, wp::float32>(var_282);
                // for efcid in range(ne + tid, nefc, wp.block_dim()):                            <L 1238>
                var_284 = wp::add(var_6, var_1);
                var_285 = builtin_block_dim();
                var_286 = wp::range(var_284, var_12, var_285);
                start_for_15:;
                    if (iter_cmp(var_286) == 0) goto end_for_15;
                    var_287 = wp::iter_next(var_286);
                    // if wp.static(IS_ELLIPTIC):                                                 <L 1239>
                    // r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(                          <L 1279>
                    // efcid,                                                                     <L 1280>
                    // lo_next_alpha,                                                             <L 1281>
                    // hi_next_alpha,                                                             <L 1282>
                    // mid_alpha,                                                                 <L 1283>
                    // ne,                                                                        <L 1284>
                    // nf,                                                                        <L 1285>
                    // efc_D_in[worldid, efcid],                                                  <L 1286>
                    var_289 = wp::address(var_efc_D_in, var_0, var_287);
                    // efc_frictionloss_in[worldid],                                              <L 1287>
                    var_290 = wp::view(var_efc_frictionloss_in, var_0);
                    // ctx_Jaref_in[worldid, efcid],                                              <L 1288>
                    var_291 = wp::address(var_ctx_Jaref_in, var_0, var_287);
                    // ctx_jv_in[worldid, efcid],                                                 <L 1289>
                    var_292 = wp::address(var_ctx_jv_in, var_0, var_287);
                    var_296 = wp::load(var_289);
                    var_297 = wp::load(var_291);
                    var_298 = wp::load(var_292);
                    _compute_efc_eval_pt_3alphas_pyramidal_0(var_287, var_268, var_274, var_277, var_6, var_9, var_296, var_290, var_297, var_298, var_293, var_294, var_295);
                    // local_lo += r_lo                                                           <L 1291>
                    var_299 = wp::add(var_279, var_293);
                    // local_hi += r_hi                                                           <L 1292>
                    var_300 = wp::add(var_281, var_294);
                    // local_mid += r_mid                                                         <L 1293>
                    var_301 = wp::add(var_283, var_295);
                    wp::assign(var_279, var_299);
                    wp::assign(var_281, var_300);
                    wp::assign(var_283, var_301);
                    goto start_for_15;
                end_for_15:;
                // local_combined = wp.mat33(                                                     <L 1296>
                // local_lo[0],                                                                   <L 1297>
                var_303 = wp::extract(var_279, var_302);
                // local_hi[0],                                                                   <L 1298>
                var_305 = wp::extract(var_281, var_304);
                // local_mid[0],                                                                  <L 1299>
                var_307 = wp::extract(var_283, var_306);
                // local_lo[1],                                                                   <L 1300>
                var_309 = wp::extract(var_279, var_308);
                // local_hi[1],                                                                   <L 1301>
                var_311 = wp::extract(var_281, var_310);
                // local_mid[1],                                                                  <L 1302>
                var_313 = wp::extract(var_283, var_312);
                // local_lo[2],                                                                   <L 1303>
                var_315 = wp::extract(var_279, var_314);
                // local_hi[2],                                                                   <L 1304>
                var_317 = wp::extract(var_281, var_316);
                // local_mid[2],                                                                  <L 1305>
                var_319 = wp::extract(var_283, var_318);
                var_320 = wp::mat_t<3, 3, wp::float32>(var_303, var_305, var_307, var_309, var_311, var_313, var_315, var_317, var_319);
                // combined_tile = wp.tile(local_combined, preserve_type=True)                    <L 1310>
                var_322 = wp::tile<wp::mat_t<3, 3, wp::float32>>(var_320);
                // combined_sum = wp.tile_reduce(wp.add, combined_tile)                           <L 1311>
                var_323 = wp::tile_reduce(wp::add, var_322);
                // result = combined_sum[0]                                                       <L 1312>
                var_325 = wp::tile_extract(var_323, var_324);
                // gauss_lo, gauss_hi, gauss_mid = _eval_pt_3alphas(ctx_quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha)       <L 1315>
                _eval_pt_3alphas_0(var_117, var_268, var_274, var_277, var_326, var_327, var_328);
                // lo_next = gauss_lo + wp.vec3(result[0, 0], result[1, 0], result[2, 0])         <L 1316>
                var_331 = wp::extract(var_325, var_329, var_330);
                var_334 = wp::extract(var_325, var_332, var_333);
                var_337 = wp::extract(var_325, var_335, var_336);
                var_338 = wp::vec_t<3, wp::float32>(var_331, var_334, var_337);
                var_339 = wp::add(var_326, var_338);
                // hi_next = gauss_hi + wp.vec3(result[0, 1], result[1, 1], result[2, 1])         <L 1317>
                var_342 = wp::extract(var_325, var_340, var_341);
                var_345 = wp::extract(var_325, var_343, var_344);
                var_348 = wp::extract(var_325, var_346, var_347);
                var_349 = wp::vec_t<3, wp::float32>(var_342, var_345, var_348);
                var_350 = wp::add(var_327, var_349);
                // mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])            <L 1318>
                var_353 = wp::extract(var_325, var_351, var_352);
                var_356 = wp::extract(var_325, var_354, var_355);
                var_359 = wp::extract(var_325, var_357, var_358);
                var_360 = wp::vec_t<3, wp::float32>(var_353, var_356, var_359);
                var_361 = wp::add(var_328, var_360);
                // conv_lo = wp.abs(lo_next[1]) < gtol_accept and lo_next[0] < 0.0                <L 1321>
                var_364 = wp::extract(var_339, var_363);
                var_365 = wp::abs(var_364);
                var_366 = (var_365 < var_175);
                var_362 = var_366;
                if (var_362) {
                    var_368 = wp::extract(var_339, var_367);
                    var_370 = (var_368 < var_369);
                    var_362 = var_362 && var_370;
                }
                // conv_hi = wp.abs(hi_next[1]) < gtol_accept and hi_next[0] < 0.0                <L 1322>
                var_373 = wp::extract(var_350, var_372);
                var_374 = wp::abs(var_373);
                var_375 = (var_374 < var_175);
                var_371 = var_375;
                if (var_371) {
                    var_377 = wp::extract(var_350, var_376);
                    var_379 = (var_377 < var_378);
                    var_371 = var_371 && var_379;
                }
                // conv_mid = wp.abs(mid[1]) < gtol_accept and mid[0] < 0.0                       <L 1323>
                var_382 = wp::extract(var_361, var_381);
                var_383 = wp::abs(var_382);
                var_384 = (var_383 < var_175);
                var_380 = var_384;
                if (var_380) {
                    var_386 = wp::extract(var_361, var_385);
                    var_388 = (var_386 < var_387);
                    var_380 = var_380 && var_388;
                }
                // converged = conv_lo or conv_hi or conv_mid                                     <L 1324>
                var_389 = var_362;
                if (!var_389) {
                    var_389 = var_389 || var_371;
                }
                if (!var_389) {
                    var_389 = var_389 || var_380;
                }
                // swap_lo = False                                                                <L 1325>
                // swap_hi = False                                                                <L 1326>
                // if converged:                                                                  <L 1327>
                if (var_389) {
                    // conv_pt = wp.vec3(types.MJ_MAXVAL, 0.0, 0.0)                               <L 1328>
                    var_396 = wp::vec_t<3, wp::float32>(var_393, var_394, var_395);
                    // conv_alpha = float(0.0)                                                    <L 1329>
                    var_398 = wp::float(var_397);
                    // if conv_lo and lo_next[0] < conv_pt[0]:                                    <L 1330>
                    var_399 = var_362;
                    if (var_399) {
                        var_401 = wp::extract(var_339, var_400);
                        var_403 = wp::extract(var_396, var_402);
                        var_404 = (var_401 < var_403);
                        var_399 = var_399 && var_404;
                    }
                    if (var_399) {
                        // conv_pt = lo_next                                                      <L 1331>
                        var_405 = wp::copy(var_339);
                        // conv_alpha = lo_next_alpha                                             <L 1332>
                        var_406 = wp::copy(var_268);
                    }
                    var_407 = wp::where(var_399, var_405, var_396);
                    var_408 = wp::where(var_399, var_406, var_398);
                    // if conv_hi and hi_next[0] < conv_pt[0]:                                    <L 1333>
                    var_409 = var_371;
                    if (var_409) {
                        var_411 = wp::extract(var_350, var_410);
                        var_413 = wp::extract(var_407, var_412);
                        var_414 = (var_411 < var_413);
                        var_409 = var_409 && var_414;
                    }
                    if (var_409) {
                        // conv_pt = hi_next                                                      <L 1334>
                        var_415 = wp::copy(var_350);
                        // conv_alpha = hi_next_alpha                                             <L 1335>
                        var_416 = wp::copy(var_274);
                    }
                    var_417 = wp::where(var_409, var_415, var_407);
                    var_418 = wp::where(var_409, var_416, var_408);
                    // if conv_mid and mid[0] < conv_pt[0]:                                       <L 1336>
                    var_419 = var_380;
                    if (var_419) {
                        var_421 = wp::extract(var_361, var_420);
                        var_423 = wp::extract(var_417, var_422);
                        var_424 = (var_421 < var_423);
                        var_419 = var_419 && var_424;
                    }
                    if (var_419) {
                        // conv_pt = mid                                                          <L 1337>
                        var_425 = wp::copy(var_361);
                        // conv_alpha = mid_alpha                                                 <L 1338>
                        var_426 = wp::copy(var_277);
                    }
                    var_427 = wp::where(var_419, var_425, var_417);
                    var_428 = wp::where(var_419, var_426, var_418);
                    // lo = conv_pt                                                               <L 1339>
                    var_429 = wp::copy(var_427);
                    // lo_alpha = conv_alpha                                                      <L 1340>
                    var_430 = wp::copy(var_428);
                    // hi = conv_pt                                                               <L 1341>
                    var_431 = wp::copy(var_427);
                    // hi_alpha = conv_alpha                                                      <L 1342>
                    var_432 = wp::copy(var_428);
                }
                if (!var_389) {
                    // swap_lo_lo_next = _in_bracket(lo, lo_next)                                 <L 1346>
                    var_433 = _in_bracket_0(var_254, var_339);
                    // lo = wp.where(swap_lo_lo_next, lo_next, lo)                                <L 1347>
                    var_434 = wp::where(var_433, var_339, var_254);
                    // lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)              <L 1348>
                    var_435 = wp::where(var_433, var_268, var_256);
                    // swap_lo_mid = _in_bracket(lo, mid)                                         <L 1349>
                    var_436 = _in_bracket_0(var_434, var_361);
                    // lo = wp.where(swap_lo_mid, mid, lo)                                        <L 1350>
                    var_437 = wp::where(var_436, var_361, var_434);
                    // lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)                      <L 1351>
                    var_438 = wp::where(var_436, var_277, var_435);
                    // swap_lo_hi_next = _in_bracket(lo, hi_next)                                 <L 1352>
                    var_439 = _in_bracket_0(var_437, var_350);
                    // lo = wp.where(swap_lo_hi_next, hi_next, lo)                                <L 1353>
                    var_440 = wp::where(var_439, var_350, var_437);
                    // lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)              <L 1354>
                    var_441 = wp::where(var_439, var_274, var_438);
                    // swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next                <L 1355>
                    var_442 = var_433;
                    if (!var_442) {
                        var_442 = var_442 || var_436;
                    }
                    if (!var_442) {
                        var_442 = var_442 || var_439;
                    }
                    // swap_hi_hi_next = _in_bracket(hi, hi_next) or (hi[1] < 0.0 and hi_next[1] > 0.0)       <L 1360>
                    var_444 = _in_bracket_0(var_257, var_350);
                    var_443 = var_444;
                    if (!var_443) {
                        var_447 = wp::extract(var_257, var_446);
                        var_449 = (var_447 < var_448);
                        var_445 = var_449;
                        if (var_445) {
                            var_451 = wp::extract(var_350, var_450);
                            var_453 = (var_451 > var_452);
                            var_445 = var_445 && var_453;
                        }
                        var_443 = var_443 || var_445;
                    }
                    // hi = wp.where(swap_hi_hi_next, hi_next, hi)                                <L 1361>
                    var_454 = wp::where(var_443, var_350, var_257);
                    // hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)              <L 1362>
                    var_455 = wp::where(var_443, var_274, var_259);
                    // swap_hi_mid = _in_bracket(hi, mid)                                         <L 1363>
                    var_456 = _in_bracket_0(var_454, var_361);
                    // hi = wp.where(swap_hi_mid, mid, hi)                                        <L 1364>
                    var_457 = wp::where(var_456, var_361, var_454);
                    // hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)                      <L 1365>
                    var_458 = wp::where(var_456, var_277, var_455);
                    // swap_hi_lo_next = _in_bracket(hi, lo_next)                                 <L 1366>
                    var_459 = _in_bracket_0(var_457, var_339);
                    // hi = wp.where(swap_hi_lo_next, lo_next, hi)                                <L 1367>
                    var_460 = wp::where(var_459, var_339, var_457);
                    // hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)              <L 1368>
                    var_461 = wp::where(var_459, var_268, var_458);
                    // swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next                <L 1369>
                    var_462 = var_443;
                    if (!var_462) {
                        var_462 = var_462 || var_456;
                    }
                    if (!var_462) {
                        var_462 = var_462 || var_459;
                    }
                }
                var_463 = wp::where(var_389, var_429, var_440);
                var_464 = wp::where(var_389, var_430, var_441);
                var_465 = wp::where(var_389, var_431, var_460);
                var_466 = wp::where(var_389, var_432, var_461);
                var_467 = wp::where(var_389, var_390, var_442);
                var_468 = wp::where(var_389, var_391, var_462);
                // ls_done = (                                                                    <L 1372>
                // converged                                                                      <L 1373>
                var_469 = var_389;
                if (!var_469) {
                    // or (not swap_lo and not swap_hi)                                           <L 1374>
                    var_471 = wp::unot(var_467);
                    var_470 = var_471;
                    if (var_470) {
                        var_472 = wp::unot(var_468);
                        var_470 = var_470 && var_472;
                    }
                    var_469 = var_469 || var_470;
                }
                if (!var_469) {
                    // or (lo[0] < 0.0 and lo[1] < 0.0 and lo[1] > -gtol)                         <L 1375>
                    var_475 = wp::extract(var_463, var_474);
                    var_477 = (var_475 < var_476);
                    var_473 = var_477;
                    if (var_473) {
                        var_479 = wp::extract(var_463, var_478);
                        var_481 = (var_479 < var_480);
                        var_473 = var_473 && var_481;
                    }
                    if (var_473) {
                        var_483 = wp::extract(var_463, var_482);
                        var_484 = wp::neg(var_68);
                        var_485 = (var_483 > var_484);
                        var_473 = var_473 && var_485;
                    }
                    var_469 = var_469 || var_473;
                }
                if (!var_469) {
                    // or (hi[0] < 0.0 and hi[1] > 0.0 and hi[1] < gtol)                          <L 1376>
                    var_488 = wp::extract(var_465, var_487);
                    var_490 = (var_488 < var_489);
                    var_486 = var_490;
                    if (var_486) {
                        var_492 = wp::extract(var_465, var_491);
                        var_494 = (var_492 > var_493);
                        var_486 = var_486 && var_494;
                    }
                    if (var_486) {
                        var_496 = wp::extract(var_465, var_495);
                        var_497 = (var_496 < var_68);
                        var_486 = var_486 && var_497;
                    }
                    var_469 = var_469 || var_486;
                }
                // improved = lo[0] < 0.0 or hi[0] < 0.0                                          <L 1380>
                var_500 = wp::extract(var_463, var_499);
                var_502 = (var_500 < var_501);
                var_498 = var_502;
                if (!var_498) {
                    var_504 = wp::extract(var_465, var_503);
                    var_506 = (var_504 < var_505);
                    var_498 = var_498 || var_506;
                }
                // lo_better = lo[0] < hi[0]                                                      <L 1381>
                var_508 = wp::extract(var_463, var_507);
                var_510 = wp::extract(var_465, var_509);
                var_511 = (var_508 < var_510);
                // best_alpha = wp.where(lo_better, lo_alpha, hi_alpha)                           <L 1382>
                var_512 = wp::where(var_511, var_464, var_466);
                // best_delta = wp.where(lo_better, lo[0], hi[0])                                 <L 1383>
                var_514 = wp::extract(var_463, var_513);
                var_516 = wp::extract(var_465, var_515);
                var_517 = wp::where(var_511, var_514, var_516);
                // alpha = wp.where(improved, best_alpha, alpha)                                  <L 1384>
                var_518 = wp::where(var_498, var_512, var_244);
                // improvement = wp.where(improved, -best_delta, improvement)                     <L 1385>
                var_519 = wp::neg(var_517);
                var_520 = wp::where(var_498, var_519, var_245);
                // if ls_done:                                                                    <L 1387>
                if (var_469) {
                    // ls_converged = True                                                        <L 1388>
                    // break                                                                      <L 1389>
                    wp::assign(var_243, var_287);
                    wp::assign(var_244, var_518);
                    wp::assign(var_245, var_520);
                    wp::assign(var_246, var_521);
                    wp::assign(var_254, var_463);
                    wp::assign(var_256, var_464);
                    wp::assign(var_257, var_465);
                    wp::assign(var_259, var_466);
                    goto end_for_13;
                }
                var_522 = wp::where(var_469, var_243, var_287);
                var_523 = wp::where(var_469, var_244, var_518);
                var_524 = wp::where(var_469, var_245, var_520);
                var_525 = wp::where(var_469, var_254, var_463);
                var_526 = wp::where(var_469, var_256, var_464);
                var_527 = wp::where(var_469, var_257, var_465);
                var_528 = wp::where(var_469, var_259, var_466);
                wp::assign(var_243, var_522);
                wp::assign(var_244, var_523);
                wp::assign(var_245, var_524);
                wp::assign(var_254, var_525);
                wp::assign(var_256, var_526);
                wp::assign(var_257, var_527);
                wp::assign(var_259, var_528);
                goto start_for_13;
            end_for_13:;
        }
        // for dofid in range(tid, nv, wp.block_dim()):                                           <L 1392>
        var_529 = builtin_block_dim();
        var_530 = wp::range(var_1, var_nv, var_529);
        start_for_17:;
            if (iter_cmp(var_530) == 0) goto end_for_17;
            var_531 = wp::iter_next(var_530);
            // qacc_out[worldid, dofid] += alpha * ctx_search_in[worldid, dofid]                  <L 1393>
            var_532 = wp::address(var_ctx_search_in, var_0, var_531);
            var_534 = wp::load(var_532);
            var_533 = wp::mul(var_244, var_534);
            var_535 = wp::atomic_add(var_qacc_out, var_0, var_531, var_533);
            // efc_Ma_out[worldid, dofid] += alpha * ctx_mv_in[worldid, dofid]                    <L 1394>
            var_536 = wp::address(var_ctx_mv_in, var_0, var_531);
            var_538 = wp::load(var_536);
            var_537 = wp::mul(var_244, var_538);
            var_539 = wp::atomic_add(var_efc_Ma_out, var_0, var_531, var_537);
            goto start_for_17;
        end_for_17:;
        // for efcid in range(tid, nefc, wp.block_dim()):                                         <L 1397>
        var_540 = builtin_block_dim();
        var_541 = wp::range(var_1, var_12, var_540);
        start_for_19:;
            if (iter_cmp(var_541) == 0) goto end_for_19;
            var_542 = wp::iter_next(var_541);
            // ctx_Jaref_out[worldid, efcid] += alpha * ctx_jv_in[worldid, efcid]                 <L 1398>
            var_543 = wp::address(var_ctx_jv_in, var_0, var_542);
            var_545 = wp::load(var_543);
            var_544 = wp::mul(var_244, var_545);
            var_546 = wp::atomic_add(var_ctx_Jaref_out, var_0, var_542, var_544);
            goto start_for_19;
        end_for_19:;
        // if tid == 0:                                                                           <L 1400>
        var_548 = (var_1 == var_547);
        if (var_548) {
            // ctx_improvement_out[worldid] = improvement                                         <L 1401>
            wp::array_store(var_ctx_improvement_out, var_0, var_245);
            // ctx_alpha_out[worldid] = alpha                                                     <L 1402>
            wp::array_store(var_ctx_alpha_out, var_0, var_244);
            // if wp.static(INCREMENTAL):                                                         <L 1403>
            // ctx_ls_exhausted_out[worldid] = wp.abs(alpha) < noise_floor                        <L 1404>
            var_550 = wp::abs(var_244);
            var_551 = (var_550 < var_184);
            wp::array_store(var_ctx_ls_exhausted_out, var_0, var_551);
            // if not ls_converged:                                                               <L 1405>
            var_552 = wp::unot(var_246);
            if (var_552) {
                // if wp.static(bool(warn_overflow & OverflowType.LS_ITERATIONS)):                <L 1406>
                // wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.LS_ITERATIONS))       <L 1412>
                var_555 = wp::atomic_or(var_overflow_out, var_0, var_554);
            }
        }
    }
}

