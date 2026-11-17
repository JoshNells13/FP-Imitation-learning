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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:52
static CUDA_CALLABLE wp::quat_t<wp::float32> axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    wp::quat_t<wp::float32> var_13;
    //---------
    // forward
    // def axis_angle_to_quat(axis: wp.vec3, angle: float) -> wp.quat:                        <L 53>
    // s, c = wp.sin(angle * 0.5), wp.cos(angle * 0.5)                                        <L 54>
    var_1 = wp::mul(var_angle, var_0);
    var_2 = wp::sin(var_1);
    var_4 = wp::mul(var_angle, var_3);
    var_5 = wp::cos(var_4);
    // axis = axis * s                                                                        <L 55>
    var_6 = wp::mul(var_axis, var_2);
    // return wp.quat(c, axis[0], axis[1], axis[2])                                           <L 56>
    var_8 = wp::extract(var_6, var_7);
    var_10 = wp::extract(var_6, var_9);
    var_12 = wp::extract(var_6, var_11);
    var_13 = wp::quat_t<wp::float32>(var_5, var_8, var_10, var_12);
    return var_13;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:23
static CUDA_CALLABLE wp::quat_t<wp::float32> mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 3;
    wp::float32 var_18;
    const wp::int32 var_19 = 3;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    const wp::int32 var_36 = 3;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 3;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 0;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    const wp::int32 var_53 = 3;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    const wp::int32 var_59 = 0;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 3;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 0;
    wp::float32 var_70;
    const wp::int32 var_71 = 3;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::int32 var_74 = 1;
    wp::float32 var_75;
    const wp::int32 var_76 = 2;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 2;
    wp::float32 var_81;
    const wp::int32 var_82 = 1;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 3;
    wp::float32 var_87;
    const wp::int32 var_88 = 0;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::quat_t<wp::float32> var_92;
    //---------
    // forward
    // def mul_quat(u: wp.quat, v: wp.quat) -> wp.quat:                                       <L 24>
    // return wp.quat(                                                                        <L 25>
    // u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],                                 <L 26>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_u, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::sub(var_4, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::sub(var_10, var_15);
    var_18 = wp::extract(var_u, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    // u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],                                 <L 27>
    var_24 = wp::extract(var_u, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_29 = wp::extract(var_u, var_28);
    var_31 = wp::extract(var_v, var_30);
    var_32 = wp::mul(var_29, var_31);
    var_33 = wp::add(var_27, var_32);
    var_35 = wp::extract(var_u, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_u, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::sub(var_39, var_44);
    // u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],                                 <L 28>
    var_47 = wp::extract(var_u, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_52 = wp::extract(var_u, var_51);
    var_54 = wp::extract(var_v, var_53);
    var_55 = wp::mul(var_52, var_54);
    var_56 = wp::sub(var_50, var_55);
    var_58 = wp::extract(var_u, var_57);
    var_60 = wp::extract(var_v, var_59);
    var_61 = wp::mul(var_58, var_60);
    var_62 = wp::add(var_56, var_61);
    var_64 = wp::extract(var_u, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    // u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],                                 <L 29>
    var_70 = wp::extract(var_u, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_75 = wp::extract(var_u, var_74);
    var_77 = wp::extract(var_v, var_76);
    var_78 = wp::mul(var_75, var_77);
    var_79 = wp::add(var_73, var_78);
    var_81 = wp::extract(var_u, var_80);
    var_83 = wp::extract(var_v, var_82);
    var_84 = wp::mul(var_81, var_83);
    var_85 = wp::sub(var_79, var_84);
    var_87 = wp::extract(var_u, var_86);
    var_89 = wp::extract(var_v, var_88);
    var_90 = wp::mul(var_87, var_89);
    var_91 = wp::add(var_85, var_90);
    var_92 = wp::quat_t<wp::float32>(var_22, var_45, var_68, var_91);
    return var_92;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:188
static CUDA_CALLABLE wp::quat_t<wp::float32> quat_integrate_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_v,
    wp::float32 var_dt)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::quat_t<wp::float32> var_3;
    wp::quat_t<wp::float32> var_4;
    wp::quat_t<wp::float32> var_5;
    wp::quat_t<wp::float32> var_6;
    //---------
    // forward
    // def quat_integrate(q: wp.quat, v: wp.vec3, dt: float) -> wp.quat:                      <L 189>
    // norm_ = wp.length(v)                                                                   <L 191>
    var_0 = wp::length(var_v);
    // v = wp.normalize(v)  # does that need proper zero gradient handling?                   <L 192>
    var_1 = wp::normalize(var_v);
    // angle = dt * norm_                                                                     <L 193>
    var_2 = wp::mul(var_dt, var_0);
    // q_res = axis_angle_to_quat(v, angle)                                                   <L 195>
    var_3 = axis_angle_to_quat_0(var_1, var_2);
    // q = wp.normalize(q)                                                                    <L 196>
    var_4 = wp::normalize(var_q);
    // q_res = mul_quat(q, q_res)                                                             <L 197>
    var_5 = mul_quat_0(var_4, var_3);
    // return wp.normalize(q_res)                                                             <L 199>
    var_6 = wp::normalize(var_5);
    return var_6;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:675
static CUDA_CALLABLE wp::float32 _sigmoid_0(
    wp::float32 var_x)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    bool var_4;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 3.0;
    wp::float32 var_9;
    const wp::float32 var_10 = 2.0;
    wp::float32 var_11;
    const wp::float32 var_12 = 5.0;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 10.0;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def _sigmoid(x: float) -> float:                                                       <L 676>
    // if x <= 0.0:                                                                           <L 678>
    var_1 = (var_x <= var_0);
    if (var_1) {
        // return 0.0                                                                         <L 679>
        return var_2;
    }
    // if x >= 1.0:                                                                           <L 681>
    var_4 = (var_x >= var_3);
    if (var_4) {
        // return 1.0                                                                         <L 682>
        return var_5;
    }
    // return x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)                                  <L 686>
    var_6 = wp::mul(var_x, var_x);
    var_7 = wp::mul(var_6, var_x);
    var_9 = wp::mul(var_8, var_x);
    var_11 = wp::mul(var_10, var_x);
    var_13 = wp::sub(var_11, var_12);
    var_14 = wp::mul(var_9, var_13);
    var_16 = wp::add(var_14, var_15);
    var_17 = wp::mul(var_7, var_16);
    return var_17;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:689
static CUDA_CALLABLE wp::float32 muscle_dynamics_timescale_0(
    wp::float32 var_dctrl,
    wp::float32 var_tau_act,
    wp::float32 var_tau_deact,
    wp::float32 var_smooth_width)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-15;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.5;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    //---------
    // forward
    // def muscle_dynamics_timescale(dctrl: float, tau_act: float, tau_deact: float, smooth_width: float) -> float:       <L 690>
    // if smooth_width < MJ_MINVAL:                                                           <L 693>
    var_1 = (var_smooth_width < var_0);
    if (var_1) {
        // if dctrl > 0.0:                                                                    <L 694>
        var_3 = (var_dctrl > var_2);
        if (var_3) {
            // return tau_act                                                                 <L 695>
            return var_tau_act;
        }
        if (!var_3) {
            // return tau_deact                                                               <L 697>
            return var_tau_deact;
        }
    }
    if (!var_1) {
        // return tau_deact + (tau_act - tau_deact) * _sigmoid(dctrl / smooth_width + 0.5)       <L 700>
        var_4 = wp::sub(var_tau_act, var_tau_deact);
        var_5 = wp::div(var_dctrl, var_smooth_width);
        var_7 = wp::add(var_5, var_6);
        var_8 = _sigmoid_0(var_7);
        var_9 = wp::mul(var_4, var_8);
        var_10 = wp::add(var_tau_deact, var_9);
        return var_10;
    }
    return {};
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:703
static CUDA_CALLABLE wp::float32 muscle_dynamics_0(
    wp::float32 var_control,
    wp::float32 var_activation,
    wp::vec_t<10, wp::float32> var_prm)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 1.0;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 1.0;
    wp::float32 var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.5;
    const wp::float32 var_9 = 1.5;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.5;
    const wp::float32 var_16 = 1.5;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 2;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 1e-15;
    wp::float32 var_25;
    wp::float32 var_26;
    //---------
    // forward
    // def muscle_dynamics(control: float, activation: float, prm: vec10) -> float:           <L 704>
    // ctrlclamp = wp.clamp(control, 0.0, 1.0)                                                <L 707>
    var_2 = wp::clamp(var_control, var_0, var_1);
    // actclamp = wp.clamp(activation, 0.0, 1.0)                                              <L 710>
    var_5 = wp::clamp(var_activation, var_3, var_4);
    // tau_act = prm[0] * (0.5 + 1.5 * actclamp)  # activation timescale                      <L 713>
    var_7 = wp::extract(var_prm, var_6);
    var_10 = wp::mul(var_9, var_5);
    var_11 = wp::add(var_8, var_10);
    var_12 = wp::mul(var_7, var_11);
    // tau_deact = prm[1] / (0.5 + 1.5 * actclamp)  # deactivation timescale                  <L 714>
    var_14 = wp::extract(var_prm, var_13);
    var_17 = wp::mul(var_16, var_5);
    var_18 = wp::add(var_15, var_17);
    var_19 = wp::div(var_14, var_18);
    // smooth_width = prm[2]  # width of smoothing sigmoid                                    <L 715>
    var_21 = wp::extract(var_prm, var_20);
    // dctrl = ctrlclamp - activation  # excess excitation                                    <L 716>
    var_22 = wp::sub(var_2, var_activation);
    // tau = muscle_dynamics_timescale(dctrl, tau_act, tau_deact, smooth_width)               <L 718>
    var_23 = muscle_dynamics_timescale_0(var_22, var_12, var_19, var_21);
    // return dctrl / wp.max(MJ_MINVAL, tau)                                                  <L 721>
    var_25 = wp::max(var_24, var_23);
    var_26 = wp::div(var_22, var_25);
    return var_26;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:724
static CUDA_CALLABLE wp::vec_t<6, wp::int32> dcmotor_slots_0(
    wp::vec_t<10, wp::float32> var_dynprm,
    wp::vec_t<10, wp::float32> var_gainprm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = -1;
    const wp::int32 var_1 = -1;
    const wp::int32 var_2 = -1;
    const wp::int32 var_3 = -1;
    const wp::int32 var_4 = -1;
    const wp::int32 var_5 = 0;
    wp::vec_t<6, wp::int32> var_6;
    const wp::int32 var_7 = 0;
    const wp::int32 var_8 = 7;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    const wp::int32 var_12 = 0;
    const wp::int32 var_13 = 1;
    wp::int32 var_14;
    wp::int32 var_15;
    const wp::int32 var_16 = 5;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.0;
    bool var_19;
    const wp::int32 var_20 = 1;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    wp::int32 var_23;
    const wp::int32 var_24 = 2;
    wp::float32 var_25;
    const wp::float32 var_26 = 0.0;
    bool var_27;
    const wp::int32 var_28 = 2;
    const wp::int32 var_29 = 1;
    wp::int32 var_30;
    wp::int32 var_31;
    const wp::int32 var_32 = 5;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.0;
    bool var_35;
    const wp::int32 var_36 = 3;
    const wp::int32 var_37 = 1;
    wp::int32 var_38;
    wp::int32 var_39;
    const wp::int32 var_40 = 0;
    wp::float32 var_41;
    const wp::float32 var_42 = 0.0;
    bool var_43;
    const wp::int32 var_44 = 4;
    const wp::int32 var_45 = 1;
    wp::int32 var_46;
    wp::int32 var_47;
    const wp::int32 var_48 = 5;
    //---------
    // forward
    // def dcmotor_slots(dynprm: types.vec10, gainprm: types.vec10) -> types.vec6i:           <L 725>
    // s = types.vec6i(-1, -1, -1, -1, -1, 0)                                                 <L 743>
    var_6 = wp::vec_t<6, wp::int32>({var_0, var_1, var_2, var_3, var_4, var_5});
    // num_slots = 0                                                                          <L 744>
    // if dynprm[7] > 0.0:                                                                    <L 745>
    var_9 = wp::extract(var_dynprm, var_8);
    var_11 = (var_9 > var_10);
    if (var_11) {
        // s[0] = num_slots                                                                   <L 746>
        wp::assign_inplace(var_6, var_12, var_7);
        // num_slots += 1                                                                     <L 747>
        var_14 = wp::add(var_7, var_13);
    }
    var_15 = wp::where(var_11, var_14, var_7);
    // if gainprm[5] > 0.0:                                                                   <L 748>
    var_17 = wp::extract(var_gainprm, var_16);
    var_19 = (var_17 > var_18);
    if (var_19) {
        // s[1] = num_slots                                                                   <L 749>
        wp::assign_inplace(var_6, var_20, var_15);
        // num_slots += 1                                                                     <L 750>
        var_22 = wp::add(var_15, var_21);
    }
    var_23 = wp::where(var_19, var_22, var_15);
    // if dynprm[2] > 0.0:                                                                    <L 751>
    var_25 = wp::extract(var_dynprm, var_24);
    var_27 = (var_25 > var_26);
    if (var_27) {
        // s[2] = num_slots                                                                   <L 752>
        wp::assign_inplace(var_6, var_28, var_23);
        // num_slots += 1                                                                     <L 753>
        var_30 = wp::add(var_23, var_29);
    }
    var_31 = wp::where(var_27, var_30, var_23);
    // if dynprm[5] > 0.0:                                                                    <L 754>
    var_33 = wp::extract(var_dynprm, var_32);
    var_35 = (var_33 > var_34);
    if (var_35) {
        // s[3] = num_slots                                                                   <L 755>
        wp::assign_inplace(var_6, var_36, var_31);
        // num_slots += 1                                                                     <L 756>
        var_38 = wp::add(var_31, var_37);
    }
    var_39 = wp::where(var_35, var_38, var_31);
    // if dynprm[0] > 0.0:                                                                    <L 757>
    var_41 = wp::extract(var_dynprm, var_40);
    var_43 = (var_41 > var_42);
    if (var_43) {
        // s[4] = num_slots                                                                   <L 758>
        wp::assign_inplace(var_6, var_44, var_39);
        // num_slots += 1                                                                     <L 759>
        var_46 = wp::add(var_39, var_45);
    }
    var_47 = wp::where(var_43, var_46, var_39);
    // s[5] = num_slots                                                                       <L 760>
    wp::assign_inplace(var_6, var_48, var_47);
    // return s                                                                               <L 761>
    return var_6;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:792
static CUDA_CALLABLE wp::vec_t<4, wp::float32> unpack_servo_inputs_0(
    wp::array_t<wp::float32> var_ctrl_in,
    wp::int32 var_worldid,
    wp::int32 var_uadr,
    wp::int32 var_spec,
    wp::float32 var_u_first)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<4, wp::float32> var_4;
    wp::int32 var_5;
    const bool var_6 = false;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    const wp::int32 var_9 = 0;
    bool var_10;
    bool var_11;
    wp::float32* var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 0;
    const bool var_17 = true;
    const wp::int32 var_18 = 1;
    wp::int32 var_19;
    wp::int32 var_20;
    bool var_21;
    const wp::int32 var_22 = 2;
    wp::int32 var_23;
    const wp::int32 var_24 = 0;
    bool var_25;
    bool var_26;
    wp::float32* var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    const bool var_32 = true;
    const wp::int32 var_33 = 1;
    wp::int32 var_34;
    wp::int32 var_35;
    bool var_36;
    const wp::int32 var_37 = 4;
    wp::int32 var_38;
    const wp::int32 var_39 = 0;
    bool var_40;
    bool var_41;
    wp::float32* var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 2;
    const bool var_47 = true;
    const wp::int32 var_48 = 1;
    wp::int32 var_49;
    wp::int32 var_50;
    bool var_51;
    const wp::int32 var_52 = 8;
    wp::int32 var_53;
    const wp::int32 var_54 = 0;
    bool var_55;
    bool var_56;
    wp::float32* var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    const wp::int32 var_61 = 3;
    //---------
    // forward
    // def unpack_servo_inputs(                                                               <L 793>
    // out = wp.vec4(0.0, 0.0, 0.0, 0.0)                                                      <L 802>
    var_4 = wp::vec_t<4, wp::float32>(var_0, var_1, var_2, var_3);
    // adr = uadr                                                                             <L 803>
    var_5 = wp::copy(var_uadr);
    // first_set = False                                                                      <L 804>
    // if (spec & 1) != 0:  # POS                                                             <L 805>
    var_8 = wp::bit_and(var_spec, var_7);
    var_10 = (var_8 != var_9);
    if (var_10) {
        // out[0] = u_first if not first_set else ctrl_in[worldid, adr]                       <L 806>
        var_11 = wp::unot(var_6);
        if (var_11) {
        }
        if (!var_11) {
            var_12 = wp::address(var_ctrl_in, var_worldid, var_5);
            var_14 = wp::load(var_12);
            var_13 = wp::copy(var_14);
        }
        var_15 = wp::where(var_11, var_u_first, var_13);
        wp::assign_inplace(var_4, var_16, var_15);
        // first_set = True                                                                   <L 807>
        // adr += 1                                                                           <L 808>
        var_19 = wp::add(var_5, var_18);
    }
    var_20 = wp::where(var_10, var_19, var_5);
    var_21 = wp::where(var_10, var_17, var_6);
    // if (spec & 2) != 0:  # VEL                                                             <L 809>
    var_23 = wp::bit_and(var_spec, var_22);
    var_25 = (var_23 != var_24);
    if (var_25) {
        // out[1] = u_first if not first_set else ctrl_in[worldid, adr]                       <L 810>
        var_26 = wp::unot(var_21);
        if (var_26) {
        }
        if (!var_26) {
            var_27 = wp::address(var_ctrl_in, var_worldid, var_20);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
        }
        var_30 = wp::where(var_26, var_u_first, var_28);
        wp::assign_inplace(var_4, var_31, var_30);
        // first_set = True                                                                   <L 811>
        // adr += 1                                                                           <L 812>
        var_34 = wp::add(var_20, var_33);
    }
    var_35 = wp::where(var_25, var_34, var_20);
    var_36 = wp::where(var_25, var_32, var_21);
    // if (spec & 4) != 0:  # FF                                                              <L 813>
    var_38 = wp::bit_and(var_spec, var_37);
    var_40 = (var_38 != var_39);
    if (var_40) {
        // out[2] = u_first if not first_set else ctrl_in[worldid, adr]                       <L 814>
        var_41 = wp::unot(var_36);
        if (var_41) {
        }
        if (!var_41) {
            var_42 = wp::address(var_ctrl_in, var_worldid, var_35);
            var_44 = wp::load(var_42);
            var_43 = wp::copy(var_44);
        }
        var_45 = wp::where(var_41, var_u_first, var_43);
        wp::assign_inplace(var_4, var_46, var_45);
        // first_set = True                                                                   <L 815>
        // adr += 1                                                                           <L 816>
        var_49 = wp::add(var_35, var_48);
    }
    var_50 = wp::where(var_40, var_49, var_35);
    var_51 = wp::where(var_40, var_47, var_36);
    // if (spec & 8) != 0:  # VOLTAGE                                                         <L 817>
    var_53 = wp::bit_and(var_spec, var_52);
    var_55 = (var_53 != var_54);
    if (var_55) {
        // out[3] = u_first if not first_set else ctrl_in[worldid, adr]                       <L 818>
        var_56 = wp::unot(var_51);
        if (var_56) {
        }
        if (!var_56) {
            var_57 = wp::address(var_ctrl_in, var_worldid, var_50);
            var_59 = wp::load(var_57);
            var_58 = wp::copy(var_59);
        }
        var_60 = wp::where(var_56, var_u_first, var_58);
        wp::assign_inplace(var_4, var_61, var_60);
    }
    // return out                                                                             <L 819>
    return var_4;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:822
static CUDA_CALLABLE wp::float32 dcmotor_voltage_0(
    wp::array_t<wp::float32> var_ctrl_in,
    wp::int32 var_worldid,
    wp::int32 var_uadr,
    wp::int32 var_spec,
    wp::float32 var_u_first,
    wp::float32 var_length,
    wp::float32 var_velocity,
    wp::float32 var_x_I,
    wp::vec_t<10, wp::float32> var_gainprm)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::vec_t<4, wp::float32> var_1;
    const wp::int32 var_2 = 7;
    wp::int32 var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    const wp::int32 var_6 = 4;
    wp::float32 var_7;
    const wp::int32 var_8 = 5;
    wp::float32 var_9;
    const wp::int32 var_10 = 6;
    wp::float32 var_11;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 7;
    wp::float32 var_35;
    const wp::float32 var_36 = 0.0;
    bool var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::int32 var_41 = 3;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    //---------
    // forward
    // def dcmotor_voltage(                                                                   <L 823>
    // voltage = 0.0                                                                          <L 836>
    // u4 = unpack_servo_inputs(ctrl_in, worldid, uadr, spec, u_first)                        <L 837>
    var_1 = unpack_servo_inputs_0(var_ctrl_in, var_worldid, var_uadr, var_spec, var_u_first);
    // if (spec & 7) != 0:  # POS (1) | VEL (2) | FF (4)                                      <L 839>
    var_3 = wp::bit_and(var_spec, var_2);
    var_5 = (var_3 != var_4);
    if (var_5) {
        // kp = gainprm[4]                                                                    <L 840>
        var_7 = wp::extract(var_gainprm, var_6);
        // ki = gainprm[5]                                                                    <L 841>
        var_9 = wp::extract(var_gainprm, var_8);
        // kd = gainprm[6]                                                                    <L 842>
        var_11 = wp::extract(var_gainprm, var_10);
        // torque = kp * (u4[0] - length) + kd * (u4[1] - velocity) + ki * x_I + u4[2]        <L 843>
        var_13 = wp::extract(var_1, var_12);
        var_14 = wp::sub(var_13, var_length);
        var_15 = wp::mul(var_7, var_14);
        var_17 = wp::extract(var_1, var_16);
        var_18 = wp::sub(var_17, var_velocity);
        var_19 = wp::mul(var_11, var_18);
        var_20 = wp::add(var_15, var_19);
        var_21 = wp::mul(var_9, var_x_I);
        var_22 = wp::add(var_20, var_21);
        var_24 = wp::extract(var_1, var_23);
        var_25 = wp::add(var_22, var_24);
        // R = gainprm[0]                                                                     <L 845>
        var_27 = wp::extract(var_gainprm, var_26);
        // K = gainprm[1]                                                                     <L 846>
        var_29 = wp::extract(var_gainprm, var_28);
        // voltage = (R / K) * torque + K * velocity                                          <L 847>
        var_30 = wp::div(var_27, var_29);
        var_31 = wp::mul(var_30, var_25);
        var_32 = wp::mul(var_29, var_velocity);
        var_33 = wp::add(var_31, var_32);
        // Vmax = gainprm[7]                                                                  <L 849>
        var_35 = wp::extract(var_gainprm, var_34);
        // if Vmax > 0.0:                                                                     <L 850>
        var_37 = (var_35 > var_36);
        if (var_37) {
            // voltage = wp.clamp(voltage, -Vmax, Vmax)                                       <L 851>
            var_38 = wp::neg(var_35);
            var_39 = wp::clamp(var_33, var_38, var_35);
        }
        var_40 = wp::where(var_37, var_39, var_33);
        // return voltage + u4[3]                                                             <L 853>
        var_42 = wp::extract(var_1, var_41);
        var_43 = wp::add(var_40, var_42);
        return var_43;
    }
    if (!var_5) {
        // return u_first                                                                     <L 855>
        return var_u_first;
    }
    var_44 = wp::where(var_5, var_40, var_0);
    return {};
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:786
static CUDA_CALLABLE wp::float32 lugre_stribeck_0(
    wp::float32 var_velocity,
    wp::float32 var_F_C,
    wp::float32 var_F_S,
    wp::float32 var_v_S)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-15;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    //---------
    // forward
    // def lugre_stribeck(velocity: float, F_C: float, F_S: float, v_S: float) -> float:       <L 787>
    // ratio = velocity / wp.max(MJ_MINVAL, v_S)                                              <L 788>
    var_1 = wp::max(var_0, var_v_S);
    var_2 = wp::div(var_velocity, var_1);
    // return F_C + (F_S - F_C) * wp.exp(-ratio * ratio)                                      <L 789>
    var_3 = wp::sub(var_F_S, var_F_C);
    var_4 = wp::neg(var_2);
    var_5 = wp::mul(var_4, var_2);
    var_6 = wp::exp(var_5);
    var_7 = wp::mul(var_3, var_6);
    var_8 = wp::add(var_F_C, var_7);
    return var_8;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:37
static CUDA_CALLABLE wp::float32 next_act_0(
    wp::float32 var_opt_timestep,
    wp::int32 var_actuator_dyntype,
    wp::vec_t<10, wp::float32> var_actuator_dynprm,
    wp::vec_t<2, wp::float32> var_actuator_actrange,
    wp::float32 var_act_in,
    wp::float32 var_act_dot_in,
    wp::float32 var_act_dot_scale,
    bool var_clamp)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    bool var_1;
    const wp::float32 var_2 = 1e-15;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 1.0;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 7;
    bool var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    //---------
    // forward
    // def next_act(                                                                          <L 38>
    // if actuator_dyntype == DynType.FILTEREXACT:                                            <L 52>
    var_1 = (var_actuator_dyntype == var_0);
    if (var_1) {
        // tau = wp.max(MJ_MINVAL, actuator_dynprm[0])                                        <L 53>
        var_4 = wp::extract(var_actuator_dynprm, var_3);
        var_5 = wp::max(var_2, var_4);
        // act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))       <L 54>
        var_6 = wp::mul(var_act_dot_scale, var_act_dot_in);
        var_7 = wp::mul(var_6, var_5);
        var_9 = wp::neg(var_opt_timestep);
        var_10 = wp::div(var_9, var_5);
        var_11 = wp::exp(var_10);
        var_12 = wp::sub(var_8, var_11);
        var_13 = wp::mul(var_7, var_12);
        var_14 = wp::add(var_act_in, var_13);
    }
    if (!var_1) {
        // elif actuator_dyntype == DynType.USER:                                             <L 55>
        var_16 = (var_actuator_dyntype == var_15);
        if (var_16) {
            // return act_in                                                                  <L 56>
            return var_act_in;
        }
        if (!var_16) {
            // act = act_in + act_dot_scale * act_dot_in * opt_timestep                       <L 58>
            var_17 = wp::mul(var_act_dot_scale, var_act_dot_in);
            var_18 = wp::mul(var_17, var_opt_timestep);
            var_19 = wp::add(var_act_in, var_18);
        }
    }
    var_20 = wp::where(var_1, var_14, var_19);
    // if clamp:                                                                              <L 61>
    if (var_clamp) {
        // act = wp.clamp(act, actuator_actrange[0], actuator_actrange[1])                    <L 62>
        var_22 = wp::extract(var_actuator_actrange, var_21);
        var_24 = wp::extract(var_actuator_actrange, var_23);
        var_25 = wp::clamp(var_20, var_22, var_24);
    }
    var_26 = wp::where(var_clamp, var_25, var_20);
    // return act                                                                             <L 64>
    return var_26;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:459
static CUDA_CALLABLE wp::float32 muscle_gain_length_0(
    wp::float32 var_length,
    wp::float32 var_lmin,
    wp::float32 var_lmax)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.5;
    const wp::float32 var_5 = 1.0;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.5;
    const wp::float32 var_9 = 1.0;
    wp::float32 var_10;
    wp::float32 var_11;
    bool var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1e-15;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 0.5;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 1.0;
    bool var_22;
    const wp::float32 var_23 = 1.0;
    wp::float32 var_24;
    const wp::float32 var_25 = 1e-15;
    const wp::float32 var_26 = 1.0;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::float32 var_30 = 1.0;
    const wp::float32 var_31 = 0.5;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    bool var_35;
    const wp::float32 var_36 = 1.0;
    wp::float32 var_37;
    const wp::float32 var_38 = 1e-15;
    const wp::float32 var_39 = 1.0;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 1.0;
    const wp::float32 var_44 = 0.5;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::float32 var_49 = 1e-15;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 0.5;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    //---------
    // forward
    // def muscle_gain_length(length: float, lmin: float, lmax: float) -> float:              <L 460>
    // if (lmin > length) or (length > lmax):                                                 <L 462>
    var_1 = (var_lmin > var_length);
    var_0 = var_1;
    if (!var_0) {
        var_2 = (var_length > var_lmax);
        var_0 = var_0 || var_2;
    }
    if (var_0) {
        // return 0.0                                                                         <L 463>
        return var_3;
    }
    // a = 0.5 * (lmin + 1.0)                                                                 <L 466>
    var_6 = wp::add(var_lmin, var_5);
    var_7 = wp::mul(var_4, var_6);
    // b = 0.5 * (1.0 + lmax)                                                                 <L 467>
    var_10 = wp::add(var_9, var_lmax);
    var_11 = wp::mul(var_8, var_10);
    // if length <= a:                                                                        <L 469>
    var_12 = (var_length <= var_7);
    if (var_12) {
        // x = (length - lmin) / wp.max(MJ_MINVAL, a - lmin)                                  <L 470>
        var_13 = wp::sub(var_length, var_lmin);
        var_15 = wp::sub(var_7, var_lmin);
        var_16 = wp::max(var_14, var_15);
        var_17 = wp::div(var_13, var_16);
        // return 0.5 * x * x                                                                 <L 471>
        var_19 = wp::mul(var_18, var_17);
        var_20 = wp::mul(var_19, var_17);
        return var_20;
    }
    if (!var_12) {
        // elif length <= 1.0:                                                                <L 472>
        var_22 = (var_length <= var_21);
        if (var_22) {
            // x = (1.0 - length) / wp.max(MJ_MINVAL, 1.0 - a)                                <L 473>
            var_24 = wp::sub(var_23, var_length);
            var_27 = wp::sub(var_26, var_7);
            var_28 = wp::max(var_25, var_27);
            var_29 = wp::div(var_24, var_28);
            // return 1.0 - 0.5 * x * x                                                       <L 474>
            var_32 = wp::mul(var_31, var_29);
            var_33 = wp::mul(var_32, var_29);
            var_34 = wp::sub(var_30, var_33);
            return var_34;
        }
        if (!var_22) {
            // elif length <= b:                                                              <L 475>
            var_35 = (var_length <= var_11);
            if (var_35) {
                // x = (length - 1.0) / wp.max(MJ_MINVAL, b - 1.0)                            <L 476>
                var_37 = wp::sub(var_length, var_36);
                var_40 = wp::sub(var_11, var_39);
                var_41 = wp::max(var_38, var_40);
                var_42 = wp::div(var_37, var_41);
                // return 1.0 - 0.5 * x * x                                                   <L 477>
                var_45 = wp::mul(var_44, var_42);
                var_46 = wp::mul(var_45, var_42);
                var_47 = wp::sub(var_43, var_46);
                return var_47;
            }
            if (!var_35) {
                // x = (lmax - length) / wp.max(MJ_MINVAL, lmax - b)                          <L 479>
                var_48 = wp::sub(var_lmax, var_length);
                var_50 = wp::sub(var_lmax, var_11);
                var_51 = wp::max(var_49, var_50);
                var_52 = wp::div(var_48, var_51);
                // return 0.5 * x * x                                                         <L 480>
                var_54 = wp::mul(var_53, var_52);
                var_55 = wp::mul(var_54, var_52);
                return var_55;
            }
            var_56 = wp::where(var_35, var_38, var_49);
            var_57 = wp::where(var_35, var_42, var_52);
        }
        var_58 = wp::where(var_22, var_25, var_56);
        var_59 = wp::where(var_22, var_29, var_57);
    }
    var_60 = wp::where(var_12, var_14, var_58);
    var_61 = wp::where(var_12, var_17, var_59);
    return {};
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:483
static CUDA_CALLABLE wp::float32 muscle_gain_0(
    wp::float32 var_len,
    wp::float32 var_vel,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::vec_t<2, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    const wp::int32 var_13 = 6;
    wp::float32 var_14;
    const wp::int32 var_15 = 8;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    const wp::float32 var_19 = 1e-15;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 0;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 0;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::float32 var_47 = 1.0;
    wp::float32 var_48;
    const wp::float32 var_49 = -1.0;
    bool var_50;
    const wp::float32 var_51 = 0.0;
    const wp::float32 var_52 = 0.0;
    bool var_53;
    const wp::float32 var_54 = 1.0;
    wp::float32 var_55;
    const wp::float32 var_56 = 1.0;
    wp::float32 var_57;
    wp::float32 var_58;
    bool var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    //---------
    // forward
    // def muscle_gain(len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 484>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 487>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 488>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 489>
    var_8 = wp::extract(var_prm, var_7);
    // lmin = prm[4]                                                                          <L 490>
    var_10 = wp::extract(var_prm, var_9);
    // lmax = prm[5]                                                                          <L 491>
    var_12 = wp::extract(var_prm, var_11);
    // vmax = prm[6]                                                                          <L 492>
    var_14 = wp::extract(var_prm, var_13);
    // fvmax = prm[8]                                                                         <L 493>
    var_16 = wp::extract(var_prm, var_15);
    // if force < 0.0:                                                                        <L 496>
    var_18 = (var_6 < var_17);
    if (var_18) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 497>
        var_20 = wp::max(var_19, var_acc0);
        var_21 = wp::div(var_8, var_20);
    }
    var_22 = wp::where(var_18, var_21, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 500>
    var_24 = wp::extract(var_lengthrange, var_23);
    var_26 = wp::extract(var_lengthrange, var_25);
    var_27 = wp::sub(var_24, var_26);
    var_29 = wp::extract(var_4, var_28);
    var_31 = wp::extract(var_4, var_30);
    var_32 = wp::sub(var_29, var_31);
    var_33 = wp::max(var_19, var_32);
    var_34 = wp::div(var_27, var_33);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 503>
    var_36 = wp::extract(var_4, var_35);
    var_38 = wp::extract(var_lengthrange, var_37);
    var_39 = wp::sub(var_len, var_38);
    var_40 = wp::max(var_19, var_34);
    var_41 = wp::div(var_39, var_40);
    var_42 = wp::add(var_36, var_41);
    // V = vel / wp.max(MJ_MINVAL, L0 * vmax)                                                 <L 504>
    var_43 = wp::mul(var_34, var_14);
    var_44 = wp::max(var_19, var_43);
    var_45 = wp::div(var_vel, var_44);
    // FL = muscle_gain_length(L, lmin, lmax)                                                 <L 507>
    var_46 = muscle_gain_length_0(var_42, var_10, var_12);
    // y = fvmax - 1.0                                                                        <L 510>
    var_48 = wp::sub(var_16, var_47);
    // if V <= -1.0:                                                                          <L 511>
    var_50 = (var_45 <= var_49);
    if (var_50) {
        // FV = 0.0                                                                           <L 512>
    }
    if (!var_50) {
        // elif V <= 0.0:                                                                     <L 513>
        var_53 = (var_45 <= var_52);
        if (var_53) {
            // FV = (V + 1.0) * (V + 1.0)                                                     <L 514>
            var_55 = wp::add(var_45, var_54);
            var_57 = wp::add(var_45, var_56);
            var_58 = wp::mul(var_55, var_57);
        }
        if (!var_53) {
            // elif V <= y:                                                                   <L 515>
            var_59 = (var_45 <= var_48);
            if (var_59) {
                // FV = fvmax - (y - V) * (y - V) / wp.max(MJ_MINVAL, y)                      <L 516>
                var_60 = wp::sub(var_48, var_45);
                var_61 = wp::sub(var_48, var_45);
                var_62 = wp::mul(var_60, var_61);
                var_63 = wp::max(var_19, var_48);
                var_64 = wp::div(var_62, var_63);
                var_65 = wp::sub(var_16, var_64);
            }
            if (!var_59) {
                // FV = fvmax                                                                 <L 518>
                var_66 = wp::copy(var_16);
            }
            var_67 = wp::where(var_59, var_65, var_66);
        }
        var_68 = wp::where(var_53, var_58, var_67);
    }
    var_69 = wp::where(var_50, var_51, var_68);
    // return -force * FL * FV                                                                <L 521>
    var_70 = wp::neg(var_22);
    var_71 = wp::mul(var_70, var_46);
    var_72 = wp::mul(var_71, var_69);
    return var_72;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:764
static CUDA_CALLABLE wp::float32 dcmotor_resistance_0(
    wp::array_t<wp::float32> var_act_in,
    wp::int32 var_worldid,
    wp::int32 var_actadr,
    wp::vec_t<10, wp::float32> var_dynprm,
    wp::vec_t<10, wp::float32> var_gainprm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    bool var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::vec_t<6, wp::int32> var_9;
    const wp::int32 var_10 = 2;
    wp::int32 var_11;
    wp::int32 var_12;
    wp::float32* var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 2;
    wp::float32 var_17;
    const wp::int32 var_18 = 3;
    wp::float32 var_19;
    const wp::int32 var_20 = 4;
    wp::float32 var_21;
    const wp::float32 var_22 = 1.0;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::float32 var_29 = 1e-15;
    wp::float32 var_30;
    //---------
    // forward
    // def dcmotor_resistance(                                                                <L 765>
    // R = gainprm[0]                                                                         <L 775>
    var_1 = wp::extract(var_gainprm, var_0);
    // if actadr >= 0 and dynprm[2] > 0.0:                                                    <L 776>
    var_4 = (var_actadr >= var_3);
    var_2 = var_4;
    if (var_2) {
        var_6 = wp::extract(var_dynprm, var_5);
        var_8 = (var_6 > var_7);
        var_2 = var_2 && var_8;
    }
    if (var_2) {
        // slots = dcmotor_slots(dynprm, gainprm)                                             <L 777>
        var_9 = dcmotor_slots_0(var_dynprm, var_gainprm);
        // T = act_in[worldid, actadr + slots[2]]                                             <L 778>
        var_11 = wp::extract(var_9, var_10);
        var_12 = wp::add(var_actadr, var_11);
        var_13 = wp::address(var_act_in, var_worldid, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // alpha = gainprm[2]                                                                 <L 779>
        var_17 = wp::extract(var_gainprm, var_16);
        // T0 = gainprm[3]                                                                    <L 780>
        var_19 = wp::extract(var_gainprm, var_18);
        // Ta = dynprm[4]                                                                     <L 781>
        var_21 = wp::extract(var_dynprm, var_20);
        // R *= 1.0 + alpha * (T + Ta - T0)                                                   <L 782>
        var_23 = wp::add(var_14, var_21);
        var_24 = wp::sub(var_23, var_19);
        var_25 = wp::mul(var_17, var_24);
        var_26 = wp::add(var_22, var_25);
        var_27 = wp::mul(var_1, var_26);
    }
    var_28 = wp::where(var_2, var_27, var_1);
    // return wp.max(MJ_MINVAL, R)                                                            <L 783>
    var_30 = wp::max(var_29, var_28);
    return var_30;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:524
static CUDA_CALLABLE wp::float32 muscle_bias_0(
    wp::float32 var_len,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::vec_t<2, wp::float32> var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 5;
    wp::float32 var_10;
    const wp::int32 var_11 = 7;
    wp::float32 var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    const wp::float32 var_15 = 1e-15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    const wp::float32 var_39 = 0.5;
    const wp::float32 var_40 = 1.0;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 1.0;
    bool var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    const wp::float32 var_47 = 1.0;
    wp::float32 var_48;
    const wp::float32 var_49 = 1.0;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    const wp::float32 var_55 = 0.5;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::float32 var_60 = 1.0;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::float32 var_66 = 0.5;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    //---------
    // forward
    // def muscle_bias(len: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 525>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 531>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 532>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 533>
    var_8 = wp::extract(var_prm, var_7);
    // lmax = prm[5]                                                                          <L 534>
    var_10 = wp::extract(var_prm, var_9);
    // fpmax = prm[7]                                                                         <L 535>
    var_12 = wp::extract(var_prm, var_11);
    // if force < 0.0:                                                                        <L 538>
    var_14 = (var_6 < var_13);
    if (var_14) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 539>
        var_16 = wp::max(var_15, var_acc0);
        var_17 = wp::div(var_8, var_16);
    }
    var_18 = wp::where(var_14, var_17, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 542>
    var_20 = wp::extract(var_lengthrange, var_19);
    var_22 = wp::extract(var_lengthrange, var_21);
    var_23 = wp::sub(var_20, var_22);
    var_25 = wp::extract(var_4, var_24);
    var_27 = wp::extract(var_4, var_26);
    var_28 = wp::sub(var_25, var_27);
    var_29 = wp::max(var_15, var_28);
    var_30 = wp::div(var_23, var_29);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 545>
    var_32 = wp::extract(var_4, var_31);
    var_34 = wp::extract(var_lengthrange, var_33);
    var_35 = wp::sub(var_len, var_34);
    var_36 = wp::max(var_15, var_30);
    var_37 = wp::div(var_35, var_36);
    var_38 = wp::add(var_32, var_37);
    // b = 0.5 * (1.0 + lmax)                                                                 <L 548>
    var_41 = wp::add(var_40, var_10);
    var_42 = wp::mul(var_39, var_41);
    // if L <= 1.0:                                                                           <L 549>
    var_44 = (var_38 <= var_43);
    if (var_44) {
        // return 0.0                                                                         <L 550>
        return var_45;
    }
    if (!var_44) {
        // elif L <= b:                                                                       <L 551>
        var_46 = (var_38 <= var_42);
        if (var_46) {
            // x = (L - 1.0) / wp.max(MJ_MINVAL, b - 1.0)                                     <L 552>
            var_48 = wp::sub(var_38, var_47);
            var_50 = wp::sub(var_42, var_49);
            var_51 = wp::max(var_15, var_50);
            var_52 = wp::div(var_48, var_51);
            // return -force * fpmax * 0.5 * x * x                                            <L 553>
            var_53 = wp::neg(var_18);
            var_54 = wp::mul(var_53, var_12);
            var_56 = wp::mul(var_54, var_55);
            var_57 = wp::mul(var_56, var_52);
            var_58 = wp::mul(var_57, var_52);
            return var_58;
        }
        if (!var_46) {
            // x = (L - b) / wp.max(MJ_MINVAL, b - 1.0)                                       <L 555>
            var_59 = wp::sub(var_38, var_42);
            var_61 = wp::sub(var_42, var_60);
            var_62 = wp::max(var_15, var_61);
            var_63 = wp::div(var_59, var_62);
            // return -force * fpmax * (0.5 + x)                                              <L 556>
            var_64 = wp::neg(var_18);
            var_65 = wp::mul(var_64, var_12);
            var_67 = wp::add(var_66, var_63);
            var_68 = wp::mul(var_65, var_67);
            return var_68;
        }
        var_69 = wp::where(var_46, var_52, var_63);
    }
    return {};
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\forward.py:167
static CUDA_CALLABLE wp::float32 _wrap_period_0(
    wp::array_t<wp::int32> var_jnt_type,
    wp::int32 var_trntype,
    wp::int32 var_dyntype,
    wp::int32 var_gaintype,
    wp::int32 var_biastype,
    wp::vec_t<2, wp::int32> var_trnid,
    wp::vec_t<10, wp::float32> var_gainprm,
    wp::vec_t<10, wp::float32> var_biasprm,
    wp::vec_t<6, wp::float32> var_gear)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    const wp::int32 var_3 = 1;
    bool var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    bool var_10;
    bool var_11;
    const wp::int32 var_12 = 0;
    bool var_13;
    const wp::int32 var_14 = 1;
    bool var_15;
    bool var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    const wp::int32 var_19 = 4;
    bool var_20;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    const wp::int32 var_23 = 0;
    bool var_24;
    const wp::int32 var_25 = 0;
    wp::float32 var_26;
    const wp::float32 var_27 = 0.0;
    bool var_28;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    const wp::float32 var_31 = 0.0;
    bool var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    const wp::float32 var_35 = 0.0;
    bool var_36;
    const wp::int32 var_37 = 3;
    wp::float32 var_38;
    const wp::int32 var_39 = 4;
    wp::float32 var_40;
    const wp::int32 var_41 = 5;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::float32 var_44 = 6.283185307179586;
    wp::float32 var_45;
    wp::float32 var_46;
    bool var_47;
    const wp::int32 var_48 = 0;
    bool var_49;
    const wp::int32 var_50 = 1;
    bool var_51;
    const wp::int32 var_52 = 0;
    wp::int32 var_53;
    wp::int32* var_54;
    const wp::int32 var_55 = 1;
    bool var_56;
    wp::int32 var_57;
    const wp::int32 var_58 = 0;
    wp::float32 var_59;
    const wp::int32 var_60 = 1;
    wp::float32 var_61;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    const wp::float32 var_65 = 6.283185307179586;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::float32 var_68 = 0.0;
    //---------
    // forward
    // def _wrap_period(                                                                      <L 168>
    // is_servo = (                                                                           <L 182>
    // gaintype == GainType.FIXED                                                             <L 183>
    var_2 = (var_gaintype == var_1);
    var_0 = var_2;
    if (var_0) {
        // and biastype == BiasType.AFFINE                                                    <L 184>
        var_4 = (var_biastype == var_3);
        var_0 = var_0 && var_4;
    }
    if (var_0) {
        // and gainprm[0] == -biasprm[1]                                                      <L 185>
        var_6 = wp::extract(var_gainprm, var_5);
        var_8 = wp::extract(var_biasprm, var_7);
        var_9 = wp::neg(var_8);
        var_10 = (var_6 == var_9);
        var_0 = var_0 && var_10;
    }
    if (var_0) {
        // and (dyntype == DynType.NONE or dyntype == DynType.INTEGRATOR)                     <L 186>
        var_13 = (var_dyntype == var_12);
        var_11 = var_13;
        if (!var_11) {
            var_15 = (var_dyntype == var_14);
            var_11 = var_11 || var_15;
        }
        var_0 = var_0 && var_11;
    }
    // if not is_servo:                                                                       <L 188>
    var_16 = wp::unot(var_0);
    if (var_16) {
        // return 0.0                                                                         <L 189>
        return var_17;
    }
    // if trntype == TrnType.SITE and trnid[1] >= 0 and gear[0] == 0.0 and gear[1] == 0.0 and gear[2] == 0.0:       <L 192>
    var_20 = (var_trntype == var_19);
    var_18 = var_20;
    if (var_18) {
        var_22 = wp::extract(var_trnid, var_21);
        var_24 = (var_22 >= var_23);
        var_18 = var_18 && var_24;
    }
    if (var_18) {
        var_26 = wp::extract(var_gear, var_25);
        var_28 = (var_26 == var_27);
        var_18 = var_18 && var_28;
    }
    if (var_18) {
        var_30 = wp::extract(var_gear, var_29);
        var_32 = (var_30 == var_31);
        var_18 = var_18 && var_32;
    }
    if (var_18) {
        var_34 = wp::extract(var_gear, var_33);
        var_36 = (var_34 == var_35);
        var_18 = var_18 && var_36;
    }
    if (var_18) {
        // rot_gear = wp.vec3(gear[3], gear[4], gear[5])                                      <L 193>
        var_38 = wp::extract(var_gear, var_37);
        var_40 = wp::extract(var_gear, var_39);
        var_42 = wp::extract(var_gear, var_41);
        var_43 = wp::vec_t<3, wp::float32>(var_38, var_40, var_42);
        // return wp.static(2.0 * wp.pi) * wp.length(rot_gear)                                <L 194>
        var_45 = wp::length(var_43);
        var_46 = wp::mul(var_44, var_45);
        return var_46;
    }
    // if trntype == TrnType.JOINT or trntype == TrnType.JOINTINPARENT:                       <L 197>
    var_49 = (var_trntype == var_48);
    var_47 = var_49;
    if (!var_47) {
        var_51 = (var_trntype == var_50);
        var_47 = var_47 || var_51;
    }
    if (var_47) {
        // if jnt_type[trnid[0]] == JointType.BALL:                                           <L 198>
        var_53 = wp::extract(var_trnid, var_52);
        var_54 = wp::address(var_jnt_type, var_53);
        var_57 = wp::load(var_54);
        var_56 = (var_57 == var_55);
        if (var_56) {
            // lin_gear = wp.vec3(gear[0], gear[1], gear[2])                                  <L 199>
            var_59 = wp::extract(var_gear, var_58);
            var_61 = wp::extract(var_gear, var_60);
            var_63 = wp::extract(var_gear, var_62);
            var_64 = wp::vec_t<3, wp::float32>(var_59, var_61, var_63);
            // return wp.static(2.0 * wp.pi) * wp.length(lin_gear)                            <L 200>
            var_66 = wp::length(var_64);
            var_67 = wp::mul(var_65, var_66);
            return var_67;
        }
    }
    // return 0.0                                                                             <L 202>
    return var_68;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:900
static CUDA_CALLABLE wp::float32 _poly_force_deriv_0(
    wp::float32 var_linear,
    wp::vec_t<2, wp::float32> var_poly,
    wp::float32 var_x,
    wp::int32 var_flg_odd)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    bool var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 2.0;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 3.0;
    const wp::int32 var_12 = 1;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    //---------
    // forward
    // def _poly_force_deriv(linear: float, poly: wp.vec2, x: float, flg_odd: int) -> float:       <L 901>
    // x_val = wp.where(flg_odd == 1, wp.abs(x), x)                                           <L 902>
    var_1 = (var_flg_odd == var_0);
    var_2 = wp::abs(var_x);
    var_3 = wp::where(var_1, var_2, var_x);
    // res = linear                                                                           <L 903>
    var_4 = wp::copy(var_linear);
    // res += 2.0 * poly[0] * x_val                                                           <L 904>
    var_7 = wp::extract(var_poly, var_6);
    var_8 = wp::mul(var_5, var_7);
    var_9 = wp::mul(var_8, var_3);
    var_10 = wp::add(var_4, var_9);
    // res += 3.0 * poly[1] * x_val * x_val                                                   <L 905>
    var_13 = wp::extract(var_poly, var_12);
    var_14 = wp::mul(var_11, var_13);
    var_15 = wp::mul(var_14, var_3);
    var_16 = wp::mul(var_15, var_3);
    var_17 = wp::add(var_10, var_16);
    // return res                                                                             <L 906>
    return var_17;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:339
static CUDA_CALLABLE void free_bias_vel_blocks_0(
    wp::float32 var_mass,
    wp::mat_t<3, 3, wp::float32> var_R,
    wp::mat_t<3, 3, wp::float32> var_Iw,
    wp::vec_t<3, wp::float32> var_s,
    wp::vec_t<3, wp::float32> var_qvel_rot,
    wp::mat_t<3, 3, wp::float32> & ret_0,
    wp::mat_t<3, 3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    const wp::int32 var_5 = 3;
    wp::mat_t<3, 3, wp::float32> var_6;
    wp::mat_t<3, 3, wp::float32> var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::mat_t<3, 3, wp::float32> var_11;
    wp::mat_t<3, 3, wp::float32> var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::mat_t<3, 3, wp::float32> var_15;
    const wp::float32 var_16 = 0.0;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    const wp::int32 var_19 = 1;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::int32 var_22 = 2;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::float32 var_25 = 0.0;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    const wp::int32 var_28 = 1;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_34;
    wp::float32 var_35;
    wp::mat_t<3, 3, wp::float32> var_36;
    wp::mat_t<3, 3, wp::float32> var_37;
    wp::mat_t<3, 3, wp::float32> var_38;
    wp::mat_t<3, 3, wp::float32> var_39;
    wp::mat_t<3, 3, wp::float32> var_40;
    wp::mat_t<3, 3, wp::float32> var_41;
    //---------
    // forward
    // def free_bias_vel_blocks(                                                              <L 340>
    // w = R @ qvel_rot                                                                       <L 349>
    var_0 = wp::mul(var_R, var_qvel_rot);
    // ws = wp.cross(w, s)                                                                    <L 351>
    var_1 = wp::cross(var_0, var_s);
    // Iww = Iw @ w                                                                           <L 352>
    var_2 = wp::mul(var_Iw, var_0);
    // w_dot_s = wp.dot(w, s)                                                                 <L 354>
    var_3 = wp::dot(var_0, var_s);
    // K = wp.outer(s, w) - wp.identity(n=3, dtype=float) * w_dot_s + wp.skew(ws)             <L 355>
    var_4 = wp::outer(var_s, var_0);
    var_6 = wp::identity<3, wp::float32>();
    var_7 = wp::mul(var_6, var_3);
    var_8 = wp::sub(var_4, var_7);
    var_9 = wp::skew(var_1);
    var_10 = wp::add(var_8, var_9);
    // lin = K @ R                                                                            <L 357>
    var_11 = wp::mul(var_10, var_R);
    // s_x_K = wp.skew(s) @ K                                                                 <L 359>
    var_12 = wp::skew(var_s);
    var_13 = wp::mul(var_12, var_10);
    // w_x_Iw = wp.skew(w) @ Iw                                                               <L 360>
    var_14 = wp::skew(var_0);
    var_15 = wp::mul(var_14, var_Iw);
    // neg_skew_Iww = wp.mat33(                                                               <L 361>
    // 0.0,                                                                                   <L 362>
    // Iww[2],                                                                                <L 363>
    var_18 = wp::extract(var_2, var_17);
    // -Iww[1],                                                                               <L 364>
    var_20 = wp::extract(var_2, var_19);
    var_21 = wp::neg(var_20);
    // -Iww[2],                                                                               <L 365>
    var_23 = wp::extract(var_2, var_22);
    var_24 = wp::neg(var_23);
    // 0.0,                                                                                   <L 366>
    // Iww[0],                                                                                <L 367>
    var_27 = wp::extract(var_2, var_26);
    // Iww[1],                                                                                <L 368>
    var_29 = wp::extract(var_2, var_28);
    // -Iww[0],                                                                               <L 369>
    var_31 = wp::extract(var_2, var_30);
    var_32 = wp::neg(var_31);
    // 0.0,                                                                                   <L 370>
    var_34 = wp::mat_t<3, 3, wp::float32>(var_16, var_18, var_21, var_24, var_25, var_27, var_29, var_32, var_33);
    // C = -mass * s_x_K + w_x_Iw + neg_skew_Iww                                              <L 373>
    var_35 = wp::neg(var_mass);
    var_36 = wp::mul(var_35, var_13);
    var_37 = wp::add(var_36, var_15);
    var_38 = wp::add(var_37, var_34);
    // rot = wp.transpose(R) @ C @ R                                                          <L 374>
    var_39 = wp::transpose(var_R);
    var_40 = wp::mul(var_39, var_38);
    var_41 = wp::mul(var_40, var_R);
    // return lin, rot                                                                        <L 376>
    ret_0 = var_11;
    ret_1 = var_41;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\passive.py:53
static CUDA_CALLABLE wp::vec_t<3, wp::float32> geom_semiaxes_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geom_type)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 2;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 3;
    bool var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    const wp::int32 var_9 = 1;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::int32 var_13 = 5;
    bool var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    //---------
    // forward
    // def geom_semiaxes(size: wp.vec3, geom_type: int) -> wp.vec3:  # kernel_analyzer: ignore       <L 54>
    // if geom_type == GeomType.SPHERE:                                                       <L 55>
    var_1 = (var_geom_type == var_0);
    if (var_1) {
        // r = size[0]                                                                        <L 56>
        var_3 = wp::extract(var_size, var_2);
        // return wp.vec3(r, r, r)                                                            <L 57>
        var_4 = wp::vec_t<3, wp::float32>(var_3, var_3, var_3);
        return var_4;
    }
    // if geom_type == GeomType.CAPSULE:                                                      <L 59>
    var_6 = (var_geom_type == var_5);
    if (var_6) {
        // radius = size[0]                                                                   <L 60>
        var_8 = wp::extract(var_size, var_7);
        // half_length = size[1]                                                              <L 61>
        var_10 = wp::extract(var_size, var_9);
        // return wp.vec3(radius, radius, half_length + radius)                               <L 62>
        var_11 = wp::add(var_10, var_8);
        var_12 = wp::vec_t<3, wp::float32>(var_8, var_8, var_11);
        return var_12;
    }
    // if geom_type == GeomType.CYLINDER:                                                     <L 64>
    var_14 = (var_geom_type == var_13);
    if (var_14) {
        // radius = size[0]                                                                   <L 65>
        var_16 = wp::extract(var_size, var_15);
        // half_length = size[1]                                                              <L 66>
        var_18 = wp::extract(var_size, var_17);
        // return wp.vec3(radius, radius, half_length)                                        <L 67>
        var_19 = wp::vec_t<3, wp::float32>(var_16, var_16, var_18);
        return var_19;
    }
    var_20 = wp::where(var_14, var_16, var_8);
    var_21 = wp::where(var_14, var_18, var_10);
    // return size                                                                            <L 70>
    return var_size;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\passive.py:47
static CUDA_CALLABLE wp::float32 _pow4_0(
    wp::float32 var_val)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    //---------
    // forward
    // def _pow4(val: float) -> float:                                                        <L 48>
    // sq = val * val                                                                         <L 49>
    var_0 = wp::mul(var_val, var_val);
    // return sq * sq                                                                         <L 50>
    var_1 = wp::mul(var_0, var_0);
    return var_1;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\passive.py:73
static CUDA_CALLABLE wp::float32 ellipsoid_max_moment_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_dir)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::int32 var_1 = 1;
    wp::int32 var_2;
    const wp::int32 var_3 = 3;
    wp::int32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 2;
    wp::int32 var_7;
    const wp::int32 var_8 = 3;
    wp::int32 var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 1.6755160819145563;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    //---------
    // forward
    // def ellipsoid_max_moment(size: wp.vec3, dir: int) -> float:                            <L 74>
    // d0 = size[dir]                                                                         <L 75>
    var_0 = wp::extract(var_size, var_dir);
    // d1 = size[(dir + 1) % 3]                                                               <L 76>
    var_2 = wp::add(var_dir, var_1);
    var_4 = wp::mod(var_2, var_3);
    var_5 = wp::extract(var_size, var_4);
    // d2 = size[(dir + 2) % 3]                                                               <L 77>
    var_7 = wp::add(var_dir, var_6);
    var_9 = wp::mod(var_7, var_8);
    var_10 = wp::extract(var_size, var_9);
    // return wp.static(8.0 / 15.0 * wp.pi) * d0 * _pow4(wp.max(d1, d2))                      <L 78>
    var_12 = wp::mul(var_11, var_0);
    var_13 = wp::max(var_5, var_10);
    var_14 = _pow4_0(var_13);
    var_15 = wp::mul(var_12, var_14);
    return var_15;
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:605
static CUDA_CALLABLE void _geom_ellipsoid_fluid_B_0(
    wp::vec_t<3, wp::float32> var_semiaxes,
    wp::float32 var_blunt_drag_coef,
    wp::float32 var_slender_drag_coef,
    wp::float32 var_ang_drag_coef,
    wp::float32 var_kutta_lift_coef,
    wp::float32 var_magnus_lift_coef,
    wp::vec_t<3, wp::float32> var_virtual_mass,
    wp::vec_t<3, wp::float32> var_virtual_inertia,
    wp::vec_t<3, wp::float32> var_ang_vel,
    wp::vec_t<3, wp::float32> var_lin_vel,
    wp::float32 var_density,
    wp::float32 var_viscosity,
    bool var_flg_dragonly,
    wp::mat_t<3, 3, wp::float32> & ret_0,
    wp::mat_t<3, 3, wp::float32> & ret_1,
    wp::mat_t<3, 3, wp::float32> & ret_2,
    wp::mat_t<3, 3, wp::float32> & ret_3)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_1;
    const wp::float32 var_2 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_3;
    const wp::float32 var_4 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_5;
    const wp::float32 var_6 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 0;
    wp::float32 var_17;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::int32 var_21 = 2;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::int32 var_24 = 0;
    wp::float32 var_25;
    const wp::int32 var_26 = 1;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 2;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::float32 var_34 = 1e-15;
    bool var_35;
    const wp::float32 var_36 = 1.0;
    wp::float32 var_37;
    const wp::float32 var_38 = 0.0;
    wp::float32 var_39;
    const wp::int32 var_40 = 0;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::int32 var_43 = 1;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 2;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    bool var_50;
    const wp::float32 var_51 = 1.0;
    wp::float32 var_52;
    const wp::float32 var_53 = 0.0;
    wp::float32 var_54;
    wp::vec_t<3, wp::float32> var_55;
    const wp::int32 var_56 = 0;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    const wp::int32 var_60 = 2;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    bool var_79;
    const wp::float32 var_80 = 1.0;
    wp::float32 var_81;
    const wp::float32 var_82 = 0.0;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    bool var_98;
    wp::mat_t<3, 3, wp::float32> var_99;
    wp::mat_t<3, 3, wp::float32> var_100;
    const wp::float32 var_101 = 0.0;
    bool var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    wp::mat_t<3, 3, wp::float32> var_107;
    wp::mat_t<3, 3, wp::float32> var_108;
    wp::mat_t<3, 3, wp::float32> var_109;
    wp::mat_t<3, 3, wp::float32> var_110;
    wp::mat_t<3, 3, wp::float32> var_111;
    wp::mat_t<3, 3, wp::float32> var_112;
    wp::mat_t<3, 3, wp::float32> var_113;
    wp::mat_t<3, 3, wp::float32> var_114;
    wp::mat_t<3, 3, wp::float32> var_115;
    wp::mat_t<3, 3, wp::float32> var_116;
    wp::mat_t<3, 3, wp::float32> var_117;
    wp::mat_t<3, 3, wp::float32> var_118;
    wp::mat_t<3, 3, wp::float32> var_119;
    wp::mat_t<3, 3, wp::float32> var_120;
    wp::mat_t<3, 3, wp::float32> var_121;
    wp::mat_t<3, 3, wp::float32> var_122;
    wp::mat_t<3, 3, wp::float32> var_123;
    wp::mat_t<3, 3, wp::float32> var_124;
    wp::mat_t<3, 3, wp::float32> var_125;
    const wp::float32 var_126 = 4.1887902047863905;
    const wp::int32 var_127 = 0;
    wp::float32 var_128;
    wp::float32 var_129;
    const wp::int32 var_130 = 1;
    wp::float32 var_131;
    wp::float32 var_132;
    const wp::int32 var_133 = 2;
    wp::float32 var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::mat_t<3, 3, wp::float32> var_138;
    wp::mat_t<3, 3, wp::float32> var_139;
    wp::mat_t<3, 3, wp::float32> var_140;
    wp::mat_t<3, 3, wp::float32> var_141;
    bool var_142;
    const wp::float32 var_143 = 3.141592653589793;
    wp::float32 var_144;
    wp::float32 var_145;
    wp::float32 var_146;
    wp::float32 var_147;
    wp::float32 var_148;
    wp::float32 var_149;
    const wp::float32 var_150 = 0.0;
    wp::float32 var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    wp::float32 var_161;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::float32 var_165;
    wp::float32 var_166;
    bool var_167;
    const wp::float32 var_168 = 1.0;
    wp::float32 var_169;
    const wp::float32 var_170 = 0.0;
    wp::float32 var_171;
    const wp::float32 var_172 = 1.0;
    wp::float32 var_173;
    wp::float32 var_174;
    wp::float32 var_175;
    wp::vec_t<3, wp::float32> var_176;
    wp::mat_t<3, 3, wp::float32> var_177;
    const wp::float32 var_178 = 2.0;
    wp::mat_t<3, 3, wp::float32> var_179;
    wp::vec_t<3, wp::float32> var_180;
    wp::float32 var_181;
    wp::float32 var_182;
    wp::float32 var_183;
    wp::float32 var_184;
    wp::float32 var_185;
    wp::float32 var_186;
    wp::float32 var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::vec_t<3, wp::float32> var_190;
    wp::mat_t<3, 3, wp::float32> var_191;
    wp::mat_t<3, 3, wp::float32> var_192;
    wp::mat_t<3, 3, wp::float32> var_193;
    wp::mat_t<3, 3, wp::float32> var_194;
    wp::mat_t<3, 3, wp::float32> var_195;
    wp::mat_t<3, 3, wp::float32> var_196;
    wp::mat_t<3, 3, wp::float32> var_197;
    wp::mat_t<3, 3, wp::float32> var_198;
    wp::mat_t<3, 3, wp::float32> var_199;
    wp::mat_t<3, 3, wp::float32> var_200;
    wp::mat_t<3, 3, wp::float32> var_201;
    wp::mat_t<3, 3, wp::float32> var_202;
    wp::mat_t<3, 3, wp::float32> var_203;
    const wp::float32 var_204 = 0.6666666666666666;
    const wp::int32 var_205 = 0;
    wp::float32 var_206;
    const wp::int32 var_207 = 1;
    wp::float32 var_208;
    wp::float32 var_209;
    const wp::int32 var_210 = 2;
    wp::float32 var_211;
    wp::float32 var_212;
    wp::float32 var_213;
    const wp::float32 var_214 = 3.141592653589793;
    wp::float32 var_215;
    wp::float32 var_216;
    const wp::float32 var_217 = 3.141592653589793;
    wp::float32 var_218;
    wp::float32 var_219;
    wp::float32 var_220;
    const wp::float32 var_221 = 9.42477796076938;
    wp::float32 var_222;
    wp::float32 var_223;
    wp::float32 var_224;
    wp::float32 var_225;
    wp::float32 var_226;
    wp::float32 var_227;
    wp::float32 var_228;
    wp::float32 var_229;
    wp::float32 var_230;
    wp::float32 var_231;
    bool var_232;
    const wp::float32 var_233 = 3.141592653589793;
    wp::float32 var_234;
    wp::float32 var_235;
    wp::float32 var_236;
    const wp::float32 var_237 = 0.0;
    wp::float32 var_238;
    wp::float32 var_239;
    wp::float32 var_240;
    wp::float32 var_241;
    wp::float32 var_242;
    wp::float32 var_243;
    wp::float32 var_244;
    wp::float32 var_245;
    wp::float32 var_246;
    wp::float32 var_247;
    wp::float32 var_248;
    wp::float32 var_249;
    wp::float32 var_250;
    wp::float32 var_251;
    wp::float32 var_252;
    wp::float32 var_253;
    wp::float32 var_254;
    wp::float32 var_255;
    wp::float32 var_256;
    wp::float32 var_257;
    wp::float32 var_258;
    wp::float32 var_259;
    wp::float32 var_260;
    wp::float32 var_261;
    wp::float32 var_262;
    wp::float32 var_263;
    wp::float32 var_264;
    wp::float32 var_265;
    wp::float32 var_266;
    wp::float32 var_267;
    wp::float32 var_268;
    wp::float32 var_269;
    wp::float32 var_270;
    wp::float32 var_271;
    wp::vec_t<3, wp::float32> var_272;
    wp::float32 var_273;
    wp::float32 var_274;
    wp::mat_t<3, 3, wp::float32> var_275;
    wp::vec_t<3, wp::float32> var_276;
    wp::mat_t<3, 3, wp::float32> var_277;
    wp::mat_t<3, 3, wp::float32> var_278;
    wp::float32 var_279;
    wp::float32 var_280;
    wp::mat_t<3, 3, wp::float32> var_281;
    wp::mat_t<3, 3, wp::float32> var_282;
    wp::mat_t<3, 3, wp::float32> var_283;
    wp::vec_t<3, wp::float32> var_284;
    wp::mat_t<3, 3, wp::float32> var_285;
    wp::mat_t<3, 3, wp::float32> var_286;
    wp::mat_t<3, 3, wp::float32> var_287;
    const wp::float32 var_288 = 3.141592653589793;
    wp::float32 var_289;
    wp::float32 var_290;
    wp::float32 var_291;
    const wp::float32 var_292 = 1.6755160819145563;
    wp::float32 var_293;
    wp::float32 var_294;
    wp::float32 var_295;
    wp::float32 var_296;
    wp::float32 var_297;
    const wp::int32 var_298 = 0;
    wp::float32 var_299;
    const wp::int32 var_300 = 1;
    wp::float32 var_301;
    const wp::int32 var_302 = 2;
    wp::float32 var_303;
    wp::vec_t<3, wp::float32> var_304;
    const wp::int32 var_305 = 0;
    wp::float32 var_306;
    wp::float32 var_307;
    const wp::int32 var_308 = 0;
    wp::float32 var_309;
    wp::float32 var_310;
    wp::float32 var_311;
    wp::float32 var_312;
    const wp::int32 var_313 = 1;
    wp::float32 var_314;
    wp::float32 var_315;
    const wp::int32 var_316 = 1;
    wp::float32 var_317;
    wp::float32 var_318;
    wp::float32 var_319;
    wp::float32 var_320;
    const wp::int32 var_321 = 2;
    wp::float32 var_322;
    wp::float32 var_323;
    const wp::int32 var_324 = 2;
    wp::float32 var_325;
    wp::float32 var_326;
    wp::float32 var_327;
    wp::float32 var_328;
    wp::vec_t<3, wp::float32> var_329;
    wp::vec_t<3, wp::float32> var_330;
    wp::float32 var_331;
    wp::float32 var_332;
    wp::float32 var_333;
    wp::vec_t<3, wp::float32> var_334;
    wp::vec_t<3, wp::float32> var_335;
    wp::vec_t<3, wp::float32> var_336;
    wp::float32 var_337;
    wp::float32 var_338;
    wp::float32 var_339;
    wp::mat_t<3, 3, wp::float32> var_340;
    wp::vec_t<3, wp::float32> var_341;
    wp::mat_t<3, 3, wp::float32> var_342;
    wp::mat_t<3, 3, wp::float32> var_343;
    wp::mat_t<3, 3, wp::float32> var_344;
    //---------
    // forward
    // def _geom_ellipsoid_fluid_B(                                                           <L 606>
    // B00 = wp.mat33(0.0)  # torque wrt ang_vel                                              <L 623>
    var_1 = wp::mat_t<3, 3, wp::float32>(var_0);
    // B01 = wp.mat33(0.0)  # torque wrt lin_vel                                              <L 624>
    var_3 = wp::mat_t<3, 3, wp::float32>(var_2);
    // B10 = wp.mat33(0.0)  # force wrt ang_vel                                               <L 625>
    var_5 = wp::mat_t<3, 3, wp::float32>(var_4);
    // B11 = wp.mat33(0.0)  # force wrt lin_vel                                               <L 626>
    var_7 = wp::mat_t<3, 3, wp::float32>(var_6);
    // d_max = wp.max(wp.max(semiaxes[0], semiaxes[1]), semiaxes[2])                          <L 628>
    var_9 = wp::extract(var_semiaxes, var_8);
    var_11 = wp::extract(var_semiaxes, var_10);
    var_12 = wp::max(var_9, var_11);
    var_14 = wp::extract(var_semiaxes, var_13);
    var_15 = wp::max(var_12, var_14);
    // d_min = wp.min(wp.min(semiaxes[0], semiaxes[1]), semiaxes[2])                          <L 629>
    var_17 = wp::extract(var_semiaxes, var_16);
    var_19 = wp::extract(var_semiaxes, var_18);
    var_20 = wp::min(var_17, var_19);
    var_22 = wp::extract(var_semiaxes, var_21);
    var_23 = wp::min(var_20, var_22);
    // d_mid = semiaxes[0] + semiaxes[1] + semiaxes[2] - d_max - d_min                        <L 630>
    var_25 = wp::extract(var_semiaxes, var_24);
    var_27 = wp::extract(var_semiaxes, var_26);
    var_28 = wp::add(var_25, var_27);
    var_30 = wp::extract(var_semiaxes, var_29);
    var_31 = wp::add(var_28, var_30);
    var_32 = wp::sub(var_31, var_15);
    var_33 = wp::sub(var_32, var_23);
    // inv_dmax = wp.where(d_max > MJ_MINVAL, 1.0 / d_max, 0.0)                               <L 631>
    var_35 = (var_15 > var_34);
    var_37 = wp::div(var_36, var_15);
    var_39 = wp::where(var_35, var_37, var_38);
    // s0 = semiaxes[0] * inv_dmax                                                            <L 632>
    var_41 = wp::extract(var_semiaxes, var_40);
    var_42 = wp::mul(var_41, var_39);
    // s1 = semiaxes[1] * inv_dmax                                                            <L 633>
    var_44 = wp::extract(var_semiaxes, var_43);
    var_45 = wp::mul(var_44, var_39);
    // s2 = semiaxes[2] * inv_dmax                                                            <L 634>
    var_47 = wp::extract(var_semiaxes, var_46);
    var_48 = wp::mul(var_47, var_39);
    // norm = wp.length(lin_vel)                                                              <L 636>
    var_49 = wp::length(var_lin_vel);
    // inv_norm = wp.where(norm > MJ_MINVAL, 1.0 / norm, 0.0)                                 <L 637>
    var_50 = (var_49 > var_34);
    var_52 = wp::div(var_51, var_49);
    var_54 = wp::where(var_50, var_52, var_53);
    // v = lin_vel * inv_norm                                                                 <L 638>
    var_55 = wp::mul(var_lin_vel, var_54);
    // x = v[0]                                                                               <L 639>
    var_57 = wp::extract(var_55, var_56);
    // y = v[1]                                                                               <L 640>
    var_59 = wp::extract(var_55, var_58);
    // z = v[2]                                                                               <L 641>
    var_61 = wp::extract(var_55, var_60);
    // xx = x * x                                                                             <L 642>
    var_62 = wp::mul(var_57, var_57);
    // yy = y * y                                                                             <L 643>
    var_63 = wp::mul(var_59, var_59);
    // zz = z * z                                                                             <L 644>
    var_64 = wp::mul(var_61, var_61);
    // s12_sq = (s1 * s2) * (s1 * s2)                                                         <L 646>
    var_65 = wp::mul(var_45, var_48);
    var_66 = wp::mul(var_45, var_48);
    var_67 = wp::mul(var_65, var_66);
    // s20_sq = (s2 * s0) * (s2 * s0)                                                         <L 647>
    var_68 = wp::mul(var_48, var_42);
    var_69 = wp::mul(var_48, var_42);
    var_70 = wp::mul(var_68, var_69);
    // s01_sq = (s0 * s1) * (s0 * s1)                                                         <L 648>
    var_71 = wp::mul(var_42, var_45);
    var_72 = wp::mul(var_42, var_45);
    var_73 = wp::mul(var_71, var_72);
    // proj_num = s12_sq * xx + s20_sq * yy + s01_sq * zz                                     <L 650>
    var_74 = wp::mul(var_67, var_62);
    var_75 = wp::mul(var_70, var_63);
    var_76 = wp::add(var_74, var_75);
    var_77 = wp::mul(var_73, var_64);
    var_78 = wp::add(var_76, var_77);
    // inv_proj_num = wp.where(proj_num > MJ_MINVAL, 1.0 / proj_num, 0.0)                     <L 651>
    var_79 = (var_78 > var_34);
    var_81 = wp::div(var_80, var_78);
    var_83 = wp::where(var_79, var_81, var_82);
    // a = s12_sq * inv_proj_num                                                              <L 652>
    var_84 = wp::mul(var_67, var_83);
    // b = s20_sq * inv_proj_num                                                              <L 653>
    var_85 = wp::mul(var_70, var_83);
    // c = s01_sq * inv_proj_num                                                              <L 654>
    var_86 = wp::mul(var_73, var_83);
    // aa = a * a                                                                             <L 655>
    var_87 = wp::mul(var_84, var_84);
    // bb = b * b                                                                             <L 656>
    var_88 = wp::mul(var_85, var_85);
    // cc = c * c                                                                             <L 657>
    var_89 = wp::mul(var_86, var_86);
    // proj_denom = aa * xx + bb * yy + cc * zz                                               <L 659>
    var_90 = wp::mul(var_87, var_62);
    var_91 = wp::mul(var_88, var_63);
    var_92 = wp::add(var_90, var_91);
    var_93 = wp::mul(var_89, var_64);
    var_94 = wp::add(var_92, var_93);
    // area_scale = d_max * d_max * wp.sqrt(proj_num)                                         <L 660>
    var_95 = wp::mul(var_15, var_15);
    var_96 = wp::sqrt(var_78);
    var_97 = wp::mul(var_95, var_96);
    // if not flg_dragonly:                                                                   <L 662>
    var_98 = wp::unot(var_flg_dragonly);
    if (var_98) {
        // skew_lin_vel = wp.skew(lin_vel)                                                    <L 663>
        var_99 = wp::skew(var_lin_vel);
        // skew_ang_vel = wp.skew(ang_vel)                                                    <L 664>
        var_100 = wp::skew(var_ang_vel);
        // if density > 0.0:                                                                  <L 665>
        var_102 = (var_density > var_101);
        if (var_102) {
            // density_vm = density * virtual_mass                                            <L 667>
            var_103 = wp::mul(var_density, var_virtual_mass);
            // density_vi = density * virtual_inertia                                         <L 668>
            var_104 = wp::mul(var_density, var_virtual_inertia);
            // virtual_lin_mom = wp.cw_mul(density_vm, lin_vel)                               <L 669>
            var_105 = wp::cw_mul(var_103, var_lin_vel);
            // virtual_ang_mom = wp.cw_mul(density_vi, ang_vel)                               <L 670>
            var_106 = wp::cw_mul(var_104, var_ang_vel);
            // skew_virtual_lin_mom = wp.skew(virtual_lin_mom)                                <L 671>
            var_107 = wp::skew(var_105);
            // B00 += wp.skew(virtual_ang_mom) - skew_ang_vel @ wp.diag(density_vi)           <L 674>
            var_108 = wp::skew(var_106);
            var_109 = wp::diag(var_104);
            var_110 = wp::mul(var_100, var_109);
            var_111 = wp::sub(var_108, var_110);
            var_112 = wp::add(var_1, var_111);
            // B01 += skew_virtual_lin_mom - skew_lin_vel @ wp.diag(density_vm)               <L 677>
            var_113 = wp::diag(var_103);
            var_114 = wp::mul(var_99, var_113);
            var_115 = wp::sub(var_107, var_114);
            var_116 = wp::add(var_3, var_115);
            // B10 += skew_virtual_lin_mom                                                    <L 680>
            var_117 = wp::add(var_5, var_107);
            // B11 += -skew_ang_vel @ wp.diag(density_vm)                                     <L 682>
            var_118 = wp::neg(var_100);
            var_119 = wp::diag(var_103);
            var_120 = wp::mul(var_118, var_119);
            var_121 = wp::add(var_7, var_120);
        }
        var_122 = wp::where(var_102, var_112, var_1);
        var_123 = wp::where(var_102, var_116, var_3);
        var_124 = wp::where(var_102, var_117, var_5);
        var_125 = wp::where(var_102, var_121, var_7);
        // volume = wp.static(4.0 / 3.0 * wp.pi) * semiaxes[0] * semiaxes[1] * semiaxes[2]       <L 685>
        var_128 = wp::extract(var_semiaxes, var_127);
        var_129 = wp::mul(var_126, var_128);
        var_131 = wp::extract(var_semiaxes, var_130);
        var_132 = wp::mul(var_129, var_131);
        var_134 = wp::extract(var_semiaxes, var_133);
        var_135 = wp::mul(var_132, var_134);
        // magnus_coef = magnus_lift_coef * density * volume                                  <L 686>
        var_136 = wp::mul(var_magnus_lift_coef, var_density);
        var_137 = wp::mul(var_136, var_135);
        // B10 -= skew_lin_vel * magnus_coef                                                  <L 687>
        var_138 = wp::mul(var_99, var_137);
        var_139 = wp::sub(var_124, var_138);
        // B11 += skew_ang_vel * magnus_coef                                                  <L 688>
        var_140 = wp::mul(var_100, var_137);
        var_141 = wp::add(var_125, var_140);
        // df_denom = wp.where(                                                               <L 691>
        // proj_denom > MJ_MINVAL,                                                            <L 692>
        var_142 = (var_94 > var_34);
        // wp.pi * kutta_lift_coef * density * area_scale * norm / wp.sqrt(proj_denom),       <L 693>
        var_144 = wp::mul(var_143, var_kutta_lift_coef);
        var_145 = wp::mul(var_144, var_density);
        var_146 = wp::mul(var_145, var_97);
        var_147 = wp::mul(var_146, var_49);
        var_148 = wp::sqrt(var_94);
        var_149 = wp::div(var_147, var_148);
        // 0.0,                                                                               <L 694>
        var_151 = wp::where(var_142, var_149, var_150);
        // dfx_coef = yy * (a - b) + zz * (a - c)                                             <L 697>
        var_152 = wp::sub(var_84, var_85);
        var_153 = wp::mul(var_63, var_152);
        var_154 = wp::sub(var_84, var_86);
        var_155 = wp::mul(var_64, var_154);
        var_156 = wp::add(var_153, var_155);
        // dfy_coef = xx * (b - a) + zz * (b - c)                                             <L 698>
        var_157 = wp::sub(var_85, var_84);
        var_158 = wp::mul(var_62, var_157);
        var_159 = wp::sub(var_85, var_86);
        var_160 = wp::mul(var_64, var_159);
        var_161 = wp::add(var_158, var_160);
        // dfz_coef = xx * (c - a) + yy * (c - b)                                             <L 699>
        var_162 = wp::sub(var_86, var_84);
        var_163 = wp::mul(var_62, var_162);
        var_164 = wp::sub(var_86, var_85);
        var_165 = wp::mul(var_63, var_164);
        var_166 = wp::add(var_163, var_165);
        // proj_term = wp.where(proj_denom > MJ_MINVAL, 1.0 / proj_denom, 0.0)                <L 700>
        var_167 = (var_94 > var_34);
        var_169 = wp::div(var_168, var_94);
        var_171 = wp::where(var_167, var_169, var_170);
        // cos_term = 1.0                                                                     <L 701>
        // D = wp.skew(wp.vec3(b - c, c - a, a - b)) * 2.0                                    <L 703>
        var_173 = wp::sub(var_85, var_86);
        var_174 = wp::sub(var_86, var_84);
        var_175 = wp::sub(var_84, var_85);
        var_176 = wp::vec_t<3, wp::float32>(var_173, var_174, var_175);
        var_177 = wp::skew(var_176);
        var_179 = wp::mul(var_177, var_178);
        // df_coef = wp.vec3(dfx_coef, dfy_coef, dfz_coef)                                    <L 705>
        var_180 = wp::vec_t<3, wp::float32>(var_156, var_161, var_166);
        // inner_term = wp.vec3(                                                              <L 706>
        // aa * proj_term - a + cos_term,                                                     <L 707>
        var_181 = wp::mul(var_87, var_171);
        var_182 = wp::sub(var_181, var_84);
        var_183 = wp::add(var_182, var_172);
        // bb * proj_term - b + cos_term,                                                     <L 708>
        var_184 = wp::mul(var_88, var_171);
        var_185 = wp::sub(var_184, var_85);
        var_186 = wp::add(var_185, var_172);
        // cc * proj_term - c + cos_term,                                                     <L 709>
        var_187 = wp::mul(var_89, var_171);
        var_188 = wp::sub(var_187, var_86);
        var_189 = wp::add(var_188, var_172);
        var_190 = wp::vec_t<3, wp::float32>(var_183, var_186, var_189);
        // D += wp.outer(df_coef, inner_term)                                                 <L 712>
        var_191 = wp::outer(var_180, var_190);
        var_192 = wp::add(var_179, var_191);
        // V = wp.diag(v)                                                                     <L 714>
        var_193 = wp::diag(var_55);
        // D = V @ D @ V - wp.diag(df_coef)                                                   <L 715>
        var_194 = wp::mul(var_193, var_192);
        var_195 = wp::mul(var_194, var_193);
        var_196 = wp::diag(var_180);
        var_197 = wp::sub(var_195, var_196);
        // D *= df_denom                                                                      <L 717>
        var_198 = wp::mul(var_197, var_151);
        // B11 += D                                                                           <L 718>
        var_199 = wp::add(var_141, var_198);
    }
    var_200 = wp::where(var_98, var_122, var_1);
    var_201 = wp::where(var_98, var_123, var_3);
    var_202 = wp::where(var_98, var_139, var_5);
    var_203 = wp::where(var_98, var_199, var_7);
    // eq_sphere_D = wp.static(2.0 / 3.0) * (semiaxes[0] + semiaxes[1] + semiaxes[2])         <L 721>
    var_206 = wp::extract(var_semiaxes, var_205);
    var_208 = wp::extract(var_semiaxes, var_207);
    var_209 = wp::add(var_206, var_208);
    var_211 = wp::extract(var_semiaxes, var_210);
    var_212 = wp::add(var_209, var_211);
    var_213 = wp::mul(var_204, var_212);
    // A_max = wp.pi * d_max * d_mid                                                          <L 722>
    var_215 = wp::mul(var_214, var_15);
    var_216 = wp::mul(var_215, var_33);
    // A_proj = wp.pi * area_scale * wp.sqrt(proj_denom)                                      <L 724>
    var_218 = wp::mul(var_217, var_97);
    var_219 = wp::sqrt(var_94);
    var_220 = wp::mul(var_218, var_219);
    // lin_coef = viscosity * wp.static(3.0 * wp.pi) * eq_sphere_D                            <L 726>
    var_222 = wp::mul(var_viscosity, var_221);
    var_223 = wp::mul(var_222, var_213);
    // quad_coef = density * (A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj))       <L 727>
    var_224 = wp::mul(var_220, var_blunt_drag_coef);
    var_225 = wp::sub(var_216, var_220);
    var_226 = wp::mul(var_slender_drag_coef, var_225);
    var_227 = wp::add(var_224, var_226);
    var_228 = wp::mul(var_density, var_227);
    // Aproj_coef = density * norm * (blunt_drag_coef - slender_drag_coef)                    <L 728>
    var_229 = wp::mul(var_density, var_49);
    var_230 = wp::sub(var_blunt_drag_coef, var_slender_drag_coef);
    var_231 = wp::mul(var_229, var_230);
    // dA_coef = wp.where(proj_denom > MJ_MINVAL, wp.pi * area_scale / wp.sqrt(proj_denom), 0.0)       <L 729>
    var_232 = (var_94 > var_34);
    var_234 = wp::mul(var_233, var_97);
    var_235 = wp::sqrt(var_94);
    var_236 = wp::div(var_234, var_235);
    var_238 = wp::where(var_232, var_236, var_237);
    // dAproj_dv = wp.vec3(                                                                   <L 731>
    // Aproj_coef * dA_coef * a * x * (b * yy * (a - b) + c * zz * (a - c)),                  <L 732>
    var_239 = wp::mul(var_231, var_238);
    var_240 = wp::mul(var_239, var_84);
    var_241 = wp::mul(var_240, var_57);
    var_242 = wp::mul(var_85, var_63);
    var_243 = wp::sub(var_84, var_85);
    var_244 = wp::mul(var_242, var_243);
    var_245 = wp::mul(var_86, var_64);
    var_246 = wp::sub(var_84, var_86);
    var_247 = wp::mul(var_245, var_246);
    var_248 = wp::add(var_244, var_247);
    var_249 = wp::mul(var_241, var_248);
    // Aproj_coef * dA_coef * b * y * (a * xx * (b - a) + c * zz * (b - c)),                  <L 733>
    var_250 = wp::mul(var_231, var_238);
    var_251 = wp::mul(var_250, var_85);
    var_252 = wp::mul(var_251, var_59);
    var_253 = wp::mul(var_84, var_62);
    var_254 = wp::sub(var_85, var_84);
    var_255 = wp::mul(var_253, var_254);
    var_256 = wp::mul(var_86, var_64);
    var_257 = wp::sub(var_85, var_86);
    var_258 = wp::mul(var_256, var_257);
    var_259 = wp::add(var_255, var_258);
    var_260 = wp::mul(var_252, var_259);
    // Aproj_coef * dA_coef * c * z * (a * xx * (c - a) + b * yy * (c - b)),                  <L 734>
    var_261 = wp::mul(var_231, var_238);
    var_262 = wp::mul(var_261, var_86);
    var_263 = wp::mul(var_262, var_61);
    var_264 = wp::mul(var_84, var_62);
    var_265 = wp::sub(var_86, var_84);
    var_266 = wp::mul(var_264, var_265);
    var_267 = wp::mul(var_85, var_63);
    var_268 = wp::sub(var_86, var_85);
    var_269 = wp::mul(var_267, var_268);
    var_270 = wp::add(var_266, var_269);
    var_271 = wp::mul(var_263, var_270);
    var_272 = wp::vec_t<3, wp::float32>(var_249, var_260, var_271);
    // inner = xx + yy + zz                                                                   <L 737>
    var_273 = wp::add(var_62, var_63);
    var_274 = wp::add(var_273, var_64);
    // D = (wp.outer(v, v) + wp.diag(wp.vec3(inner))) * (-quad_coef * norm)                   <L 738>
    var_275 = wp::outer(var_55, var_55);
    var_276 = wp::vec_t<3, wp::float32>(var_274);
    var_277 = wp::diag(var_276);
    var_278 = wp::add(var_275, var_277);
    var_279 = wp::neg(var_228);
    var_280 = wp::mul(var_279, var_49);
    var_281 = wp::mul(var_278, var_280);
    // D -= wp.outer(v, dAproj_dv)                                                            <L 739>
    var_282 = wp::outer(var_55, var_272);
    var_283 = wp::sub(var_281, var_282);
    // D -= wp.diag(wp.vec3(lin_coef))                                                        <L 740>
    var_284 = wp::vec_t<3, wp::float32>(var_223);
    var_285 = wp::diag(var_284);
    var_286 = wp::sub(var_283, var_285);
    // B11 += D                                                                               <L 742>
    var_287 = wp::add(var_203, var_286);
    // lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D                   <L 745>
    var_289 = wp::mul(var_288, var_213);
    var_290 = wp::mul(var_289, var_213);
    var_291 = wp::mul(var_290, var_213);
    // I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * d_max * d_max * d_max * d_max          <L 746>
    var_293 = wp::mul(var_292, var_33);
    var_294 = wp::mul(var_293, var_15);
    var_295 = wp::mul(var_294, var_15);
    var_296 = wp::mul(var_295, var_15);
    var_297 = wp::mul(var_296, var_15);
    // II = wp.vec3(                                                                          <L 747>
    // ellipsoid_max_moment(semiaxes, 0),                                                     <L 748>
    var_299 = ellipsoid_max_moment_0(var_semiaxes, var_298);
    // ellipsoid_max_moment(semiaxes, 1),                                                     <L 749>
    var_301 = ellipsoid_max_moment_0(var_semiaxes, var_300);
    // ellipsoid_max_moment(semiaxes, 2),                                                     <L 750>
    var_303 = ellipsoid_max_moment_0(var_semiaxes, var_302);
    var_304 = wp::vec_t<3, wp::float32>(var_299, var_301, var_303);
    // mom_coef = wp.vec3(                                                                    <L 753>
    // ang_drag_coef * II[0] + slender_drag_coef * (I_max - II[0]),                           <L 754>
    var_306 = wp::extract(var_304, var_305);
    var_307 = wp::mul(var_ang_drag_coef, var_306);
    var_309 = wp::extract(var_304, var_308);
    var_310 = wp::sub(var_297, var_309);
    var_311 = wp::mul(var_slender_drag_coef, var_310);
    var_312 = wp::add(var_307, var_311);
    // ang_drag_coef * II[1] + slender_drag_coef * (I_max - II[1]),                           <L 755>
    var_314 = wp::extract(var_304, var_313);
    var_315 = wp::mul(var_ang_drag_coef, var_314);
    var_317 = wp::extract(var_304, var_316);
    var_318 = wp::sub(var_297, var_317);
    var_319 = wp::mul(var_slender_drag_coef, var_318);
    var_320 = wp::add(var_315, var_319);
    // ang_drag_coef * II[2] + slender_drag_coef * (I_max - II[2]),                           <L 756>
    var_322 = wp::extract(var_304, var_321);
    var_323 = wp::mul(var_ang_drag_coef, var_322);
    var_325 = wp::extract(var_304, var_324);
    var_326 = wp::sub(var_297, var_325);
    var_327 = wp::mul(var_slender_drag_coef, var_326);
    var_328 = wp::add(var_323, var_327);
    var_329 = wp::vec_t<3, wp::float32>(var_312, var_320, var_328);
    // mom_visc = wp.cw_mul(ang_vel, mom_coef)                                                <L 759>
    var_330 = wp::cw_mul(var_ang_vel, var_329);
    // norm_mom = wp.length(mom_visc)                                                         <L 760>
    var_331 = wp::length(var_330);
    // density_scaled = math.safe_div(density, norm_mom)                                      <L 761>
    var_332 = safe_div_0(var_density, var_331);
    // mom_sq = -density_scaled * wp.cw_mul(wp.cw_mul(ang_vel, mom_coef), mom_coef)           <L 763>
    var_333 = wp::neg(var_332);
    var_334 = wp::cw_mul(var_ang_vel, var_329);
    var_335 = wp::cw_mul(var_334, var_329);
    var_336 = wp::mul(var_333, var_335);
    // torq_lin_coef = viscosity * lin_visc_torq_coef                                         <L 765>
    var_337 = wp::mul(var_viscosity, var_291);
    // diag_val = wp.dot(ang_vel, mom_sq) - torq_lin_coef                                     <L 766>
    var_338 = wp::dot(var_ang_vel, var_336);
    var_339 = wp::sub(var_338, var_337);
    // D = wp.outer(ang_vel, mom_sq) + wp.diag(wp.vec3(diag_val))                             <L 768>
    var_340 = wp::outer(var_ang_vel, var_336);
    var_341 = wp::vec_t<3, wp::float32>(var_339);
    var_342 = wp::diag(var_341);
    var_343 = wp::add(var_340, var_342);
    // B00 += D                                                                               <L 769>
    var_344 = wp::add(var_200, var_343);
    // return B00, B01, B10, B11                                                              <L 771>
    ret_0 = var_344;
    ret_1 = var_201;
    ret_2 = var_202;
    ret_3 = var_287;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\forward.py:734
static CUDA_CALLABLE wp::mat_t<6, 6, wp::float32> _project_spatial_B_0(
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::int32 var_worldid,
    wp::int32 var_dof_adr,
    wp::mat_t<3, 3, wp::float32> var_geom_rotT,
    wp::vec_t<3, wp::float32> var_offset,
    wp::mat_t<3, 3, wp::float32> var_B00,
    wp::mat_t<3, 3, wp::float32> var_B01,
    wp::mat_t<3, 3, wp::float32> var_B10,
    wp::mat_t<3, 3, wp::float32> var_B11)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::vec_t<6, wp::float32>* var_2;
    wp::vec_t<6, wp::float32> var_3;
    wp::vec_t<6, wp::float32> var_4;
    const wp::int32 var_5 = 1;
    wp::int32 var_6;
    wp::vec_t<6, wp::float32>* var_7;
    wp::vec_t<6, wp::float32> var_8;
    wp::vec_t<6, wp::float32> var_9;
    const wp::int32 var_10 = 2;
    wp::int32 var_11;
    wp::vec_t<6, wp::float32>* var_12;
    wp::vec_t<6, wp::float32> var_13;
    wp::vec_t<6, wp::float32> var_14;
    const wp::int32 var_15 = 3;
    wp::int32 var_16;
    wp::vec_t<6, wp::float32>* var_17;
    wp::vec_t<6, wp::float32> var_18;
    wp::vec_t<6, wp::float32> var_19;
    const wp::int32 var_20 = 4;
    wp::int32 var_21;
    wp::vec_t<6, wp::float32>* var_22;
    wp::vec_t<6, wp::float32> var_23;
    wp::vec_t<6, wp::float32> var_24;
    const wp::int32 var_25 = 5;
    wp::int32 var_26;
    wp::vec_t<6, wp::float32>* var_27;
    wp::vec_t<6, wp::float32> var_28;
    wp::vec_t<6, wp::float32> var_29;
    const wp::int32 var_30 = 3;
    wp::float32 var_31;
    const wp::int32 var_32 = 4;
    wp::float32 var_33;
    const wp::int32 var_34 = 5;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    const wp::int32 var_38 = 3;
    wp::float32 var_39;
    const wp::int32 var_40 = 4;
    wp::float32 var_41;
    const wp::int32 var_42 = 5;
    wp::float32 var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    const wp::int32 var_46 = 3;
    wp::float32 var_47;
    const wp::int32 var_48 = 4;
    wp::float32 var_49;
    const wp::int32 var_50 = 5;
    wp::float32 var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    const wp::int32 var_54 = 0;
    wp::float32 var_55;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    const wp::int32 var_58 = 2;
    wp::float32 var_59;
    wp::vec_t<3, wp::float32> var_60;
    const wp::int32 var_61 = 0;
    wp::float32 var_62;
    const wp::int32 var_63 = 1;
    wp::float32 var_64;
    const wp::int32 var_65 = 2;
    wp::float32 var_66;
    wp::vec_t<3, wp::float32> var_67;
    const wp::int32 var_68 = 0;
    wp::float32 var_69;
    const wp::int32 var_70 = 1;
    wp::float32 var_71;
    const wp::int32 var_72 = 2;
    wp::float32 var_73;
    wp::vec_t<3, wp::float32> var_74;
    const wp::int32 var_75 = 3;
    wp::float32 var_76;
    const wp::int32 var_77 = 4;
    wp::float32 var_78;
    const wp::int32 var_79 = 5;
    wp::float32 var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    const wp::int32 var_84 = 3;
    wp::float32 var_85;
    const wp::int32 var_86 = 4;
    wp::float32 var_87;
    const wp::int32 var_88 = 5;
    wp::float32 var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::vec_t<3, wp::float32> var_92;
    const wp::int32 var_93 = 3;
    wp::float32 var_94;
    const wp::int32 var_95 = 4;
    wp::float32 var_96;
    const wp::int32 var_97 = 5;
    wp::float32 var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    wp::vec_t<3, wp::float32> var_107;
    const wp::int32 var_108 = 0;
    wp::float32 var_109;
    const wp::int32 var_110 = 0;
    wp::float32 var_111;
    const wp::int32 var_112 = 0;
    wp::float32 var_113;
    const wp::int32 var_114 = 1;
    wp::float32 var_115;
    const wp::int32 var_116 = 1;
    wp::float32 var_117;
    const wp::int32 var_118 = 1;
    wp::float32 var_119;
    const wp::int32 var_120 = 2;
    wp::float32 var_121;
    const wp::int32 var_122 = 2;
    wp::float32 var_123;
    const wp::int32 var_124 = 2;
    wp::float32 var_125;
    wp::mat_t<3, 3, wp::float32> var_126;
    const wp::int32 var_127 = 0;
    wp::float32 var_128;
    const wp::int32 var_129 = 0;
    wp::float32 var_130;
    const wp::int32 var_131 = 0;
    wp::float32 var_132;
    const wp::int32 var_133 = 1;
    wp::float32 var_134;
    const wp::int32 var_135 = 1;
    wp::float32 var_136;
    const wp::int32 var_137 = 1;
    wp::float32 var_138;
    const wp::int32 var_139 = 2;
    wp::float32 var_140;
    const wp::int32 var_141 = 2;
    wp::float32 var_142;
    const wp::int32 var_143 = 2;
    wp::float32 var_144;
    wp::mat_t<3, 3, wp::float32> var_145;
    const wp::int32 var_146 = 0;
    wp::float32 var_147;
    const wp::int32 var_148 = 0;
    wp::float32 var_149;
    const wp::int32 var_150 = 0;
    wp::float32 var_151;
    const wp::int32 var_152 = 1;
    wp::float32 var_153;
    const wp::int32 var_154 = 1;
    wp::float32 var_155;
    const wp::int32 var_156 = 1;
    wp::float32 var_157;
    const wp::int32 var_158 = 2;
    wp::float32 var_159;
    const wp::int32 var_160 = 2;
    wp::float32 var_161;
    const wp::int32 var_162 = 2;
    wp::float32 var_163;
    wp::mat_t<3, 3, wp::float32> var_164;
    wp::mat_t<3, 3, wp::float32> var_165;
    wp::mat_t<3, 3, wp::float32> var_166;
    wp::mat_t<3, 3, wp::float32> var_167;
    wp::mat_t<3, 3, wp::float32> var_168;
    wp::mat_t<3, 3, wp::float32> var_169;
    wp::mat_t<3, 3, wp::float32> var_170;
    wp::mat_t<3, 3, wp::float32> var_171;
    wp::mat_t<3, 3, wp::float32> var_172;
    wp::mat_t<3, 3, wp::float32> var_173;
    wp::mat_t<3, 3, wp::float32> var_174;
    wp::mat_t<3, 3, wp::float32> var_175;
    wp::mat_t<3, 3, wp::float32> var_176;
    wp::mat_t<3, 3, wp::float32> var_177;
    wp::mat_t<3, 3, wp::float32> var_178;
    wp::mat_t<3, 3, wp::float32> var_179;
    wp::mat_t<3, 3, wp::float32> var_180;
    wp::mat_t<3, 3, wp::float32> var_181;
    wp::mat_t<3, 3, wp::float32> var_182;
    wp::mat_t<3, 3, wp::float32> var_183;
    const wp::float32 var_184 = 0.0;
    wp::mat_t<6, 6, wp::float32> var_185;
    const wp::int32 var_186 = 0;
    const wp::int32 var_187 = 0;
    wp::float32 var_188;
    wp::float32 var_189;
    const wp::int32 var_190 = 3;
    wp::int32 var_191;
    wp::float32 var_192;
    const wp::int32 var_193 = 3;
    wp::int32 var_194;
    wp::float32 var_195;
    const wp::int32 var_196 = 3;
    wp::int32 var_197;
    const wp::int32 var_198 = 3;
    wp::int32 var_199;
    const wp::int32 var_200 = 1;
    wp::float32 var_201;
    wp::float32 var_202;
    const wp::int32 var_203 = 3;
    wp::int32 var_204;
    wp::float32 var_205;
    const wp::int32 var_206 = 3;
    wp::int32 var_207;
    wp::float32 var_208;
    const wp::int32 var_209 = 3;
    wp::int32 var_210;
    const wp::int32 var_211 = 3;
    wp::int32 var_212;
    const wp::int32 var_213 = 2;
    wp::float32 var_214;
    wp::float32 var_215;
    const wp::int32 var_216 = 3;
    wp::int32 var_217;
    wp::float32 var_218;
    const wp::int32 var_219 = 3;
    wp::int32 var_220;
    wp::float32 var_221;
    const wp::int32 var_222 = 3;
    wp::int32 var_223;
    const wp::int32 var_224 = 3;
    wp::int32 var_225;
    const wp::int32 var_226 = 1;
    const wp::int32 var_227 = 0;
    wp::float32 var_228;
    wp::float32 var_229;
    const wp::int32 var_230 = 3;
    wp::int32 var_231;
    wp::float32 var_232;
    const wp::int32 var_233 = 3;
    wp::int32 var_234;
    wp::float32 var_235;
    const wp::int32 var_236 = 3;
    wp::int32 var_237;
    const wp::int32 var_238 = 3;
    wp::int32 var_239;
    const wp::int32 var_240 = 1;
    wp::float32 var_241;
    wp::float32 var_242;
    const wp::int32 var_243 = 3;
    wp::int32 var_244;
    wp::float32 var_245;
    const wp::int32 var_246 = 3;
    wp::int32 var_247;
    wp::float32 var_248;
    const wp::int32 var_249 = 3;
    wp::int32 var_250;
    const wp::int32 var_251 = 3;
    wp::int32 var_252;
    const wp::int32 var_253 = 2;
    wp::float32 var_254;
    wp::float32 var_255;
    const wp::int32 var_256 = 3;
    wp::int32 var_257;
    wp::float32 var_258;
    const wp::int32 var_259 = 3;
    wp::int32 var_260;
    wp::float32 var_261;
    const wp::int32 var_262 = 3;
    wp::int32 var_263;
    const wp::int32 var_264 = 3;
    wp::int32 var_265;
    const wp::int32 var_266 = 2;
    const wp::int32 var_267 = 0;
    wp::float32 var_268;
    wp::float32 var_269;
    const wp::int32 var_270 = 3;
    wp::int32 var_271;
    wp::float32 var_272;
    const wp::int32 var_273 = 3;
    wp::int32 var_274;
    wp::float32 var_275;
    const wp::int32 var_276 = 3;
    wp::int32 var_277;
    const wp::int32 var_278 = 3;
    wp::int32 var_279;
    const wp::int32 var_280 = 1;
    wp::float32 var_281;
    wp::float32 var_282;
    const wp::int32 var_283 = 3;
    wp::int32 var_284;
    wp::float32 var_285;
    const wp::int32 var_286 = 3;
    wp::int32 var_287;
    wp::float32 var_288;
    const wp::int32 var_289 = 3;
    wp::int32 var_290;
    const wp::int32 var_291 = 3;
    wp::int32 var_292;
    const wp::int32 var_293 = 2;
    wp::float32 var_294;
    wp::float32 var_295;
    const wp::int32 var_296 = 3;
    wp::int32 var_297;
    wp::float32 var_298;
    const wp::int32 var_299 = 3;
    wp::int32 var_300;
    wp::float32 var_301;
    const wp::int32 var_302 = 3;
    wp::int32 var_303;
    const wp::int32 var_304 = 3;
    wp::int32 var_305;
    //---------
    // forward
    // def _project_spatial_B(                                                                <L 735>
    // c0 = cdof_in[worldid, dof_adr + 0]                                                     <L 748>
    var_1 = wp::add(var_dof_adr, var_0);
    var_2 = wp::address(var_cdof_in, var_worldid, var_1);
    var_4 = wp::load(var_2);
    var_3 = wp::copy(var_4);
    // c1 = cdof_in[worldid, dof_adr + 1]                                                     <L 749>
    var_6 = wp::add(var_dof_adr, var_5);
    var_7 = wp::address(var_cdof_in, var_worldid, var_6);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // c2 = cdof_in[worldid, dof_adr + 2]                                                     <L 750>
    var_11 = wp::add(var_dof_adr, var_10);
    var_12 = wp::address(var_cdof_in, var_worldid, var_11);
    var_14 = wp::load(var_12);
    var_13 = wp::copy(var_14);
    // c3 = cdof_in[worldid, dof_adr + 3]                                                     <L 751>
    var_16 = wp::add(var_dof_adr, var_15);
    var_17 = wp::address(var_cdof_in, var_worldid, var_16);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // c4 = cdof_in[worldid, dof_adr + 4]                                                     <L 752>
    var_21 = wp::add(var_dof_adr, var_20);
    var_22 = wp::address(var_cdof_in, var_worldid, var_21);
    var_24 = wp::load(var_22);
    var_23 = wp::copy(var_24);
    // c5 = cdof_in[worldid, dof_adr + 5]                                                     <L 753>
    var_26 = wp::add(var_dof_adr, var_25);
    var_27 = wp::address(var_cdof_in, var_worldid, var_26);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // ll0 = geom_rotT @ wp.vec3(c0[3], c0[4], c0[5])                                         <L 755>
    var_31 = wp::extract(var_3, var_30);
    var_33 = wp::extract(var_3, var_32);
    var_35 = wp::extract(var_3, var_34);
    var_36 = wp::vec_t<3, wp::float32>(var_31, var_33, var_35);
    var_37 = wp::mul(var_geom_rotT, var_36);
    // ll1 = geom_rotT @ wp.vec3(c1[3], c1[4], c1[5])                                         <L 756>
    var_39 = wp::extract(var_8, var_38);
    var_41 = wp::extract(var_8, var_40);
    var_43 = wp::extract(var_8, var_42);
    var_44 = wp::vec_t<3, wp::float32>(var_39, var_41, var_43);
    var_45 = wp::mul(var_geom_rotT, var_44);
    // ll2 = geom_rotT @ wp.vec3(c2[3], c2[4], c2[5])                                         <L 757>
    var_47 = wp::extract(var_13, var_46);
    var_49 = wp::extract(var_13, var_48);
    var_51 = wp::extract(var_13, var_50);
    var_52 = wp::vec_t<3, wp::float32>(var_47, var_49, var_51);
    var_53 = wp::mul(var_geom_rotT, var_52);
    // a3 = wp.vec3(c3[0], c3[1], c3[2])                                                      <L 759>
    var_55 = wp::extract(var_18, var_54);
    var_57 = wp::extract(var_18, var_56);
    var_59 = wp::extract(var_18, var_58);
    var_60 = wp::vec_t<3, wp::float32>(var_55, var_57, var_59);
    // a4 = wp.vec3(c4[0], c4[1], c4[2])                                                      <L 760>
    var_62 = wp::extract(var_23, var_61);
    var_64 = wp::extract(var_23, var_63);
    var_66 = wp::extract(var_23, var_65);
    var_67 = wp::vec_t<3, wp::float32>(var_62, var_64, var_66);
    // a5 = wp.vec3(c5[0], c5[1], c5[2])                                                      <L 761>
    var_69 = wp::extract(var_28, var_68);
    var_71 = wp::extract(var_28, var_70);
    var_73 = wp::extract(var_28, var_72);
    var_74 = wp::vec_t<3, wp::float32>(var_69, var_71, var_73);
    // p3 = wp.vec3(c3[3], c3[4], c3[5]) + wp.cross(a3, offset)                               <L 763>
    var_76 = wp::extract(var_18, var_75);
    var_78 = wp::extract(var_18, var_77);
    var_80 = wp::extract(var_18, var_79);
    var_81 = wp::vec_t<3, wp::float32>(var_76, var_78, var_80);
    var_82 = wp::cross(var_60, var_offset);
    var_83 = wp::add(var_81, var_82);
    // p4 = wp.vec3(c4[3], c4[4], c4[5]) + wp.cross(a4, offset)                               <L 764>
    var_85 = wp::extract(var_23, var_84);
    var_87 = wp::extract(var_23, var_86);
    var_89 = wp::extract(var_23, var_88);
    var_90 = wp::vec_t<3, wp::float32>(var_85, var_87, var_89);
    var_91 = wp::cross(var_67, var_offset);
    var_92 = wp::add(var_90, var_91);
    // p5 = wp.vec3(c5[3], c5[4], c5[5]) + wp.cross(a5, offset)                               <L 765>
    var_94 = wp::extract(var_28, var_93);
    var_96 = wp::extract(var_28, var_95);
    var_98 = wp::extract(var_28, var_97);
    var_99 = wp::vec_t<3, wp::float32>(var_94, var_96, var_98);
    var_100 = wp::cross(var_74, var_offset);
    var_101 = wp::add(var_99, var_100);
    // la3 = geom_rotT @ a3                                                                   <L 767>
    var_102 = wp::mul(var_geom_rotT, var_60);
    // la4 = geom_rotT @ a4                                                                   <L 768>
    var_103 = wp::mul(var_geom_rotT, var_67);
    // la5 = geom_rotT @ a5                                                                   <L 769>
    var_104 = wp::mul(var_geom_rotT, var_74);
    // ll3 = geom_rotT @ p3                                                                   <L 771>
    var_105 = wp::mul(var_geom_rotT, var_83);
    // ll4 = geom_rotT @ p4                                                                   <L 772>
    var_106 = wp::mul(var_geom_rotT, var_92);
    // ll5 = geom_rotT @ p5                                                                   <L 773>
    var_107 = wp::mul(var_geom_rotT, var_101);
    // ll_trans = wp.mat33(                                                                   <L 775>
    // ll0[0],                                                                                <L 776>
    var_109 = wp::extract(var_37, var_108);
    // ll1[0],                                                                                <L 777>
    var_111 = wp::extract(var_45, var_110);
    // ll2[0],                                                                                <L 778>
    var_113 = wp::extract(var_53, var_112);
    // ll0[1],                                                                                <L 779>
    var_115 = wp::extract(var_37, var_114);
    // ll1[1],                                                                                <L 780>
    var_117 = wp::extract(var_45, var_116);
    // ll2[1],                                                                                <L 781>
    var_119 = wp::extract(var_53, var_118);
    // ll0[2],                                                                                <L 782>
    var_121 = wp::extract(var_37, var_120);
    // ll1[2],                                                                                <L 783>
    var_123 = wp::extract(var_45, var_122);
    // ll2[2],                                                                                <L 784>
    var_125 = wp::extract(var_53, var_124);
    var_126 = wp::mat_t<3, 3, wp::float32>(var_109, var_111, var_113, var_115, var_117, var_119, var_121, var_123, var_125);
    // la_rot = wp.mat33(                                                                     <L 786>
    // la3[0],                                                                                <L 787>
    var_128 = wp::extract(var_102, var_127);
    // la4[0],                                                                                <L 788>
    var_130 = wp::extract(var_103, var_129);
    // la5[0],                                                                                <L 789>
    var_132 = wp::extract(var_104, var_131);
    // la3[1],                                                                                <L 790>
    var_134 = wp::extract(var_102, var_133);
    // la4[1],                                                                                <L 791>
    var_136 = wp::extract(var_103, var_135);
    // la5[1],                                                                                <L 792>
    var_138 = wp::extract(var_104, var_137);
    // la3[2],                                                                                <L 793>
    var_140 = wp::extract(var_102, var_139);
    // la4[2],                                                                                <L 794>
    var_142 = wp::extract(var_103, var_141);
    // la5[2],                                                                                <L 795>
    var_144 = wp::extract(var_104, var_143);
    var_145 = wp::mat_t<3, 3, wp::float32>(var_128, var_130, var_132, var_134, var_136, var_138, var_140, var_142, var_144);
    // ll_rot = wp.mat33(                                                                     <L 797>
    // ll3[0],                                                                                <L 798>
    var_147 = wp::extract(var_105, var_146);
    // ll4[0],                                                                                <L 799>
    var_149 = wp::extract(var_106, var_148);
    // ll5[0],                                                                                <L 800>
    var_151 = wp::extract(var_107, var_150);
    // ll3[1],                                                                                <L 801>
    var_153 = wp::extract(var_105, var_152);
    // ll4[1],                                                                                <L 802>
    var_155 = wp::extract(var_106, var_154);
    // ll5[1],                                                                                <L 803>
    var_157 = wp::extract(var_107, var_156);
    // ll3[2],                                                                                <L 804>
    var_159 = wp::extract(var_105, var_158);
    // ll4[2],                                                                                <L 805>
    var_161 = wp::extract(var_106, var_160);
    // ll5[2],                                                                                <L 806>
    var_163 = wp::extract(var_107, var_162);
    var_164 = wp::mat_t<3, 3, wp::float32>(var_147, var_149, var_151, var_153, var_155, var_157, var_159, var_161, var_163);
    // ll_trans_T = wp.transpose(ll_trans)                                                    <L 809>
    var_165 = wp::transpose(var_126);
    // la_rot_T = wp.transpose(la_rot)                                                        <L 810>
    var_166 = wp::transpose(var_145);
    // ll_rot_T = wp.transpose(ll_rot)                                                        <L 811>
    var_167 = wp::transpose(var_164);
    // T_lin = B11 @ ll_trans                                                                 <L 813>
    var_168 = wp::mul(var_B11, var_126);
    // T_rot = B10 @ la_rot + B11 @ ll_rot                                                    <L 814>
    var_169 = wp::mul(var_B10, var_145);
    var_170 = wp::mul(var_B11, var_164);
    var_171 = wp::add(var_169, var_170);
    // top_left = ll_trans_T @ T_lin                                                          <L 816>
    var_172 = wp::mul(var_165, var_168);
    // top_right = ll_trans_T @ T_rot                                                         <L 817>
    var_173 = wp::mul(var_165, var_171);
    // bot_left = la_rot_T @ (B01 @ ll_trans) + ll_rot_T @ T_lin                              <L 819>
    var_174 = wp::mul(var_B01, var_126);
    var_175 = wp::mul(var_166, var_174);
    var_176 = wp::mul(var_167, var_168);
    var_177 = wp::add(var_175, var_176);
    // bot_right = la_rot_T @ (B00 @ la_rot + B01 @ ll_rot) + ll_rot_T @ T_rot                <L 820>
    var_178 = wp::mul(var_B00, var_145);
    var_179 = wp::mul(var_B01, var_164);
    var_180 = wp::add(var_178, var_179);
    var_181 = wp::mul(var_166, var_180);
    var_182 = wp::mul(var_167, var_171);
    var_183 = wp::add(var_181, var_182);
    // out = mat66(0.0)                                                                       <L 822>
    var_185 = wp::mat_t<6, 6, wp::float32>(var_184);
    // for r in range(3):                                                                     <L 823>
    // for c in range(3):                                                                     <L 824>
    // out[r, c] = top_left[r, c]                                                             <L 825>
    var_188 = wp::extract(var_172, var_186, var_187);
    wp::assign_inplace(var_185, var_186, var_187, var_188);
    // out[r, 3 + c] = top_right[r, c]                                                        <L 826>
    var_189 = wp::extract(var_173, var_186, var_187);
    var_191 = wp::add(var_190, var_187);
    wp::assign_inplace(var_185, var_186, var_191, var_189);
    // out[3 + r, c] = bot_left[r, c]                                                         <L 827>
    var_192 = wp::extract(var_177, var_186, var_187);
    var_194 = wp::add(var_193, var_186);
    wp::assign_inplace(var_185, var_194, var_187, var_192);
    // out[3 + r, 3 + c] = bot_right[r, c]                                                    <L 828>
    var_195 = wp::extract(var_183, var_186, var_187);
    var_197 = wp::add(var_196, var_186);
    var_199 = wp::add(var_198, var_187);
    wp::assign_inplace(var_185, var_197, var_199, var_195);
    // out[r, c] = top_left[r, c]                                                             <L 825>
    var_201 = wp::extract(var_172, var_186, var_200);
    wp::assign_inplace(var_185, var_186, var_200, var_201);
    // out[r, 3 + c] = top_right[r, c]                                                        <L 826>
    var_202 = wp::extract(var_173, var_186, var_200);
    var_204 = wp::add(var_203, var_200);
    wp::assign_inplace(var_185, var_186, var_204, var_202);
    // out[3 + r, c] = bot_left[r, c]                                                         <L 827>
    var_205 = wp::extract(var_177, var_186, var_200);
    var_207 = wp::add(var_206, var_186);
    wp::assign_inplace(var_185, var_207, var_200, var_205);
    // out[3 + r, 3 + c] = bot_right[r, c]                                                    <L 828>
    var_208 = wp::extract(var_183, var_186, var_200);
    var_210 = wp::add(var_209, var_186);
    var_212 = wp::add(var_211, var_200);
    wp::assign_inplace(var_185, var_210, var_212, var_208);
    // out[r, c] = top_left[r, c]                                                             <L 825>
    var_214 = wp::extract(var_172, var_186, var_213);
    wp::assign_inplace(var_185, var_186, var_213, var_214);
    // out[r, 3 + c] = top_right[r, c]                                                        <L 826>
    var_215 = wp::extract(var_173, var_186, var_213);
    var_217 = wp::add(var_216, var_213);
    wp::assign_inplace(var_185, var_186, var_217, var_215);
    // out[3 + r, c] = bot_left[r, c]                                                         <L 827>
    var_218 = wp::extract(var_177, var_186, var_213);
    var_220 = wp::add(var_219, var_186);
    wp::assign_inplace(var_185, var_220, var_213, var_218);
    // out[3 + r, 3 + c] = bot_right[r, c]                                                    <L 828>
    var_221 = wp::extract(var_183, var_186, var_213);
    var_223 = wp::add(var_222, var_186);
    var_225 = wp::add(var_224, var_213);
    wp::assign_inplace(var_185, var_223, var_225, var_221);
    // for c in range(3):                                                                     <L 824>
    // out[r, c] = top_left[r, c]                                                             <L 825>
    var_228 = wp::extract(var_172, var_226, var_227);
    wp::assign_inplace(var_185, var_226, var_227, var_228);
    // out[r, 3 + c] = top_right[r, c]                                                        <L 826>
    var_229 = wp::extract(var_173, var_226, var_227);
    var_231 = wp::add(var_230, var_227);
    wp::assign_inplace(var_185, var_226, var_231, var_229);
    // out[3 + r, c] = bot_left[r, c]                                                         <L 827>
    var_232 = wp::extract(var_177, var_226, var_227);
    var_234 = wp::add(var_233, var_226);
    wp::assign_inplace(var_185, var_234, var_227, var_232);
    // out[3 + r, 3 + c] = bot_right[r, c]                                                    <L 828>
    var_235 = wp::extract(var_183, var_226, var_227);
    var_237 = wp::add(var_236, var_226);
    var_239 = wp::add(var_238, var_227);
    wp::assign_inplace(var_185, var_237, var_239, var_235);
    // out[r, c] = top_left[r, c]                                                             <L 825>
    var_241 = wp::extract(var_172, var_226, var_240);
    wp::assign_inplace(var_185, var_226, var_240, var_241);
    // out[r, 3 + c] = top_right[r, c]                                                        <L 826>
    var_242 = wp::extract(var_173, var_226, var_240);
    var_244 = wp::add(var_243, var_240);
    wp::assign_inplace(var_185, var_226, var_244, var_242);
    // out[3 + r, c] = bot_left[r, c]                                                         <L 827>
    var_245 = wp::extract(var_177, var_226, var_240);
    var_247 = wp::add(var_246, var_226);
    wp::assign_inplace(var_185, var_247, var_240, var_245);
    // out[3 + r, 3 + c] = bot_right[r, c]                                                    <L 828>
    var_248 = wp::extract(var_183, var_226, var_240);
    var_250 = wp::add(var_249, var_226);
    var_252 = wp::add(var_251, var_240);
    wp::assign_inplace(var_185, var_250, var_252, var_248);
    // out[r, c] = top_left[r, c]                                                             <L 825>
    var_254 = wp::extract(var_172, var_226, var_253);
    wp::assign_inplace(var_185, var_226, var_253, var_254);
    // out[r, 3 + c] = top_right[r, c]                                                        <L 826>
    var_255 = wp::extract(var_173, var_226, var_253);
    var_257 = wp::add(var_256, var_253);
    wp::assign_inplace(var_185, var_226, var_257, var_255);
    // out[3 + r, c] = bot_left[r, c]                                                         <L 827>
    var_258 = wp::extract(var_177, var_226, var_253);
    var_260 = wp::add(var_259, var_226);
    wp::assign_inplace(var_185, var_260, var_253, var_258);
    // out[3 + r, 3 + c] = bot_right[r, c]                                                    <L 828>
    var_261 = wp::extract(var_183, var_226, var_253);
    var_263 = wp::add(var_262, var_226);
    var_265 = wp::add(var_264, var_253);
    wp::assign_inplace(var_185, var_263, var_265, var_261);
    // for c in range(3):                                                                     <L 824>
    // out[r, c] = top_left[r, c]                                                             <L 825>
    var_268 = wp::extract(var_172, var_266, var_267);
    wp::assign_inplace(var_185, var_266, var_267, var_268);
    // out[r, 3 + c] = top_right[r, c]                                                        <L 826>
    var_269 = wp::extract(var_173, var_266, var_267);
    var_271 = wp::add(var_270, var_267);
    wp::assign_inplace(var_185, var_266, var_271, var_269);
    // out[3 + r, c] = bot_left[r, c]                                                         <L 827>
    var_272 = wp::extract(var_177, var_266, var_267);
    var_274 = wp::add(var_273, var_266);
    wp::assign_inplace(var_185, var_274, var_267, var_272);
    // out[3 + r, 3 + c] = bot_right[r, c]                                                    <L 828>
    var_275 = wp::extract(var_183, var_266, var_267);
    var_277 = wp::add(var_276, var_266);
    var_279 = wp::add(var_278, var_267);
    wp::assign_inplace(var_185, var_277, var_279, var_275);
    // out[r, c] = top_left[r, c]                                                             <L 825>
    var_281 = wp::extract(var_172, var_266, var_280);
    wp::assign_inplace(var_185, var_266, var_280, var_281);
    // out[r, 3 + c] = top_right[r, c]                                                        <L 826>
    var_282 = wp::extract(var_173, var_266, var_280);
    var_284 = wp::add(var_283, var_280);
    wp::assign_inplace(var_185, var_266, var_284, var_282);
    // out[3 + r, c] = bot_left[r, c]                                                         <L 827>
    var_285 = wp::extract(var_177, var_266, var_280);
    var_287 = wp::add(var_286, var_266);
    wp::assign_inplace(var_185, var_287, var_280, var_285);
    // out[3 + r, 3 + c] = bot_right[r, c]                                                    <L 828>
    var_288 = wp::extract(var_183, var_266, var_280);
    var_290 = wp::add(var_289, var_266);
    var_292 = wp::add(var_291, var_280);
    wp::assign_inplace(var_185, var_290, var_292, var_288);
    // out[r, c] = top_left[r, c]                                                             <L 825>
    var_294 = wp::extract(var_172, var_266, var_293);
    wp::assign_inplace(var_185, var_266, var_293, var_294);
    // out[r, 3 + c] = top_right[r, c]                                                        <L 826>
    var_295 = wp::extract(var_173, var_266, var_293);
    var_297 = wp::add(var_296, var_293);
    wp::assign_inplace(var_185, var_266, var_297, var_295);
    // out[3 + r, c] = bot_left[r, c]                                                         <L 827>
    var_298 = wp::extract(var_177, var_266, var_293);
    var_300 = wp::add(var_299, var_266);
    wp::assign_inplace(var_185, var_300, var_293, var_298);
    // out[3 + r, 3 + c] = bot_right[r, c]                                                    <L 828>
    var_301 = wp::extract(var_183, var_266, var_293);
    var_303 = wp::add(var_302, var_266);
    var_305 = wp::add(var_304, var_293);
    wp::assign_inplace(var_185, var_303, var_305, var_301);
    // return out                                                                             <L 829>
    return var_185;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:996
static CUDA_CALLABLE wp::mat_t<6, 6, wp::float32> _deriv_box_fluid_0(
    wp::int32 var_opt_integrator,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::int32 var_worldid,
    wp::int32 var_bodyid,
    wp::vec_t<6, wp::float32> var_lvel,
    wp::float32 var_density,
    wp::float32 var_viscosity)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    wp::mat_t<6, 6, wp::float32> var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::int32 var_6;
    wp::float32* var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::shape_t* var_10;
    const wp::int32 var_11 = 0;
    wp::int32 var_12;
    wp::shape_t var_13;
    wp::int32 var_14;
    wp::vec_t<3, wp::float32>* var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    const wp::float32 var_18 = 6.0;
    wp::float32 var_19;
    const wp::float32 var_20 = 1e-15;
    const wp::int32 var_21 = 1;
    wp::float32 var_22;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::int32 var_26 = 0;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::int32 var_32 = 0;
    wp::float32 var_33;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::vec_t<3, wp::float32> var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::float32 var_64;
    const wp::float32 var_65 = 0.3333333333333333;
    wp::float32 var_66;
    const wp::float32 var_67 = 3.141592653589793;
    const wp::float32 var_68 = -3.141592653589793;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::int32 var_73 = 0;
    const wp::int32 var_74 = 0;
    const wp::int32 var_75 = 1;
    const wp::int32 var_76 = 1;
    const wp::int32 var_77 = 2;
    const wp::int32 var_78 = 2;
    const wp::float32 var_79 = -9.42477796076938;
    wp::float32 var_80;
    wp::float32 var_81;
    const wp::int32 var_82 = 3;
    const wp::int32 var_83 = 3;
    const wp::int32 var_84 = 4;
    const wp::int32 var_85 = 4;
    const wp::int32 var_86 = 5;
    const wp::int32 var_87 = 5;
    const wp::float32 var_88 = 0.0;
    bool var_89;
    const wp::int32 var_90 = 1;
    wp::float32 var_91;
    const wp::int32 var_92 = 1;
    wp::float32 var_93;
    wp::float32 var_94;
    const wp::int32 var_95 = 1;
    wp::float32 var_96;
    wp::float32 var_97;
    const wp::int32 var_98 = 1;
    wp::float32 var_99;
    wp::float32 var_100;
    const wp::int32 var_101 = 2;
    wp::float32 var_102;
    const wp::int32 var_103 = 2;
    wp::float32 var_104;
    wp::float32 var_105;
    const wp::int32 var_106 = 2;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::int32 var_109 = 2;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::int32 var_113 = 0;
    wp::float32 var_114;
    const wp::int32 var_115 = 0;
    wp::float32 var_116;
    wp::float32 var_117;
    const wp::int32 var_118 = 0;
    wp::float32 var_119;
    wp::float32 var_120;
    const wp::int32 var_121 = 0;
    wp::float32 var_122;
    wp::float32 var_123;
    const wp::int32 var_124 = 2;
    wp::float32 var_125;
    const wp::int32 var_126 = 2;
    wp::float32 var_127;
    wp::float32 var_128;
    const wp::int32 var_129 = 2;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 2;
    wp::float32 var_133;
    wp::float32 var_134;
    wp::float32 var_135;
    const wp::int32 var_136 = 0;
    wp::float32 var_137;
    const wp::int32 var_138 = 0;
    wp::float32 var_139;
    wp::float32 var_140;
    const wp::int32 var_141 = 0;
    wp::float32 var_142;
    wp::float32 var_143;
    const wp::int32 var_144 = 0;
    wp::float32 var_145;
    wp::float32 var_146;
    const wp::int32 var_147 = 1;
    wp::float32 var_148;
    const wp::int32 var_149 = 1;
    wp::float32 var_150;
    wp::float32 var_151;
    const wp::int32 var_152 = 1;
    wp::float32 var_153;
    wp::float32 var_154;
    const wp::int32 var_155 = 1;
    wp::float32 var_156;
    wp::float32 var_157;
    wp::float32 var_158;
    const wp::float32 var_159 = 0.03125;
    const wp::int32 var_160 = 0;
    const wp::int32 var_161 = 0;
    const wp::int32 var_162 = 0;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::float32 var_165;
    const wp::int32 var_166 = 0;
    wp::float32 var_167;
    wp::float32 var_168;
    wp::float32 var_169;
    wp::float32 var_170;
    const wp::int32 var_171 = 1;
    const wp::int32 var_172 = 1;
    const wp::int32 var_173 = 1;
    wp::float32 var_174;
    wp::float32 var_175;
    wp::float32 var_176;
    const wp::int32 var_177 = 1;
    wp::float32 var_178;
    wp::float32 var_179;
    wp::float32 var_180;
    wp::float32 var_181;
    const wp::int32 var_182 = 2;
    const wp::int32 var_183 = 2;
    const wp::int32 var_184 = 2;
    wp::float32 var_185;
    wp::float32 var_186;
    wp::float32 var_187;
    const wp::int32 var_188 = 2;
    wp::float32 var_189;
    wp::float32 var_190;
    wp::float32 var_191;
    wp::float32 var_192;
    const wp::int32 var_193 = 3;
    const wp::int32 var_194 = 3;
    const wp::int32 var_195 = 1;
    wp::float32 var_196;
    wp::float32 var_197;
    const wp::int32 var_198 = 2;
    wp::float32 var_199;
    wp::float32 var_200;
    const wp::int32 var_201 = 3;
    wp::float32 var_202;
    wp::float32 var_203;
    wp::float32 var_204;
    const wp::int32 var_205 = 4;
    const wp::int32 var_206 = 4;
    const wp::int32 var_207 = 0;
    wp::float32 var_208;
    wp::float32 var_209;
    const wp::int32 var_210 = 2;
    wp::float32 var_211;
    wp::float32 var_212;
    const wp::int32 var_213 = 4;
    wp::float32 var_214;
    wp::float32 var_215;
    wp::float32 var_216;
    const wp::int32 var_217 = 5;
    const wp::int32 var_218 = 5;
    const wp::int32 var_219 = 0;
    wp::float32 var_220;
    wp::float32 var_221;
    const wp::int32 var_222 = 1;
    wp::float32 var_223;
    wp::float32 var_224;
    const wp::int32 var_225 = 5;
    wp::float32 var_226;
    wp::float32 var_227;
    wp::float32 var_228;
    bool var_229;
    const wp::int32 var_230 = 3;
    bool var_231;
    const wp::int32 var_232 = 4;
    bool var_233;
    const wp::float32 var_234 = 0.5;
    wp::mat_t<6, 6, wp::float32> var_235;
    wp::mat_t<6, 6, wp::float32> var_236;
    wp::mat_t<6, 6, wp::float32> var_237;
    wp::mat_t<6, 6, wp::float32> var_238;
    //---------
    // forward
    // def _deriv_box_fluid(                                                                  <L 997>
    // B = wp.spatial_matrix(0.0)                                                             <L 1009>
    var_1 = wp::mat_t<6, 6, wp::float32>(var_0);
    // mass = body_mass[worldid % body_mass.shape[0], bodyid]                                 <L 1011>
    var_2 = &(var_body_mass.shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    var_6 = wp::mod(var_worldid, var_4);
    var_7 = wp::address(var_body_mass, var_6, var_bodyid);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // inertia = body_inertia[worldid % body_inertia.shape[0], bodyid]                        <L 1012>
    var_10 = &(var_body_inertia.shape);
    var_13 = wp::load(var_10);
    var_12 = wp::extract(var_13, var_11);
    var_14 = wp::mod(var_worldid, var_12);
    var_15 = wp::address(var_body_inertia, var_14, var_bodyid);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    // scl = 6.0 / mass                                                                       <L 1013>
    var_19 = wp::div(var_18, var_8);
    // box = wp.vec3(                                                                         <L 1016>
    // wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl),                <L 1017>
    var_22 = wp::extract(var_16, var_21);
    var_24 = wp::extract(var_16, var_23);
    var_25 = wp::add(var_22, var_24);
    var_27 = wp::extract(var_16, var_26);
    var_28 = wp::sub(var_25, var_27);
    var_29 = wp::max(var_20, var_28);
    var_30 = wp::mul(var_29, var_19);
    var_31 = wp::sqrt(var_30);
    // wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl),                <L 1018>
    var_33 = wp::extract(var_16, var_32);
    var_35 = wp::extract(var_16, var_34);
    var_36 = wp::add(var_33, var_35);
    var_38 = wp::extract(var_16, var_37);
    var_39 = wp::sub(var_36, var_38);
    var_40 = wp::max(var_20, var_39);
    var_41 = wp::mul(var_40, var_19);
    var_42 = wp::sqrt(var_41);
    // wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl),                <L 1019>
    var_44 = wp::extract(var_16, var_43);
    var_46 = wp::extract(var_16, var_45);
    var_47 = wp::add(var_44, var_46);
    var_49 = wp::extract(var_16, var_48);
    var_50 = wp::sub(var_47, var_49);
    var_51 = wp::max(var_20, var_50);
    var_52 = wp::mul(var_51, var_19);
    var_53 = wp::sqrt(var_52);
    var_54 = wp::vec_t<3, wp::float32>(var_31, var_42, var_53);
    // if viscosity > 0.0:                                                                    <L 1023>
    var_56 = (var_viscosity > var_55);
    if (var_56) {
        // diam = (box[0] + box[1] + box[2]) * wp.static(1.0 / 3.0)                           <L 1024>
        var_58 = wp::extract(var_54, var_57);
        var_60 = wp::extract(var_54, var_59);
        var_61 = wp::add(var_58, var_60);
        var_63 = wp::extract(var_54, var_62);
        var_64 = wp::add(var_61, var_63);
        var_66 = wp::mul(var_64, var_65);
        // visc_rot = -wp.pi * diam * diam * diam * viscosity                                 <L 1027>
        var_69 = wp::mul(var_68, var_66);
        var_70 = wp::mul(var_69, var_66);
        var_71 = wp::mul(var_70, var_66);
        var_72 = wp::mul(var_71, var_viscosity);
        // B[0, 0] += visc_rot                                                                <L 1028>
        wp::add_inplace(var_1, var_73, var_74, var_72);
        // B[1, 1] += visc_rot                                                                <L 1029>
        wp::add_inplace(var_1, var_75, var_76, var_72);
        // B[2, 2] += visc_rot                                                                <L 1030>
        wp::add_inplace(var_1, var_77, var_78, var_72);
        // visc_lin = wp.static(-3.0 * wp.pi) * diam * viscosity                              <L 1033>
        var_80 = wp::mul(var_79, var_66);
        var_81 = wp::mul(var_80, var_viscosity);
        // B[3, 3] += visc_lin                                                                <L 1034>
        wp::add_inplace(var_1, var_82, var_83, var_81);
        // B[4, 4] += visc_lin                                                                <L 1035>
        wp::add_inplace(var_1, var_84, var_85, var_81);
        // B[5, 5] += visc_lin                                                                <L 1036>
        wp::add_inplace(var_1, var_86, var_87, var_81);
    }
    // if density > 0.0:                                                                      <L 1039>
    var_89 = (var_density > var_88);
    if (var_89) {
        // term0 = box[1] * box[1] * box[1] * box[1] + box[2] * box[2] * box[2] * box[2]       <L 1040>
        var_91 = wp::extract(var_54, var_90);
        var_93 = wp::extract(var_54, var_92);
        var_94 = wp::mul(var_91, var_93);
        var_96 = wp::extract(var_54, var_95);
        var_97 = wp::mul(var_94, var_96);
        var_99 = wp::extract(var_54, var_98);
        var_100 = wp::mul(var_97, var_99);
        var_102 = wp::extract(var_54, var_101);
        var_104 = wp::extract(var_54, var_103);
        var_105 = wp::mul(var_102, var_104);
        var_107 = wp::extract(var_54, var_106);
        var_108 = wp::mul(var_105, var_107);
        var_110 = wp::extract(var_54, var_109);
        var_111 = wp::mul(var_108, var_110);
        var_112 = wp::add(var_100, var_111);
        // term1 = box[0] * box[0] * box[0] * box[0] + box[2] * box[2] * box[2] * box[2]       <L 1041>
        var_114 = wp::extract(var_54, var_113);
        var_116 = wp::extract(var_54, var_115);
        var_117 = wp::mul(var_114, var_116);
        var_119 = wp::extract(var_54, var_118);
        var_120 = wp::mul(var_117, var_119);
        var_122 = wp::extract(var_54, var_121);
        var_123 = wp::mul(var_120, var_122);
        var_125 = wp::extract(var_54, var_124);
        var_127 = wp::extract(var_54, var_126);
        var_128 = wp::mul(var_125, var_127);
        var_130 = wp::extract(var_54, var_129);
        var_131 = wp::mul(var_128, var_130);
        var_133 = wp::extract(var_54, var_132);
        var_134 = wp::mul(var_131, var_133);
        var_135 = wp::add(var_123, var_134);
        // term2 = box[0] * box[0] * box[0] * box[0] + box[1] * box[1] * box[1] * box[1]       <L 1042>
        var_137 = wp::extract(var_54, var_136);
        var_139 = wp::extract(var_54, var_138);
        var_140 = wp::mul(var_137, var_139);
        var_142 = wp::extract(var_54, var_141);
        var_143 = wp::mul(var_140, var_142);
        var_145 = wp::extract(var_54, var_144);
        var_146 = wp::mul(var_143, var_145);
        var_148 = wp::extract(var_54, var_147);
        var_150 = wp::extract(var_54, var_149);
        var_151 = wp::mul(var_148, var_150);
        var_153 = wp::extract(var_54, var_152);
        var_154 = wp::mul(var_151, var_153);
        var_156 = wp::extract(var_54, var_155);
        var_157 = wp::mul(var_154, var_156);
        var_158 = wp::add(var_146, var_157);
        // inv_32 = wp.static(1.0 / 32.0)                                                     <L 1044>
        // B[0, 0] -= density * box[0] * term0 * wp.abs(lvel[0]) * inv_32                     <L 1045>
        var_163 = wp::extract(var_54, var_162);
        var_164 = wp::mul(var_density, var_163);
        var_165 = wp::mul(var_164, var_112);
        var_167 = wp::extract(var_lvel, var_166);
        var_168 = wp::abs(var_167);
        var_169 = wp::mul(var_165, var_168);
        var_170 = wp::mul(var_169, var_159);
        wp::sub_inplace(var_1, var_160, var_161, var_170);
        // B[1, 1] -= density * box[1] * term1 * wp.abs(lvel[1]) * inv_32                     <L 1046>
        var_174 = wp::extract(var_54, var_173);
        var_175 = wp::mul(var_density, var_174);
        var_176 = wp::mul(var_175, var_135);
        var_178 = wp::extract(var_lvel, var_177);
        var_179 = wp::abs(var_178);
        var_180 = wp::mul(var_176, var_179);
        var_181 = wp::mul(var_180, var_159);
        wp::sub_inplace(var_1, var_171, var_172, var_181);
        // B[2, 2] -= density * box[2] * term2 * wp.abs(lvel[2]) * inv_32                     <L 1047>
        var_185 = wp::extract(var_54, var_184);
        var_186 = wp::mul(var_density, var_185);
        var_187 = wp::mul(var_186, var_158);
        var_189 = wp::extract(var_lvel, var_188);
        var_190 = wp::abs(var_189);
        var_191 = wp::mul(var_187, var_190);
        var_192 = wp::mul(var_191, var_159);
        wp::sub_inplace(var_1, var_182, var_183, var_192);
        // B[3, 3] -= density * box[1] * box[2] * wp.abs(lvel[3])                             <L 1049>
        var_196 = wp::extract(var_54, var_195);
        var_197 = wp::mul(var_density, var_196);
        var_199 = wp::extract(var_54, var_198);
        var_200 = wp::mul(var_197, var_199);
        var_202 = wp::extract(var_lvel, var_201);
        var_203 = wp::abs(var_202);
        var_204 = wp::mul(var_200, var_203);
        wp::sub_inplace(var_1, var_193, var_194, var_204);
        // B[4, 4] -= density * box[0] * box[2] * wp.abs(lvel[4])                             <L 1050>
        var_208 = wp::extract(var_54, var_207);
        var_209 = wp::mul(var_density, var_208);
        var_211 = wp::extract(var_54, var_210);
        var_212 = wp::mul(var_209, var_211);
        var_214 = wp::extract(var_lvel, var_213);
        var_215 = wp::abs(var_214);
        var_216 = wp::mul(var_212, var_215);
        wp::sub_inplace(var_1, var_205, var_206, var_216);
        // B[5, 5] -= density * box[0] * box[1] * wp.abs(lvel[5])                             <L 1051>
        var_220 = wp::extract(var_54, var_219);
        var_221 = wp::mul(var_density, var_220);
        var_223 = wp::extract(var_54, var_222);
        var_224 = wp::mul(var_221, var_223);
        var_226 = wp::extract(var_lvel, var_225);
        var_227 = wp::abs(var_226);
        var_228 = wp::mul(var_224, var_227);
        wp::sub_inplace(var_1, var_217, var_218, var_228);
    }
    // if opt_integrator == IntegratorType.IMPLICITFAST or opt_integrator == IntegratorType.DISCRETE:       <L 1053>
    var_231 = (var_opt_integrator == var_230);
    var_229 = var_231;
    if (!var_229) {
        var_233 = (var_opt_integrator == var_232);
        var_229 = var_229 || var_233;
    }
    if (var_229) {
        // B = 0.5 * (B + wp.transpose(B))                                                    <L 1054>
        var_235 = wp::transpose(var_1);
        var_236 = wp::add(var_1, var_235);
        var_237 = wp::mul(var_234, var_236);
    }
    var_238 = wp::where(var_229, var_237, var_1);
    // return B                                                                               <L 1056>
    return var_238;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:379
static CUDA_CALLABLE void lu_factor_6x6_0(
    wp::mat_t<6, 6, wp::float32> var_A_in,
    wp::mat_t<6, 6, wp::float32> & ret_0,
    wp::vec_t<6, wp::int32> & ret_1,
    bool & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<6, 6, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::vec_t<6, wp::int32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::int32 var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    const wp::int32 var_9 = 6;
    wp::range_t var_10;
    wp::int32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    bool var_14;
    wp::float32 var_15;
    wp::int32 var_16;
    wp::float32 var_17;
    wp::int32 var_18;
    const wp::float32 var_19 = 1e-15;
    bool var_20;
    const bool var_21 = false;
    bool var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::int32 var_26 = 1;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 2;
    wp::float32 var_30;
    wp::float32 var_31;
    const wp::int32 var_32 = 3;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 4;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::int32 var_38 = 5;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::float32 var_41 = 1.0;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::int32 var_44 = 1;
    wp::int32 var_45;
    const wp::int32 var_46 = 6;
    wp::range_t var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 1;
    wp::int32 var_54;
    const wp::int32 var_55 = 6;
    wp::range_t var_56;
    wp::int32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::int32 var_60 = 1;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::int32 var_63;
    const wp::int32 var_64 = 1;
    wp::int32 var_65;
    const wp::int32 var_66 = 6;
    wp::range_t var_67;
    wp::int32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    bool var_71;
    wp::float32 var_72;
    wp::int32 var_73;
    wp::float32 var_74;
    wp::int32 var_75;
    const wp::float32 var_76 = 1e-15;
    bool var_77;
    const bool var_78 = false;
    bool var_79;
    const wp::int32 var_80 = 0;
    wp::float32 var_81;
    wp::float32 var_82;
    const wp::int32 var_83 = 1;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 2;
    wp::float32 var_87;
    wp::float32 var_88;
    const wp::int32 var_89 = 3;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 4;
    wp::float32 var_93;
    wp::float32 var_94;
    const wp::int32 var_95 = 5;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::int32 var_98;
    wp::float32 var_99;
    const wp::float32 var_100 = 1.0;
    wp::float32 var_101;
    wp::float32 var_102;
    const wp::int32 var_103 = 1;
    wp::int32 var_104;
    const wp::int32 var_105 = 6;
    wp::range_t var_106;
    wp::int32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    const wp::int32 var_111 = 1;
    wp::int32 var_112;
    const wp::int32 var_113 = 6;
    wp::range_t var_114;
    wp::int32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    const wp::int32 var_118 = 2;
    wp::float32 var_119;
    wp::float32 var_120;
    wp::int32 var_121;
    const wp::int32 var_122 = 1;
    wp::int32 var_123;
    const wp::int32 var_124 = 6;
    wp::range_t var_125;
    wp::int32 var_126;
    wp::float32 var_127;
    wp::float32 var_128;
    bool var_129;
    wp::float32 var_130;
    wp::int32 var_131;
    wp::float32 var_132;
    wp::int32 var_133;
    const wp::float32 var_134 = 1e-15;
    bool var_135;
    const bool var_136 = false;
    bool var_137;
    const wp::int32 var_138 = 0;
    wp::float32 var_139;
    wp::float32 var_140;
    const wp::int32 var_141 = 1;
    wp::float32 var_142;
    wp::float32 var_143;
    const wp::int32 var_144 = 2;
    wp::float32 var_145;
    wp::float32 var_146;
    const wp::int32 var_147 = 3;
    wp::float32 var_148;
    wp::float32 var_149;
    const wp::int32 var_150 = 4;
    wp::float32 var_151;
    wp::float32 var_152;
    const wp::int32 var_153 = 5;
    wp::float32 var_154;
    wp::float32 var_155;
    wp::int32 var_156;
    wp::float32 var_157;
    const wp::float32 var_158 = 1.0;
    wp::float32 var_159;
    wp::float32 var_160;
    const wp::int32 var_161 = 1;
    wp::int32 var_162;
    const wp::int32 var_163 = 6;
    wp::range_t var_164;
    wp::int32 var_165;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    const wp::int32 var_169 = 1;
    wp::int32 var_170;
    const wp::int32 var_171 = 6;
    wp::range_t var_172;
    wp::int32 var_173;
    wp::float32 var_174;
    wp::float32 var_175;
    const wp::int32 var_176 = 3;
    wp::float32 var_177;
    wp::float32 var_178;
    wp::int32 var_179;
    const wp::int32 var_180 = 1;
    wp::int32 var_181;
    const wp::int32 var_182 = 6;
    wp::range_t var_183;
    wp::int32 var_184;
    wp::float32 var_185;
    wp::float32 var_186;
    bool var_187;
    wp::float32 var_188;
    wp::int32 var_189;
    wp::float32 var_190;
    wp::int32 var_191;
    const wp::float32 var_192 = 1e-15;
    bool var_193;
    const bool var_194 = false;
    bool var_195;
    const wp::int32 var_196 = 0;
    wp::float32 var_197;
    wp::float32 var_198;
    const wp::int32 var_199 = 1;
    wp::float32 var_200;
    wp::float32 var_201;
    const wp::int32 var_202 = 2;
    wp::float32 var_203;
    wp::float32 var_204;
    const wp::int32 var_205 = 3;
    wp::float32 var_206;
    wp::float32 var_207;
    const wp::int32 var_208 = 4;
    wp::float32 var_209;
    wp::float32 var_210;
    const wp::int32 var_211 = 5;
    wp::float32 var_212;
    wp::float32 var_213;
    wp::int32 var_214;
    wp::float32 var_215;
    const wp::float32 var_216 = 1.0;
    wp::float32 var_217;
    wp::float32 var_218;
    const wp::int32 var_219 = 1;
    wp::int32 var_220;
    const wp::int32 var_221 = 6;
    wp::range_t var_222;
    wp::int32 var_223;
    wp::float32 var_224;
    wp::float32 var_225;
    wp::float32 var_226;
    const wp::int32 var_227 = 1;
    wp::int32 var_228;
    const wp::int32 var_229 = 6;
    wp::range_t var_230;
    wp::int32 var_231;
    wp::float32 var_232;
    wp::float32 var_233;
    const wp::int32 var_234 = 4;
    wp::float32 var_235;
    wp::float32 var_236;
    wp::int32 var_237;
    const wp::int32 var_238 = 1;
    wp::int32 var_239;
    const wp::int32 var_240 = 6;
    wp::range_t var_241;
    wp::int32 var_242;
    wp::float32 var_243;
    wp::float32 var_244;
    bool var_245;
    wp::float32 var_246;
    wp::int32 var_247;
    wp::float32 var_248;
    wp::int32 var_249;
    const wp::float32 var_250 = 1e-15;
    bool var_251;
    const bool var_252 = false;
    bool var_253;
    const wp::int32 var_254 = 0;
    wp::float32 var_255;
    wp::float32 var_256;
    const wp::int32 var_257 = 1;
    wp::float32 var_258;
    wp::float32 var_259;
    const wp::int32 var_260 = 2;
    wp::float32 var_261;
    wp::float32 var_262;
    const wp::int32 var_263 = 3;
    wp::float32 var_264;
    wp::float32 var_265;
    const wp::int32 var_266 = 4;
    wp::float32 var_267;
    wp::float32 var_268;
    const wp::int32 var_269 = 5;
    wp::float32 var_270;
    wp::float32 var_271;
    wp::int32 var_272;
    wp::float32 var_273;
    const wp::float32 var_274 = 1.0;
    wp::float32 var_275;
    wp::float32 var_276;
    const wp::int32 var_277 = 1;
    wp::int32 var_278;
    const wp::int32 var_279 = 6;
    wp::range_t var_280;
    wp::int32 var_281;
    wp::float32 var_282;
    wp::float32 var_283;
    wp::float32 var_284;
    const wp::int32 var_285 = 1;
    wp::int32 var_286;
    const wp::int32 var_287 = 6;
    wp::range_t var_288;
    wp::int32 var_289;
    wp::float32 var_290;
    wp::float32 var_291;
    const wp::int32 var_292 = 5;
    wp::float32 var_293;
    wp::float32 var_294;
    wp::int32 var_295;
    const wp::int32 var_296 = 1;
    wp::int32 var_297;
    const wp::int32 var_298 = 6;
    wp::range_t var_299;
    wp::int32 var_300;
    wp::float32 var_301;
    wp::float32 var_302;
    bool var_303;
    wp::float32 var_304;
    wp::int32 var_305;
    wp::float32 var_306;
    wp::int32 var_307;
    const wp::float32 var_308 = 1e-15;
    bool var_309;
    const bool var_310 = false;
    bool var_311;
    const wp::int32 var_312 = 0;
    wp::float32 var_313;
    wp::float32 var_314;
    const wp::int32 var_315 = 1;
    wp::float32 var_316;
    wp::float32 var_317;
    const wp::int32 var_318 = 2;
    wp::float32 var_319;
    wp::float32 var_320;
    const wp::int32 var_321 = 3;
    wp::float32 var_322;
    wp::float32 var_323;
    const wp::int32 var_324 = 4;
    wp::float32 var_325;
    wp::float32 var_326;
    const wp::int32 var_327 = 5;
    wp::float32 var_328;
    wp::float32 var_329;
    wp::int32 var_330;
    wp::float32 var_331;
    const wp::float32 var_332 = 1.0;
    wp::float32 var_333;
    wp::float32 var_334;
    const wp::int32 var_335 = 1;
    wp::int32 var_336;
    const wp::int32 var_337 = 6;
    wp::range_t var_338;
    wp::int32 var_339;
    wp::float32 var_340;
    wp::float32 var_341;
    wp::float32 var_342;
    const wp::int32 var_343 = 1;
    wp::int32 var_344;
    const wp::int32 var_345 = 6;
    wp::range_t var_346;
    wp::int32 var_347;
    wp::float32 var_348;
    wp::float32 var_349;
    const bool var_350 = true;
    //---------
    // forward
    // def lu_factor_6x6(                                                                     <L 380>
    // A = A_in                                                                               <L 385>
    var_0 = wp::copy(var_A_in);
    // pivot = types.vec6i(0)                                                                 <L 386>
    var_2 = wp::vec_t<6, wp::int32>(var_1);
    // for k in range(6):                                                                     <L 387>
    // maxval = wp.abs(A[k, k])                                                               <L 388>
    var_4 = wp::extract(var_0, var_3, var_3);
    var_5 = wp::abs(var_4);
    // maxrow = k                                                                             <L 389>
    var_6 = wp::copy(var_3);
    // for i in range(k + 1, 6):                                                              <L 390>
    var_8 = wp::add(var_3, var_7);
    var_10 = wp::range(var_8, var_9);
    start_for_0:;
        if (iter_cmp(var_10) == 0) goto end_for_0;
        var_11 = wp::iter_next(var_10);
        // val = wp.abs(A[i, k])                                                              <L 391>
        var_12 = wp::extract(var_0, var_11, var_3);
        var_13 = wp::abs(var_12);
        // if val > maxval:                                                                   <L 392>
        var_14 = (var_13 > var_5);
        if (var_14) {
            // maxval = val                                                                   <L 393>
            var_15 = wp::copy(var_13);
            // maxrow = i                                                                     <L 394>
            var_16 = wp::copy(var_11);
        }
        var_17 = wp::where(var_14, var_15, var_5);
        var_18 = wp::where(var_14, var_16, var_6);
        wp::assign(var_5, var_17);
        wp::assign(var_6, var_18);
        goto start_for_0;
    end_for_0:;
    // if maxval < types.MJ_MINVAL:                                                           <L 395>
    var_20 = (var_5 < var_19);
    if (var_20) {
        // return A, pivot, False                                                             <L 396>
        ret_0 = var_0;
        ret_1 = var_2;
        ret_2 = var_21;
        return;
    }
    // pivot[k] = maxrow                                                                      <L 397>
    wp::assign_inplace(var_2, var_3, var_6);
    // if maxrow != k:                                                                        <L 398>
    var_22 = (var_6 != var_3);
    if (var_22) {
        // for j in range(6):                                                                 <L 399>
        // tmp = A[k, j]                                                                      <L 400>
        var_24 = wp::extract(var_0, var_3, var_23);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_25 = wp::extract(var_0, var_6, var_23);
        wp::assign_inplace(var_0, var_3, var_23, var_25);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_6, var_23, var_24);
        // tmp = A[k, j]                                                                      <L 400>
        var_27 = wp::extract(var_0, var_3, var_26);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_28 = wp::extract(var_0, var_6, var_26);
        wp::assign_inplace(var_0, var_3, var_26, var_28);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_6, var_26, var_27);
        // tmp = A[k, j]                                                                      <L 400>
        var_30 = wp::extract(var_0, var_3, var_29);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_31 = wp::extract(var_0, var_6, var_29);
        wp::assign_inplace(var_0, var_3, var_29, var_31);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_6, var_29, var_30);
        // tmp = A[k, j]                                                                      <L 400>
        var_33 = wp::extract(var_0, var_3, var_32);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_34 = wp::extract(var_0, var_6, var_32);
        wp::assign_inplace(var_0, var_3, var_32, var_34);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_6, var_32, var_33);
        // tmp = A[k, j]                                                                      <L 400>
        var_36 = wp::extract(var_0, var_3, var_35);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_37 = wp::extract(var_0, var_6, var_35);
        wp::assign_inplace(var_0, var_3, var_35, var_37);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_6, var_35, var_36);
        // tmp = A[k, j]                                                                      <L 400>
        var_39 = wp::extract(var_0, var_3, var_38);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_40 = wp::extract(var_0, var_6, var_38);
        wp::assign_inplace(var_0, var_3, var_38, var_40);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_6, var_38, var_39);
    }
    // diaginv = 1.0 / A[k, k]                                                                <L 403>
    var_42 = wp::extract(var_0, var_3, var_3);
    var_43 = wp::div(var_41, var_42);
    // for i in range(k + 1, 6):                                                              <L 404>
    var_45 = wp::add(var_3, var_44);
    var_47 = wp::range(var_45, var_46);
    var_48 = wp::copy(var_38);
    start_for_3:;
        if (iter_cmp(var_47) == 0) goto end_for_3;
        var_49 = wp::iter_next(var_47);
        // A[i, k] *= diaginv                                                                 <L 405>
        var_50 = wp::extract(var_0, var_49, var_3);
        var_51 = wp::mul(var_50, var_43);
        wp::assign_inplace(var_0, var_49, var_3, var_51);
        // Aik = A[i, k]                                                                      <L 406>
        var_52 = wp::extract(var_0, var_49, var_3);
        // for j in range(k + 1, 6):                                                          <L 407>
        var_54 = wp::add(var_3, var_53);
        var_56 = wp::range(var_54, var_55);
        start_for_5:;
            if (iter_cmp(var_56) == 0) goto end_for_5;
            var_57 = wp::iter_next(var_56);
            // A[i, j] -= Aik * A[k, j]                                                       <L 408>
            var_58 = wp::extract(var_0, var_3, var_57);
            var_59 = wp::mul(var_52, var_58);
            wp::sub_inplace(var_0, var_49, var_57, var_59);
            goto start_for_5;
        end_for_5:;
        wp::assign(var_48, var_57);
        goto start_for_3;
    end_for_3:;
    // maxval = wp.abs(A[k, k])                                                               <L 388>
    var_61 = wp::extract(var_0, var_60, var_60);
    var_62 = wp::abs(var_61);
    // maxrow = k                                                                             <L 389>
    var_63 = wp::copy(var_60);
    // for i in range(k + 1, 6):                                                              <L 390>
    var_65 = wp::add(var_60, var_64);
    var_67 = wp::range(var_65, var_66);
    start_for_7:;
        if (iter_cmp(var_67) == 0) goto end_for_7;
        var_68 = wp::iter_next(var_67);
        // val = wp.abs(A[i, k])                                                              <L 391>
        var_69 = wp::extract(var_0, var_68, var_60);
        var_70 = wp::abs(var_69);
        // if val > maxval:                                                                   <L 392>
        var_71 = (var_70 > var_62);
        if (var_71) {
            // maxval = val                                                                   <L 393>
            var_72 = wp::copy(var_70);
            // maxrow = i                                                                     <L 394>
            var_73 = wp::copy(var_68);
        }
        var_74 = wp::where(var_71, var_72, var_62);
        var_75 = wp::where(var_71, var_73, var_63);
        wp::assign(var_62, var_74);
        wp::assign(var_63, var_75);
        wp::assign(var_13, var_70);
        goto start_for_7;
    end_for_7:;
    // if maxval < types.MJ_MINVAL:                                                           <L 395>
    var_77 = (var_62 < var_76);
    if (var_77) {
        // return A, pivot, False                                                             <L 396>
        ret_0 = var_0;
        ret_1 = var_2;
        ret_2 = var_78;
        return;
    }
    // pivot[k] = maxrow                                                                      <L 397>
    wp::assign_inplace(var_2, var_60, var_63);
    // if maxrow != k:                                                                        <L 398>
    var_79 = (var_63 != var_60);
    if (var_79) {
        // for j in range(6):                                                                 <L 399>
        // tmp = A[k, j]                                                                      <L 400>
        var_81 = wp::extract(var_0, var_60, var_80);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_82 = wp::extract(var_0, var_63, var_80);
        wp::assign_inplace(var_0, var_60, var_80, var_82);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_63, var_80, var_81);
        // tmp = A[k, j]                                                                      <L 400>
        var_84 = wp::extract(var_0, var_60, var_83);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_85 = wp::extract(var_0, var_63, var_83);
        wp::assign_inplace(var_0, var_60, var_83, var_85);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_63, var_83, var_84);
        // tmp = A[k, j]                                                                      <L 400>
        var_87 = wp::extract(var_0, var_60, var_86);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_88 = wp::extract(var_0, var_63, var_86);
        wp::assign_inplace(var_0, var_60, var_86, var_88);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_63, var_86, var_87);
        // tmp = A[k, j]                                                                      <L 400>
        var_90 = wp::extract(var_0, var_60, var_89);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_91 = wp::extract(var_0, var_63, var_89);
        wp::assign_inplace(var_0, var_60, var_89, var_91);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_63, var_89, var_90);
        // tmp = A[k, j]                                                                      <L 400>
        var_93 = wp::extract(var_0, var_60, var_92);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_94 = wp::extract(var_0, var_63, var_92);
        wp::assign_inplace(var_0, var_60, var_92, var_94);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_63, var_92, var_93);
        // tmp = A[k, j]                                                                      <L 400>
        var_96 = wp::extract(var_0, var_60, var_95);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_97 = wp::extract(var_0, var_63, var_95);
        wp::assign_inplace(var_0, var_60, var_95, var_97);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_63, var_95, var_96);
    }
    var_98 = wp::where(var_79, var_95, var_48);
    var_99 = wp::where(var_79, var_96, var_39);
    // diaginv = 1.0 / A[k, k]                                                                <L 403>
    var_101 = wp::extract(var_0, var_60, var_60);
    var_102 = wp::div(var_100, var_101);
    // for i in range(k + 1, 6):                                                              <L 404>
    var_104 = wp::add(var_60, var_103);
    var_106 = wp::range(var_104, var_105);
    start_for_10:;
        if (iter_cmp(var_106) == 0) goto end_for_10;
        var_107 = wp::iter_next(var_106);
        // A[i, k] *= diaginv                                                                 <L 405>
        var_108 = wp::extract(var_0, var_107, var_60);
        var_109 = wp::mul(var_108, var_102);
        wp::assign_inplace(var_0, var_107, var_60, var_109);
        // Aik = A[i, k]                                                                      <L 406>
        var_110 = wp::extract(var_0, var_107, var_60);
        // for j in range(k + 1, 6):                                                          <L 407>
        var_112 = wp::add(var_60, var_111);
        var_114 = wp::range(var_112, var_113);
        start_for_12:;
            if (iter_cmp(var_114) == 0) goto end_for_12;
            var_115 = wp::iter_next(var_114);
            // A[i, j] -= Aik * A[k, j]                                                       <L 408>
            var_116 = wp::extract(var_0, var_60, var_115);
            var_117 = wp::mul(var_110, var_116);
            wp::sub_inplace(var_0, var_107, var_115, var_117);
            goto start_for_12;
        end_for_12:;
        wp::assign(var_98, var_115);
        wp::assign(var_52, var_110);
        goto start_for_10;
    end_for_10:;
    // maxval = wp.abs(A[k, k])                                                               <L 388>
    var_119 = wp::extract(var_0, var_118, var_118);
    var_120 = wp::abs(var_119);
    // maxrow = k                                                                             <L 389>
    var_121 = wp::copy(var_118);
    // for i in range(k + 1, 6):                                                              <L 390>
    var_123 = wp::add(var_118, var_122);
    var_125 = wp::range(var_123, var_124);
    start_for_14:;
        if (iter_cmp(var_125) == 0) goto end_for_14;
        var_126 = wp::iter_next(var_125);
        // val = wp.abs(A[i, k])                                                              <L 391>
        var_127 = wp::extract(var_0, var_126, var_118);
        var_128 = wp::abs(var_127);
        // if val > maxval:                                                                   <L 392>
        var_129 = (var_128 > var_120);
        if (var_129) {
            // maxval = val                                                                   <L 393>
            var_130 = wp::copy(var_128);
            // maxrow = i                                                                     <L 394>
            var_131 = wp::copy(var_126);
        }
        var_132 = wp::where(var_129, var_130, var_120);
        var_133 = wp::where(var_129, var_131, var_121);
        wp::assign(var_120, var_132);
        wp::assign(var_121, var_133);
        wp::assign(var_13, var_128);
        goto start_for_14;
    end_for_14:;
    // if maxval < types.MJ_MINVAL:                                                           <L 395>
    var_135 = (var_120 < var_134);
    if (var_135) {
        // return A, pivot, False                                                             <L 396>
        ret_0 = var_0;
        ret_1 = var_2;
        ret_2 = var_136;
        return;
    }
    // pivot[k] = maxrow                                                                      <L 397>
    wp::assign_inplace(var_2, var_118, var_121);
    // if maxrow != k:                                                                        <L 398>
    var_137 = (var_121 != var_118);
    if (var_137) {
        // for j in range(6):                                                                 <L 399>
        // tmp = A[k, j]                                                                      <L 400>
        var_139 = wp::extract(var_0, var_118, var_138);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_140 = wp::extract(var_0, var_121, var_138);
        wp::assign_inplace(var_0, var_118, var_138, var_140);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_121, var_138, var_139);
        // tmp = A[k, j]                                                                      <L 400>
        var_142 = wp::extract(var_0, var_118, var_141);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_143 = wp::extract(var_0, var_121, var_141);
        wp::assign_inplace(var_0, var_118, var_141, var_143);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_121, var_141, var_142);
        // tmp = A[k, j]                                                                      <L 400>
        var_145 = wp::extract(var_0, var_118, var_144);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_146 = wp::extract(var_0, var_121, var_144);
        wp::assign_inplace(var_0, var_118, var_144, var_146);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_121, var_144, var_145);
        // tmp = A[k, j]                                                                      <L 400>
        var_148 = wp::extract(var_0, var_118, var_147);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_149 = wp::extract(var_0, var_121, var_147);
        wp::assign_inplace(var_0, var_118, var_147, var_149);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_121, var_147, var_148);
        // tmp = A[k, j]                                                                      <L 400>
        var_151 = wp::extract(var_0, var_118, var_150);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_152 = wp::extract(var_0, var_121, var_150);
        wp::assign_inplace(var_0, var_118, var_150, var_152);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_121, var_150, var_151);
        // tmp = A[k, j]                                                                      <L 400>
        var_154 = wp::extract(var_0, var_118, var_153);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_155 = wp::extract(var_0, var_121, var_153);
        wp::assign_inplace(var_0, var_118, var_153, var_155);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_121, var_153, var_154);
    }
    var_156 = wp::where(var_137, var_153, var_98);
    var_157 = wp::where(var_137, var_154, var_99);
    // diaginv = 1.0 / A[k, k]                                                                <L 403>
    var_159 = wp::extract(var_0, var_118, var_118);
    var_160 = wp::div(var_158, var_159);
    // for i in range(k + 1, 6):                                                              <L 404>
    var_162 = wp::add(var_118, var_161);
    var_164 = wp::range(var_162, var_163);
    start_for_17:;
        if (iter_cmp(var_164) == 0) goto end_for_17;
        var_165 = wp::iter_next(var_164);
        // A[i, k] *= diaginv                                                                 <L 405>
        var_166 = wp::extract(var_0, var_165, var_118);
        var_167 = wp::mul(var_166, var_160);
        wp::assign_inplace(var_0, var_165, var_118, var_167);
        // Aik = A[i, k]                                                                      <L 406>
        var_168 = wp::extract(var_0, var_165, var_118);
        // for j in range(k + 1, 6):                                                          <L 407>
        var_170 = wp::add(var_118, var_169);
        var_172 = wp::range(var_170, var_171);
        start_for_19:;
            if (iter_cmp(var_172) == 0) goto end_for_19;
            var_173 = wp::iter_next(var_172);
            // A[i, j] -= Aik * A[k, j]                                                       <L 408>
            var_174 = wp::extract(var_0, var_118, var_173);
            var_175 = wp::mul(var_168, var_174);
            wp::sub_inplace(var_0, var_165, var_173, var_175);
            goto start_for_19;
        end_for_19:;
        wp::assign(var_156, var_173);
        wp::assign(var_52, var_168);
        goto start_for_17;
    end_for_17:;
    // maxval = wp.abs(A[k, k])                                                               <L 388>
    var_177 = wp::extract(var_0, var_176, var_176);
    var_178 = wp::abs(var_177);
    // maxrow = k                                                                             <L 389>
    var_179 = wp::copy(var_176);
    // for i in range(k + 1, 6):                                                              <L 390>
    var_181 = wp::add(var_176, var_180);
    var_183 = wp::range(var_181, var_182);
    start_for_21:;
        if (iter_cmp(var_183) == 0) goto end_for_21;
        var_184 = wp::iter_next(var_183);
        // val = wp.abs(A[i, k])                                                              <L 391>
        var_185 = wp::extract(var_0, var_184, var_176);
        var_186 = wp::abs(var_185);
        // if val > maxval:                                                                   <L 392>
        var_187 = (var_186 > var_178);
        if (var_187) {
            // maxval = val                                                                   <L 393>
            var_188 = wp::copy(var_186);
            // maxrow = i                                                                     <L 394>
            var_189 = wp::copy(var_184);
        }
        var_190 = wp::where(var_187, var_188, var_178);
        var_191 = wp::where(var_187, var_189, var_179);
        wp::assign(var_178, var_190);
        wp::assign(var_179, var_191);
        wp::assign(var_13, var_186);
        goto start_for_21;
    end_for_21:;
    // if maxval < types.MJ_MINVAL:                                                           <L 395>
    var_193 = (var_178 < var_192);
    if (var_193) {
        // return A, pivot, False                                                             <L 396>
        ret_0 = var_0;
        ret_1 = var_2;
        ret_2 = var_194;
        return;
    }
    // pivot[k] = maxrow                                                                      <L 397>
    wp::assign_inplace(var_2, var_176, var_179);
    // if maxrow != k:                                                                        <L 398>
    var_195 = (var_179 != var_176);
    if (var_195) {
        // for j in range(6):                                                                 <L 399>
        // tmp = A[k, j]                                                                      <L 400>
        var_197 = wp::extract(var_0, var_176, var_196);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_198 = wp::extract(var_0, var_179, var_196);
        wp::assign_inplace(var_0, var_176, var_196, var_198);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_179, var_196, var_197);
        // tmp = A[k, j]                                                                      <L 400>
        var_200 = wp::extract(var_0, var_176, var_199);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_201 = wp::extract(var_0, var_179, var_199);
        wp::assign_inplace(var_0, var_176, var_199, var_201);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_179, var_199, var_200);
        // tmp = A[k, j]                                                                      <L 400>
        var_203 = wp::extract(var_0, var_176, var_202);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_204 = wp::extract(var_0, var_179, var_202);
        wp::assign_inplace(var_0, var_176, var_202, var_204);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_179, var_202, var_203);
        // tmp = A[k, j]                                                                      <L 400>
        var_206 = wp::extract(var_0, var_176, var_205);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_207 = wp::extract(var_0, var_179, var_205);
        wp::assign_inplace(var_0, var_176, var_205, var_207);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_179, var_205, var_206);
        // tmp = A[k, j]                                                                      <L 400>
        var_209 = wp::extract(var_0, var_176, var_208);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_210 = wp::extract(var_0, var_179, var_208);
        wp::assign_inplace(var_0, var_176, var_208, var_210);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_179, var_208, var_209);
        // tmp = A[k, j]                                                                      <L 400>
        var_212 = wp::extract(var_0, var_176, var_211);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_213 = wp::extract(var_0, var_179, var_211);
        wp::assign_inplace(var_0, var_176, var_211, var_213);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_179, var_211, var_212);
    }
    var_214 = wp::where(var_195, var_211, var_156);
    var_215 = wp::where(var_195, var_212, var_157);
    // diaginv = 1.0 / A[k, k]                                                                <L 403>
    var_217 = wp::extract(var_0, var_176, var_176);
    var_218 = wp::div(var_216, var_217);
    // for i in range(k + 1, 6):                                                              <L 404>
    var_220 = wp::add(var_176, var_219);
    var_222 = wp::range(var_220, var_221);
    start_for_24:;
        if (iter_cmp(var_222) == 0) goto end_for_24;
        var_223 = wp::iter_next(var_222);
        // A[i, k] *= diaginv                                                                 <L 405>
        var_224 = wp::extract(var_0, var_223, var_176);
        var_225 = wp::mul(var_224, var_218);
        wp::assign_inplace(var_0, var_223, var_176, var_225);
        // Aik = A[i, k]                                                                      <L 406>
        var_226 = wp::extract(var_0, var_223, var_176);
        // for j in range(k + 1, 6):                                                          <L 407>
        var_228 = wp::add(var_176, var_227);
        var_230 = wp::range(var_228, var_229);
        start_for_26:;
            if (iter_cmp(var_230) == 0) goto end_for_26;
            var_231 = wp::iter_next(var_230);
            // A[i, j] -= Aik * A[k, j]                                                       <L 408>
            var_232 = wp::extract(var_0, var_176, var_231);
            var_233 = wp::mul(var_226, var_232);
            wp::sub_inplace(var_0, var_223, var_231, var_233);
            goto start_for_26;
        end_for_26:;
        wp::assign(var_214, var_231);
        wp::assign(var_52, var_226);
        goto start_for_24;
    end_for_24:;
    // maxval = wp.abs(A[k, k])                                                               <L 388>
    var_235 = wp::extract(var_0, var_234, var_234);
    var_236 = wp::abs(var_235);
    // maxrow = k                                                                             <L 389>
    var_237 = wp::copy(var_234);
    // for i in range(k + 1, 6):                                                              <L 390>
    var_239 = wp::add(var_234, var_238);
    var_241 = wp::range(var_239, var_240);
    start_for_28:;
        if (iter_cmp(var_241) == 0) goto end_for_28;
        var_242 = wp::iter_next(var_241);
        // val = wp.abs(A[i, k])                                                              <L 391>
        var_243 = wp::extract(var_0, var_242, var_234);
        var_244 = wp::abs(var_243);
        // if val > maxval:                                                                   <L 392>
        var_245 = (var_244 > var_236);
        if (var_245) {
            // maxval = val                                                                   <L 393>
            var_246 = wp::copy(var_244);
            // maxrow = i                                                                     <L 394>
            var_247 = wp::copy(var_242);
        }
        var_248 = wp::where(var_245, var_246, var_236);
        var_249 = wp::where(var_245, var_247, var_237);
        wp::assign(var_236, var_248);
        wp::assign(var_237, var_249);
        wp::assign(var_13, var_244);
        goto start_for_28;
    end_for_28:;
    // if maxval < types.MJ_MINVAL:                                                           <L 395>
    var_251 = (var_236 < var_250);
    if (var_251) {
        // return A, pivot, False                                                             <L 396>
        ret_0 = var_0;
        ret_1 = var_2;
        ret_2 = var_252;
        return;
    }
    // pivot[k] = maxrow                                                                      <L 397>
    wp::assign_inplace(var_2, var_234, var_237);
    // if maxrow != k:                                                                        <L 398>
    var_253 = (var_237 != var_234);
    if (var_253) {
        // for j in range(6):                                                                 <L 399>
        // tmp = A[k, j]                                                                      <L 400>
        var_255 = wp::extract(var_0, var_234, var_254);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_256 = wp::extract(var_0, var_237, var_254);
        wp::assign_inplace(var_0, var_234, var_254, var_256);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_237, var_254, var_255);
        // tmp = A[k, j]                                                                      <L 400>
        var_258 = wp::extract(var_0, var_234, var_257);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_259 = wp::extract(var_0, var_237, var_257);
        wp::assign_inplace(var_0, var_234, var_257, var_259);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_237, var_257, var_258);
        // tmp = A[k, j]                                                                      <L 400>
        var_261 = wp::extract(var_0, var_234, var_260);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_262 = wp::extract(var_0, var_237, var_260);
        wp::assign_inplace(var_0, var_234, var_260, var_262);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_237, var_260, var_261);
        // tmp = A[k, j]                                                                      <L 400>
        var_264 = wp::extract(var_0, var_234, var_263);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_265 = wp::extract(var_0, var_237, var_263);
        wp::assign_inplace(var_0, var_234, var_263, var_265);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_237, var_263, var_264);
        // tmp = A[k, j]                                                                      <L 400>
        var_267 = wp::extract(var_0, var_234, var_266);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_268 = wp::extract(var_0, var_237, var_266);
        wp::assign_inplace(var_0, var_234, var_266, var_268);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_237, var_266, var_267);
        // tmp = A[k, j]                                                                      <L 400>
        var_270 = wp::extract(var_0, var_234, var_269);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_271 = wp::extract(var_0, var_237, var_269);
        wp::assign_inplace(var_0, var_234, var_269, var_271);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_237, var_269, var_270);
    }
    var_272 = wp::where(var_253, var_269, var_214);
    var_273 = wp::where(var_253, var_270, var_215);
    // diaginv = 1.0 / A[k, k]                                                                <L 403>
    var_275 = wp::extract(var_0, var_234, var_234);
    var_276 = wp::div(var_274, var_275);
    // for i in range(k + 1, 6):                                                              <L 404>
    var_278 = wp::add(var_234, var_277);
    var_280 = wp::range(var_278, var_279);
    start_for_31:;
        if (iter_cmp(var_280) == 0) goto end_for_31;
        var_281 = wp::iter_next(var_280);
        // A[i, k] *= diaginv                                                                 <L 405>
        var_282 = wp::extract(var_0, var_281, var_234);
        var_283 = wp::mul(var_282, var_276);
        wp::assign_inplace(var_0, var_281, var_234, var_283);
        // Aik = A[i, k]                                                                      <L 406>
        var_284 = wp::extract(var_0, var_281, var_234);
        // for j in range(k + 1, 6):                                                          <L 407>
        var_286 = wp::add(var_234, var_285);
        var_288 = wp::range(var_286, var_287);
        start_for_33:;
            if (iter_cmp(var_288) == 0) goto end_for_33;
            var_289 = wp::iter_next(var_288);
            // A[i, j] -= Aik * A[k, j]                                                       <L 408>
            var_290 = wp::extract(var_0, var_234, var_289);
            var_291 = wp::mul(var_284, var_290);
            wp::sub_inplace(var_0, var_281, var_289, var_291);
            goto start_for_33;
        end_for_33:;
        wp::assign(var_272, var_289);
        wp::assign(var_52, var_284);
        goto start_for_31;
    end_for_31:;
    // maxval = wp.abs(A[k, k])                                                               <L 388>
    var_293 = wp::extract(var_0, var_292, var_292);
    var_294 = wp::abs(var_293);
    // maxrow = k                                                                             <L 389>
    var_295 = wp::copy(var_292);
    // for i in range(k + 1, 6):                                                              <L 390>
    var_297 = wp::add(var_292, var_296);
    var_299 = wp::range(var_297, var_298);
    start_for_35:;
        if (iter_cmp(var_299) == 0) goto end_for_35;
        var_300 = wp::iter_next(var_299);
        // val = wp.abs(A[i, k])                                                              <L 391>
        var_301 = wp::extract(var_0, var_300, var_292);
        var_302 = wp::abs(var_301);
        // if val > maxval:                                                                   <L 392>
        var_303 = (var_302 > var_294);
        if (var_303) {
            // maxval = val                                                                   <L 393>
            var_304 = wp::copy(var_302);
            // maxrow = i                                                                     <L 394>
            var_305 = wp::copy(var_300);
        }
        var_306 = wp::where(var_303, var_304, var_294);
        var_307 = wp::where(var_303, var_305, var_295);
        wp::assign(var_294, var_306);
        wp::assign(var_295, var_307);
        wp::assign(var_13, var_302);
        goto start_for_35;
    end_for_35:;
    // if maxval < types.MJ_MINVAL:                                                           <L 395>
    var_309 = (var_294 < var_308);
    if (var_309) {
        // return A, pivot, False                                                             <L 396>
        ret_0 = var_0;
        ret_1 = var_2;
        ret_2 = var_310;
        return;
    }
    // pivot[k] = maxrow                                                                      <L 397>
    wp::assign_inplace(var_2, var_292, var_295);
    // if maxrow != k:                                                                        <L 398>
    var_311 = (var_295 != var_292);
    if (var_311) {
        // for j in range(6):                                                                 <L 399>
        // tmp = A[k, j]                                                                      <L 400>
        var_313 = wp::extract(var_0, var_292, var_312);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_314 = wp::extract(var_0, var_295, var_312);
        wp::assign_inplace(var_0, var_292, var_312, var_314);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_295, var_312, var_313);
        // tmp = A[k, j]                                                                      <L 400>
        var_316 = wp::extract(var_0, var_292, var_315);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_317 = wp::extract(var_0, var_295, var_315);
        wp::assign_inplace(var_0, var_292, var_315, var_317);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_295, var_315, var_316);
        // tmp = A[k, j]                                                                      <L 400>
        var_319 = wp::extract(var_0, var_292, var_318);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_320 = wp::extract(var_0, var_295, var_318);
        wp::assign_inplace(var_0, var_292, var_318, var_320);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_295, var_318, var_319);
        // tmp = A[k, j]                                                                      <L 400>
        var_322 = wp::extract(var_0, var_292, var_321);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_323 = wp::extract(var_0, var_295, var_321);
        wp::assign_inplace(var_0, var_292, var_321, var_323);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_295, var_321, var_322);
        // tmp = A[k, j]                                                                      <L 400>
        var_325 = wp::extract(var_0, var_292, var_324);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_326 = wp::extract(var_0, var_295, var_324);
        wp::assign_inplace(var_0, var_292, var_324, var_326);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_295, var_324, var_325);
        // tmp = A[k, j]                                                                      <L 400>
        var_328 = wp::extract(var_0, var_292, var_327);
        // A[k, j] = A[maxrow, j]                                                             <L 401>
        var_329 = wp::extract(var_0, var_295, var_327);
        wp::assign_inplace(var_0, var_292, var_327, var_329);
        // A[maxrow, j] = tmp                                                                 <L 402>
        wp::assign_inplace(var_0, var_295, var_327, var_328);
    }
    var_330 = wp::where(var_311, var_327, var_272);
    var_331 = wp::where(var_311, var_328, var_273);
    // diaginv = 1.0 / A[k, k]                                                                <L 403>
    var_333 = wp::extract(var_0, var_292, var_292);
    var_334 = wp::div(var_332, var_333);
    // for i in range(k + 1, 6):                                                              <L 404>
    var_336 = wp::add(var_292, var_335);
    var_338 = wp::range(var_336, var_337);
    start_for_38:;
        if (iter_cmp(var_338) == 0) goto end_for_38;
        var_339 = wp::iter_next(var_338);
        // A[i, k] *= diaginv                                                                 <L 405>
        var_340 = wp::extract(var_0, var_339, var_292);
        var_341 = wp::mul(var_340, var_334);
        wp::assign_inplace(var_0, var_339, var_292, var_341);
        // Aik = A[i, k]                                                                      <L 406>
        var_342 = wp::extract(var_0, var_339, var_292);
        // for j in range(k + 1, 6):                                                          <L 407>
        var_344 = wp::add(var_292, var_343);
        var_346 = wp::range(var_344, var_345);
        start_for_40:;
            if (iter_cmp(var_346) == 0) goto end_for_40;
            var_347 = wp::iter_next(var_346);
            // A[i, j] -= Aik * A[k, j]                                                       <L 408>
            var_348 = wp::extract(var_0, var_292, var_347);
            var_349 = wp::mul(var_342, var_348);
            wp::sub_inplace(var_0, var_339, var_347, var_349);
            goto start_for_40;
        end_for_40:;
        wp::assign(var_330, var_347);
        wp::assign(var_52, var_342);
        goto start_for_38;
    end_for_38:;
    // return A, pivot, True                                                                  <L 409>
    ret_0 = var_0;
    ret_1 = var_2;
    ret_2 = var_350;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:412
static CUDA_CALLABLE wp::vec_t<6, wp::float32> lu_solve_6x6_0(
    wp::mat_t<6, 6, wp::float32> var_A_in,
    wp::vec_t<6, wp::int32> var_pivot_in,
    wp::vec_t<6, wp::float32> var_b_in)
{
    //---------
    // primal vars
    wp::vec_t<6, wp::float32> var_0;
    const wp::int32 var_1 = 0;
    wp::int32 var_2;
    bool var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    bool var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 2;
    wp::int32 var_17;
    bool var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::int32 var_22 = 0;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    const wp::int32 var_26 = 1;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::int32 var_30 = 3;
    wp::int32 var_31;
    bool var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::int32 var_36 = 0;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 1;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::int32 var_44 = 2;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    const wp::int32 var_48 = 4;
    wp::int32 var_49;
    bool var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::int32 var_54 = 0;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::int32 var_66 = 3;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    const wp::int32 var_70 = 5;
    wp::int32 var_71;
    bool var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    const wp::int32 var_76 = 0;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 1;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    const wp::int32 var_84 = 2;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    const wp::int32 var_88 = 3;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 4;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    const wp::int32 var_96 = 5;
    const wp::int32 var_97 = 1;
    wp::int32 var_98;
    const wp::int32 var_99 = 6;
    wp::range_t var_100;
    wp::int32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    const wp::int32 var_108 = 4;
    const wp::int32 var_109 = 1;
    wp::int32 var_110;
    const wp::int32 var_111 = 6;
    wp::range_t var_112;
    wp::int32 var_113;
    wp::float32 var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    wp::float32 var_119;
    const wp::int32 var_120 = 3;
    const wp::int32 var_121 = 1;
    wp::int32 var_122;
    const wp::int32 var_123 = 6;
    wp::range_t var_124;
    wp::int32 var_125;
    wp::float32 var_126;
    wp::float32 var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 2;
    const wp::int32 var_133 = 1;
    wp::int32 var_134;
    const wp::int32 var_135 = 6;
    wp::range_t var_136;
    wp::int32 var_137;
    wp::float32 var_138;
    wp::float32 var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    wp::float32 var_142;
    wp::float32 var_143;
    const wp::int32 var_144 = 1;
    const wp::int32 var_145 = 1;
    wp::int32 var_146;
    const wp::int32 var_147 = 6;
    wp::range_t var_148;
    wp::int32 var_149;
    wp::float32 var_150;
    wp::float32 var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::float32 var_154;
    wp::float32 var_155;
    const wp::int32 var_156 = 0;
    const wp::int32 var_157 = 1;
    wp::int32 var_158;
    const wp::int32 var_159 = 6;
    wp::range_t var_160;
    wp::int32 var_161;
    wp::float32 var_162;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::float32 var_165;
    wp::float32 var_166;
    wp::float32 var_167;
    //---------
    // forward
    // def lu_solve_6x6(                                                                      <L 413>
    // x = b_in                                                                               <L 420>
    var_0 = wp::copy(var_b_in);
    // for i in range(6):                                                                     <L 421>
    // p = pivot_in[i]                                                                        <L 422>
    var_2 = wp::extract(var_pivot_in, var_1);
    // if p != i:                                                                             <L 423>
    var_3 = (var_2 != var_1);
    if (var_3) {
        // tmp = x[i]                                                                         <L 424>
        var_4 = wp::extract(var_0, var_1);
        // x[i] = x[p]                                                                        <L 425>
        var_5 = wp::extract(var_0, var_2);
        wp::assign_inplace(var_0, var_1, var_5);
        // x[p] = tmp                                                                         <L 426>
        wp::assign_inplace(var_0, var_2, var_4);
    }
    // for j in range(i):                                                                     <L 427>
    // p = pivot_in[i]                                                                        <L 422>
    var_7 = wp::extract(var_pivot_in, var_6);
    // if p != i:                                                                             <L 423>
    var_8 = (var_7 != var_6);
    if (var_8) {
        // tmp = x[i]                                                                         <L 424>
        var_9 = wp::extract(var_0, var_6);
        // x[i] = x[p]                                                                        <L 425>
        var_10 = wp::extract(var_0, var_7);
        wp::assign_inplace(var_0, var_6, var_10);
        // x[p] = tmp                                                                         <L 426>
        wp::assign_inplace(var_0, var_7, var_9);
    }
    var_11 = wp::where(var_8, var_9, var_4);
    // for j in range(i):                                                                     <L 427>
    // x[i] -= A_in[i, j] * x[j]                                                              <L 428>
    var_13 = wp::extract(var_A_in, var_6, var_12);
    var_14 = wp::extract(var_0, var_12);
    var_15 = wp::mul(var_13, var_14);
    wp::sub_inplace(var_0, var_6, var_15);
    // p = pivot_in[i]                                                                        <L 422>
    var_17 = wp::extract(var_pivot_in, var_16);
    // if p != i:                                                                             <L 423>
    var_18 = (var_17 != var_16);
    if (var_18) {
        // tmp = x[i]                                                                         <L 424>
        var_19 = wp::extract(var_0, var_16);
        // x[i] = x[p]                                                                        <L 425>
        var_20 = wp::extract(var_0, var_17);
        wp::assign_inplace(var_0, var_16, var_20);
        // x[p] = tmp                                                                         <L 426>
        wp::assign_inplace(var_0, var_17, var_19);
    }
    var_21 = wp::where(var_18, var_19, var_11);
    // for j in range(i):                                                                     <L 427>
    // x[i] -= A_in[i, j] * x[j]                                                              <L 428>
    var_23 = wp::extract(var_A_in, var_16, var_22);
    var_24 = wp::extract(var_0, var_22);
    var_25 = wp::mul(var_23, var_24);
    wp::sub_inplace(var_0, var_16, var_25);
    var_27 = wp::extract(var_A_in, var_16, var_26);
    var_28 = wp::extract(var_0, var_26);
    var_29 = wp::mul(var_27, var_28);
    wp::sub_inplace(var_0, var_16, var_29);
    // p = pivot_in[i]                                                                        <L 422>
    var_31 = wp::extract(var_pivot_in, var_30);
    // if p != i:                                                                             <L 423>
    var_32 = (var_31 != var_30);
    if (var_32) {
        // tmp = x[i]                                                                         <L 424>
        var_33 = wp::extract(var_0, var_30);
        // x[i] = x[p]                                                                        <L 425>
        var_34 = wp::extract(var_0, var_31);
        wp::assign_inplace(var_0, var_30, var_34);
        // x[p] = tmp                                                                         <L 426>
        wp::assign_inplace(var_0, var_31, var_33);
    }
    var_35 = wp::where(var_32, var_33, var_21);
    // for j in range(i):                                                                     <L 427>
    // x[i] -= A_in[i, j] * x[j]                                                              <L 428>
    var_37 = wp::extract(var_A_in, var_30, var_36);
    var_38 = wp::extract(var_0, var_36);
    var_39 = wp::mul(var_37, var_38);
    wp::sub_inplace(var_0, var_30, var_39);
    var_41 = wp::extract(var_A_in, var_30, var_40);
    var_42 = wp::extract(var_0, var_40);
    var_43 = wp::mul(var_41, var_42);
    wp::sub_inplace(var_0, var_30, var_43);
    var_45 = wp::extract(var_A_in, var_30, var_44);
    var_46 = wp::extract(var_0, var_44);
    var_47 = wp::mul(var_45, var_46);
    wp::sub_inplace(var_0, var_30, var_47);
    // p = pivot_in[i]                                                                        <L 422>
    var_49 = wp::extract(var_pivot_in, var_48);
    // if p != i:                                                                             <L 423>
    var_50 = (var_49 != var_48);
    if (var_50) {
        // tmp = x[i]                                                                         <L 424>
        var_51 = wp::extract(var_0, var_48);
        // x[i] = x[p]                                                                        <L 425>
        var_52 = wp::extract(var_0, var_49);
        wp::assign_inplace(var_0, var_48, var_52);
        // x[p] = tmp                                                                         <L 426>
        wp::assign_inplace(var_0, var_49, var_51);
    }
    var_53 = wp::where(var_50, var_51, var_35);
    // for j in range(i):                                                                     <L 427>
    // x[i] -= A_in[i, j] * x[j]                                                              <L 428>
    var_55 = wp::extract(var_A_in, var_48, var_54);
    var_56 = wp::extract(var_0, var_54);
    var_57 = wp::mul(var_55, var_56);
    wp::sub_inplace(var_0, var_48, var_57);
    var_59 = wp::extract(var_A_in, var_48, var_58);
    var_60 = wp::extract(var_0, var_58);
    var_61 = wp::mul(var_59, var_60);
    wp::sub_inplace(var_0, var_48, var_61);
    var_63 = wp::extract(var_A_in, var_48, var_62);
    var_64 = wp::extract(var_0, var_62);
    var_65 = wp::mul(var_63, var_64);
    wp::sub_inplace(var_0, var_48, var_65);
    var_67 = wp::extract(var_A_in, var_48, var_66);
    var_68 = wp::extract(var_0, var_66);
    var_69 = wp::mul(var_67, var_68);
    wp::sub_inplace(var_0, var_48, var_69);
    // p = pivot_in[i]                                                                        <L 422>
    var_71 = wp::extract(var_pivot_in, var_70);
    // if p != i:                                                                             <L 423>
    var_72 = (var_71 != var_70);
    if (var_72) {
        // tmp = x[i]                                                                         <L 424>
        var_73 = wp::extract(var_0, var_70);
        // x[i] = x[p]                                                                        <L 425>
        var_74 = wp::extract(var_0, var_71);
        wp::assign_inplace(var_0, var_70, var_74);
        // x[p] = tmp                                                                         <L 426>
        wp::assign_inplace(var_0, var_71, var_73);
    }
    var_75 = wp::where(var_72, var_73, var_53);
    // for j in range(i):                                                                     <L 427>
    // x[i] -= A_in[i, j] * x[j]                                                              <L 428>
    var_77 = wp::extract(var_A_in, var_70, var_76);
    var_78 = wp::extract(var_0, var_76);
    var_79 = wp::mul(var_77, var_78);
    wp::sub_inplace(var_0, var_70, var_79);
    var_81 = wp::extract(var_A_in, var_70, var_80);
    var_82 = wp::extract(var_0, var_80);
    var_83 = wp::mul(var_81, var_82);
    wp::sub_inplace(var_0, var_70, var_83);
    var_85 = wp::extract(var_A_in, var_70, var_84);
    var_86 = wp::extract(var_0, var_84);
    var_87 = wp::mul(var_85, var_86);
    wp::sub_inplace(var_0, var_70, var_87);
    var_89 = wp::extract(var_A_in, var_70, var_88);
    var_90 = wp::extract(var_0, var_88);
    var_91 = wp::mul(var_89, var_90);
    wp::sub_inplace(var_0, var_70, var_91);
    var_93 = wp::extract(var_A_in, var_70, var_92);
    var_94 = wp::extract(var_0, var_92);
    var_95 = wp::mul(var_93, var_94);
    wp::sub_inplace(var_0, var_70, var_95);
    // for i in range(5, -1, -1):                                                             <L 429>
    // for j in range(i + 1, 6):                                                              <L 430>
    var_98 = wp::add(var_96, var_97);
    var_100 = wp::range(var_98, var_99);
    start_for_0:;
        if (iter_cmp(var_100) == 0) goto end_for_0;
        var_101 = wp::iter_next(var_100);
        // x[i] -= A_in[i, j] * x[j]                                                          <L 431>
        var_102 = wp::extract(var_A_in, var_96, var_101);
        var_103 = wp::extract(var_0, var_101);
        var_104 = wp::mul(var_102, var_103);
        wp::sub_inplace(var_0, var_96, var_104);
        goto start_for_0;
    end_for_0:;
    // x[i] /= A_in[i, i]                                                                     <L 432>
    var_105 = wp::extract(var_0, var_96);
    var_106 = wp::extract(var_A_in, var_96, var_96);
    var_107 = wp::div(var_105, var_106);
    wp::assign_inplace(var_0, var_96, var_107);
    // for j in range(i + 1, 6):                                                              <L 430>
    var_110 = wp::add(var_108, var_109);
    var_112 = wp::range(var_110, var_111);
    start_for_2:;
        if (iter_cmp(var_112) == 0) goto end_for_2;
        var_113 = wp::iter_next(var_112);
        // x[i] -= A_in[i, j] * x[j]                                                          <L 431>
        var_114 = wp::extract(var_A_in, var_108, var_113);
        var_115 = wp::extract(var_0, var_113);
        var_116 = wp::mul(var_114, var_115);
        wp::sub_inplace(var_0, var_108, var_116);
        goto start_for_2;
    end_for_2:;
    // x[i] /= A_in[i, i]                                                                     <L 432>
    var_117 = wp::extract(var_0, var_108);
    var_118 = wp::extract(var_A_in, var_108, var_108);
    var_119 = wp::div(var_117, var_118);
    wp::assign_inplace(var_0, var_108, var_119);
    // for j in range(i + 1, 6):                                                              <L 430>
    var_122 = wp::add(var_120, var_121);
    var_124 = wp::range(var_122, var_123);
    start_for_4:;
        if (iter_cmp(var_124) == 0) goto end_for_4;
        var_125 = wp::iter_next(var_124);
        // x[i] -= A_in[i, j] * x[j]                                                          <L 431>
        var_126 = wp::extract(var_A_in, var_120, var_125);
        var_127 = wp::extract(var_0, var_125);
        var_128 = wp::mul(var_126, var_127);
        wp::sub_inplace(var_0, var_120, var_128);
        goto start_for_4;
    end_for_4:;
    // x[i] /= A_in[i, i]                                                                     <L 432>
    var_129 = wp::extract(var_0, var_120);
    var_130 = wp::extract(var_A_in, var_120, var_120);
    var_131 = wp::div(var_129, var_130);
    wp::assign_inplace(var_0, var_120, var_131);
    // for j in range(i + 1, 6):                                                              <L 430>
    var_134 = wp::add(var_132, var_133);
    var_136 = wp::range(var_134, var_135);
    start_for_6:;
        if (iter_cmp(var_136) == 0) goto end_for_6;
        var_137 = wp::iter_next(var_136);
        // x[i] -= A_in[i, j] * x[j]                                                          <L 431>
        var_138 = wp::extract(var_A_in, var_132, var_137);
        var_139 = wp::extract(var_0, var_137);
        var_140 = wp::mul(var_138, var_139);
        wp::sub_inplace(var_0, var_132, var_140);
        goto start_for_6;
    end_for_6:;
    // x[i] /= A_in[i, i]                                                                     <L 432>
    var_141 = wp::extract(var_0, var_132);
    var_142 = wp::extract(var_A_in, var_132, var_132);
    var_143 = wp::div(var_141, var_142);
    wp::assign_inplace(var_0, var_132, var_143);
    // for j in range(i + 1, 6):                                                              <L 430>
    var_146 = wp::add(var_144, var_145);
    var_148 = wp::range(var_146, var_147);
    start_for_8:;
        if (iter_cmp(var_148) == 0) goto end_for_8;
        var_149 = wp::iter_next(var_148);
        // x[i] -= A_in[i, j] * x[j]                                                          <L 431>
        var_150 = wp::extract(var_A_in, var_144, var_149);
        var_151 = wp::extract(var_0, var_149);
        var_152 = wp::mul(var_150, var_151);
        wp::sub_inplace(var_0, var_144, var_152);
        goto start_for_8;
    end_for_8:;
    // x[i] /= A_in[i, i]                                                                     <L 432>
    var_153 = wp::extract(var_0, var_144);
    var_154 = wp::extract(var_A_in, var_144, var_144);
    var_155 = wp::div(var_153, var_154);
    wp::assign_inplace(var_0, var_144, var_155);
    // for j in range(i + 1, 6):                                                              <L 430>
    var_158 = wp::add(var_156, var_157);
    var_160 = wp::range(var_158, var_159);
    start_for_10:;
        if (iter_cmp(var_160) == 0) goto end_for_10;
        var_161 = wp::iter_next(var_160);
        // x[i] -= A_in[i, j] * x[j]                                                          <L 431>
        var_162 = wp::extract(var_A_in, var_156, var_161);
        var_163 = wp::extract(var_0, var_161);
        var_164 = wp::mul(var_162, var_163);
        wp::sub_inplace(var_0, var_156, var_164);
        goto start_for_10;
    end_for_10:;
    // x[i] /= A_in[i, i]                                                                     <L 432>
    var_165 = wp::extract(var_0, var_156);
    var_166 = wp::extract(var_A_in, var_156, var_156);
    var_167 = wp::div(var_165, var_166);
    wp::assign_inplace(var_0, var_156, var_167);
    // return x                                                                               <L 433>
    return var_0;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:52
static CUDA_CALLABLE void adj_axis_angle_to_quat_0(
    wp::vec_t<3, wp::float32> var_axis,
    wp::float32 var_angle,
    wp::vec_t<3, wp::float32> & adj_axis,
    wp::float32 & adj_angle,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:23
static CUDA_CALLABLE void adj_mul_quat_0(
    wp::quat_t<wp::float32> var_u,
    wp::quat_t<wp::float32> var_v,
    wp::quat_t<wp::float32> & adj_u,
    wp::quat_t<wp::float32> & adj_v,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:188
static CUDA_CALLABLE void adj_quat_integrate_0(
    wp::quat_t<wp::float32> var_q,
    wp::vec_t<3, wp::float32> var_v,
    wp::float32 var_dt,
    wp::quat_t<wp::float32> & adj_q,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_dt,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:675
static CUDA_CALLABLE void adj__sigmoid_0(
    wp::float32 var_x,
    wp::float32 & adj_x,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:689
static CUDA_CALLABLE void adj_muscle_dynamics_timescale_0(
    wp::float32 var_dctrl,
    wp::float32 var_tau_act,
    wp::float32 var_tau_deact,
    wp::float32 var_smooth_width,
    wp::float32 & adj_dctrl,
    wp::float32 & adj_tau_act,
    wp::float32 & adj_tau_deact,
    wp::float32 & adj_smooth_width,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:703
static CUDA_CALLABLE void adj_muscle_dynamics_0(
    wp::float32 var_control,
    wp::float32 var_activation,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_control,
    wp::float32 & adj_activation,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:724
static CUDA_CALLABLE void adj_dcmotor_slots_0(
    wp::vec_t<10, wp::float32> var_dynprm,
    wp::vec_t<10, wp::float32> var_gainprm,
    wp::vec_t<10, wp::float32> & adj_dynprm,
    wp::vec_t<10, wp::float32> & adj_gainprm,
    wp::vec_t<6, wp::int32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:792
static CUDA_CALLABLE void adj_unpack_servo_inputs_0(
    wp::array_t<wp::float32> var_ctrl_in,
    wp::int32 var_worldid,
    wp::int32 var_uadr,
    wp::int32 var_spec,
    wp::float32 var_u_first,
    wp::array_t<wp::float32> & adj_ctrl_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_uadr,
    wp::int32 & adj_spec,
    wp::float32 & adj_u_first,
    wp::vec_t<4, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:822
static CUDA_CALLABLE void adj_dcmotor_voltage_0(
    wp::array_t<wp::float32> var_ctrl_in,
    wp::int32 var_worldid,
    wp::int32 var_uadr,
    wp::int32 var_spec,
    wp::float32 var_u_first,
    wp::float32 var_length,
    wp::float32 var_velocity,
    wp::float32 var_x_I,
    wp::vec_t<10, wp::float32> var_gainprm,
    wp::array_t<wp::float32> & adj_ctrl_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_uadr,
    wp::int32 & adj_spec,
    wp::float32 & adj_u_first,
    wp::float32 & adj_length,
    wp::float32 & adj_velocity,
    wp::float32 & adj_x_I,
    wp::vec_t<10, wp::float32> & adj_gainprm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:786
static CUDA_CALLABLE void adj_lugre_stribeck_0(
    wp::float32 var_velocity,
    wp::float32 var_F_C,
    wp::float32 var_F_S,
    wp::float32 var_v_S,
    wp::float32 & adj_velocity,
    wp::float32 & adj_F_C,
    wp::float32 & adj_F_S,
    wp::float32 & adj_v_S,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:37
static CUDA_CALLABLE void adj_next_act_0(
    wp::float32 var_opt_timestep,
    wp::int32 var_actuator_dyntype,
    wp::vec_t<10, wp::float32> var_actuator_dynprm,
    wp::vec_t<2, wp::float32> var_actuator_actrange,
    wp::float32 var_act_in,
    wp::float32 var_act_dot_in,
    wp::float32 var_act_dot_scale,
    bool var_clamp,
    wp::float32 & adj_opt_timestep,
    wp::int32 & adj_actuator_dyntype,
    wp::vec_t<10, wp::float32> & adj_actuator_dynprm,
    wp::vec_t<2, wp::float32> & adj_actuator_actrange,
    wp::float32 & adj_act_in,
    wp::float32 & adj_act_dot_in,
    wp::float32 & adj_act_dot_scale,
    bool & adj_clamp,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:459
static CUDA_CALLABLE void adj_muscle_gain_length_0(
    wp::float32 var_length,
    wp::float32 var_lmin,
    wp::float32 var_lmax,
    wp::float32 & adj_length,
    wp::float32 & adj_lmin,
    wp::float32 & adj_lmax,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:483
static CUDA_CALLABLE void adj_muscle_gain_0(
    wp::float32 var_len,
    wp::float32 var_vel,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_len,
    wp::float32 & adj_vel,
    wp::vec_t<2, wp::float32> & adj_lengthrange,
    wp::float32 & adj_acc0,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:764
static CUDA_CALLABLE void adj_dcmotor_resistance_0(
    wp::array_t<wp::float32> var_act_in,
    wp::int32 var_worldid,
    wp::int32 var_actadr,
    wp::vec_t<10, wp::float32> var_dynprm,
    wp::vec_t<10, wp::float32> var_gainprm,
    wp::array_t<wp::float32> & adj_act_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_actadr,
    wp::vec_t<10, wp::float32> & adj_dynprm,
    wp::vec_t<10, wp::float32> & adj_gainprm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:524
static CUDA_CALLABLE void adj_muscle_bias_0(
    wp::float32 var_len,
    wp::vec_t<2, wp::float32> var_lengthrange,
    wp::float32 var_acc0,
    wp::vec_t<10, wp::float32> var_prm,
    wp::float32 & adj_len,
    wp::vec_t<2, wp::float32> & adj_lengthrange,
    wp::float32 & adj_acc0,
    wp::vec_t<10, wp::float32> & adj_prm,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\forward.py:167
static CUDA_CALLABLE void adj__wrap_period_0(
    wp::array_t<wp::int32> var_jnt_type,
    wp::int32 var_trntype,
    wp::int32 var_dyntype,
    wp::int32 var_gaintype,
    wp::int32 var_biastype,
    wp::vec_t<2, wp::int32> var_trnid,
    wp::vec_t<10, wp::float32> var_gainprm,
    wp::vec_t<10, wp::float32> var_biasprm,
    wp::vec_t<6, wp::float32> var_gear,
    wp::array_t<wp::int32> & adj_jnt_type,
    wp::int32 & adj_trntype,
    wp::int32 & adj_dyntype,
    wp::int32 & adj_gaintype,
    wp::int32 & adj_biastype,
    wp::vec_t<2, wp::int32> & adj_trnid,
    wp::vec_t<10, wp::float32> & adj_gainprm,
    wp::vec_t<10, wp::float32> & adj_biasprm,
    wp::vec_t<6, wp::float32> & adj_gear,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:900
static CUDA_CALLABLE void adj__poly_force_deriv_0(
    wp::float32 var_linear,
    wp::vec_t<2, wp::float32> var_poly,
    wp::float32 var_x,
    wp::int32 var_flg_odd,
    wp::float32 & adj_linear,
    wp::vec_t<2, wp::float32> & adj_poly,
    wp::float32 & adj_x,
    wp::int32 & adj_flg_odd,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:339
static CUDA_CALLABLE void adj_free_bias_vel_blocks_0(
    wp::float32 var_mass,
    wp::mat_t<3, 3, wp::float32> var_R,
    wp::mat_t<3, 3, wp::float32> var_Iw,
    wp::vec_t<3, wp::float32> var_s,
    wp::vec_t<3, wp::float32> var_qvel_rot,
    wp::mat_t<3, 3, wp::float32> & ret_0,
    wp::mat_t<3, 3, wp::float32> & ret_1,
    wp::float32 & adj_mass,
    wp::mat_t<3, 3, wp::float32> & adj_R,
    wp::mat_t<3, 3, wp::float32> & adj_Iw,
    wp::vec_t<3, wp::float32> & adj_s,
    wp::vec_t<3, wp::float32> & adj_qvel_rot,
    wp::mat_t<3, 3, wp::float32> & adj_ret_0,
    wp::mat_t<3, 3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\passive.py:53
static CUDA_CALLABLE void adj_geom_semiaxes_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_geom_type,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::int32 & adj_geom_type,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\passive.py:47
static CUDA_CALLABLE void adj__pow4_0(
    wp::float32 var_val,
    wp::float32 & adj_val,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\passive.py:73
static CUDA_CALLABLE void adj_ellipsoid_max_moment_0(
    wp::vec_t<3, wp::float32> var_size,
    wp::int32 var_dir,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::int32 & adj_dir,
    wp::float32 & adj_ret)
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:605
static CUDA_CALLABLE void adj__geom_ellipsoid_fluid_B_0(
    wp::vec_t<3, wp::float32> var_semiaxes,
    wp::float32 var_blunt_drag_coef,
    wp::float32 var_slender_drag_coef,
    wp::float32 var_ang_drag_coef,
    wp::float32 var_kutta_lift_coef,
    wp::float32 var_magnus_lift_coef,
    wp::vec_t<3, wp::float32> var_virtual_mass,
    wp::vec_t<3, wp::float32> var_virtual_inertia,
    wp::vec_t<3, wp::float32> var_ang_vel,
    wp::vec_t<3, wp::float32> var_lin_vel,
    wp::float32 var_density,
    wp::float32 var_viscosity,
    bool var_flg_dragonly,
    wp::mat_t<3, 3, wp::float32> & ret_0,
    wp::mat_t<3, 3, wp::float32> & ret_1,
    wp::mat_t<3, 3, wp::float32> & ret_2,
    wp::mat_t<3, 3, wp::float32> & ret_3,
    wp::vec_t<3, wp::float32> & adj_semiaxes,
    wp::float32 & adj_blunt_drag_coef,
    wp::float32 & adj_slender_drag_coef,
    wp::float32 & adj_ang_drag_coef,
    wp::float32 & adj_kutta_lift_coef,
    wp::float32 & adj_magnus_lift_coef,
    wp::vec_t<3, wp::float32> & adj_virtual_mass,
    wp::vec_t<3, wp::float32> & adj_virtual_inertia,
    wp::vec_t<3, wp::float32> & adj_ang_vel,
    wp::vec_t<3, wp::float32> & adj_lin_vel,
    wp::float32 & adj_density,
    wp::float32 & adj_viscosity,
    bool & adj_flg_dragonly,
    wp::mat_t<3, 3, wp::float32> & adj_ret_0,
    wp::mat_t<3, 3, wp::float32> & adj_ret_1,
    wp::mat_t<3, 3, wp::float32> & adj_ret_2,
    wp::mat_t<3, 3, wp::float32> & adj_ret_3)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\forward.py:734
static CUDA_CALLABLE void adj__project_spatial_B_0(
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::int32 var_worldid,
    wp::int32 var_dof_adr,
    wp::mat_t<3, 3, wp::float32> var_geom_rotT,
    wp::vec_t<3, wp::float32> var_offset,
    wp::mat_t<3, 3, wp::float32> var_B00,
    wp::mat_t<3, 3, wp::float32> var_B01,
    wp::mat_t<3, 3, wp::float32> var_B10,
    wp::mat_t<3, 3, wp::float32> var_B11,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_dof_adr,
    wp::mat_t<3, 3, wp::float32> & adj_geom_rotT,
    wp::vec_t<3, wp::float32> & adj_offset,
    wp::mat_t<3, 3, wp::float32> & adj_B00,
    wp::mat_t<3, 3, wp::float32> & adj_B01,
    wp::mat_t<3, 3, wp::float32> & adj_B10,
    wp::mat_t<3, 3, wp::float32> & adj_B11,
    wp::mat_t<6, 6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:996
static CUDA_CALLABLE void adj__deriv_box_fluid_0(
    wp::int32 var_opt_integrator,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::int32 var_worldid,
    wp::int32 var_bodyid,
    wp::vec_t<6, wp::float32> var_lvel,
    wp::float32 var_density,
    wp::float32 var_viscosity,
    wp::int32 & adj_opt_integrator,
    wp::array_t<wp::float32> & adj_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_body_inertia,
    wp::int32 & adj_worldid,
    wp::int32 & adj_bodyid,
    wp::vec_t<6, wp::float32> & adj_lvel,
    wp::float32 & adj_density,
    wp::float32 & adj_viscosity,
    wp::mat_t<6, 6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:379
static CUDA_CALLABLE void adj_lu_factor_6x6_0(
    wp::mat_t<6, 6, wp::float32> var_A_in,
    wp::mat_t<6, 6, wp::float32> & ret_0,
    wp::vec_t<6, wp::int32> & ret_1,
    bool & ret_2,
    wp::mat_t<6, 6, wp::float32> & adj_A_in,
    wp::mat_t<6, 6, wp::float32> & adj_ret_0,
    wp::vec_t<6, wp::int32> & adj_ret_1,
    bool & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:412
static CUDA_CALLABLE void adj_lu_solve_6x6_0(
    wp::mat_t<6, 6, wp::float32> var_A_in,
    wp::vec_t<6, wp::int32> var_pivot_in,
    wp::vec_t<6, wp::float32> var_b_in,
    wp::mat_t<6, 6, wp::float32> & adj_A_in,
    wp::vec_t<6, wp::int32> & adj_pivot_in,
    wp::vec_t<6, wp::float32> & adj_b_in,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _actuator_velocity_00a052f5_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_actuator_velocity_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::float32 var_8 = 0.0;
        wp::float32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::float32* var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        //---------
        // forward
        // def _actuator_velocity(                                                                <L 1479>
        // worldid, actid = wp.tid()                                                              <L 1489>
        builtin_tid2d(var_0, var_1);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 1491>
        var_2 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 1492>
        var_5 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // vel = float(0.0)                                                                       <L 1494>
        var_9 = wp::float(var_8);
        // for i in range(rownnz):                                                                <L 1495>
        var_10 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // sparseid = rowadr + i                                                              <L 1496>
            var_12 = wp::add(var_6, var_11);
            // colind = moment_colind_in[worldid, sparseid]                                       <L 1497>
            var_13 = wp::address(var_moment_colind_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // vel += actuator_moment_in[worldid, sparseid] * qvel_in[worldid, colind]            <L 1498>
            var_16 = wp::address(var_actuator_moment_in, var_0, var_12);
            var_17 = wp::address(var_qvel_in, var_0, var_14);
            var_19 = wp::load(var_16);
            var_20 = wp::load(var_17);
            var_18 = wp::mul(var_19, var_20);
            var_21 = wp::add(var_9, var_18);
            wp::assign(var_9, var_21);
            goto start_for_0;
        end_for_0:;
        // actuator_velocity_out[worldid, actid] = vel                                            <L 1500>
        wp::array_store(var_actuator_velocity_out, var_0, var_1, var_9);
    }
}



extern "C" __global__ void _next_position_1c101e8d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_qvel_in,
    wp::float32 var_qvel_scale_in,
    wp::array_t<wp::float32> var_qpos_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::array_t<wp::float32> var_19;
        wp::array_t<wp::float32> var_20;
        wp::array_t<wp::float32> var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::float32* var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::float32* var_27;
        const wp::int32 var_28 = 2;
        wp::int32 var_29;
        wp::float32* var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32* var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::float32* var_38;
        const wp::int32 var_39 = 2;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        const wp::int32 var_49 = 3;
        wp::int32 var_50;
        wp::float32* var_51;
        const wp::int32 var_52 = 4;
        wp::int32 var_53;
        wp::float32* var_54;
        const wp::int32 var_55 = 5;
        wp::int32 var_56;
        wp::float32* var_57;
        const wp::int32 var_58 = 6;
        wp::int32 var_59;
        wp::float32* var_60;
        wp::quat_t<wp::float32> var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        const wp::int32 var_66 = 3;
        wp::int32 var_67;
        wp::float32* var_68;
        const wp::int32 var_69 = 4;
        wp::int32 var_70;
        wp::float32* var_71;
        const wp::int32 var_72 = 5;
        wp::int32 var_73;
        wp::float32* var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::quat_t<wp::float32> var_80;
        const wp::int32 var_81 = 0;
        wp::float32 var_82;
        const wp::int32 var_83 = 0;
        wp::int32 var_84;
        const wp::int32 var_85 = 1;
        wp::float32 var_86;
        const wp::int32 var_87 = 1;
        wp::int32 var_88;
        const wp::int32 var_89 = 2;
        wp::float32 var_90;
        const wp::int32 var_91 = 2;
        wp::int32 var_92;
        const wp::int32 var_93 = 0;
        wp::float32 var_94;
        const wp::int32 var_95 = 3;
        wp::int32 var_96;
        const wp::int32 var_97 = 1;
        wp::float32 var_98;
        const wp::int32 var_99 = 4;
        wp::int32 var_100;
        const wp::int32 var_101 = 2;
        wp::float32 var_102;
        const wp::int32 var_103 = 5;
        wp::int32 var_104;
        const wp::int32 var_105 = 3;
        wp::float32 var_106;
        const wp::int32 var_107 = 6;
        wp::int32 var_108;
        const wp::int32 var_109 = 1;
        bool var_110;
        const wp::int32 var_111 = 0;
        wp::int32 var_112;
        wp::float32* var_113;
        const wp::int32 var_114 = 1;
        wp::int32 var_115;
        wp::float32* var_116;
        const wp::int32 var_117 = 2;
        wp::int32 var_118;
        wp::float32* var_119;
        const wp::int32 var_120 = 3;
        wp::int32 var_121;
        wp::float32* var_122;
        wp::quat_t<wp::float32> var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::float32* var_128;
        const wp::int32 var_129 = 1;
        wp::int32 var_130;
        wp::float32* var_131;
        const wp::int32 var_132 = 2;
        wp::int32 var_133;
        wp::float32* var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::quat_t<wp::float32> var_140;
        const wp::int32 var_141 = 0;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::int32 var_144;
        const wp::int32 var_145 = 1;
        wp::float32 var_146;
        const wp::int32 var_147 = 1;
        wp::int32 var_148;
        const wp::int32 var_149 = 2;
        wp::float32 var_150;
        const wp::int32 var_151 = 2;
        wp::int32 var_152;
        const wp::int32 var_153 = 3;
        wp::float32 var_154;
        const wp::int32 var_155 = 3;
        wp::int32 var_156;
        wp::float32* var_157;
        wp::float32* var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        wp::float32 var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::quat_t<wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::quat_t<wp::float32> var_166;
        //---------
        // forward
        // def _next_position(                                                                    <L 87>
        // worldid, jntid = wp.tid()                                                              <L 101>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 102>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // jnttype = jnt_type[jntid]                                                              <L 104>
        var_10 = wp::address(var_jnt_type, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // qpos_adr = jnt_qposadr[jntid]                                                          <L 105>
        var_13 = wp::address(var_jnt_qposadr, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // dof_adr = jnt_dofadr[jntid]                                                            <L 106>
        var_16 = wp::address(var_jnt_dofadr, var_1);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // qpos = qpos_in[worldid]                                                                <L 107>
        var_19 = wp::view(var_qpos_in, var_0);
        // qpos_next = qpos_out[worldid]                                                          <L 108>
        var_20 = wp::view(var_qpos_out, var_0);
        // qvel = qvel_in[worldid]                                                                <L 109>
        var_21 = wp::view(var_qvel_in, var_0);
        // if jnttype == JointType.FREE:                                                          <L 111>
        var_23 = (var_11 == var_22);
        if (var_23) {
            // qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])         <L 112>
            var_24 = wp::address(var_19, var_14);
            var_26 = wp::add(var_14, var_25);
            var_27 = wp::address(var_19, var_26);
            var_29 = wp::add(var_14, var_28);
            var_30 = wp::address(var_19, var_29);
            var_32 = wp::load(var_24);
            var_33 = wp::load(var_27);
            var_34 = wp::load(var_30);
            var_31 = wp::vec_t<3, wp::float32>(var_32, var_33, var_34);
            // qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in       <L 113>
            var_35 = wp::address(var_21, var_17);
            var_37 = wp::add(var_17, var_36);
            var_38 = wp::address(var_21, var_37);
            var_40 = wp::add(var_17, var_39);
            var_41 = wp::address(var_21, var_40);
            var_43 = wp::load(var_35);
            var_44 = wp::load(var_38);
            var_45 = wp::load(var_41);
            var_42 = wp::vec_t<3, wp::float32>(var_43, var_44, var_45);
            var_46 = wp::mul(var_42, var_qvel_scale_in);
            // qpos_new = qpos_pos + timestep * qvel_lin                                          <L 115>
            var_47 = wp::mul(var_8, var_46);
            var_48 = wp::add(var_31, var_47);
            // qpos_quat = wp.quat(                                                               <L 117>
            // qpos[qpos_adr + 3],                                                                <L 118>
            var_50 = wp::add(var_14, var_49);
            var_51 = wp::address(var_19, var_50);
            // qpos[qpos_adr + 4],                                                                <L 119>
            var_53 = wp::add(var_14, var_52);
            var_54 = wp::address(var_19, var_53);
            // qpos[qpos_adr + 5],                                                                <L 120>
            var_56 = wp::add(var_14, var_55);
            var_57 = wp::address(var_19, var_56);
            // qpos[qpos_adr + 6],                                                                <L 121>
            var_59 = wp::add(var_14, var_58);
            var_60 = wp::address(var_19, var_59);
            var_62 = wp::load(var_51);
            var_63 = wp::load(var_54);
            var_64 = wp::load(var_57);
            var_65 = wp::load(var_60);
            var_61 = wp::quat_t<wp::float32>(var_62, var_63, var_64, var_65);
            // qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5]) * qvel_scale_in       <L 123>
            var_67 = wp::add(var_17, var_66);
            var_68 = wp::address(var_21, var_67);
            var_70 = wp::add(var_17, var_69);
            var_71 = wp::address(var_21, var_70);
            var_73 = wp::add(var_17, var_72);
            var_74 = wp::address(var_21, var_73);
            var_76 = wp::load(var_68);
            var_77 = wp::load(var_71);
            var_78 = wp::load(var_74);
            var_75 = wp::vec_t<3, wp::float32>(var_76, var_77, var_78);
            var_79 = wp::mul(var_75, var_qvel_scale_in);
            // qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)                 <L 125>
            var_80 = quat_integrate_0(var_61, var_79, var_8);
            // qpos_next[qpos_adr + 0] = qpos_new[0]                                              <L 127>
            var_82 = wp::extract(var_48, var_81);
            var_84 = wp::add(var_14, var_83);
            wp::array_store(var_20, var_84, var_82);
            // qpos_next[qpos_adr + 1] = qpos_new[1]                                              <L 128>
            var_86 = wp::extract(var_48, var_85);
            var_88 = wp::add(var_14, var_87);
            wp::array_store(var_20, var_88, var_86);
            // qpos_next[qpos_adr + 2] = qpos_new[2]                                              <L 129>
            var_90 = wp::extract(var_48, var_89);
            var_92 = wp::add(var_14, var_91);
            wp::array_store(var_20, var_92, var_90);
            // qpos_next[qpos_adr + 3] = qpos_quat_new[0]                                         <L 130>
            var_94 = wp::extract(var_80, var_93);
            var_96 = wp::add(var_14, var_95);
            wp::array_store(var_20, var_96, var_94);
            // qpos_next[qpos_adr + 4] = qpos_quat_new[1]                                         <L 131>
            var_98 = wp::extract(var_80, var_97);
            var_100 = wp::add(var_14, var_99);
            wp::array_store(var_20, var_100, var_98);
            // qpos_next[qpos_adr + 5] = qpos_quat_new[2]                                         <L 132>
            var_102 = wp::extract(var_80, var_101);
            var_104 = wp::add(var_14, var_103);
            wp::array_store(var_20, var_104, var_102);
            // qpos_next[qpos_adr + 6] = qpos_quat_new[3]                                         <L 133>
            var_106 = wp::extract(var_80, var_105);
            var_108 = wp::add(var_14, var_107);
            wp::array_store(var_20, var_108, var_106);
        }
        if (!var_23) {
            // elif jnttype == JointType.BALL:                                                    <L 135>
            var_110 = (var_11 == var_109);
            if (var_110) {
                // qpos_quat = wp.quat(qpos[qpos_adr + 0], qpos[qpos_adr + 1], qpos[qpos_adr + 2], qpos[qpos_adr + 3])       <L 136>
                var_112 = wp::add(var_14, var_111);
                var_113 = wp::address(var_19, var_112);
                var_115 = wp::add(var_14, var_114);
                var_116 = wp::address(var_19, var_115);
                var_118 = wp::add(var_14, var_117);
                var_119 = wp::address(var_19, var_118);
                var_121 = wp::add(var_14, var_120);
                var_122 = wp::address(var_19, var_121);
                var_124 = wp::load(var_113);
                var_125 = wp::load(var_116);
                var_126 = wp::load(var_119);
                var_127 = wp::load(var_122);
                var_123 = wp::quat_t<wp::float32>(var_124, var_125, var_126, var_127);
                // qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in       <L 137>
                var_128 = wp::address(var_21, var_17);
                var_130 = wp::add(var_17, var_129);
                var_131 = wp::address(var_21, var_130);
                var_133 = wp::add(var_17, var_132);
                var_134 = wp::address(var_21, var_133);
                var_136 = wp::load(var_128);
                var_137 = wp::load(var_131);
                var_138 = wp::load(var_134);
                var_135 = wp::vec_t<3, wp::float32>(var_136, var_137, var_138);
                var_139 = wp::mul(var_135, var_qvel_scale_in);
                // qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)             <L 139>
                var_140 = quat_integrate_0(var_123, var_139, var_8);
                // qpos_next[qpos_adr + 0] = qpos_quat_new[0]                                     <L 141>
                var_142 = wp::extract(var_140, var_141);
                var_144 = wp::add(var_14, var_143);
                wp::array_store(var_20, var_144, var_142);
                // qpos_next[qpos_adr + 1] = qpos_quat_new[1]                                     <L 142>
                var_146 = wp::extract(var_140, var_145);
                var_148 = wp::add(var_14, var_147);
                wp::array_store(var_20, var_148, var_146);
                // qpos_next[qpos_adr + 2] = qpos_quat_new[2]                                     <L 143>
                var_150 = wp::extract(var_140, var_149);
                var_152 = wp::add(var_14, var_151);
                wp::array_store(var_20, var_152, var_150);
                // qpos_next[qpos_adr + 3] = qpos_quat_new[3]                                     <L 144>
                var_154 = wp::extract(var_140, var_153);
                var_156 = wp::add(var_14, var_155);
                wp::array_store(var_20, var_156, var_154);
            }
            if (!var_110) {
                // qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr] * qvel_scale_in       <L 147>
                var_157 = wp::address(var_19, var_14);
                var_158 = wp::address(var_21, var_17);
                var_160 = wp::load(var_158);
                var_159 = wp::mul(var_8, var_160);
                var_161 = wp::mul(var_159, var_qvel_scale_in);
                var_163 = wp::load(var_157);
                var_162 = wp::add(var_163, var_161);
                wp::array_store(var_20, var_14, var_162);
            }
        }
        var_164 = wp::where(var_23, var_61, var_123);
        var_165 = wp::where(var_23, var_79, var_139);
        var_166 = wp::where(var_23, var_80, var_140);
    }
}



extern "C" __global__ void _map_m2d_273b5a0a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_mapM2D,
    wp::array_t<wp::float32> var_qH_M,
    wp::array_t<wp::float32> var_qLU_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        //---------
        // forward
        // def _map_m2d(                                                                          <L 693>
        // worldid, elemid = wp.tid()                                                             <L 702>
        builtin_tid2d(var_0, var_1);
        // m_idx = mapM2D[elemid]                                                                 <L 703>
        var_2 = wp::address(var_mapM2D, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if m_idx >= 0:                                                                         <L 704>
        var_6 = (var_3 >= var_5);
        if (var_6) {
            // qLU_out[worldid, elemid] = qH_M[worldid, m_idx]                                    <L 705>
            var_7 = wp::address(var_qH_M, var_0, var_3);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            wp::array_store(var_qLU_out, var_0, var_1, var_8);
        }
        if (!var_6) {
            // qLU_out[worldid, elemid] = 0.0                                                     <L 707>
            wp::array_store(var_qLU_out, var_0, var_1, var_10);
        }
    }
}



extern "C" __global__ void _tendon_actuator_force_2da6552b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_ten_actfrc_out)
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
        wp::int32* var_2;
        const wp::int32 var_3 = 3;
        bool var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        wp::array_t<wp::float32> var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        //---------
        // forward
        // def _tendon_actuator_force(                                                            <L 1889>
        // worldid, actid = wp.tid()                                                              <L 1898>
        builtin_tid2d(var_0, var_1);
        // if actuator_trntype[actid] == TrnType.TENDON:                                          <L 1900>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // tenid = actuator_trnid[actid][0]                                                   <L 1901>
            var_6 = wp::address(var_actuator_trnid, var_1);
            var_9 = wp::load(var_6);
            var_8 = wp::extract(var_9, var_7);
            // wp.atomic_add(ten_actfrc_out[worldid], tenid, actuator_force_in[worldid, actid])       <L 1903>
            var_10 = wp::view(var_ten_actfrc_out, var_0);
            var_11 = wp::address(var_actuator_force_in, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::atomic_add(var_10, var_8, var_13);
        }
    }
}



extern "C" __global__ void _rk_accumulate_activation_velocity_5935ab3f_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::float32 var_scale,
    wp::array_t<wp::float32> var_act_dot_out)
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
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        //---------
        // forward
        // def _rk_accumulate_activation_velocity(                                                <L 618>
        // worldid, actid = wp.tid()                                                              <L 626>
        builtin_tid2d(var_0, var_1);
        // act_dot_out[worldid, actid] += scale * act_dot_in[worldid, actid]                      <L 627>
        var_2 = wp::address(var_act_dot_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::mul(var_scale, var_4);
        var_5 = wp::atomic_add(var_act_dot_out, var_0, var_1, var_3);
    }
}



extern "C" __global__ void _qfrc_actuator_6de5c51c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_qfrc_actuator_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::range_t var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::array_t<wp::float32> var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def _qfrc_actuator(                                                                    <L 1933>
        // worldid, actid = wp.tid()                                                              <L 1943>
        builtin_tid2d(var_0, var_1);
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 1945>
        var_2 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 1946>
        var_5 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for i in range(rownnz):                                                                <L 1948>
        var_8 = wp::range(var_3);
        start_for_0:;
            if (iter_cmp(var_8) == 0) goto end_for_0;
            var_9 = wp::iter_next(var_8);
            // sparseid = rowadr + i                                                              <L 1949>
            var_10 = wp::add(var_6, var_9);
            // colind = moment_colind_in[worldid, sparseid]                                       <L 1950>
            var_11 = wp::address(var_moment_colind_in, var_0, var_10);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // qfrc = actuator_moment_in[worldid, sparseid] * actuator_force_in[worldid, actid]       <L 1951>
            var_14 = wp::address(var_actuator_moment_in, var_0, var_10);
            var_15 = wp::address(var_actuator_force_in, var_0, var_1);
            var_17 = wp::load(var_14);
            var_18 = wp::load(var_15);
            var_16 = wp::mul(var_17, var_18);
            // wp.atomic_add(qfrc_actuator_out[worldid], colind, qfrc)                            <L 1952>
            var_19 = wp::view(var_qfrc_actuator_out, var_0);
            var_20 = wp::atomic_add(var_19, var_12, var_16);
            goto start_for_0;
        end_for_0:;
    }
}



extern "C" __global__ void _rk_accumulate_velocity_acceleration_829bb7cc_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::float32 var_scale,
    wp::array_t<wp::float32> var_qvel_out,
    wp::array_t<wp::float32> var_qacc_out)
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
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        //---------
        // forward
        // def _rk_accumulate_velocity_acceleration(                                              <L 602>
        // worldid, dofid = wp.tid()                                                              <L 612>
        builtin_tid2d(var_0, var_1);
        // qvel_out[worldid, dofid] += scale * qvel_in[worldid, dofid]                            <L 613>
        var_2 = wp::address(var_qvel_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::mul(var_scale, var_4);
        var_5 = wp::atomic_add(var_qvel_out, var_0, var_1, var_3);
        // qacc_out[worldid, dofid] += scale * qacc_in[worldid, dofid]                            <L 614>
        var_6 = wp::address(var_qacc_in, var_0, var_1);
        var_8 = wp::load(var_6);
        var_7 = wp::mul(var_scale, var_8);
        var_9 = wp::atomic_add(var_qacc_out, var_0, var_1, var_7);
    }
}



extern "C" __global__ void _actuator_force_a8c1d316_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_na,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_gaintype,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::int32> var_actuator_ctrladr,
    wp::array_t<wp::int32> var_actuator_ctrlnum,
    wp::array_t<wp::int32> var_actuator_ctrlspec,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<bool> var_actuator_actearly,
    wp::array_t<bool> var_actuator_forcelimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_forcerange,
    wp::array_t<bool> var_actuator_ctrllimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_ctrlrange,
    wp::array_t<wp::vec_t<6, wp::float32>> var_actuator_gear,
    wp::array_t<wp::float32> var_actuator_acc0,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_lengthrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_ctrl_in,
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::int32 var_dsbl_clampctrl,
    wp::array_t<wp::float32> var_act_dot_out,
    wp::array_t<wp::float32> var_actuator_force_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::float32 var_13 = 0.0;
        const wp::int32 var_14 = 0;
        bool var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        bool var_19;
        bool* var_20;
        bool var_21;
        bool var_22;
        wp::vec_t<2, wp::float32>* var_23;
        wp::vec_t<2, wp::float32> var_24;
        wp::vec_t<2, wp::float32> var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        const wp::int32 var_28 = 1;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        bool var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::shape_t* var_49;
        const wp::int32 var_50 = 0;
        wp::int32 var_51;
        wp::shape_t var_52;
        wp::int32 var_53;
        wp::vec_t<10, wp::float32>* var_54;
        wp::vec_t<10, wp::float32> var_55;
        wp::vec_t<10, wp::float32> var_56;
        const wp::int32 var_57 = 1;
        bool var_58;
        wp::float32 var_59;
        bool var_60;
        const wp::int32 var_61 = 2;
        bool var_62;
        const wp::int32 var_63 = 3;
        bool var_64;
        wp::float32* var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 0;
        wp::float32 var_70;
        const wp::float32 var_71 = 1e-15;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::int32 var_74 = 4;
        bool var_75;
        wp::shape_t* var_76;
        const wp::int32 var_77 = 0;
        wp::int32 var_78;
        wp::shape_t var_79;
        wp::int32 var_80;
        wp::vec_t<10, wp::float32>* var_81;
        wp::vec_t<10, wp::float32> var_82;
        wp::vec_t<10, wp::float32> var_83;
        wp::float32* var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        const wp::int32 var_88 = 5;
        bool var_89;
        wp::shape_t* var_90;
        const wp::int32 var_91 = 0;
        wp::int32 var_92;
        wp::shape_t var_93;
        wp::int32 var_94;
        wp::vec_t<10, wp::float32>* var_95;
        wp::vec_t<10, wp::float32> var_96;
        wp::vec_t<10, wp::float32> var_97;
        wp::vec_t<6, wp::int32> var_98;
        wp::int32 var_99;
        const wp::float32 var_100 = 0.0;
        const wp::int32 var_101 = 0;
        wp::int32 var_102;
        const wp::int32 var_103 = 0;
        bool var_104;
        wp::float32* var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        const wp::int32 var_108 = 7;
        wp::float32 var_109;
        wp::shape_t* var_110;
        const wp::int32 var_111 = 0;
        wp::int32 var_112;
        wp::shape_t var_113;
        wp::int32 var_114;
        wp::float32* var_115;
        wp::float32 var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::shape_t* var_122;
        const wp::int32 var_123 = 0;
        wp::int32 var_124;
        wp::shape_t var_125;
        wp::int32 var_126;
        wp::float32* var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        const wp::int32 var_131 = 1;
        wp::int32 var_132;
        wp::float32 var_133;
        wp::int32 var_134;
        wp::float32 var_135;
        const wp::float32 var_136 = 0.0;
        const wp::int32 var_137 = 1;
        wp::int32 var_138;
        const wp::int32 var_139 = 0;
        bool var_140;
        wp::float32* var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        const wp::int32 var_144 = 8;
        wp::float32 var_145;
        wp::float32* var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        const wp::float32 var_149 = 0.0;
        bool var_150;
        bool var_151;
        const wp::float32 var_152 = 0.0;
        wp::float32 var_153;
        wp::float32 var_154;
        bool var_155;
        const wp::float32 var_156 = 0.0;
        wp::float32 var_157;
        wp::float32 var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        const wp::int32 var_161 = 1;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        const wp::float32 var_166 = 0.0;
        const wp::int32 var_167 = 0;
        bool var_168;
        wp::int32* var_169;
        wp::float32* var_170;
        wp::float32* var_171;
        wp::float32 var_172;
        wp::int32 var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        const wp::int32 var_177 = 0;
        wp::float32 var_178;
        const wp::int32 var_179 = 1;
        wp::float32 var_180;
        const wp::float32 var_181 = 1e-15;
        const wp::int32 var_182 = 0;
        wp::float32 var_183;
        wp::float32 var_184;
        const wp::int32 var_185 = 2;
        wp::int32 var_186;
        const wp::int32 var_187 = 0;
        bool var_188;
        const wp::int32 var_189 = 2;
        wp::float32 var_190;
        const wp::int32 var_191 = 3;
        wp::float32 var_192;
        const wp::int32 var_193 = 4;
        wp::float32 var_194;
        const wp::int32 var_195 = 2;
        wp::float32 var_196;
        const wp::int32 var_197 = 3;
        wp::float32 var_198;
        wp::float32* var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        const wp::float32 var_202 = 1.0;
        wp::float32 var_203;
        wp::float32 var_204;
        wp::float32 var_205;
        wp::float32 var_206;
        wp::float32 var_207;
        wp::float32* var_208;
        wp::float32 var_209;
        wp::float32 var_210;
        wp::float32 var_211;
        wp::float32 var_212;
        const wp::int32 var_213 = 4;
        wp::int32 var_214;
        const wp::int32 var_215 = 0;
        bool var_216;
        wp::float32* var_217;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        wp::float32 var_224;
        wp::float32 var_225;
        const wp::int32 var_226 = 1;
        wp::int32 var_227;
        wp::float32 var_228;
        wp::int32 var_229;
        wp::float32 var_230;
        wp::float32 var_231;
        const wp::int32 var_232 = 3;
        wp::int32 var_233;
        const wp::int32 var_234 = 0;
        bool var_235;
        const wp::int32 var_236 = 5;
        wp::float32 var_237;
        wp::shape_t* var_238;
        const wp::int32 var_239 = 0;
        wp::int32 var_240;
        wp::shape_t var_241;
        wp::int32 var_242;
        wp::vec_t<10, wp::float32>* var_243;
        wp::vec_t<10, wp::float32> var_244;
        wp::vec_t<10, wp::float32> var_245;
        const wp::int32 var_246 = 3;
        wp::float32 var_247;
        const wp::int32 var_248 = 4;
        wp::float32 var_249;
        const wp::int32 var_250 = 5;
        wp::float32 var_251;
        wp::float32* var_252;
        wp::float32 var_253;
        wp::float32 var_254;
        wp::float32* var_255;
        wp::float32 var_256;
        wp::float32 var_257;
        wp::float32 var_258;
        wp::float32* var_259;
        wp::float32 var_260;
        wp::float32 var_261;
        wp::float32 var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        wp::float32 var_265;
        wp::float32* var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        const wp::int32 var_269 = 1;
        wp::int32 var_270;
        wp::int32 var_271;
        wp::float32 var_272;
        const wp::int32 var_273 = 4;
        wp::int32 var_274;
        const wp::int32 var_275 = 0;
        bool var_276;
        const wp::int32 var_277 = 1;
        wp::float32 var_278;
        wp::float32 var_279;
        wp::float32 var_280;
        wp::float32* var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        wp::float32 var_284;
        wp::float32* var_285;
        wp::float32 var_286;
        wp::float32 var_287;
        wp::float32 var_288;
        const wp::float32 var_289 = 0.0;
        bool var_290;
        wp::float32 var_291;
        wp::float32 var_292;
        wp::float32 var_293;
        wp::float32 var_294;
        const wp::int32 var_295 = 7;
        bool var_296;
        const wp::float32 var_297 = 0.0;
        const wp::float32 var_298 = 0.0;
        wp::float32 var_299;
        wp::float32 var_300;
        wp::float32 var_301;
        wp::float32 var_302;
        wp::vec_t<10, wp::float32> var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::vec_t<10, wp::float32> var_306;
        wp::float32 var_307;
        wp::float32 var_308;
        wp::float32 var_309;
        wp::float32 var_310;
        wp::vec_t<10, wp::float32> var_311;
        wp::float32 var_312;
        bool* var_313;
        bool var_314;
        bool var_315;
        const wp::int32 var_316 = 1;
        bool var_317;
        const wp::int32 var_318 = 0;
        bool var_319;
        const wp::int32 var_320 = 5;
        bool var_321;
        wp::float32* var_322;
        wp::float32 var_323;
        wp::float32 var_324;
        wp::float32 var_325;
        const wp::int32 var_326 = 5;
        bool var_327;
        wp::shape_t* var_328;
        const wp::int32 var_329 = 0;
        wp::int32 var_330;
        wp::shape_t var_331;
        wp::int32 var_332;
        wp::vec_t<10, wp::float32>* var_333;
        wp::vec_t<10, wp::float32> var_334;
        wp::vec_t<10, wp::float32> var_335;
        wp::vec_t<6, wp::int32> var_336;
        wp::int32* var_337;
        const wp::int32 var_338 = 1;
        wp::int32 var_339;
        wp::int32 var_340;
        const wp::int32 var_341 = 4;
        wp::int32 var_342;
        bool var_343;
        const wp::int32 var_344 = 0;
        wp::float32 var_345;
        wp::float32 var_346;
        wp::float32 var_347;
        const wp::float32 var_348 = 1.0;
        wp::shape_t* var_349;
        const wp::int32 var_350 = 0;
        wp::int32 var_351;
        wp::shape_t var_352;
        wp::int32 var_353;
        wp::float32* var_354;
        wp::float32 var_355;
        wp::float32 var_356;
        wp::float32 var_357;
        wp::float32 var_358;
        wp::float32 var_359;
        wp::float32 var_360;
        wp::float32 var_361;
        const wp::int32 var_362 = 3;
        wp::int32 var_363;
        bool var_364;
        const wp::int32 var_365 = 5;
        wp::float32 var_366;
        wp::shape_t* var_367;
        const wp::int32 var_368 = 0;
        wp::int32 var_369;
        wp::shape_t var_370;
        wp::int32 var_371;
        wp::vec_t<10, wp::float32>* var_372;
        wp::vec_t<10, wp::float32> var_373;
        wp::vec_t<10, wp::float32> var_374;
        const wp::int32 var_375 = 3;
        wp::float32 var_376;
        const wp::int32 var_377 = 4;
        wp::float32 var_378;
        const wp::int32 var_379 = 5;
        wp::float32 var_380;
        wp::float32* var_381;
        wp::float32 var_382;
        wp::float32 var_383;
        wp::float32 var_384;
        wp::float32 var_385;
        wp::float32 var_386;
        wp::float32 var_387;
        wp::float32 var_388;
        wp::float32 var_389;
        wp::shape_t* var_390;
        const wp::int32 var_391 = 0;
        wp::int32 var_392;
        wp::shape_t var_393;
        wp::int32 var_394;
        wp::float32* var_395;
        wp::float32 var_396;
        wp::float32 var_397;
        wp::float32 var_398;
        wp::float32 var_399;
        wp::float32 var_400;
        wp::float32 var_401;
        bool var_402;
        const wp::float32 var_403 = 1.0;
        wp::float32 var_404;
        wp::float32 var_405;
        wp::float32 var_406;
        wp::float32 var_407;
        wp::float32 var_408;
        wp::float32 var_409;
        const wp::int32 var_410 = 1;
        wp::int32 var_411;
        bool var_412;
        wp::shape_t* var_413;
        const wp::int32 var_414 = 0;
        wp::int32 var_415;
        wp::shape_t var_416;
        wp::int32 var_417;
        wp::float32* var_418;
        wp::float32 var_419;
        wp::float32 var_420;
        wp::float32 var_421;
        const wp::int32 var_422 = 8;
        wp::float32 var_423;
        const wp::float32 var_424 = 0.0;
        bool var_425;
        wp::float32 var_426;
        wp::float32 var_427;
        wp::float32 var_428;
        wp::shape_t* var_429;
        const wp::int32 var_430 = 0;
        wp::int32 var_431;
        wp::shape_t var_432;
        wp::int32 var_433;
        wp::float32* var_434;
        wp::float32 var_435;
        wp::float32 var_436;
        wp::float32 var_437;
        wp::float32 var_438;
        wp::float32 var_439;
        wp::float32 var_440;
        wp::float32 var_441;
        wp::float32 var_442;
        wp::vec_t<10, wp::float32> var_443;
        wp::float32 var_444;
        wp::float32 var_445;
        wp::float32 var_446;
        wp::float32 var_447;
        wp::float32 var_448;
        wp::float32 var_449;
        wp::float32 var_450;
        wp::float32 var_451;
        wp::float32 var_452;
        wp::vec_t<10, wp::float32> var_453;
        wp::float32 var_454;
        wp::float32 var_455;
        wp::float32 var_456;
        wp::float32 var_457;
        wp::float32 var_458;
        bool* var_459;
        bool var_460;
        wp::shape_t* var_461;
        const wp::int32 var_462 = 0;
        wp::int32 var_463;
        wp::shape_t var_464;
        wp::int32 var_465;
        wp::vec_t<2, wp::float32>* var_466;
        wp::vec_t<2, wp::float32> var_467;
        wp::vec_t<2, wp::float32> var_468;
        const wp::int32 var_469 = 0;
        wp::float32 var_470;
        const wp::int32 var_471 = 1;
        wp::float32 var_472;
        wp::float32 var_473;
        bool var_474;
        wp::float32 var_475;
        bool var_476;
        wp::shape_t* var_477;
        const wp::int32 var_478 = 0;
        wp::int32 var_479;
        wp::shape_t var_480;
        wp::int32 var_481;
        wp::float32* var_482;
        wp::shape_t* var_483;
        const wp::int32 var_484 = 0;
        wp::int32 var_485;
        wp::shape_t var_486;
        wp::int32 var_487;
        wp::vec_t<2, wp::float32>* var_488;
        const wp::float32 var_489 = 1.0;
        bool* var_490;
        wp::float32 var_491;
        wp::float32 var_492;
        wp::vec_t<2, wp::float32> var_493;
        bool var_494;
        wp::float32 var_495;
        wp::vec_t<10, wp::float32> var_496;
        wp::vec_t<6, wp::int32> var_497;
        wp::float32 var_498;
        wp::float32 var_499;
        wp::float32 var_500;
        wp::vec_t<10, wp::float32> var_501;
        wp::float32 var_502;
        wp::float32 var_503;
        wp::float32 var_504;
        wp::float32 var_505;
        wp::float32 var_506;
        bool var_507;
        bool var_508;
        wp::float32* var_509;
        wp::float32 var_510;
        wp::float32 var_511;
        bool var_512;
        wp::float32 var_513;
        bool var_514;
        wp::float32 var_515;
        bool var_516;
        wp::vec_t<10, wp::float32> var_517;
        bool var_518;
        wp::vec_t<6, wp::int32> var_519;
        bool var_520;
        wp::float32 var_521;
        bool var_522;
        wp::float32 var_523;
        bool var_524;
        wp::float32 var_525;
        bool var_526;
        wp::vec_t<10, wp::float32> var_527;
        bool var_528;
        wp::float32 var_529;
        bool var_530;
        wp::float32 var_531;
        bool var_532;
        wp::float32 var_533;
        bool var_534;
        wp::float32 var_535;
        bool var_536;
        wp::float32 var_537;
        bool var_538;
        wp::float32 var_539;
        wp::float32 var_540;
        wp::float32* var_541;
        wp::float32 var_542;
        wp::float32 var_543;
        wp::float32* var_544;
        wp::float32 var_545;
        wp::float32 var_546;
        wp::int32* var_547;
        wp::int32 var_548;
        wp::int32 var_549;
        wp::shape_t* var_550;
        const wp::int32 var_551 = 0;
        wp::int32 var_552;
        wp::shape_t var_553;
        wp::int32 var_554;
        wp::vec_t<10, wp::float32>* var_555;
        wp::vec_t<10, wp::float32> var_556;
        wp::vec_t<10, wp::float32> var_557;
        wp::shape_t* var_558;
        const wp::int32 var_559 = 0;
        wp::int32 var_560;
        wp::shape_t var_561;
        wp::int32 var_562;
        wp::vec_t<10, wp::float32>* var_563;
        wp::vec_t<10, wp::float32> var_564;
        wp::vec_t<10, wp::float32> var_565;
        const wp::float32 var_566 = 0.0;
        const wp::int32 var_567 = 0;
        bool var_568;
        const wp::int32 var_569 = 0;
        wp::float32 var_570;
        const wp::int32 var_571 = 1;
        bool var_572;
        const wp::int32 var_573 = 0;
        wp::float32 var_574;
        const wp::int32 var_575 = 1;
        wp::float32 var_576;
        wp::float32 var_577;
        wp::float32 var_578;
        const wp::int32 var_579 = 2;
        wp::float32 var_580;
        wp::float32 var_581;
        wp::float32 var_582;
        const wp::int32 var_583 = 2;
        bool var_584;
        wp::shape_t* var_585;
        const wp::int32 var_586 = 0;
        wp::int32 var_587;
        wp::shape_t var_588;
        wp::int32 var_589;
        wp::float32* var_590;
        wp::float32 var_591;
        wp::float32 var_592;
        wp::shape_t* var_593;
        const wp::int32 var_594 = 0;
        wp::int32 var_595;
        wp::shape_t var_596;
        wp::int32 var_597;
        wp::vec_t<2, wp::float32>* var_598;
        wp::vec_t<2, wp::float32> var_599;
        wp::vec_t<2, wp::float32> var_600;
        wp::float32 var_601;
        const wp::int32 var_602 = 3;
        bool var_603;
        const wp::int32 var_604 = 1;
        wp::float32 var_605;
        const wp::int32 var_606 = 0;
        wp::float32 var_607;
        wp::float32 var_608;
        const wp::float32 var_609 = 0.0;
        bool var_610;
        wp::float32 var_611;
        wp::float32 var_612;
        const wp::float32 var_613 = 0.0;
        bool var_614;
        const wp::int32 var_615 = 0;
        bool var_616;
        const wp::float32 var_617 = 0.0;
        wp::int32* var_618;
        const wp::int32 var_619 = 7;
        wp::int32 var_620;
        wp::int32 var_621;
        const wp::int32 var_622 = 0;
        bool var_623;
        const wp::float32 var_624 = 0.0;
        bool var_625;
        const wp::int32 var_626 = 0;
        bool var_627;
        wp::vec_t<6, wp::int32> var_628;
        const wp::int32 var_629 = 1;
        wp::int32 var_630;
        const wp::int32 var_631 = 0;
        bool var_632;
        const wp::int32 var_633 = 1;
        wp::int32 var_634;
        wp::int32 var_635;
        wp::float32* var_636;
        wp::float32 var_637;
        wp::float32 var_638;
        wp::float32 var_639;
        wp::vec_t<6, wp::int32> var_640;
        wp::float32 var_641;
        wp::int32* var_642;
        wp::float32 var_643;
        wp::int32 var_644;
        wp::float32 var_645;
        wp::float32 var_646;
        wp::vec_t<6, wp::int32> var_647;
        wp::float32 var_648;
        wp::float32 var_649;
        wp::vec_t<6, wp::int32> var_650;
        wp::float32 var_651;
        wp::float32 var_652;
        wp::vec_t<6, wp::int32> var_653;
        wp::float32 var_654;
        wp::float32 var_655;
        wp::vec_t<6, wp::int32> var_656;
        wp::float32 var_657;
        wp::float32 var_658;
        wp::float32 var_659;
        wp::float32 var_660;
        wp::float32 var_661;
        wp::float32 var_662;
        wp::vec_t<6, wp::int32> var_663;
        wp::float32 var_664;
        wp::float32 var_665;
        wp::float32 var_666;
        wp::float32 var_667;
        wp::float32 var_668;
        wp::float32 var_669;
        wp::vec_t<6, wp::int32> var_670;
        wp::float32 var_671;
        wp::float32 var_672;
        wp::float32 var_673;
        wp::float32 var_674;
        wp::float32 var_675;
        wp::float32 var_676;
        wp::vec_t<6, wp::int32> var_677;
        wp::float32 var_678;
        wp::float32 var_679;
        wp::float32 var_680;
        wp::float32 var_681;
        wp::float32 var_682;
        wp::int32* var_683;
        wp::int32 var_684;
        wp::int32 var_685;
        wp::shape_t* var_686;
        const wp::int32 var_687 = 0;
        wp::int32 var_688;
        wp::shape_t var_689;
        wp::int32 var_690;
        wp::vec_t<10, wp::float32>* var_691;
        wp::vec_t<10, wp::float32> var_692;
        wp::vec_t<10, wp::float32> var_693;
        const wp::float32 var_694 = 0.0;
        const wp::int32 var_695 = 1;
        bool var_696;
        const wp::int32 var_697 = 0;
        wp::float32 var_698;
        const wp::int32 var_699 = 1;
        wp::float32 var_700;
        wp::float32 var_701;
        wp::float32 var_702;
        const wp::int32 var_703 = 2;
        wp::float32 var_704;
        wp::float32 var_705;
        wp::float32 var_706;
        const wp::int32 var_707 = 2;
        bool var_708;
        wp::shape_t* var_709;
        const wp::int32 var_710 = 0;
        wp::int32 var_711;
        wp::shape_t var_712;
        wp::int32 var_713;
        wp::float32* var_714;
        wp::float32 var_715;
        wp::float32 var_716;
        wp::shape_t* var_717;
        const wp::int32 var_718 = 0;
        wp::int32 var_719;
        wp::shape_t var_720;
        wp::int32 var_721;
        wp::vec_t<2, wp::float32>* var_722;
        wp::vec_t<2, wp::float32> var_723;
        wp::vec_t<2, wp::float32> var_724;
        wp::float32 var_725;
        const wp::int32 var_726 = 3;
        bool var_727;
        const wp::int32 var_728 = 0;
        wp::float32 var_729;
        const wp::float32 var_730 = 0.0;
        bool var_731;
        const wp::int32 var_732 = 1;
        wp::float32 var_733;
        wp::float32 var_734;
        wp::float32 var_735;
        wp::float32 var_736;
        wp::float32 var_737;
        wp::float32 var_738;
        wp::float32 var_739;
        wp::float32 var_740;
        wp::float32 var_741;
        wp::float32 var_742;
        wp::vec_t<2, wp::float32> var_743;
        wp::float32 var_744;
        wp::float32 var_745;
        wp::float32 var_746;
        wp::vec_t<2, wp::float32> var_747;
        wp::float32 var_748;
        wp::int32* var_749;
        wp::int32 var_750;
        wp::int32 var_751;
        wp::vec_t<2, wp::int32>* var_752;
        wp::vec_t<2, wp::int32> var_753;
        wp::vec_t<2, wp::int32> var_754;
        wp::shape_t* var_755;
        const wp::int32 var_756 = 0;
        wp::int32 var_757;
        wp::shape_t var_758;
        wp::int32 var_759;
        wp::vec_t<6, wp::float32>* var_760;
        wp::vec_t<6, wp::float32> var_761;
        wp::vec_t<6, wp::float32> var_762;
        wp::float32 var_763;
        const wp::float32 var_764 = 0.0;
        bool var_765;
        wp::float32 var_766;
        wp::float32 var_767;
        wp::float32 var_768;
        wp::float32 var_769;
        wp::float32 var_770;
        wp::float32 var_771;
        wp::float32 var_772;
        wp::float32 var_773;
        bool* var_774;
        bool var_775;
        wp::shape_t* var_776;
        const wp::int32 var_777 = 0;
        wp::int32 var_778;
        wp::shape_t var_779;
        wp::int32 var_780;
        wp::vec_t<2, wp::float32>* var_781;
        wp::vec_t<2, wp::float32> var_782;
        wp::vec_t<2, wp::float32> var_783;
        const wp::int32 var_784 = 0;
        wp::float32 var_785;
        const wp::int32 var_786 = 1;
        wp::float32 var_787;
        wp::float32 var_788;
        bool var_789;
        wp::float32 var_790;
        bool var_791;
        const wp::int32 var_792 = 3;
        bool var_793;
        const wp::int32 var_794 = 0;
        wp::float32 var_795;
        const wp::float32 var_796 = 0.0;
        bool var_797;
        const wp::int32 var_798 = 1;
        wp::float32 var_799;
        const wp::int32 var_800 = 2;
        wp::float32 var_801;
        wp::float32 var_802;
        wp::float32 var_803;
        wp::float32 var_804;
        wp::float32 var_805;
        wp::float32 var_806;
        wp::float32 var_807;
        const wp::int32 var_808 = 5;
        wp::float32 var_809;
        const wp::float32 var_810 = 0.0;
        bool var_811;
        const wp::int32 var_812 = 6;
        wp::float32 var_813;
        wp::vec_t<6, wp::int32> var_814;
        const wp::int32 var_815 = 3;
        wp::int32 var_816;
        wp::int32 var_817;
        wp::float32* var_818;
        wp::float32 var_819;
        wp::float32 var_820;
        wp::float32* var_821;
        wp::float32 var_822;
        wp::float32 var_823;
        wp::float32 var_824;
        wp::float32 var_825;
        wp::float32 var_826;
        wp::float32 var_827;
        wp::vec_t<6, wp::int32> var_828;
        wp::int32 var_829;
        wp::float32 var_830;
        wp::float32 var_831;
        wp::vec_t<6, wp::int32> var_832;
        wp::int32 var_833;
        wp::float32 var_834;
        wp::float32 var_835;
        wp::float32 var_836;
        //---------
        // forward
        // def _actuator_force(                                                                   <L 1556>
        // worldid, uid = wp.tid()                                                                <L 1595>
        builtin_tid2d(var_0, var_1);
        // actuator_ctrlrange_id = worldid % actuator_ctrlrange.shape[0]                          <L 1597>
        var_2 = &(var_actuator_ctrlrange.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // uadr = actuator_ctrladr[uid]                                                           <L 1599>
        var_7 = wp::address(var_actuator_ctrladr, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // ctrlnum = actuator_ctrlnum[uid]                                                        <L 1600>
        var_10 = wp::address(var_actuator_ctrlnum, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // ctrl = 0.0                                                                             <L 1601>
        // if ctrlnum > 0:                                                                        <L 1602>
        var_15 = (var_11 > var_14);
        if (var_15) {
            // ctrl = ctrl_in[worldid, uadr]                                                      <L 1603>
            var_16 = wp::address(var_ctrl_in, var_0, var_8);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // if actuator_ctrllimited[uadr] and not dsbl_clampctrl:                              <L 1604>
            var_20 = wp::address(var_actuator_ctrllimited, var_8);
            var_21 = wp::load(var_20);
            var_19 = var_21;
            if (var_19) {
                var_22 = wp::unot(var_dsbl_clampctrl);
                var_19 = var_19 && var_22;
            }
            if (var_19) {
                // ctrlrange = actuator_ctrlrange[actuator_ctrlrange_id, uadr]                    <L 1605>
                var_23 = wp::address(var_actuator_ctrlrange, var_6, var_8);
                var_25 = wp::load(var_23);
                var_24 = wp::copy(var_25);
                // ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])                              <L 1606>
                var_27 = wp::extract(var_24, var_26);
                var_29 = wp::extract(var_24, var_28);
                var_30 = wp::clamp(var_17, var_27, var_29);
            }
            var_31 = wp::where(var_19, var_30, var_17);
        }
        var_32 = wp::where(var_15, var_31, var_13);
        // ctrl_act = ctrl                                                                        <L 1607>
        var_33 = wp::copy(var_32);
        // u_first = ctrl                                                                         <L 1608>
        var_34 = wp::copy(var_32);
        // dyntype = actuator_dyntype[uid]                                                        <L 1610>
        var_35 = wp::address(var_actuator_dyntype, var_1);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // act_first = actuator_actadr[uid]                                                       <L 1611>
        var_38 = wp::address(var_actuator_actadr, var_1);
        var_40 = wp::load(var_38);
        var_39 = wp::copy(var_40);
        // if na and act_first >= 0:                                                              <L 1612>
        var_41 = var_na;
        if (var_41) {
            var_43 = (var_39 >= var_42);
            var_41 = var_41 && var_43;
        }
        if (var_41) {
            // act_last = act_first + actuator_actnum[uid] - 1                                    <L 1613>
            var_44 = wp::address(var_actuator_actnum, var_1);
            var_46 = wp::load(var_44);
            var_45 = wp::add(var_39, var_46);
            var_48 = wp::sub(var_45, var_47);
            // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]                  <L 1614>
            var_49 = &(var_actuator_dynprm.shape);
            var_52 = wp::load(var_49);
            var_51 = wp::extract(var_52, var_50);
            var_53 = wp::mod(var_0, var_51);
            var_54 = wp::address(var_actuator_dynprm, var_53, var_1);
            var_56 = wp::load(var_54);
            var_55 = wp::copy(var_56);
            // if dyntype == DynType.INTEGRATOR:                                                  <L 1616>
            var_58 = (var_36 == var_57);
            if (var_58) {
                // act_dot = ctrl                                                                 <L 1617>
                var_59 = wp::copy(var_32);
            }
            if (!var_58) {
                // elif dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT:              <L 1618>
                var_62 = (var_36 == var_61);
                var_60 = var_62;
                if (!var_60) {
                    var_64 = (var_36 == var_63);
                    var_60 = var_60 || var_64;
                }
                if (var_60) {
                    // act = act_in[worldid, act_last]                                            <L 1619>
                    var_65 = wp::address(var_act_in, var_0, var_48);
                    var_67 = wp::load(var_65);
                    var_66 = wp::copy(var_67);
                    // act_dot = (ctrl - act) / wp.max(dynprm[0], MJ_MINVAL)                      <L 1620>
                    var_68 = wp::sub(var_32, var_66);
                    var_70 = wp::extract(var_55, var_69);
                    var_72 = wp::max(var_70, var_71);
                    var_73 = wp::div(var_68, var_72);
                }
                if (!var_60) {
                    // elif dyntype == DynType.MUSCLE:                                            <L 1621>
                    var_75 = (var_36 == var_74);
                    if (var_75) {
                        // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]       <L 1622>
                        var_76 = &(var_actuator_dynprm.shape);
                        var_79 = wp::load(var_76);
                        var_78 = wp::extract(var_79, var_77);
                        var_80 = wp::mod(var_0, var_78);
                        var_81 = wp::address(var_actuator_dynprm, var_80, var_1);
                        var_83 = wp::load(var_81);
                        var_82 = wp::copy(var_83);
                        // act = act_in[worldid, act_last]                                        <L 1623>
                        var_84 = wp::address(var_act_in, var_0, var_48);
                        var_86 = wp::load(var_84);
                        var_85 = wp::copy(var_86);
                        // act_dot = util_misc.muscle_dynamics(ctrl, act, dynprm)                 <L 1624>
                        var_87 = muscle_dynamics_0(var_32, var_85, var_82);
                    }
                    if (!var_75) {
                        // elif dyntype == DynType.DCMOTOR:                                       <L 1625>
                        var_89 = (var_36 == var_88);
                        if (var_89) {
                            // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]       <L 1626>
                            var_90 = &(var_actuator_gainprm.shape);
                            var_93 = wp::load(var_90);
                            var_92 = wp::extract(var_93, var_91);
                            var_94 = wp::mod(var_0, var_92);
                            var_95 = wp::address(var_actuator_gainprm, var_94, var_1);
                            var_97 = wp::load(var_95);
                            var_96 = wp::copy(var_97);
                            // slots = util_misc.dcmotor_slots(dynprm, gainprm)                   <L 1627>
                            var_98 = dcmotor_slots_0(var_55, var_96);
                            // adr = act_first                                                    <L 1628>
                            var_99 = wp::copy(var_39);
                            // act_dot = 0.0                                                      <L 1630>
                            // if slots[0] >= 0:                                                  <L 1633>
                            var_102 = wp::extract(var_98, var_101);
                            var_104 = (var_102 >= var_103);
                            if (var_104) {
                                // u_prev = act_in[worldid, adr]                                  <L 1634>
                                var_105 = wp::address(var_act_in, var_0, var_99);
                                var_107 = wp::load(var_105);
                                var_106 = wp::copy(var_107);
                                // slew_s = dynprm[7]                                             <L 1635>
                                var_109 = wp::extract(var_55, var_108);
                                // slew = slew_s * opt_timestep[worldid % opt_timestep.shape[0]]       <L 1636>
                                var_110 = &(var_opt_timestep.shape);
                                var_113 = wp::load(var_110);
                                var_112 = wp::extract(var_113, var_111);
                                var_114 = wp::mod(var_0, var_112);
                                var_115 = wp::address(var_opt_timestep, var_114);
                                var_117 = wp::load(var_115);
                                var_116 = wp::mul(var_109, var_117);
                                // u_eff = wp.clamp(u_first, u_prev - slew, u_prev + slew)        <L 1637>
                                var_118 = wp::sub(var_106, var_116);
                                var_119 = wp::add(var_106, var_116);
                                var_120 = wp::clamp(var_34, var_118, var_119);
                                // act_dot = (u_eff - u_prev) / opt_timestep[worldid % opt_timestep.shape[0]]       <L 1638>
                                var_121 = wp::sub(var_120, var_106);
                                var_122 = &(var_opt_timestep.shape);
                                var_125 = wp::load(var_122);
                                var_124 = wp::extract(var_125, var_123);
                                var_126 = wp::mod(var_0, var_124);
                                var_127 = wp::address(var_opt_timestep, var_126);
                                var_129 = wp::load(var_127);
                                var_128 = wp::div(var_121, var_129);
                                // act_dot_out[worldid, adr] = act_dot                            <L 1639>
                                wp::array_store(var_act_dot_out, var_0, var_99, var_128);
                                // u_first = u_eff                                                <L 1640>
                                var_130 = wp::copy(var_120);
                                // adr += 1                                                       <L 1641>
                                var_132 = wp::add(var_99, var_131);
                            }
                            var_133 = wp::where(var_104, var_130, var_34);
                            var_134 = wp::where(var_104, var_132, var_99);
                            var_135 = wp::where(var_104, var_128, var_100);
                            // x_I = 0.0                                                          <L 1644>
                            // if slots[1] >= 0:                                                  <L 1645>
                            var_138 = wp::extract(var_98, var_137);
                            var_140 = (var_138 >= var_139);
                            if (var_140) {
                                // x_I = act_in[worldid, adr]                                     <L 1646>
                                var_141 = wp::address(var_act_in, var_0, var_134);
                                var_143 = wp::load(var_141);
                                var_142 = wp::copy(var_143);
                                // Imax = dynprm[8]                                               <L 1647>
                                var_145 = wp::extract(var_55, var_144);
                                // act_dot = u_first - actuator_length_in[worldid, uid]           <L 1648>
                                var_146 = wp::address(var_actuator_length_in, var_0, var_1);
                                var_148 = wp::load(var_146);
                                var_147 = wp::sub(var_133, var_148);
                                // if Imax > 0.0:                                                 <L 1650>
                                var_150 = (var_145 > var_149);
                                if (var_150) {
                                    // if x_I >= Imax:                                            <L 1651>
                                    var_151 = (var_142 >= var_145);
                                    if (var_151) {
                                        // act_dot = wp.min(act_dot, 0.0)                         <L 1652>
                                        var_153 = wp::min(var_147, var_152);
                                    }
                                    if (!var_151) {
                                        // elif x_I <= -Imax:                                     <L 1653>
                                        var_154 = wp::neg(var_145);
                                        var_155 = (var_142 <= var_154);
                                        if (var_155) {
                                            // act_dot = wp.max(act_dot, 0.0)                     <L 1654>
                                            var_157 = wp::max(var_147, var_156);
                                        }
                                        var_158 = wp::where(var_155, var_157, var_147);
                                    }
                                    var_159 = wp::where(var_151, var_153, var_158);
                                }
                                var_160 = wp::where(var_150, var_159, var_147);
                                // act_dot_out[worldid, adr] = act_dot                            <L 1656>
                                wp::array_store(var_act_dot_out, var_0, var_134, var_160);
                                // adr += 1                                                       <L 1657>
                                var_162 = wp::add(var_134, var_161);
                            }
                            var_163 = wp::where(var_140, var_162, var_134);
                            var_164 = wp::where(var_140, var_160, var_135);
                            var_165 = wp::where(var_140, var_142, var_136);
                            // V = 0.0                                                            <L 1660>
                            // if ctrlnum > 0:                                                    <L 1661>
                            var_168 = (var_11 > var_167);
                            if (var_168) {
                                // V = util_misc.dcmotor_voltage(                                 <L 1662>
                                // ctrl_in,                                                       <L 1663>
                                // worldid,                                                       <L 1664>
                                // uadr,                                                          <L 1665>
                                // actuator_ctrlspec[uid],                                        <L 1666>
                                var_169 = wp::address(var_actuator_ctrlspec, var_1);
                                // u_first,                                                       <L 1667>
                                // actuator_length_in[worldid, uid],                              <L 1668>
                                var_170 = wp::address(var_actuator_length_in, var_0, var_1);
                                // actuator_velocity_in[worldid, uid],                            <L 1669>
                                var_171 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                // x_I,                                                           <L 1670>
                                // gainprm,                                                       <L 1671>
                                var_173 = wp::load(var_169);
                                var_174 = wp::load(var_170);
                                var_175 = wp::load(var_171);
                                var_172 = dcmotor_voltage_0(var_ctrl_in, var_0, var_8, var_173, var_133, var_174, var_175, var_165, var_96);
                            }
                            var_176 = wp::where(var_168, var_172, var_166);
                            // R = gainprm[0]                                                     <L 1675>
                            var_178 = wp::extract(var_96, var_177);
                            // K = gainprm[1]                                                     <L 1676>
                            var_180 = wp::extract(var_96, var_179);
                            // te = wp.max(MJ_MINVAL, dynprm[0])                                  <L 1677>
                            var_183 = wp::extract(var_55, var_182);
                            var_184 = wp::max(var_181, var_183);
                            // if slots[2] >= 0:                                                  <L 1679>
                            var_186 = wp::extract(var_98, var_185);
                            var_188 = (var_186 >= var_187);
                            if (var_188) {
                                // RT = dynprm[2]                                                 <L 1680>
                                var_190 = wp::extract(var_55, var_189);
                                // C = dynprm[3]                                                  <L 1681>
                                var_192 = wp::extract(var_55, var_191);
                                // Ta = dynprm[4]                                                 <L 1682>
                                var_194 = wp::extract(var_55, var_193);
                                // alpha = gainprm[2]                                             <L 1683>
                                var_196 = wp::extract(var_96, var_195);
                                // T0 = gainprm[3]                                                <L 1684>
                                var_198 = wp::extract(var_96, var_197);
                                // T = act_in[worldid, adr]                                       <L 1685>
                                var_199 = wp::address(var_act_in, var_0, var_163);
                                var_201 = wp::load(var_199);
                                var_200 = wp::copy(var_201);
                                // R_eff = R * (1.0 + alpha * (T + Ta - T0))                      <L 1686>
                                var_203 = wp::add(var_200, var_194);
                                var_204 = wp::sub(var_203, var_198);
                                var_205 = wp::mul(var_196, var_204);
                                var_206 = wp::add(var_202, var_205);
                                var_207 = wp::mul(var_178, var_206);
                                // current = (V - K * actuator_velocity_in[worldid, uid]) / R_eff       <L 1688>
                                var_208 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_210 = wp::load(var_208);
                                var_209 = wp::mul(var_180, var_210);
                                var_211 = wp::sub(var_176, var_209);
                                var_212 = wp::div(var_211, var_207);
                                // if slots[4] >= 0:                                              <L 1689>
                                var_214 = wp::extract(var_98, var_213);
                                var_216 = (var_214 >= var_215);
                                if (var_216) {
                                    // current = act_in[worldid, act_last]                        <L 1690>
                                    var_217 = wp::address(var_act_in, var_0, var_48);
                                    var_219 = wp::load(var_217);
                                    var_218 = wp::copy(var_219);
                                }
                                var_220 = wp::where(var_216, var_218, var_212);
                                // act_dot = (R_eff * current * current - T / RT) / C             <L 1692>
                                var_221 = wp::mul(var_207, var_220);
                                var_222 = wp::mul(var_221, var_220);
                                var_223 = wp::div(var_200, var_190);
                                var_224 = wp::sub(var_222, var_223);
                                var_225 = wp::div(var_224, var_192);
                                // act_dot_out[worldid, adr] = act_dot                            <L 1693>
                                wp::array_store(var_act_dot_out, var_0, var_163, var_225);
                                // adr += 1                                                       <L 1694>
                                var_227 = wp::add(var_163, var_226);
                                // R = R_eff                                                      <L 1695>
                                var_228 = wp::copy(var_207);
                            }
                            var_229 = wp::where(var_188, var_227, var_163);
                            var_230 = wp::where(var_188, var_225, var_164);
                            var_231 = wp::where(var_188, var_228, var_178);
                            // if slots[3] >= 0:                                                  <L 1698>
                            var_233 = wp::extract(var_98, var_232);
                            var_235 = (var_233 >= var_234);
                            if (var_235) {
                                // sigma0 = dynprm[5]                                             <L 1699>
                                var_237 = wp::extract(var_55, var_236);
                                // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]       <L 1700>
                                var_238 = &(var_actuator_biasprm.shape);
                                var_241 = wp::load(var_238);
                                var_240 = wp::extract(var_241, var_239);
                                var_242 = wp::mod(var_0, var_240);
                                var_243 = wp::address(var_actuator_biasprm, var_242, var_1);
                                var_245 = wp::load(var_243);
                                var_244 = wp::copy(var_245);
                                // F_C = biasprm[3]                                               <L 1701>
                                var_247 = wp::extract(var_244, var_246);
                                // F_S = biasprm[4]                                               <L 1702>
                                var_249 = wp::extract(var_244, var_248);
                                // v_S = biasprm[5]                                               <L 1703>
                                var_251 = wp::extract(var_244, var_250);
                                // z = act_in[worldid, adr]                                       <L 1704>
                                var_252 = wp::address(var_act_in, var_0, var_229);
                                var_254 = wp::load(var_252);
                                var_253 = wp::copy(var_254);
                                // g = util_misc.lugre_stribeck(actuator_velocity_in[worldid, uid], F_C, F_S, v_S)       <L 1705>
                                var_255 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_257 = wp::load(var_255);
                                var_256 = lugre_stribeck_0(var_257, var_247, var_249, var_251);
                                // a = -sigma0 * wp.abs(actuator_velocity_in[worldid, uid]) / wp.max(MJ_MINVAL, g)       <L 1706>
                                var_258 = wp::neg(var_237);
                                var_259 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_261 = wp::load(var_259);
                                var_260 = wp::abs(var_261);
                                var_262 = wp::mul(var_258, var_260);
                                var_263 = wp::max(var_181, var_256);
                                var_264 = wp::div(var_262, var_263);
                                // act_dot = a * z + actuator_velocity_in[worldid, uid]           <L 1707>
                                var_265 = wp::mul(var_264, var_253);
                                var_266 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_268 = wp::load(var_266);
                                var_267 = wp::add(var_265, var_268);
                                // act_dot_out[worldid, adr] = act_dot                            <L 1708>
                                wp::array_store(var_act_dot_out, var_0, var_229, var_267);
                                // adr += 1                                                       <L 1709>
                                var_270 = wp::add(var_229, var_269);
                            }
                            var_271 = wp::where(var_235, var_270, var_229);
                            var_272 = wp::where(var_235, var_267, var_230);
                            // if slots[4] >= 0:                                                  <L 1712>
                            var_274 = wp::extract(var_98, var_273);
                            var_276 = (var_274 >= var_275);
                            if (var_276) {
                                // dimax = dynprm[1]                                              <L 1713>
                                var_278 = wp::extract(var_55, var_277);
                                // act_dot = (V / R - K / R * actuator_velocity_in[worldid, uid] - act_in[worldid, act_last]) / te       <L 1714>
                                var_279 = wp::div(var_176, var_231);
                                var_280 = wp::div(var_180, var_231);
                                var_281 = wp::address(var_actuator_velocity_in, var_0, var_1);
                                var_283 = wp::load(var_281);
                                var_282 = wp::mul(var_280, var_283);
                                var_284 = wp::sub(var_279, var_282);
                                var_285 = wp::address(var_act_in, var_0, var_48);
                                var_287 = wp::load(var_285);
                                var_286 = wp::sub(var_284, var_287);
                                var_288 = wp::div(var_286, var_184);
                                // if dimax > 0.0:                                                <L 1715>
                                var_290 = (var_278 > var_289);
                                if (var_290) {
                                    // act_dot = wp.clamp(act_dot, -dimax, dimax)                 <L 1716>
                                    var_291 = wp::neg(var_278);
                                    var_292 = wp::clamp(var_288, var_291, var_278);
                                }
                                var_293 = wp::where(var_290, var_292, var_288);
                                // act_dot_out[worldid, act_last] = act_dot                       <L 1717>
                                wp::array_store(var_act_dot_out, var_0, var_48, var_293);
                            }
                            var_294 = wp::where(var_276, var_293, var_272);
                        }
                        if (!var_89) {
                            // elif dyntype == DynType.USER:                                      <L 1719>
                            var_296 = (var_36 == var_295);
                            if (var_296) {
                                // act_dot = 0.0  # set by act_dyn_callback                       <L 1720>
                            }
                            if (!var_296) {
                                // act_dot = 0.0                                                  <L 1722>
                            }
                            var_299 = wp::where(var_296, var_297, var_298);
                        }
                        var_300 = wp::where(var_89, var_133, var_34);
                        var_301 = wp::where(var_89, var_294, var_299);
                    }
                    var_302 = wp::where(var_75, var_34, var_300);
                    var_303 = wp::where(var_75, var_82, var_55);
                    var_304 = wp::where(var_75, var_87, var_301);
                }
                var_305 = wp::where(var_60, var_34, var_302);
                var_306 = wp::where(var_60, var_55, var_303);
                var_307 = wp::where(var_60, var_66, var_85);
                var_308 = wp::where(var_60, var_71, var_181);
                var_309 = wp::where(var_60, var_73, var_304);
            }
            var_310 = wp::where(var_58, var_34, var_305);
            var_311 = wp::where(var_58, var_55, var_306);
            var_312 = wp::where(var_58, var_59, var_309);
            // act_dot_out[worldid, act_last] = act_dot                                           <L 1724>
            wp::array_store(var_act_dot_out, var_0, var_48, var_312);
            // if actuator_actearly[uid]:                                                         <L 1726>
            var_313 = wp::address(var_actuator_actearly, var_1);
            var_314 = wp::load(var_313);
            if (var_314) {
                // if dyntype == DynType.INTEGRATOR or dyntype == DynType.NONE or dyntype == DynType.DCMOTOR:       <L 1727>
                var_317 = (var_36 == var_316);
                var_315 = var_317;
                if (!var_315) {
                    var_319 = (var_36 == var_318);
                    var_315 = var_315 || var_319;
                }
                if (!var_315) {
                    var_321 = (var_36 == var_320);
                    var_315 = var_315 || var_321;
                }
                if (var_315) {
                    // act = act_in[worldid, act_last]                                            <L 1728>
                    var_322 = wp::address(var_act_in, var_0, var_48);
                    var_324 = wp::load(var_322);
                    var_323 = wp::copy(var_324);
                }
                var_325 = wp::where(var_315, var_323, var_307);
                // if dyntype == DynType.DCMOTOR:                                                 <L 1730>
                var_327 = (var_36 == var_326);
                if (var_327) {
                    // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]       <L 1731>
                    var_328 = &(var_actuator_gainprm.shape);
                    var_331 = wp::load(var_328);
                    var_330 = wp::extract(var_331, var_329);
                    var_332 = wp::mod(var_0, var_330);
                    var_333 = wp::address(var_actuator_gainprm, var_332, var_1);
                    var_335 = wp::load(var_333);
                    var_334 = wp::copy(var_335);
                    // slots = util_misc.dcmotor_slots(dynprm, gainprm)                           <L 1732>
                    var_336 = dcmotor_slots_0(var_311, var_334);
                    // offset = actuator_actnum[uid] - 1                                          <L 1733>
                    var_337 = wp::address(var_actuator_actnum, var_1);
                    var_340 = wp::load(var_337);
                    var_339 = wp::sub(var_340, var_338);
                    // if offset == slots[4]:  # current                                          <L 1735>
                    var_342 = wp::extract(var_336, var_341);
                    var_343 = (var_339 == var_342);
                    if (var_343) {
                        // te = wp.max(MJ_MINVAL, dynprm[0])                                      <L 1736>
                        var_345 = wp::extract(var_311, var_344);
                        var_346 = wp::max(var_308, var_345);
                        // ctrl_act = act + act_dot * te * (1.0 - wp.exp(-opt_timestep[worldid % opt_timestep.shape[0]] / te))       <L 1737>
                        var_347 = wp::mul(var_312, var_346);
                        var_349 = &(var_opt_timestep.shape);
                        var_352 = wp::load(var_349);
                        var_351 = wp::extract(var_352, var_350);
                        var_353 = wp::mod(var_0, var_351);
                        var_354 = wp::address(var_opt_timestep, var_353);
                        var_356 = wp::load(var_354);
                        var_355 = wp::neg(var_356);
                        var_357 = wp::div(var_355, var_346);
                        var_358 = wp::exp(var_357);
                        var_359 = wp::sub(var_348, var_358);
                        var_360 = wp::mul(var_347, var_359);
                        var_361 = wp::add(var_325, var_360);
                    }
                    if (!var_343) {
                        // elif offset == slots[3]:  # bristle                                    <L 1738>
                        var_363 = wp::extract(var_336, var_362);
                        var_364 = (var_339 == var_363);
                        if (var_364) {
                            // sigma0 = dynprm[5]                                                 <L 1739>
                            var_366 = wp::extract(var_311, var_365);
                            // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]       <L 1740>
                            var_367 = &(var_actuator_biasprm.shape);
                            var_370 = wp::load(var_367);
                            var_369 = wp::extract(var_370, var_368);
                            var_371 = wp::mod(var_0, var_369);
                            var_372 = wp::address(var_actuator_biasprm, var_371, var_1);
                            var_374 = wp::load(var_372);
                            var_373 = wp::copy(var_374);
                            // F_C = biasprm[3]                                                   <L 1741>
                            var_376 = wp::extract(var_373, var_375);
                            // F_S = biasprm[4]                                                   <L 1742>
                            var_378 = wp::extract(var_373, var_377);
                            // v_S = biasprm[5]                                                   <L 1743>
                            var_380 = wp::extract(var_373, var_379);
                            // velocity = actuator_velocity_in[worldid, uid]                      <L 1744>
                            var_381 = wp::address(var_actuator_velocity_in, var_0, var_1);
                            var_383 = wp::load(var_381);
                            var_382 = wp::copy(var_383);
                            // g = util_misc.lugre_stribeck(velocity, F_C, F_S, v_S)              <L 1745>
                            var_384 = lugre_stribeck_0(var_382, var_376, var_378, var_380);
                            // a = -sigma0 * wp.abs(velocity) / wp.max(MJ_MINVAL, g)              <L 1746>
                            var_385 = wp::neg(var_366);
                            var_386 = wp::abs(var_382);
                            var_387 = wp::mul(var_385, var_386);
                            var_388 = wp::max(var_308, var_384);
                            var_389 = wp::div(var_387, var_388);
                            // h = opt_timestep[worldid % opt_timestep.shape[0]]                  <L 1747>
                            var_390 = &(var_opt_timestep.shape);
                            var_393 = wp::load(var_390);
                            var_392 = wp::extract(var_393, var_391);
                            var_394 = wp::mod(var_0, var_392);
                            var_395 = wp::address(var_opt_timestep, var_394);
                            var_397 = wp::load(var_395);
                            var_396 = wp::copy(var_397);
                            // exp_ah = wp.exp(a * h)                                             <L 1748>
                            var_398 = wp::mul(var_389, var_396);
                            var_399 = wp::exp(var_398);
                            // int_h = h                                                          <L 1749>
                            var_400 = wp::copy(var_396);
                            // if wp.abs(a) > MJ_MINVAL:                                          <L 1750>
                            var_401 = wp::abs(var_389);
                            var_402 = (var_401 > var_308);
                            if (var_402) {
                                // int_h = (exp_ah - 1.0) / a                                     <L 1751>
                                var_404 = wp::sub(var_399, var_403);
                                var_405 = wp::div(var_404, var_389);
                            }
                            var_406 = wp::where(var_402, var_405, var_400);
                            // ctrl_act = exp_ah * act + int_h * velocity                         <L 1752>
                            var_407 = wp::mul(var_399, var_325);
                            var_408 = wp::mul(var_406, var_382);
                            var_409 = wp::add(var_407, var_408);
                        }
                        if (!var_364) {
                            // elif offset == slots[1]:  # integral                               <L 1753>
                            var_411 = wp::extract(var_336, var_410);
                            var_412 = (var_339 == var_411);
                            if (var_412) {
                                // ctrl_act = act + act_dot * opt_timestep[worldid % opt_timestep.shape[0]]       <L 1754>
                                var_413 = &(var_opt_timestep.shape);
                                var_416 = wp::load(var_413);
                                var_415 = wp::extract(var_416, var_414);
                                var_417 = wp::mod(var_0, var_415);
                                var_418 = wp::address(var_opt_timestep, var_417);
                                var_420 = wp::load(var_418);
                                var_419 = wp::mul(var_312, var_420);
                                var_421 = wp::add(var_325, var_419);
                                // Imax = dynprm[8]                                               <L 1755>
                                var_423 = wp::extract(var_311, var_422);
                                // if Imax > 0.0:                                                 <L 1756>
                                var_425 = (var_423 > var_424);
                                if (var_425) {
                                    // ctrl_act = wp.clamp(ctrl_act, -Imax, Imax)                 <L 1757>
                                    var_426 = wp::neg(var_423);
                                    var_427 = wp::clamp(var_421, var_426, var_423);
                                }
                                var_428 = wp::where(var_425, var_427, var_421);
                            }
                            if (!var_412) {
                                // ctrl_act = act + act_dot * opt_timestep[worldid % opt_timestep.shape[0]]       <L 1759>
                                var_429 = &(var_opt_timestep.shape);
                                var_432 = wp::load(var_429);
                                var_431 = wp::extract(var_432, var_430);
                                var_433 = wp::mod(var_0, var_431);
                                var_434 = wp::address(var_opt_timestep, var_433);
                                var_436 = wp::load(var_434);
                                var_435 = wp::mul(var_312, var_436);
                                var_437 = wp::add(var_325, var_435);
                            }
                            var_438 = wp::where(var_412, var_428, var_437);
                            var_439 = wp::where(var_412, var_423, var_145);
                        }
                        var_440 = wp::where(var_364, var_409, var_438);
                        var_441 = wp::where(var_364, var_145, var_439);
                        var_442 = wp::where(var_364, var_366, var_237);
                        var_443 = wp::where(var_364, var_373, var_244);
                        var_444 = wp::where(var_364, var_376, var_247);
                        var_445 = wp::where(var_364, var_378, var_249);
                        var_446 = wp::where(var_364, var_380, var_251);
                        var_447 = wp::where(var_364, var_384, var_256);
                        var_448 = wp::where(var_364, var_389, var_264);
                    }
                    var_449 = wp::where(var_343, var_361, var_440);
                    var_450 = wp::where(var_343, var_145, var_441);
                    var_451 = wp::where(var_343, var_346, var_184);
                    var_452 = wp::where(var_343, var_237, var_442);
                    var_453 = wp::where(var_343, var_244, var_443);
                    var_454 = wp::where(var_343, var_247, var_444);
                    var_455 = wp::where(var_343, var_249, var_445);
                    var_456 = wp::where(var_343, var_251, var_446);
                    var_457 = wp::where(var_343, var_256, var_447);
                    var_458 = wp::where(var_343, var_264, var_448);
                    // if actuator_actlimited[uid]:                                               <L 1761>
                    var_459 = wp::address(var_actuator_actlimited, var_1);
                    var_460 = wp::load(var_459);
                    if (var_460) {
                        // actrange = actuator_actrange[worldid % actuator_actrange.shape[0], uid]       <L 1762>
                        var_461 = &(var_actuator_actrange.shape);
                        var_464 = wp::load(var_461);
                        var_463 = wp::extract(var_464, var_462);
                        var_465 = wp::mod(var_0, var_463);
                        var_466 = wp::address(var_actuator_actrange, var_465, var_1);
                        var_468 = wp::load(var_466);
                        var_467 = wp::copy(var_468);
                        // ctrl_act = wp.clamp(ctrl_act, actrange[0], actrange[1])                <L 1763>
                        var_470 = wp::extract(var_467, var_469);
                        var_472 = wp::extract(var_467, var_471);
                        var_473 = wp::clamp(var_449, var_470, var_472);
                    }
                    var_474 = wp::load(var_459);
                    var_476 = wp::load(var_459);
                    var_475 = wp::where(var_476, var_473, var_449);
                }
                if (!var_327) {
                    // ctrl_act = next_act(                                                       <L 1765>
                    // opt_timestep[worldid % opt_timestep.shape[0]],                             <L 1766>
                    var_477 = &(var_opt_timestep.shape);
                    var_480 = wp::load(var_477);
                    var_479 = wp::extract(var_480, var_478);
                    var_481 = wp::mod(var_0, var_479);
                    var_482 = wp::address(var_opt_timestep, var_481);
                    // dyntype,                                                                   <L 1767>
                    // dynprm,                                                                    <L 1768>
                    // actuator_actrange[worldid % actuator_actrange.shape[0], uid],              <L 1769>
                    var_483 = &(var_actuator_actrange.shape);
                    var_486 = wp::load(var_483);
                    var_485 = wp::extract(var_486, var_484);
                    var_487 = wp::mod(var_0, var_485);
                    var_488 = wp::address(var_actuator_actrange, var_487, var_1);
                    // act,                                                                       <L 1770>
                    // act_dot,                                                                   <L 1771>
                    // 1.0,                                                                       <L 1772>
                    // actuator_actlimited[uid],                                                  <L 1773>
                    var_490 = wp::address(var_actuator_actlimited, var_1);
                    var_492 = wp::load(var_482);
                    var_493 = wp::load(var_488);
                    var_494 = wp::load(var_490);
                    var_491 = next_act_0(var_492, var_36, var_311, var_493, var_325, var_312, var_489, var_494);
                }
                var_495 = wp::where(var_327, var_475, var_491);
                var_496 = wp::where(var_327, var_334, var_96);
                var_497 = wp::where(var_327, var_336, var_98);
                var_498 = wp::where(var_327, var_450, var_145);
                var_499 = wp::where(var_327, var_451, var_184);
                var_500 = wp::where(var_327, var_452, var_237);
                var_501 = wp::where(var_327, var_453, var_244);
                var_502 = wp::where(var_327, var_454, var_247);
                var_503 = wp::where(var_327, var_455, var_249);
                var_504 = wp::where(var_327, var_456, var_251);
                var_505 = wp::where(var_327, var_457, var_256);
                var_506 = wp::where(var_327, var_458, var_264);
            }
            var_507 = wp::load(var_313);
            var_508 = wp::load(var_313);
            if (!var_508) {
                // ctrl_act = act_in[worldid, act_last]                                           <L 1776>
                var_509 = wp::address(var_act_in, var_0, var_48);
                var_511 = wp::load(var_509);
                var_510 = wp::copy(var_511);
            }
            var_512 = wp::load(var_313);
            var_514 = wp::load(var_313);
            var_513 = wp::where(var_514, var_495, var_510);
            var_516 = wp::load(var_313);
            var_515 = wp::where(var_516, var_325, var_307);
            var_518 = wp::load(var_313);
            var_517 = wp::where(var_518, var_496, var_96);
            var_520 = wp::load(var_313);
            var_519 = wp::where(var_520, var_497, var_98);
            var_522 = wp::load(var_313);
            var_521 = wp::where(var_522, var_498, var_145);
            var_524 = wp::load(var_313);
            var_523 = wp::where(var_524, var_499, var_184);
            var_526 = wp::load(var_313);
            var_525 = wp::where(var_526, var_500, var_237);
            var_528 = wp::load(var_313);
            var_527 = wp::where(var_528, var_501, var_244);
            var_530 = wp::load(var_313);
            var_529 = wp::where(var_530, var_502, var_247);
            var_532 = wp::load(var_313);
            var_531 = wp::where(var_532, var_503, var_249);
            var_534 = wp::load(var_313);
            var_533 = wp::where(var_534, var_504, var_251);
            var_536 = wp::load(var_313);
            var_535 = wp::where(var_536, var_505, var_256);
            var_538 = wp::load(var_313);
            var_537 = wp::where(var_538, var_506, var_264);
        }
        var_539 = wp::where(var_41, var_513, var_33);
        var_540 = wp::where(var_41, var_310, var_34);
        // length = actuator_length_in[worldid, uid]                                              <L 1778>
        var_541 = wp::address(var_actuator_length_in, var_0, var_1);
        var_543 = wp::load(var_541);
        var_542 = wp::copy(var_543);
        // velocity = actuator_velocity_in[worldid, uid]                                          <L 1779>
        var_544 = wp::address(var_actuator_velocity_in, var_0, var_1);
        var_546 = wp::load(var_544);
        var_545 = wp::copy(var_546);
        // gaintype = actuator_gaintype[uid]                                                      <L 1782>
        var_547 = wp::address(var_actuator_gaintype, var_1);
        var_549 = wp::load(var_547);
        var_548 = wp::copy(var_549);
        // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], uid]                   <L 1783>
        var_550 = &(var_actuator_gainprm.shape);
        var_553 = wp::load(var_550);
        var_552 = wp::extract(var_553, var_551);
        var_554 = wp::mod(var_0, var_552);
        var_555 = wp::address(var_actuator_gainprm, var_554, var_1);
        var_557 = wp::load(var_555);
        var_556 = wp::copy(var_557);
        // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]                      <L 1784>
        var_558 = &(var_actuator_dynprm.shape);
        var_561 = wp::load(var_558);
        var_560 = wp::extract(var_561, var_559);
        var_562 = wp::mod(var_0, var_560);
        var_563 = wp::address(var_actuator_dynprm, var_562, var_1);
        var_565 = wp::load(var_563);
        var_564 = wp::copy(var_565);
        // gain = 0.0                                                                             <L 1786>
        // if gaintype == GainType.FIXED:                                                         <L 1787>
        var_568 = (var_548 == var_567);
        if (var_568) {
            // gain = gainprm[0]                                                                  <L 1788>
            var_570 = wp::extract(var_556, var_569);
        }
        if (!var_568) {
            // elif gaintype == GainType.AFFINE:                                                  <L 1789>
            var_572 = (var_548 == var_571);
            if (var_572) {
                // gain = gainprm[0] + gainprm[1] * length + gainprm[2] * velocity                <L 1790>
                var_574 = wp::extract(var_556, var_573);
                var_576 = wp::extract(var_556, var_575);
                var_577 = wp::mul(var_576, var_542);
                var_578 = wp::add(var_574, var_577);
                var_580 = wp::extract(var_556, var_579);
                var_581 = wp::mul(var_580, var_545);
                var_582 = wp::add(var_578, var_581);
            }
            if (!var_572) {
                // elif gaintype == GainType.MUSCLE:                                              <L 1791>
                var_584 = (var_548 == var_583);
                if (var_584) {
                    // acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]                <L 1792>
                    var_585 = &(var_actuator_acc0.shape);
                    var_588 = wp::load(var_585);
                    var_587 = wp::extract(var_588, var_586);
                    var_589 = wp::mod(var_0, var_587);
                    var_590 = wp::address(var_actuator_acc0, var_589, var_1);
                    var_592 = wp::load(var_590);
                    var_591 = wp::copy(var_592);
                    // lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]       <L 1793>
                    var_593 = &(var_actuator_lengthrange.shape);
                    var_596 = wp::load(var_593);
                    var_595 = wp::extract(var_596, var_594);
                    var_597 = wp::mod(var_0, var_595);
                    var_598 = wp::address(var_actuator_lengthrange, var_597, var_1);
                    var_600 = wp::load(var_598);
                    var_599 = wp::copy(var_600);
                    // gain = util_misc.muscle_gain(length, velocity, lengthrange, acc0, gainprm)       <L 1794>
                    var_601 = muscle_gain_0(var_542, var_545, var_599, var_591, var_556);
                }
                if (!var_584) {
                    // elif gaintype == GainType.DCMOTOR:                                         <L 1795>
                    var_603 = (var_548 == var_602);
                    if (var_603) {
                        // K = gainprm[1]                                                         <L 1796>
                        var_605 = wp::extract(var_556, var_604);
                        // te = dynprm[0]                                                         <L 1797>
                        var_607 = wp::extract(var_564, var_606);
                        // R = util_misc.dcmotor_resistance(act_in, worldid, act_first, dynprm, gainprm)       <L 1798>
                        var_608 = dcmotor_resistance_0(var_act_in, var_0, var_39, var_564, var_556);
                        // gain = K if te > 0.0 else K / R                                        <L 1800>
                        var_610 = (var_607 > var_609);
                        if (var_610) {
                        }
                        if (!var_610) {
                            var_611 = wp::div(var_605, var_608);
                        }
                        var_612 = wp::where(var_610, var_605, var_611);
                        // if te <= 0.0:                                                          <L 1802>
                        var_614 = (var_607 <= var_613);
                        if (var_614) {
                            // if ctrlnum == 0:                                                   <L 1803>
                            var_616 = (var_11 == var_615);
                            if (var_616) {
                                // ctrl_act = 0.0                                                 <L 1804>
                            }
                            if (!var_616) {
                                // elif (actuator_ctrlspec[uid] & 7) != 0:                        <L 1805>
                                var_618 = wp::address(var_actuator_ctrlspec, var_1);
                                var_621 = wp::load(var_618);
                                var_620 = wp::bit_and(var_621, var_619);
                                var_623 = (var_620 != var_622);
                                if (var_623) {
                                    // x_I = 0.0                                                  <L 1806>
                                    // if na and act_first >= 0:                                  <L 1807>
                                    var_625 = var_na;
                                    if (var_625) {
                                        var_627 = (var_39 >= var_626);
                                        var_625 = var_625 && var_627;
                                    }
                                    if (var_625) {
                                        // slots = util_misc.dcmotor_slots(dynprm, gainprm)       <L 1808>
                                        var_628 = dcmotor_slots_0(var_564, var_556);
                                        // if slots[1] >= 0:                                      <L 1809>
                                        var_630 = wp::extract(var_628, var_629);
                                        var_632 = (var_630 >= var_631);
                                        if (var_632) {
                                            // x_I = act_in[worldid, act_first + slots[1]]        <L 1810>
                                            var_634 = wp::extract(var_628, var_633);
                                            var_635 = wp::add(var_39, var_634);
                                            var_636 = wp::address(var_act_in, var_0, var_635);
                                            var_638 = wp::load(var_636);
                                            var_637 = wp::copy(var_638);
                                        }
                                        var_639 = wp::where(var_632, var_637, var_624);
                                    }
                                    var_640 = wp::where(var_625, var_628, var_519);
                                    var_641 = wp::where(var_625, var_639, var_624);
                                    // ctrl_act = util_misc.dcmotor_voltage(                      <L 1811>
                                    // ctrl_in,                                                   <L 1812>
                                    // worldid,                                                   <L 1813>
                                    // uadr,                                                      <L 1814>
                                    // actuator_ctrlspec[uid],                                    <L 1815>
                                    var_642 = wp::address(var_actuator_ctrlspec, var_1);
                                    // u_first,                                                   <L 1816>
                                    // length,                                                    <L 1817>
                                    // velocity,                                                  <L 1818>
                                    // x_I,                                                       <L 1819>
                                    // gainprm,                                                   <L 1820>
                                    var_644 = wp::load(var_642);
                                    var_643 = dcmotor_voltage_0(var_ctrl_in, var_0, var_8, var_644, var_540, var_542, var_545, var_641, var_556);
                                }
                                if (!var_623) {
                                    // ctrl_act = u_first                                         <L 1823>
                                    var_645 = wp::copy(var_540);
                                }
                                var_646 = wp::where(var_623, var_643, var_645);
                                var_647 = wp::where(var_623, var_640, var_519);
                                var_648 = wp::where(var_623, var_641, var_165);
                            }
                            var_649 = wp::where(var_616, var_617, var_646);
                            var_650 = wp::where(var_616, var_519, var_647);
                            var_651 = wp::where(var_616, var_165, var_648);
                        }
                        var_652 = wp::where(var_614, var_649, var_539);
                        var_653 = wp::where(var_614, var_650, var_519);
                        var_654 = wp::where(var_614, var_651, var_165);
                    }
                    var_655 = wp::where(var_603, var_652, var_539);
                    var_656 = wp::where(var_603, var_653, var_519);
                    var_657 = wp::where(var_603, var_654, var_165);
                    var_658 = wp::where(var_603, var_608, var_231);
                    var_659 = wp::where(var_603, var_605, var_180);
                    var_660 = wp::where(var_603, var_607, var_523);
                    var_661 = wp::where(var_603, var_612, var_566);
                }
                var_662 = wp::where(var_584, var_539, var_655);
                var_663 = wp::where(var_584, var_519, var_656);
                var_664 = wp::where(var_584, var_165, var_657);
                var_665 = wp::where(var_584, var_231, var_658);
                var_666 = wp::where(var_584, var_180, var_659);
                var_667 = wp::where(var_584, var_523, var_660);
                var_668 = wp::where(var_584, var_601, var_661);
            }
            var_669 = wp::where(var_572, var_539, var_662);
            var_670 = wp::where(var_572, var_519, var_663);
            var_671 = wp::where(var_572, var_165, var_664);
            var_672 = wp::where(var_572, var_231, var_665);
            var_673 = wp::where(var_572, var_180, var_666);
            var_674 = wp::where(var_572, var_523, var_667);
            var_675 = wp::where(var_572, var_582, var_668);
        }
        var_676 = wp::where(var_568, var_539, var_669);
        var_677 = wp::where(var_568, var_519, var_670);
        var_678 = wp::where(var_568, var_165, var_671);
        var_679 = wp::where(var_568, var_231, var_672);
        var_680 = wp::where(var_568, var_180, var_673);
        var_681 = wp::where(var_568, var_523, var_674);
        var_682 = wp::where(var_568, var_570, var_675);
        // biastype = actuator_biastype[uid]                                                      <L 1827>
        var_683 = wp::address(var_actuator_biastype, var_1);
        var_685 = wp::load(var_683);
        var_684 = wp::copy(var_685);
        // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], uid]                   <L 1828>
        var_686 = &(var_actuator_biasprm.shape);
        var_689 = wp::load(var_686);
        var_688 = wp::extract(var_689, var_687);
        var_690 = wp::mod(var_0, var_688);
        var_691 = wp::address(var_actuator_biasprm, var_690, var_1);
        var_693 = wp::load(var_691);
        var_692 = wp::copy(var_693);
        // bias = 0.0  # BiasType.NONE or BiasType.USER (modified by act_bias_callback)           <L 1830>
        // if biastype == BiasType.AFFINE:                                                        <L 1831>
        var_696 = (var_684 == var_695);
        if (var_696) {
            // bias = biasprm[0] + biasprm[1] * length + biasprm[2] * velocity                    <L 1832>
            var_698 = wp::extract(var_692, var_697);
            var_700 = wp::extract(var_692, var_699);
            var_701 = wp::mul(var_700, var_542);
            var_702 = wp::add(var_698, var_701);
            var_704 = wp::extract(var_692, var_703);
            var_705 = wp::mul(var_704, var_545);
            var_706 = wp::add(var_702, var_705);
        }
        if (!var_696) {
            // elif biastype == BiasType.MUSCLE:                                                  <L 1833>
            var_708 = (var_684 == var_707);
            if (var_708) {
                // acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], uid]                    <L 1834>
                var_709 = &(var_actuator_acc0.shape);
                var_712 = wp::load(var_709);
                var_711 = wp::extract(var_712, var_710);
                var_713 = wp::mod(var_0, var_711);
                var_714 = wp::address(var_actuator_acc0, var_713, var_1);
                var_716 = wp::load(var_714);
                var_715 = wp::copy(var_716);
                // lengthrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], uid]       <L 1835>
                var_717 = &(var_actuator_lengthrange.shape);
                var_720 = wp::load(var_717);
                var_719 = wp::extract(var_720, var_718);
                var_721 = wp::mod(var_0, var_719);
                var_722 = wp::address(var_actuator_lengthrange, var_721, var_1);
                var_724 = wp::load(var_722);
                var_723 = wp::copy(var_724);
                // bias = util_misc.muscle_bias(length, lengthrange, acc0, biasprm)               <L 1836>
                var_725 = muscle_bias_0(var_542, var_723, var_715, var_692);
            }
            if (!var_708) {
                // elif biastype == BiasType.DCMOTOR:                                             <L 1837>
                var_727 = (var_684 == var_726);
                if (var_727) {
                    // if dynprm[0] <= 0.0:                                                       <L 1838>
                    var_729 = wp::extract(var_564, var_728);
                    var_731 = (var_729 <= var_730);
                    if (var_731) {
                        // K = gainprm[1]                                                         <L 1839>
                        var_733 = wp::extract(var_556, var_732);
                        // bias -= gain * K * velocity                                            <L 1840>
                        var_734 = wp::mul(var_682, var_733);
                        var_735 = wp::mul(var_734, var_545);
                        var_736 = wp::sub(var_694, var_735);
                    }
                    var_737 = wp::where(var_731, var_733, var_680);
                    var_738 = wp::where(var_731, var_736, var_694);
                }
                var_739 = wp::where(var_727, var_737, var_680);
                var_740 = wp::where(var_727, var_738, var_694);
            }
            var_741 = wp::where(var_708, var_680, var_739);
            var_742 = wp::where(var_708, var_715, var_591);
            var_743 = wp::where(var_708, var_723, var_599);
            var_744 = wp::where(var_708, var_725, var_740);
        }
        var_745 = wp::where(var_696, var_680, var_741);
        var_746 = wp::where(var_696, var_591, var_742);
        var_747 = wp::where(var_696, var_599, var_743);
        var_748 = wp::where(var_696, var_706, var_744);
        // trntype = actuator_trntype[uid]                                                        <L 1842>
        var_749 = wp::address(var_actuator_trntype, var_1);
        var_751 = wp::load(var_749);
        var_750 = wp::copy(var_751);
        // trnid = actuator_trnid[uid]                                                            <L 1843>
        var_752 = wp::address(var_actuator_trnid, var_1);
        var_754 = wp::load(var_752);
        var_753 = wp::copy(var_754);
        // gear = actuator_gear[worldid % actuator_gear.shape[0], uid]                            <L 1844>
        var_755 = &(var_actuator_gear.shape);
        var_758 = wp::load(var_755);
        var_757 = wp::extract(var_758, var_756);
        var_759 = wp::mod(var_0, var_757);
        var_760 = wp::address(var_actuator_gear, var_759, var_1);
        var_762 = wp::load(var_760);
        var_761 = wp::copy(var_762);
        // period = _wrap_period(                                                                 <L 1845>
        // jnt_type,                                                                              <L 1846>
        // trntype,                                                                               <L 1847>
        // dyntype,                                                                               <L 1848>
        // gaintype,                                                                              <L 1849>
        // biastype,                                                                              <L 1850>
        // trnid,                                                                                 <L 1851>
        // gainprm,                                                                               <L 1852>
        // biasprm,                                                                               <L 1853>
        // gear,                                                                                  <L 1854>
        var_763 = _wrap_period_0(var_jnt_type, var_750, var_36, var_548, var_684, var_753, var_556, var_692, var_761);
        // if period > 0.0:                                                                       <L 1856>
        var_765 = (var_763 > var_764);
        if (var_765) {
            // err = ctrl_act - length                                                            <L 1857>
            var_766 = wp::sub(var_676, var_542);
            // ctrl_act -= period * wp.round(err / period)                                        <L 1858>
            var_767 = wp::div(var_766, var_763);
            var_768 = wp::round(var_767);
            var_769 = wp::mul(var_763, var_768);
            var_770 = wp::sub(var_676, var_769);
        }
        var_771 = wp::where(var_765, var_770, var_676);
        // force = gain * ctrl_act + bias                                                         <L 1860>
        var_772 = wp::mul(var_682, var_771);
        var_773 = wp::add(var_772, var_748);
        // if actuator_forcelimited[uid]:                                                         <L 1862>
        var_774 = wp::address(var_actuator_forcelimited, var_1);
        var_775 = wp::load(var_774);
        if (var_775) {
            // forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], uid]       <L 1863>
            var_776 = &(var_actuator_forcerange.shape);
            var_779 = wp::load(var_776);
            var_778 = wp::extract(var_779, var_777);
            var_780 = wp::mod(var_0, var_778);
            var_781 = wp::address(var_actuator_forcerange, var_780, var_1);
            var_783 = wp::load(var_781);
            var_782 = wp::copy(var_783);
            // force = wp.clamp(force, forcerange[0], forcerange[1])                              <L 1864>
            var_785 = wp::extract(var_782, var_784);
            var_787 = wp::extract(var_782, var_786);
            var_788 = wp::clamp(var_773, var_785, var_787);
        }
        var_789 = wp::load(var_774);
        var_791 = wp::load(var_774);
        var_790 = wp::where(var_791, var_788, var_773);
        // if biastype == BiasType.DCMOTOR:                                                       <L 1867>
        var_793 = (var_684 == var_792);
        if (var_793) {
            // A = biasprm[0]                                                                     <L 1869>
            var_795 = wp::extract(var_692, var_794);
            // if A != 0.0:                                                                       <L 1870>
            var_797 = (var_795 != var_796);
            if (var_797) {
                // Np = biasprm[1]                                                                <L 1871>
                var_799 = wp::extract(var_692, var_798);
                // phi = biasprm[2]                                                               <L 1872>
                var_801 = wp::extract(var_692, var_800);
                // force += A * wp.sin(Np * length + phi)                                         <L 1873>
                var_802 = wp::mul(var_799, var_542);
                var_803 = wp::add(var_802, var_801);
                var_804 = wp::sin(var_803);
                var_805 = wp::mul(var_795, var_804);
                var_806 = wp::add(var_790, var_805);
            }
            var_807 = wp::where(var_797, var_806, var_790);
            // sigma0 = dynprm[5]                                                                 <L 1876>
            var_809 = wp::extract(var_564, var_808);
            // if sigma0 > 0.0:                                                                   <L 1877>
            var_811 = (var_809 > var_810);
            if (var_811) {
                // sigma1 = dynprm[6]                                                             <L 1878>
                var_813 = wp::extract(var_564, var_812);
                // slots = util_misc.dcmotor_slots(dynprm, gainprm)                               <L 1879>
                var_814 = dcmotor_slots_0(var_564, var_556);
                // adr = act_first + slots[3]  # slots[3] is bristle                              <L 1880>
                var_816 = wp::extract(var_814, var_815);
                var_817 = wp::add(var_39, var_816);
                // z = act_in[worldid, adr]                                                       <L 1881>
                var_818 = wp::address(var_act_in, var_0, var_817);
                var_820 = wp::load(var_818);
                var_819 = wp::copy(var_820);
                // z_dot = act_dot_out[worldid, adr]                                              <L 1882>
                var_821 = wp::address(var_act_dot_out, var_0, var_817);
                var_823 = wp::load(var_821);
                var_822 = wp::copy(var_823);
                // force -= sigma0 * z + sigma1 * z_dot                                           <L 1883>
                var_824 = wp::mul(var_809, var_819);
                var_825 = wp::mul(var_813, var_822);
                var_826 = wp::add(var_824, var_825);
                var_827 = wp::sub(var_807, var_826);
            }
            var_828 = wp::where(var_811, var_814, var_677);
            var_829 = wp::where(var_811, var_817, var_271);
            var_830 = wp::where(var_811, var_819, var_253);
            var_831 = wp::where(var_811, var_827, var_807);
        }
        var_832 = wp::where(var_793, var_828, var_677);
        var_833 = wp::where(var_793, var_829, var_271);
        var_834 = wp::where(var_793, var_809, var_525);
        var_835 = wp::where(var_793, var_830, var_253);
        var_836 = wp::where(var_793, var_831, var_790);
        // actuator_force_out[worldid, uid] = force                                               <L 1885>
        wp::array_store(var_actuator_force_out, var_0, var_1, var_836);
    }
}



extern "C" __global__ void _qfrc_actuator_gravcomp_limits_a91eb6a0_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_jnt_actfrclimited,
    wp::array_t<wp::int32> var_jnt_actgravcomp,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_actfrcrange,
    wp::array_t<wp::int32> var_dof_jntid,
    wp::array_t<wp::float32> var_qfrc_gravcomp_in,
    wp::array_t<wp::float32> var_qfrc_actuator_in,
    bool var_gravity_enabled,
    wp::array_t<wp::float32> var_qfrc_actuator_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        bool var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        bool* var_15;
        bool var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::vec_t<2, wp::float32>* var_22;
        wp::vec_t<2, wp::float32> var_23;
        wp::vec_t<2, wp::float32> var_24;
        const wp::int32 var_25 = 0;
        wp::float32 var_26;
        const wp::int32 var_27 = 1;
        wp::float32 var_28;
        wp::float32 var_29;
        bool var_30;
        wp::float32 var_31;
        bool var_32;
        //---------
        // forward
        // def _qfrc_actuator_gravcomp_limits(                                                    <L 1956>
        // worldid, dofid = wp.tid()                                                              <L 1970>
        builtin_tid2d(var_0, var_1);
        // jntid = dof_jntid[dofid]                                                               <L 1971>
        var_2 = wp::address(var_dof_jntid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // qfrc = qfrc_actuator_in[worldid, dofid]                                                <L 1973>
        var_5 = wp::address(var_qfrc_actuator_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if gravity_enabled and jnt_actgravcomp[jntid]:                                         <L 1976>
        var_8 = var_gravity_enabled;
        if (var_8) {
            var_9 = wp::address(var_jnt_actgravcomp, var_3);
            var_10 = wp::load(var_9);
            var_8 = var_8 && var_10;
        }
        if (var_8) {
            // qfrc += qfrc_gravcomp_in[worldid, dofid]                                           <L 1977>
            var_11 = wp::address(var_qfrc_gravcomp_in, var_0, var_1);
            var_13 = wp::load(var_11);
            var_12 = wp::add(var_6, var_13);
        }
        var_14 = wp::where(var_8, var_12, var_6);
        // if jnt_actfrclimited[jntid]:                                                           <L 1980>
        var_15 = wp::address(var_jnt_actfrclimited, var_3);
        var_16 = wp::load(var_15);
        if (var_16) {
            // frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], jntid]              <L 1981>
            var_17 = &(var_jnt_actfrcrange.shape);
            var_20 = wp::load(var_17);
            var_19 = wp::extract(var_20, var_18);
            var_21 = wp::mod(var_0, var_19);
            var_22 = wp::address(var_jnt_actfrcrange, var_21, var_3);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])                                    <L 1982>
            var_26 = wp::extract(var_23, var_25);
            var_28 = wp::extract(var_23, var_27);
            var_29 = wp::clamp(var_14, var_26, var_28);
        }
        var_30 = wp::load(var_15);
        var_32 = wp::load(var_15);
        var_31 = wp::where(var_32, var_29, var_14);
        // qfrc_actuator_out[worldid, dofid] = qfrc                                               <L 1984>
        wp::array_store(var_qfrc_actuator_out, var_0, var_1, var_31);
    }
}



extern "C" __global__ void _tendon_velocity_ca816a63_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_velocity_out)
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
        const wp::float32 var_2 = 0.0;
        wp::float32 var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::float32 var_16 = 0.0;
        bool var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        //---------
        // forward
        // def _tendon_velocity(                                                                  <L 1504>
        // worldid, tenid = wp.tid()                                                              <L 1515>
        builtin_tid2d(var_0, var_1);
        // velocity = float(0.0)                                                                  <L 1517>
        var_3 = wp::float(var_2);
        // rownnz = ten_J_rownnz[tenid]                                                           <L 1518>
        var_4 = wp::address(var_ten_J_rownnz, var_1);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // rowadr = ten_J_rowadr[tenid]                                                           <L 1519>
        var_7 = wp::address(var_ten_J_rowadr, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // for i in range(rownnz):                                                                <L 1520>
        var_10 = wp::range(var_5);
        start_for_0:;
            if (iter_cmp(var_10) == 0) goto end_for_0;
            var_11 = wp::iter_next(var_10);
            // sparseid = rowadr + i                                                              <L 1521>
            var_12 = wp::add(var_8, var_11);
            // J = ten_J_in[worldid, sparseid]                                                    <L 1522>
            var_13 = wp::address(var_ten_J_in, var_0, var_12);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // if J != 0.0:                                                                       <L 1523>
            var_17 = (var_14 != var_16);
            if (var_17) {
                // colind = ten_J_colind[sparseid]                                                <L 1524>
                var_18 = wp::address(var_ten_J_colind, var_12);
                var_20 = wp::load(var_18);
                var_19 = wp::copy(var_20);
                // velocity += J * qvel_in[worldid, colind]                                       <L 1525>
                var_21 = wp::address(var_qvel_in, var_0, var_19);
                var_23 = wp::load(var_21);
                var_22 = wp::mul(var_14, var_23);
                var_24 = wp::add(var_3, var_22);
            }
            var_25 = wp::where(var_17, var_24, var_3);
            wp::assign(var_3, var_25);
            goto start_for_0;
        end_for_0:;
        // ten_velocity_out[worldid, tenid] = velocity                                            <L 1527>
        wp::array_store(var_ten_velocity_out, var_0, var_1, var_3);
    }
}



extern "C" __global__ void _next_activation_cf227b4a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_gaintype,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<wp::vec_t<6, wp::float32>> var_actuator_gear,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::float32 var_act_dot_scale,
    bool var_limit,
    wp::array_t<wp::float32> var_act_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::shape_t* var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::shape_t var_10;
        wp::int32 var_11;
        wp::shape_t* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::shape_t var_15;
        wp::int32 var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::shape_t* var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::shape_t var_25;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 5;
        bool var_37;
        wp::vec_t<10, wp::float32>* var_38;
        wp::vec_t<10, wp::float32> var_39;
        wp::vec_t<10, wp::float32> var_40;
        wp::vec_t<10, wp::float32>* var_41;
        wp::vec_t<10, wp::float32> var_42;
        wp::vec_t<10, wp::float32> var_43;
        wp::vec_t<10, wp::float32>* var_44;
        wp::vec_t<10, wp::float32> var_45;
        wp::vec_t<10, wp::float32> var_46;
        wp::vec_t<6, wp::int32> var_47;
        wp::int32 var_48;
        wp::range_t var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::float32* var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 4;
        wp::int32 var_59;
        bool var_60;
        const wp::int32 var_61 = 0;
        wp::float32 var_62;
        const wp::float32 var_63 = 1e-15;
        const wp::int32 var_64 = 0;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::float32 var_68 = 1.0;
        wp::float32* var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        const wp::int32 var_77 = 3;
        wp::int32 var_78;
        bool var_79;
        const wp::int32 var_80 = 3;
        wp::float32 var_81;
        const wp::int32 var_82 = 4;
        wp::float32 var_83;
        const wp::int32 var_84 = 5;
        wp::float32 var_85;
        const wp::int32 var_86 = 5;
        wp::float32 var_87;
        wp::float32* var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        const wp::float32 var_95 = 1e-15;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32* var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        bool var_105;
        const wp::float32 var_106 = 1.0;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        const wp::int32 var_113 = 1;
        wp::int32 var_114;
        bool var_115;
        wp::float32* var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::int32 var_120 = 8;
        wp::float32 var_121;
        const wp::float32 var_122 = 0.0;
        bool var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32* var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::int32 var_135;
        wp::range_t var_136;
        wp::int32 var_137;
        wp::float32* var_138;
        wp::vec_t<10, wp::float32>* var_139;
        wp::vec_t<2, wp::float32>* var_140;
        wp::float32* var_141;
        wp::float32* var_142;
        bool var_143;
        bool* var_144;
        bool var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::vec_t<10, wp::float32> var_148;
        wp::vec_t<2, wp::float32> var_149;
        wp::float32 var_150;
        wp::float32 var_151;
        bool var_152;
        const wp::int32 var_153 = 1;
        bool var_154;
        wp::int32* var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32* var_158;
        wp::int32 var_159;
        wp::int32 var_160;
        wp::vec_t<10, wp::float32>* var_161;
        wp::vec_t<10, wp::float32> var_162;
        wp::vec_t<10, wp::float32> var_163;
        wp::vec_t<10, wp::float32>* var_164;
        wp::vec_t<10, wp::float32> var_165;
        wp::vec_t<10, wp::float32> var_166;
        wp::int32* var_167;
        wp::int32 var_168;
        wp::int32 var_169;
        wp::vec_t<2, wp::int32>* var_170;
        wp::vec_t<2, wp::int32> var_171;
        wp::vec_t<2, wp::int32> var_172;
        wp::shape_t* var_173;
        const wp::int32 var_174 = 0;
        wp::int32 var_175;
        wp::shape_t var_176;
        wp::int32 var_177;
        wp::vec_t<6, wp::float32>* var_178;
        wp::vec_t<6, wp::float32> var_179;
        wp::vec_t<6, wp::float32> var_180;
        wp::float32 var_181;
        const wp::float32 var_182 = 0.0;
        bool var_183;
        wp::float32* var_184;
        wp::float32 var_185;
        wp::float32 var_186;
        wp::float32 var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::float32 var_192;
        wp::float32 var_193;
        wp::vec_t<10, wp::float32> var_194;
        wp::vec_t<10, wp::float32> var_195;
        wp::int32 var_196;
        wp::float32 var_197;
        //---------
        // forward
        // def _next_activation(                                                                  <L 206>
        // worldid, uid = wp.tid()                                                                <L 234>
        builtin_tid2d(var_0, var_1);
        // opt_timestep_id = worldid % opt_timestep.shape[0]                                      <L 235>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // actuator_dynprm_id = worldid % actuator_dynprm.shape[0]                                <L 236>
        var_7 = &(var_actuator_dynprm.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // actuator_actrange_id = worldid % actuator_actrange.shape[0]                            <L 237>
        var_12 = &(var_actuator_actrange.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        // actuator_gainprm_id = worldid % actuator_gainprm.shape[0]                              <L 238>
        var_17 = &(var_actuator_gainprm.shape);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        var_21 = wp::mod(var_0, var_19);
        // actuator_biasprm_id = worldid % actuator_biasprm.shape[0]                              <L 239>
        var_22 = &(var_actuator_biasprm.shape);
        var_25 = wp::load(var_22);
        var_24 = wp::extract(var_25, var_23);
        var_26 = wp::mod(var_0, var_24);
        // actadr = actuator_actadr[uid]                                                          <L 241>
        var_27 = wp::address(var_actuator_actadr, var_1);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // actnum = actuator_actnum[uid]                                                          <L 242>
        var_30 = wp::address(var_actuator_actnum, var_1);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // dyntype = actuator_dyntype[uid]                                                        <L 243>
        var_33 = wp::address(var_actuator_dyntype, var_1);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // if dyntype == DynType.DCMOTOR:                                                         <L 245>
        var_37 = (var_34 == var_36);
        if (var_37) {
            // dynprm = actuator_dynprm[actuator_dynprm_id, uid]                                  <L 246>
            var_38 = wp::address(var_actuator_dynprm, var_11, var_1);
            var_40 = wp::load(var_38);
            var_39 = wp::copy(var_40);
            // gainprm = actuator_gainprm[actuator_gainprm_id, uid]                               <L 247>
            var_41 = wp::address(var_actuator_gainprm, var_21, var_1);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // biasprm = actuator_biasprm[actuator_biasprm_id, uid]                               <L 248>
            var_44 = wp::address(var_actuator_biasprm, var_26, var_1);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // slots = util_misc.dcmotor_slots(dynprm, gainprm)                                   <L 249>
            var_47 = dcmotor_slots_0(var_39, var_42);
            // for j in range(actadr, actadr + actnum):                                           <L 251>
            var_48 = wp::add(var_28, var_31);
            var_49 = wp::range(var_28, var_48);
            start_for_0:;
                if (iter_cmp(var_49) == 0) goto end_for_0;
                var_50 = wp::iter_next(var_49);
                // offset = j - actadr                                                            <L 252>
                var_51 = wp::sub(var_50, var_28);
                // act = act_in[worldid, j]                                                       <L 253>
                var_52 = wp::address(var_act_in, var_0, var_50);
                var_54 = wp::load(var_52);
                var_53 = wp::copy(var_54);
                // act_dot = act_dot_in[worldid, j]                                               <L 254>
                var_55 = wp::address(var_act_dot_in, var_0, var_50);
                var_57 = wp::load(var_55);
                var_56 = wp::copy(var_57);
                // if offset == slots[4]:  # current                                              <L 256>
                var_59 = wp::extract(var_47, var_58);
                var_60 = (var_51 == var_59);
                if (var_60) {
                    // R = gainprm[0]                                                             <L 257>
                    var_62 = wp::extract(var_42, var_61);
                    // te = wp.max(MJ_MINVAL, dynprm[0])                                          <L 258>
                    var_65 = wp::extract(var_39, var_64);
                    var_66 = wp::max(var_63, var_65);
                    // act = act + act_dot * te * (1.0 - wp.exp(-opt_timestep[opt_timestep_id] / te))       <L 259>
                    var_67 = wp::mul(var_56, var_66);
                    var_69 = wp::address(var_opt_timestep, var_6);
                    var_71 = wp::load(var_69);
                    var_70 = wp::neg(var_71);
                    var_72 = wp::div(var_70, var_66);
                    var_73 = wp::exp(var_72);
                    var_74 = wp::sub(var_68, var_73);
                    var_75 = wp::mul(var_67, var_74);
                    var_76 = wp::add(var_53, var_75);
                }
                if (!var_60) {
                    // elif offset == slots[3]:  # bristle                                        <L 260>
                    var_78 = wp::extract(var_47, var_77);
                    var_79 = (var_51 == var_78);
                    if (var_79) {
                        // F_C = biasprm[3]                                                       <L 261>
                        var_81 = wp::extract(var_45, var_80);
                        // F_S = biasprm[4]                                                       <L 262>
                        var_83 = wp::extract(var_45, var_82);
                        // v_S = biasprm[5]                                                       <L 263>
                        var_85 = wp::extract(var_45, var_84);
                        // sigma0 = dynprm[5]                                                     <L 264>
                        var_87 = wp::extract(var_39, var_86);
                        // velocity = actuator_velocity_in[worldid, uid]                          <L 265>
                        var_88 = wp::address(var_actuator_velocity_in, var_0, var_1);
                        var_90 = wp::load(var_88);
                        var_89 = wp::copy(var_90);
                        // g = util_misc.lugre_stribeck(velocity, F_C, F_S, v_S)                  <L 266>
                        var_91 = lugre_stribeck_0(var_89, var_81, var_83, var_85);
                        // a = -sigma0 * wp.abs(velocity) / wp.max(MJ_MINVAL, g)                  <L 268>
                        var_92 = wp::neg(var_87);
                        var_93 = wp::abs(var_89);
                        var_94 = wp::mul(var_92, var_93);
                        var_96 = wp::max(var_95, var_91);
                        var_97 = wp::div(var_94, var_96);
                        // h = opt_timestep[opt_timestep_id]                                      <L 269>
                        var_98 = wp::address(var_opt_timestep, var_6);
                        var_100 = wp::load(var_98);
                        var_99 = wp::copy(var_100);
                        // exp_ah = wp.exp(a * h)                                                 <L 270>
                        var_101 = wp::mul(var_97, var_99);
                        var_102 = wp::exp(var_101);
                        // int_h = h                                                              <L 271>
                        var_103 = wp::copy(var_99);
                        // if wp.abs(a) > MJ_MINVAL:                                              <L 272>
                        var_104 = wp::abs(var_97);
                        var_105 = (var_104 > var_95);
                        if (var_105) {
                            // int_h = (exp_ah - 1.0) / a                                         <L 273>
                            var_107 = wp::sub(var_102, var_106);
                            var_108 = wp::div(var_107, var_97);
                        }
                        var_109 = wp::where(var_105, var_108, var_103);
                        // act = exp_ah * act + int_h * velocity                                  <L 274>
                        var_110 = wp::mul(var_102, var_53);
                        var_111 = wp::mul(var_109, var_89);
                        var_112 = wp::add(var_110, var_111);
                    }
                    if (!var_79) {
                        // elif offset == slots[1]:  # integral                                   <L 275>
                        var_114 = wp::extract(var_47, var_113);
                        var_115 = (var_51 == var_114);
                        if (var_115) {
                            // act = act + act_dot * opt_timestep[opt_timestep_id]                <L 276>
                            var_116 = wp::address(var_opt_timestep, var_6);
                            var_118 = wp::load(var_116);
                            var_117 = wp::mul(var_56, var_118);
                            var_119 = wp::add(var_53, var_117);
                            // Imax = dynprm[8]                                                   <L 277>
                            var_121 = wp::extract(var_39, var_120);
                            // if Imax > 0.0:                                                     <L 278>
                            var_123 = (var_121 > var_122);
                            if (var_123) {
                                // act = wp.clamp(act, -Imax, Imax)                               <L 279>
                                var_124 = wp::neg(var_121);
                                var_125 = wp::clamp(var_119, var_124, var_121);
                            }
                            var_126 = wp::where(var_123, var_125, var_119);
                        }
                        if (!var_115) {
                            // act = act + act_dot * opt_timestep[opt_timestep_id]                <L 281>
                            var_127 = wp::address(var_opt_timestep, var_6);
                            var_129 = wp::load(var_127);
                            var_128 = wp::mul(var_56, var_129);
                            var_130 = wp::add(var_53, var_128);
                        }
                        var_131 = wp::where(var_115, var_126, var_130);
                    }
                    var_132 = wp::where(var_79, var_112, var_131);
                }
                var_133 = wp::where(var_60, var_76, var_132);
                var_134 = wp::where(var_60, var_63, var_95);
                // act_out[worldid, j] = act                                                      <L 283>
                wp::array_store(var_act_out, var_0, var_50, var_133);
                goto start_for_0;
            end_for_0:;
        }
        if (!var_37) {
            // for j in range(actadr, actadr + actnum):                                           <L 285>
            var_135 = wp::add(var_28, var_31);
            var_136 = wp::range(var_28, var_135);
            start_for_2:;
                if (iter_cmp(var_136) == 0) goto end_for_2;
                var_137 = wp::iter_next(var_136);
                // act = next_act(                                                                <L 286>
                // opt_timestep[opt_timestep_id],                                                 <L 287>
                var_138 = wp::address(var_opt_timestep, var_6);
                // dyntype,                                                                       <L 288>
                // actuator_dynprm[actuator_dynprm_id, uid],                                      <L 289>
                var_139 = wp::address(var_actuator_dynprm, var_11, var_1);
                // actuator_actrange[actuator_actrange_id, uid],                                  <L 290>
                var_140 = wp::address(var_actuator_actrange, var_16, var_1);
                // act_in[worldid, j],                                                            <L 291>
                var_141 = wp::address(var_act_in, var_0, var_137);
                // act_dot_in[worldid, j],                                                        <L 292>
                var_142 = wp::address(var_act_dot_in, var_0, var_137);
                // act_dot_scale,                                                                 <L 293>
                // limit and actuator_actlimited[uid],                                            <L 294>
                var_143 = var_limit;
                if (var_143) {
                    var_144 = wp::address(var_actuator_actlimited, var_1);
                    var_145 = wp::load(var_144);
                    var_143 = var_143 && var_145;
                }
                var_147 = wp::load(var_138);
                var_148 = wp::load(var_139);
                var_149 = wp::load(var_140);
                var_150 = wp::load(var_141);
                var_151 = wp::load(var_142);
                var_146 = next_act_0(var_147, var_34, var_148, var_149, var_150, var_151, var_act_dot_scale, var_143);
                // if limit and dyntype == DynType.INTEGRATOR:                                    <L 296>
                var_152 = var_limit;
                if (var_152) {
                    var_154 = (var_34 == var_153);
                    var_152 = var_152 && var_154;
                }
                if (var_152) {
                    // gaintype = actuator_gaintype[uid]                                          <L 297>
                    var_155 = wp::address(var_actuator_gaintype, var_1);
                    var_157 = wp::load(var_155);
                    var_156 = wp::copy(var_157);
                    // biastype = actuator_biastype[uid]                                          <L 298>
                    var_158 = wp::address(var_actuator_biastype, var_1);
                    var_160 = wp::load(var_158);
                    var_159 = wp::copy(var_160);
                    // gainprm = actuator_gainprm[actuator_gainprm_id, uid]                       <L 299>
                    var_161 = wp::address(var_actuator_gainprm, var_21, var_1);
                    var_163 = wp::load(var_161);
                    var_162 = wp::copy(var_163);
                    // biasprm = actuator_biasprm[actuator_biasprm_id, uid]                       <L 300>
                    var_164 = wp::address(var_actuator_biasprm, var_26, var_1);
                    var_166 = wp::load(var_164);
                    var_165 = wp::copy(var_166);
                    // trntype = actuator_trntype[uid]                                            <L 301>
                    var_167 = wp::address(var_actuator_trntype, var_1);
                    var_169 = wp::load(var_167);
                    var_168 = wp::copy(var_169);
                    // trnid = actuator_trnid[uid]                                                <L 302>
                    var_170 = wp::address(var_actuator_trnid, var_1);
                    var_172 = wp::load(var_170);
                    var_171 = wp::copy(var_172);
                    // gear = actuator_gear[worldid % actuator_gear.shape[0], uid]                <L 303>
                    var_173 = &(var_actuator_gear.shape);
                    var_176 = wp::load(var_173);
                    var_175 = wp::extract(var_176, var_174);
                    var_177 = wp::mod(var_0, var_175);
                    var_178 = wp::address(var_actuator_gear, var_177, var_1);
                    var_180 = wp::load(var_178);
                    var_179 = wp::copy(var_180);
                    // period = _wrap_period(                                                     <L 304>
                    // jnt_type,                                                                  <L 305>
                    // trntype,                                                                   <L 306>
                    // dyntype,                                                                   <L 307>
                    // gaintype,                                                                  <L 308>
                    // biastype,                                                                  <L 309>
                    // trnid,                                                                     <L 310>
                    // gainprm,                                                                   <L 311>
                    // biasprm,                                                                   <L 312>
                    // gear,                                                                      <L 313>
                    var_181 = _wrap_period_0(var_jnt_type, var_168, var_34, var_156, var_159, var_171, var_162, var_165, var_179);
                    // if period > 0.0:                                                           <L 315>
                    var_183 = (var_181 > var_182);
                    if (var_183) {
                        // length = actuator_length_in[worldid, uid]                              <L 316>
                        var_184 = wp::address(var_actuator_length_in, var_0, var_1);
                        var_186 = wp::load(var_184);
                        var_185 = wp::copy(var_186);
                        // err = act - length                                                     <L 317>
                        var_187 = wp::sub(var_146, var_185);
                        // act -= period * wp.round(err / period)                                 <L 318>
                        var_188 = wp::div(var_187, var_181);
                        var_189 = wp::round(var_188);
                        var_190 = wp::mul(var_181, var_189);
                        var_191 = wp::sub(var_146, var_190);
                    }
                    var_192 = wp::where(var_183, var_191, var_146);
                }
                var_193 = wp::where(var_152, var_192, var_146);
                // act_out[worldid, j] = act                                                      <L 320>
                wp::array_store(var_act_out, var_0, var_137, var_193);
                goto start_for_2;
            end_for_2:;
        }
        var_194 = wp::where(var_37, var_42, var_162);
        var_195 = wp::where(var_37, var_45, var_165);
        var_196 = wp::where(var_37, var_50, var_137);
        var_197 = wp::where(var_37, var_133, var_193);
    }
}



extern "C" __global__ void _compute_damping_deriv_cfbf3e5a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_dof_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_dof_dampingpoly,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_deriv_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::shape_t* var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        wp::shape_t var_13;
        wp::int32 var_14;
        wp::vec_t<2, wp::float32>* var_15;
        wp::vec_t<2, wp::float32> var_16;
        wp::vec_t<2, wp::float32> var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::int32 var_21 = 1;
        wp::float32 var_22;
        //---------
        // forward
        // def _compute_damping_deriv(                                                            <L 478>
        // worldid, tid = wp.tid()                                                                <L 487>
        builtin_tid2d(var_0, var_1);
        // damping = dof_damping[worldid % dof_damping.shape[0], tid]                             <L 488>
        var_2 = &(var_dof_damping.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_dof_damping, var_6, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], tid]                       <L 489>
        var_10 = &(var_dof_dampingpoly.shape);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_14 = wp::mod(var_0, var_12);
        var_15 = wp::address(var_dof_dampingpoly, var_14, var_1);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // v = qvel_in[worldid, tid]                                                              <L 490>
        var_18 = wp::address(var_qvel_in, var_0, var_1);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // deriv_out[worldid, tid] = util_misc._poly_force_deriv(damping, dpoly, v, 1)            <L 491>
        var_22 = _poly_force_deriv_0(var_8, var_16, var_19, var_21);
        wp::array_store(var_deriv_out, var_0, var_1, var_22);
    }
}



extern "C" __global__ void _next_velocity_d66c2f53_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_qacc_in,
    wp::float32 var_qacc_scale_in,
    wp::array_t<wp::float32> var_qvel_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        //---------
        // forward
        // def _next_velocity(                                                                    <L 151>
        // worldid, dofid = wp.tid()                                                              <L 162>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 163>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_scale_in * qacc_in[worldid, dofid] * timestep       <L 164>
        var_10 = wp::address(var_qvel_in, var_0, var_1);
        var_11 = wp::address(var_qacc_in, var_0, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::mul(var_qacc_scale_in, var_13);
        var_14 = wp::mul(var_12, var_8);
        var_16 = wp::load(var_10);
        var_15 = wp::add(var_16, var_14);
        wp::array_store(var_qvel_out, var_0, var_1, var_15);
    }
}



extern "C" __global__ void _implicit_free_body_solve_dcb26f31_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nbody,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_wind,
    wp::array_t<wp::float32> var_opt_density,
    wp::array_t<wp::float32> var_opt_viscosity,
    wp::int32 var_opt_integrator,
    wp::int32 var_opt_disableflags,
    wp::int32 var_opt_enableflags,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_body_geomnum,
    wp::array_t<wp::int32> var_body_geomadr,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::int32> var_dof_treeid,
    wp::array_t<wp::float32> var_dof_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_dof_dampingpoly,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::float32> var_geom_fluid,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_M_colind,
    wp::array_t<bool> var_body_fluid_ellipsoid,
    wp::array_t<wp::int32> var_body_freeadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_crb_in,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_tree_awake_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::float32> var_qfrc_in,
    wp::array_t<wp::float32> var_qacc_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        bool var_11;
        const wp::int32 var_12 = 16;
        wp::int32 var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32 var_17;
        wp::shape_t* var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        wp::shape_t var_21;
        wp::int32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        const wp::float32 var_26 = 0.0;
        wp::mat_t<6, 6, wp::float32> var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::range_t var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::float32* var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32* var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::range_t var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32* var_57;
        wp::int32 var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        wp::float32* var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 2;
        wp::int32 var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        wp::int32* var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::range_t var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::int32* var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        wp::float32* var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        const wp::int32 var_82 = 3;
        wp::int32 var_83;
        wp::int32* var_84;
        wp::int32 var_85;
        wp::int32 var_86;
        wp::int32* var_87;
        wp::int32 var_88;
        wp::int32 var_89;
        wp::range_t var_90;
        wp::int32 var_91;
        wp::int32 var_92;
        wp::int32* var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        wp::float32* var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        const wp::int32 var_100 = 4;
        wp::int32 var_101;
        wp::int32* var_102;
        wp::int32 var_103;
        wp::int32 var_104;
        wp::int32* var_105;
        wp::int32 var_106;
        wp::int32 var_107;
        wp::range_t var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::int32* var_111;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::float32* var_115;
        wp::float32 var_116;
        wp::float32 var_117;
        const wp::int32 var_118 = 5;
        wp::int32 var_119;
        wp::int32* var_120;
        wp::int32 var_121;
        wp::int32 var_122;
        wp::int32* var_123;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::range_t var_126;
        wp::int32 var_127;
        wp::int32 var_128;
        wp::int32* var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        wp::float32* var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        const wp::int32 var_136 = 64;
        wp::int32 var_137;
        bool var_138;
        const wp::int32 var_139 = 0;
        wp::int32 var_140;
        wp::shape_t* var_141;
        const wp::int32 var_142 = 0;
        wp::int32 var_143;
        wp::shape_t var_144;
        wp::int32 var_145;
        wp::float32* var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::shape_t* var_149;
        const wp::int32 var_150 = 0;
        wp::int32 var_151;
        wp::shape_t var_152;
        wp::int32 var_153;
        wp::vec_t<2, wp::float32>* var_154;
        wp::vec_t<2, wp::float32> var_155;
        wp::vec_t<2, wp::float32> var_156;
        wp::float32* var_157;
        wp::float32 var_158;
        wp::float32 var_159;
        const wp::int32 var_160 = 1;
        wp::float32 var_161;
        wp::float32 var_162;
        const wp::int32 var_163 = 1;
        wp::int32 var_164;
        wp::shape_t* var_165;
        const wp::int32 var_166 = 0;
        wp::int32 var_167;
        wp::shape_t var_168;
        wp::int32 var_169;
        wp::float32* var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        wp::shape_t* var_173;
        const wp::int32 var_174 = 0;
        wp::int32 var_175;
        wp::shape_t var_176;
        wp::int32 var_177;
        wp::vec_t<2, wp::float32>* var_178;
        wp::vec_t<2, wp::float32> var_179;
        wp::vec_t<2, wp::float32> var_180;
        wp::float32* var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        const wp::int32 var_184 = 1;
        wp::float32 var_185;
        wp::float32 var_186;
        const wp::int32 var_187 = 2;
        wp::int32 var_188;
        wp::shape_t* var_189;
        const wp::int32 var_190 = 0;
        wp::int32 var_191;
        wp::shape_t var_192;
        wp::int32 var_193;
        wp::float32* var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        wp::shape_t* var_197;
        const wp::int32 var_198 = 0;
        wp::int32 var_199;
        wp::shape_t var_200;
        wp::int32 var_201;
        wp::vec_t<2, wp::float32>* var_202;
        wp::vec_t<2, wp::float32> var_203;
        wp::vec_t<2, wp::float32> var_204;
        wp::float32* var_205;
        wp::float32 var_206;
        wp::float32 var_207;
        const wp::int32 var_208 = 1;
        wp::float32 var_209;
        wp::float32 var_210;
        const wp::int32 var_211 = 3;
        wp::int32 var_212;
        wp::shape_t* var_213;
        const wp::int32 var_214 = 0;
        wp::int32 var_215;
        wp::shape_t var_216;
        wp::int32 var_217;
        wp::float32* var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        wp::shape_t* var_221;
        const wp::int32 var_222 = 0;
        wp::int32 var_223;
        wp::shape_t var_224;
        wp::int32 var_225;
        wp::vec_t<2, wp::float32>* var_226;
        wp::vec_t<2, wp::float32> var_227;
        wp::vec_t<2, wp::float32> var_228;
        wp::float32* var_229;
        wp::float32 var_230;
        wp::float32 var_231;
        const wp::int32 var_232 = 1;
        wp::float32 var_233;
        wp::float32 var_234;
        const wp::int32 var_235 = 4;
        wp::int32 var_236;
        wp::shape_t* var_237;
        const wp::int32 var_238 = 0;
        wp::int32 var_239;
        wp::shape_t var_240;
        wp::int32 var_241;
        wp::float32* var_242;
        wp::float32 var_243;
        wp::float32 var_244;
        wp::shape_t* var_245;
        const wp::int32 var_246 = 0;
        wp::int32 var_247;
        wp::shape_t var_248;
        wp::int32 var_249;
        wp::vec_t<2, wp::float32>* var_250;
        wp::vec_t<2, wp::float32> var_251;
        wp::vec_t<2, wp::float32> var_252;
        wp::float32* var_253;
        wp::float32 var_254;
        wp::float32 var_255;
        const wp::int32 var_256 = 1;
        wp::float32 var_257;
        wp::float32 var_258;
        const wp::int32 var_259 = 5;
        wp::int32 var_260;
        wp::shape_t* var_261;
        const wp::int32 var_262 = 0;
        wp::int32 var_263;
        wp::shape_t var_264;
        wp::int32 var_265;
        wp::float32* var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        wp::shape_t* var_269;
        const wp::int32 var_270 = 0;
        wp::int32 var_271;
        wp::shape_t var_272;
        wp::int32 var_273;
        wp::vec_t<2, wp::float32>* var_274;
        wp::vec_t<2, wp::float32> var_275;
        wp::vec_t<2, wp::float32> var_276;
        wp::float32* var_277;
        wp::float32 var_278;
        wp::float32 var_279;
        const wp::int32 var_280 = 1;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::int32 var_283;
        wp::vec_t<10, wp::float32>* var_284;
        wp::vec_t<10, wp::float32> var_285;
        wp::vec_t<10, wp::float32> var_286;
        const wp::int32 var_287 = 9;
        wp::float32 var_288;
        wp::mat_t<3, 3, wp::float32>* var_289;
        wp::mat_t<3, 3, wp::float32> var_290;
        wp::mat_t<3, 3, wp::float32> var_291;
        const wp::int32 var_292 = 0;
        wp::float32 var_293;
        const wp::int32 var_294 = 3;
        wp::float32 var_295;
        const wp::int32 var_296 = 4;
        wp::float32 var_297;
        const wp::int32 var_298 = 3;
        wp::float32 var_299;
        const wp::int32 var_300 = 1;
        wp::float32 var_301;
        const wp::int32 var_302 = 5;
        wp::float32 var_303;
        const wp::int32 var_304 = 4;
        wp::float32 var_305;
        const wp::int32 var_306 = 5;
        wp::float32 var_307;
        const wp::int32 var_308 = 2;
        wp::float32 var_309;
        wp::mat_t<3, 3, wp::float32> var_310;
        wp::vec_t<3, wp::float32>* var_311;
        wp::vec_t<3, wp::float32>* var_312;
        wp::vec_t<3, wp::float32> var_313;
        wp::vec_t<3, wp::float32> var_314;
        wp::vec_t<3, wp::float32> var_315;
        const wp::int32 var_316 = 3;
        wp::int32 var_317;
        wp::float32* var_318;
        const wp::int32 var_319 = 4;
        wp::int32 var_320;
        wp::float32* var_321;
        const wp::int32 var_322 = 5;
        wp::int32 var_323;
        wp::float32* var_324;
        wp::vec_t<3, wp::float32> var_325;
        wp::float32 var_326;
        wp::float32 var_327;
        wp::float32 var_328;
        wp::mat_t<3, 3, wp::float32> var_329;
        wp::mat_t<3, 3, wp::float32> var_330;
        wp::float32 var_331;
        wp::float32 var_332;
        const wp::int32 var_333 = 0;
        const wp::int32 var_334 = 0;
        const wp::int32 var_335 = 3;
        wp::int32 var_336;
        wp::float32 var_337;
        wp::float32 var_338;
        const wp::int32 var_339 = 3;
        wp::int32 var_340;
        const wp::int32 var_341 = 3;
        wp::int32 var_342;
        wp::float32 var_343;
        wp::float32 var_344;
        const wp::int32 var_345 = 1;
        const wp::int32 var_346 = 3;
        wp::int32 var_347;
        wp::float32 var_348;
        wp::float32 var_349;
        const wp::int32 var_350 = 3;
        wp::int32 var_351;
        const wp::int32 var_352 = 3;
        wp::int32 var_353;
        wp::float32 var_354;
        wp::float32 var_355;
        const wp::int32 var_356 = 2;
        const wp::int32 var_357 = 3;
        wp::int32 var_358;
        wp::float32 var_359;
        wp::float32 var_360;
        const wp::int32 var_361 = 3;
        wp::int32 var_362;
        const wp::int32 var_363 = 3;
        wp::int32 var_364;
        wp::float32 var_365;
        wp::float32 var_366;
        const wp::int32 var_367 = 1;
        const wp::int32 var_368 = 0;
        const wp::int32 var_369 = 3;
        wp::int32 var_370;
        wp::float32 var_371;
        wp::float32 var_372;
        const wp::int32 var_373 = 3;
        wp::int32 var_374;
        const wp::int32 var_375 = 3;
        wp::int32 var_376;
        wp::float32 var_377;
        wp::float32 var_378;
        const wp::int32 var_379 = 1;
        const wp::int32 var_380 = 3;
        wp::int32 var_381;
        wp::float32 var_382;
        wp::float32 var_383;
        const wp::int32 var_384 = 3;
        wp::int32 var_385;
        const wp::int32 var_386 = 3;
        wp::int32 var_387;
        wp::float32 var_388;
        wp::float32 var_389;
        const wp::int32 var_390 = 2;
        const wp::int32 var_391 = 3;
        wp::int32 var_392;
        wp::float32 var_393;
        wp::float32 var_394;
        const wp::int32 var_395 = 3;
        wp::int32 var_396;
        const wp::int32 var_397 = 3;
        wp::int32 var_398;
        wp::float32 var_399;
        wp::float32 var_400;
        const wp::int32 var_401 = 2;
        const wp::int32 var_402 = 0;
        const wp::int32 var_403 = 3;
        wp::int32 var_404;
        wp::float32 var_405;
        wp::float32 var_406;
        const wp::int32 var_407 = 3;
        wp::int32 var_408;
        const wp::int32 var_409 = 3;
        wp::int32 var_410;
        wp::float32 var_411;
        wp::float32 var_412;
        const wp::int32 var_413 = 1;
        const wp::int32 var_414 = 3;
        wp::int32 var_415;
        wp::float32 var_416;
        wp::float32 var_417;
        const wp::int32 var_418 = 3;
        wp::int32 var_419;
        const wp::int32 var_420 = 3;
        wp::int32 var_421;
        wp::float32 var_422;
        wp::float32 var_423;
        const wp::int32 var_424 = 2;
        const wp::int32 var_425 = 3;
        wp::int32 var_426;
        wp::float32 var_427;
        wp::float32 var_428;
        const wp::int32 var_429 = 3;
        wp::int32 var_430;
        const wp::int32 var_431 = 3;
        wp::int32 var_432;
        wp::float32 var_433;
        wp::float32 var_434;
        wp::shape_t* var_435;
        const wp::int32 var_436 = 0;
        wp::int32 var_437;
        wp::shape_t var_438;
        wp::int32 var_439;
        wp::float32* var_440;
        wp::float32 var_441;
        wp::float32 var_442;
        wp::shape_t* var_443;
        const wp::int32 var_444 = 0;
        wp::int32 var_445;
        wp::shape_t var_446;
        wp::int32 var_447;
        wp::float32* var_448;
        wp::float32 var_449;
        wp::float32 var_450;
        wp::shape_t* var_451;
        const wp::int32 var_452 = 0;
        wp::int32 var_453;
        wp::shape_t var_454;
        wp::int32 var_455;
        wp::vec_t<3, wp::float32>* var_456;
        wp::vec_t<3, wp::float32> var_457;
        wp::vec_t<3, wp::float32> var_458;
        bool var_459;
        const wp::float32 var_460 = 0.0;
        bool var_461;
        const wp::float32 var_462 = 0.0;
        bool var_463;
        wp::vec_t<3, wp::float32>* var_464;
        wp::vec_t<3, wp::float32> var_465;
        wp::vec_t<3, wp::float32> var_466;
        wp::range_t var_467;
        wp::int32 var_468;
        wp::int32 var_469;
        wp::int32 var_470;
        wp::int32* var_471;
        bool var_472;
        wp::int32 var_473;
        bool* var_474;
        bool var_475;
        wp::vec_t<3, wp::float32>* var_476;
        wp::vec_t<3, wp::float32> var_477;
        wp::vec_t<3, wp::float32> var_478;
        wp::vec_t<6, wp::float32>* var_479;
        wp::vec_t<6, wp::float32> var_480;
        wp::vec_t<6, wp::float32> var_481;
        wp::vec_t<3, wp::float32> var_482;
        wp::vec_t<3, wp::float32> var_483;
        wp::vec_t<3, wp::float32> var_484;
        wp::vec_t<3, wp::float32> var_485;
        wp::vec_t<3, wp::float32> var_486;
        wp::int32* var_487;
        wp::int32 var_488;
        wp::int32 var_489;
        wp::int32* var_490;
        wp::int32 var_491;
        wp::int32 var_492;
        wp::range_t var_493;
        wp::int32 var_494;
        wp::int32 var_495;
        const wp::int32 var_496 = 0;
        wp::float32* var_497;
        wp::float32 var_498;
        wp::float32 var_499;
        const wp::float32 var_500 = 0.0;
        bool var_501;
        wp::shape_t* var_502;
        const wp::int32 var_503 = 0;
        wp::int32 var_504;
        wp::shape_t var_505;
        wp::int32 var_506;
        wp::vec_t<3, wp::float32>* var_507;
        wp::vec_t<3, wp::float32> var_508;
        wp::vec_t<3, wp::float32> var_509;
        wp::int32* var_510;
        wp::vec_t<3, wp::float32> var_511;
        wp::int32 var_512;
        wp::mat_t<3, 3, wp::float32>* var_513;
        wp::mat_t<3, 3, wp::float32> var_514;
        wp::mat_t<3, 3, wp::float32> var_515;
        wp::mat_t<3, 3, wp::float32> var_516;
        wp::vec_t<3, wp::float32>* var_517;
        wp::vec_t<3, wp::float32> var_518;
        wp::vec_t<3, wp::float32> var_519;
        wp::vec_t<3, wp::float32> var_520;
        wp::vec_t<3, wp::float32> var_521;
        wp::vec_t<3, wp::float32> var_522;
        wp::vec_t<3, wp::float32> var_523;
        wp::vec_t<3, wp::float32> var_524;
        bool var_525;
        const wp::int32 var_526 = 0;
        wp::float32 var_527;
        const wp::float32 var_528 = 0.0;
        bool var_529;
        const wp::int32 var_530 = 1;
        wp::float32 var_531;
        const wp::float32 var_532 = 0.0;
        bool var_533;
        const wp::int32 var_534 = 2;
        wp::float32 var_535;
        const wp::float32 var_536 = 0.0;
        bool var_537;
        wp::vec_t<3, wp::float32> var_538;
        wp::vec_t<3, wp::float32> var_539;
        wp::vec_t<3, wp::float32> var_540;
        wp::vec_t<3, wp::float32> var_541;
        wp::vec_t<3, wp::float32> var_542;
        const wp::int32 var_543 = 1;
        wp::float32* var_544;
        wp::float32 var_545;
        wp::float32 var_546;
        const wp::int32 var_547 = 2;
        wp::float32* var_548;
        wp::float32 var_549;
        wp::float32 var_550;
        const wp::int32 var_551 = 3;
        wp::float32* var_552;
        wp::float32 var_553;
        wp::float32 var_554;
        const wp::int32 var_555 = 4;
        wp::float32* var_556;
        wp::float32 var_557;
        wp::float32 var_558;
        const wp::int32 var_559 = 5;
        wp::float32* var_560;
        wp::float32 var_561;
        wp::float32 var_562;
        const wp::int32 var_563 = 6;
        wp::float32* var_564;
        const wp::int32 var_565 = 7;
        wp::float32* var_566;
        const wp::int32 var_567 = 8;
        wp::float32* var_568;
        wp::vec_t<3, wp::float32> var_569;
        wp::float32 var_570;
        wp::float32 var_571;
        wp::float32 var_572;
        const wp::int32 var_573 = 9;
        wp::float32* var_574;
        const wp::int32 var_575 = 10;
        wp::float32* var_576;
        const wp::int32 var_577 = 11;
        wp::float32* var_578;
        wp::vec_t<3, wp::float32> var_579;
        wp::float32 var_580;
        wp::float32 var_581;
        wp::float32 var_582;
        const bool var_583 = false;
        wp::mat_t<3, 3, wp::float32> var_584;
        wp::mat_t<3, 3, wp::float32> var_585;
        wp::mat_t<3, 3, wp::float32> var_586;
        wp::mat_t<3, 3, wp::float32> var_587;
        wp::vec_t<3, wp::float32> var_588;
        wp::mat_t<6, 6, wp::float32> var_589;
        const wp::int32 var_590 = 0;
        const wp::int32 var_591 = 0;
        wp::float32 var_592;
        wp::float32 var_593;
        const wp::int32 var_594 = 1;
        wp::float32 var_595;
        wp::float32 var_596;
        const wp::int32 var_597 = 2;
        wp::float32 var_598;
        wp::float32 var_599;
        const wp::int32 var_600 = 3;
        wp::float32 var_601;
        wp::float32 var_602;
        const wp::int32 var_603 = 4;
        wp::float32 var_604;
        wp::float32 var_605;
        const wp::int32 var_606 = 5;
        wp::float32 var_607;
        wp::float32 var_608;
        const wp::int32 var_609 = 1;
        const wp::int32 var_610 = 0;
        wp::float32 var_611;
        wp::float32 var_612;
        const wp::int32 var_613 = 1;
        wp::float32 var_614;
        wp::float32 var_615;
        const wp::int32 var_616 = 2;
        wp::float32 var_617;
        wp::float32 var_618;
        const wp::int32 var_619 = 3;
        wp::float32 var_620;
        wp::float32 var_621;
        const wp::int32 var_622 = 4;
        wp::float32 var_623;
        wp::float32 var_624;
        const wp::int32 var_625 = 5;
        wp::float32 var_626;
        wp::float32 var_627;
        const wp::int32 var_628 = 2;
        const wp::int32 var_629 = 0;
        wp::float32 var_630;
        wp::float32 var_631;
        const wp::int32 var_632 = 1;
        wp::float32 var_633;
        wp::float32 var_634;
        const wp::int32 var_635 = 2;
        wp::float32 var_636;
        wp::float32 var_637;
        const wp::int32 var_638 = 3;
        wp::float32 var_639;
        wp::float32 var_640;
        const wp::int32 var_641 = 4;
        wp::float32 var_642;
        wp::float32 var_643;
        const wp::int32 var_644 = 5;
        wp::float32 var_645;
        wp::float32 var_646;
        const wp::int32 var_647 = 3;
        const wp::int32 var_648 = 0;
        wp::float32 var_649;
        wp::float32 var_650;
        const wp::int32 var_651 = 1;
        wp::float32 var_652;
        wp::float32 var_653;
        const wp::int32 var_654 = 2;
        wp::float32 var_655;
        wp::float32 var_656;
        const wp::int32 var_657 = 3;
        wp::float32 var_658;
        wp::float32 var_659;
        const wp::int32 var_660 = 4;
        wp::float32 var_661;
        wp::float32 var_662;
        const wp::int32 var_663 = 5;
        wp::float32 var_664;
        wp::float32 var_665;
        const wp::int32 var_666 = 4;
        const wp::int32 var_667 = 0;
        wp::float32 var_668;
        wp::float32 var_669;
        const wp::int32 var_670 = 1;
        wp::float32 var_671;
        wp::float32 var_672;
        const wp::int32 var_673 = 2;
        wp::float32 var_674;
        wp::float32 var_675;
        const wp::int32 var_676 = 3;
        wp::float32 var_677;
        wp::float32 var_678;
        const wp::int32 var_679 = 4;
        wp::float32 var_680;
        wp::float32 var_681;
        const wp::int32 var_682 = 5;
        wp::float32 var_683;
        wp::float32 var_684;
        const wp::int32 var_685 = 5;
        const wp::int32 var_686 = 0;
        wp::float32 var_687;
        wp::float32 var_688;
        const wp::int32 var_689 = 1;
        wp::float32 var_690;
        wp::float32 var_691;
        const wp::int32 var_692 = 2;
        wp::float32 var_693;
        wp::float32 var_694;
        const wp::int32 var_695 = 3;
        wp::float32 var_696;
        wp::float32 var_697;
        const wp::int32 var_698 = 4;
        wp::float32 var_699;
        wp::float32 var_700;
        const wp::int32 var_701 = 5;
        wp::float32 var_702;
        wp::float32 var_703;
        bool var_704;
        bool var_705;
        wp::shape_t* var_706;
        const wp::int32 var_707 = 0;
        wp::int32 var_708;
        wp::shape_t var_709;
        wp::int32 var_710;
        wp::float32* var_711;
        const wp::float32 var_712 = 1e-15;
        bool var_713;
        wp::float32 var_714;
        wp::vec_t<3, wp::float32>* var_715;
        wp::vec_t<3, wp::float32> var_716;
        wp::vec_t<3, wp::float32> var_717;
        wp::mat_t<3, 3, wp::float32>* var_718;
        wp::mat_t<3, 3, wp::float32> var_719;
        wp::mat_t<3, 3, wp::float32> var_720;
        wp::vec_t<6, wp::float32>* var_721;
        wp::vec_t<6, wp::float32> var_722;
        wp::vec_t<6, wp::float32> var_723;
        const wp::int32 var_724 = 0;
        wp::float32 var_725;
        const wp::int32 var_726 = 1;
        wp::float32 var_727;
        const wp::int32 var_728 = 2;
        wp::float32 var_729;
        wp::vec_t<3, wp::float32> var_730;
        const wp::int32 var_731 = 3;
        wp::float32 var_732;
        const wp::int32 var_733 = 4;
        wp::float32 var_734;
        const wp::int32 var_735 = 5;
        wp::float32 var_736;
        wp::vec_t<3, wp::float32> var_737;
        wp::vec_t<3, wp::float32> var_738;
        wp::vec_t<3, wp::float32> var_739;
        wp::vec_t<3, wp::float32> var_740;
        wp::mat_t<3, 3, wp::float32> var_741;
        wp::vec_t<3, wp::float32> var_742;
        wp::vec_t<3, wp::float32> var_743;
        wp::vec_t<3, wp::float32> var_744;
        const wp::int32 var_745 = 0;
        wp::float32 var_746;
        const wp::int32 var_747 = 1;
        wp::float32 var_748;
        const wp::int32 var_749 = 2;
        wp::float32 var_750;
        const wp::int32 var_751 = 0;
        wp::float32 var_752;
        const wp::int32 var_753 = 1;
        wp::float32 var_754;
        const wp::int32 var_755 = 2;
        wp::float32 var_756;
        wp::vec_t<6, wp::float32> var_757;
        wp::mat_t<6, 6, wp::float32> var_758;
        const wp::int32 var_759 = 0;
        const wp::int32 var_760 = 0;
        wp::float32 var_761;
        const wp::int32 var_762 = 1;
        const wp::int32 var_763 = 1;
        wp::float32 var_764;
        const wp::int32 var_765 = 2;
        const wp::int32 var_766 = 2;
        wp::float32 var_767;
        wp::vec_t<3, wp::float32> var_768;
        wp::mat_t<3, 3, wp::float32> var_769;
        const wp::int32 var_770 = 3;
        const wp::int32 var_771 = 3;
        wp::float32 var_772;
        const wp::int32 var_773 = 4;
        const wp::int32 var_774 = 4;
        wp::float32 var_775;
        const wp::int32 var_776 = 5;
        const wp::int32 var_777 = 5;
        wp::float32 var_778;
        wp::vec_t<3, wp::float32> var_779;
        wp::mat_t<3, 3, wp::float32> var_780;
        const wp::float32 var_781 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_782;
        wp::vec_t<3, wp::float32> var_783;
        wp::mat_t<6, 6, wp::float32> var_784;
        const wp::int32 var_785 = 0;
        const wp::int32 var_786 = 0;
        wp::float32 var_787;
        wp::float32 var_788;
        const wp::int32 var_789 = 1;
        wp::float32 var_790;
        wp::float32 var_791;
        const wp::int32 var_792 = 2;
        wp::float32 var_793;
        wp::float32 var_794;
        const wp::int32 var_795 = 3;
        wp::float32 var_796;
        wp::float32 var_797;
        const wp::int32 var_798 = 4;
        wp::float32 var_799;
        wp::float32 var_800;
        const wp::int32 var_801 = 5;
        wp::float32 var_802;
        wp::float32 var_803;
        const wp::int32 var_804 = 1;
        const wp::int32 var_805 = 0;
        wp::float32 var_806;
        wp::float32 var_807;
        const wp::int32 var_808 = 1;
        wp::float32 var_809;
        wp::float32 var_810;
        const wp::int32 var_811 = 2;
        wp::float32 var_812;
        wp::float32 var_813;
        const wp::int32 var_814 = 3;
        wp::float32 var_815;
        wp::float32 var_816;
        const wp::int32 var_817 = 4;
        wp::float32 var_818;
        wp::float32 var_819;
        const wp::int32 var_820 = 5;
        wp::float32 var_821;
        wp::float32 var_822;
        const wp::int32 var_823 = 2;
        const wp::int32 var_824 = 0;
        wp::float32 var_825;
        wp::float32 var_826;
        const wp::int32 var_827 = 1;
        wp::float32 var_828;
        wp::float32 var_829;
        const wp::int32 var_830 = 2;
        wp::float32 var_831;
        wp::float32 var_832;
        const wp::int32 var_833 = 3;
        wp::float32 var_834;
        wp::float32 var_835;
        const wp::int32 var_836 = 4;
        wp::float32 var_837;
        wp::float32 var_838;
        const wp::int32 var_839 = 5;
        wp::float32 var_840;
        wp::float32 var_841;
        const wp::int32 var_842 = 3;
        const wp::int32 var_843 = 0;
        wp::float32 var_844;
        wp::float32 var_845;
        const wp::int32 var_846 = 1;
        wp::float32 var_847;
        wp::float32 var_848;
        const wp::int32 var_849 = 2;
        wp::float32 var_850;
        wp::float32 var_851;
        const wp::int32 var_852 = 3;
        wp::float32 var_853;
        wp::float32 var_854;
        const wp::int32 var_855 = 4;
        wp::float32 var_856;
        wp::float32 var_857;
        const wp::int32 var_858 = 5;
        wp::float32 var_859;
        wp::float32 var_860;
        const wp::int32 var_861 = 4;
        const wp::int32 var_862 = 0;
        wp::float32 var_863;
        wp::float32 var_864;
        const wp::int32 var_865 = 1;
        wp::float32 var_866;
        wp::float32 var_867;
        const wp::int32 var_868 = 2;
        wp::float32 var_869;
        wp::float32 var_870;
        const wp::int32 var_871 = 3;
        wp::float32 var_872;
        wp::float32 var_873;
        const wp::int32 var_874 = 4;
        wp::float32 var_875;
        wp::float32 var_876;
        const wp::int32 var_877 = 5;
        wp::float32 var_878;
        wp::float32 var_879;
        const wp::int32 var_880 = 5;
        const wp::int32 var_881 = 0;
        wp::float32 var_882;
        wp::float32 var_883;
        const wp::int32 var_884 = 1;
        wp::float32 var_885;
        wp::float32 var_886;
        const wp::int32 var_887 = 2;
        wp::float32 var_888;
        wp::float32 var_889;
        const wp::int32 var_890 = 3;
        wp::float32 var_891;
        wp::float32 var_892;
        const wp::int32 var_893 = 4;
        wp::float32 var_894;
        wp::float32 var_895;
        const wp::int32 var_896 = 5;
        wp::float32 var_897;
        wp::float32 var_898;
        wp::int32 var_899;
        wp::int32 var_900;
        bool var_901;
        wp::int32 var_902;
        bool var_903;
        wp::int32 var_904;
        bool var_905;
        wp::vec_t<6, wp::float32> var_906;
        bool var_907;
        wp::vec_t<3, wp::float32> var_908;
        bool var_909;
        wp::mat_t<3, 3, wp::float32> var_910;
        bool var_911;
        wp::mat_t<3, 3, wp::float32> var_912;
        bool var_913;
        wp::mat_t<6, 6, wp::float32> var_914;
        bool var_915;
        wp::int32 var_916;
        wp::int32 var_917;
        wp::mat_t<6, 6, wp::float32> var_918;
        wp::vec_t<6, wp::int32> var_919;
        bool var_920;
        const wp::float32 var_921 = 0.0;
        wp::vec_t<6, wp::float32> var_922;
        const wp::int32 var_923 = 0;
        wp::int32 var_924;
        wp::float32* var_925;
        wp::float32 var_926;
        wp::float32 var_927;
        const wp::int32 var_928 = 1;
        wp::int32 var_929;
        wp::float32* var_930;
        wp::float32 var_931;
        wp::float32 var_932;
        const wp::int32 var_933 = 2;
        wp::int32 var_934;
        wp::float32* var_935;
        wp::float32 var_936;
        wp::float32 var_937;
        const wp::int32 var_938 = 3;
        wp::int32 var_939;
        wp::float32* var_940;
        wp::float32 var_941;
        wp::float32 var_942;
        const wp::int32 var_943 = 4;
        wp::int32 var_944;
        wp::float32* var_945;
        wp::float32 var_946;
        wp::float32 var_947;
        const wp::int32 var_948 = 5;
        wp::int32 var_949;
        wp::float32* var_950;
        wp::float32 var_951;
        wp::float32 var_952;
        wp::vec_t<6, wp::float32> var_953;
        const wp::int32 var_954 = 0;
        wp::float32 var_955;
        wp::int32 var_956;
        const wp::int32 var_957 = 1;
        wp::float32 var_958;
        wp::int32 var_959;
        const wp::int32 var_960 = 2;
        wp::float32 var_961;
        wp::int32 var_962;
        const wp::int32 var_963 = 3;
        wp::float32 var_964;
        wp::int32 var_965;
        const wp::int32 var_966 = 4;
        wp::float32 var_967;
        wp::int32 var_968;
        const wp::int32 var_969 = 5;
        wp::float32 var_970;
        wp::int32 var_971;
        wp::int32 var_972;
        //---------
        // forward
        // def _implicit_free_body_solve(                                                         <L 833>
        // worldid, freeid = wp.tid()                                                             <L 879>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_freeadr[freeid]                                                          <L 880>
        var_2 = wp::address(var_body_freeadr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dof_adr = body_dofadr[bodyid]                                                          <L 882>
        var_5 = wp::address(var_body_dofadr, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // treeid = dof_treeid[dof_adr]                                                           <L 885>
        var_8 = wp::address(var_dof_treeid, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if (opt_enableflags & EnableBit.SLEEP) and (tree_awake_in[worldid, treeid] == 0):       <L 886>
        var_13 = wp::bit_and(var_opt_enableflags, var_12);
        var_11 = var_13;
        if (var_11) {
            var_14 = wp::address(var_tree_awake_in, var_0, var_9);
            var_17 = wp::load(var_14);
            var_16 = (var_17 == var_15);
            var_11 = var_11 && var_16;
        }
        if (var_11) {
            // return                                                                             <L 887>
            continue;
        }
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 889>
        var_18 = &(var_opt_timestep.shape);
        var_21 = wp::load(var_18);
        var_20 = wp::extract(var_21, var_19);
        var_22 = wp::mod(var_0, var_20);
        var_23 = wp::address(var_opt_timestep, var_22);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // A = mat66(0.0)                                                                         <L 892>
        var_27 = wp::mat_t<6, 6, wp::float32>(var_26);
        // for r in range(6):                                                                     <L 893>
        // row = dof_adr + r                                                                      <L 894>
        var_29 = wp::add(var_6, var_28);
        // start = M_rowadr[row]                                                                  <L 895>
        var_30 = wp::address(var_M_rowadr, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // nnz = M_rownnz[row]                                                                    <L 896>
        var_33 = wp::address(var_M_rownnz, var_29);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // for k in range(nnz):                                                                   <L 897>
        var_36 = wp::range(var_34);
        start_for_1:;
            if (iter_cmp(var_36) == 0) goto end_for_1;
            var_37 = wp::iter_next(var_36);
            // c = M_colind[start + k] - dof_adr                                                  <L 898>
            var_38 = wp::add(var_31, var_37);
            var_39 = wp::address(var_M_colind, var_38);
            var_41 = wp::load(var_39);
            var_40 = wp::sub(var_41, var_6);
            // val = M_in[worldid, start + k]                                                     <L 899>
            var_42 = wp::add(var_31, var_37);
            var_43 = wp::address(var_M_in, var_0, var_42);
            var_45 = wp::load(var_43);
            var_44 = wp::copy(var_45);
            // A[r, c] = val                                                                      <L 900>
            wp::assign_inplace(var_27, var_28, var_40, var_44);
            // A[c, r] = val                                                                      <L 901>
            wp::assign_inplace(var_27, var_40, var_28, var_44);
            goto start_for_1;
        end_for_1:;
        // row = dof_adr + r                                                                      <L 894>
        var_47 = wp::add(var_6, var_46);
        // start = M_rowadr[row]                                                                  <L 895>
        var_48 = wp::address(var_M_rowadr, var_47);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // nnz = M_rownnz[row]                                                                    <L 896>
        var_51 = wp::address(var_M_rownnz, var_47);
        var_53 = wp::load(var_51);
        var_52 = wp::copy(var_53);
        // for k in range(nnz):                                                                   <L 897>
        var_54 = wp::range(var_52);
        start_for_3:;
            if (iter_cmp(var_54) == 0) goto end_for_3;
            var_55 = wp::iter_next(var_54);
            // c = M_colind[start + k] - dof_adr                                                  <L 898>
            var_56 = wp::add(var_49, var_55);
            var_57 = wp::address(var_M_colind, var_56);
            var_59 = wp::load(var_57);
            var_58 = wp::sub(var_59, var_6);
            // val = M_in[worldid, start + k]                                                     <L 899>
            var_60 = wp::add(var_49, var_55);
            var_61 = wp::address(var_M_in, var_0, var_60);
            var_63 = wp::load(var_61);
            var_62 = wp::copy(var_63);
            // A[r, c] = val                                                                      <L 900>
            wp::assign_inplace(var_27, var_46, var_58, var_62);
            // A[c, r] = val                                                                      <L 901>
            wp::assign_inplace(var_27, var_58, var_46, var_62);
            wp::assign(var_40, var_58);
            wp::assign(var_44, var_62);
            goto start_for_3;
        end_for_3:;
        // row = dof_adr + r                                                                      <L 894>
        var_65 = wp::add(var_6, var_64);
        // start = M_rowadr[row]                                                                  <L 895>
        var_66 = wp::address(var_M_rowadr, var_65);
        var_68 = wp::load(var_66);
        var_67 = wp::copy(var_68);
        // nnz = M_rownnz[row]                                                                    <L 896>
        var_69 = wp::address(var_M_rownnz, var_65);
        var_71 = wp::load(var_69);
        var_70 = wp::copy(var_71);
        // for k in range(nnz):                                                                   <L 897>
        var_72 = wp::range(var_70);
        start_for_5:;
            if (iter_cmp(var_72) == 0) goto end_for_5;
            var_73 = wp::iter_next(var_72);
            // c = M_colind[start + k] - dof_adr                                                  <L 898>
            var_74 = wp::add(var_67, var_73);
            var_75 = wp::address(var_M_colind, var_74);
            var_77 = wp::load(var_75);
            var_76 = wp::sub(var_77, var_6);
            // val = M_in[worldid, start + k]                                                     <L 899>
            var_78 = wp::add(var_67, var_73);
            var_79 = wp::address(var_M_in, var_0, var_78);
            var_81 = wp::load(var_79);
            var_80 = wp::copy(var_81);
            // A[r, c] = val                                                                      <L 900>
            wp::assign_inplace(var_27, var_64, var_76, var_80);
            // A[c, r] = val                                                                      <L 901>
            wp::assign_inplace(var_27, var_76, var_64, var_80);
            wp::assign(var_40, var_76);
            wp::assign(var_44, var_80);
            goto start_for_5;
        end_for_5:;
        // row = dof_adr + r                                                                      <L 894>
        var_83 = wp::add(var_6, var_82);
        // start = M_rowadr[row]                                                                  <L 895>
        var_84 = wp::address(var_M_rowadr, var_83);
        var_86 = wp::load(var_84);
        var_85 = wp::copy(var_86);
        // nnz = M_rownnz[row]                                                                    <L 896>
        var_87 = wp::address(var_M_rownnz, var_83);
        var_89 = wp::load(var_87);
        var_88 = wp::copy(var_89);
        // for k in range(nnz):                                                                   <L 897>
        var_90 = wp::range(var_88);
        start_for_7:;
            if (iter_cmp(var_90) == 0) goto end_for_7;
            var_91 = wp::iter_next(var_90);
            // c = M_colind[start + k] - dof_adr                                                  <L 898>
            var_92 = wp::add(var_85, var_91);
            var_93 = wp::address(var_M_colind, var_92);
            var_95 = wp::load(var_93);
            var_94 = wp::sub(var_95, var_6);
            // val = M_in[worldid, start + k]                                                     <L 899>
            var_96 = wp::add(var_85, var_91);
            var_97 = wp::address(var_M_in, var_0, var_96);
            var_99 = wp::load(var_97);
            var_98 = wp::copy(var_99);
            // A[r, c] = val                                                                      <L 900>
            wp::assign_inplace(var_27, var_82, var_94, var_98);
            // A[c, r] = val                                                                      <L 901>
            wp::assign_inplace(var_27, var_94, var_82, var_98);
            wp::assign(var_40, var_94);
            wp::assign(var_44, var_98);
            goto start_for_7;
        end_for_7:;
        // row = dof_adr + r                                                                      <L 894>
        var_101 = wp::add(var_6, var_100);
        // start = M_rowadr[row]                                                                  <L 895>
        var_102 = wp::address(var_M_rowadr, var_101);
        var_104 = wp::load(var_102);
        var_103 = wp::copy(var_104);
        // nnz = M_rownnz[row]                                                                    <L 896>
        var_105 = wp::address(var_M_rownnz, var_101);
        var_107 = wp::load(var_105);
        var_106 = wp::copy(var_107);
        // for k in range(nnz):                                                                   <L 897>
        var_108 = wp::range(var_106);
        start_for_9:;
            if (iter_cmp(var_108) == 0) goto end_for_9;
            var_109 = wp::iter_next(var_108);
            // c = M_colind[start + k] - dof_adr                                                  <L 898>
            var_110 = wp::add(var_103, var_109);
            var_111 = wp::address(var_M_colind, var_110);
            var_113 = wp::load(var_111);
            var_112 = wp::sub(var_113, var_6);
            // val = M_in[worldid, start + k]                                                     <L 899>
            var_114 = wp::add(var_103, var_109);
            var_115 = wp::address(var_M_in, var_0, var_114);
            var_117 = wp::load(var_115);
            var_116 = wp::copy(var_117);
            // A[r, c] = val                                                                      <L 900>
            wp::assign_inplace(var_27, var_100, var_112, var_116);
            // A[c, r] = val                                                                      <L 901>
            wp::assign_inplace(var_27, var_112, var_100, var_116);
            wp::assign(var_40, var_112);
            wp::assign(var_44, var_116);
            goto start_for_9;
        end_for_9:;
        // row = dof_adr + r                                                                      <L 894>
        var_119 = wp::add(var_6, var_118);
        // start = M_rowadr[row]                                                                  <L 895>
        var_120 = wp::address(var_M_rowadr, var_119);
        var_122 = wp::load(var_120);
        var_121 = wp::copy(var_122);
        // nnz = M_rownnz[row]                                                                    <L 896>
        var_123 = wp::address(var_M_rownnz, var_119);
        var_125 = wp::load(var_123);
        var_124 = wp::copy(var_125);
        // for k in range(nnz):                                                                   <L 897>
        var_126 = wp::range(var_124);
        start_for_11:;
            if (iter_cmp(var_126) == 0) goto end_for_11;
            var_127 = wp::iter_next(var_126);
            // c = M_colind[start + k] - dof_adr                                                  <L 898>
            var_128 = wp::add(var_121, var_127);
            var_129 = wp::address(var_M_colind, var_128);
            var_131 = wp::load(var_129);
            var_130 = wp::sub(var_131, var_6);
            // val = M_in[worldid, start + k]                                                     <L 899>
            var_132 = wp::add(var_121, var_127);
            var_133 = wp::address(var_M_in, var_0, var_132);
            var_135 = wp::load(var_133);
            var_134 = wp::copy(var_135);
            // A[r, c] = val                                                                      <L 900>
            wp::assign_inplace(var_27, var_118, var_130, var_134);
            // A[c, r] = val                                                                      <L 901>
            wp::assign_inplace(var_27, var_130, var_118, var_134);
            wp::assign(var_40, var_130);
            wp::assign(var_44, var_134);
            goto start_for_11;
        end_for_11:;
        // if not (opt_disableflags & DisableBit.DAMPER):                                         <L 904>
        var_137 = wp::bit_and(var_opt_disableflags, var_136);
        var_138 = wp::unot(var_137);
        if (var_138) {
            // for r in range(6):                                                                 <L 905>
            // dof = dof_adr + r                                                                  <L 906>
            var_140 = wp::add(var_6, var_139);
            // damp = dof_damping[worldid % dof_damping.shape[0], dof]                            <L 907>
            var_141 = &(var_dof_damping.shape);
            var_144 = wp::load(var_141);
            var_143 = wp::extract(var_144, var_142);
            var_145 = wp::mod(var_0, var_143);
            var_146 = wp::address(var_dof_damping, var_145, var_140);
            var_148 = wp::load(var_146);
            var_147 = wp::copy(var_148);
            // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dof]                   <L 908>
            var_149 = &(var_dof_dampingpoly.shape);
            var_152 = wp::load(var_149);
            var_151 = wp::extract(var_152, var_150);
            var_153 = wp::mod(var_0, var_151);
            var_154 = wp::address(var_dof_dampingpoly, var_153, var_140);
            var_156 = wp::load(var_154);
            var_155 = wp::copy(var_156);
            // v = qvel_in[worldid, dof]                                                          <L 909>
            var_157 = wp::address(var_qvel_in, var_0, var_140);
            var_159 = wp::load(var_157);
            var_158 = wp::copy(var_159);
            // A[r, r] += timestep * util_misc._poly_force_deriv(damp, dpoly, v, 1)               <L 910>
            var_161 = _poly_force_deriv_0(var_147, var_155, var_158, var_160);
            var_162 = wp::mul(var_24, var_161);
            wp::add_inplace(var_27, var_139, var_139, var_162);
            // dof = dof_adr + r                                                                  <L 906>
            var_164 = wp::add(var_6, var_163);
            // damp = dof_damping[worldid % dof_damping.shape[0], dof]                            <L 907>
            var_165 = &(var_dof_damping.shape);
            var_168 = wp::load(var_165);
            var_167 = wp::extract(var_168, var_166);
            var_169 = wp::mod(var_0, var_167);
            var_170 = wp::address(var_dof_damping, var_169, var_164);
            var_172 = wp::load(var_170);
            var_171 = wp::copy(var_172);
            // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dof]                   <L 908>
            var_173 = &(var_dof_dampingpoly.shape);
            var_176 = wp::load(var_173);
            var_175 = wp::extract(var_176, var_174);
            var_177 = wp::mod(var_0, var_175);
            var_178 = wp::address(var_dof_dampingpoly, var_177, var_164);
            var_180 = wp::load(var_178);
            var_179 = wp::copy(var_180);
            // v = qvel_in[worldid, dof]                                                          <L 909>
            var_181 = wp::address(var_qvel_in, var_0, var_164);
            var_183 = wp::load(var_181);
            var_182 = wp::copy(var_183);
            // A[r, r] += timestep * util_misc._poly_force_deriv(damp, dpoly, v, 1)               <L 910>
            var_185 = _poly_force_deriv_0(var_171, var_179, var_182, var_184);
            var_186 = wp::mul(var_24, var_185);
            wp::add_inplace(var_27, var_163, var_163, var_186);
            // dof = dof_adr + r                                                                  <L 906>
            var_188 = wp::add(var_6, var_187);
            // damp = dof_damping[worldid % dof_damping.shape[0], dof]                            <L 907>
            var_189 = &(var_dof_damping.shape);
            var_192 = wp::load(var_189);
            var_191 = wp::extract(var_192, var_190);
            var_193 = wp::mod(var_0, var_191);
            var_194 = wp::address(var_dof_damping, var_193, var_188);
            var_196 = wp::load(var_194);
            var_195 = wp::copy(var_196);
            // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dof]                   <L 908>
            var_197 = &(var_dof_dampingpoly.shape);
            var_200 = wp::load(var_197);
            var_199 = wp::extract(var_200, var_198);
            var_201 = wp::mod(var_0, var_199);
            var_202 = wp::address(var_dof_dampingpoly, var_201, var_188);
            var_204 = wp::load(var_202);
            var_203 = wp::copy(var_204);
            // v = qvel_in[worldid, dof]                                                          <L 909>
            var_205 = wp::address(var_qvel_in, var_0, var_188);
            var_207 = wp::load(var_205);
            var_206 = wp::copy(var_207);
            // A[r, r] += timestep * util_misc._poly_force_deriv(damp, dpoly, v, 1)               <L 910>
            var_209 = _poly_force_deriv_0(var_195, var_203, var_206, var_208);
            var_210 = wp::mul(var_24, var_209);
            wp::add_inplace(var_27, var_187, var_187, var_210);
            // dof = dof_adr + r                                                                  <L 906>
            var_212 = wp::add(var_6, var_211);
            // damp = dof_damping[worldid % dof_damping.shape[0], dof]                            <L 907>
            var_213 = &(var_dof_damping.shape);
            var_216 = wp::load(var_213);
            var_215 = wp::extract(var_216, var_214);
            var_217 = wp::mod(var_0, var_215);
            var_218 = wp::address(var_dof_damping, var_217, var_212);
            var_220 = wp::load(var_218);
            var_219 = wp::copy(var_220);
            // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dof]                   <L 908>
            var_221 = &(var_dof_dampingpoly.shape);
            var_224 = wp::load(var_221);
            var_223 = wp::extract(var_224, var_222);
            var_225 = wp::mod(var_0, var_223);
            var_226 = wp::address(var_dof_dampingpoly, var_225, var_212);
            var_228 = wp::load(var_226);
            var_227 = wp::copy(var_228);
            // v = qvel_in[worldid, dof]                                                          <L 909>
            var_229 = wp::address(var_qvel_in, var_0, var_212);
            var_231 = wp::load(var_229);
            var_230 = wp::copy(var_231);
            // A[r, r] += timestep * util_misc._poly_force_deriv(damp, dpoly, v, 1)               <L 910>
            var_233 = _poly_force_deriv_0(var_219, var_227, var_230, var_232);
            var_234 = wp::mul(var_24, var_233);
            wp::add_inplace(var_27, var_211, var_211, var_234);
            // dof = dof_adr + r                                                                  <L 906>
            var_236 = wp::add(var_6, var_235);
            // damp = dof_damping[worldid % dof_damping.shape[0], dof]                            <L 907>
            var_237 = &(var_dof_damping.shape);
            var_240 = wp::load(var_237);
            var_239 = wp::extract(var_240, var_238);
            var_241 = wp::mod(var_0, var_239);
            var_242 = wp::address(var_dof_damping, var_241, var_236);
            var_244 = wp::load(var_242);
            var_243 = wp::copy(var_244);
            // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dof]                   <L 908>
            var_245 = &(var_dof_dampingpoly.shape);
            var_248 = wp::load(var_245);
            var_247 = wp::extract(var_248, var_246);
            var_249 = wp::mod(var_0, var_247);
            var_250 = wp::address(var_dof_dampingpoly, var_249, var_236);
            var_252 = wp::load(var_250);
            var_251 = wp::copy(var_252);
            // v = qvel_in[worldid, dof]                                                          <L 909>
            var_253 = wp::address(var_qvel_in, var_0, var_236);
            var_255 = wp::load(var_253);
            var_254 = wp::copy(var_255);
            // A[r, r] += timestep * util_misc._poly_force_deriv(damp, dpoly, v, 1)               <L 910>
            var_257 = _poly_force_deriv_0(var_243, var_251, var_254, var_256);
            var_258 = wp::mul(var_24, var_257);
            wp::add_inplace(var_27, var_235, var_235, var_258);
            // dof = dof_adr + r                                                                  <L 906>
            var_260 = wp::add(var_6, var_259);
            // damp = dof_damping[worldid % dof_damping.shape[0], dof]                            <L 907>
            var_261 = &(var_dof_damping.shape);
            var_264 = wp::load(var_261);
            var_263 = wp::extract(var_264, var_262);
            var_265 = wp::mod(var_0, var_263);
            var_266 = wp::address(var_dof_damping, var_265, var_260);
            var_268 = wp::load(var_266);
            var_267 = wp::copy(var_268);
            // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dof]                   <L 908>
            var_269 = &(var_dof_dampingpoly.shape);
            var_272 = wp::load(var_269);
            var_271 = wp::extract(var_272, var_270);
            var_273 = wp::mod(var_0, var_271);
            var_274 = wp::address(var_dof_dampingpoly, var_273, var_260);
            var_276 = wp::load(var_274);
            var_275 = wp::copy(var_276);
            // v = qvel_in[worldid, dof]                                                          <L 909>
            var_277 = wp::address(var_qvel_in, var_0, var_260);
            var_279 = wp::load(var_277);
            var_278 = wp::copy(var_279);
            // A[r, r] += timestep * util_misc._poly_force_deriv(damp, dpoly, v, 1)               <L 910>
            var_281 = _poly_force_deriv_0(var_267, var_275, var_278, var_280);
            var_282 = wp::mul(var_24, var_281);
            wp::add_inplace(var_27, var_259, var_259, var_282);
        }
        var_283 = wp::where(var_138, var_259, var_118);
        // crb = crb_in[worldid, bodyid]                                                          <L 913>
        var_284 = wp::address(var_crb_in, var_0, var_3);
        var_286 = wp::load(var_284);
        var_285 = wp::copy(var_286);
        // subtree_mass = crb[9]                                                                  <L 914>
        var_288 = wp::extract(var_285, var_287);
        // R = xmat_in[worldid, bodyid]                                                           <L 915>
        var_289 = wp::address(var_xmat_in, var_0, var_3);
        var_291 = wp::load(var_289);
        var_290 = wp::copy(var_291);
        // Iw = wp.mat33(                                                                         <L 916>
        // crb[0],                                                                                <L 917>
        var_293 = wp::extract(var_285, var_292);
        // crb[3],                                                                                <L 918>
        var_295 = wp::extract(var_285, var_294);
        // crb[4],                                                                                <L 919>
        var_297 = wp::extract(var_285, var_296);
        // crb[3],                                                                                <L 920>
        var_299 = wp::extract(var_285, var_298);
        // crb[1],                                                                                <L 921>
        var_301 = wp::extract(var_285, var_300);
        // crb[5],                                                                                <L 922>
        var_303 = wp::extract(var_285, var_302);
        // crb[4],                                                                                <L 923>
        var_305 = wp::extract(var_285, var_304);
        // crb[5],                                                                                <L 924>
        var_307 = wp::extract(var_285, var_306);
        // crb[2],                                                                                <L 925>
        var_309 = wp::extract(var_285, var_308);
        var_310 = wp::mat_t<3, 3, wp::float32>(var_293, var_295, var_297, var_299, var_301, var_303, var_305, var_307, var_309);
        // s = subtree_com_in[worldid, bodyid] - xpos_in[worldid, bodyid]                         <L 927>
        var_311 = wp::address(var_subtree_com_in, var_0, var_3);
        var_312 = wp::address(var_xpos_in, var_0, var_3);
        var_314 = wp::load(var_311);
        var_315 = wp::load(var_312);
        var_313 = wp::sub(var_314, var_315);
        // qvel_rot = wp.vec3(                                                                    <L 928>
        // qvel_in[worldid, dof_adr + 3],                                                         <L 929>
        var_317 = wp::add(var_6, var_316);
        var_318 = wp::address(var_qvel_in, var_0, var_317);
        // qvel_in[worldid, dof_adr + 4],                                                         <L 930>
        var_320 = wp::add(var_6, var_319);
        var_321 = wp::address(var_qvel_in, var_0, var_320);
        // qvel_in[worldid, dof_adr + 5],                                                         <L 931>
        var_323 = wp::add(var_6, var_322);
        var_324 = wp::address(var_qvel_in, var_0, var_323);
        var_326 = wp::load(var_318);
        var_327 = wp::load(var_321);
        var_328 = wp::load(var_324);
        var_325 = wp::vec_t<3, wp::float32>(var_326, var_327, var_328);
        // lin, rot = math.free_bias_vel_blocks(subtree_mass, R, Iw, s, qvel_rot)                 <L 933>
        free_bias_vel_blocks_0(var_288, var_290, var_310, var_313, var_325, var_329, var_330);
        // h_mass = -timestep * subtree_mass                                                      <L 934>
        var_331 = wp::neg(var_24);
        var_332 = wp::mul(var_331, var_288);
        // for r in range(3):                                                                     <L 935>
        // for c in range(3):                                                                     <L 936>
        // A[r, 3 + c] += h_mass * lin[r, c]                                                      <L 937>
        var_336 = wp::add(var_335, var_334);
        var_337 = wp::extract(var_329, var_333, var_334);
        var_338 = wp::mul(var_332, var_337);
        wp::add_inplace(var_27, var_333, var_336, var_338);
        // A[3 + r, 3 + c] += timestep * rot[r, c]                                                <L 938>
        var_340 = wp::add(var_339, var_333);
        var_342 = wp::add(var_341, var_334);
        var_343 = wp::extract(var_330, var_333, var_334);
        var_344 = wp::mul(var_24, var_343);
        wp::add_inplace(var_27, var_340, var_342, var_344);
        // A[r, 3 + c] += h_mass * lin[r, c]                                                      <L 937>
        var_347 = wp::add(var_346, var_345);
        var_348 = wp::extract(var_329, var_333, var_345);
        var_349 = wp::mul(var_332, var_348);
        wp::add_inplace(var_27, var_333, var_347, var_349);
        // A[3 + r, 3 + c] += timestep * rot[r, c]                                                <L 938>
        var_351 = wp::add(var_350, var_333);
        var_353 = wp::add(var_352, var_345);
        var_354 = wp::extract(var_330, var_333, var_345);
        var_355 = wp::mul(var_24, var_354);
        wp::add_inplace(var_27, var_351, var_353, var_355);
        // A[r, 3 + c] += h_mass * lin[r, c]                                                      <L 937>
        var_358 = wp::add(var_357, var_356);
        var_359 = wp::extract(var_329, var_333, var_356);
        var_360 = wp::mul(var_332, var_359);
        wp::add_inplace(var_27, var_333, var_358, var_360);
        // A[3 + r, 3 + c] += timestep * rot[r, c]                                                <L 938>
        var_362 = wp::add(var_361, var_333);
        var_364 = wp::add(var_363, var_356);
        var_365 = wp::extract(var_330, var_333, var_356);
        var_366 = wp::mul(var_24, var_365);
        wp::add_inplace(var_27, var_362, var_364, var_366);
        // for c in range(3):                                                                     <L 936>
        // A[r, 3 + c] += h_mass * lin[r, c]                                                      <L 937>
        var_370 = wp::add(var_369, var_368);
        var_371 = wp::extract(var_329, var_367, var_368);
        var_372 = wp::mul(var_332, var_371);
        wp::add_inplace(var_27, var_367, var_370, var_372);
        // A[3 + r, 3 + c] += timestep * rot[r, c]                                                <L 938>
        var_374 = wp::add(var_373, var_367);
        var_376 = wp::add(var_375, var_368);
        var_377 = wp::extract(var_330, var_367, var_368);
        var_378 = wp::mul(var_24, var_377);
        wp::add_inplace(var_27, var_374, var_376, var_378);
        // A[r, 3 + c] += h_mass * lin[r, c]                                                      <L 937>
        var_381 = wp::add(var_380, var_379);
        var_382 = wp::extract(var_329, var_367, var_379);
        var_383 = wp::mul(var_332, var_382);
        wp::add_inplace(var_27, var_367, var_381, var_383);
        // A[3 + r, 3 + c] += timestep * rot[r, c]                                                <L 938>
        var_385 = wp::add(var_384, var_367);
        var_387 = wp::add(var_386, var_379);
        var_388 = wp::extract(var_330, var_367, var_379);
        var_389 = wp::mul(var_24, var_388);
        wp::add_inplace(var_27, var_385, var_387, var_389);
        // A[r, 3 + c] += h_mass * lin[r, c]                                                      <L 937>
        var_392 = wp::add(var_391, var_390);
        var_393 = wp::extract(var_329, var_367, var_390);
        var_394 = wp::mul(var_332, var_393);
        wp::add_inplace(var_27, var_367, var_392, var_394);
        // A[3 + r, 3 + c] += timestep * rot[r, c]                                                <L 938>
        var_396 = wp::add(var_395, var_367);
        var_398 = wp::add(var_397, var_390);
        var_399 = wp::extract(var_330, var_367, var_390);
        var_400 = wp::mul(var_24, var_399);
        wp::add_inplace(var_27, var_396, var_398, var_400);
        // for c in range(3):                                                                     <L 936>
        // A[r, 3 + c] += h_mass * lin[r, c]                                                      <L 937>
        var_404 = wp::add(var_403, var_402);
        var_405 = wp::extract(var_329, var_401, var_402);
        var_406 = wp::mul(var_332, var_405);
        wp::add_inplace(var_27, var_401, var_404, var_406);
        // A[3 + r, 3 + c] += timestep * rot[r, c]                                                <L 938>
        var_408 = wp::add(var_407, var_401);
        var_410 = wp::add(var_409, var_402);
        var_411 = wp::extract(var_330, var_401, var_402);
        var_412 = wp::mul(var_24, var_411);
        wp::add_inplace(var_27, var_408, var_410, var_412);
        // A[r, 3 + c] += h_mass * lin[r, c]                                                      <L 937>
        var_415 = wp::add(var_414, var_413);
        var_416 = wp::extract(var_329, var_401, var_413);
        var_417 = wp::mul(var_332, var_416);
        wp::add_inplace(var_27, var_401, var_415, var_417);
        // A[3 + r, 3 + c] += timestep * rot[r, c]                                                <L 938>
        var_419 = wp::add(var_418, var_401);
        var_421 = wp::add(var_420, var_413);
        var_422 = wp::extract(var_330, var_401, var_413);
        var_423 = wp::mul(var_24, var_422);
        wp::add_inplace(var_27, var_419, var_421, var_423);
        // A[r, 3 + c] += h_mass * lin[r, c]                                                      <L 937>
        var_426 = wp::add(var_425, var_424);
        var_427 = wp::extract(var_329, var_401, var_424);
        var_428 = wp::mul(var_332, var_427);
        wp::add_inplace(var_27, var_401, var_426, var_428);
        // A[3 + r, 3 + c] += timestep * rot[r, c]                                                <L 938>
        var_430 = wp::add(var_429, var_401);
        var_432 = wp::add(var_431, var_424);
        var_433 = wp::extract(var_330, var_401, var_424);
        var_434 = wp::mul(var_24, var_433);
        wp::add_inplace(var_27, var_430, var_432, var_434);
        // density = opt_density[worldid % opt_density.shape[0]]                                  <L 941>
        var_435 = &(var_opt_density.shape);
        var_438 = wp::load(var_435);
        var_437 = wp::extract(var_438, var_436);
        var_439 = wp::mod(var_0, var_437);
        var_440 = wp::address(var_opt_density, var_439);
        var_442 = wp::load(var_440);
        var_441 = wp::copy(var_442);
        // viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]                            <L 942>
        var_443 = &(var_opt_viscosity.shape);
        var_446 = wp::load(var_443);
        var_445 = wp::extract(var_446, var_444);
        var_447 = wp::mod(var_0, var_445);
        var_448 = wp::address(var_opt_viscosity, var_447);
        var_450 = wp::load(var_448);
        var_449 = wp::copy(var_450);
        // wind = opt_wind[worldid % opt_wind.shape[0]]                                           <L 943>
        var_451 = &(var_opt_wind.shape);
        var_454 = wp::load(var_451);
        var_453 = wp::extract(var_454, var_452);
        var_455 = wp::mod(var_0, var_453);
        var_456 = wp::address(var_opt_wind, var_455);
        var_458 = wp::load(var_456);
        var_457 = wp::copy(var_458);
        // if density > 0.0 or viscosity > 0.0:                                                   <L 945>
        var_461 = (var_441 > var_460);
        var_459 = var_461;
        if (!var_459) {
            var_463 = (var_449 > var_462);
            var_459 = var_459 || var_463;
        }
        if (var_459) {
            // subtree_root = subtree_com_in[worldid, bodyid]                                     <L 946>
            var_464 = wp::address(var_subtree_com_in, var_0, var_3);
            var_466 = wp::load(var_464);
            var_465 = wp::copy(var_466);
            // for b in range(bodyid, nbody):                                                     <L 948>
            var_467 = wp::range(var_3, var_nbody);
            var_468 = wp::copy(var_401);
            var_469 = wp::copy(var_424);
            start_for_13:;
                if (iter_cmp(var_467) == 0) goto end_for_13;
                var_470 = wp::iter_next(var_467);
                // if body_rootid[b] != bodyid:                                                   <L 949>
                var_471 = wp::address(var_body_rootid, var_470);
                var_473 = wp::load(var_471);
                var_472 = (var_473 != var_3);
                if (var_472) {
                    // break                                                                      <L 950>
                    goto end_for_13;
                }
                // if body_fluid_ellipsoid[b]:                                                    <L 952>
                var_474 = wp::address(var_body_fluid_ellipsoid, var_470);
                var_475 = wp::load(var_474);
                if (var_475) {
                    // xipos = xipos_in[worldid, b]                                               <L 953>
                    var_476 = wp::address(var_xipos_in, var_0, var_470);
                    var_478 = wp::load(var_476);
                    var_477 = wp::copy(var_478);
                    // cvel = cvel_in[worldid, b]                                                 <L 954>
                    var_479 = wp::address(var_cvel_in, var_0, var_470);
                    var_481 = wp::load(var_479);
                    var_480 = wp::copy(var_481);
                    // ang_global = wp.spatial_top(cvel)                                          <L 955>
                    var_482 = wp::spatial_top(var_480);
                    // lin_global = wp.spatial_bottom(cvel)                                       <L 956>
                    var_483 = wp::spatial_bottom(var_480);
                    // lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)          <L 957>
                    var_484 = wp::sub(var_477, var_465);
                    var_485 = wp::cross(var_484, var_482);
                    var_486 = wp::sub(var_483, var_485);
                    // geomadr = body_geomadr[b]                                                  <L 959>
                    var_487 = wp::address(var_body_geomadr, var_470);
                    var_489 = wp::load(var_487);
                    var_488 = wp::copy(var_489);
                    // geomnum = body_geomnum[b]                                                  <L 960>
                    var_490 = wp::address(var_body_geomnum, var_470);
                    var_492 = wp::load(var_490);
                    var_491 = wp::copy(var_492);
                    // for g in range(geomnum):                                                   <L 962>
                    var_493 = wp::range(var_491);
                    start_for_15:;
                        if (iter_cmp(var_493) == 0) goto end_for_15;
                        var_494 = wp::iter_next(var_493);
                        // geomid = geomadr + g                                                   <L 963>
                        var_495 = wp::add(var_488, var_494);
                        // coef = geom_fluid[geomid, 0]                                           <L 964>
                        var_497 = wp::address(var_geom_fluid, var_495, var_496);
                        var_499 = wp::load(var_497);
                        var_498 = wp::copy(var_499);
                        // if coef <= 0.0:                                                        <L 965>
                        var_501 = (var_498 <= var_500);
                        if (var_501) {
                            // continue                                                           <L 966>
                            goto start_for_15;
                        }
                        // size = geom_size[worldid % geom_size.shape[0], geomid]                 <L 968>
                        var_502 = &(var_geom_size.shape);
                        var_505 = wp::load(var_502);
                        var_504 = wp::extract(var_505, var_503);
                        var_506 = wp::mod(var_0, var_504);
                        var_507 = wp::address(var_geom_size, var_506, var_495);
                        var_509 = wp::load(var_507);
                        var_508 = wp::copy(var_509);
                        // semiaxes = passive.geom_semiaxes(size, geom_type[geomid])              <L 969>
                        var_510 = wp::address(var_geom_type, var_495);
                        var_512 = wp::load(var_510);
                        var_511 = geom_semiaxes_0(var_508, var_512);
                        // geom_rot = geom_xmat_in[worldid, geomid]                               <L 970>
                        var_513 = wp::address(var_geom_xmat_in, var_0, var_495);
                        var_515 = wp::load(var_513);
                        var_514 = wp::copy(var_515);
                        // geom_rotT = wp.transpose(geom_rot)                                     <L 971>
                        var_516 = wp::transpose(var_514);
                        // geom_pos = geom_xpos_in[worldid, geomid]                               <L 972>
                        var_517 = wp::address(var_geom_xpos_in, var_0, var_495);
                        var_519 = wp::load(var_517);
                        var_518 = wp::copy(var_519);
                        // lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)           <L 975>
                        var_520 = wp::sub(var_518, var_477);
                        var_521 = wp::cross(var_482, var_520);
                        var_522 = wp::add(var_486, var_521);
                        // l_ang = geom_rotT @ ang_global                                         <L 976>
                        var_523 = wp::mul(var_516, var_482);
                        // l_lin = geom_rotT @ lin_point                                          <L 977>
                        var_524 = wp::mul(var_516, var_522);
                        // if wind[0] != 0.0 or wind[1] != 0.0 or wind[2] != 0.0:                 <L 979>
                        var_527 = wp::extract(var_457, var_526);
                        var_529 = (var_527 != var_528);
                        var_525 = var_529;
                        if (!var_525) {
                            var_531 = wp::extract(var_457, var_530);
                            var_533 = (var_531 != var_532);
                            var_525 = var_525 || var_533;
                        }
                        if (!var_525) {
                            var_535 = wp::extract(var_457, var_534);
                            var_537 = (var_535 != var_536);
                            var_525 = var_525 || var_537;
                        }
                        if (var_525) {
                            // l_lin -= geom_rotT @ wind                                          <L 980>
                            var_538 = wp::mul(var_516, var_457);
                            var_539 = wp::sub(var_524, var_538);
                        }
                        var_540 = wp::where(var_525, var_539, var_524);
                        // ang_vel = l_ang                                                        <L 982>
                        var_541 = wp::copy(var_523);
                        // lin_vel = l_lin                                                        <L 983>
                        var_542 = wp::copy(var_540);
                        // blunt_drag_coef = geom_fluid[geomid, 1]                                <L 985>
                        var_544 = wp::address(var_geom_fluid, var_495, var_543);
                        var_546 = wp::load(var_544);
                        var_545 = wp::copy(var_546);
                        // slender_drag_coef = geom_fluid[geomid, 2]                              <L 986>
                        var_548 = wp::address(var_geom_fluid, var_495, var_547);
                        var_550 = wp::load(var_548);
                        var_549 = wp::copy(var_550);
                        // ang_drag_coef = geom_fluid[geomid, 3]                                  <L 987>
                        var_552 = wp::address(var_geom_fluid, var_495, var_551);
                        var_554 = wp::load(var_552);
                        var_553 = wp::copy(var_554);
                        // kutta_lift_coef = geom_fluid[geomid, 4]                                <L 988>
                        var_556 = wp::address(var_geom_fluid, var_495, var_555);
                        var_558 = wp::load(var_556);
                        var_557 = wp::copy(var_558);
                        // magnus_lift_coef = geom_fluid[geomid, 5]                               <L 989>
                        var_560 = wp::address(var_geom_fluid, var_495, var_559);
                        var_562 = wp::load(var_560);
                        var_561 = wp::copy(var_562);
                        // virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])       <L 990>
                        var_564 = wp::address(var_geom_fluid, var_495, var_563);
                        var_566 = wp::address(var_geom_fluid, var_495, var_565);
                        var_568 = wp::address(var_geom_fluid, var_495, var_567);
                        var_570 = wp::load(var_564);
                        var_571 = wp::load(var_566);
                        var_572 = wp::load(var_568);
                        var_569 = wp::vec_t<3, wp::float32>(var_570, var_571, var_572);
                        // virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])       <L 991>
                        var_574 = wp::address(var_geom_fluid, var_495, var_573);
                        var_576 = wp::address(var_geom_fluid, var_495, var_575);
                        var_578 = wp::address(var_geom_fluid, var_495, var_577);
                        var_580 = wp::load(var_574);
                        var_581 = wp::load(var_576);
                        var_582 = wp::load(var_578);
                        var_579 = wp::vec_t<3, wp::float32>(var_580, var_581, var_582);
                        // B00, B01, B10, B11 = derivative._geom_ellipsoid_fluid_B(               <L 994>
                        // semiaxes,                                                              <L 995>
                        // blunt_drag_coef,                                                       <L 996>
                        // slender_drag_coef,                                                     <L 997>
                        // ang_drag_coef,                                                         <L 998>
                        // kutta_lift_coef,                                                       <L 999>
                        // magnus_lift_coef,                                                      <L 1000>
                        // virtual_mass,                                                          <L 1001>
                        // virtual_inertia,                                                       <L 1002>
                        // ang_vel,                                                               <L 1003>
                        // lin_vel,                                                               <L 1004>
                        // density,                                                               <L 1005>
                        // viscosity,                                                             <L 1006>
                        // False,                                                                 <L 1007>
                        _geom_ellipsoid_fluid_B_0(var_511, var_545, var_549, var_553, var_557, var_561, var_569, var_579, var_541, var_542, var_441, var_449, var_583, var_584, var_585, var_586, var_587);
                        // offset = geom_pos - subtree_root                                       <L 1011>
                        var_588 = wp::sub(var_518, var_465);
                        // J_T_B_J = _project_spatial_B(cdof_in, worldid, dof_adr, geom_rotT, offset, B00, B01, B10, B11)       <L 1012>
                        var_589 = _project_spatial_B_0(var_cdof_in, var_0, var_6, var_516, var_588, var_584, var_585, var_586, var_587);
                        // for r in range(6):                                                     <L 1013>
                        // for c in range(6):                                                     <L 1014>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1015>
                        var_592 = wp::extract(var_589, var_590, var_591);
                        var_593 = wp::mul(var_24, var_592);
                        wp::sub_inplace(var_27, var_590, var_591, var_593);
                        var_595 = wp::extract(var_589, var_590, var_594);
                        var_596 = wp::mul(var_24, var_595);
                        wp::sub_inplace(var_27, var_590, var_594, var_596);
                        var_598 = wp::extract(var_589, var_590, var_597);
                        var_599 = wp::mul(var_24, var_598);
                        wp::sub_inplace(var_27, var_590, var_597, var_599);
                        var_601 = wp::extract(var_589, var_590, var_600);
                        var_602 = wp::mul(var_24, var_601);
                        wp::sub_inplace(var_27, var_590, var_600, var_602);
                        var_604 = wp::extract(var_589, var_590, var_603);
                        var_605 = wp::mul(var_24, var_604);
                        wp::sub_inplace(var_27, var_590, var_603, var_605);
                        var_607 = wp::extract(var_589, var_590, var_606);
                        var_608 = wp::mul(var_24, var_607);
                        wp::sub_inplace(var_27, var_590, var_606, var_608);
                        // for c in range(6):                                                     <L 1014>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1015>
                        var_611 = wp::extract(var_589, var_609, var_610);
                        var_612 = wp::mul(var_24, var_611);
                        wp::sub_inplace(var_27, var_609, var_610, var_612);
                        var_614 = wp::extract(var_589, var_609, var_613);
                        var_615 = wp::mul(var_24, var_614);
                        wp::sub_inplace(var_27, var_609, var_613, var_615);
                        var_617 = wp::extract(var_589, var_609, var_616);
                        var_618 = wp::mul(var_24, var_617);
                        wp::sub_inplace(var_27, var_609, var_616, var_618);
                        var_620 = wp::extract(var_589, var_609, var_619);
                        var_621 = wp::mul(var_24, var_620);
                        wp::sub_inplace(var_27, var_609, var_619, var_621);
                        var_623 = wp::extract(var_589, var_609, var_622);
                        var_624 = wp::mul(var_24, var_623);
                        wp::sub_inplace(var_27, var_609, var_622, var_624);
                        var_626 = wp::extract(var_589, var_609, var_625);
                        var_627 = wp::mul(var_24, var_626);
                        wp::sub_inplace(var_27, var_609, var_625, var_627);
                        // for c in range(6):                                                     <L 1014>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1015>
                        var_630 = wp::extract(var_589, var_628, var_629);
                        var_631 = wp::mul(var_24, var_630);
                        wp::sub_inplace(var_27, var_628, var_629, var_631);
                        var_633 = wp::extract(var_589, var_628, var_632);
                        var_634 = wp::mul(var_24, var_633);
                        wp::sub_inplace(var_27, var_628, var_632, var_634);
                        var_636 = wp::extract(var_589, var_628, var_635);
                        var_637 = wp::mul(var_24, var_636);
                        wp::sub_inplace(var_27, var_628, var_635, var_637);
                        var_639 = wp::extract(var_589, var_628, var_638);
                        var_640 = wp::mul(var_24, var_639);
                        wp::sub_inplace(var_27, var_628, var_638, var_640);
                        var_642 = wp::extract(var_589, var_628, var_641);
                        var_643 = wp::mul(var_24, var_642);
                        wp::sub_inplace(var_27, var_628, var_641, var_643);
                        var_645 = wp::extract(var_589, var_628, var_644);
                        var_646 = wp::mul(var_24, var_645);
                        wp::sub_inplace(var_27, var_628, var_644, var_646);
                        // for c in range(6):                                                     <L 1014>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1015>
                        var_649 = wp::extract(var_589, var_647, var_648);
                        var_650 = wp::mul(var_24, var_649);
                        wp::sub_inplace(var_27, var_647, var_648, var_650);
                        var_652 = wp::extract(var_589, var_647, var_651);
                        var_653 = wp::mul(var_24, var_652);
                        wp::sub_inplace(var_27, var_647, var_651, var_653);
                        var_655 = wp::extract(var_589, var_647, var_654);
                        var_656 = wp::mul(var_24, var_655);
                        wp::sub_inplace(var_27, var_647, var_654, var_656);
                        var_658 = wp::extract(var_589, var_647, var_657);
                        var_659 = wp::mul(var_24, var_658);
                        wp::sub_inplace(var_27, var_647, var_657, var_659);
                        var_661 = wp::extract(var_589, var_647, var_660);
                        var_662 = wp::mul(var_24, var_661);
                        wp::sub_inplace(var_27, var_647, var_660, var_662);
                        var_664 = wp::extract(var_589, var_647, var_663);
                        var_665 = wp::mul(var_24, var_664);
                        wp::sub_inplace(var_27, var_647, var_663, var_665);
                        // for c in range(6):                                                     <L 1014>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1015>
                        var_668 = wp::extract(var_589, var_666, var_667);
                        var_669 = wp::mul(var_24, var_668);
                        wp::sub_inplace(var_27, var_666, var_667, var_669);
                        var_671 = wp::extract(var_589, var_666, var_670);
                        var_672 = wp::mul(var_24, var_671);
                        wp::sub_inplace(var_27, var_666, var_670, var_672);
                        var_674 = wp::extract(var_589, var_666, var_673);
                        var_675 = wp::mul(var_24, var_674);
                        wp::sub_inplace(var_27, var_666, var_673, var_675);
                        var_677 = wp::extract(var_589, var_666, var_676);
                        var_678 = wp::mul(var_24, var_677);
                        wp::sub_inplace(var_27, var_666, var_676, var_678);
                        var_680 = wp::extract(var_589, var_666, var_679);
                        var_681 = wp::mul(var_24, var_680);
                        wp::sub_inplace(var_27, var_666, var_679, var_681);
                        var_683 = wp::extract(var_589, var_666, var_682);
                        var_684 = wp::mul(var_24, var_683);
                        wp::sub_inplace(var_27, var_666, var_682, var_684);
                        // for c in range(6):                                                     <L 1014>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1015>
                        var_687 = wp::extract(var_589, var_685, var_686);
                        var_688 = wp::mul(var_24, var_687);
                        wp::sub_inplace(var_27, var_685, var_686, var_688);
                        var_690 = wp::extract(var_589, var_685, var_689);
                        var_691 = wp::mul(var_24, var_690);
                        wp::sub_inplace(var_27, var_685, var_689, var_691);
                        var_693 = wp::extract(var_589, var_685, var_692);
                        var_694 = wp::mul(var_24, var_693);
                        wp::sub_inplace(var_27, var_685, var_692, var_694);
                        var_696 = wp::extract(var_589, var_685, var_695);
                        var_697 = wp::mul(var_24, var_696);
                        wp::sub_inplace(var_27, var_685, var_695, var_697);
                        var_699 = wp::extract(var_589, var_685, var_698);
                        var_700 = wp::mul(var_24, var_699);
                        wp::sub_inplace(var_27, var_685, var_698, var_700);
                        var_702 = wp::extract(var_589, var_685, var_701);
                        var_703 = wp::mul(var_24, var_702);
                        wp::sub_inplace(var_27, var_685, var_701, var_703);
                        wp::assign(var_468, var_685);
                        wp::assign(var_469, var_701);
                        goto start_for_15;
                    end_for_15:;
                }
                var_704 = wp::load(var_474);
                var_705 = wp::load(var_474);
                if (!var_705) {
                    // elif body_mass[worldid % body_mass.shape[0], b] >= MJ_MINVAL:              <L 1017>
                    var_706 = &(var_body_mass.shape);
                    var_709 = wp::load(var_706);
                    var_708 = wp::extract(var_709, var_707);
                    var_710 = wp::mod(var_0, var_708);
                    var_711 = wp::address(var_body_mass, var_710, var_470);
                    var_714 = wp::load(var_711);
                    var_713 = (var_714 >= var_712);
                    if (var_713) {
                        // b_ipos = xipos_in[worldid, b]                                          <L 1018>
                        var_715 = wp::address(var_xipos_in, var_0, var_470);
                        var_717 = wp::load(var_715);
                        var_716 = wp::copy(var_717);
                        // b_imat = ximat_in[worldid, b]                                          <L 1019>
                        var_718 = wp::address(var_ximat_in, var_0, var_470);
                        var_720 = wp::load(var_718);
                        var_719 = wp::copy(var_720);
                        // cvel = cvel_in[worldid, b]                                             <L 1021>
                        var_721 = wp::address(var_cvel_in, var_0, var_470);
                        var_723 = wp::load(var_721);
                        var_722 = wp::copy(var_723);
                        // v_subtree_ang = wp.vec3(cvel[0], cvel[1], cvel[2])                     <L 1022>
                        var_725 = wp::extract(var_722, var_724);
                        var_727 = wp::extract(var_722, var_726);
                        var_729 = wp::extract(var_722, var_728);
                        var_730 = wp::vec_t<3, wp::float32>(var_725, var_727, var_729);
                        // v_subtree_lin = wp.vec3(cvel[3], cvel[4], cvel[5])                     <L 1023>
                        var_732 = wp::extract(var_722, var_731);
                        var_734 = wp::extract(var_722, var_733);
                        var_736 = wp::extract(var_722, var_735);
                        var_737 = wp::vec_t<3, wp::float32>(var_732, var_734, var_736);
                        // lin_com = v_subtree_lin - wp.cross(b_ipos - subtree_root, v_subtree_ang)       <L 1025>
                        var_738 = wp::sub(var_716, var_465);
                        var_739 = wp::cross(var_738, var_730);
                        var_740 = wp::sub(var_737, var_739);
                        // b_imat_T = wp.transpose(b_imat)                                        <L 1026>
                        var_741 = wp::transpose(var_719);
                        // v_local_ang = b_imat_T @ v_subtree_ang                                 <L 1027>
                        var_742 = wp::mul(var_741, var_730);
                        // v_local_lin = b_imat_T @ (lin_com - wind)                              <L 1028>
                        var_743 = wp::sub(var_740, var_457);
                        var_744 = wp::mul(var_741, var_743);
                        // lvel = wp.spatial_vector(                                              <L 1030>
                        // v_local_ang[0],                                                        <L 1031>
                        var_746 = wp::extract(var_742, var_745);
                        // v_local_ang[1],                                                        <L 1032>
                        var_748 = wp::extract(var_742, var_747);
                        // v_local_ang[2],                                                        <L 1033>
                        var_750 = wp::extract(var_742, var_749);
                        // v_local_lin[0],                                                        <L 1034>
                        var_752 = wp::extract(var_744, var_751);
                        // v_local_lin[1],                                                        <L 1035>
                        var_754 = wp::extract(var_744, var_753);
                        // v_local_lin[2],                                                        <L 1036>
                        var_756 = wp::extract(var_744, var_755);
                        var_757 = wp::vec_t<6, wp::float32>({var_746, var_748, var_750, var_752, var_754, var_756});
                        // B_box = derivative._deriv_box_fluid(                                   <L 1039>
                        // opt_integrator,                                                        <L 1040>
                        // body_mass,                                                             <L 1041>
                        // body_inertia,                                                          <L 1042>
                        // worldid,                                                               <L 1043>
                        // b,                                                                     <L 1044>
                        // lvel,                                                                  <L 1045>
                        // density,                                                               <L 1046>
                        // viscosity,                                                             <L 1047>
                        var_758 = _deriv_box_fluid_0(var_opt_integrator, var_body_mass, var_body_inertia, var_0, var_470, var_757, var_441, var_449);
                        // B00 = wp.diag(wp.vec3(B_box[0, 0], B_box[1, 1], B_box[2, 2]))          <L 1049>
                        var_761 = wp::extract(var_758, var_759, var_760);
                        var_764 = wp::extract(var_758, var_762, var_763);
                        var_767 = wp::extract(var_758, var_765, var_766);
                        var_768 = wp::vec_t<3, wp::float32>(var_761, var_764, var_767);
                        var_769 = wp::diag(var_768);
                        // B11 = wp.diag(wp.vec3(B_box[3, 3], B_box[4, 4], B_box[5, 5]))          <L 1050>
                        var_772 = wp::extract(var_758, var_770, var_771);
                        var_775 = wp::extract(var_758, var_773, var_774);
                        var_778 = wp::extract(var_758, var_776, var_777);
                        var_779 = wp::vec_t<3, wp::float32>(var_772, var_775, var_778);
                        var_780 = wp::diag(var_779);
                        // zero33 = wp.mat33(0.0)                                                 <L 1051>
                        var_782 = wp::mat_t<3, 3, wp::float32>(var_781);
                        // offset_box = b_ipos - subtree_root                                     <L 1052>
                        var_783 = wp::sub(var_716, var_465);
                        // J_T_B_J = _project_spatial_B(cdof_in, worldid, dof_adr, b_imat_T, offset_box, B00, zero33, zero33, B11)       <L 1053>
                        var_784 = _project_spatial_B_0(var_cdof_in, var_0, var_6, var_741, var_783, var_769, var_782, var_782, var_780);
                        // for r in range(6):                                                     <L 1054>
                        // for c in range(6):                                                     <L 1055>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1056>
                        var_787 = wp::extract(var_784, var_785, var_786);
                        var_788 = wp::mul(var_24, var_787);
                        wp::sub_inplace(var_27, var_785, var_786, var_788);
                        var_790 = wp::extract(var_784, var_785, var_789);
                        var_791 = wp::mul(var_24, var_790);
                        wp::sub_inplace(var_27, var_785, var_789, var_791);
                        var_793 = wp::extract(var_784, var_785, var_792);
                        var_794 = wp::mul(var_24, var_793);
                        wp::sub_inplace(var_27, var_785, var_792, var_794);
                        var_796 = wp::extract(var_784, var_785, var_795);
                        var_797 = wp::mul(var_24, var_796);
                        wp::sub_inplace(var_27, var_785, var_795, var_797);
                        var_799 = wp::extract(var_784, var_785, var_798);
                        var_800 = wp::mul(var_24, var_799);
                        wp::sub_inplace(var_27, var_785, var_798, var_800);
                        var_802 = wp::extract(var_784, var_785, var_801);
                        var_803 = wp::mul(var_24, var_802);
                        wp::sub_inplace(var_27, var_785, var_801, var_803);
                        // for c in range(6):                                                     <L 1055>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1056>
                        var_806 = wp::extract(var_784, var_804, var_805);
                        var_807 = wp::mul(var_24, var_806);
                        wp::sub_inplace(var_27, var_804, var_805, var_807);
                        var_809 = wp::extract(var_784, var_804, var_808);
                        var_810 = wp::mul(var_24, var_809);
                        wp::sub_inplace(var_27, var_804, var_808, var_810);
                        var_812 = wp::extract(var_784, var_804, var_811);
                        var_813 = wp::mul(var_24, var_812);
                        wp::sub_inplace(var_27, var_804, var_811, var_813);
                        var_815 = wp::extract(var_784, var_804, var_814);
                        var_816 = wp::mul(var_24, var_815);
                        wp::sub_inplace(var_27, var_804, var_814, var_816);
                        var_818 = wp::extract(var_784, var_804, var_817);
                        var_819 = wp::mul(var_24, var_818);
                        wp::sub_inplace(var_27, var_804, var_817, var_819);
                        var_821 = wp::extract(var_784, var_804, var_820);
                        var_822 = wp::mul(var_24, var_821);
                        wp::sub_inplace(var_27, var_804, var_820, var_822);
                        // for c in range(6):                                                     <L 1055>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1056>
                        var_825 = wp::extract(var_784, var_823, var_824);
                        var_826 = wp::mul(var_24, var_825);
                        wp::sub_inplace(var_27, var_823, var_824, var_826);
                        var_828 = wp::extract(var_784, var_823, var_827);
                        var_829 = wp::mul(var_24, var_828);
                        wp::sub_inplace(var_27, var_823, var_827, var_829);
                        var_831 = wp::extract(var_784, var_823, var_830);
                        var_832 = wp::mul(var_24, var_831);
                        wp::sub_inplace(var_27, var_823, var_830, var_832);
                        var_834 = wp::extract(var_784, var_823, var_833);
                        var_835 = wp::mul(var_24, var_834);
                        wp::sub_inplace(var_27, var_823, var_833, var_835);
                        var_837 = wp::extract(var_784, var_823, var_836);
                        var_838 = wp::mul(var_24, var_837);
                        wp::sub_inplace(var_27, var_823, var_836, var_838);
                        var_840 = wp::extract(var_784, var_823, var_839);
                        var_841 = wp::mul(var_24, var_840);
                        wp::sub_inplace(var_27, var_823, var_839, var_841);
                        // for c in range(6):                                                     <L 1055>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1056>
                        var_844 = wp::extract(var_784, var_842, var_843);
                        var_845 = wp::mul(var_24, var_844);
                        wp::sub_inplace(var_27, var_842, var_843, var_845);
                        var_847 = wp::extract(var_784, var_842, var_846);
                        var_848 = wp::mul(var_24, var_847);
                        wp::sub_inplace(var_27, var_842, var_846, var_848);
                        var_850 = wp::extract(var_784, var_842, var_849);
                        var_851 = wp::mul(var_24, var_850);
                        wp::sub_inplace(var_27, var_842, var_849, var_851);
                        var_853 = wp::extract(var_784, var_842, var_852);
                        var_854 = wp::mul(var_24, var_853);
                        wp::sub_inplace(var_27, var_842, var_852, var_854);
                        var_856 = wp::extract(var_784, var_842, var_855);
                        var_857 = wp::mul(var_24, var_856);
                        wp::sub_inplace(var_27, var_842, var_855, var_857);
                        var_859 = wp::extract(var_784, var_842, var_858);
                        var_860 = wp::mul(var_24, var_859);
                        wp::sub_inplace(var_27, var_842, var_858, var_860);
                        // for c in range(6):                                                     <L 1055>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1056>
                        var_863 = wp::extract(var_784, var_861, var_862);
                        var_864 = wp::mul(var_24, var_863);
                        wp::sub_inplace(var_27, var_861, var_862, var_864);
                        var_866 = wp::extract(var_784, var_861, var_865);
                        var_867 = wp::mul(var_24, var_866);
                        wp::sub_inplace(var_27, var_861, var_865, var_867);
                        var_869 = wp::extract(var_784, var_861, var_868);
                        var_870 = wp::mul(var_24, var_869);
                        wp::sub_inplace(var_27, var_861, var_868, var_870);
                        var_872 = wp::extract(var_784, var_861, var_871);
                        var_873 = wp::mul(var_24, var_872);
                        wp::sub_inplace(var_27, var_861, var_871, var_873);
                        var_875 = wp::extract(var_784, var_861, var_874);
                        var_876 = wp::mul(var_24, var_875);
                        wp::sub_inplace(var_27, var_861, var_874, var_876);
                        var_878 = wp::extract(var_784, var_861, var_877);
                        var_879 = wp::mul(var_24, var_878);
                        wp::sub_inplace(var_27, var_861, var_877, var_879);
                        // for c in range(6):                                                     <L 1055>
                        // A[r, c] -= timestep * J_T_B_J[r, c]                                    <L 1056>
                        var_882 = wp::extract(var_784, var_880, var_881);
                        var_883 = wp::mul(var_24, var_882);
                        wp::sub_inplace(var_27, var_880, var_881, var_883);
                        var_885 = wp::extract(var_784, var_880, var_884);
                        var_886 = wp::mul(var_24, var_885);
                        wp::sub_inplace(var_27, var_880, var_884, var_886);
                        var_888 = wp::extract(var_784, var_880, var_887);
                        var_889 = wp::mul(var_24, var_888);
                        wp::sub_inplace(var_27, var_880, var_887, var_889);
                        var_891 = wp::extract(var_784, var_880, var_890);
                        var_892 = wp::mul(var_24, var_891);
                        wp::sub_inplace(var_27, var_880, var_890, var_892);
                        var_894 = wp::extract(var_784, var_880, var_893);
                        var_895 = wp::mul(var_24, var_894);
                        wp::sub_inplace(var_27, var_880, var_893, var_895);
                        var_897 = wp::extract(var_784, var_880, var_896);
                        var_898 = wp::mul(var_24, var_897);
                        wp::sub_inplace(var_27, var_880, var_896, var_898);
                    }
                    var_899 = wp::where(var_713, var_880, var_468);
                    var_900 = wp::where(var_713, var_896, var_469);
                }
                var_901 = wp::load(var_474);
                var_903 = wp::load(var_474);
                var_902 = wp::where(var_903, var_468, var_899);
                var_905 = wp::load(var_474);
                var_904 = wp::where(var_905, var_469, var_900);
                var_907 = wp::load(var_474);
                var_906 = wp::where(var_907, var_480, var_722);
                var_909 = wp::load(var_474);
                var_908 = wp::where(var_909, var_486, var_740);
                var_911 = wp::load(var_474);
                var_910 = wp::where(var_911, var_584, var_769);
                var_913 = wp::load(var_474);
                var_912 = wp::where(var_913, var_587, var_780);
                var_915 = wp::load(var_474);
                var_914 = wp::where(var_915, var_589, var_784);
                wp::assign(var_468, var_902);
                wp::assign(var_469, var_904);
                goto start_for_13;
            end_for_13:;
        }
        var_916 = wp::where(var_459, var_468, var_401);
        var_917 = wp::where(var_459, var_469, var_424);
        // A_fact, pivot, ok = math.lu_factor_6x6(A)                                              <L 1059>
        lu_factor_6x6_0(var_27, var_918, var_919, var_920);
        // if ok:                                                                                 <L 1060>
        if (var_920) {
            // b_vec = vec6(0.0)                                                                  <L 1061>
            var_922 = wp::vec_t<6, wp::float32>(var_921);
            // for r in range(6):                                                                 <L 1062>
            // b_vec[r] = qfrc_in[worldid, dof_adr + r]                                           <L 1063>
            var_924 = wp::add(var_6, var_923);
            var_925 = wp::address(var_qfrc_in, var_0, var_924);
            var_927 = wp::load(var_925);
            var_926 = wp::copy(var_927);
            wp::assign_inplace(var_922, var_923, var_926);
            var_929 = wp::add(var_6, var_928);
            var_930 = wp::address(var_qfrc_in, var_0, var_929);
            var_932 = wp::load(var_930);
            var_931 = wp::copy(var_932);
            wp::assign_inplace(var_922, var_928, var_931);
            var_934 = wp::add(var_6, var_933);
            var_935 = wp::address(var_qfrc_in, var_0, var_934);
            var_937 = wp::load(var_935);
            var_936 = wp::copy(var_937);
            wp::assign_inplace(var_922, var_933, var_936);
            var_939 = wp::add(var_6, var_938);
            var_940 = wp::address(var_qfrc_in, var_0, var_939);
            var_942 = wp::load(var_940);
            var_941 = wp::copy(var_942);
            wp::assign_inplace(var_922, var_938, var_941);
            var_944 = wp::add(var_6, var_943);
            var_945 = wp::address(var_qfrc_in, var_0, var_944);
            var_947 = wp::load(var_945);
            var_946 = wp::copy(var_947);
            wp::assign_inplace(var_922, var_943, var_946);
            var_949 = wp::add(var_6, var_948);
            var_950 = wp::address(var_qfrc_in, var_0, var_949);
            var_952 = wp::load(var_950);
            var_951 = wp::copy(var_952);
            wp::assign_inplace(var_922, var_948, var_951);
            // x = math.lu_solve_6x6(A_fact, pivot, b_vec)                                        <L 1064>
            var_953 = lu_solve_6x6_0(var_918, var_919, var_922);
            // for r in range(6):                                                                 <L 1065>
            // qacc_out[worldid, dof_adr + r] = x[r]                                              <L 1066>
            var_955 = wp::extract(var_953, var_954);
            var_956 = wp::add(var_6, var_954);
            wp::array_store(var_qacc_out, var_0, var_956, var_955);
            var_958 = wp::extract(var_953, var_957);
            var_959 = wp::add(var_6, var_957);
            wp::array_store(var_qacc_out, var_0, var_959, var_958);
            var_961 = wp::extract(var_953, var_960);
            var_962 = wp::add(var_6, var_960);
            wp::array_store(var_qacc_out, var_0, var_962, var_961);
            var_964 = wp::extract(var_953, var_963);
            var_965 = wp::add(var_6, var_963);
            wp::array_store(var_qacc_out, var_0, var_965, var_964);
            var_967 = wp::extract(var_953, var_966);
            var_968 = wp::add(var_6, var_966);
            wp::array_store(var_qacc_out, var_0, var_968, var_967);
            var_970 = wp::extract(var_953, var_969);
            var_971 = wp::add(var_6, var_969);
            wp::array_store(var_qacc_out, var_0, var_971, var_970);
        }
        var_972 = wp::where(var_920, var_969, var_916);
    }
}



extern "C" __global__ void _euler_damp_qfrc_de139d07_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_damp_deriv,
    wp::array_t<wp::float32> var_M_integration_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        //---------
        // forward
        // def _euler_damp_qfrc(                                                                  <L 495>
        // worldid, tid = wp.tid()                                                                <L 505>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 506>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // adr = M_rowadr[tid] + M_rownnz[tid] - 1                                                <L 508>
        var_10 = wp::address(var_M_rowadr, var_1);
        var_11 = wp::address(var_M_rownnz, var_1);
        var_13 = wp::load(var_10);
        var_14 = wp::load(var_11);
        var_12 = wp::add(var_13, var_14);
        var_16 = wp::sub(var_12, var_15);
        // M_integration_out[worldid, adr] += timestep * damp_deriv[worldid, tid]                 <L 509>
        var_17 = wp::address(var_damp_deriv, var_0, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::mul(var_8, var_19);
        var_20 = wp::atomic_add(var_M_integration_out, var_0, var_16, var_18);
    }
}



extern "C" __global__ void _implicit_free_body_reset_m_df0a068d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_body_freeadr,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_qH_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::range_t var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::range_t var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 2;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::int32* var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::range_t var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::float32* var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::int32 var_52;
        const wp::int32 var_53 = 3;
        wp::int32 var_54;
        wp::int32* var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::int32* var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        wp::range_t var_61;
        wp::int32 var_62;
        wp::int32 var_63;
        wp::float32* var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 4;
        wp::int32 var_69;
        wp::int32* var_70;
        wp::int32 var_71;
        wp::int32 var_72;
        wp::int32* var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        wp::range_t var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        wp::float32* var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::int32 var_82;
        const wp::int32 var_83 = 5;
        wp::int32 var_84;
        wp::int32* var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        wp::int32* var_88;
        wp::int32 var_89;
        wp::int32 var_90;
        wp::range_t var_91;
        wp::int32 var_92;
        wp::int32 var_93;
        wp::float32* var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::int32 var_97;
        //---------
        // forward
        // def _implicit_free_body_reset_m(                                                       <L 711>
        // worldid, freeid = wp.tid()                                                             <L 722>
        builtin_tid2d(var_0, var_1);
        // bodyid = body_freeadr[freeid]                                                          <L 723>
        var_2 = wp::address(var_body_freeadr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dof_adr = body_dofadr[bodyid]                                                          <L 725>
        var_5 = wp::address(var_body_dofadr, var_3);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // for r in range(6):                                                                     <L 726>
        // row = dof_adr + r                                                                      <L 727>
        var_9 = wp::add(var_6, var_8);
        // start = M_rowadr[row]                                                                  <L 728>
        var_10 = wp::address(var_M_rowadr, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // nnz = M_rownnz[row]                                                                    <L 729>
        var_13 = wp::address(var_M_rownnz, var_9);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // for k in range(nnz):                                                                   <L 730>
        var_16 = wp::range(var_14);
        start_for_0:;
            if (iter_cmp(var_16) == 0) goto end_for_0;
            var_17 = wp::iter_next(var_16);
            // qH_out[worldid, start + k] = M_in[worldid, start + k]                              <L 731>
            var_18 = wp::add(var_11, var_17);
            var_19 = wp::address(var_M_in, var_0, var_18);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            var_22 = wp::add(var_11, var_17);
            wp::array_store(var_qH_out, var_0, var_22, var_20);
            goto start_for_0;
        end_for_0:;
        // row = dof_adr + r                                                                      <L 727>
        var_24 = wp::add(var_6, var_23);
        // start = M_rowadr[row]                                                                  <L 728>
        var_25 = wp::address(var_M_rowadr, var_24);
        var_27 = wp::load(var_25);
        var_26 = wp::copy(var_27);
        // nnz = M_rownnz[row]                                                                    <L 729>
        var_28 = wp::address(var_M_rownnz, var_24);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // for k in range(nnz):                                                                   <L 730>
        var_31 = wp::range(var_29);
        start_for_2:;
            if (iter_cmp(var_31) == 0) goto end_for_2;
            var_32 = wp::iter_next(var_31);
            // qH_out[worldid, start + k] = M_in[worldid, start + k]                              <L 731>
            var_33 = wp::add(var_26, var_32);
            var_34 = wp::address(var_M_in, var_0, var_33);
            var_36 = wp::load(var_34);
            var_35 = wp::copy(var_36);
            var_37 = wp::add(var_26, var_32);
            wp::array_store(var_qH_out, var_0, var_37, var_35);
            goto start_for_2;
        end_for_2:;
        // row = dof_adr + r                                                                      <L 727>
        var_39 = wp::add(var_6, var_38);
        // start = M_rowadr[row]                                                                  <L 728>
        var_40 = wp::address(var_M_rowadr, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // nnz = M_rownnz[row]                                                                    <L 729>
        var_43 = wp::address(var_M_rownnz, var_39);
        var_45 = wp::load(var_43);
        var_44 = wp::copy(var_45);
        // for k in range(nnz):                                                                   <L 730>
        var_46 = wp::range(var_44);
        start_for_4:;
            if (iter_cmp(var_46) == 0) goto end_for_4;
            var_47 = wp::iter_next(var_46);
            // qH_out[worldid, start + k] = M_in[worldid, start + k]                              <L 731>
            var_48 = wp::add(var_41, var_47);
            var_49 = wp::address(var_M_in, var_0, var_48);
            var_51 = wp::load(var_49);
            var_50 = wp::copy(var_51);
            var_52 = wp::add(var_41, var_47);
            wp::array_store(var_qH_out, var_0, var_52, var_50);
            goto start_for_4;
        end_for_4:;
        // row = dof_adr + r                                                                      <L 727>
        var_54 = wp::add(var_6, var_53);
        // start = M_rowadr[row]                                                                  <L 728>
        var_55 = wp::address(var_M_rowadr, var_54);
        var_57 = wp::load(var_55);
        var_56 = wp::copy(var_57);
        // nnz = M_rownnz[row]                                                                    <L 729>
        var_58 = wp::address(var_M_rownnz, var_54);
        var_60 = wp::load(var_58);
        var_59 = wp::copy(var_60);
        // for k in range(nnz):                                                                   <L 730>
        var_61 = wp::range(var_59);
        start_for_6:;
            if (iter_cmp(var_61) == 0) goto end_for_6;
            var_62 = wp::iter_next(var_61);
            // qH_out[worldid, start + k] = M_in[worldid, start + k]                              <L 731>
            var_63 = wp::add(var_56, var_62);
            var_64 = wp::address(var_M_in, var_0, var_63);
            var_66 = wp::load(var_64);
            var_65 = wp::copy(var_66);
            var_67 = wp::add(var_56, var_62);
            wp::array_store(var_qH_out, var_0, var_67, var_65);
            goto start_for_6;
        end_for_6:;
        // row = dof_adr + r                                                                      <L 727>
        var_69 = wp::add(var_6, var_68);
        // start = M_rowadr[row]                                                                  <L 728>
        var_70 = wp::address(var_M_rowadr, var_69);
        var_72 = wp::load(var_70);
        var_71 = wp::copy(var_72);
        // nnz = M_rownnz[row]                                                                    <L 729>
        var_73 = wp::address(var_M_rownnz, var_69);
        var_75 = wp::load(var_73);
        var_74 = wp::copy(var_75);
        // for k in range(nnz):                                                                   <L 730>
        var_76 = wp::range(var_74);
        start_for_8:;
            if (iter_cmp(var_76) == 0) goto end_for_8;
            var_77 = wp::iter_next(var_76);
            // qH_out[worldid, start + k] = M_in[worldid, start + k]                              <L 731>
            var_78 = wp::add(var_71, var_77);
            var_79 = wp::address(var_M_in, var_0, var_78);
            var_81 = wp::load(var_79);
            var_80 = wp::copy(var_81);
            var_82 = wp::add(var_71, var_77);
            wp::array_store(var_qH_out, var_0, var_82, var_80);
            goto start_for_8;
        end_for_8:;
        // row = dof_adr + r                                                                      <L 727>
        var_84 = wp::add(var_6, var_83);
        // start = M_rowadr[row]                                                                  <L 728>
        var_85 = wp::address(var_M_rowadr, var_84);
        var_87 = wp::load(var_85);
        var_86 = wp::copy(var_87);
        // nnz = M_rownnz[row]                                                                    <L 729>
        var_88 = wp::address(var_M_rownnz, var_84);
        var_90 = wp::load(var_88);
        var_89 = wp::copy(var_90);
        // for k in range(nnz):                                                                   <L 730>
        var_91 = wp::range(var_89);
        start_for_10:;
            if (iter_cmp(var_91) == 0) goto end_for_10;
            var_92 = wp::iter_next(var_91);
            // qH_out[worldid, start + k] = M_in[worldid, start + k]                              <L 731>
            var_93 = wp::add(var_86, var_92);
            var_94 = wp::address(var_M_in, var_0, var_93);
            var_96 = wp::load(var_94);
            var_95 = wp::copy(var_96);
            var_97 = wp::add(var_86, var_92);
            wp::array_store(var_qH_out, var_0, var_97, var_95);
            goto start_for_10;
        end_for_10:;
    }
}



extern "C" __global__ void _tendon_actuator_force_clamp_fd85b13d_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<bool> var_tendon_actfrclimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_actfrcrange,
    wp::array_t<wp::int32> var_actuator_trntype,
    wp::array_t<wp::vec_t<2, wp::int32>> var_actuator_trnid,
    wp::array_t<wp::float32> var_ten_actfrc_in,
    wp::array_t<wp::float32> var_actuator_force_out)
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
        wp::int32* var_2;
        const wp::int32 var_3 = 3;
        bool var_4;
        wp::int32 var_5;
        wp::vec_t<2, wp::int32>* var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::vec_t<2, wp::int32> var_9;
        bool* var_10;
        bool var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::shape_t* var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::shape_t var_18;
        wp::int32 var_19;
        wp::vec_t<2, wp::float32>* var_20;
        wp::vec_t<2, wp::float32> var_21;
        wp::vec_t<2, wp::float32> var_22;
        const wp::int32 var_23 = 0;
        wp::float32 var_24;
        bool var_25;
        wp::float32* var_26;
        const wp::int32 var_27 = 0;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        bool var_34;
        wp::float32* var_35;
        const wp::int32 var_36 = 1;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        bool var_41;
        //---------
        // forward
        // def _tendon_actuator_force_clamp(                                                      <L 1907>
        // worldid, actid = wp.tid()                                                              <L 1918>
        builtin_tid2d(var_0, var_1);
        // if actuator_trntype[actid] == TrnType.TENDON:                                          <L 1920>
        var_2 = wp::address(var_actuator_trntype, var_1);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // tenid = actuator_trnid[actid][0]                                                   <L 1921>
            var_6 = wp::address(var_actuator_trnid, var_1);
            var_9 = wp::load(var_6);
            var_8 = wp::extract(var_9, var_7);
            // if tendon_actfrclimited[tenid]:                                                    <L 1922>
            var_10 = wp::address(var_tendon_actfrclimited, var_8);
            var_11 = wp::load(var_10);
            if (var_11) {
                // ten_actfrc = ten_actfrc_in[worldid, tenid]                                     <L 1923>
                var_12 = wp::address(var_ten_actfrc_in, var_0, var_8);
                var_14 = wp::load(var_12);
                var_13 = wp::copy(var_14);
                // actfrcrange = tendon_actfrcrange[worldid % tendon_actfrcrange.shape[0], tenid]       <L 1924>
                var_15 = &(var_tendon_actfrcrange.shape);
                var_18 = wp::load(var_15);
                var_17 = wp::extract(var_18, var_16);
                var_19 = wp::mod(var_0, var_17);
                var_20 = wp::address(var_tendon_actfrcrange, var_19, var_8);
                var_22 = wp::load(var_20);
                var_21 = wp::copy(var_22);
                // if ten_actfrc < actfrcrange[0]:                                                <L 1926>
                var_24 = wp::extract(var_21, var_23);
                var_25 = (var_13 < var_24);
                if (var_25) {
                    // actuator_force_out[worldid, actid] *= actfrcrange[0] / ten_actfrc          <L 1927>
                    var_26 = wp::address(var_actuator_force_out, var_0, var_1);
                    var_28 = wp::extract(var_21, var_27);
                    var_29 = wp::div(var_28, var_13);
                    var_31 = wp::load(var_26);
                    var_30 = wp::mul(var_31, var_29);
                    wp::array_store(var_actuator_force_out, var_0, var_1, var_30);
                }
                if (!var_25) {
                    // elif ten_actfrc > actfrcrange[1]:                                          <L 1928>
                    var_33 = wp::extract(var_21, var_32);
                    var_34 = (var_13 > var_33);
                    if (var_34) {
                        // actuator_force_out[worldid, actid] *= actfrcrange[1] / ten_actfrc       <L 1929>
                        var_35 = wp::address(var_actuator_force_out, var_0, var_1);
                        var_37 = wp::extract(var_21, var_36);
                        var_38 = wp::div(var_37, var_13);
                        var_40 = wp::load(var_35);
                        var_39 = wp::mul(var_40, var_38);
                        wp::array_store(var_actuator_force_out, var_0, var_1, var_39);
                    }
                }
            }
            var_41 = wp::load(var_10);
        }
    }
}

