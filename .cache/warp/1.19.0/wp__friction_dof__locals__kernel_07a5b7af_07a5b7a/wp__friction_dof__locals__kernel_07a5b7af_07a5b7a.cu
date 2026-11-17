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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\constraint.py:83
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _contact_kbimp_0(
    wp::int32 var_opt_disableflags,
    wp::float32 var_timestep,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_pos_imp,
    bool var_is_discrete)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 0;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    wp::float32 var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 3;
    wp::float32 var_12;
    const wp::int32 var_13 = 4;
    wp::float32 var_14;
    bool var_15;
    const wp::int32 var_16 = 4096;
    wp::int32 var_17;
    bool var_18;
    bool var_19;
    const wp::float32 var_20 = 2.0;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 0.0001;
    const wp::float32 var_25 = 0.0001;
    const wp::float32 var_26 = 0.9999;
    const wp::float32 var_27 = 0.9999;
    wp::float32 var_28;
    const wp::float32 var_29 = 0.0001;
    const wp::float32 var_30 = 0.0001;
    const wp::float32 var_31 = 0.9999;
    const wp::float32 var_32 = 0.9999;
    wp::float32 var_33;
    const wp::float32 var_34 = 1e-15;
    const wp::float32 var_35 = 1e-15;
    wp::float32 var_36;
    const wp::float32 var_37 = 0.0001;
    const wp::float32 var_38 = 0.0001;
    const wp::float32 var_39 = 0.9999;
    const wp::float32 var_40 = 0.9999;
    wp::float32 var_41;
    const wp::float32 var_42 = 1.0;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    const wp::float32 var_47 = 0.0;
    bool var_48;
    const wp::int32 var_49 = 0;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 1e-15;
    const wp::float32 var_53 = 1e-15;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 1.0;
    const wp::float32 var_57 = 1e-15;
    const wp::float32 var_58 = 1e-15;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::int32 var_66 = 1;
    wp::float32 var_67;
    const wp::float32 var_68 = 0.0;
    bool var_69;
    const wp::int32 var_70 = 1;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::float32 var_73 = 1e-15;
    const wp::float32 var_74 = 1e-15;
    wp::float32 var_75;
    wp::float32 var_76;
    const wp::float32 var_77 = 2.0;
    const wp::float32 var_78 = 1e-15;
    const wp::float32 var_79 = 1e-15;
    wp::float32 var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    bool var_86;
    bool var_87;
    const wp::float32 var_88 = 1e-15;
    bool var_89;
    const wp::float32 var_90 = 0.5;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::float32 var_93 = 0.0;
    bool var_94;
    wp::float32 var_95;
    const wp::float32 var_96 = 1.0;
    bool var_97;
    wp::float32 var_98;
    const wp::float32 var_99 = 1.0;
    bool var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    bool var_104;
    const wp::float32 var_105 = 1.0;
    const wp::float32 var_106 = 1.0;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    wp::float32 var_115;
    const wp::float32 var_116 = 1.0;
    const wp::float32 var_117 = 1.0;
    const wp::float32 var_118 = 1.0;
    wp::float32 var_119;
    const wp::float32 var_120 = 1.0;
    wp::float32 var_121;
    wp::float32 var_122;
    wp::float32 var_123;
    const wp::float32 var_124 = 1.0;
    wp::float32 var_125;
    wp::float32 var_126;
    wp::float32 var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    wp::float32 var_133;
    wp::float32 var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::float32 var_138;
    wp::vec_t<3, wp::float32> var_139;
    //---------
    // forward
    // def _contact_kbimp(                                                                    <L 84>
    // timeconst = solref[0]                                                                  <L 95>
    var_1 = wp::extract(var_solref, var_0);
    // dampratio = solref[1]                                                                  <L 96>
    var_3 = wp::extract(var_solref, var_2);
    // dmin = solimp[0]                                                                       <L 97>
    var_5 = wp::extract(var_solimp, var_4);
    // dmax = solimp[1]                                                                       <L 98>
    var_7 = wp::extract(var_solimp, var_6);
    // width_raw = solimp[2]                                                                  <L 99>
    var_9 = wp::extract(var_solimp, var_8);
    // width = width_raw                                                                      <L 100>
    var_10 = wp::copy(var_9);
    // mid = solimp[3]                                                                        <L 101>
    var_12 = wp::extract(var_solimp, var_11);
    // power = solimp[4]                                                                      <L 102>
    var_14 = wp::extract(var_solimp, var_13);
    // if not (opt_disableflags & DisableBit.REFSAFE) and not is_discrete:                    <L 104>
    var_17 = wp::bit_and(var_opt_disableflags, var_16);
    var_18 = wp::unot(var_17);
    var_15 = var_18;
    if (var_15) {
        var_19 = wp::unot(var_is_discrete);
        var_15 = var_15 && var_19;
    }
    if (var_15) {
        // timeconst = wp.max(timeconst, 2.0 * timestep)                                      <L 105>
        var_21 = wp::mul(var_20, var_timestep);
        var_22 = wp::max(var_1, var_21);
    }
    var_23 = wp::where(var_15, var_22, var_1);
    // dmin = wp.clamp(dmin, types.MJ_MINIMP, types.MJ_MAXIMP)                                <L 107>
    var_28 = wp::clamp(var_5, var_25, var_27);
    // dmax = wp.clamp(dmax, types.MJ_MINIMP, types.MJ_MAXIMP)                                <L 108>
    var_33 = wp::clamp(var_7, var_30, var_32);
    // width = wp.max(types.MJ_MINVAL, width)                                                 <L 109>
    var_36 = wp::max(var_35, var_10);
    // mid = wp.clamp(mid, types.MJ_MINIMP, types.MJ_MAXIMP)                                  <L 110>
    var_41 = wp::clamp(var_12, var_38, var_40);
    // power = wp.max(1.0, power)                                                             <L 111>
    var_43 = wp::max(var_42, var_14);
    // dmax_sq = dmax * dmax                                                                  <L 114>
    var_44 = wp::mul(var_33, var_33);
    // if solref[0] <= 0.0:                                                                   <L 117>
    var_46 = wp::extract(var_solref, var_45);
    var_48 = (var_46 <= var_47);
    if (var_48) {
        // k = -solref[0] / wp.max(types.MJ_MINVAL, dmax_sq)                                  <L 118>
        var_50 = wp::extract(var_solref, var_49);
        var_51 = wp::neg(var_50);
        var_54 = wp::max(var_53, var_44);
        var_55 = wp::div(var_51, var_54);
    }
    if (!var_48) {
        // k = 1.0 / wp.max(types.MJ_MINVAL, dmax_sq * timeconst * timeconst * dampratio * dampratio)       <L 120>
        var_59 = wp::mul(var_44, var_23);
        var_60 = wp::mul(var_59, var_23);
        var_61 = wp::mul(var_60, var_3);
        var_62 = wp::mul(var_61, var_3);
        var_63 = wp::max(var_58, var_62);
        var_64 = wp::div(var_56, var_63);
    }
    var_65 = wp::where(var_48, var_55, var_64);
    // if solref[1] <= 0.0:                                                                   <L 121>
    var_67 = wp::extract(var_solref, var_66);
    var_69 = (var_67 <= var_68);
    if (var_69) {
        // b = -solref[1] / wp.max(types.MJ_MINVAL, dmax)                                     <L 122>
        var_71 = wp::extract(var_solref, var_70);
        var_72 = wp::neg(var_71);
        var_75 = wp::max(var_74, var_33);
        var_76 = wp::div(var_72, var_75);
    }
    if (!var_69) {
        // b = 2.0 / wp.max(types.MJ_MINVAL, dmax * timeconst)                                <L 124>
        var_80 = wp::mul(var_33, var_23);
        var_81 = wp::max(var_79, var_80);
        var_82 = wp::div(var_77, var_81);
    }
    var_83 = wp::where(var_69, var_76, var_82);
    // imp_x = wp.abs(pos_imp) / width                                                        <L 126>
    var_84 = wp::abs(var_pos_imp);
    var_85 = wp::div(var_84, var_36);
    // if dmin == dmax or width_raw <= types.MJ_MINVAL:                                       <L 129>
    var_87 = (var_28 == var_33);
    var_86 = var_87;
    if (!var_86) {
        var_89 = (var_9 <= var_88);
        var_86 = var_86 || var_89;
    }
    if (var_86) {
        // imp = 0.5 * (dmin + dmax)                                                          <L 130>
        var_91 = wp::add(var_28, var_33);
        var_92 = wp::mul(var_90, var_91);
    }
    if (!var_86) {
        // elif imp_x <= 0.0:                                                                 <L 131>
        var_94 = (var_85 <= var_93);
        if (var_94) {
            // imp = dmin                                                                     <L 132>
            var_95 = wp::copy(var_28);
        }
        if (!var_94) {
            // elif imp_x >= 1.0:                                                             <L 133>
            var_97 = (var_85 >= var_96);
            if (var_97) {
                // imp = dmax                                                                 <L 134>
                var_98 = wp::copy(var_33);
            }
            if (!var_97) {
                // elif power == 1.0:                                                         <L 135>
                var_100 = (var_43 == var_99);
                if (var_100) {
                    // imp = dmin + imp_x * (dmax - dmin)                                     <L 136>
                    var_101 = wp::sub(var_33, var_28);
                    var_102 = wp::mul(var_85, var_101);
                    var_103 = wp::add(var_28, var_102);
                }
                if (!var_100) {
                    // elif imp_x <= mid:                                                     <L 137>
                    var_104 = (var_85 <= var_41);
                    if (var_104) {
                        // imp_y = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)       <L 138>
                        var_107 = wp::sub(var_43, var_106);
                        var_108 = wp::pow(var_41, var_107);
                        var_109 = wp::div(var_105, var_108);
                        var_110 = wp::pow(var_85, var_43);
                        var_111 = wp::mul(var_109, var_110);
                        // imp = wp.clamp(dmin + imp_y * (dmax - dmin), dmin, dmax)           <L 139>
                        var_112 = wp::sub(var_33, var_28);
                        var_113 = wp::mul(var_111, var_112);
                        var_114 = wp::add(var_28, var_113);
                        var_115 = wp::clamp(var_114, var_28, var_33);
                    }
                    if (!var_104) {
                        // imp_y = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)       <L 141>
                        var_119 = wp::sub(var_118, var_41);
                        var_121 = wp::sub(var_43, var_120);
                        var_122 = wp::pow(var_119, var_121);
                        var_123 = wp::div(var_117, var_122);
                        var_125 = wp::sub(var_124, var_85);
                        var_126 = wp::pow(var_125, var_43);
                        var_127 = wp::mul(var_123, var_126);
                        var_128 = wp::sub(var_116, var_127);
                        // imp = wp.clamp(dmin + imp_y * (dmax - dmin), dmin, dmax)           <L 142>
                        var_129 = wp::sub(var_33, var_28);
                        var_130 = wp::mul(var_128, var_129);
                        var_131 = wp::add(var_28, var_130);
                        var_132 = wp::clamp(var_131, var_28, var_33);
                    }
                    var_133 = wp::where(var_104, var_111, var_128);
                    var_134 = wp::where(var_104, var_115, var_132);
                }
                var_135 = wp::where(var_100, var_103, var_134);
            }
            var_136 = wp::where(var_97, var_98, var_135);
        }
        var_137 = wp::where(var_94, var_95, var_136);
    }
    var_138 = wp::where(var_86, var_92, var_137);
    // return wp.vec3(k, b, imp)                                                              <L 144>
    var_139 = wp::vec_t<3, wp::float32>(var_65, var_83, var_138);
    return var_139;
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\constraint.py:147
static CUDA_CALLABLE wp::float32 _efc_D_0(
    wp::float32 var_invweight,
    wp::float32 var_imp)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1.0;
    const wp::float32 var_1 = 1.0;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::float32 var_5 = 1e-15;
    const wp::float32 var_6 = 1e-15;
    wp::float32 var_7;
    wp::float32 var_8;
    //---------
    // forward
    // def _efc_D(invweight: float, imp: float) -> float:                                     <L 148>
    // return 1.0 / wp.max(invweight * (1.0 - imp) / imp, types.MJ_MINVAL)                    <L 149>
    var_2 = wp::sub(var_1, var_imp);
    var_3 = wp::mul(var_invweight, var_2);
    var_4 = wp::div(var_3, var_imp);
    var_7 = wp::max(var_4, var_6);
    var_8 = wp::div(var_0, var_7);
    return var_8;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\constraint.py:152
static CUDA_CALLABLE void _efc_row_0(
    wp::int32 var_opt_disableflags,
    wp::int32 var_worldid,
    wp::float32 var_timestep,
    wp::int32 var_efcid,
    wp::float32 var_pos_aref,
    wp::float32 var_pos_imp,
    wp::float32 var_invweight,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_margin,
    wp::float32 var_vel,
    wp::float32 var_frictionloss,
    wp::int32 var_type,
    wp::int32 var_id,
    bool var_is_discrete,
    wp::float32 var_aref_shift,
    wp::array_t<wp::int32> var_type_out,
    wp::array_t<wp::int32> var_id_out,
    wp::array_t<wp::float32> var_pos_out,
    wp::array_t<wp::float32> var_margin_out,
    wp::array_t<wp::float32> var_D_out,
    wp::array_t<wp::float32> var_vel_out,
    wp::array_t<wp::float32> var_aref_out,
    wp::array_t<wp::float32> var_frictionloss_out)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    bool var_1;
    const wp::int32 var_2 = 1;
    bool var_3;
    const wp::int32 var_4 = 2;
    bool var_5;
    const wp::float32 var_6 = 0.0;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::float32 var_14;
    bool var_15;
    const wp::int32 var_16 = 4096;
    wp::int32 var_17;
    bool var_18;
    const wp::int32 var_19 = 0;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    const wp::float32 var_23 = 0.0;
    bool var_24;
    bool var_25;
    const wp::int32 var_26 = 3;
    bool var_27;
    const wp::int32 var_28 = 4;
    bool var_29;
    const wp::int32 var_30 = 5;
    bool var_31;
    const wp::int32 var_32 = 6;
    bool var_33;
    const wp::int32 var_34 = 7;
    bool var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::float32 var_38 = 1.0;
    bool var_39;
    const wp::float32 var_40 = 0.9999;
    const wp::float32 var_41 = 1.0;
    wp::float32 var_42;
    wp::float32 var_43;
    const wp::float32 var_44 = 1.0;
    const wp::float32 var_45 = 0.9999;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    const wp::float32 var_49 = 1.0;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 0.0;
    const wp::float32 var_55 = 2.0;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    const wp::float32 var_64 = 1.0;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    const wp::float32 var_70 = 1e-15;
    const wp::float32 var_71 = 1e-15;
    const wp::float32 var_72 = 1.0;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    const wp::float32 var_77 = 1e-15;
    const wp::float32 var_78 = 1e-15;
    const wp::float32 var_79 = 1.0;
    const wp::float32 var_80 = 0.9999;
    wp::float32 var_81;
    wp::float32 var_82;
    const wp::float32 var_83 = 0.9999;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    const wp::float32 var_88 = 1.0;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    //---------
    // forward
    // def _efc_row(                                                                          <L 153>
    // kbimp = _contact_kbimp(opt_disableflags, timestep, solref, solimp, pos_imp, is_discrete)       <L 183>
    var_0 = _contact_kbimp_0(var_opt_disableflags, var_timestep, var_solref, var_solimp, var_pos_imp, var_is_discrete);
    // k = 0.0 if (type == ConstraintType.FRICTION_DOF or type == ConstraintType.FRICTION_TENDON) else kbimp[0]       <L 184>
    var_3 = (var_type == var_2);
    var_1 = var_3;
    if (!var_1) {
        var_5 = (var_type == var_4);
        var_1 = var_1 || var_5;
    }
    if (var_1) {
    }
    if (!var_1) {
        var_8 = wp::extract(var_0, var_7);
    }
    var_9 = wp::where(var_1, var_6, var_8);
    // b = kbimp[1]                                                                           <L 185>
    var_11 = wp::extract(var_0, var_10);
    // imp = kbimp[2]                                                                         <L 186>
    var_13 = wp::extract(var_0, var_12);
    // if is_discrete:                                                                        <L 188>
    if (var_is_discrete) {
        // timestep_sq = timestep * timestep                                                  <L 189>
        var_14 = wp::mul(var_timestep, var_timestep);
        // if (                                                                               <L 191>
        // not (opt_disableflags & DisableBit.REFSAFE)                                        <L 192>
        var_17 = wp::bit_and(var_opt_disableflags, var_16);
        var_18 = wp::unot(var_17);
        var_15 = var_18;
        if (var_15) {
            // and solref[0] > 0.0                                                            <L 193>
            var_20 = wp::extract(var_solref, var_19);
            var_22 = (var_20 > var_21);
            var_15 = var_15 && var_22;
        }
        if (var_15) {
            // and k > 0.0                                                                    <L 194>
            var_24 = (var_9 > var_23);
            var_15 = var_15 && var_24;
        }
        if (var_15) {
            // type == ConstraintType.LIMIT_JOINT                                             <L 196>
            var_27 = (var_type == var_26);
            var_25 = var_27;
            if (!var_25) {
                // or type == ConstraintType.LIMIT_TENDON                                     <L 197>
                var_29 = (var_type == var_28);
                var_25 = var_25 || var_29;
            }
            if (!var_25) {
                // or type == ConstraintType.CONTACT_FRICTIONLESS                             <L 198>
                var_31 = (var_type == var_30);
                var_25 = var_25 || var_31;
            }
            if (!var_25) {
                // or type == ConstraintType.CONTACT_PYRAMIDAL                                <L 199>
                var_33 = (var_type == var_32);
                var_25 = var_25 || var_33;
            }
            if (!var_25) {
                // or type == ConstraintType.CONTACT_ELLIPTIC                                 <L 200>
                var_35 = (var_type == var_34);
                var_25 = var_25 || var_35;
            }
            var_15 = var_15 && var_25;
        }
        if (var_15) {
            // excess = timestep_sq * k * imp                                                 <L 203>
            var_36 = wp::mul(var_14, var_9);
            var_37 = wp::mul(var_36, var_13);
            // if excess > 1.0:                                                               <L 204>
            var_39 = (var_37 > var_38);
            if (var_39) {
                // fmax = math.safe_div(types.MJ_MAXIMP * (1.0 - imp), imp * (1.0 - types.MJ_MAXIMP))       <L 205>
                var_42 = wp::sub(var_41, var_13);
                var_43 = wp::mul(var_40, var_42);
                var_46 = wp::sub(var_44, var_45);
                var_47 = wp::mul(var_13, var_46);
                var_48 = safe_div_0(var_43, var_47);
                // k = 1.0 / (timestep_sq * imp)                                              <L 206>
                var_50 = wp::mul(var_14, var_13);
                var_51 = wp::div(var_49, var_50);
                // b = wp.min(b / wp.sqrt(excess), wp.max(0.0, fmax - 2.0) / timestep)        <L 207>
                var_52 = wp::sqrt(var_37);
                var_53 = wp::div(var_11, var_52);
                var_56 = wp::sub(var_48, var_55);
                var_57 = wp::max(var_54, var_56);
                var_58 = wp::div(var_57, var_timestep);
                var_59 = wp::min(var_53, var_58);
            }
            var_60 = wp::where(var_39, var_51, var_9);
            var_61 = wp::where(var_39, var_59, var_11);
        }
        var_62 = wp::where(var_15, var_60, var_9);
        var_63 = wp::where(var_15, var_61, var_11);
        // f = 1.0 + timestep * b + timestep_sq * k * imp                                     <L 209>
        var_65 = wp::mul(var_timestep, var_63);
        var_66 = wp::add(var_64, var_65);
        var_67 = wp::mul(var_14, var_62);
        var_68 = wp::mul(var_67, var_13);
        var_69 = wp::add(var_66, var_68);
        // R0 = wp.max(types.MJ_MINVAL, (1.0 - imp) * invweight / imp)                        <L 210>
        var_73 = wp::sub(var_72, var_13);
        var_74 = wp::mul(var_73, var_invweight);
        var_75 = wp::div(var_74, var_13);
        var_76 = wp::max(var_71, var_75);
        // Rmin = wp.max(types.MJ_MINVAL, (1.0 - types.MJ_MAXIMP) * invweight / types.MJ_MAXIMP)       <L 211>
        var_81 = wp::sub(var_79, var_80);
        var_82 = wp::mul(var_81, var_invweight);
        var_84 = wp::div(var_82, var_83);
        var_85 = wp::max(var_78, var_84);
        // R = wp.max(R0 / f, Rmin)                                                           <L 212>
        var_86 = wp::div(var_76, var_69);
        var_87 = wp::max(var_86, var_85);
        // D_out[worldid, efcid] = 1.0 / R                                                    <L 213>
        var_89 = wp::div(var_88, var_87);
        wp::array_store(var_D_out, var_worldid, var_efcid, var_89);
        // aref_out[worldid, efcid] = (-k * imp * pos_aref - (b + timestep * k * imp) * vel - aref_shift) / f       <L 214>
        var_90 = wp::neg(var_62);
        var_91 = wp::mul(var_90, var_13);
        var_92 = wp::mul(var_91, var_pos_aref);
        var_93 = wp::mul(var_timestep, var_62);
        var_94 = wp::mul(var_93, var_13);
        var_95 = wp::add(var_63, var_94);
        var_96 = wp::mul(var_95, var_vel);
        var_97 = wp::sub(var_92, var_96);
        var_98 = wp::sub(var_97, var_aref_shift);
        var_99 = wp::div(var_98, var_69);
        wp::array_store(var_aref_out, var_worldid, var_efcid, var_99);
    }
    if (!var_is_discrete) {
        // D_out[worldid, efcid] = _efc_D(invweight, imp)                                     <L 216>
        var_100 = _efc_D_0(var_invweight, var_13);
        wp::array_store(var_D_out, var_worldid, var_efcid, var_100);
        // aref_out[worldid, efcid] = -k * imp * pos_aref - b * vel - aref_shift              <L 217>
        var_101 = wp::neg(var_9);
        var_102 = wp::mul(var_101, var_13);
        var_103 = wp::mul(var_102, var_pos_aref);
        var_104 = wp::mul(var_11, var_vel);
        var_105 = wp::sub(var_103, var_104);
        var_106 = wp::sub(var_105, var_aref_shift);
        wp::array_store(var_aref_out, var_worldid, var_efcid, var_106);
    }
    var_107 = wp::where(var_is_discrete, var_62, var_9);
    var_108 = wp::where(var_is_discrete, var_63, var_11);
    // vel_out[worldid, efcid] = vel                                                          <L 219>
    wp::array_store(var_vel_out, var_worldid, var_efcid, var_vel);
    // pos_out[worldid, efcid] = pos_aref + margin                                            <L 220>
    var_109 = wp::add(var_pos_aref, var_margin);
    wp::array_store(var_pos_out, var_worldid, var_efcid, var_109);
    // margin_out[worldid, efcid] = margin                                                    <L 221>
    wp::array_store(var_margin_out, var_worldid, var_efcid, var_margin);
    // frictionloss_out[worldid, efcid] = frictionloss                                        <L 222>
    wp::array_store(var_frictionloss_out, var_worldid, var_efcid, var_frictionloss);
    // type_out[worldid, efcid] = type                                                        <L 223>
    wp::array_store(var_type_out, var_worldid, var_efcid, var_type);
    // id_out[worldid, efcid] = id                                                            <L 224>
    wp::array_store(var_id_out, var_worldid, var_efcid, var_id);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\constraint.py:83
static CUDA_CALLABLE void adj__contact_kbimp_0(
    wp::int32 var_opt_disableflags,
    wp::float32 var_timestep,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_pos_imp,
    bool var_is_discrete,
    wp::int32 & adj_opt_disableflags,
    wp::float32 & adj_timestep,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::float32 & adj_pos_imp,
    bool & adj_is_discrete,
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\constraint.py:147
static CUDA_CALLABLE void adj__efc_D_0(
    wp::float32 var_invweight,
    wp::float32 var_imp,
    wp::float32 & adj_invweight,
    wp::float32 & adj_imp,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\constraint.py:152
static CUDA_CALLABLE void adj__efc_row_0(
    wp::int32 var_opt_disableflags,
    wp::int32 var_worldid,
    wp::float32 var_timestep,
    wp::int32 var_efcid,
    wp::float32 var_pos_aref,
    wp::float32 var_pos_imp,
    wp::float32 var_invweight,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_margin,
    wp::float32 var_vel,
    wp::float32 var_frictionloss,
    wp::int32 var_type,
    wp::int32 var_id,
    bool var_is_discrete,
    wp::float32 var_aref_shift,
    wp::array_t<wp::int32> var_type_out,
    wp::array_t<wp::int32> var_id_out,
    wp::array_t<wp::float32> var_pos_out,
    wp::array_t<wp::float32> var_margin_out,
    wp::array_t<wp::float32> var_D_out,
    wp::array_t<wp::float32> var_vel_out,
    wp::array_t<wp::float32> var_aref_out,
    wp::array_t<wp::float32> var_frictionloss_out,
    wp::int32 & adj_opt_disableflags,
    wp::int32 & adj_worldid,
    wp::float32 & adj_timestep,
    wp::int32 & adj_efcid,
    wp::float32 & adj_pos_aref,
    wp::float32 & adj_pos_imp,
    wp::float32 & adj_invweight,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::float32 & adj_margin,
    wp::float32 & adj_vel,
    wp::float32 & adj_frictionloss,
    wp::int32 & adj_type,
    wp::int32 & adj_id,
    bool & adj_is_discrete,
    wp::float32 & adj_aref_shift,
    wp::array_t<wp::int32> & adj_type_out,
    wp::array_t<wp::int32> & adj_id_out,
    wp::array_t<wp::float32> & adj_pos_out,
    wp::array_t<wp::float32> & adj_margin_out,
    wp::array_t<wp::float32> & adj_D_out,
    wp::array_t<wp::float32> & adj_vel_out,
    wp::array_t<wp::float32> & adj_aref_out,
    wp::array_t<wp::float32> & adj_frictionloss_out)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _friction_dof__locals__kernel_3b050432_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::vec_t<2, wp::float32>> var_dof_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_dof_solimp,
    wp::array_t<wp::float32> var_dof_frictionloss,
    wp::array_t<wp::float32> var_dof_invweight0,
    wp::array_t<wp::float32> var_qvel_in,
    wp::int32 var_njmax_in,
    wp::int32 var_njmax_nnz_in,
    wp::array_t<wp::int32> var_nf_out,
    wp::array_t<wp::int32> var_nefc_out,
    wp::array_t<wp::int32> var_efc_type_out,
    wp::array_t<wp::int32> var_efc_id_out,
    wp::array_t<wp::int32> var_efc_jtdaj_adr_out,
    wp::array_t<wp::int32> var_efc_jtdaj_nrow_out,
    wp::array_t<wp::int32> var_efc_jtdaj_nblock_out,
    wp::array_t<wp::int32> var_efc_J_rownnz_out,
    wp::array_t<wp::int32> var_efc_J_rowadr_out,
    wp::array_t<wp::int32> var_efc_J_colind_out,
    wp::array_t<wp::float32> var_efc_J_out,
    wp::array_t<wp::float32> var_efc_pos_out,
    wp::array_t<wp::float32> var_efc_margin_out,
    wp::array_t<wp::float32> var_efc_D_out,
    wp::array_t<wp::float32> var_efc_vel_out,
    wp::array_t<wp::float32> var_efc_aref_out,
    wp::array_t<wp::float32> var_efc_frictionloss_out,
    wp::array_t<wp::int32> var_efc_nnz_out)
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
        wp::shape_t* var_2;
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        wp::shape_t var_5;
        wp::int32 var_6;
        wp::float32* var_7;
        const wp::float32 var_8 = 0.0;
        bool var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        bool var_15;
        const bool var_16 = false;
        const bool var_17 = false;
        wp::range_t var_18;
        wp::int32 var_19;
        const wp::float32 var_20 = 0.0;
        const wp::float32 var_21 = 1.0;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::shape_t* var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::shape_t var_28;
        wp::int32 var_29;
        wp::shape_t* var_30;
        const wp::int32 var_31 = 0;
        wp::int32 var_32;
        wp::shape_t var_33;
        wp::int32 var_34;
        wp::shape_t* var_35;
        const wp::int32 var_36 = 0;
        wp::int32 var_37;
        wp::shape_t var_38;
        wp::int32 var_39;
        wp::shape_t* var_40;
        const wp::int32 var_41 = 0;
        wp::int32 var_42;
        wp::shape_t var_43;
        wp::int32 var_44;
        wp::float32* var_45;
        const wp::float32 var_46 = 0.0;
        const wp::float32 var_47 = 0.0;
        wp::float32* var_48;
        wp::vec_t<2, wp::float32>* var_49;
        wp::vec_t<5, wp::float32>* var_50;
        const wp::float32 var_51 = 0.0;
        wp::float32* var_52;
        const wp::int32 var_53 = 1;
        const wp::int32 var_54 = 1;
        const bool var_55 = false;
        const wp::float32 var_56 = 0.0;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::vec_t<2, wp::float32> var_59;
        wp::vec_t<5, wp::float32> var_60;
        wp::float32 var_61;
        //---------
        // forward
        // def kernel(                                                                            <L 1852>
        // worldid, dofid = wp.tid()                                                              <L 1886>
        builtin_tid2d(var_0, var_1);
        // dof_frictionloss_id = worldid % dof_frictionloss.shape[0]                              <L 1888>
        var_2 = &(var_dof_frictionloss.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // if dof_frictionloss[dof_frictionloss_id, dofid] <= 0.0:                                <L 1890>
        var_7 = wp::address(var_dof_frictionloss, var_6, var_1);
        var_10 = wp::load(var_7);
        var_9 = (var_10 <= var_8);
        if (var_9) {
            // return                                                                             <L 1891>
            return;
        }
        // wp.atomic_add(nf_out, worldid, 1)                                                      <L 1893>
        var_12 = wp::atomic_add(var_nf_out, var_0, var_11);
        // efcid = wp.atomic_add(nefc_out, worldid, 1)                                            <L 1894>
        var_14 = wp::atomic_add(var_nefc_out, var_0, var_13);
        // if efcid >= njmax_in:                                                                  <L 1896>
        var_15 = (var_14 >= var_njmax_in);
        if (var_15) {
            // return                                                                             <L 1897>
            return;
        }
        // if wp.static(is_sparse and newton):                                                    <L 1899>
        // if wp.static(is_sparse):                                                               <L 1904>
        // for i in range(nv):                                                                    <L 1913>
        var_18 = wp::range(var_nv);
        start_for_2:;
            if (iter_cmp(var_18) == 0) goto end_for_2;
            var_19 = wp::iter_next(var_18);
            // efc_J_out[worldid, efcid, i] = 0.0                                                 <L 1914>
            wp::array_store(var_efc_J_out, var_0, var_14, var_19, var_20);
            goto start_for_2;
        end_for_2:;
        // efc_J_out[worldid, efcid, dofid] = 1.0                                                 <L 1915>
        wp::array_store(var_efc_J_out, var_0, var_14, var_1, var_21);
        // Jqvel = qvel_in[worldid, dofid]                                                        <L 1917>
        var_22 = wp::address(var_qvel_in, var_0, var_1);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // dof_invweight0_id = worldid % dof_invweight0.shape[0]                                  <L 1919>
        var_25 = &(var_dof_invweight0.shape);
        var_28 = wp::load(var_25);
        var_27 = wp::extract(var_28, var_26);
        var_29 = wp::mod(var_0, var_27);
        // dof_solref_id = worldid % dof_solref.shape[0]                                          <L 1920>
        var_30 = &(var_dof_solref.shape);
        var_33 = wp::load(var_30);
        var_32 = wp::extract(var_33, var_31);
        var_34 = wp::mod(var_0, var_32);
        // dof_solimp_id = worldid % dof_solimp.shape[0]                                          <L 1921>
        var_35 = &(var_dof_solimp.shape);
        var_38 = wp::load(var_35);
        var_37 = wp::extract(var_38, var_36);
        var_39 = wp::mod(var_0, var_37);
        // _efc_row(                                                                              <L 1922>
        // opt_disableflags,                                                                      <L 1923>
        // worldid,                                                                               <L 1924>
        // opt_timestep[worldid % opt_timestep.shape[0]],                                         <L 1925>
        var_40 = &(var_opt_timestep.shape);
        var_43 = wp::load(var_40);
        var_42 = wp::extract(var_43, var_41);
        var_44 = wp::mod(var_0, var_42);
        var_45 = wp::address(var_opt_timestep, var_44);
        // efcid,                                                                                 <L 1926>
        // 0.0,                                                                                   <L 1927>
        // 0.0,                                                                                   <L 1928>
        // dof_invweight0[dof_invweight0_id, dofid],                                              <L 1929>
        var_48 = wp::address(var_dof_invweight0, var_29, var_1);
        // dof_solref[dof_solref_id, dofid],                                                      <L 1930>
        var_49 = wp::address(var_dof_solref, var_34, var_1);
        // dof_solimp[dof_solimp_id, dofid],                                                      <L 1931>
        var_50 = wp::address(var_dof_solimp, var_39, var_1);
        // 0.0,                                                                                   <L 1932>
        // Jqvel,                                                                                 <L 1933>
        // dof_frictionloss[dof_frictionloss_id, dofid],                                          <L 1934>
        var_52 = wp::address(var_dof_frictionloss, var_6, var_1);
        // ConstraintType.FRICTION_DOF,                                                           <L 1935>
        // dofid,                                                                                 <L 1936>
        // is_discrete,                                                                           <L 1937>
        // 0.0,                                                                                   <L 1938>
        // efc_type_out,                                                                          <L 1939>
        // efc_id_out,                                                                            <L 1940>
        // efc_pos_out,                                                                           <L 1941>
        // efc_margin_out,                                                                        <L 1942>
        // efc_D_out,                                                                             <L 1943>
        // efc_vel_out,                                                                           <L 1944>
        // efc_aref_out,                                                                          <L 1945>
        // efc_frictionloss_out,                                                                  <L 1946>
        var_57 = wp::load(var_45);
        var_58 = wp::load(var_48);
        var_59 = wp::load(var_49);
        var_60 = wp::load(var_50);
        var_61 = wp::load(var_52);
        _efc_row_0(var_opt_disableflags, var_0, var_57, var_14, var_46, var_47, var_58, var_59, var_60, var_51, var_23, var_61, var_54, var_1, var_55, var_56, var_efc_type_out, var_efc_id_out, var_efc_pos_out, var_efc_margin_out, var_efc_D_out, var_efc_vel_out, var_efc_aref_out, var_efc_frictionloss_out);
}

