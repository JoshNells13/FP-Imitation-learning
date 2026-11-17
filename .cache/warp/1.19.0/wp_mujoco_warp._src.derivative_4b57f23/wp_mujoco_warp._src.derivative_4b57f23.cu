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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:1059
static CUDA_CALLABLE wp::vec_t<6, wp::float32> _get_jac_column_local_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point_global,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::mat_t<3, 3, wp::float32> var_b_imat)
{
    //---------
    // primal vars
    wp::int32* var_0;
    wp::int32 var_1;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<6, wp::float32>* var_5;
    wp::vec_t<6, wp::float32> var_6;
    wp::vec_t<6, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<6, wp::float32> var_16;
    //---------
    // forward
    // def _get_jac_column_local(                                                             <L 1060>
    // offset = point_global - subtree_com_in[worldid, body_rootid[bodyid]]                   <L 1075>
    var_0 = wp::address(var_body_rootid, var_bodyid);
    var_1 = wp::load(var_0);
    var_2 = wp::address(var_subtree_com_in, var_worldid, var_1);
    var_4 = wp::load(var_2);
    var_3 = wp::sub(var_point_global, var_4);
    // cdof_val = cdof_in[worldid, dofid]                                                     <L 1076>
    var_5 = wp::address(var_cdof_in, var_worldid, var_dofid);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // cdof_ang = wp.spatial_top(cdof_val)                                                    <L 1077>
    var_8 = wp::spatial_top(var_6);
    // cdof_lin = wp.spatial_bottom(cdof_val)                                                 <L 1078>
    var_9 = wp::spatial_bottom(var_6);
    // jacp = cdof_lin + wp.cross(cdof_ang, offset)                                           <L 1080>
    var_10 = wp::cross(var_8, var_3);
    var_11 = wp::add(var_9, var_10);
    // jacr = cdof_ang                                                                        <L 1081>
    var_12 = wp::copy(var_8);
    // b_imat_T = wp.transpose(b_imat)                                                        <L 1083>
    var_13 = wp::transpose(var_b_imat);
    // jacp_loc = b_imat_T @ jacp                                                             <L 1084>
    var_14 = wp::mul(var_13, var_11);
    // jacr_loc = b_imat_T @ jacr                                                             <L 1085>
    var_15 = wp::mul(var_13, var_12);
    // return wp.spatial_vector(jacr_loc, jacp_loc)                                           <L 1086>
    var_16 = wp::vec_t<6, wp::float32>(var_15, var_14);
    return var_16;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:2376
static CUDA_CALLABLE void _write_cholesky33_0(
    wp::int32 var_worldid,
    wp::int32 var_base,
    wp::float32 var_B00,
    wp::float32 var_B10,
    wp::float32 var_B11,
    wp::float32 var_B20,
    wp::float32 var_B21,
    wp::float32 var_B22,
    wp::array_t<wp::float32> var_out)
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
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::int32 var_18 = 0;
    wp::int32 var_19;
    const wp::float32 var_20 = 0.0;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    const wp::float32 var_23 = 0.0;
    const wp::int32 var_24 = 2;
    wp::int32 var_25;
    const wp::int32 var_26 = 3;
    wp::int32 var_27;
    const wp::int32 var_28 = 4;
    wp::int32 var_29;
    const wp::float32 var_30 = 0.0;
    const wp::int32 var_31 = 5;
    wp::int32 var_32;
    const wp::int32 var_33 = 6;
    wp::int32 var_34;
    const wp::int32 var_35 = 7;
    wp::int32 var_36;
    const wp::int32 var_37 = 8;
    wp::int32 var_38;
    //---------
    // forward
    // def _write_cholesky33(                                                                 <L 2377>
    // L00 = wp.sqrt(wp.max(MJ_MINVAL, B00))                                                  <L 2390>
    var_1 = wp::max(var_0, var_B00);
    var_2 = wp::sqrt(var_1);
    // L10 = B10 / L00                                                                        <L 2391>
    var_3 = wp::div(var_B10, var_2);
    // L20 = B20 / L00                                                                        <L 2392>
    var_4 = wp::div(var_B20, var_2);
    // L11 = wp.sqrt(wp.max(MJ_MINVAL, B11 - L10 * L10))                                      <L 2393>
    var_5 = wp::mul(var_3, var_3);
    var_6 = wp::sub(var_B11, var_5);
    var_7 = wp::max(var_0, var_6);
    var_8 = wp::sqrt(var_7);
    // L21 = (B21 - L20 * L10) / L11                                                          <L 2394>
    var_9 = wp::mul(var_4, var_3);
    var_10 = wp::sub(var_B21, var_9);
    var_11 = wp::div(var_10, var_8);
    // L22 = wp.sqrt(wp.max(MJ_MINVAL, B22 - (L20 * L20 + L21 * L21)))                        <L 2395>
    var_12 = wp::mul(var_4, var_4);
    var_13 = wp::mul(var_11, var_11);
    var_14 = wp::add(var_12, var_13);
    var_15 = wp::sub(var_B22, var_14);
    var_16 = wp::max(var_0, var_15);
    var_17 = wp::sqrt(var_16);
    // out[worldid, base + 0] = L00                                                           <L 2397>
    var_19 = wp::add(var_base, var_18);
    wp::array_store(var_out, var_worldid, var_19, var_2);
    // out[worldid, base + 1] = 0.0                                                           <L 2398>
    var_22 = wp::add(var_base, var_21);
    wp::array_store(var_out, var_worldid, var_22, var_20);
    // out[worldid, base + 2] = 0.0                                                           <L 2399>
    var_25 = wp::add(var_base, var_24);
    wp::array_store(var_out, var_worldid, var_25, var_23);
    // out[worldid, base + 3] = L10                                                           <L 2400>
    var_27 = wp::add(var_base, var_26);
    wp::array_store(var_out, var_worldid, var_27, var_3);
    // out[worldid, base + 4] = L11                                                           <L 2401>
    var_29 = wp::add(var_base, var_28);
    wp::array_store(var_out, var_worldid, var_29, var_8);
    // out[worldid, base + 5] = 0.0                                                           <L 2402>
    var_32 = wp::add(var_base, var_31);
    wp::array_store(var_out, var_worldid, var_32, var_30);
    // out[worldid, base + 6] = L20                                                           <L 2403>
    var_34 = wp::add(var_base, var_33);
    wp::array_store(var_out, var_worldid, var_34, var_4);
    // out[worldid, base + 7] = L21                                                           <L 2404>
    var_36 = wp::add(var_base, var_35);
    wp::array_store(var_out, var_worldid, var_36, var_11);
    // out[worldid, base + 8] = L22                                                           <L 2405>
    var_38 = wp::add(var_base, var_37);
    wp::array_store(var_out, var_worldid, var_38, var_17);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:2341
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> _extract_3x3_diag_block_0(
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_M_colind,
    wp::array_t<wp::int32> var_efm_K_rownnz,
    wp::array_t<wp::int32> var_efm_K_rowadr,
    wp::array_t<wp::int32> var_efm_K_colind,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_efm_K_val_in,
    wp::int32 var_i,
    wp::int32 var_worldid)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    wp::int32 var_11;
    wp::int32* var_12;
    wp::int32 var_13;
    wp::int32 var_14;
    wp::int32* var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::range_t var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    wp::int32* var_21;
    wp::int32 var_22;
    wp::int32 var_23;
    bool var_24;
    bool var_25;
    const wp::int32 var_26 = 3;
    wp::int32 var_27;
    bool var_28;
    wp::int32 var_29;
    wp::int32 var_30;
    wp::float32* var_31;
    wp::float32 var_32;
    wp::int32* var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::int32* var_36;
    wp::int32 var_37;
    wp::int32 var_38;
    wp::range_t var_39;
    wp::int32 var_40;
    wp::int32 var_41;
    wp::int32* var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    bool var_45;
    bool var_46;
    const wp::int32 var_47 = 3;
    wp::int32 var_48;
    bool var_49;
    wp::int32 var_50;
    wp::int32 var_51;
    wp::float32* var_52;
    wp::float32 var_53;
    const wp::int32 var_54 = 1;
    wp::int32 var_55;
    wp::int32* var_56;
    wp::int32 var_57;
    wp::int32 var_58;
    wp::int32* var_59;
    wp::int32 var_60;
    wp::int32 var_61;
    wp::range_t var_62;
    wp::int32 var_63;
    wp::int32 var_64;
    wp::int32* var_65;
    wp::int32 var_66;
    wp::int32 var_67;
    bool var_68;
    bool var_69;
    const wp::int32 var_70 = 3;
    wp::int32 var_71;
    bool var_72;
    wp::int32 var_73;
    wp::int32 var_74;
    wp::float32* var_75;
    wp::float32 var_76;
    wp::int32* var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    wp::int32* var_80;
    wp::int32 var_81;
    wp::int32 var_82;
    wp::range_t var_83;
    wp::int32 var_84;
    wp::int32 var_85;
    wp::int32* var_86;
    wp::int32 var_87;
    wp::int32 var_88;
    bool var_89;
    bool var_90;
    const wp::int32 var_91 = 3;
    wp::int32 var_92;
    bool var_93;
    wp::int32 var_94;
    wp::int32 var_95;
    wp::float32* var_96;
    wp::float32 var_97;
    const wp::int32 var_98 = 2;
    wp::int32 var_99;
    wp::int32* var_100;
    wp::int32 var_101;
    wp::int32 var_102;
    wp::int32* var_103;
    wp::int32 var_104;
    wp::int32 var_105;
    wp::range_t var_106;
    wp::int32 var_107;
    wp::int32 var_108;
    wp::int32* var_109;
    wp::int32 var_110;
    wp::int32 var_111;
    bool var_112;
    bool var_113;
    const wp::int32 var_114 = 3;
    wp::int32 var_115;
    bool var_116;
    wp::int32 var_117;
    wp::int32 var_118;
    wp::float32* var_119;
    wp::float32 var_120;
    wp::int32* var_121;
    wp::int32 var_122;
    wp::int32 var_123;
    wp::int32* var_124;
    wp::int32 var_125;
    wp::int32 var_126;
    wp::range_t var_127;
    wp::int32 var_128;
    wp::int32 var_129;
    wp::int32* var_130;
    wp::int32 var_131;
    wp::int32 var_132;
    bool var_133;
    bool var_134;
    const wp::int32 var_135 = 3;
    wp::int32 var_136;
    bool var_137;
    wp::int32 var_138;
    wp::int32 var_139;
    wp::float32* var_140;
    wp::float32 var_141;
    //---------
    // forward
    // def _extract_3x3_diag_block(                                                           <L 2342>
    // B = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                              <L 2357>
    var_9 = wp::mat_t<3, 3, wp::float32>(var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8);
    // for r in range(3):                                                                     <L 2358>
    // row = i + r                                                                            <L 2359>
    var_11 = wp::add(var_i, var_10);
    // m_adr = M_rowadr[row]                                                                  <L 2360>
    var_12 = wp::address(var_M_rowadr, var_11);
    var_14 = wp::load(var_12);
    var_13 = wp::copy(var_14);
    // m_nnz = M_rownnz[row]                                                                  <L 2361>
    var_15 = wp::address(var_M_rownnz, var_11);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    // for a in range(m_nnz):                                                                 <L 2362>
    var_18 = wp::range(var_16);
    start_for_0:;
        if (iter_cmp(var_18) == 0) goto end_for_0;
        var_19 = wp::iter_next(var_18);
        // c = M_colind[m_adr + a]                                                            <L 2363>
        var_20 = wp::add(var_13, var_19);
        var_21 = wp::address(var_M_colind, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if c >= i and c < i + 3:                                                           <L 2364>
        var_25 = (var_22 >= var_i);
        var_24 = var_25;
        if (var_24) {
            var_27 = wp::add(var_i, var_26);
            var_28 = (var_22 < var_27);
            var_24 = var_24 && var_28;
        }
        if (var_24) {
            // B[r, c - i] += M_in[worldid, m_adr + a]                                        <L 2365>
            var_29 = wp::sub(var_22, var_i);
            var_30 = wp::add(var_13, var_19);
            var_31 = wp::address(var_M_in, var_worldid, var_30);
            var_32 = wp::load(var_31);
            wp::add_inplace(var_9, var_10, var_29, var_32);
        }
        goto start_for_0;
    end_for_0:;
    // k_adr = efm_K_rowadr[row]                                                              <L 2367>
    var_33 = wp::address(var_efm_K_rowadr, var_11);
    var_35 = wp::load(var_33);
    var_34 = wp::copy(var_35);
    // k_nnz = efm_K_rownnz[row]                                                              <L 2368>
    var_36 = wp::address(var_efm_K_rownnz, var_11);
    var_38 = wp::load(var_36);
    var_37 = wp::copy(var_38);
    // for a in range(k_nnz):                                                                 <L 2369>
    var_39 = wp::range(var_37);
    start_for_2:;
        if (iter_cmp(var_39) == 0) goto end_for_2;
        var_40 = wp::iter_next(var_39);
        // c = efm_K_colind[k_adr + a]                                                        <L 2370>
        var_41 = wp::add(var_34, var_40);
        var_42 = wp::address(var_efm_K_colind, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // if c >= i and c < i + 3:                                                           <L 2371>
        var_46 = (var_43 >= var_i);
        var_45 = var_46;
        if (var_45) {
            var_48 = wp::add(var_i, var_47);
            var_49 = (var_43 < var_48);
            var_45 = var_45 && var_49;
        }
        if (var_45) {
            // B[r, c - i] += efm_K_val_in[worldid, k_adr + a]                                <L 2372>
            var_50 = wp::sub(var_43, var_i);
            var_51 = wp::add(var_34, var_40);
            var_52 = wp::address(var_efm_K_val_in, var_worldid, var_51);
            var_53 = wp::load(var_52);
            wp::add_inplace(var_9, var_10, var_50, var_53);
        }
        wp::assign(var_22, var_43);
        goto start_for_2;
    end_for_2:;
    // row = i + r                                                                            <L 2359>
    var_55 = wp::add(var_i, var_54);
    // m_adr = M_rowadr[row]                                                                  <L 2360>
    var_56 = wp::address(var_M_rowadr, var_55);
    var_58 = wp::load(var_56);
    var_57 = wp::copy(var_58);
    // m_nnz = M_rownnz[row]                                                                  <L 2361>
    var_59 = wp::address(var_M_rownnz, var_55);
    var_61 = wp::load(var_59);
    var_60 = wp::copy(var_61);
    // for a in range(m_nnz):                                                                 <L 2362>
    var_62 = wp::range(var_60);
    start_for_4:;
        if (iter_cmp(var_62) == 0) goto end_for_4;
        var_63 = wp::iter_next(var_62);
        // c = M_colind[m_adr + a]                                                            <L 2363>
        var_64 = wp::add(var_57, var_63);
        var_65 = wp::address(var_M_colind, var_64);
        var_67 = wp::load(var_65);
        var_66 = wp::copy(var_67);
        // if c >= i and c < i + 3:                                                           <L 2364>
        var_69 = (var_66 >= var_i);
        var_68 = var_69;
        if (var_68) {
            var_71 = wp::add(var_i, var_70);
            var_72 = (var_66 < var_71);
            var_68 = var_68 && var_72;
        }
        if (var_68) {
            // B[r, c - i] += M_in[worldid, m_adr + a]                                        <L 2365>
            var_73 = wp::sub(var_66, var_i);
            var_74 = wp::add(var_57, var_63);
            var_75 = wp::address(var_M_in, var_worldid, var_74);
            var_76 = wp::load(var_75);
            wp::add_inplace(var_9, var_54, var_73, var_76);
        }
        wp::assign(var_22, var_66);
        goto start_for_4;
    end_for_4:;
    // k_adr = efm_K_rowadr[row]                                                              <L 2367>
    var_77 = wp::address(var_efm_K_rowadr, var_55);
    var_79 = wp::load(var_77);
    var_78 = wp::copy(var_79);
    // k_nnz = efm_K_rownnz[row]                                                              <L 2368>
    var_80 = wp::address(var_efm_K_rownnz, var_55);
    var_82 = wp::load(var_80);
    var_81 = wp::copy(var_82);
    // for a in range(k_nnz):                                                                 <L 2369>
    var_83 = wp::range(var_81);
    start_for_6:;
        if (iter_cmp(var_83) == 0) goto end_for_6;
        var_84 = wp::iter_next(var_83);
        // c = efm_K_colind[k_adr + a]                                                        <L 2370>
        var_85 = wp::add(var_78, var_84);
        var_86 = wp::address(var_efm_K_colind, var_85);
        var_88 = wp::load(var_86);
        var_87 = wp::copy(var_88);
        // if c >= i and c < i + 3:                                                           <L 2371>
        var_90 = (var_87 >= var_i);
        var_89 = var_90;
        if (var_89) {
            var_92 = wp::add(var_i, var_91);
            var_93 = (var_87 < var_92);
            var_89 = var_89 && var_93;
        }
        if (var_89) {
            // B[r, c - i] += efm_K_val_in[worldid, k_adr + a]                                <L 2372>
            var_94 = wp::sub(var_87, var_i);
            var_95 = wp::add(var_78, var_84);
            var_96 = wp::address(var_efm_K_val_in, var_worldid, var_95);
            var_97 = wp::load(var_96);
            wp::add_inplace(var_9, var_54, var_94, var_97);
        }
        wp::assign(var_22, var_87);
        goto start_for_6;
    end_for_6:;
    // row = i + r                                                                            <L 2359>
    var_99 = wp::add(var_i, var_98);
    // m_adr = M_rowadr[row]                                                                  <L 2360>
    var_100 = wp::address(var_M_rowadr, var_99);
    var_102 = wp::load(var_100);
    var_101 = wp::copy(var_102);
    // m_nnz = M_rownnz[row]                                                                  <L 2361>
    var_103 = wp::address(var_M_rownnz, var_99);
    var_105 = wp::load(var_103);
    var_104 = wp::copy(var_105);
    // for a in range(m_nnz):                                                                 <L 2362>
    var_106 = wp::range(var_104);
    start_for_8:;
        if (iter_cmp(var_106) == 0) goto end_for_8;
        var_107 = wp::iter_next(var_106);
        // c = M_colind[m_adr + a]                                                            <L 2363>
        var_108 = wp::add(var_101, var_107);
        var_109 = wp::address(var_M_colind, var_108);
        var_111 = wp::load(var_109);
        var_110 = wp::copy(var_111);
        // if c >= i and c < i + 3:                                                           <L 2364>
        var_113 = (var_110 >= var_i);
        var_112 = var_113;
        if (var_112) {
            var_115 = wp::add(var_i, var_114);
            var_116 = (var_110 < var_115);
            var_112 = var_112 && var_116;
        }
        if (var_112) {
            // B[r, c - i] += M_in[worldid, m_adr + a]                                        <L 2365>
            var_117 = wp::sub(var_110, var_i);
            var_118 = wp::add(var_101, var_107);
            var_119 = wp::address(var_M_in, var_worldid, var_118);
            var_120 = wp::load(var_119);
            wp::add_inplace(var_9, var_98, var_117, var_120);
        }
        wp::assign(var_22, var_110);
        goto start_for_8;
    end_for_8:;
    // k_adr = efm_K_rowadr[row]                                                              <L 2367>
    var_121 = wp::address(var_efm_K_rowadr, var_99);
    var_123 = wp::load(var_121);
    var_122 = wp::copy(var_123);
    // k_nnz = efm_K_rownnz[row]                                                              <L 2368>
    var_124 = wp::address(var_efm_K_rownnz, var_99);
    var_126 = wp::load(var_124);
    var_125 = wp::copy(var_126);
    // for a in range(k_nnz):                                                                 <L 2369>
    var_127 = wp::range(var_125);
    start_for_10:;
        if (iter_cmp(var_127) == 0) goto end_for_10;
        var_128 = wp::iter_next(var_127);
        // c = efm_K_colind[k_adr + a]                                                        <L 2370>
        var_129 = wp::add(var_122, var_128);
        var_130 = wp::address(var_efm_K_colind, var_129);
        var_132 = wp::load(var_130);
        var_131 = wp::copy(var_132);
        // if c >= i and c < i + 3:                                                           <L 2371>
        var_134 = (var_131 >= var_i);
        var_133 = var_134;
        if (var_133) {
            var_136 = wp::add(var_i, var_135);
            var_137 = (var_131 < var_136);
            var_133 = var_133 && var_137;
        }
        if (var_133) {
            // B[r, c - i] += efm_K_val_in[worldid, k_adr + a]                                <L 2372>
            var_138 = wp::sub(var_131, var_i);
            var_139 = wp::add(var_122, var_128);
            var_140 = wp::address(var_efm_K_val_in, var_worldid, var_139);
            var_141 = wp::load(var_140);
            wp::add_inplace(var_9, var_98, var_138, var_141);
        }
        wp::assign(var_22, var_131);
        goto start_for_10;
    end_for_10:;
    // return B                                                                               <L 2373>
    return var_9;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:1981
static CUDA_CALLABLE wp::int32 _find_csr_col_0(
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::int32 var_target_col,
    wp::array_t<wp::int32> var_colind)
{
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    bool var_4;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    wp::int32 var_7;
    wp::int32* var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    bool var_11;
    bool var_12;
    const wp::int32 var_13 = 1;
    wp::int32 var_14;
    const wp::int32 var_15 = 1;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    const wp::int32 var_21 = -1;
    //---------
    // forward
    // def _find_csr_col(rowadr: int, rownnz: int, target_col: int, colind: wp.array[int]) -> int:       <L 1982>
    // lo = rowadr                                                                            <L 1983>
    var_0 = wp::copy(var_rowadr);
    // hi = rowadr + rownnz - 1                                                               <L 1984>
    var_1 = wp::add(var_rowadr, var_rownnz);
    var_3 = wp::sub(var_1, var_2);
    // while lo <= hi:                                                                        <L 1985>
    start_while_0:;
    var_4 = (var_0 <= var_3);
    if ((var_4) == false) goto end_while_0;
        // mid = (lo + hi) // 2                                                               <L 1986>
        var_5 = wp::add(var_0, var_3);
        var_7 = wp::floordiv(var_5, var_6);
        // c = colind[mid]                                                                    <L 1987>
        var_8 = wp::address(var_colind, var_7);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if c == target_col:                                                                <L 1988>
        var_11 = (var_9 == var_target_col);
        if (var_11) {
            // return mid                                                                     <L 1989>
            return var_7;
        }
        if (!var_11) {
            // elif c < target_col:                                                           <L 1990>
            var_12 = (var_9 < var_target_col);
            if (var_12) {
                // lo = mid + 1                                                               <L 1991>
                var_14 = wp::add(var_7, var_13);
            }
            if (!var_12) {
                // hi = mid - 1                                                               <L 1993>
                var_16 = wp::sub(var_7, var_15);
            }
            var_17 = wp::where(var_12, var_14, var_0);
            var_18 = wp::where(var_12, var_3, var_16);
        }
        var_19 = wp::where(var_11, var_0, var_17);
        var_20 = wp::where(var_11, var_3, var_18);
        wp::assign(var_0, var_19);
        wp::assign(var_3, var_20);
    goto start_while_0;
    end_while_0:;
    // return -1                                                                              <L 1994>
    return var_21;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:67
static CUDA_CALLABLE wp::quat_t<wp::float32> mat33_to_quat_polar_0(
    wp::mat_t<3, 3, wp::float32> var_F)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 1.0;
    wp::quat_t<wp::float32> var_4;
    const wp::int32 var_5 = 50;
    wp::range_t var_6;
    wp::int32 var_7;
    wp::mat_t<3, 3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::int32 var_12 = 1;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 2;
    wp::vec_t<3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32> var_16;
    const wp::int32 var_17 = 0;
    wp::vec_t<3, wp::float32> var_18;
    const wp::int32 var_19 = 1;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 2;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::float32 var_34 = 1e-10;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::float32 var_37;
    const wp::float32 var_38 = 1e-06;
    bool var_39;
    wp::vec_t<3, wp::float32> var_40;
    const wp::float32 var_41 = 0.5;
    wp::float32 var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::int32 var_51 = 2;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::quat_t<wp::float32> var_56;
    wp::quat_t<wp::float32> var_57;
    wp::quat_t<wp::float32> var_58;
    //---------
    // forward
    // def mat33_to_quat_polar(F: wp.mat33) -> wp.quat:                                       <L 68>
    // cell_quat = wp.quat(0.0, 0.0, 0.0, 1.0)                                                <L 69>
    var_4 = wp::quat_t<wp::float32>(var_0, var_1, var_2, var_3);
    // for _iter in range(50):                                                                <L 70>
    var_6 = wp::range(var_5);
    start_for_0:;
        if (iter_cmp(var_6) == 0) goto end_for_0;
        var_7 = wp::iter_next(var_6);
        // rot = wp.quat_to_matrix(cell_quat)                                                 <L 71>
        var_8 = wp::quat_to_matrix(var_4);
        // rot_t = wp.transpose(rot)                                                          <L 72>
        var_9 = wp::transpose(var_8);
        // col1_rot = rot_t[0]                                                                <L 73>
        var_11 = wp::extract(var_9, var_10);
        // col2_rot = rot_t[1]                                                                <L 74>
        var_13 = wp::extract(var_9, var_12);
        // col3_rot = rot_t[2]                                                                <L 75>
        var_15 = wp::extract(var_9, var_14);
        // F_t = wp.transpose(F)                                                              <L 76>
        var_16 = wp::transpose(var_F);
        // col1_mat = F_t[0]                                                                  <L 77>
        var_18 = wp::extract(var_16, var_17);
        // col2_mat = F_t[1]                                                                  <L 78>
        var_20 = wp::extract(var_16, var_19);
        // col3_mat = F_t[2]                                                                  <L 79>
        var_22 = wp::extract(var_16, var_21);
        // omega = wp.cross(col1_rot, col1_mat) + wp.cross(col2_rot, col2_mat) + wp.cross(col3_rot, col3_mat)       <L 81>
        var_23 = wp::cross(var_11, var_18);
        var_24 = wp::cross(var_13, var_20);
        var_25 = wp::add(var_23, var_24);
        var_26 = wp::cross(var_15, var_22);
        var_27 = wp::add(var_25, var_26);
        // denom = wp.abs(wp.dot(col1_rot, col1_mat) + wp.dot(col2_rot, col2_mat) + wp.dot(col3_rot, col3_mat)) + 1.0e-10       <L 82>
        var_28 = wp::dot(var_11, var_18);
        var_29 = wp::dot(var_13, var_20);
        var_30 = wp::add(var_28, var_29);
        var_31 = wp::dot(var_15, var_22);
        var_32 = wp::add(var_30, var_31);
        var_33 = wp::abs(var_32);
        var_35 = wp::add(var_33, var_34);
        // omega = omega / denom                                                              <L 83>
        var_36 = wp::div(var_27, var_35);
        // w = wp.length(omega)                                                               <L 85>
        var_37 = wp::length(var_36);
        // if w < 1.0e-6:                                                                     <L 86>
        var_39 = (var_37 < var_38);
        if (var_39) {
            // break                                                                          <L 87>
            goto end_for_0;
        }
        // axis = omega / w                                                                   <L 89>
        var_40 = wp::div(var_36, var_37);
        // half_w = 0.5 * w                                                                   <L 90>
        var_42 = wp::mul(var_41, var_37);
        // qrot = wp.quat(                                                                    <L 91>
        // axis[0] * wp.sin(half_w),                                                          <L 92>
        var_44 = wp::extract(var_40, var_43);
        var_45 = wp::sin(var_42);
        var_46 = wp::mul(var_44, var_45);
        // axis[1] * wp.sin(half_w),                                                          <L 93>
        var_48 = wp::extract(var_40, var_47);
        var_49 = wp::sin(var_42);
        var_50 = wp::mul(var_48, var_49);
        // axis[2] * wp.sin(half_w),                                                          <L 94>
        var_52 = wp::extract(var_40, var_51);
        var_53 = wp::sin(var_42);
        var_54 = wp::mul(var_52, var_53);
        // wp.cos(half_w),                                                                    <L 95>
        var_55 = wp::cos(var_42);
        var_56 = wp::quat_t<wp::float32>(var_46, var_50, var_54, var_55);
        // cell_quat = wp.normalize(qrot * cell_quat)                                         <L 97>
        var_57 = wp::mul(var_56, var_4);
        var_58 = wp::normalize(var_57);
        wp::assign(var_4, var_58);
        goto start_for_0;
    end_for_0:;
    // return cell_quat                                                                       <L 98>
    return var_4;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:101
static CUDA_CALLABLE wp::quat_t<wp::float32> compute_interp_cell_quat_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::int32 var_order,
    wp::int32 var_ci,
    wp::int32 var_cj,
    wp::int32 var_ck,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_ny_g,
    wp::int32 var_nz_g,
    wp::int32 var_nstart,
    wp::int32 var_worldid)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    wp::int32 var_3;
    wp::int32 var_4;
    const wp::int32 var_5 = 1;
    wp::int32 var_6;
    wp::int32 var_7;
    const wp::float32 var_8 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    wp::int32 var_11;
    const wp::int32 var_12 = 1;
    wp::int32 var_13;
    wp::range_t var_14;
    wp::int32 var_15;
    const wp::int32 var_16 = 1;
    wp::int32 var_17;
    wp::range_t var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 1;
    wp::int32 var_21;
    wp::range_t var_22;
    wp::int32 var_23;
    bool var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    wp::int32 var_29;
    wp::int32 var_30;
    wp::int32 var_31;
    wp::int32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::int32 var_36;
    wp::vec_t<3, wp::float32>* var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    const wp::int32 var_40 = 0;
    bool var_41;
    const wp::int32 var_42 = -1;
    wp::float32 var_43;
    const wp::int32 var_44 = 1;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::int32 var_47 = 0;
    bool var_48;
    const wp::int32 var_49 = -1;
    wp::float32 var_50;
    const wp::int32 var_51 = 1;
    wp::float32 var_52;
    wp::float32 var_53;
    const wp::int32 var_54 = 0;
    bool var_55;
    const wp::int32 var_56 = -1;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    wp::float32 var_60;
    const wp::float32 var_61 = 0.5;
    wp::float32 var_62;
    const wp::float32 var_63 = 0.5;
    wp::float32 var_64;
    const wp::float32 var_65 = 0.5;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::int32 var_73 = 0;
    const wp::int32 var_74 = 0;
    wp::float32 var_75;
    wp::float32 var_76;
    const wp::int32 var_77 = 1;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 2;
    wp::float32 var_81;
    wp::float32 var_82;
    const wp::int32 var_83 = 1;
    const wp::int32 var_84 = 0;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 1;
    wp::float32 var_88;
    wp::float32 var_89;
    const wp::int32 var_90 = 2;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::int32 var_93 = 2;
    const wp::int32 var_94 = 0;
    wp::float32 var_95;
    wp::float32 var_96;
    const wp::int32 var_97 = 1;
    wp::float32 var_98;
    wp::float32 var_99;
    const wp::int32 var_100 = 2;
    wp::float32 var_101;
    wp::float32 var_102;
    const wp::int32 var_103 = 1;
    wp::int32 var_104;
    wp::int32 var_105;
    wp::quat_t<wp::float32> var_106;
    //---------
    // forward
    // def compute_interp_cell_quat(                                                          <L 102>
    // npc = (order + 1) * (order + 1) * (order + 1)                                          <L 118>
    var_1 = wp::add(var_order, var_0);
    var_3 = wp::add(var_order, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::add(var_order, var_5);
    var_7 = wp::mul(var_4, var_6);
    // F = wp.mat33(0.0)                                                                      <L 119>
    var_9 = wp::mat_t<3, 3, wp::float32>(var_8);
    // idx = int(0)                                                                           <L 120>
    var_11 = wp::int(var_10);
    // for li in range(order + 1):                                                            <L 121>
    var_13 = wp::add(var_order, var_12);
    var_14 = wp::range(var_13);
    start_for_0:;
        if (iter_cmp(var_14) == 0) goto end_for_0;
        var_15 = wp::iter_next(var_14);
        // for lj in range(order + 1):                                                        <L 122>
        var_17 = wp::add(var_order, var_16);
        var_18 = wp::range(var_17);
        start_for_2:;
            if (iter_cmp(var_18) == 0) goto end_for_2;
            var_19 = wp::iter_next(var_18);
            // for lk in range(order + 1):                                                    <L 123>
            var_21 = wp::add(var_order, var_20);
            var_22 = wp::range(var_21);
            start_for_4:;
                if (iter_cmp(var_22) == 0) goto end_for_4;
                var_23 = wp::iter_next(var_22);
                // if idx < npc:                                                              <L 124>
                var_24 = (var_11 < var_7);
                if (var_24) {
                    // gi = ci * order + li                                                   <L 125>
                    var_25 = wp::mul(var_ci, var_order);
                    var_26 = wp::add(var_25, var_15);
                    // gj = cj * order + lj                                                   <L 126>
                    var_27 = wp::mul(var_cj, var_order);
                    var_28 = wp::add(var_27, var_19);
                    // gk = ck * order + lk                                                   <L 127>
                    var_29 = wp::mul(var_ck, var_order);
                    var_30 = wp::add(var_29, var_23);
                    // gidx = gi * ny_g * nz_g + gj * nz_g + gk                               <L 128>
                    var_31 = wp::mul(var_26, var_ny_g);
                    var_32 = wp::mul(var_31, var_nz_g);
                    var_33 = wp::mul(var_28, var_nz_g);
                    var_34 = wp::add(var_32, var_33);
                    var_35 = wp::add(var_34, var_30);
                    // node_pos = flexnode_xpos_in[worldid, nstart + gidx]                    <L 130>
                    var_36 = wp::add(var_nstart, var_35);
                    var_37 = wp::address(var_flexnode_xpos_in, var_worldid, var_36);
                    var_39 = wp::load(var_37);
                    var_38 = wp::copy(var_39);
                    // dphi_x = float(-1) if li == 0 else float(1)                            <L 132>
                    var_41 = (var_15 == var_40);
                    if (var_41) {
                        var_43 = wp::float(var_42);
                    }
                    if (!var_41) {
                        var_45 = wp::float(var_44);
                    }
                    var_46 = wp::where(var_41, var_43, var_45);
                    // dphi_y = float(-1) if lj == 0 else float(1)                            <L 133>
                    var_48 = (var_19 == var_47);
                    if (var_48) {
                        var_50 = wp::float(var_49);
                    }
                    if (!var_48) {
                        var_52 = wp::float(var_51);
                    }
                    var_53 = wp::where(var_48, var_50, var_52);
                    // dphi_z = float(-1) if lk == 0 else float(1)                            <L 134>
                    var_55 = (var_23 == var_54);
                    if (var_55) {
                        var_57 = wp::float(var_56);
                    }
                    if (!var_55) {
                        var_59 = wp::float(var_58);
                    }
                    var_60 = wp::where(var_55, var_57, var_59);
                    // phi_x = float(0.5)                                                     <L 135>
                    var_62 = wp::float(var_61);
                    // phi_y = float(0.5)                                                     <L 136>
                    var_64 = wp::float(var_63);
                    // phi_z = float(0.5)                                                     <L 137>
                    var_66 = wp::float(var_65);
                    // grad_x = dphi_x * phi_y * phi_z                                        <L 139>
                    var_67 = wp::mul(var_46, var_64);
                    var_68 = wp::mul(var_67, var_66);
                    // grad_y = phi_x * dphi_y * phi_z                                        <L 140>
                    var_69 = wp::mul(var_62, var_53);
                    var_70 = wp::mul(var_69, var_66);
                    // grad_z = phi_x * phi_y * dphi_z                                        <L 141>
                    var_71 = wp::mul(var_62, var_64);
                    var_72 = wp::mul(var_71, var_60);
                    // for r in range(3):                                                     <L 143>
                    // F[r, 0] += node_pos[r] * grad_x                                        <L 144>
                    var_75 = wp::extract(var_38, var_73);
                    var_76 = wp::mul(var_75, var_68);
                    wp::add_inplace(var_9, var_73, var_74, var_76);
                    // F[r, 1] += node_pos[r] * grad_y                                        <L 145>
                    var_78 = wp::extract(var_38, var_73);
                    var_79 = wp::mul(var_78, var_70);
                    wp::add_inplace(var_9, var_73, var_77, var_79);
                    // F[r, 2] += node_pos[r] * grad_z                                        <L 146>
                    var_81 = wp::extract(var_38, var_73);
                    var_82 = wp::mul(var_81, var_72);
                    wp::add_inplace(var_9, var_73, var_80, var_82);
                    // F[r, 0] += node_pos[r] * grad_x                                        <L 144>
                    var_85 = wp::extract(var_38, var_83);
                    var_86 = wp::mul(var_85, var_68);
                    wp::add_inplace(var_9, var_83, var_84, var_86);
                    // F[r, 1] += node_pos[r] * grad_y                                        <L 145>
                    var_88 = wp::extract(var_38, var_83);
                    var_89 = wp::mul(var_88, var_70);
                    wp::add_inplace(var_9, var_83, var_87, var_89);
                    // F[r, 2] += node_pos[r] * grad_z                                        <L 146>
                    var_91 = wp::extract(var_38, var_83);
                    var_92 = wp::mul(var_91, var_72);
                    wp::add_inplace(var_9, var_83, var_90, var_92);
                    // F[r, 0] += node_pos[r] * grad_x                                        <L 144>
                    var_95 = wp::extract(var_38, var_93);
                    var_96 = wp::mul(var_95, var_68);
                    wp::add_inplace(var_9, var_93, var_94, var_96);
                    // F[r, 1] += node_pos[r] * grad_y                                        <L 145>
                    var_98 = wp::extract(var_38, var_93);
                    var_99 = wp::mul(var_98, var_70);
                    wp::add_inplace(var_9, var_93, var_97, var_99);
                    // F[r, 2] += node_pos[r] * grad_z                                        <L 146>
                    var_101 = wp::extract(var_38, var_93);
                    var_102 = wp::mul(var_101, var_72);
                    wp::add_inplace(var_9, var_93, var_100, var_102);
                    // idx += 1                                                               <L 148>
                    var_104 = wp::add(var_11, var_103);
                }
                var_105 = wp::where(var_24, var_104, var_11);
                wp::assign(var_11, var_105);
                goto start_for_4;
            end_for_4:;
            goto start_for_2;
        end_for_2:;
        goto start_for_0;
    end_for_0:;
    // return mat33_to_quat_polar(F)                                                          <L 150>
    var_106 = mat33_to_quat_polar_0(var_9);
    return var_106;
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:774
static CUDA_CALLABLE wp::float32 _deriv_ellipsoid_fluid_0(
    wp::int32 var_opt_integrator,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::float32> var_geom_fluid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_bodyid,
    wp::int32 var_rootid,
    wp::int32 var_geomadr,
    wp::int32 var_geomnum,
    wp::vec_t<6, wp::float32> var_cdof_i,
    wp::vec_t<6, wp::float32> var_cdof_j,
    wp::vec_t<3, wp::float32> var_wind,
    wp::float32 var_density,
    wp::float32 var_viscosity)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 4;
    bool var_1;
    bool var_2;
    const wp::int32 var_3 = 3;
    bool var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<6, wp::float32>* var_8;
    wp::vec_t<6, wp::float32> var_9;
    wp::vec_t<6, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32>* var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    const wp::float32 var_19 = 0.0;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 2;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    const wp::int32 var_28 = 3;
    wp::float32 var_29;
    const wp::int32 var_30 = 4;
    wp::float32 var_31;
    const wp::int32 var_32 = 5;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    const wp::int32 var_39 = 2;
    wp::float32 var_40;
    wp::vec_t<3, wp::float32> var_41;
    const wp::int32 var_42 = 3;
    wp::float32 var_43;
    const wp::int32 var_44 = 4;
    wp::float32 var_45;
    const wp::int32 var_46 = 5;
    wp::float32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::range_t var_49;
    wp::int32 var_50;
    wp::int32 var_51;
    const wp::int32 var_52 = 0;
    wp::float32* var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 0.0;
    bool var_57;
    wp::shape_t* var_58;
    const wp::int32 var_59 = 0;
    wp::int32 var_60;
    wp::shape_t var_61;
    wp::int32 var_62;
    wp::vec_t<3, wp::float32>* var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::int32* var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::int32 var_68;
    wp::mat_t<3, 3, wp::float32>* var_69;
    wp::mat_t<3, 3, wp::float32> var_70;
    wp::mat_t<3, 3, wp::float32> var_71;
    wp::mat_t<3, 3, wp::float32> var_72;
    wp::vec_t<3, wp::float32>* var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::vec_t<3, wp::float32> var_79;
    wp::vec_t<3, wp::float32> var_80;
    bool var_81;
    const wp::int32 var_82 = 0;
    wp::float32 var_83;
    const wp::float32 var_84 = 0.0;
    bool var_85;
    const wp::int32 var_86 = 1;
    wp::float32 var_87;
    const wp::float32 var_88 = 0.0;
    bool var_89;
    const wp::int32 var_90 = 2;
    wp::float32 var_91;
    const wp::float32 var_92 = 0.0;
    bool var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    const wp::int32 var_99 = 1;
    wp::float32* var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    const wp::int32 var_103 = 2;
    wp::float32* var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    const wp::int32 var_107 = 3;
    wp::float32* var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    const wp::int32 var_111 = 4;
    wp::float32* var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 5;
    wp::float32* var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    const wp::int32 var_119 = 6;
    wp::float32* var_120;
    const wp::int32 var_121 = 7;
    wp::float32* var_122;
    const wp::int32 var_123 = 8;
    wp::float32* var_124;
    wp::vec_t<3, wp::float32> var_125;
    wp::float32 var_126;
    wp::float32 var_127;
    wp::float32 var_128;
    const wp::int32 var_129 = 9;
    wp::float32* var_130;
    const wp::int32 var_131 = 10;
    wp::float32* var_132;
    const wp::int32 var_133 = 11;
    wp::float32* var_134;
    wp::vec_t<3, wp::float32> var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::float32 var_138;
    wp::mat_t<3, 3, wp::float32> var_139;
    wp::mat_t<3, 3, wp::float32> var_140;
    wp::mat_t<3, 3, wp::float32> var_141;
    wp::mat_t<3, 3, wp::float32> var_142;
    const wp::float32 var_143 = 0.5;
    wp::mat_t<3, 3, wp::float32> var_144;
    wp::mat_t<3, 3, wp::float32> var_145;
    wp::mat_t<3, 3, wp::float32> var_146;
    const wp::float32 var_147 = 0.5;
    wp::mat_t<3, 3, wp::float32> var_148;
    wp::mat_t<3, 3, wp::float32> var_149;
    wp::mat_t<3, 3, wp::float32> var_150;
    const wp::float32 var_151 = 0.5;
    wp::mat_t<3, 3, wp::float32> var_152;
    wp::mat_t<3, 3, wp::float32> var_153;
    wp::mat_t<3, 3, wp::float32> var_154;
    wp::mat_t<3, 3, wp::float32> var_155;
    wp::mat_t<3, 3, wp::float32> var_156;
    wp::mat_t<3, 3, wp::float32> var_157;
    wp::mat_t<3, 3, wp::float32> var_158;
    wp::mat_t<3, 3, wp::float32> var_159;
    wp::mat_t<3, 3, wp::float32> var_160;
    wp::vec_t<3, wp::float32> var_161;
    wp::vec_t<3, wp::float32> var_162;
    wp::vec_t<3, wp::float32> var_163;
    wp::vec_t<3, wp::float32> var_164;
    wp::vec_t<3, wp::float32> var_165;
    wp::vec_t<3, wp::float32> var_166;
    wp::vec_t<3, wp::float32> var_167;
    wp::vec_t<3, wp::float32> var_168;
    wp::vec_t<3, wp::float32> var_169;
    wp::vec_t<3, wp::float32> var_170;
    wp::vec_t<3, wp::float32> var_171;
    wp::vec_t<3, wp::float32> var_172;
    wp::vec_t<3, wp::float32> var_173;
    wp::vec_t<3, wp::float32> var_174;
    wp::vec_t<3, wp::float32> var_175;
    wp::float32 var_176;
    wp::float32 var_177;
    wp::float32 var_178;
    wp::float32 var_179;
    //---------
    // forward
    // def _deriv_ellipsoid_fluid(                                                            <L 775>
    // flg_dragonly = opt_integrator == IntegratorType.DISCRETE                               <L 803>
    var_1 = (var_opt_integrator == var_0);
    // is_symmetrize = flg_dragonly or opt_integrator == IntegratorType.IMPLICITFAST          <L 804>
    var_2 = var_1;
    if (!var_2) {
        var_4 = (var_opt_integrator == var_3);
        var_2 = var_2 || var_4;
    }
    // xipos = xipos_in[worldid, bodyid]                                                      <L 807>
    var_5 = wp::address(var_xipos_in, var_worldid, var_bodyid);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // cvel = cvel_in[worldid, bodyid]                                                        <L 808>
    var_8 = wp::address(var_cvel_in, var_worldid, var_bodyid);
    var_10 = wp::load(var_8);
    var_9 = wp::copy(var_10);
    // ang_global = wp.spatial_top(cvel)                                                      <L 809>
    var_11 = wp::spatial_top(var_9);
    // lin_global = wp.spatial_bottom(cvel)                                                   <L 810>
    var_12 = wp::spatial_bottom(var_9);
    // subtree_root = subtree_com_in[worldid, rootid]                                         <L 811>
    var_13 = wp::address(var_subtree_com_in, var_worldid, var_rootid);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)                      <L 812>
    var_16 = wp::sub(var_6, var_14);
    var_17 = wp::cross(var_16, var_11);
    var_18 = wp::sub(var_12, var_17);
    // qderiv_contrib = float(0.0)                                                            <L 814>
    var_20 = wp::float(var_19);
    // cdof_ang_i = wp.vec3(cdof_i[0], cdof_i[1], cdof_i[2])                                  <L 816>
    var_22 = wp::extract(var_cdof_i, var_21);
    var_24 = wp::extract(var_cdof_i, var_23);
    var_26 = wp::extract(var_cdof_i, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // cdof_lin_i = wp.vec3(cdof_i[3], cdof_i[4], cdof_i[5])                                  <L 817>
    var_29 = wp::extract(var_cdof_i, var_28);
    var_31 = wp::extract(var_cdof_i, var_30);
    var_33 = wp::extract(var_cdof_i, var_32);
    var_34 = wp::vec_t<3, wp::float32>(var_29, var_31, var_33);
    // cdof_ang_j = wp.vec3(cdof_j[0], cdof_j[1], cdof_j[2])                                  <L 818>
    var_36 = wp::extract(var_cdof_j, var_35);
    var_38 = wp::extract(var_cdof_j, var_37);
    var_40 = wp::extract(var_cdof_j, var_39);
    var_41 = wp::vec_t<3, wp::float32>(var_36, var_38, var_40);
    // cdof_lin_j = wp.vec3(cdof_j[3], cdof_j[4], cdof_j[5])                                  <L 819>
    var_43 = wp::extract(var_cdof_j, var_42);
    var_45 = wp::extract(var_cdof_j, var_44);
    var_47 = wp::extract(var_cdof_j, var_46);
    var_48 = wp::vec_t<3, wp::float32>(var_43, var_45, var_47);
    // for g in range(geomnum):                                                               <L 821>
    var_49 = wp::range(var_geomnum);
    start_for_0:;
        if (iter_cmp(var_49) == 0) goto end_for_0;
        var_50 = wp::iter_next(var_49);
        // geomid = geomadr + g                                                               <L 822>
        var_51 = wp::add(var_geomadr, var_50);
        // coef = geom_fluid[geomid, 0]                                                       <L 823>
        var_53 = wp::address(var_geom_fluid, var_51, var_52);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // if coef <= 0.0:                                                                    <L 824>
        var_57 = (var_54 <= var_56);
        if (var_57) {
            // continue                                                                       <L 825>
            goto start_for_0;
        }
        // size = geom_size[worldid % geom_size.shape[0], geomid]                             <L 827>
        var_58 = &(var_geom_size.shape);
        var_61 = wp::load(var_58);
        var_60 = wp::extract(var_61, var_59);
        var_62 = wp::mod(var_worldid, var_60);
        var_63 = wp::address(var_geom_size, var_62, var_51);
        var_65 = wp::load(var_63);
        var_64 = wp::copy(var_65);
        // semiaxes = geom_semiaxes(size, geom_type[geomid])                                  <L 828>
        var_66 = wp::address(var_geom_type, var_51);
        var_68 = wp::load(var_66);
        var_67 = geom_semiaxes_0(var_64, var_68);
        // geom_rot = geom_xmat_in[worldid, geomid]                                           <L 829>
        var_69 = wp::address(var_geom_xmat_in, var_worldid, var_51);
        var_71 = wp::load(var_69);
        var_70 = wp::copy(var_71);
        // geom_rotT = wp.transpose(geom_rot)                                                 <L 830>
        var_72 = wp::transpose(var_70);
        // geom_pos = geom_xpos_in[worldid, geomid]                                           <L 831>
        var_73 = wp::address(var_geom_xpos_in, var_worldid, var_51);
        var_75 = wp::load(var_73);
        var_74 = wp::copy(var_75);
        // lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)                       <L 834>
        var_76 = wp::sub(var_74, var_6);
        var_77 = wp::cross(var_11, var_76);
        var_78 = wp::add(var_18, var_77);
        // l_ang = geom_rotT @ ang_global                                                     <L 835>
        var_79 = wp::mul(var_72, var_11);
        // l_lin = geom_rotT @ lin_point                                                      <L 836>
        var_80 = wp::mul(var_72, var_78);
        // if wind[0] != 0.0 or wind[1] != 0.0 or wind[2] != 0.0:                             <L 838>
        var_83 = wp::extract(var_wind, var_82);
        var_85 = (var_83 != var_84);
        var_81 = var_85;
        if (!var_81) {
            var_87 = wp::extract(var_wind, var_86);
            var_89 = (var_87 != var_88);
            var_81 = var_81 || var_89;
        }
        if (!var_81) {
            var_91 = wp::extract(var_wind, var_90);
            var_93 = (var_91 != var_92);
            var_81 = var_81 || var_93;
        }
        if (var_81) {
            // l_lin -= geom_rotT @ wind                                                      <L 839>
            var_94 = wp::mul(var_72, var_wind);
            var_95 = wp::sub(var_80, var_94);
        }
        var_96 = wp::where(var_81, var_95, var_80);
        // ang_vel = l_ang                                                                    <L 841>
        var_97 = wp::copy(var_79);
        // lin_vel = l_lin                                                                    <L 842>
        var_98 = wp::copy(var_96);
        // blunt_drag_coef = geom_fluid[geomid, 1]                                            <L 845>
        var_100 = wp::address(var_geom_fluid, var_51, var_99);
        var_102 = wp::load(var_100);
        var_101 = wp::copy(var_102);
        // slender_drag_coef = geom_fluid[geomid, 2]                                          <L 846>
        var_104 = wp::address(var_geom_fluid, var_51, var_103);
        var_106 = wp::load(var_104);
        var_105 = wp::copy(var_106);
        // ang_drag_coef = geom_fluid[geomid, 3]                                              <L 847>
        var_108 = wp::address(var_geom_fluid, var_51, var_107);
        var_110 = wp::load(var_108);
        var_109 = wp::copy(var_110);
        // kutta_lift_coef = geom_fluid[geomid, 4]                                            <L 848>
        var_112 = wp::address(var_geom_fluid, var_51, var_111);
        var_114 = wp::load(var_112);
        var_113 = wp::copy(var_114);
        // magnus_lift_coef = geom_fluid[geomid, 5]                                           <L 849>
        var_116 = wp::address(var_geom_fluid, var_51, var_115);
        var_118 = wp::load(var_116);
        var_117 = wp::copy(var_118);
        // virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])       <L 850>
        var_120 = wp::address(var_geom_fluid, var_51, var_119);
        var_122 = wp::address(var_geom_fluid, var_51, var_121);
        var_124 = wp::address(var_geom_fluid, var_51, var_123);
        var_126 = wp::load(var_120);
        var_127 = wp::load(var_122);
        var_128 = wp::load(var_124);
        var_125 = wp::vec_t<3, wp::float32>(var_126, var_127, var_128);
        // virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])       <L 851>
        var_130 = wp::address(var_geom_fluid, var_51, var_129);
        var_132 = wp::address(var_geom_fluid, var_51, var_131);
        var_134 = wp::address(var_geom_fluid, var_51, var_133);
        var_136 = wp::load(var_130);
        var_137 = wp::load(var_132);
        var_138 = wp::load(var_134);
        var_135 = wp::vec_t<3, wp::float32>(var_136, var_137, var_138);
        // B00, B01, B10, B11 = _geom_ellipsoid_fluid_B(                                      <L 853>
        // semiaxes,                                                                          <L 854>
        // blunt_drag_coef,                                                                   <L 855>
        // slender_drag_coef,                                                                 <L 856>
        // ang_drag_coef,                                                                     <L 857>
        // kutta_lift_coef,                                                                   <L 858>
        // magnus_lift_coef,                                                                  <L 859>
        // virtual_mass,                                                                      <L 860>
        // virtual_inertia,                                                                   <L 861>
        // ang_vel,                                                                           <L 862>
        // lin_vel,                                                                           <L 863>
        // density,                                                                           <L 864>
        // viscosity,                                                                         <L 865>
        // flg_dragonly=flg_dragonly,                                                         <L 866>
        _geom_ellipsoid_fluid_B_0(var_67, var_101, var_105, var_109, var_113, var_117, var_125, var_135, var_97, var_98, var_density, var_viscosity, var_1, var_139, var_140, var_141, var_142);
        // if is_symmetrize:                                                                  <L 870>
        if (var_2) {
            // B00 = 0.5 * (B00 + wp.transpose(B00))                                          <L 871>
            var_144 = wp::transpose(var_139);
            var_145 = wp::add(var_139, var_144);
            var_146 = wp::mul(var_143, var_145);
            // B11 = 0.5 * (B11 + wp.transpose(B11))                                          <L 872>
            var_148 = wp::transpose(var_142);
            var_149 = wp::add(var_142, var_148);
            var_150 = wp::mul(var_147, var_149);
            // B01_sym = 0.5 * (B01 + wp.transpose(B10))                                      <L 873>
            var_152 = wp::transpose(var_141);
            var_153 = wp::add(var_140, var_152);
            var_154 = wp::mul(var_151, var_153);
            // B01 = B01_sym                                                                  <L 874>
            var_155 = wp::copy(var_154);
            // B10 = wp.transpose(B01_sym)                                                    <L 875>
            var_156 = wp::transpose(var_154);
        }
        var_157 = wp::where(var_2, var_146, var_139);
        var_158 = wp::where(var_2, var_155, var_140);
        var_159 = wp::where(var_2, var_156, var_141);
        var_160 = wp::where(var_2, var_150, var_142);
        // offset = geom_pos - subtree_root                                                   <L 878>
        var_161 = wp::sub(var_74, var_14);
        // jac_p_i = cdof_lin_i + wp.cross(cdof_ang_i, offset)                                <L 880>
        var_162 = wp::cross(var_27, var_161);
        var_163 = wp::add(var_34, var_162);
        // la_i = geom_rotT @ cdof_ang_i                                                      <L 881>
        var_164 = wp::mul(var_72, var_27);
        // ll_i = geom_rotT @ jac_p_i                                                         <L 882>
        var_165 = wp::mul(var_72, var_163);
        // jac_p_j = cdof_lin_j + wp.cross(cdof_ang_j, offset)                                <L 884>
        var_166 = wp::cross(var_41, var_161);
        var_167 = wp::add(var_48, var_166);
        // la_j = geom_rotT @ cdof_ang_j                                                      <L 885>
        var_168 = wp::mul(var_72, var_41);
        // ll_j = geom_rotT @ jac_p_j                                                         <L 886>
        var_169 = wp::mul(var_72, var_167);
        // Bj_ang = B00 @ la_j + B01 @ ll_j                                                   <L 889>
        var_170 = wp::mul(var_157, var_168);
        var_171 = wp::mul(var_158, var_169);
        var_172 = wp::add(var_170, var_171);
        // Bj_lin = B10 @ la_j + B11 @ ll_j                                                   <L 890>
        var_173 = wp::mul(var_159, var_168);
        var_174 = wp::mul(var_160, var_169);
        var_175 = wp::add(var_173, var_174);
        // qderiv_contrib += wp.dot(la_i, Bj_ang) + wp.dot(ll_i, Bj_lin)                      <L 893>
        var_176 = wp::dot(var_164, var_172);
        var_177 = wp::dot(var_165, var_175);
        var_178 = wp::add(var_176, var_177);
        var_179 = wp::add(var_20, var_178);
        wp::assign(var_20, var_179);
        goto start_for_0;
    end_for_0:;
    // return qderiv_contrib                                                                  <L 895>
    return var_20;
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\warp\_src\math.py:0
static CUDA_CALLABLE wp::float32 norm_l2_0(
    wp::vec_t<3, wp::float32> var_v)
{
    //---------
    // primal vars
    wp::float32 var_0;
    //---------
    // forward
    // def norm_l2(v: Any) -> float:                                                          <L 1>
    // return wp.length(v)                                                                    <L 12>
    var_0 = wp::length(var_v);
    return var_0;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:161
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::float32 var_1;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    const wp::int32 var_4 = 3;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::float32 var_10 = 0.0;
    wp::vec_t<3, wp::float32> var_11;
    const wp::float32 var_12 = 2.0;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 3.141592653589793;
    bool var_18;
    const wp::float32 var_19 = 2.0;
    const wp::float32 var_20 = 3.141592653589793;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    //---------
    // forward
    // def quat_to_vel(quat: wp.quat) -> wp.vec3:                                             <L 162>
    // axis = wp.vec3(quat[1], quat[2], quat[3])                                              <L 163>
    var_1 = wp::extract(var_quat, var_0);
    var_3 = wp::extract(var_quat, var_2);
    var_5 = wp::extract(var_quat, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // sin_a_2 = wp.norm_l2(axis)                                                             <L 164>
    var_7 = norm_l2_0(var_6);
    // if sin_a_2 == 0.0:                                                                     <L 166>
    var_9 = (var_7 == var_8);
    if (var_9) {
        // return wp.vec3(0.0)                                                                <L 167>
        var_11 = wp::vec_t<3, wp::float32>(var_10);
        return var_11;
    }
    // speed = 2.0 * wp.atan2(sin_a_2, quat[0])                                               <L 169>
    var_14 = wp::extract(var_quat, var_13);
    var_15 = wp::atan2(var_7, var_14);
    var_16 = wp::mul(var_12, var_15);
    // if speed > wp.pi:                                                                      <L 171>
    var_18 = (var_16 > var_17);
    if (var_18) {
        // speed -= 2.0 * wp.pi                                                               <L 172>
        var_21 = wp::mul(var_19, var_20);
        var_22 = wp::sub(var_16, var_21);
    }
    var_23 = wp::where(var_18, var_22, var_16);
    // return axis * speed / sin_a_2                                                          <L 174>
    var_24 = wp::mul(var_6, var_23);
    var_25 = wp::div(var_24, var_7);
    return var_25;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:177
static CUDA_CALLABLE wp::vec_t<3, wp::float32> quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 2;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 3;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::quat_t<wp::float32> var_11;
    wp::quat_t<wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    //---------
    // forward
    // def quat_sub(qa: wp.quat, qb: wp.quat) -> wp.vec3:                                     <L 178>
    // qneg = wp.quat(qb[0], -qb[1], -qb[2], -qb[3])                                          <L 181>
    var_1 = wp::extract(var_qb, var_0);
    var_3 = wp::extract(var_qb, var_2);
    var_4 = wp::neg(var_3);
    var_6 = wp::extract(var_qb, var_5);
    var_7 = wp::neg(var_6);
    var_9 = wp::extract(var_qb, var_8);
    var_10 = wp::neg(var_9);
    var_11 = wp::quat_t<wp::float32>(var_1, var_4, var_7, var_10);
    // qdif = mul_quat(qneg, qa)                                                              <L 182>
    var_12 = mul_quat_0(var_11, var_qa);
    // return quat_to_vel(qdif)                                                               <L 185>
    var_13 = quat_to_vel_0(var_12);
    return var_13;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:120
static CUDA_CALLABLE wp::vec_t<6, wp::float32> inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 3;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 4;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 8;
    wp::float32 var_18;
    const wp::int32 var_19 = 4;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    const wp::int32 var_23 = 7;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    const wp::int32 var_29 = 3;
    wp::float32 var_30;
    const wp::int32 var_31 = 0;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    const wp::int32 var_36 = 1;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 5;
    wp::float32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 8;
    wp::float32 var_47;
    const wp::int32 var_48 = 3;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::int32 var_52 = 6;
    wp::float32 var_53;
    const wp::int32 var_54 = 5;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 4;
    wp::float32 var_59;
    const wp::int32 var_60 = 0;
    wp::float32 var_61;
    wp::float32 var_62;
    const wp::int32 var_63 = 5;
    wp::float32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::float32 var_70;
    const wp::int32 var_71 = 2;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    const wp::int32 var_75 = 7;
    wp::float32 var_76;
    const wp::int32 var_77 = 3;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 6;
    wp::float32 var_82;
    const wp::int32 var_83 = 4;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::int32 var_87 = 8;
    wp::float32 var_88;
    const wp::int32 var_89 = 1;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::int32 var_92 = 7;
    wp::float32 var_93;
    const wp::int32 var_94 = 2;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    const wp::int32 var_98 = 9;
    wp::float32 var_99;
    const wp::int32 var_100 = 3;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    const wp::int32 var_104 = 6;
    wp::float32 var_105;
    const wp::int32 var_106 = 2;
    wp::float32 var_107;
    wp::float32 var_108;
    const wp::int32 var_109 = 8;
    wp::float32 var_110;
    const wp::int32 var_111 = 0;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 9;
    wp::float32 var_116;
    const wp::int32 var_117 = 4;
    wp::float32 var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    const wp::int32 var_121 = 7;
    wp::float32 var_122;
    const wp::int32 var_123 = 0;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 6;
    wp::float32 var_127;
    const wp::int32 var_128 = 1;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    const wp::int32 var_132 = 9;
    wp::float32 var_133;
    const wp::int32 var_134 = 5;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::vec_t<6, wp::float32> var_138;
    //---------
    // forward
    // def inert_vec(i: types.vec10, v: wp.spatial_vector) -> wp.spatial_vector:              <L 121>
    // return wp.spatial_vector(                                                              <L 123>
    // i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],                   <L 124>
    var_1 = wp::extract(var_i, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_i, var_5);
    var_8 = wp::extract(var_v, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::add(var_4, var_9);
    var_12 = wp::extract(var_i, var_11);
    var_14 = wp::extract(var_v, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::add(var_10, var_15);
    var_18 = wp::extract(var_i, var_17);
    var_20 = wp::extract(var_v, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::sub(var_16, var_21);
    var_24 = wp::extract(var_i, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_28 = wp::add(var_22, var_27);
    // i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],                   <L 125>
    var_30 = wp::extract(var_i, var_29);
    var_32 = wp::extract(var_v, var_31);
    var_33 = wp::mul(var_30, var_32);
    var_35 = wp::extract(var_i, var_34);
    var_37 = wp::extract(var_v, var_36);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_33, var_38);
    var_41 = wp::extract(var_i, var_40);
    var_43 = wp::extract(var_v, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::add(var_39, var_44);
    var_47 = wp::extract(var_i, var_46);
    var_49 = wp::extract(var_v, var_48);
    var_50 = wp::mul(var_47, var_49);
    var_51 = wp::add(var_45, var_50);
    var_53 = wp::extract(var_i, var_52);
    var_55 = wp::extract(var_v, var_54);
    var_56 = wp::mul(var_53, var_55);
    var_57 = wp::sub(var_51, var_56);
    // i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],                   <L 126>
    var_59 = wp::extract(var_i, var_58);
    var_61 = wp::extract(var_v, var_60);
    var_62 = wp::mul(var_59, var_61);
    var_64 = wp::extract(var_i, var_63);
    var_66 = wp::extract(var_v, var_65);
    var_67 = wp::mul(var_64, var_66);
    var_68 = wp::add(var_62, var_67);
    var_70 = wp::extract(var_i, var_69);
    var_72 = wp::extract(var_v, var_71);
    var_73 = wp::mul(var_70, var_72);
    var_74 = wp::add(var_68, var_73);
    var_76 = wp::extract(var_i, var_75);
    var_78 = wp::extract(var_v, var_77);
    var_79 = wp::mul(var_76, var_78);
    var_80 = wp::sub(var_74, var_79);
    var_82 = wp::extract(var_i, var_81);
    var_84 = wp::extract(var_v, var_83);
    var_85 = wp::mul(var_82, var_84);
    var_86 = wp::add(var_80, var_85);
    // i[8] * v[1] - i[7] * v[2] + i[9] * v[3],                                               <L 127>
    var_88 = wp::extract(var_i, var_87);
    var_90 = wp::extract(var_v, var_89);
    var_91 = wp::mul(var_88, var_90);
    var_93 = wp::extract(var_i, var_92);
    var_95 = wp::extract(var_v, var_94);
    var_96 = wp::mul(var_93, var_95);
    var_97 = wp::sub(var_91, var_96);
    var_99 = wp::extract(var_i, var_98);
    var_101 = wp::extract(var_v, var_100);
    var_102 = wp::mul(var_99, var_101);
    var_103 = wp::add(var_97, var_102);
    // i[6] * v[2] - i[8] * v[0] + i[9] * v[4],                                               <L 128>
    var_105 = wp::extract(var_i, var_104);
    var_107 = wp::extract(var_v, var_106);
    var_108 = wp::mul(var_105, var_107);
    var_110 = wp::extract(var_i, var_109);
    var_112 = wp::extract(var_v, var_111);
    var_113 = wp::mul(var_110, var_112);
    var_114 = wp::sub(var_108, var_113);
    var_116 = wp::extract(var_i, var_115);
    var_118 = wp::extract(var_v, var_117);
    var_119 = wp::mul(var_116, var_118);
    var_120 = wp::add(var_114, var_119);
    // i[7] * v[0] - i[6] * v[1] + i[9] * v[5],                                               <L 129>
    var_122 = wp::extract(var_i, var_121);
    var_124 = wp::extract(var_v, var_123);
    var_125 = wp::mul(var_122, var_124);
    var_127 = wp::extract(var_i, var_126);
    var_129 = wp::extract(var_v, var_128);
    var_130 = wp::mul(var_127, var_129);
    var_131 = wp::sub(var_125, var_130);
    var_133 = wp::extract(var_i, var_132);
    var_135 = wp::extract(var_v, var_134);
    var_136 = wp::mul(var_133, var_135);
    var_137 = wp::add(var_131, var_136);
    var_138 = wp::vec_t<6, wp::float32>({var_28, var_57, var_86, var_103, var_120, var_137});
    return var_138;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:147
static CUDA_CALLABLE wp::vec_t<6, wp::float32> motion_cross_force_0(
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> var_f)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 3;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<6, wp::float32> var_32;
    //---------
    // forward
    // def motion_cross_force(v: wp.spatial_vector, f: wp.spatial_vector) -> wp.spatial_vector:       <L 148>
    // v0 = wp.vec3(v[0], v[1], v[2])                                                         <L 150>
    var_1 = wp::extract(var_v, var_0);
    var_3 = wp::extract(var_v, var_2);
    var_5 = wp::extract(var_v, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // v1 = wp.vec3(v[3], v[4], v[5])                                                         <L 151>
    var_8 = wp::extract(var_v, var_7);
    var_10 = wp::extract(var_v, var_9);
    var_12 = wp::extract(var_v, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_8, var_10, var_12);
    // f0 = wp.vec3(f[0], f[1], f[2])                                                         <L 152>
    var_15 = wp::extract(var_f, var_14);
    var_17 = wp::extract(var_f, var_16);
    var_19 = wp::extract(var_f, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_15, var_17, var_19);
    // f1 = wp.vec3(f[3], f[4], f[5])                                                         <L 153>
    var_22 = wp::extract(var_f, var_21);
    var_24 = wp::extract(var_f, var_23);
    var_26 = wp::extract(var_f, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // ang = wp.cross(v0, f0) + wp.cross(v1, f1)                                              <L 155>
    var_28 = wp::cross(var_6, var_20);
    var_29 = wp::cross(var_13, var_27);
    var_30 = wp::add(var_28, var_29);
    // vel = wp.cross(v0, f1)                                                                 <L 156>
    var_31 = wp::cross(var_6, var_27);
    // return wp.spatial_vector(ang, vel)                                                     <L 158>
    var_32 = wp::vec_t<6, wp::float32>(var_30, var_31);
    return var_32;
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:642
static CUDA_CALLABLE wp::float32 muscle_gain_vel_deriv_0(
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
    const wp::float32 var_54 = 2.0;
    wp::float32 var_55;
    const wp::float32 var_56 = 2.0;
    wp::float32 var_57;
    bool var_58;
    const wp::float32 var_59 = -2.0;
    wp::float32 var_60;
    const wp::float32 var_61 = 2.0;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    const wp::float32 var_66 = 0.0;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    //---------
    // forward
    // def muscle_gain_vel_deriv(len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 643>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 645>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 646>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 647>
    var_8 = wp::extract(var_prm, var_7);
    // lmin = prm[4]                                                                          <L 648>
    var_10 = wp::extract(var_prm, var_9);
    // lmax = prm[5]                                                                          <L 649>
    var_12 = wp::extract(var_prm, var_11);
    // vmax = prm[6]                                                                          <L 650>
    var_14 = wp::extract(var_prm, var_13);
    // fvmax = prm[8]                                                                         <L 651>
    var_16 = wp::extract(var_prm, var_15);
    // if force < 0.0:                                                                        <L 653>
    var_18 = (var_6 < var_17);
    if (var_18) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 654>
        var_20 = wp::max(var_19, var_acc0);
        var_21 = wp::div(var_8, var_20);
    }
    var_22 = wp::where(var_18, var_21, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 656>
    var_24 = wp::extract(var_lengthrange, var_23);
    var_26 = wp::extract(var_lengthrange, var_25);
    var_27 = wp::sub(var_24, var_26);
    var_29 = wp::extract(var_4, var_28);
    var_31 = wp::extract(var_4, var_30);
    var_32 = wp::sub(var_29, var_31);
    var_33 = wp::max(var_19, var_32);
    var_34 = wp::div(var_27, var_33);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 657>
    var_36 = wp::extract(var_4, var_35);
    var_38 = wp::extract(var_lengthrange, var_37);
    var_39 = wp::sub(var_len, var_38);
    var_40 = wp::max(var_19, var_34);
    var_41 = wp::div(var_39, var_40);
    var_42 = wp::add(var_36, var_41);
    // V = vel / wp.max(MJ_MINVAL, L0 * vmax)                                                 <L 658>
    var_43 = wp::mul(var_34, var_14);
    var_44 = wp::max(var_19, var_43);
    var_45 = wp::div(var_vel, var_44);
    // FL = muscle_gain_length(L, lmin, lmax)                                                 <L 660>
    var_46 = muscle_gain_length_0(var_42, var_10, var_12);
    // y = fvmax - 1.0                                                                        <L 662>
    var_48 = wp::sub(var_16, var_47);
    // if V <= -1.0:                                                                          <L 663>
    var_50 = (var_45 <= var_49);
    if (var_50) {
        // dFV = 0.0                                                                          <L 664>
    }
    if (!var_50) {
        // elif V <= 0.0:                                                                     <L 665>
        var_53 = (var_45 <= var_52);
        if (var_53) {
            // dFV = 2.0 * V + 2.0                                                            <L 666>
            var_55 = wp::mul(var_54, var_45);
            var_57 = wp::add(var_55, var_56);
        }
        if (!var_53) {
            // elif V <= y:                                                                   <L 667>
            var_58 = (var_45 <= var_48);
            if (var_58) {
                // dFV = (-2.0 * V + 2.0 * y) / wp.max(MJ_MINVAL, y)                          <L 668>
                var_60 = wp::mul(var_59, var_45);
                var_62 = wp::mul(var_61, var_48);
                var_63 = wp::add(var_60, var_62);
                var_64 = wp::max(var_19, var_48);
                var_65 = wp::div(var_63, var_64);
            }
            if (!var_58) {
                // dFV = 0.0                                                                  <L 670>
            }
            var_67 = wp::where(var_58, var_65, var_66);
        }
        var_68 = wp::where(var_53, var_57, var_67);
    }
    var_69 = wp::where(var_50, var_51, var_68);
    // return -force * FL * dFV / wp.max(MJ_MINVAL, L0 * vmax)                                <L 672>
    var_70 = wp::neg(var_22);
    var_71 = wp::mul(var_70, var_46);
    var_72 = wp::mul(var_71, var_69);
    var_73 = wp::mul(var_34, var_14);
    var_74 = wp::max(var_19, var_73);
    var_75 = wp::div(var_72, var_74);
    return var_75;
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:133
static CUDA_CALLABLE wp::vec_t<6, wp::float32> motion_cross_0(
    wp::vec_t<6, wp::float32> var_u,
    wp::vec_t<6, wp::float32> var_v)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 1;
    wp::float32 var_3;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 3;
    wp::float32 var_8;
    const wp::int32 var_9 = 4;
    wp::float32 var_10;
    const wp::int32 var_11 = 5;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 3;
    wp::float32 var_22;
    const wp::int32 var_23 = 4;
    wp::float32 var_24;
    const wp::int32 var_25 = 5;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<6, wp::float32> var_32;
    //---------
    // forward
    // def motion_cross(u: wp.spatial_vector, v: wp.spatial_vector) -> wp.spatial_vector:       <L 134>
    // u0 = wp.vec3(u[0], u[1], u[2])                                                         <L 136>
    var_1 = wp::extract(var_u, var_0);
    var_3 = wp::extract(var_u, var_2);
    var_5 = wp::extract(var_u, var_4);
    var_6 = wp::vec_t<3, wp::float32>(var_1, var_3, var_5);
    // u1 = wp.vec3(u[3], u[4], u[5])                                                         <L 137>
    var_8 = wp::extract(var_u, var_7);
    var_10 = wp::extract(var_u, var_9);
    var_12 = wp::extract(var_u, var_11);
    var_13 = wp::vec_t<3, wp::float32>(var_8, var_10, var_12);
    // v0 = wp.vec3(v[0], v[1], v[2])                                                         <L 138>
    var_15 = wp::extract(var_v, var_14);
    var_17 = wp::extract(var_v, var_16);
    var_19 = wp::extract(var_v, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_15, var_17, var_19);
    // v1 = wp.vec3(v[3], v[4], v[5])                                                         <L 139>
    var_22 = wp::extract(var_v, var_21);
    var_24 = wp::extract(var_v, var_23);
    var_26 = wp::extract(var_v, var_25);
    var_27 = wp::vec_t<3, wp::float32>(var_22, var_24, var_26);
    // ang = wp.cross(u0, v0)                                                                 <L 141>
    var_28 = wp::cross(var_6, var_20);
    // vel = wp.cross(u1, v0) + wp.cross(u0, v1)                                              <L 142>
    var_29 = wp::cross(var_13, var_20);
    var_30 = wp::cross(var_6, var_27);
    var_31 = wp::add(var_29, var_30);
    // return wp.spatial_vector(ang, vel)                                                     <L 144>
    var_32 = wp::vec_t<6, wp::float32>(var_28, var_31);
    return var_32;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:615
static CUDA_CALLABLE wp::float32 muscle_bias_len_deriv_0(
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
    const wp::float32 var_53 = 1.0;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    const wp::float32 var_57 = 1.0;
    const wp::float32 var_58 = 1.0;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    //---------
    // forward
    // def muscle_bias_len_deriv(len: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 616>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 618>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 619>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 620>
    var_8 = wp::extract(var_prm, var_7);
    // lmax = prm[5]                                                                          <L 621>
    var_10 = wp::extract(var_prm, var_9);
    // fpmax = prm[7]                                                                         <L 622>
    var_12 = wp::extract(var_prm, var_11);
    // if force < 0.0:                                                                        <L 624>
    var_14 = (var_6 < var_13);
    if (var_14) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 625>
        var_16 = wp::max(var_15, var_acc0);
        var_17 = wp::div(var_8, var_16);
    }
    var_18 = wp::where(var_14, var_17, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 627>
    var_20 = wp::extract(var_lengthrange, var_19);
    var_22 = wp::extract(var_lengthrange, var_21);
    var_23 = wp::sub(var_20, var_22);
    var_25 = wp::extract(var_4, var_24);
    var_27 = wp::extract(var_4, var_26);
    var_28 = wp::sub(var_25, var_27);
    var_29 = wp::max(var_15, var_28);
    var_30 = wp::div(var_23, var_29);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 628>
    var_32 = wp::extract(var_4, var_31);
    var_34 = wp::extract(var_lengthrange, var_33);
    var_35 = wp::sub(var_len, var_34);
    var_36 = wp::max(var_15, var_30);
    var_37 = wp::div(var_35, var_36);
    var_38 = wp::add(var_32, var_37);
    // b = 0.5 * (1.0 + lmax)                                                                 <L 630>
    var_41 = wp::add(var_40, var_10);
    var_42 = wp::mul(var_39, var_41);
    // if L <= 1.0:                                                                           <L 631>
    var_44 = (var_38 <= var_43);
    if (var_44) {
        // dFP = 0.0                                                                          <L 632>
    }
    if (!var_44) {
        // elif L <= b:                                                                       <L 633>
        var_46 = (var_38 <= var_42);
        if (var_46) {
            // x = (L - 1.0) / wp.max(MJ_MINVAL, b - 1.0)                                     <L 634>
            var_48 = wp::sub(var_38, var_47);
            var_50 = wp::sub(var_42, var_49);
            var_51 = wp::max(var_15, var_50);
            var_52 = wp::div(var_48, var_51);
            // dFP = x / wp.max(MJ_MINVAL, b - 1.0)                                           <L 635>
            var_54 = wp::sub(var_42, var_53);
            var_55 = wp::max(var_15, var_54);
            var_56 = wp::div(var_52, var_55);
        }
        if (!var_46) {
            // dFP = 1.0 / wp.max(MJ_MINVAL, b - 1.0)                                         <L 637>
            var_59 = wp::sub(var_42, var_58);
            var_60 = wp::max(var_15, var_59);
            var_61 = wp::div(var_57, var_60);
        }
        var_62 = wp::where(var_46, var_56, var_61);
    }
    var_63 = wp::where(var_44, var_45, var_62);
    // return -force * fpmax * dFP / wp.max(MJ_MINVAL, L0)                                    <L 639>
    var_64 = wp::neg(var_18);
    var_65 = wp::mul(var_64, var_12);
    var_66 = wp::mul(var_65, var_63);
    var_67 = wp::max(var_15, var_30);
    var_68 = wp::div(var_66, var_67);
    return var_68;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:559
static CUDA_CALLABLE wp::float32 muscle_gain_length_deriv_0(
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
    wp::float32 var_18;
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
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    bool var_34;
    const wp::float32 var_35 = 1.0;
    wp::float32 var_36;
    const wp::float32 var_37 = 1e-15;
    const wp::float32 var_38 = 1.0;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    const wp::float32 var_43 = 1.0;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    const wp::float32 var_48 = 1e-15;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
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
    // def muscle_gain_length_deriv(length: float, lmin: float, lmax: float) -> float:        <L 560>
    // if (lmin > length) or (length > lmax):                                                 <L 562>
    var_1 = (var_lmin > var_length);
    var_0 = var_1;
    if (!var_0) {
        var_2 = (var_length > var_lmax);
        var_0 = var_0 || var_2;
    }
    if (var_0) {
        // return 0.0                                                                         <L 563>
        return var_3;
    }
    // a = 0.5 * (lmin + 1.0)                                                                 <L 565>
    var_6 = wp::add(var_lmin, var_5);
    var_7 = wp::mul(var_4, var_6);
    // b = 0.5 * (1.0 + lmax)                                                                 <L 566>
    var_10 = wp::add(var_9, var_lmax);
    var_11 = wp::mul(var_8, var_10);
    // if length <= a:                                                                        <L 568>
    var_12 = (var_length <= var_7);
    if (var_12) {
        // x = (length - lmin) / wp.max(MJ_MINVAL, a - lmin)                                  <L 569>
        var_13 = wp::sub(var_length, var_lmin);
        var_15 = wp::sub(var_7, var_lmin);
        var_16 = wp::max(var_14, var_15);
        var_17 = wp::div(var_13, var_16);
        // return x / wp.max(MJ_MINVAL, a - lmin)                                             <L 570>
        var_18 = wp::sub(var_7, var_lmin);
        var_19 = wp::max(var_14, var_18);
        var_20 = wp::div(var_17, var_19);
        return var_20;
    }
    if (!var_12) {
        // elif length <= 1.0:                                                                <L 571>
        var_22 = (var_length <= var_21);
        if (var_22) {
            // x = (1.0 - length) / wp.max(MJ_MINVAL, 1.0 - a)                                <L 572>
            var_24 = wp::sub(var_23, var_length);
            var_27 = wp::sub(var_26, var_7);
            var_28 = wp::max(var_25, var_27);
            var_29 = wp::div(var_24, var_28);
            // return x / wp.max(MJ_MINVAL, 1.0 - a)                                          <L 573>
            var_31 = wp::sub(var_30, var_7);
            var_32 = wp::max(var_25, var_31);
            var_33 = wp::div(var_29, var_32);
            return var_33;
        }
        if (!var_22) {
            // elif length <= b:                                                              <L 574>
            var_34 = (var_length <= var_11);
            if (var_34) {
                // x = (length - 1.0) / wp.max(MJ_MINVAL, b - 1.0)                            <L 575>
                var_36 = wp::sub(var_length, var_35);
                var_39 = wp::sub(var_11, var_38);
                var_40 = wp::max(var_37, var_39);
                var_41 = wp::div(var_36, var_40);
                // return -x / wp.max(MJ_MINVAL, b - 1.0)                                     <L 576>
                var_42 = wp::neg(var_41);
                var_44 = wp::sub(var_11, var_43);
                var_45 = wp::max(var_37, var_44);
                var_46 = wp::div(var_42, var_45);
                return var_46;
            }
            if (!var_34) {
                // x = (lmax - length) / wp.max(MJ_MINVAL, lmax - b)                          <L 578>
                var_47 = wp::sub(var_lmax, var_length);
                var_49 = wp::sub(var_lmax, var_11);
                var_50 = wp::max(var_48, var_49);
                var_51 = wp::div(var_47, var_50);
                // return -x / wp.max(MJ_MINVAL, lmax - b)                                    <L 579>
                var_52 = wp::neg(var_51);
                var_53 = wp::sub(var_lmax, var_11);
                var_54 = wp::max(var_48, var_53);
                var_55 = wp::div(var_52, var_54);
                return var_55;
            }
            var_56 = wp::where(var_34, var_37, var_48);
            var_57 = wp::where(var_34, var_41, var_51);
        }
        var_58 = wp::where(var_22, var_25, var_56);
        var_59 = wp::where(var_22, var_29, var_57);
    }
    var_60 = wp::where(var_12, var_14, var_58);
    var_61 = wp::where(var_12, var_17, var_59);
    return {};
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:582
static CUDA_CALLABLE wp::float32 muscle_gain_len_deriv_0(
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
    wp::float32 var_73;
    wp::float32 var_74;
    //---------
    // forward
    // def muscle_gain_len_deriv(len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:       <L 583>
    // range_ = wp.vec2(prm[0], prm[1])                                                       <L 585>
    var_1 = wp::extract(var_prm, var_0);
    var_3 = wp::extract(var_prm, var_2);
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // force = prm[2]                                                                         <L 586>
    var_6 = wp::extract(var_prm, var_5);
    // scale = prm[3]                                                                         <L 587>
    var_8 = wp::extract(var_prm, var_7);
    // lmin = prm[4]                                                                          <L 588>
    var_10 = wp::extract(var_prm, var_9);
    // lmax = prm[5]                                                                          <L 589>
    var_12 = wp::extract(var_prm, var_11);
    // vmax = prm[6]                                                                          <L 590>
    var_14 = wp::extract(var_prm, var_13);
    // fvmax = prm[8]                                                                         <L 591>
    var_16 = wp::extract(var_prm, var_15);
    // if force < 0.0:                                                                        <L 593>
    var_18 = (var_6 < var_17);
    if (var_18) {
        // force = scale / wp.max(MJ_MINVAL, acc0)                                            <L 594>
        var_20 = wp::max(var_19, var_acc0);
        var_21 = wp::div(var_8, var_20);
    }
    var_22 = wp::where(var_18, var_21, var_6);
    // L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])       <L 596>
    var_24 = wp::extract(var_lengthrange, var_23);
    var_26 = wp::extract(var_lengthrange, var_25);
    var_27 = wp::sub(var_24, var_26);
    var_29 = wp::extract(var_4, var_28);
    var_31 = wp::extract(var_4, var_30);
    var_32 = wp::sub(var_29, var_31);
    var_33 = wp::max(var_19, var_32);
    var_34 = wp::div(var_27, var_33);
    // L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)                         <L 597>
    var_36 = wp::extract(var_4, var_35);
    var_38 = wp::extract(var_lengthrange, var_37);
    var_39 = wp::sub(var_len, var_38);
    var_40 = wp::max(var_19, var_34);
    var_41 = wp::div(var_39, var_40);
    var_42 = wp::add(var_36, var_41);
    // V = vel / wp.max(MJ_MINVAL, L0 * vmax)                                                 <L 598>
    var_43 = wp::mul(var_34, var_14);
    var_44 = wp::max(var_19, var_43);
    var_45 = wp::div(var_vel, var_44);
    // dFL = muscle_gain_length_deriv(L, lmin, lmax)                                          <L 600>
    var_46 = muscle_gain_length_deriv_0(var_42, var_10, var_12);
    // y = fvmax - 1.0                                                                        <L 602>
    var_48 = wp::sub(var_16, var_47);
    // if V <= -1.0:                                                                          <L 603>
    var_50 = (var_45 <= var_49);
    if (var_50) {
        // FV = 0.0                                                                           <L 604>
    }
    if (!var_50) {
        // elif V <= 0.0:                                                                     <L 605>
        var_53 = (var_45 <= var_52);
        if (var_53) {
            // FV = (V + 1.0) * (V + 1.0)                                                     <L 606>
            var_55 = wp::add(var_45, var_54);
            var_57 = wp::add(var_45, var_56);
            var_58 = wp::mul(var_55, var_57);
        }
        if (!var_53) {
            // elif V <= y:                                                                   <L 607>
            var_59 = (var_45 <= var_48);
            if (var_59) {
                // FV = fvmax - (y - V) * (y - V) / wp.max(MJ_MINVAL, y)                      <L 608>
                var_60 = wp::sub(var_48, var_45);
                var_61 = wp::sub(var_48, var_45);
                var_62 = wp::mul(var_60, var_61);
                var_63 = wp::max(var_19, var_48);
                var_64 = wp::div(var_62, var_63);
                var_65 = wp::sub(var_16, var_64);
            }
            if (!var_59) {
                // FV = fvmax                                                                 <L 610>
                var_66 = wp::copy(var_16);
            }
            var_67 = wp::where(var_59, var_65, var_66);
        }
        var_68 = wp::where(var_53, var_58, var_67);
    }
    var_69 = wp::where(var_50, var_51, var_68);
    // return -force * dFL * FV / wp.max(MJ_MINVAL, L0)                                       <L 612>
    var_70 = wp::neg(var_22);
    var_71 = wp::mul(var_70, var_46);
    var_72 = wp::mul(var_71, var_69);
    var_73 = wp::max(var_19, var_34);
    var_74 = wp::div(var_72, var_73);
    return var_74;
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:1059
static CUDA_CALLABLE void adj__get_jac_column_local_0(
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::vec_t<3, wp::float32> var_point_global,
    wp::int32 var_bodyid,
    wp::int32 var_dofid,
    wp::int32 var_worldid,
    wp::mat_t<3, 3, wp::float32> var_b_imat,
    wp::array_t<wp::int32> & adj_body_parentid,
    wp::array_t<wp::int32> & adj_body_rootid,
    wp::array_t<wp::int32> & adj_dof_bodyid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cdof_in,
    wp::vec_t<3, wp::float32> & adj_point_global,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_dofid,
    wp::int32 & adj_worldid,
    wp::mat_t<3, 3, wp::float32> & adj_b_imat,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:2376
static CUDA_CALLABLE void adj__write_cholesky33_0(
    wp::int32 var_worldid,
    wp::int32 var_base,
    wp::float32 var_B00,
    wp::float32 var_B10,
    wp::float32 var_B11,
    wp::float32 var_B20,
    wp::float32 var_B21,
    wp::float32 var_B22,
    wp::array_t<wp::float32> var_out,
    wp::int32 & adj_worldid,
    wp::int32 & adj_base,
    wp::float32 & adj_B00,
    wp::float32 & adj_B10,
    wp::float32 & adj_B11,
    wp::float32 & adj_B20,
    wp::float32 & adj_B21,
    wp::float32 & adj_B22,
    wp::array_t<wp::float32> & adj_out)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:2341
static CUDA_CALLABLE void adj__extract_3x3_diag_block_0(
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_M_colind,
    wp::array_t<wp::int32> var_efm_K_rownnz,
    wp::array_t<wp::int32> var_efm_K_rowadr,
    wp::array_t<wp::int32> var_efm_K_colind,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_efm_K_val_in,
    wp::int32 var_i,
    wp::int32 var_worldid,
    wp::array_t<wp::int32> & adj_M_rownnz,
    wp::array_t<wp::int32> & adj_M_rowadr,
    wp::array_t<wp::int32> & adj_M_colind,
    wp::array_t<wp::int32> & adj_efm_K_rownnz,
    wp::array_t<wp::int32> & adj_efm_K_rowadr,
    wp::array_t<wp::int32> & adj_efm_K_colind,
    wp::array_t<wp::float32> & adj_M_in,
    wp::array_t<wp::float32> & adj_efm_K_val_in,
    wp::int32 & adj_i,
    wp::int32 & adj_worldid,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:1981
static CUDA_CALLABLE void adj__find_csr_col_0(
    wp::int32 var_rowadr,
    wp::int32 var_rownnz,
    wp::int32 var_target_col,
    wp::array_t<wp::int32> var_colind,
    wp::int32 & adj_rowadr,
    wp::int32 & adj_rownnz,
    wp::int32 & adj_target_col,
    wp::array_t<wp::int32> & adj_colind,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:67
static CUDA_CALLABLE void adj_mat33_to_quat_polar_0(
    wp::mat_t<3, 3, wp::float32> var_F,
    wp::mat_t<3, 3, wp::float32> & adj_F,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\support.py:101
static CUDA_CALLABLE void adj_compute_interp_cell_quat_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::int32 var_order,
    wp::int32 var_ci,
    wp::int32 var_cj,
    wp::int32 var_ck,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_ny_g,
    wp::int32 var_nz_g,
    wp::int32 var_nstart,
    wp::int32 var_worldid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_flexnode_xpos_in,
    wp::int32 & adj_order,
    wp::int32 & adj_ci,
    wp::int32 & adj_cj,
    wp::int32 & adj_ck,
    wp::int32 & adj_cy,
    wp::int32 & adj_cz,
    wp::int32 & adj_ny_g,
    wp::int32 & adj_nz_g,
    wp::int32 & adj_nstart,
    wp::int32 & adj_worldid,
    wp::quat_t<wp::float32> & adj_ret)
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\derivative.py:774
static CUDA_CALLABLE void adj__deriv_ellipsoid_fluid_0(
    wp::int32 var_opt_integrator,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::float32> var_geom_fluid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::int32 var_worldid,
    wp::int32 var_bodyid,
    wp::int32 var_rootid,
    wp::int32 var_geomadr,
    wp::int32 var_geomnum,
    wp::vec_t<6, wp::float32> var_cdof_i,
    wp::vec_t<6, wp::float32> var_cdof_j,
    wp::vec_t<3, wp::float32> var_wind,
    wp::float32 var_density,
    wp::float32 var_viscosity,
    wp::int32 & adj_opt_integrator,
    wp::array_t<wp::int32> & adj_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_size,
    wp::array_t<wp::float32> & adj_geom_fluid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> & adj_cvel_in,
    wp::int32 & adj_worldid,
    wp::int32 & adj_bodyid,
    wp::int32 & adj_rootid,
    wp::int32 & adj_geomadr,
    wp::int32 & adj_geomnum,
    wp::vec_t<6, wp::float32> & adj_cdof_i,
    wp::vec_t<6, wp::float32> & adj_cdof_j,
    wp::vec_t<3, wp::float32> & adj_wind,
    wp::float32 & adj_density,
    wp::float32 & adj_viscosity,
    wp::float32 & adj_ret)
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\warp\_src\math.py:0
static CUDA_CALLABLE void adj_norm_l2_0(
    wp::vec_t<3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:161
static CUDA_CALLABLE void adj_quat_to_vel_0(
    wp::quat_t<wp::float32> var_quat,
    wp::quat_t<wp::float32> & adj_quat,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:177
static CUDA_CALLABLE void adj_quat_sub_0(
    wp::quat_t<wp::float32> var_qa,
    wp::quat_t<wp::float32> var_qb,
    wp::quat_t<wp::float32> & adj_qa,
    wp::quat_t<wp::float32> & adj_qb,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:120
static CUDA_CALLABLE void adj_inert_vec_0(
    wp::vec_t<10, wp::float32> var_i,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<10, wp::float32> & adj_i,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:147
static CUDA_CALLABLE void adj_motion_cross_force_0(
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> var_f,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_f,
    wp::vec_t<6, wp::float32> & adj_ret)
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:642
static CUDA_CALLABLE void adj_muscle_gain_vel_deriv_0(
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:133
static CUDA_CALLABLE void adj_motion_cross_0(
    wp::vec_t<6, wp::float32> var_u,
    wp::vec_t<6, wp::float32> var_v,
    wp::vec_t<6, wp::float32> & adj_u,
    wp::vec_t<6, wp::float32> & adj_v,
    wp::vec_t<6, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:615
static CUDA_CALLABLE void adj_muscle_bias_len_deriv_0(
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:559
static CUDA_CALLABLE void adj_muscle_gain_length_deriv_0(
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\util_misc.py:582
static CUDA_CALLABLE void adj_muscle_gain_len_deriv_0(
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



extern "C" __global__ void deriv_rne_body2jnt_sparse_08cf4a2e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::float32> var_timestep,
    wp::array_t<wp::int32> var_Di,
    wp::array_t<wp::int32> var_Dj,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcfrcbody_in,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::vec_t<6, wp::float32>* var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::vec_t<6, wp::float32> var_21;
        wp::vec_t<6, wp::float32>* var_22;
        wp::float32 var_23;
        wp::vec_t<6, wp::float32> var_24;
        wp::array_t<wp::float32> var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        //---------
        // forward
        // def deriv_rne_body2jnt_sparse(                                                         <L 506>
        // worldid, elemid = wp.tid()                                                             <L 520>
        builtin_tid2d(var_0, var_1);
        // dt = timestep[worldid % timestep.shape[0]]                                             <L 521>
        var_2 = &(var_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // i = Di[elemid]                                                                         <L 523>
        var_10 = wp::address(var_Di, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // j = Dj[elemid]                                                                         <L 524>
        var_13 = wp::address(var_Dj, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // body_i = dof_bodyid[i]                                                                 <L 526>
        var_16 = wp::address(var_dof_bodyid, var_11);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // dcfrc = Dcfrcbody_in[worldid, body_i, j]                                               <L 527>
        var_19 = wp::address(var_Dcfrcbody_in, var_0, var_17, var_14);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // term = wp.dot(cdof_in[worldid, i], dcfrc)                                              <L 528>
        var_22 = wp::address(var_cdof_in, var_0, var_11);
        var_24 = wp::load(var_22);
        var_23 = wp::dot(var_24, var_20);
        // wp.atomic_add(qDeriv_out[worldid], elemid, dt * term)                                  <L 530>
        var_25 = wp::view(var_qDeriv_out, var_0);
        var_26 = wp::mul(var_8, var_23);
        var_27 = wp::atomic_add(var_25, var_1, var_26);
}



extern "C" __global__ void _qderiv_box_fluid_15c70aee_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_wind,
    wp::array_t<wp::float32> var_opt_density,
    wp::array_t<wp::float32> var_opt_viscosity,
    wp::int32 var_opt_integrator,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::float32> var_body_mass,
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_inertia,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_body_fluid_box_adr,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_ximat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::int32> var_Mi,
    wp::array_t<wp::int32> var_Mj,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32 var_25;
        wp::shape_t* var_26;
        const wp::int32 var_27 = 0;
        wp::int32 var_28;
        wp::shape_t var_29;
        wp::int32 var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::shape_t* var_34;
        const wp::int32 var_35 = 0;
        wp::int32 var_36;
        wp::shape_t var_37;
        wp::int32 var_38;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::shape_t* var_42;
        const wp::int32 var_43 = 0;
        wp::int32 var_44;
        wp::shape_t var_45;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::shape_t* var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::shape_t var_53;
        wp::int32 var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        bool var_58;
        const wp::float32 var_59 = 0.0;
        bool var_60;
        const wp::float32 var_61 = 0.0;
        bool var_62;
        wp::vec_t<3, wp::float32>* var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::mat_t<3, 3, wp::float32>* var_66;
        wp::mat_t<3, 3, wp::float32> var_67;
        wp::mat_t<3, 3, wp::float32> var_68;
        wp::int32* var_69;
        wp::int32 var_70;
        wp::vec_t<3, wp::float32>* var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<6, wp::float32>* var_74;
        wp::vec_t<6, wp::float32> var_75;
        wp::vec_t<6, wp::float32> var_76;
        const wp::int32 var_77 = 0;
        wp::float32 var_78;
        const wp::int32 var_79 = 1;
        wp::float32 var_80;
        const wp::int32 var_81 = 2;
        wp::float32 var_82;
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
        wp::vec_t<3, wp::float32> var_93;
        wp::mat_t<3, 3, wp::float32> var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::vec_t<3, wp::float32> var_98;
        wp::vec_t<6, wp::float32> var_99;
        wp::mat_t<6, 6, wp::float32> var_100;
        wp::vec_t<6, wp::float32> var_101;
        wp::vec_t<6, wp::float32> var_102;
        wp::vec_t<6, wp::float32> var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        const wp::float32 var_106 = 0.0;
        bool var_107;
        wp::array_t<wp::float32> var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        //---------
        // forward
        // def _qderiv_box_fluid(                                                                 <L 1090>
        // worldid, fluid_idx, elemid = wp.tid()                                                  <L 1117>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid = body_fluid_box_adr[fluid_idx]                                                 <L 1119>
        var_3 = wp::address(var_body_fluid_box_adr, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // dofiid = Mi[elemid]                                                                    <L 1121>
        var_6 = wp::address(var_Mi, var_2);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // dofjid = Mj[elemid]                                                                    <L 1122>
        var_9 = wp::address(var_Mj, var_2);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // madr = M_elemid[dofiid, dofjid]                                                        <L 1124>
        var_12 = wp::address(var_M_elemid, var_7, var_10);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if madr < 0:                                                                           <L 1125>
        var_16 = (var_13 < var_15);
        if (var_16) {
            // return                                                                             <L 1126>
            return;
        }
        // bodyid_i = dof_bodyid[dofiid]                                                          <L 1128>
        var_17 = wp::address(var_dof_bodyid, var_7);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // if bodyid_i == 0:                                                                      <L 1130>
        var_21 = (var_18 == var_20);
        if (var_21) {
            // return                                                                             <L 1131>
            return;
        }
        // if body_isdofancestor[bodyid, dofiid] == 0:                                            <L 1133>
        var_22 = wp::address(var_body_isdofancestor, var_4, var_7);
        var_25 = wp::load(var_22);
        var_24 = (var_25 == var_23);
        if (var_24) {
            // return                                                                             <L 1134>
            return;
        }
        // wind = opt_wind[worldid % opt_wind.shape[0]]                                           <L 1136>
        var_26 = &(var_opt_wind.shape);
        var_29 = wp::load(var_26);
        var_28 = wp::extract(var_29, var_27);
        var_30 = wp::mod(var_0, var_28);
        var_31 = wp::address(var_opt_wind, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // density = opt_density[worldid % opt_density.shape[0]]                                  <L 1137>
        var_34 = &(var_opt_density.shape);
        var_37 = wp::load(var_34);
        var_36 = wp::extract(var_37, var_35);
        var_38 = wp::mod(var_0, var_36);
        var_39 = wp::address(var_opt_density, var_38);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]                            <L 1138>
        var_42 = &(var_opt_viscosity.shape);
        var_45 = wp::load(var_42);
        var_44 = wp::extract(var_45, var_43);
        var_46 = wp::mod(var_0, var_44);
        var_47 = wp::address(var_opt_viscosity, var_46);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 1139>
        var_50 = &(var_opt_timestep.shape);
        var_53 = wp::load(var_50);
        var_52 = wp::extract(var_53, var_51);
        var_54 = wp::mod(var_0, var_52);
        var_55 = wp::address(var_opt_timestep, var_54);
        var_57 = wp::load(var_55);
        var_56 = wp::copy(var_57);
        // if density <= 0.0 and viscosity <= 0.0:                                                <L 1141>
        var_60 = (var_40 <= var_59);
        var_58 = var_60;
        if (var_58) {
            var_62 = (var_48 <= var_61);
            var_58 = var_58 && var_62;
        }
        if (var_58) {
            // return                                                                             <L 1142>
            return;
        }
        // b_ipos = xipos_in[worldid, bodyid]                                                     <L 1145>
        var_63 = wp::address(var_xipos_in, var_0, var_4);
        var_65 = wp::load(var_63);
        var_64 = wp::copy(var_65);
        // b_imat = ximat_in[worldid, bodyid]                                                     <L 1146>
        var_66 = wp::address(var_ximat_in, var_0, var_4);
        var_68 = wp::load(var_66);
        var_67 = wp::copy(var_68);
        // subtree_root = subtree_com_in[worldid, body_rootid[bodyid]]                            <L 1147>
        var_69 = wp::address(var_body_rootid, var_4);
        var_70 = wp::load(var_69);
        var_71 = wp::address(var_subtree_com_in, var_0, var_70);
        var_73 = wp::load(var_71);
        var_72 = wp::copy(var_73);
        // vel_subtree = cvel_in[worldid, bodyid]                                                 <L 1149>
        var_74 = wp::address(var_cvel_in, var_0, var_4);
        var_76 = wp::load(var_74);
        var_75 = wp::copy(var_76);
        // v_subtree_ang = wp.vec3(vel_subtree[0], vel_subtree[1], vel_subtree[2])                <L 1150>
        var_78 = wp::extract(var_75, var_77);
        var_80 = wp::extract(var_75, var_79);
        var_82 = wp::extract(var_75, var_81);
        var_83 = wp::vec_t<3, wp::float32>(var_78, var_80, var_82);
        // v_subtree_lin = wp.vec3(vel_subtree[3], vel_subtree[4], vel_subtree[5])                <L 1151>
        var_85 = wp::extract(var_75, var_84);
        var_87 = wp::extract(var_75, var_86);
        var_89 = wp::extract(var_75, var_88);
        var_90 = wp::vec_t<3, wp::float32>(var_85, var_87, var_89);
        // lin_com = v_subtree_lin - wp.cross(b_ipos - subtree_root, v_subtree_ang)               <L 1153>
        var_91 = wp::sub(var_64, var_72);
        var_92 = wp::cross(var_91, var_83);
        var_93 = wp::sub(var_90, var_92);
        // b_imat_T = wp.transpose(b_imat)                                                        <L 1154>
        var_94 = wp::transpose(var_67);
        // v_local_ang = b_imat_T @ v_subtree_ang                                                 <L 1155>
        var_95 = wp::mul(var_94, var_83);
        // v_local_lin = b_imat_T @ lin_com                                                       <L 1156>
        var_96 = wp::mul(var_94, var_93);
        // wind_local = b_imat_T @ wind                                                           <L 1157>
        var_97 = wp::mul(var_94, var_32);
        // lvel = wp.spatial_vector(v_local_ang, v_local_lin - wind_local)                        <L 1159>
        var_98 = wp::sub(var_96, var_97);
        var_99 = wp::vec_t<6, wp::float32>(var_95, var_98);
        // B = _deriv_box_fluid(                                                                  <L 1161>
        // opt_integrator,                                                                        <L 1162>
        // body_mass,                                                                             <L 1163>
        // body_inertia,                                                                          <L 1164>
        // worldid,                                                                               <L 1165>
        // bodyid,                                                                                <L 1166>
        // lvel,                                                                                  <L 1167>
        // density,                                                                               <L 1168>
        // viscosity,                                                                             <L 1169>
        var_100 = _deriv_box_fluid_0(var_opt_integrator, var_body_mass, var_body_inertia, var_0, var_4, var_99, var_40, var_48);
        // J_i = _get_jac_column_local(                                                           <L 1173>
        // body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, b_ipos, bodyid, dofiid, worldid, b_imat       <L 1174>
        var_101 = _get_jac_column_local_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_64, var_4, var_7, var_0, var_67);
        // J_j = _get_jac_column_local(                                                           <L 1176>
        // body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, b_ipos, bodyid, dofjid, worldid, b_imat       <L 1177>
        var_102 = _get_jac_column_local_0(var_body_parentid, var_body_rootid, var_dof_bodyid, var_subtree_com_in, var_cdof_in, var_64, var_4, var_10, var_0, var_67);
        // contrib = wp.dot(J_i, B @ J_j) * timestep                                              <L 1180>
        var_103 = wp::mul(var_100, var_102);
        var_104 = wp::dot(var_101, var_103);
        var_105 = wp::mul(var_104, var_56);
        // if contrib != 0.0:                                                                     <L 1182>
        var_107 = (var_105 != var_106);
        if (var_107) {
            // wp.atomic_add(qDeriv_out[worldid], madr, -contrib)                                 <L 1183>
            var_108 = wp::view(var_qDeriv_out, var_0);
            var_109 = wp::neg(var_105);
            var_110 = wp::atomic_add(var_108, var_13, var_109);
        }
}



extern "C" __global__ void _eff_init_sdiag_24e84c44_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_efm_diag_in,
    wp::array_t<wp::float32> var_qH_out)
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
        wp::int32* var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def _eff_init_sdiag(                                                                   <L 1710>
        // worldid, dofid = wp.tid()                                                              <L 1719>
        builtin_tid2d(var_0, var_1);
        // diag_elem = M_rowadr[dofid] + M_rownnz[dofid] - 1                                      <L 1720>
        var_2 = wp::address(var_M_rowadr, var_1);
        var_3 = wp::address(var_M_rownnz, var_1);
        var_5 = wp::load(var_2);
        var_6 = wp::load(var_3);
        var_4 = wp::add(var_5, var_6);
        var_8 = wp::sub(var_4, var_7);
        // qH_out[worldid, diag_elem] += efm_diag_in[worldid, dofid]                              <L 1721>
        var_9 = wp::address(var_efm_diag_in, var_0, var_1);
        var_11 = wp::load(var_9);
        var_10 = wp::atomic_add(var_qH_out, var_0, var_8, var_11);
}



extern "C" __global__ void _eff_fold_efc_sparse_27bbfa83_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_efm_dofid,
    wp::array_t<wp::int32> var_efm_dofblk,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::int32> var_efc_J_rownnz_in,
    wp::array_t<wp::int32> var_efc_J_rowadr_in,
    wp::array_t<wp::int32> var_efc_J_colind_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_epB_out)
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
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        const wp::float32 var_8 = 0.0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::range_t var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 0;
        bool var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 0;
        wp::int32 var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::range_t var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 0;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::int32* var_43;
        bool var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 0;
        wp::int32 var_47;
        wp::float32* var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        const wp::int32 var_51 = 9;
        wp::int32 var_52;
        const wp::int32 var_53 = 3;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        //---------
        // forward
        // def _eff_fold_efc_sparse(                                                              <L 3658>
        // worldid, r = wp.tid()                                                                  <L 3672>
        builtin_tid2d(var_0, var_1);
        // if r >= nefc_in[worldid]:                                                              <L 3673>
        var_2 = wp::address(var_nefc_in, var_0);
        var_4 = wp::load(var_2);
        var_3 = (var_1 >= var_4);
        if (var_3) {
            // return                                                                             <L 3674>
            return;
        }
        // D = efc_D_in[worldid, r]                                                               <L 3675>
        var_5 = wp::address(var_efc_D_in, var_0, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if D == 0.0:                                                                           <L 3676>
        var_9 = (var_6 == var_8);
        if (var_9) {
            // return                                                                             <L 3677>
            return;
        }
        // adr = efc_J_rowadr_in[worldid, r]                                                      <L 3678>
        var_10 = wp::address(var_efc_J_rowadr_in, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // nnz = efc_J_rownnz_in[worldid, r]                                                      <L 3679>
        var_13 = wp::address(var_efc_J_rownnz_in, var_0, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // for a in range(nnz):                                                                   <L 3680>
        var_16 = wp::range(var_14);
        start_for_2:;
            if (iter_cmp(var_16) == 0) goto end_for_2;
            var_17 = wp::iter_next(var_16);
            // ia = efc_J_colind_in[worldid, 0, adr + a]                                          <L 3681>
            var_19 = wp::add(var_11, var_17);
            var_20 = wp::address(var_efc_J_colind_in, var_0, var_18, var_19);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // k = efm_dofblk[ia]                                                                 <L 3682>
            var_23 = wp::address(var_efm_dofblk, var_21);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if k < 0:                                                                          <L 3683>
            var_27 = (var_24 < var_26);
            if (var_27) {
                // continue                                                                       <L 3684>
                goto start_for_2;
            }
            // base = efm_dofid[k]                                                                <L 3685>
            var_28 = wp::address(var_efm_dofid, var_24);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // D_ja = D * efc_J_in[worldid, 0, adr + a]                                           <L 3686>
            var_32 = wp::add(var_11, var_17);
            var_33 = wp::address(var_efc_J_in, var_0, var_31, var_32);
            var_35 = wp::load(var_33);
            var_34 = wp::mul(var_6, var_35);
            // for b in range(nnz):                                                               <L 3687>
            var_36 = wp::range(var_14);
            start_for_4:;
                if (iter_cmp(var_36) == 0) goto end_for_4;
                var_37 = wp::iter_next(var_36);
                // ib = efc_J_colind_in[worldid, 0, adr + b]                                      <L 3688>
                var_39 = wp::add(var_11, var_37);
                var_40 = wp::address(var_efc_J_colind_in, var_0, var_38, var_39);
                var_42 = wp::load(var_40);
                var_41 = wp::copy(var_42);
                // if efm_dofblk[ib] == k:                                                        <L 3689>
                var_43 = wp::address(var_efm_dofblk, var_41);
                var_45 = wp::load(var_43);
                var_44 = (var_45 == var_24);
                if (var_44) {
                    // jb = efc_J_in[worldid, 0, adr + b]                                         <L 3690>
                    var_47 = wp::add(var_11, var_37);
                    var_48 = wp::address(var_efc_J_in, var_0, var_46, var_47);
                    var_50 = wp::load(var_48);
                    var_49 = wp::copy(var_50);
                    // wp.atomic_add(epB_out, worldid, 9 * k + 3 * (ia - base) + (ib - base), D_ja * jb)       <L 3691>
                    var_52 = wp::mul(var_51, var_24);
                    var_54 = wp::sub(var_21, var_29);
                    var_55 = wp::mul(var_53, var_54);
                    var_56 = wp::add(var_52, var_55);
                    var_57 = wp::sub(var_41, var_29);
                    var_58 = wp::add(var_56, var_57);
                    var_59 = wp::mul(var_34, var_49);
                    var_60 = wp::atomic_add(var_epB_out, var_0, var_58, var_59);
                }
                goto start_for_4;
            end_for_4:;
            goto start_for_2;
        end_for_2:;
}



extern "C" __global__ void _eff_factor_folded_blocks_2eaa6f9e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_epB_in,
    wp::array_t<wp::float32> var_epL_out)
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
        const wp::int32 var_2 = 9;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::float32* var_6;
        const wp::int32 var_7 = 3;
        wp::int32 var_8;
        wp::float32* var_9;
        const wp::int32 var_10 = 4;
        wp::int32 var_11;
        wp::float32* var_12;
        const wp::int32 var_13 = 6;
        wp::int32 var_14;
        wp::float32* var_15;
        const wp::int32 var_16 = 7;
        wp::int32 var_17;
        wp::float32* var_18;
        const wp::int32 var_19 = 8;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::float32 var_27;
        //---------
        // forward
        // def _eff_factor_folded_blocks(                                                         <L 3733>
        // worldid, k = wp.tid()                                                                  <L 3739>
        builtin_tid2d(var_0, var_1);
        // base = 9 * k                                                                           <L 3740>
        var_3 = wp::mul(var_2, var_1);
        // _write_cholesky33(                                                                     <L 3741>
        // worldid,                                                                               <L 3742>
        // base,                                                                                  <L 3743>
        // epB_in[worldid, base + 0],                                                             <L 3744>
        var_5 = wp::add(var_3, var_4);
        var_6 = wp::address(var_epB_in, var_0, var_5);
        // epB_in[worldid, base + 3],                                                             <L 3745>
        var_8 = wp::add(var_3, var_7);
        var_9 = wp::address(var_epB_in, var_0, var_8);
        // epB_in[worldid, base + 4],                                                             <L 3746>
        var_11 = wp::add(var_3, var_10);
        var_12 = wp::address(var_epB_in, var_0, var_11);
        // epB_in[worldid, base + 6],                                                             <L 3747>
        var_14 = wp::add(var_3, var_13);
        var_15 = wp::address(var_epB_in, var_0, var_14);
        // epB_in[worldid, base + 7],                                                             <L 3748>
        var_17 = wp::add(var_3, var_16);
        var_18 = wp::address(var_epB_in, var_0, var_17);
        // epB_in[worldid, base + 8],                                                             <L 3749>
        var_20 = wp::add(var_3, var_19);
        var_21 = wp::address(var_epB_in, var_0, var_20);
        // epL_out,                                                                               <L 3750>
        var_22 = wp::load(var_6);
        var_23 = wp::load(var_9);
        var_24 = wp::load(var_12);
        var_25 = wp::load(var_15);
        var_26 = wp::load(var_18);
        var_27 = wp::load(var_21);
        _write_cholesky33_0(var_0, var_3, var_22, var_23, var_24, var_25, var_26, var_27, var_epL_out);
}



extern "C" __global__ void _eff_build_blocks_raw_321b45a4_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_M_colind,
    wp::array_t<wp::int32> var_efm_K_rownnz,
    wp::array_t<wp::int32> var_efm_K_rowadr,
    wp::array_t<wp::int32> var_efm_K_colind,
    wp::array_t<wp::int32> var_efm_dofid,
    bool var_is_sparse,
    wp::array_t<wp::int32> var_nefc_in,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_efm_K_val_in,
    wp::array_t<wp::float32> var_efc_J_in,
    wp::array_t<wp::float32> var_efc_D_in,
    wp::array_t<wp::float32> var_epB_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::mat_t<3, 3, wp::float32> var_5;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::range_t var_10;
        wp::int32 var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        bool var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::float32* var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        bool var_30;
        const wp::float32 var_31 = 0.0;
        bool var_32;
        const wp::float32 var_33 = 0.0;
        bool var_34;
        const wp::float32 var_35 = 0.0;
        bool var_36;
        const wp::int32 var_37 = 0;
        const wp::int32 var_38 = 0;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 0;
        const wp::int32 var_42 = 1;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 0;
        const wp::int32 var_46 = 2;
        wp::float32 var_47;
        wp::float32 var_48;
        const wp::int32 var_49 = 1;
        const wp::int32 var_50 = 0;
        wp::float32 var_51;
        wp::float32 var_52;
        const wp::int32 var_53 = 1;
        const wp::int32 var_54 = 1;
        wp::float32 var_55;
        wp::float32 var_56;
        const wp::int32 var_57 = 1;
        const wp::int32 var_58 = 2;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 2;
        const wp::int32 var_62 = 0;
        wp::float32 var_63;
        wp::float32 var_64;
        const wp::int32 var_65 = 2;
        const wp::int32 var_66 = 1;
        wp::float32 var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 2;
        const wp::int32 var_70 = 2;
        wp::float32 var_71;
        wp::float32 var_72;
        const wp::int32 var_73 = 9;
        wp::int32 var_74;
        const wp::int32 var_75 = 0;
        const wp::int32 var_76 = 0;
        wp::float32 var_77;
        const wp::int32 var_78 = 3;
        wp::int32 var_79;
        wp::int32 var_80;
        wp::int32 var_81;
        const wp::int32 var_82 = 1;
        wp::float32 var_83;
        const wp::int32 var_84 = 3;
        wp::int32 var_85;
        wp::int32 var_86;
        wp::int32 var_87;
        const wp::int32 var_88 = 2;
        wp::float32 var_89;
        const wp::int32 var_90 = 3;
        wp::int32 var_91;
        wp::int32 var_92;
        wp::int32 var_93;
        const wp::int32 var_94 = 1;
        const wp::int32 var_95 = 0;
        wp::float32 var_96;
        const wp::int32 var_97 = 3;
        wp::int32 var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        const wp::int32 var_101 = 1;
        wp::float32 var_102;
        const wp::int32 var_103 = 3;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        const wp::int32 var_107 = 2;
        wp::float32 var_108;
        const wp::int32 var_109 = 3;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        const wp::int32 var_113 = 2;
        const wp::int32 var_114 = 0;
        wp::float32 var_115;
        const wp::int32 var_116 = 3;
        wp::int32 var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        const wp::int32 var_120 = 1;
        wp::float32 var_121;
        const wp::int32 var_122 = 3;
        wp::int32 var_123;
        wp::int32 var_124;
        wp::int32 var_125;
        const wp::int32 var_126 = 2;
        wp::float32 var_127;
        const wp::int32 var_128 = 3;
        wp::int32 var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        //---------
        // forward
        // def _eff_build_blocks_raw(                                                             <L 3539>
        // worldid, k = wp.tid()                                                                  <L 3558>
        builtin_tid2d(var_0, var_1);
        // i = efm_dofid[k]                                                                       <L 3559>
        var_2 = wp::address(var_efm_dofid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // B = _extract_3x3_diag_block(                                                           <L 3560>
        // M_rownnz, M_rowadr, M_colind, efm_K_rownnz, efm_K_rowadr, efm_K_colind, M_in, efm_K_val_in, i, worldid       <L 3561>
        var_5 = _extract_3x3_diag_block_0(var_M_rownnz, var_M_rowadr, var_M_colind, var_efm_K_rownnz, var_efm_K_rowadr, var_efm_K_colind, var_M_in, var_efm_K_val_in, var_3, var_0);
        // if not is_sparse:                                                                      <L 3564>
        var_6 = wp::unot(var_is_sparse);
        if (var_6) {
            // nefc = nefc_in[worldid]                                                            <L 3565>
            var_7 = wp::address(var_nefc_in, var_0);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // for r in range(nefc):                                                              <L 3566>
            var_10 = wp::range(var_8);
            start_for_0:;
                if (iter_cmp(var_10) == 0) goto end_for_0;
                var_11 = wp::iter_next(var_10);
                // D = efc_D_in[worldid, r]                                                       <L 3567>
                var_12 = wp::address(var_efc_D_in, var_0, var_11);
                var_14 = wp::load(var_12);
                var_13 = wp::copy(var_14);
                // if D != 0.0:                                                                   <L 3568>
                var_16 = (var_13 != var_15);
                if (var_16) {
                    // j0 = efc_J_in[worldid, r, i]                                               <L 3569>
                    var_17 = wp::address(var_efc_J_in, var_0, var_11, var_3);
                    var_19 = wp::load(var_17);
                    var_18 = wp::copy(var_19);
                    // j1 = efc_J_in[worldid, r, i + 1]                                           <L 3570>
                    var_21 = wp::add(var_3, var_20);
                    var_22 = wp::address(var_efc_J_in, var_0, var_11, var_21);
                    var_24 = wp::load(var_22);
                    var_23 = wp::copy(var_24);
                    // j2 = efc_J_in[worldid, r, i + 2]                                           <L 3571>
                    var_26 = wp::add(var_3, var_25);
                    var_27 = wp::address(var_efc_J_in, var_0, var_11, var_26);
                    var_29 = wp::load(var_27);
                    var_28 = wp::copy(var_29);
                    // if j0 != 0.0 or j1 != 0.0 or j2 != 0.0:                                    <L 3572>
                    var_32 = (var_18 != var_31);
                    var_30 = var_32;
                    if (!var_30) {
                        var_34 = (var_23 != var_33);
                        var_30 = var_30 || var_34;
                    }
                    if (!var_30) {
                        var_36 = (var_28 != var_35);
                        var_30 = var_30 || var_36;
                    }
                    if (var_30) {
                        // B[0, 0] += D * j0 * j0                                                 <L 3573>
                        var_39 = wp::mul(var_13, var_18);
                        var_40 = wp::mul(var_39, var_18);
                        wp::add_inplace(var_5, var_37, var_38, var_40);
                        // B[0, 1] += D * j0 * j1                                                 <L 3574>
                        var_43 = wp::mul(var_13, var_18);
                        var_44 = wp::mul(var_43, var_23);
                        wp::add_inplace(var_5, var_41, var_42, var_44);
                        // B[0, 2] += D * j0 * j2                                                 <L 3575>
                        var_47 = wp::mul(var_13, var_18);
                        var_48 = wp::mul(var_47, var_28);
                        wp::add_inplace(var_5, var_45, var_46, var_48);
                        // B[1, 0] += D * j1 * j0                                                 <L 3576>
                        var_51 = wp::mul(var_13, var_23);
                        var_52 = wp::mul(var_51, var_18);
                        wp::add_inplace(var_5, var_49, var_50, var_52);
                        // B[1, 1] += D * j1 * j1                                                 <L 3577>
                        var_55 = wp::mul(var_13, var_23);
                        var_56 = wp::mul(var_55, var_23);
                        wp::add_inplace(var_5, var_53, var_54, var_56);
                        // B[1, 2] += D * j1 * j2                                                 <L 3578>
                        var_59 = wp::mul(var_13, var_23);
                        var_60 = wp::mul(var_59, var_28);
                        wp::add_inplace(var_5, var_57, var_58, var_60);
                        // B[2, 0] += D * j2 * j0                                                 <L 3579>
                        var_63 = wp::mul(var_13, var_28);
                        var_64 = wp::mul(var_63, var_18);
                        wp::add_inplace(var_5, var_61, var_62, var_64);
                        // B[2, 1] += D * j2 * j1                                                 <L 3580>
                        var_67 = wp::mul(var_13, var_28);
                        var_68 = wp::mul(var_67, var_23);
                        wp::add_inplace(var_5, var_65, var_66, var_68);
                        // B[2, 2] += D * j2 * j2                                                 <L 3581>
                        var_71 = wp::mul(var_13, var_28);
                        var_72 = wp::mul(var_71, var_28);
                        wp::add_inplace(var_5, var_69, var_70, var_72);
                    }
                }
                goto start_for_0;
            end_for_0:;
        }
        // base = 9 * k                                                                           <L 3583>
        var_74 = wp::mul(var_73, var_1);
        // for r in range(3):                                                                     <L 3584>
        // for c in range(3):                                                                     <L 3585>
        // epB_out[worldid, base + 3 * r + c] = B[r, c]                                           <L 3586>
        var_77 = wp::extract(var_5, var_75, var_76);
        var_79 = wp::mul(var_78, var_75);
        var_80 = wp::add(var_74, var_79);
        var_81 = wp::add(var_80, var_76);
        wp::array_store(var_epB_out, var_0, var_81, var_77);
        var_83 = wp::extract(var_5, var_75, var_82);
        var_85 = wp::mul(var_84, var_75);
        var_86 = wp::add(var_74, var_85);
        var_87 = wp::add(var_86, var_82);
        wp::array_store(var_epB_out, var_0, var_87, var_83);
        var_89 = wp::extract(var_5, var_75, var_88);
        var_91 = wp::mul(var_90, var_75);
        var_92 = wp::add(var_74, var_91);
        var_93 = wp::add(var_92, var_88);
        wp::array_store(var_epB_out, var_0, var_93, var_89);
        // for c in range(3):                                                                     <L 3585>
        // epB_out[worldid, base + 3 * r + c] = B[r, c]                                           <L 3586>
        var_96 = wp::extract(var_5, var_94, var_95);
        var_98 = wp::mul(var_97, var_94);
        var_99 = wp::add(var_74, var_98);
        var_100 = wp::add(var_99, var_95);
        wp::array_store(var_epB_out, var_0, var_100, var_96);
        var_102 = wp::extract(var_5, var_94, var_101);
        var_104 = wp::mul(var_103, var_94);
        var_105 = wp::add(var_74, var_104);
        var_106 = wp::add(var_105, var_101);
        wp::array_store(var_epB_out, var_0, var_106, var_102);
        var_108 = wp::extract(var_5, var_94, var_107);
        var_110 = wp::mul(var_109, var_94);
        var_111 = wp::add(var_74, var_110);
        var_112 = wp::add(var_111, var_107);
        wp::array_store(var_epB_out, var_0, var_112, var_108);
        // for c in range(3):                                                                     <L 3585>
        // epB_out[worldid, base + 3 * r + c] = B[r, c]                                           <L 3586>
        var_115 = wp::extract(var_5, var_113, var_114);
        var_117 = wp::mul(var_116, var_113);
        var_118 = wp::add(var_74, var_117);
        var_119 = wp::add(var_118, var_114);
        wp::array_store(var_epB_out, var_0, var_119, var_115);
        var_121 = wp::extract(var_5, var_113, var_120);
        var_123 = wp::mul(var_122, var_113);
        var_124 = wp::add(var_74, var_123);
        var_125 = wp::add(var_124, var_120);
        wp::array_store(var_epB_out, var_0, var_125, var_121);
        var_127 = wp::extract(var_5, var_113, var_126);
        var_129 = wp::mul(var_128, var_113);
        var_130 = wp::add(var_74, var_129);
        var_131 = wp::add(var_130, var_126);
        wp::array_store(var_epB_out, var_0, var_131, var_127);
}



extern "C" __global__ void deriv_rne_cfrcbody_backward_32e4b713_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcfrcbody_out)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::vec_t<6, wp::float32>* var_9;
        wp::vec_t<6, wp::float32> var_10;
        wp::vec_t<6, wp::float32> var_11;
        wp::array_t<wp::vec_t<6, wp::float32>> var_12;
        wp::vec_t<6, wp::float32> var_13;
        //---------
        // forward
        // def deriv_rne_cfrcbody_backward(                                                       <L 486>
        // worldid, nodeid, dofid = wp.tid()                                                      <L 495>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid = body_tree_[nodeid]                                                            <L 496>
        var_3 = wp::address(var_body_tree_, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // pid = body_parentid[bodyid]                                                            <L 497>
        var_6 = wp::address(var_body_parentid, var_4);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // val = Dcfrcbody_out[worldid, bodyid, dofid]                                            <L 501>
        var_9 = wp::address(var_Dcfrcbody_out, var_0, var_4, var_2);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // wp.atomic_add(Dcfrcbody_out[worldid, pid], dofid, val)                                 <L 502>
        var_12 = wp::view(var_Dcfrcbody_out, var_0, var_7);
        var_13 = wp::atomic_add(var_12, var_2, var_10);
}



extern "C" __global__ void _eff_flex_bend_stiff_3f1f8139_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_flex_dim,
    wp::array_t<wp::int32> var_flex_interp,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_edgeadr,
    wp::array_t<wp::int32> var_flex_edgenum,
    wp::array_t<wp::int32> var_flex_bendingadr,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edgeflap,
    wp::array_t<wp::float32> var_flex_bending,
    wp::array_t<wp::float32> var_flex_damping,
    wp::array_t<bool> var_flex_rigid,
    wp::array_t<bool> var_flex_centered,
    wp::array_t<wp::int32> var_efm_K_rownnz,
    wp::array_t<wp::int32> var_efm_K_rowadr,
    wp::array_t<wp::int32> var_efm_K_colind,
    wp::array_t<wp::int32> var_flex_edgeflexid,
    wp::array_t<bool> var_flex_simple,
    wp::array_t<wp::quat_t<wp::float32>> var_xquat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::float32> var_efm_K_val_out)
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
        wp::float32 var_8;
        wp::float32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        bool var_15;
        wp::int32* var_16;
        const wp::int32 var_17 = 0;
        bool var_18;
        wp::int32 var_19;
        bool* var_20;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 2;
        bool var_24;
        wp::int32 var_25;
        bool* var_26;
        bool var_27;
        bool var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        const wp::int32 var_34 = 32;
        wp::int32 var_35;
        const wp::float32 var_36 = 0.0;
        wp::float32 var_37;
        wp::float32 var_38;
        const wp::int32 var_39 = 64;
        wp::int32 var_40;
        const wp::float32 var_41 = 0.0;
        wp::float32 var_42;
        wp::float32* var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::float32 var_47 = 0.0;
        bool var_48;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::vec_t<2, wp::int32>* var_55;
        wp::vec_t<2, wp::int32> var_56;
        wp::vec_t<2, wp::int32> var_57;
        wp::vec_t<2, wp::int32>* var_58;
        wp::vec_t<2, wp::int32> var_59;
        wp::vec_t<2, wp::int32> var_60;
        const wp::int32 var_61 = 1;
        wp::int32 var_62;
        const wp::int32 var_63 = -1;
        bool var_64;
        const wp::int32 var_65 = 0;
        wp::int32 var_66;
        const wp::int32 var_67 = 1;
        wp::int32 var_68;
        const wp::int32 var_69 = 0;
        wp::int32 var_70;
        const wp::int32 var_71 = 1;
        wp::int32 var_72;
        wp::vec_t<4, wp::int32> var_73;
        const wp::int32 var_74 = 17;
        wp::int32 var_75;
        wp::int32 var_76;
        const wp::int32 var_77 = -1;
        const wp::int32 var_78 = -1;
        const wp::int32 var_79 = -1;
        const wp::int32 var_80 = -1;
        wp::vec_t<4, wp::int32> var_81;
        const wp::int32 var_82 = -1;
        const wp::int32 var_83 = -1;
        const wp::int32 var_84 = -1;
        const wp::int32 var_85 = -1;
        wp::vec_t<4, wp::int32> var_86;
        const wp::int32 var_87 = 0;
        wp::int32 var_88;
        wp::int32 var_89;
        wp::int32* var_90;
        wp::int32 var_91;
        wp::int32* var_92;
        wp::int32 var_93;
        wp::int32 var_94;
        wp::int32* var_95;
        const wp::int32 var_96 = 3;
        bool var_97;
        wp::int32 var_98;
        wp::int32* var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        const wp::int32 var_102 = 1;
        wp::int32 var_103;
        wp::int32 var_104;
        wp::int32* var_105;
        wp::int32 var_106;
        wp::int32* var_107;
        wp::int32 var_108;
        wp::int32 var_109;
        wp::int32* var_110;
        const wp::int32 var_111 = 3;
        bool var_112;
        wp::int32 var_113;
        wp::int32* var_114;
        wp::int32 var_115;
        wp::int32 var_116;
        const wp::int32 var_117 = 2;
        wp::int32 var_118;
        wp::int32 var_119;
        wp::int32* var_120;
        wp::int32 var_121;
        wp::int32* var_122;
        wp::int32 var_123;
        wp::int32 var_124;
        wp::int32* var_125;
        const wp::int32 var_126 = 3;
        bool var_127;
        wp::int32 var_128;
        wp::int32* var_129;
        wp::int32 var_130;
        wp::int32 var_131;
        const wp::int32 var_132 = 3;
        wp::int32 var_133;
        wp::int32 var_134;
        wp::int32* var_135;
        wp::int32 var_136;
        wp::int32* var_137;
        wp::int32 var_138;
        wp::int32 var_139;
        wp::int32* var_140;
        const wp::int32 var_141 = 3;
        bool var_142;
        wp::int32 var_143;
        wp::int32* var_144;
        wp::int32 var_145;
        wp::int32 var_146;
        const wp::int32 var_147 = 0;
        const bool var_148 = true;
        wp::int32 var_149;
        bool var_150;
        const wp::int32 var_151 = 0;
        bool var_152;
        wp::int32* var_153;
        const wp::int32 var_154 = 0;
        bool var_155;
        wp::int32 var_156;
        const bool var_157 = false;
        bool var_158;
        wp::int32 var_159;
        wp::mat_t<3, 3, wp::float32>* var_160;
        wp::mat_t<3, 3, wp::float32> var_161;
        wp::mat_t<3, 3, wp::float32> var_162;
        wp::quat_t<wp::float32>* var_163;
        wp::quat_t<wp::float32> var_164;
        wp::quat_t<wp::float32> var_165;
        const wp::int32 var_166 = 0;
        const bool var_167 = true;
        wp::int32 var_168;
        const wp::int32 var_169 = 0;
        bool var_170;
        const bool var_171 = false;
        bool var_172;
        const wp::int32 var_173 = 4;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::int32 var_176;
        wp::float32* var_177;
        wp::float32 var_178;
        wp::float32 var_179;
        const wp::float32 var_180 = 0.0;
        bool var_181;
        const bool var_182 = false;
        bool var_183;
        wp::int32 var_184;
        bool var_185;
        bool var_186;
        wp::quat_t<wp::float32>* var_187;
        bool var_188;
        wp::quat_t<wp::float32> var_189;
        const wp::int32 var_190 = 0;
        wp::int32 var_191;
        wp::int32* var_192;
        wp::int32* var_193;
        wp::int32 var_194;
        wp::int32 var_195;
        wp::int32 var_196;
        const wp::int32 var_197 = 0;
        bool var_198;
        wp::int32 var_199;
        wp::float32 var_200;
        const wp::int32 var_201 = 1;
        wp::int32 var_202;
        wp::int32* var_203;
        wp::int32* var_204;
        wp::int32 var_205;
        wp::int32 var_206;
        wp::int32 var_207;
        const wp::int32 var_208 = 0;
        bool var_209;
        wp::int32 var_210;
        wp::float32 var_211;
        const wp::int32 var_212 = 2;
        wp::int32 var_213;
        wp::int32* var_214;
        wp::int32* var_215;
        wp::int32 var_216;
        wp::int32 var_217;
        wp::int32 var_218;
        const wp::int32 var_219 = 0;
        bool var_220;
        wp::int32 var_221;
        wp::float32 var_222;
        wp::mat_t<3, 3, wp::float32>* var_223;
        wp::mat_t<3, 3, wp::float32> var_224;
        wp::mat_t<3, 3, wp::float32> var_225;
        wp::mat_t<3, 3, wp::float32> var_226;
        const wp::int32 var_227 = 0;
        wp::int32 var_228;
        wp::int32* var_229;
        wp::int32* var_230;
        wp::int32 var_231;
        wp::int32 var_232;
        wp::int32 var_233;
        const wp::int32 var_234 = 0;
        bool var_235;
        const wp::int32 var_236 = 0;
        wp::int32 var_237;
        wp::float32 var_238;
        wp::float32 var_239;
        const wp::int32 var_240 = 1;
        wp::int32 var_241;
        wp::float32 var_242;
        wp::float32 var_243;
        const wp::int32 var_244 = 2;
        wp::int32 var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        const wp::int32 var_248 = 1;
        wp::int32 var_249;
        wp::int32* var_250;
        wp::int32* var_251;
        wp::int32 var_252;
        wp::int32 var_253;
        wp::int32 var_254;
        const wp::int32 var_255 = 0;
        bool var_256;
        const wp::int32 var_257 = 0;
        wp::int32 var_258;
        wp::float32 var_259;
        wp::float32 var_260;
        const wp::int32 var_261 = 1;
        wp::int32 var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        const wp::int32 var_265 = 2;
        wp::int32 var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        wp::int32 var_269;
        const wp::int32 var_270 = 2;
        wp::int32 var_271;
        wp::int32* var_272;
        wp::int32* var_273;
        wp::int32 var_274;
        wp::int32 var_275;
        wp::int32 var_276;
        const wp::int32 var_277 = 0;
        bool var_278;
        const wp::int32 var_279 = 0;
        wp::int32 var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        const wp::int32 var_283 = 1;
        wp::int32 var_284;
        wp::float32 var_285;
        wp::float32 var_286;
        const wp::int32 var_287 = 2;
        wp::int32 var_288;
        wp::float32 var_289;
        wp::float32 var_290;
        wp::int32 var_291;
        wp::int32 var_292;
        wp::int32 var_293;
        wp::int32 var_294;
        bool var_295;
        const wp::int32 var_296 = 1;
        const bool var_297 = true;
        wp::int32 var_298;
        const wp::int32 var_299 = 0;
        bool var_300;
        const bool var_301 = false;
        bool var_302;
        const wp::int32 var_303 = 4;
        wp::int32 var_304;
        wp::int32 var_305;
        wp::int32 var_306;
        wp::float32* var_307;
        wp::float32 var_308;
        wp::float32 var_309;
        const wp::float32 var_310 = 0.0;
        bool var_311;
        const bool var_312 = false;
        bool var_313;
        wp::int32 var_314;
        bool var_315;
        bool var_316;
        wp::quat_t<wp::float32>* var_317;
        bool var_318;
        wp::quat_t<wp::float32> var_319;
        const wp::int32 var_320 = 0;
        wp::int32 var_321;
        wp::int32* var_322;
        wp::int32* var_323;
        wp::int32 var_324;
        wp::int32 var_325;
        wp::int32 var_326;
        const wp::int32 var_327 = 0;
        bool var_328;
        wp::int32 var_329;
        wp::float32 var_330;
        const wp::int32 var_331 = 1;
        wp::int32 var_332;
        wp::int32* var_333;
        wp::int32* var_334;
        wp::int32 var_335;
        wp::int32 var_336;
        wp::int32 var_337;
        const wp::int32 var_338 = 0;
        bool var_339;
        wp::int32 var_340;
        wp::float32 var_341;
        const wp::int32 var_342 = 2;
        wp::int32 var_343;
        wp::int32* var_344;
        wp::int32* var_345;
        wp::int32 var_346;
        wp::int32 var_347;
        wp::int32 var_348;
        const wp::int32 var_349 = 0;
        bool var_350;
        wp::int32 var_351;
        wp::float32 var_352;
        wp::mat_t<3, 3, wp::float32>* var_353;
        wp::mat_t<3, 3, wp::float32> var_354;
        wp::mat_t<3, 3, wp::float32> var_355;
        wp::mat_t<3, 3, wp::float32> var_356;
        const wp::int32 var_357 = 0;
        wp::int32 var_358;
        wp::int32* var_359;
        wp::int32* var_360;
        wp::int32 var_361;
        wp::int32 var_362;
        wp::int32 var_363;
        const wp::int32 var_364 = 0;
        bool var_365;
        const wp::int32 var_366 = 0;
        wp::int32 var_367;
        wp::float32 var_368;
        wp::float32 var_369;
        const wp::int32 var_370 = 1;
        wp::int32 var_371;
        wp::float32 var_372;
        wp::float32 var_373;
        const wp::int32 var_374 = 2;
        wp::int32 var_375;
        wp::float32 var_376;
        wp::float32 var_377;
        wp::int32 var_378;
        const wp::int32 var_379 = 1;
        wp::int32 var_380;
        wp::int32* var_381;
        wp::int32* var_382;
        wp::int32 var_383;
        wp::int32 var_384;
        wp::int32 var_385;
        const wp::int32 var_386 = 0;
        bool var_387;
        const wp::int32 var_388 = 0;
        wp::int32 var_389;
        wp::float32 var_390;
        wp::float32 var_391;
        const wp::int32 var_392 = 1;
        wp::int32 var_393;
        wp::float32 var_394;
        wp::float32 var_395;
        const wp::int32 var_396 = 2;
        wp::int32 var_397;
        wp::float32 var_398;
        wp::float32 var_399;
        wp::int32 var_400;
        const wp::int32 var_401 = 2;
        wp::int32 var_402;
        wp::int32* var_403;
        wp::int32* var_404;
        wp::int32 var_405;
        wp::int32 var_406;
        wp::int32 var_407;
        const wp::int32 var_408 = 0;
        bool var_409;
        const wp::int32 var_410 = 0;
        wp::int32 var_411;
        wp::float32 var_412;
        wp::float32 var_413;
        const wp::int32 var_414 = 1;
        wp::int32 var_415;
        wp::float32 var_416;
        wp::float32 var_417;
        const wp::int32 var_418 = 2;
        wp::int32 var_419;
        wp::float32 var_420;
        wp::float32 var_421;
        wp::int32 var_422;
        wp::int32 var_423;
        wp::int32 var_424;
        wp::int32 var_425;
        wp::mat_t<3, 3, wp::float32> var_426;
        wp::int32 var_427;
        wp::int32 var_428;
        wp::int32 var_429;
        wp::int32 var_430;
        wp::int32 var_431;
        wp::mat_t<3, 3, wp::float32> var_432;
        wp::int32 var_433;
        bool var_434;
        wp::float32 var_435;
        wp::int32 var_436;
        wp::int32 var_437;
        wp::int32 var_438;
        wp::int32 var_439;
        wp::mat_t<3, 3, wp::float32> var_440;
        wp::int32 var_441;
        const wp::int32 var_442 = 2;
        const bool var_443 = true;
        wp::int32 var_444;
        const wp::int32 var_445 = 0;
        bool var_446;
        const bool var_447 = false;
        bool var_448;
        const wp::int32 var_449 = 4;
        wp::int32 var_450;
        wp::int32 var_451;
        wp::int32 var_452;
        wp::float32* var_453;
        wp::float32 var_454;
        wp::float32 var_455;
        const wp::float32 var_456 = 0.0;
        bool var_457;
        const bool var_458 = false;
        bool var_459;
        wp::int32 var_460;
        bool var_461;
        bool var_462;
        wp::quat_t<wp::float32>* var_463;
        bool var_464;
        wp::quat_t<wp::float32> var_465;
        const wp::int32 var_466 = 0;
        wp::int32 var_467;
        wp::int32* var_468;
        wp::int32* var_469;
        wp::int32 var_470;
        wp::int32 var_471;
        wp::int32 var_472;
        const wp::int32 var_473 = 0;
        bool var_474;
        wp::int32 var_475;
        wp::float32 var_476;
        const wp::int32 var_477 = 1;
        wp::int32 var_478;
        wp::int32* var_479;
        wp::int32* var_480;
        wp::int32 var_481;
        wp::int32 var_482;
        wp::int32 var_483;
        const wp::int32 var_484 = 0;
        bool var_485;
        wp::int32 var_486;
        wp::float32 var_487;
        const wp::int32 var_488 = 2;
        wp::int32 var_489;
        wp::int32* var_490;
        wp::int32* var_491;
        wp::int32 var_492;
        wp::int32 var_493;
        wp::int32 var_494;
        const wp::int32 var_495 = 0;
        bool var_496;
        wp::int32 var_497;
        wp::float32 var_498;
        wp::mat_t<3, 3, wp::float32>* var_499;
        wp::mat_t<3, 3, wp::float32> var_500;
        wp::mat_t<3, 3, wp::float32> var_501;
        wp::mat_t<3, 3, wp::float32> var_502;
        const wp::int32 var_503 = 0;
        wp::int32 var_504;
        wp::int32* var_505;
        wp::int32* var_506;
        wp::int32 var_507;
        wp::int32 var_508;
        wp::int32 var_509;
        const wp::int32 var_510 = 0;
        bool var_511;
        const wp::int32 var_512 = 0;
        wp::int32 var_513;
        wp::float32 var_514;
        wp::float32 var_515;
        const wp::int32 var_516 = 1;
        wp::int32 var_517;
        wp::float32 var_518;
        wp::float32 var_519;
        const wp::int32 var_520 = 2;
        wp::int32 var_521;
        wp::float32 var_522;
        wp::float32 var_523;
        wp::int32 var_524;
        const wp::int32 var_525 = 1;
        wp::int32 var_526;
        wp::int32* var_527;
        wp::int32* var_528;
        wp::int32 var_529;
        wp::int32 var_530;
        wp::int32 var_531;
        const wp::int32 var_532 = 0;
        bool var_533;
        const wp::int32 var_534 = 0;
        wp::int32 var_535;
        wp::float32 var_536;
        wp::float32 var_537;
        const wp::int32 var_538 = 1;
        wp::int32 var_539;
        wp::float32 var_540;
        wp::float32 var_541;
        const wp::int32 var_542 = 2;
        wp::int32 var_543;
        wp::float32 var_544;
        wp::float32 var_545;
        wp::int32 var_546;
        const wp::int32 var_547 = 2;
        wp::int32 var_548;
        wp::int32* var_549;
        wp::int32* var_550;
        wp::int32 var_551;
        wp::int32 var_552;
        wp::int32 var_553;
        const wp::int32 var_554 = 0;
        bool var_555;
        const wp::int32 var_556 = 0;
        wp::int32 var_557;
        wp::float32 var_558;
        wp::float32 var_559;
        const wp::int32 var_560 = 1;
        wp::int32 var_561;
        wp::float32 var_562;
        wp::float32 var_563;
        const wp::int32 var_564 = 2;
        wp::int32 var_565;
        wp::float32 var_566;
        wp::float32 var_567;
        wp::int32 var_568;
        wp::int32 var_569;
        wp::int32 var_570;
        wp::int32 var_571;
        wp::mat_t<3, 3, wp::float32> var_572;
        wp::int32 var_573;
        wp::int32 var_574;
        wp::int32 var_575;
        wp::int32 var_576;
        wp::int32 var_577;
        wp::mat_t<3, 3, wp::float32> var_578;
        wp::int32 var_579;
        bool var_580;
        wp::float32 var_581;
        wp::int32 var_582;
        wp::int32 var_583;
        wp::int32 var_584;
        wp::int32 var_585;
        wp::mat_t<3, 3, wp::float32> var_586;
        wp::int32 var_587;
        const wp::int32 var_588 = 3;
        const bool var_589 = true;
        wp::int32 var_590;
        const wp::int32 var_591 = 0;
        bool var_592;
        const bool var_593 = false;
        bool var_594;
        const wp::int32 var_595 = 4;
        wp::int32 var_596;
        wp::int32 var_597;
        wp::int32 var_598;
        wp::float32* var_599;
        wp::float32 var_600;
        wp::float32 var_601;
        const wp::float32 var_602 = 0.0;
        bool var_603;
        const bool var_604 = false;
        bool var_605;
        wp::int32 var_606;
        bool var_607;
        bool var_608;
        wp::quat_t<wp::float32>* var_609;
        bool var_610;
        wp::quat_t<wp::float32> var_611;
        const wp::int32 var_612 = 0;
        wp::int32 var_613;
        wp::int32* var_614;
        wp::int32* var_615;
        wp::int32 var_616;
        wp::int32 var_617;
        wp::int32 var_618;
        const wp::int32 var_619 = 0;
        bool var_620;
        wp::int32 var_621;
        wp::float32 var_622;
        const wp::int32 var_623 = 1;
        wp::int32 var_624;
        wp::int32* var_625;
        wp::int32* var_626;
        wp::int32 var_627;
        wp::int32 var_628;
        wp::int32 var_629;
        const wp::int32 var_630 = 0;
        bool var_631;
        wp::int32 var_632;
        wp::float32 var_633;
        const wp::int32 var_634 = 2;
        wp::int32 var_635;
        wp::int32* var_636;
        wp::int32* var_637;
        wp::int32 var_638;
        wp::int32 var_639;
        wp::int32 var_640;
        const wp::int32 var_641 = 0;
        bool var_642;
        wp::int32 var_643;
        wp::float32 var_644;
        wp::mat_t<3, 3, wp::float32>* var_645;
        wp::mat_t<3, 3, wp::float32> var_646;
        wp::mat_t<3, 3, wp::float32> var_647;
        wp::mat_t<3, 3, wp::float32> var_648;
        const wp::int32 var_649 = 0;
        wp::int32 var_650;
        wp::int32* var_651;
        wp::int32* var_652;
        wp::int32 var_653;
        wp::int32 var_654;
        wp::int32 var_655;
        const wp::int32 var_656 = 0;
        bool var_657;
        const wp::int32 var_658 = 0;
        wp::int32 var_659;
        wp::float32 var_660;
        wp::float32 var_661;
        const wp::int32 var_662 = 1;
        wp::int32 var_663;
        wp::float32 var_664;
        wp::float32 var_665;
        const wp::int32 var_666 = 2;
        wp::int32 var_667;
        wp::float32 var_668;
        wp::float32 var_669;
        wp::int32 var_670;
        const wp::int32 var_671 = 1;
        wp::int32 var_672;
        wp::int32* var_673;
        wp::int32* var_674;
        wp::int32 var_675;
        wp::int32 var_676;
        wp::int32 var_677;
        const wp::int32 var_678 = 0;
        bool var_679;
        const wp::int32 var_680 = 0;
        wp::int32 var_681;
        wp::float32 var_682;
        wp::float32 var_683;
        const wp::int32 var_684 = 1;
        wp::int32 var_685;
        wp::float32 var_686;
        wp::float32 var_687;
        const wp::int32 var_688 = 2;
        wp::int32 var_689;
        wp::float32 var_690;
        wp::float32 var_691;
        wp::int32 var_692;
        const wp::int32 var_693 = 2;
        wp::int32 var_694;
        wp::int32* var_695;
        wp::int32* var_696;
        wp::int32 var_697;
        wp::int32 var_698;
        wp::int32 var_699;
        const wp::int32 var_700 = 0;
        bool var_701;
        const wp::int32 var_702 = 0;
        wp::int32 var_703;
        wp::float32 var_704;
        wp::float32 var_705;
        const wp::int32 var_706 = 1;
        wp::int32 var_707;
        wp::float32 var_708;
        wp::float32 var_709;
        const wp::int32 var_710 = 2;
        wp::int32 var_711;
        wp::float32 var_712;
        wp::float32 var_713;
        wp::int32 var_714;
        wp::int32 var_715;
        wp::int32 var_716;
        wp::int32 var_717;
        wp::mat_t<3, 3, wp::float32> var_718;
        wp::int32 var_719;
        wp::int32 var_720;
        wp::int32 var_721;
        wp::int32 var_722;
        wp::int32 var_723;
        wp::mat_t<3, 3, wp::float32> var_724;
        wp::int32 var_725;
        bool var_726;
        wp::float32 var_727;
        wp::int32 var_728;
        wp::int32 var_729;
        wp::int32 var_730;
        wp::int32 var_731;
        wp::mat_t<3, 3, wp::float32> var_732;
        wp::int32 var_733;
        const wp::int32 var_734 = 1;
        const bool var_735 = true;
        wp::int32 var_736;
        bool var_737;
        const wp::int32 var_738 = 0;
        bool var_739;
        wp::int32* var_740;
        const wp::int32 var_741 = 0;
        bool var_742;
        wp::int32 var_743;
        const bool var_744 = false;
        bool var_745;
        wp::int32 var_746;
        wp::mat_t<3, 3, wp::float32>* var_747;
        wp::mat_t<3, 3, wp::float32> var_748;
        wp::mat_t<3, 3, wp::float32> var_749;
        wp::quat_t<wp::float32>* var_750;
        wp::quat_t<wp::float32> var_751;
        wp::quat_t<wp::float32> var_752;
        const wp::int32 var_753 = 0;
        const bool var_754 = true;
        wp::int32 var_755;
        const wp::int32 var_756 = 0;
        bool var_757;
        const bool var_758 = false;
        bool var_759;
        const wp::int32 var_760 = 4;
        wp::int32 var_761;
        wp::int32 var_762;
        wp::int32 var_763;
        wp::float32* var_764;
        wp::float32 var_765;
        wp::float32 var_766;
        const wp::float32 var_767 = 0.0;
        bool var_768;
        const bool var_769 = false;
        bool var_770;
        wp::int32 var_771;
        bool var_772;
        bool var_773;
        wp::quat_t<wp::float32>* var_774;
        bool var_775;
        wp::quat_t<wp::float32> var_776;
        const wp::int32 var_777 = 0;
        wp::int32 var_778;
        wp::int32* var_779;
        wp::int32* var_780;
        wp::int32 var_781;
        wp::int32 var_782;
        wp::int32 var_783;
        const wp::int32 var_784 = 0;
        bool var_785;
        wp::int32 var_786;
        wp::float32 var_787;
        const wp::int32 var_788 = 1;
        wp::int32 var_789;
        wp::int32* var_790;
        wp::int32* var_791;
        wp::int32 var_792;
        wp::int32 var_793;
        wp::int32 var_794;
        const wp::int32 var_795 = 0;
        bool var_796;
        wp::int32 var_797;
        wp::float32 var_798;
        const wp::int32 var_799 = 2;
        wp::int32 var_800;
        wp::int32* var_801;
        wp::int32* var_802;
        wp::int32 var_803;
        wp::int32 var_804;
        wp::int32 var_805;
        const wp::int32 var_806 = 0;
        bool var_807;
        wp::int32 var_808;
        wp::float32 var_809;
        wp::mat_t<3, 3, wp::float32>* var_810;
        wp::mat_t<3, 3, wp::float32> var_811;
        wp::mat_t<3, 3, wp::float32> var_812;
        wp::mat_t<3, 3, wp::float32> var_813;
        const wp::int32 var_814 = 0;
        wp::int32 var_815;
        wp::int32* var_816;
        wp::int32* var_817;
        wp::int32 var_818;
        wp::int32 var_819;
        wp::int32 var_820;
        const wp::int32 var_821 = 0;
        bool var_822;
        const wp::int32 var_823 = 0;
        wp::int32 var_824;
        wp::float32 var_825;
        wp::float32 var_826;
        const wp::int32 var_827 = 1;
        wp::int32 var_828;
        wp::float32 var_829;
        wp::float32 var_830;
        const wp::int32 var_831 = 2;
        wp::int32 var_832;
        wp::float32 var_833;
        wp::float32 var_834;
        wp::int32 var_835;
        const wp::int32 var_836 = 1;
        wp::int32 var_837;
        wp::int32* var_838;
        wp::int32* var_839;
        wp::int32 var_840;
        wp::int32 var_841;
        wp::int32 var_842;
        const wp::int32 var_843 = 0;
        bool var_844;
        const wp::int32 var_845 = 0;
        wp::int32 var_846;
        wp::float32 var_847;
        wp::float32 var_848;
        const wp::int32 var_849 = 1;
        wp::int32 var_850;
        wp::float32 var_851;
        wp::float32 var_852;
        const wp::int32 var_853 = 2;
        wp::int32 var_854;
        wp::float32 var_855;
        wp::float32 var_856;
        wp::int32 var_857;
        const wp::int32 var_858 = 2;
        wp::int32 var_859;
        wp::int32* var_860;
        wp::int32* var_861;
        wp::int32 var_862;
        wp::int32 var_863;
        wp::int32 var_864;
        const wp::int32 var_865 = 0;
        bool var_866;
        const wp::int32 var_867 = 0;
        wp::int32 var_868;
        wp::float32 var_869;
        wp::float32 var_870;
        const wp::int32 var_871 = 1;
        wp::int32 var_872;
        wp::float32 var_873;
        wp::float32 var_874;
        const wp::int32 var_875 = 2;
        wp::int32 var_876;
        wp::float32 var_877;
        wp::float32 var_878;
        wp::int32 var_879;
        wp::int32 var_880;
        wp::int32 var_881;
        wp::int32 var_882;
        wp::mat_t<3, 3, wp::float32> var_883;
        wp::int32 var_884;
        wp::int32 var_885;
        wp::int32 var_886;
        wp::int32 var_887;
        wp::int32 var_888;
        wp::mat_t<3, 3, wp::float32> var_889;
        wp::int32 var_890;
        wp::float32 var_891;
        wp::int32 var_892;
        wp::int32 var_893;
        wp::int32 var_894;
        wp::int32 var_895;
        wp::mat_t<3, 3, wp::float32> var_896;
        wp::int32 var_897;
        bool var_898;
        const wp::int32 var_899 = 1;
        const bool var_900 = true;
        wp::int32 var_901;
        const wp::int32 var_902 = 0;
        bool var_903;
        const bool var_904 = false;
        bool var_905;
        const wp::int32 var_906 = 4;
        wp::int32 var_907;
        wp::int32 var_908;
        wp::int32 var_909;
        wp::float32* var_910;
        wp::float32 var_911;
        wp::float32 var_912;
        const wp::float32 var_913 = 0.0;
        bool var_914;
        const bool var_915 = false;
        bool var_916;
        wp::int32 var_917;
        bool var_918;
        bool var_919;
        wp::quat_t<wp::float32>* var_920;
        bool var_921;
        wp::quat_t<wp::float32> var_922;
        const wp::int32 var_923 = 0;
        wp::int32 var_924;
        wp::int32* var_925;
        wp::int32* var_926;
        wp::int32 var_927;
        wp::int32 var_928;
        wp::int32 var_929;
        const wp::int32 var_930 = 0;
        bool var_931;
        wp::int32 var_932;
        wp::float32 var_933;
        const wp::int32 var_934 = 1;
        wp::int32 var_935;
        wp::int32* var_936;
        wp::int32* var_937;
        wp::int32 var_938;
        wp::int32 var_939;
        wp::int32 var_940;
        const wp::int32 var_941 = 0;
        bool var_942;
        wp::int32 var_943;
        wp::float32 var_944;
        const wp::int32 var_945 = 2;
        wp::int32 var_946;
        wp::int32* var_947;
        wp::int32* var_948;
        wp::int32 var_949;
        wp::int32 var_950;
        wp::int32 var_951;
        const wp::int32 var_952 = 0;
        bool var_953;
        wp::int32 var_954;
        wp::float32 var_955;
        wp::mat_t<3, 3, wp::float32>* var_956;
        wp::mat_t<3, 3, wp::float32> var_957;
        wp::mat_t<3, 3, wp::float32> var_958;
        wp::mat_t<3, 3, wp::float32> var_959;
        const wp::int32 var_960 = 0;
        wp::int32 var_961;
        wp::int32* var_962;
        wp::int32* var_963;
        wp::int32 var_964;
        wp::int32 var_965;
        wp::int32 var_966;
        const wp::int32 var_967 = 0;
        bool var_968;
        const wp::int32 var_969 = 0;
        wp::int32 var_970;
        wp::float32 var_971;
        wp::float32 var_972;
        const wp::int32 var_973 = 1;
        wp::int32 var_974;
        wp::float32 var_975;
        wp::float32 var_976;
        const wp::int32 var_977 = 2;
        wp::int32 var_978;
        wp::float32 var_979;
        wp::float32 var_980;
        wp::int32 var_981;
        const wp::int32 var_982 = 1;
        wp::int32 var_983;
        wp::int32* var_984;
        wp::int32* var_985;
        wp::int32 var_986;
        wp::int32 var_987;
        wp::int32 var_988;
        const wp::int32 var_989 = 0;
        bool var_990;
        const wp::int32 var_991 = 0;
        wp::int32 var_992;
        wp::float32 var_993;
        wp::float32 var_994;
        const wp::int32 var_995 = 1;
        wp::int32 var_996;
        wp::float32 var_997;
        wp::float32 var_998;
        const wp::int32 var_999 = 2;
        wp::int32 var_1000;
        wp::float32 var_1001;
        wp::float32 var_1002;
        wp::int32 var_1003;
        const wp::int32 var_1004 = 2;
        wp::int32 var_1005;
        wp::int32* var_1006;
        wp::int32* var_1007;
        wp::int32 var_1008;
        wp::int32 var_1009;
        wp::int32 var_1010;
        const wp::int32 var_1011 = 0;
        bool var_1012;
        const wp::int32 var_1013 = 0;
        wp::int32 var_1014;
        wp::float32 var_1015;
        wp::float32 var_1016;
        const wp::int32 var_1017 = 1;
        wp::int32 var_1018;
        wp::float32 var_1019;
        wp::float32 var_1020;
        const wp::int32 var_1021 = 2;
        wp::int32 var_1022;
        wp::float32 var_1023;
        wp::float32 var_1024;
        wp::int32 var_1025;
        wp::int32 var_1026;
        wp::int32 var_1027;
        wp::int32 var_1028;
        wp::mat_t<3, 3, wp::float32> var_1029;
        wp::int32 var_1030;
        wp::int32 var_1031;
        wp::int32 var_1032;
        wp::int32 var_1033;
        wp::int32 var_1034;
        wp::mat_t<3, 3, wp::float32> var_1035;
        wp::int32 var_1036;
        wp::float32 var_1037;
        wp::int32 var_1038;
        wp::int32 var_1039;
        wp::int32 var_1040;
        wp::int32 var_1041;
        wp::mat_t<3, 3, wp::float32> var_1042;
        wp::int32 var_1043;
        bool var_1044;
        const wp::int32 var_1045 = 2;
        const bool var_1046 = true;
        wp::int32 var_1047;
        const wp::int32 var_1048 = 0;
        bool var_1049;
        const bool var_1050 = false;
        bool var_1051;
        const wp::int32 var_1052 = 4;
        wp::int32 var_1053;
        wp::int32 var_1054;
        wp::int32 var_1055;
        wp::float32* var_1056;
        wp::float32 var_1057;
        wp::float32 var_1058;
        const wp::float32 var_1059 = 0.0;
        bool var_1060;
        const bool var_1061 = false;
        bool var_1062;
        wp::int32 var_1063;
        bool var_1064;
        bool var_1065;
        wp::quat_t<wp::float32>* var_1066;
        bool var_1067;
        wp::quat_t<wp::float32> var_1068;
        const wp::int32 var_1069 = 0;
        wp::int32 var_1070;
        wp::int32* var_1071;
        wp::int32* var_1072;
        wp::int32 var_1073;
        wp::int32 var_1074;
        wp::int32 var_1075;
        const wp::int32 var_1076 = 0;
        bool var_1077;
        wp::int32 var_1078;
        wp::float32 var_1079;
        const wp::int32 var_1080 = 1;
        wp::int32 var_1081;
        wp::int32* var_1082;
        wp::int32* var_1083;
        wp::int32 var_1084;
        wp::int32 var_1085;
        wp::int32 var_1086;
        const wp::int32 var_1087 = 0;
        bool var_1088;
        wp::int32 var_1089;
        wp::float32 var_1090;
        const wp::int32 var_1091 = 2;
        wp::int32 var_1092;
        wp::int32* var_1093;
        wp::int32* var_1094;
        wp::int32 var_1095;
        wp::int32 var_1096;
        wp::int32 var_1097;
        const wp::int32 var_1098 = 0;
        bool var_1099;
        wp::int32 var_1100;
        wp::float32 var_1101;
        wp::mat_t<3, 3, wp::float32>* var_1102;
        wp::mat_t<3, 3, wp::float32> var_1103;
        wp::mat_t<3, 3, wp::float32> var_1104;
        wp::mat_t<3, 3, wp::float32> var_1105;
        const wp::int32 var_1106 = 0;
        wp::int32 var_1107;
        wp::int32* var_1108;
        wp::int32* var_1109;
        wp::int32 var_1110;
        wp::int32 var_1111;
        wp::int32 var_1112;
        const wp::int32 var_1113 = 0;
        bool var_1114;
        const wp::int32 var_1115 = 0;
        wp::int32 var_1116;
        wp::float32 var_1117;
        wp::float32 var_1118;
        const wp::int32 var_1119 = 1;
        wp::int32 var_1120;
        wp::float32 var_1121;
        wp::float32 var_1122;
        const wp::int32 var_1123 = 2;
        wp::int32 var_1124;
        wp::float32 var_1125;
        wp::float32 var_1126;
        wp::int32 var_1127;
        const wp::int32 var_1128 = 1;
        wp::int32 var_1129;
        wp::int32* var_1130;
        wp::int32* var_1131;
        wp::int32 var_1132;
        wp::int32 var_1133;
        wp::int32 var_1134;
        const wp::int32 var_1135 = 0;
        bool var_1136;
        const wp::int32 var_1137 = 0;
        wp::int32 var_1138;
        wp::float32 var_1139;
        wp::float32 var_1140;
        const wp::int32 var_1141 = 1;
        wp::int32 var_1142;
        wp::float32 var_1143;
        wp::float32 var_1144;
        const wp::int32 var_1145 = 2;
        wp::int32 var_1146;
        wp::float32 var_1147;
        wp::float32 var_1148;
        wp::int32 var_1149;
        const wp::int32 var_1150 = 2;
        wp::int32 var_1151;
        wp::int32* var_1152;
        wp::int32* var_1153;
        wp::int32 var_1154;
        wp::int32 var_1155;
        wp::int32 var_1156;
        const wp::int32 var_1157 = 0;
        bool var_1158;
        const wp::int32 var_1159 = 0;
        wp::int32 var_1160;
        wp::float32 var_1161;
        wp::float32 var_1162;
        const wp::int32 var_1163 = 1;
        wp::int32 var_1164;
        wp::float32 var_1165;
        wp::float32 var_1166;
        const wp::int32 var_1167 = 2;
        wp::int32 var_1168;
        wp::float32 var_1169;
        wp::float32 var_1170;
        wp::int32 var_1171;
        wp::int32 var_1172;
        wp::int32 var_1173;
        wp::int32 var_1174;
        wp::mat_t<3, 3, wp::float32> var_1175;
        wp::int32 var_1176;
        wp::int32 var_1177;
        wp::int32 var_1178;
        wp::int32 var_1179;
        wp::int32 var_1180;
        wp::mat_t<3, 3, wp::float32> var_1181;
        wp::int32 var_1182;
        wp::float32 var_1183;
        wp::int32 var_1184;
        wp::int32 var_1185;
        wp::int32 var_1186;
        wp::int32 var_1187;
        wp::mat_t<3, 3, wp::float32> var_1188;
        wp::int32 var_1189;
        bool var_1190;
        const wp::int32 var_1191 = 3;
        const bool var_1192 = true;
        wp::int32 var_1193;
        const wp::int32 var_1194 = 0;
        bool var_1195;
        const bool var_1196 = false;
        bool var_1197;
        const wp::int32 var_1198 = 4;
        wp::int32 var_1199;
        wp::int32 var_1200;
        wp::int32 var_1201;
        wp::float32* var_1202;
        wp::float32 var_1203;
        wp::float32 var_1204;
        const wp::float32 var_1205 = 0.0;
        bool var_1206;
        const bool var_1207 = false;
        bool var_1208;
        wp::int32 var_1209;
        bool var_1210;
        bool var_1211;
        wp::quat_t<wp::float32>* var_1212;
        bool var_1213;
        wp::quat_t<wp::float32> var_1214;
        const wp::int32 var_1215 = 0;
        wp::int32 var_1216;
        wp::int32* var_1217;
        wp::int32* var_1218;
        wp::int32 var_1219;
        wp::int32 var_1220;
        wp::int32 var_1221;
        const wp::int32 var_1222 = 0;
        bool var_1223;
        wp::int32 var_1224;
        wp::float32 var_1225;
        const wp::int32 var_1226 = 1;
        wp::int32 var_1227;
        wp::int32* var_1228;
        wp::int32* var_1229;
        wp::int32 var_1230;
        wp::int32 var_1231;
        wp::int32 var_1232;
        const wp::int32 var_1233 = 0;
        bool var_1234;
        wp::int32 var_1235;
        wp::float32 var_1236;
        const wp::int32 var_1237 = 2;
        wp::int32 var_1238;
        wp::int32* var_1239;
        wp::int32* var_1240;
        wp::int32 var_1241;
        wp::int32 var_1242;
        wp::int32 var_1243;
        const wp::int32 var_1244 = 0;
        bool var_1245;
        wp::int32 var_1246;
        wp::float32 var_1247;
        wp::mat_t<3, 3, wp::float32>* var_1248;
        wp::mat_t<3, 3, wp::float32> var_1249;
        wp::mat_t<3, 3, wp::float32> var_1250;
        wp::mat_t<3, 3, wp::float32> var_1251;
        const wp::int32 var_1252 = 0;
        wp::int32 var_1253;
        wp::int32* var_1254;
        wp::int32* var_1255;
        wp::int32 var_1256;
        wp::int32 var_1257;
        wp::int32 var_1258;
        const wp::int32 var_1259 = 0;
        bool var_1260;
        const wp::int32 var_1261 = 0;
        wp::int32 var_1262;
        wp::float32 var_1263;
        wp::float32 var_1264;
        const wp::int32 var_1265 = 1;
        wp::int32 var_1266;
        wp::float32 var_1267;
        wp::float32 var_1268;
        const wp::int32 var_1269 = 2;
        wp::int32 var_1270;
        wp::float32 var_1271;
        wp::float32 var_1272;
        wp::int32 var_1273;
        const wp::int32 var_1274 = 1;
        wp::int32 var_1275;
        wp::int32* var_1276;
        wp::int32* var_1277;
        wp::int32 var_1278;
        wp::int32 var_1279;
        wp::int32 var_1280;
        const wp::int32 var_1281 = 0;
        bool var_1282;
        const wp::int32 var_1283 = 0;
        wp::int32 var_1284;
        wp::float32 var_1285;
        wp::float32 var_1286;
        const wp::int32 var_1287 = 1;
        wp::int32 var_1288;
        wp::float32 var_1289;
        wp::float32 var_1290;
        const wp::int32 var_1291 = 2;
        wp::int32 var_1292;
        wp::float32 var_1293;
        wp::float32 var_1294;
        wp::int32 var_1295;
        const wp::int32 var_1296 = 2;
        wp::int32 var_1297;
        wp::int32* var_1298;
        wp::int32* var_1299;
        wp::int32 var_1300;
        wp::int32 var_1301;
        wp::int32 var_1302;
        const wp::int32 var_1303 = 0;
        bool var_1304;
        const wp::int32 var_1305 = 0;
        wp::int32 var_1306;
        wp::float32 var_1307;
        wp::float32 var_1308;
        const wp::int32 var_1309 = 1;
        wp::int32 var_1310;
        wp::float32 var_1311;
        wp::float32 var_1312;
        const wp::int32 var_1313 = 2;
        wp::int32 var_1314;
        wp::float32 var_1315;
        wp::float32 var_1316;
        wp::int32 var_1317;
        wp::int32 var_1318;
        wp::int32 var_1319;
        wp::int32 var_1320;
        wp::mat_t<3, 3, wp::float32> var_1321;
        wp::int32 var_1322;
        wp::int32 var_1323;
        wp::int32 var_1324;
        wp::int32 var_1325;
        wp::int32 var_1326;
        wp::mat_t<3, 3, wp::float32> var_1327;
        wp::int32 var_1328;
        wp::float32 var_1329;
        wp::int32 var_1330;
        wp::int32 var_1331;
        wp::int32 var_1332;
        wp::int32 var_1333;
        wp::mat_t<3, 3, wp::float32> var_1334;
        wp::int32 var_1335;
        bool var_1336;
        wp::int32 var_1337;
        wp::mat_t<3, 3, wp::float32> var_1338;
        wp::quat_t<wp::float32> var_1339;
        wp::int32 var_1340;
        wp::int32 var_1341;
        wp::float32 var_1342;
        wp::int32 var_1343;
        wp::int32 var_1344;
        wp::int32 var_1345;
        wp::int32 var_1346;
        wp::mat_t<3, 3, wp::float32> var_1347;
        wp::int32 var_1348;
        const wp::int32 var_1349 = 2;
        const bool var_1350 = true;
        wp::int32 var_1351;
        bool var_1352;
        const wp::int32 var_1353 = 0;
        bool var_1354;
        wp::int32* var_1355;
        const wp::int32 var_1356 = 0;
        bool var_1357;
        wp::int32 var_1358;
        const bool var_1359 = false;
        bool var_1360;
        wp::int32 var_1361;
        wp::mat_t<3, 3, wp::float32>* var_1362;
        wp::mat_t<3, 3, wp::float32> var_1363;
        wp::mat_t<3, 3, wp::float32> var_1364;
        wp::quat_t<wp::float32>* var_1365;
        wp::quat_t<wp::float32> var_1366;
        wp::quat_t<wp::float32> var_1367;
        const wp::int32 var_1368 = 0;
        const bool var_1369 = true;
        wp::int32 var_1370;
        const wp::int32 var_1371 = 0;
        bool var_1372;
        const bool var_1373 = false;
        bool var_1374;
        const wp::int32 var_1375 = 4;
        wp::int32 var_1376;
        wp::int32 var_1377;
        wp::int32 var_1378;
        wp::float32* var_1379;
        wp::float32 var_1380;
        wp::float32 var_1381;
        const wp::float32 var_1382 = 0.0;
        bool var_1383;
        const bool var_1384 = false;
        bool var_1385;
        wp::int32 var_1386;
        bool var_1387;
        bool var_1388;
        wp::quat_t<wp::float32>* var_1389;
        bool var_1390;
        wp::quat_t<wp::float32> var_1391;
        const wp::int32 var_1392 = 0;
        wp::int32 var_1393;
        wp::int32* var_1394;
        wp::int32* var_1395;
        wp::int32 var_1396;
        wp::int32 var_1397;
        wp::int32 var_1398;
        const wp::int32 var_1399 = 0;
        bool var_1400;
        wp::int32 var_1401;
        wp::float32 var_1402;
        const wp::int32 var_1403 = 1;
        wp::int32 var_1404;
        wp::int32* var_1405;
        wp::int32* var_1406;
        wp::int32 var_1407;
        wp::int32 var_1408;
        wp::int32 var_1409;
        const wp::int32 var_1410 = 0;
        bool var_1411;
        wp::int32 var_1412;
        wp::float32 var_1413;
        const wp::int32 var_1414 = 2;
        wp::int32 var_1415;
        wp::int32* var_1416;
        wp::int32* var_1417;
        wp::int32 var_1418;
        wp::int32 var_1419;
        wp::int32 var_1420;
        const wp::int32 var_1421 = 0;
        bool var_1422;
        wp::int32 var_1423;
        wp::float32 var_1424;
        wp::mat_t<3, 3, wp::float32>* var_1425;
        wp::mat_t<3, 3, wp::float32> var_1426;
        wp::mat_t<3, 3, wp::float32> var_1427;
        wp::mat_t<3, 3, wp::float32> var_1428;
        const wp::int32 var_1429 = 0;
        wp::int32 var_1430;
        wp::int32* var_1431;
        wp::int32* var_1432;
        wp::int32 var_1433;
        wp::int32 var_1434;
        wp::int32 var_1435;
        const wp::int32 var_1436 = 0;
        bool var_1437;
        const wp::int32 var_1438 = 0;
        wp::int32 var_1439;
        wp::float32 var_1440;
        wp::float32 var_1441;
        const wp::int32 var_1442 = 1;
        wp::int32 var_1443;
        wp::float32 var_1444;
        wp::float32 var_1445;
        const wp::int32 var_1446 = 2;
        wp::int32 var_1447;
        wp::float32 var_1448;
        wp::float32 var_1449;
        wp::int32 var_1450;
        const wp::int32 var_1451 = 1;
        wp::int32 var_1452;
        wp::int32* var_1453;
        wp::int32* var_1454;
        wp::int32 var_1455;
        wp::int32 var_1456;
        wp::int32 var_1457;
        const wp::int32 var_1458 = 0;
        bool var_1459;
        const wp::int32 var_1460 = 0;
        wp::int32 var_1461;
        wp::float32 var_1462;
        wp::float32 var_1463;
        const wp::int32 var_1464 = 1;
        wp::int32 var_1465;
        wp::float32 var_1466;
        wp::float32 var_1467;
        const wp::int32 var_1468 = 2;
        wp::int32 var_1469;
        wp::float32 var_1470;
        wp::float32 var_1471;
        wp::int32 var_1472;
        const wp::int32 var_1473 = 2;
        wp::int32 var_1474;
        wp::int32* var_1475;
        wp::int32* var_1476;
        wp::int32 var_1477;
        wp::int32 var_1478;
        wp::int32 var_1479;
        const wp::int32 var_1480 = 0;
        bool var_1481;
        const wp::int32 var_1482 = 0;
        wp::int32 var_1483;
        wp::float32 var_1484;
        wp::float32 var_1485;
        const wp::int32 var_1486 = 1;
        wp::int32 var_1487;
        wp::float32 var_1488;
        wp::float32 var_1489;
        const wp::int32 var_1490 = 2;
        wp::int32 var_1491;
        wp::float32 var_1492;
        wp::float32 var_1493;
        wp::int32 var_1494;
        wp::int32 var_1495;
        wp::int32 var_1496;
        wp::int32 var_1497;
        wp::mat_t<3, 3, wp::float32> var_1498;
        wp::int32 var_1499;
        wp::int32 var_1500;
        wp::int32 var_1501;
        wp::int32 var_1502;
        wp::int32 var_1503;
        wp::mat_t<3, 3, wp::float32> var_1504;
        wp::int32 var_1505;
        wp::float32 var_1506;
        wp::int32 var_1507;
        wp::int32 var_1508;
        wp::int32 var_1509;
        wp::int32 var_1510;
        wp::mat_t<3, 3, wp::float32> var_1511;
        wp::int32 var_1512;
        bool var_1513;
        const wp::int32 var_1514 = 1;
        const bool var_1515 = true;
        wp::int32 var_1516;
        const wp::int32 var_1517 = 0;
        bool var_1518;
        const bool var_1519 = false;
        bool var_1520;
        const wp::int32 var_1521 = 4;
        wp::int32 var_1522;
        wp::int32 var_1523;
        wp::int32 var_1524;
        wp::float32* var_1525;
        wp::float32 var_1526;
        wp::float32 var_1527;
        const wp::float32 var_1528 = 0.0;
        bool var_1529;
        const bool var_1530 = false;
        bool var_1531;
        wp::int32 var_1532;
        bool var_1533;
        bool var_1534;
        wp::quat_t<wp::float32>* var_1535;
        bool var_1536;
        wp::quat_t<wp::float32> var_1537;
        const wp::int32 var_1538 = 0;
        wp::int32 var_1539;
        wp::int32* var_1540;
        wp::int32* var_1541;
        wp::int32 var_1542;
        wp::int32 var_1543;
        wp::int32 var_1544;
        const wp::int32 var_1545 = 0;
        bool var_1546;
        wp::int32 var_1547;
        wp::float32 var_1548;
        const wp::int32 var_1549 = 1;
        wp::int32 var_1550;
        wp::int32* var_1551;
        wp::int32* var_1552;
        wp::int32 var_1553;
        wp::int32 var_1554;
        wp::int32 var_1555;
        const wp::int32 var_1556 = 0;
        bool var_1557;
        wp::int32 var_1558;
        wp::float32 var_1559;
        const wp::int32 var_1560 = 2;
        wp::int32 var_1561;
        wp::int32* var_1562;
        wp::int32* var_1563;
        wp::int32 var_1564;
        wp::int32 var_1565;
        wp::int32 var_1566;
        const wp::int32 var_1567 = 0;
        bool var_1568;
        wp::int32 var_1569;
        wp::float32 var_1570;
        wp::mat_t<3, 3, wp::float32>* var_1571;
        wp::mat_t<3, 3, wp::float32> var_1572;
        wp::mat_t<3, 3, wp::float32> var_1573;
        wp::mat_t<3, 3, wp::float32> var_1574;
        const wp::int32 var_1575 = 0;
        wp::int32 var_1576;
        wp::int32* var_1577;
        wp::int32* var_1578;
        wp::int32 var_1579;
        wp::int32 var_1580;
        wp::int32 var_1581;
        const wp::int32 var_1582 = 0;
        bool var_1583;
        const wp::int32 var_1584 = 0;
        wp::int32 var_1585;
        wp::float32 var_1586;
        wp::float32 var_1587;
        const wp::int32 var_1588 = 1;
        wp::int32 var_1589;
        wp::float32 var_1590;
        wp::float32 var_1591;
        const wp::int32 var_1592 = 2;
        wp::int32 var_1593;
        wp::float32 var_1594;
        wp::float32 var_1595;
        wp::int32 var_1596;
        const wp::int32 var_1597 = 1;
        wp::int32 var_1598;
        wp::int32* var_1599;
        wp::int32* var_1600;
        wp::int32 var_1601;
        wp::int32 var_1602;
        wp::int32 var_1603;
        const wp::int32 var_1604 = 0;
        bool var_1605;
        const wp::int32 var_1606 = 0;
        wp::int32 var_1607;
        wp::float32 var_1608;
        wp::float32 var_1609;
        const wp::int32 var_1610 = 1;
        wp::int32 var_1611;
        wp::float32 var_1612;
        wp::float32 var_1613;
        const wp::int32 var_1614 = 2;
        wp::int32 var_1615;
        wp::float32 var_1616;
        wp::float32 var_1617;
        wp::int32 var_1618;
        const wp::int32 var_1619 = 2;
        wp::int32 var_1620;
        wp::int32* var_1621;
        wp::int32* var_1622;
        wp::int32 var_1623;
        wp::int32 var_1624;
        wp::int32 var_1625;
        const wp::int32 var_1626 = 0;
        bool var_1627;
        const wp::int32 var_1628 = 0;
        wp::int32 var_1629;
        wp::float32 var_1630;
        wp::float32 var_1631;
        const wp::int32 var_1632 = 1;
        wp::int32 var_1633;
        wp::float32 var_1634;
        wp::float32 var_1635;
        const wp::int32 var_1636 = 2;
        wp::int32 var_1637;
        wp::float32 var_1638;
        wp::float32 var_1639;
        wp::int32 var_1640;
        wp::int32 var_1641;
        wp::int32 var_1642;
        wp::int32 var_1643;
        wp::mat_t<3, 3, wp::float32> var_1644;
        wp::int32 var_1645;
        wp::int32 var_1646;
        wp::int32 var_1647;
        wp::int32 var_1648;
        wp::int32 var_1649;
        wp::mat_t<3, 3, wp::float32> var_1650;
        wp::int32 var_1651;
        wp::float32 var_1652;
        wp::int32 var_1653;
        wp::int32 var_1654;
        wp::int32 var_1655;
        wp::int32 var_1656;
        wp::mat_t<3, 3, wp::float32> var_1657;
        wp::int32 var_1658;
        bool var_1659;
        const wp::int32 var_1660 = 2;
        const bool var_1661 = true;
        wp::int32 var_1662;
        const wp::int32 var_1663 = 0;
        bool var_1664;
        const bool var_1665 = false;
        bool var_1666;
        const wp::int32 var_1667 = 4;
        wp::int32 var_1668;
        wp::int32 var_1669;
        wp::int32 var_1670;
        wp::float32* var_1671;
        wp::float32 var_1672;
        wp::float32 var_1673;
        const wp::float32 var_1674 = 0.0;
        bool var_1675;
        const bool var_1676 = false;
        bool var_1677;
        wp::int32 var_1678;
        bool var_1679;
        bool var_1680;
        wp::quat_t<wp::float32>* var_1681;
        bool var_1682;
        wp::quat_t<wp::float32> var_1683;
        const wp::int32 var_1684 = 0;
        wp::int32 var_1685;
        wp::int32* var_1686;
        wp::int32* var_1687;
        wp::int32 var_1688;
        wp::int32 var_1689;
        wp::int32 var_1690;
        const wp::int32 var_1691 = 0;
        bool var_1692;
        wp::int32 var_1693;
        wp::float32 var_1694;
        const wp::int32 var_1695 = 1;
        wp::int32 var_1696;
        wp::int32* var_1697;
        wp::int32* var_1698;
        wp::int32 var_1699;
        wp::int32 var_1700;
        wp::int32 var_1701;
        const wp::int32 var_1702 = 0;
        bool var_1703;
        wp::int32 var_1704;
        wp::float32 var_1705;
        const wp::int32 var_1706 = 2;
        wp::int32 var_1707;
        wp::int32* var_1708;
        wp::int32* var_1709;
        wp::int32 var_1710;
        wp::int32 var_1711;
        wp::int32 var_1712;
        const wp::int32 var_1713 = 0;
        bool var_1714;
        wp::int32 var_1715;
        wp::float32 var_1716;
        wp::mat_t<3, 3, wp::float32>* var_1717;
        wp::mat_t<3, 3, wp::float32> var_1718;
        wp::mat_t<3, 3, wp::float32> var_1719;
        wp::mat_t<3, 3, wp::float32> var_1720;
        const wp::int32 var_1721 = 0;
        wp::int32 var_1722;
        wp::int32* var_1723;
        wp::int32* var_1724;
        wp::int32 var_1725;
        wp::int32 var_1726;
        wp::int32 var_1727;
        const wp::int32 var_1728 = 0;
        bool var_1729;
        const wp::int32 var_1730 = 0;
        wp::int32 var_1731;
        wp::float32 var_1732;
        wp::float32 var_1733;
        const wp::int32 var_1734 = 1;
        wp::int32 var_1735;
        wp::float32 var_1736;
        wp::float32 var_1737;
        const wp::int32 var_1738 = 2;
        wp::int32 var_1739;
        wp::float32 var_1740;
        wp::float32 var_1741;
        wp::int32 var_1742;
        const wp::int32 var_1743 = 1;
        wp::int32 var_1744;
        wp::int32* var_1745;
        wp::int32* var_1746;
        wp::int32 var_1747;
        wp::int32 var_1748;
        wp::int32 var_1749;
        const wp::int32 var_1750 = 0;
        bool var_1751;
        const wp::int32 var_1752 = 0;
        wp::int32 var_1753;
        wp::float32 var_1754;
        wp::float32 var_1755;
        const wp::int32 var_1756 = 1;
        wp::int32 var_1757;
        wp::float32 var_1758;
        wp::float32 var_1759;
        const wp::int32 var_1760 = 2;
        wp::int32 var_1761;
        wp::float32 var_1762;
        wp::float32 var_1763;
        wp::int32 var_1764;
        const wp::int32 var_1765 = 2;
        wp::int32 var_1766;
        wp::int32* var_1767;
        wp::int32* var_1768;
        wp::int32 var_1769;
        wp::int32 var_1770;
        wp::int32 var_1771;
        const wp::int32 var_1772 = 0;
        bool var_1773;
        const wp::int32 var_1774 = 0;
        wp::int32 var_1775;
        wp::float32 var_1776;
        wp::float32 var_1777;
        const wp::int32 var_1778 = 1;
        wp::int32 var_1779;
        wp::float32 var_1780;
        wp::float32 var_1781;
        const wp::int32 var_1782 = 2;
        wp::int32 var_1783;
        wp::float32 var_1784;
        wp::float32 var_1785;
        wp::int32 var_1786;
        wp::int32 var_1787;
        wp::int32 var_1788;
        wp::int32 var_1789;
        wp::mat_t<3, 3, wp::float32> var_1790;
        wp::int32 var_1791;
        wp::int32 var_1792;
        wp::int32 var_1793;
        wp::int32 var_1794;
        wp::int32 var_1795;
        wp::mat_t<3, 3, wp::float32> var_1796;
        wp::int32 var_1797;
        wp::float32 var_1798;
        wp::int32 var_1799;
        wp::int32 var_1800;
        wp::int32 var_1801;
        wp::int32 var_1802;
        wp::mat_t<3, 3, wp::float32> var_1803;
        wp::int32 var_1804;
        bool var_1805;
        const wp::int32 var_1806 = 3;
        const bool var_1807 = true;
        wp::int32 var_1808;
        const wp::int32 var_1809 = 0;
        bool var_1810;
        const bool var_1811 = false;
        bool var_1812;
        const wp::int32 var_1813 = 4;
        wp::int32 var_1814;
        wp::int32 var_1815;
        wp::int32 var_1816;
        wp::float32* var_1817;
        wp::float32 var_1818;
        wp::float32 var_1819;
        const wp::float32 var_1820 = 0.0;
        bool var_1821;
        const bool var_1822 = false;
        bool var_1823;
        wp::int32 var_1824;
        bool var_1825;
        bool var_1826;
        wp::quat_t<wp::float32>* var_1827;
        bool var_1828;
        wp::quat_t<wp::float32> var_1829;
        const wp::int32 var_1830 = 0;
        wp::int32 var_1831;
        wp::int32* var_1832;
        wp::int32* var_1833;
        wp::int32 var_1834;
        wp::int32 var_1835;
        wp::int32 var_1836;
        const wp::int32 var_1837 = 0;
        bool var_1838;
        wp::int32 var_1839;
        wp::float32 var_1840;
        const wp::int32 var_1841 = 1;
        wp::int32 var_1842;
        wp::int32* var_1843;
        wp::int32* var_1844;
        wp::int32 var_1845;
        wp::int32 var_1846;
        wp::int32 var_1847;
        const wp::int32 var_1848 = 0;
        bool var_1849;
        wp::int32 var_1850;
        wp::float32 var_1851;
        const wp::int32 var_1852 = 2;
        wp::int32 var_1853;
        wp::int32* var_1854;
        wp::int32* var_1855;
        wp::int32 var_1856;
        wp::int32 var_1857;
        wp::int32 var_1858;
        const wp::int32 var_1859 = 0;
        bool var_1860;
        wp::int32 var_1861;
        wp::float32 var_1862;
        wp::mat_t<3, 3, wp::float32>* var_1863;
        wp::mat_t<3, 3, wp::float32> var_1864;
        wp::mat_t<3, 3, wp::float32> var_1865;
        wp::mat_t<3, 3, wp::float32> var_1866;
        const wp::int32 var_1867 = 0;
        wp::int32 var_1868;
        wp::int32* var_1869;
        wp::int32* var_1870;
        wp::int32 var_1871;
        wp::int32 var_1872;
        wp::int32 var_1873;
        const wp::int32 var_1874 = 0;
        bool var_1875;
        const wp::int32 var_1876 = 0;
        wp::int32 var_1877;
        wp::float32 var_1878;
        wp::float32 var_1879;
        const wp::int32 var_1880 = 1;
        wp::int32 var_1881;
        wp::float32 var_1882;
        wp::float32 var_1883;
        const wp::int32 var_1884 = 2;
        wp::int32 var_1885;
        wp::float32 var_1886;
        wp::float32 var_1887;
        wp::int32 var_1888;
        const wp::int32 var_1889 = 1;
        wp::int32 var_1890;
        wp::int32* var_1891;
        wp::int32* var_1892;
        wp::int32 var_1893;
        wp::int32 var_1894;
        wp::int32 var_1895;
        const wp::int32 var_1896 = 0;
        bool var_1897;
        const wp::int32 var_1898 = 0;
        wp::int32 var_1899;
        wp::float32 var_1900;
        wp::float32 var_1901;
        const wp::int32 var_1902 = 1;
        wp::int32 var_1903;
        wp::float32 var_1904;
        wp::float32 var_1905;
        const wp::int32 var_1906 = 2;
        wp::int32 var_1907;
        wp::float32 var_1908;
        wp::float32 var_1909;
        wp::int32 var_1910;
        const wp::int32 var_1911 = 2;
        wp::int32 var_1912;
        wp::int32* var_1913;
        wp::int32* var_1914;
        wp::int32 var_1915;
        wp::int32 var_1916;
        wp::int32 var_1917;
        const wp::int32 var_1918 = 0;
        bool var_1919;
        const wp::int32 var_1920 = 0;
        wp::int32 var_1921;
        wp::float32 var_1922;
        wp::float32 var_1923;
        const wp::int32 var_1924 = 1;
        wp::int32 var_1925;
        wp::float32 var_1926;
        wp::float32 var_1927;
        const wp::int32 var_1928 = 2;
        wp::int32 var_1929;
        wp::float32 var_1930;
        wp::float32 var_1931;
        wp::int32 var_1932;
        wp::int32 var_1933;
        wp::int32 var_1934;
        wp::int32 var_1935;
        wp::mat_t<3, 3, wp::float32> var_1936;
        wp::int32 var_1937;
        wp::int32 var_1938;
        wp::int32 var_1939;
        wp::int32 var_1940;
        wp::int32 var_1941;
        wp::mat_t<3, 3, wp::float32> var_1942;
        wp::int32 var_1943;
        wp::float32 var_1944;
        wp::int32 var_1945;
        wp::int32 var_1946;
        wp::int32 var_1947;
        wp::int32 var_1948;
        wp::mat_t<3, 3, wp::float32> var_1949;
        wp::int32 var_1950;
        bool var_1951;
        wp::int32 var_1952;
        wp::mat_t<3, 3, wp::float32> var_1953;
        wp::quat_t<wp::float32> var_1954;
        wp::int32 var_1955;
        wp::int32 var_1956;
        wp::float32 var_1957;
        wp::int32 var_1958;
        wp::int32 var_1959;
        wp::int32 var_1960;
        wp::int32 var_1961;
        wp::mat_t<3, 3, wp::float32> var_1962;
        wp::int32 var_1963;
        const wp::int32 var_1964 = 3;
        const bool var_1965 = true;
        wp::int32 var_1966;
        bool var_1967;
        const wp::int32 var_1968 = 0;
        bool var_1969;
        wp::int32* var_1970;
        const wp::int32 var_1971 = 0;
        bool var_1972;
        wp::int32 var_1973;
        const bool var_1974 = false;
        bool var_1975;
        wp::int32 var_1976;
        wp::mat_t<3, 3, wp::float32>* var_1977;
        wp::mat_t<3, 3, wp::float32> var_1978;
        wp::mat_t<3, 3, wp::float32> var_1979;
        wp::quat_t<wp::float32>* var_1980;
        wp::quat_t<wp::float32> var_1981;
        wp::quat_t<wp::float32> var_1982;
        const wp::int32 var_1983 = 0;
        const bool var_1984 = true;
        wp::int32 var_1985;
        const wp::int32 var_1986 = 0;
        bool var_1987;
        const bool var_1988 = false;
        bool var_1989;
        const wp::int32 var_1990 = 4;
        wp::int32 var_1991;
        wp::int32 var_1992;
        wp::int32 var_1993;
        wp::float32* var_1994;
        wp::float32 var_1995;
        wp::float32 var_1996;
        const wp::float32 var_1997 = 0.0;
        bool var_1998;
        const bool var_1999 = false;
        bool var_2000;
        wp::int32 var_2001;
        bool var_2002;
        bool var_2003;
        wp::quat_t<wp::float32>* var_2004;
        bool var_2005;
        wp::quat_t<wp::float32> var_2006;
        const wp::int32 var_2007 = 0;
        wp::int32 var_2008;
        wp::int32* var_2009;
        wp::int32* var_2010;
        wp::int32 var_2011;
        wp::int32 var_2012;
        wp::int32 var_2013;
        const wp::int32 var_2014 = 0;
        bool var_2015;
        wp::int32 var_2016;
        wp::float32 var_2017;
        const wp::int32 var_2018 = 1;
        wp::int32 var_2019;
        wp::int32* var_2020;
        wp::int32* var_2021;
        wp::int32 var_2022;
        wp::int32 var_2023;
        wp::int32 var_2024;
        const wp::int32 var_2025 = 0;
        bool var_2026;
        wp::int32 var_2027;
        wp::float32 var_2028;
        const wp::int32 var_2029 = 2;
        wp::int32 var_2030;
        wp::int32* var_2031;
        wp::int32* var_2032;
        wp::int32 var_2033;
        wp::int32 var_2034;
        wp::int32 var_2035;
        const wp::int32 var_2036 = 0;
        bool var_2037;
        wp::int32 var_2038;
        wp::float32 var_2039;
        wp::mat_t<3, 3, wp::float32>* var_2040;
        wp::mat_t<3, 3, wp::float32> var_2041;
        wp::mat_t<3, 3, wp::float32> var_2042;
        wp::mat_t<3, 3, wp::float32> var_2043;
        const wp::int32 var_2044 = 0;
        wp::int32 var_2045;
        wp::int32* var_2046;
        wp::int32* var_2047;
        wp::int32 var_2048;
        wp::int32 var_2049;
        wp::int32 var_2050;
        const wp::int32 var_2051 = 0;
        bool var_2052;
        const wp::int32 var_2053 = 0;
        wp::int32 var_2054;
        wp::float32 var_2055;
        wp::float32 var_2056;
        const wp::int32 var_2057 = 1;
        wp::int32 var_2058;
        wp::float32 var_2059;
        wp::float32 var_2060;
        const wp::int32 var_2061 = 2;
        wp::int32 var_2062;
        wp::float32 var_2063;
        wp::float32 var_2064;
        wp::int32 var_2065;
        const wp::int32 var_2066 = 1;
        wp::int32 var_2067;
        wp::int32* var_2068;
        wp::int32* var_2069;
        wp::int32 var_2070;
        wp::int32 var_2071;
        wp::int32 var_2072;
        const wp::int32 var_2073 = 0;
        bool var_2074;
        const wp::int32 var_2075 = 0;
        wp::int32 var_2076;
        wp::float32 var_2077;
        wp::float32 var_2078;
        const wp::int32 var_2079 = 1;
        wp::int32 var_2080;
        wp::float32 var_2081;
        wp::float32 var_2082;
        const wp::int32 var_2083 = 2;
        wp::int32 var_2084;
        wp::float32 var_2085;
        wp::float32 var_2086;
        wp::int32 var_2087;
        const wp::int32 var_2088 = 2;
        wp::int32 var_2089;
        wp::int32* var_2090;
        wp::int32* var_2091;
        wp::int32 var_2092;
        wp::int32 var_2093;
        wp::int32 var_2094;
        const wp::int32 var_2095 = 0;
        bool var_2096;
        const wp::int32 var_2097 = 0;
        wp::int32 var_2098;
        wp::float32 var_2099;
        wp::float32 var_2100;
        const wp::int32 var_2101 = 1;
        wp::int32 var_2102;
        wp::float32 var_2103;
        wp::float32 var_2104;
        const wp::int32 var_2105 = 2;
        wp::int32 var_2106;
        wp::float32 var_2107;
        wp::float32 var_2108;
        wp::int32 var_2109;
        wp::int32 var_2110;
        wp::int32 var_2111;
        wp::int32 var_2112;
        wp::mat_t<3, 3, wp::float32> var_2113;
        wp::int32 var_2114;
        wp::int32 var_2115;
        wp::int32 var_2116;
        wp::int32 var_2117;
        wp::int32 var_2118;
        wp::mat_t<3, 3, wp::float32> var_2119;
        wp::int32 var_2120;
        wp::float32 var_2121;
        wp::int32 var_2122;
        wp::int32 var_2123;
        wp::int32 var_2124;
        wp::int32 var_2125;
        wp::mat_t<3, 3, wp::float32> var_2126;
        wp::int32 var_2127;
        bool var_2128;
        const wp::int32 var_2129 = 1;
        const bool var_2130 = true;
        wp::int32 var_2131;
        const wp::int32 var_2132 = 0;
        bool var_2133;
        const bool var_2134 = false;
        bool var_2135;
        const wp::int32 var_2136 = 4;
        wp::int32 var_2137;
        wp::int32 var_2138;
        wp::int32 var_2139;
        wp::float32* var_2140;
        wp::float32 var_2141;
        wp::float32 var_2142;
        const wp::float32 var_2143 = 0.0;
        bool var_2144;
        const bool var_2145 = false;
        bool var_2146;
        wp::int32 var_2147;
        bool var_2148;
        bool var_2149;
        wp::quat_t<wp::float32>* var_2150;
        bool var_2151;
        wp::quat_t<wp::float32> var_2152;
        const wp::int32 var_2153 = 0;
        wp::int32 var_2154;
        wp::int32* var_2155;
        wp::int32* var_2156;
        wp::int32 var_2157;
        wp::int32 var_2158;
        wp::int32 var_2159;
        const wp::int32 var_2160 = 0;
        bool var_2161;
        wp::int32 var_2162;
        wp::float32 var_2163;
        const wp::int32 var_2164 = 1;
        wp::int32 var_2165;
        wp::int32* var_2166;
        wp::int32* var_2167;
        wp::int32 var_2168;
        wp::int32 var_2169;
        wp::int32 var_2170;
        const wp::int32 var_2171 = 0;
        bool var_2172;
        wp::int32 var_2173;
        wp::float32 var_2174;
        const wp::int32 var_2175 = 2;
        wp::int32 var_2176;
        wp::int32* var_2177;
        wp::int32* var_2178;
        wp::int32 var_2179;
        wp::int32 var_2180;
        wp::int32 var_2181;
        const wp::int32 var_2182 = 0;
        bool var_2183;
        wp::int32 var_2184;
        wp::float32 var_2185;
        wp::mat_t<3, 3, wp::float32>* var_2186;
        wp::mat_t<3, 3, wp::float32> var_2187;
        wp::mat_t<3, 3, wp::float32> var_2188;
        wp::mat_t<3, 3, wp::float32> var_2189;
        const wp::int32 var_2190 = 0;
        wp::int32 var_2191;
        wp::int32* var_2192;
        wp::int32* var_2193;
        wp::int32 var_2194;
        wp::int32 var_2195;
        wp::int32 var_2196;
        const wp::int32 var_2197 = 0;
        bool var_2198;
        const wp::int32 var_2199 = 0;
        wp::int32 var_2200;
        wp::float32 var_2201;
        wp::float32 var_2202;
        const wp::int32 var_2203 = 1;
        wp::int32 var_2204;
        wp::float32 var_2205;
        wp::float32 var_2206;
        const wp::int32 var_2207 = 2;
        wp::int32 var_2208;
        wp::float32 var_2209;
        wp::float32 var_2210;
        wp::int32 var_2211;
        const wp::int32 var_2212 = 1;
        wp::int32 var_2213;
        wp::int32* var_2214;
        wp::int32* var_2215;
        wp::int32 var_2216;
        wp::int32 var_2217;
        wp::int32 var_2218;
        const wp::int32 var_2219 = 0;
        bool var_2220;
        const wp::int32 var_2221 = 0;
        wp::int32 var_2222;
        wp::float32 var_2223;
        wp::float32 var_2224;
        const wp::int32 var_2225 = 1;
        wp::int32 var_2226;
        wp::float32 var_2227;
        wp::float32 var_2228;
        const wp::int32 var_2229 = 2;
        wp::int32 var_2230;
        wp::float32 var_2231;
        wp::float32 var_2232;
        wp::int32 var_2233;
        const wp::int32 var_2234 = 2;
        wp::int32 var_2235;
        wp::int32* var_2236;
        wp::int32* var_2237;
        wp::int32 var_2238;
        wp::int32 var_2239;
        wp::int32 var_2240;
        const wp::int32 var_2241 = 0;
        bool var_2242;
        const wp::int32 var_2243 = 0;
        wp::int32 var_2244;
        wp::float32 var_2245;
        wp::float32 var_2246;
        const wp::int32 var_2247 = 1;
        wp::int32 var_2248;
        wp::float32 var_2249;
        wp::float32 var_2250;
        const wp::int32 var_2251 = 2;
        wp::int32 var_2252;
        wp::float32 var_2253;
        wp::float32 var_2254;
        wp::int32 var_2255;
        wp::int32 var_2256;
        wp::int32 var_2257;
        wp::int32 var_2258;
        wp::mat_t<3, 3, wp::float32> var_2259;
        wp::int32 var_2260;
        wp::int32 var_2261;
        wp::int32 var_2262;
        wp::int32 var_2263;
        wp::int32 var_2264;
        wp::mat_t<3, 3, wp::float32> var_2265;
        wp::int32 var_2266;
        wp::float32 var_2267;
        wp::int32 var_2268;
        wp::int32 var_2269;
        wp::int32 var_2270;
        wp::int32 var_2271;
        wp::mat_t<3, 3, wp::float32> var_2272;
        wp::int32 var_2273;
        bool var_2274;
        const wp::int32 var_2275 = 2;
        const bool var_2276 = true;
        wp::int32 var_2277;
        const wp::int32 var_2278 = 0;
        bool var_2279;
        const bool var_2280 = false;
        bool var_2281;
        const wp::int32 var_2282 = 4;
        wp::int32 var_2283;
        wp::int32 var_2284;
        wp::int32 var_2285;
        wp::float32* var_2286;
        wp::float32 var_2287;
        wp::float32 var_2288;
        const wp::float32 var_2289 = 0.0;
        bool var_2290;
        const bool var_2291 = false;
        bool var_2292;
        wp::int32 var_2293;
        bool var_2294;
        bool var_2295;
        wp::quat_t<wp::float32>* var_2296;
        bool var_2297;
        wp::quat_t<wp::float32> var_2298;
        const wp::int32 var_2299 = 0;
        wp::int32 var_2300;
        wp::int32* var_2301;
        wp::int32* var_2302;
        wp::int32 var_2303;
        wp::int32 var_2304;
        wp::int32 var_2305;
        const wp::int32 var_2306 = 0;
        bool var_2307;
        wp::int32 var_2308;
        wp::float32 var_2309;
        const wp::int32 var_2310 = 1;
        wp::int32 var_2311;
        wp::int32* var_2312;
        wp::int32* var_2313;
        wp::int32 var_2314;
        wp::int32 var_2315;
        wp::int32 var_2316;
        const wp::int32 var_2317 = 0;
        bool var_2318;
        wp::int32 var_2319;
        wp::float32 var_2320;
        const wp::int32 var_2321 = 2;
        wp::int32 var_2322;
        wp::int32* var_2323;
        wp::int32* var_2324;
        wp::int32 var_2325;
        wp::int32 var_2326;
        wp::int32 var_2327;
        const wp::int32 var_2328 = 0;
        bool var_2329;
        wp::int32 var_2330;
        wp::float32 var_2331;
        wp::mat_t<3, 3, wp::float32>* var_2332;
        wp::mat_t<3, 3, wp::float32> var_2333;
        wp::mat_t<3, 3, wp::float32> var_2334;
        wp::mat_t<3, 3, wp::float32> var_2335;
        const wp::int32 var_2336 = 0;
        wp::int32 var_2337;
        wp::int32* var_2338;
        wp::int32* var_2339;
        wp::int32 var_2340;
        wp::int32 var_2341;
        wp::int32 var_2342;
        const wp::int32 var_2343 = 0;
        bool var_2344;
        const wp::int32 var_2345 = 0;
        wp::int32 var_2346;
        wp::float32 var_2347;
        wp::float32 var_2348;
        const wp::int32 var_2349 = 1;
        wp::int32 var_2350;
        wp::float32 var_2351;
        wp::float32 var_2352;
        const wp::int32 var_2353 = 2;
        wp::int32 var_2354;
        wp::float32 var_2355;
        wp::float32 var_2356;
        wp::int32 var_2357;
        const wp::int32 var_2358 = 1;
        wp::int32 var_2359;
        wp::int32* var_2360;
        wp::int32* var_2361;
        wp::int32 var_2362;
        wp::int32 var_2363;
        wp::int32 var_2364;
        const wp::int32 var_2365 = 0;
        bool var_2366;
        const wp::int32 var_2367 = 0;
        wp::int32 var_2368;
        wp::float32 var_2369;
        wp::float32 var_2370;
        const wp::int32 var_2371 = 1;
        wp::int32 var_2372;
        wp::float32 var_2373;
        wp::float32 var_2374;
        const wp::int32 var_2375 = 2;
        wp::int32 var_2376;
        wp::float32 var_2377;
        wp::float32 var_2378;
        wp::int32 var_2379;
        const wp::int32 var_2380 = 2;
        wp::int32 var_2381;
        wp::int32* var_2382;
        wp::int32* var_2383;
        wp::int32 var_2384;
        wp::int32 var_2385;
        wp::int32 var_2386;
        const wp::int32 var_2387 = 0;
        bool var_2388;
        const wp::int32 var_2389 = 0;
        wp::int32 var_2390;
        wp::float32 var_2391;
        wp::float32 var_2392;
        const wp::int32 var_2393 = 1;
        wp::int32 var_2394;
        wp::float32 var_2395;
        wp::float32 var_2396;
        const wp::int32 var_2397 = 2;
        wp::int32 var_2398;
        wp::float32 var_2399;
        wp::float32 var_2400;
        wp::int32 var_2401;
        wp::int32 var_2402;
        wp::int32 var_2403;
        wp::int32 var_2404;
        wp::mat_t<3, 3, wp::float32> var_2405;
        wp::int32 var_2406;
        wp::int32 var_2407;
        wp::int32 var_2408;
        wp::int32 var_2409;
        wp::int32 var_2410;
        wp::mat_t<3, 3, wp::float32> var_2411;
        wp::int32 var_2412;
        wp::float32 var_2413;
        wp::int32 var_2414;
        wp::int32 var_2415;
        wp::int32 var_2416;
        wp::int32 var_2417;
        wp::mat_t<3, 3, wp::float32> var_2418;
        wp::int32 var_2419;
        bool var_2420;
        const wp::int32 var_2421 = 3;
        const bool var_2422 = true;
        wp::int32 var_2423;
        const wp::int32 var_2424 = 0;
        bool var_2425;
        const bool var_2426 = false;
        bool var_2427;
        const wp::int32 var_2428 = 4;
        wp::int32 var_2429;
        wp::int32 var_2430;
        wp::int32 var_2431;
        wp::float32* var_2432;
        wp::float32 var_2433;
        wp::float32 var_2434;
        const wp::float32 var_2435 = 0.0;
        bool var_2436;
        const bool var_2437 = false;
        bool var_2438;
        wp::int32 var_2439;
        bool var_2440;
        bool var_2441;
        wp::quat_t<wp::float32>* var_2442;
        bool var_2443;
        wp::quat_t<wp::float32> var_2444;
        const wp::int32 var_2445 = 0;
        wp::int32 var_2446;
        wp::int32* var_2447;
        wp::int32* var_2448;
        wp::int32 var_2449;
        wp::int32 var_2450;
        wp::int32 var_2451;
        const wp::int32 var_2452 = 0;
        bool var_2453;
        wp::int32 var_2454;
        wp::float32 var_2455;
        const wp::int32 var_2456 = 1;
        wp::int32 var_2457;
        wp::int32* var_2458;
        wp::int32* var_2459;
        wp::int32 var_2460;
        wp::int32 var_2461;
        wp::int32 var_2462;
        const wp::int32 var_2463 = 0;
        bool var_2464;
        wp::int32 var_2465;
        wp::float32 var_2466;
        const wp::int32 var_2467 = 2;
        wp::int32 var_2468;
        wp::int32* var_2469;
        wp::int32* var_2470;
        wp::int32 var_2471;
        wp::int32 var_2472;
        wp::int32 var_2473;
        const wp::int32 var_2474 = 0;
        bool var_2475;
        wp::int32 var_2476;
        wp::float32 var_2477;
        wp::mat_t<3, 3, wp::float32>* var_2478;
        wp::mat_t<3, 3, wp::float32> var_2479;
        wp::mat_t<3, 3, wp::float32> var_2480;
        wp::mat_t<3, 3, wp::float32> var_2481;
        const wp::int32 var_2482 = 0;
        wp::int32 var_2483;
        wp::int32* var_2484;
        wp::int32* var_2485;
        wp::int32 var_2486;
        wp::int32 var_2487;
        wp::int32 var_2488;
        const wp::int32 var_2489 = 0;
        bool var_2490;
        const wp::int32 var_2491 = 0;
        wp::int32 var_2492;
        wp::float32 var_2493;
        wp::float32 var_2494;
        const wp::int32 var_2495 = 1;
        wp::int32 var_2496;
        wp::float32 var_2497;
        wp::float32 var_2498;
        const wp::int32 var_2499 = 2;
        wp::int32 var_2500;
        wp::float32 var_2501;
        wp::float32 var_2502;
        wp::int32 var_2503;
        const wp::int32 var_2504 = 1;
        wp::int32 var_2505;
        wp::int32* var_2506;
        wp::int32* var_2507;
        wp::int32 var_2508;
        wp::int32 var_2509;
        wp::int32 var_2510;
        const wp::int32 var_2511 = 0;
        bool var_2512;
        const wp::int32 var_2513 = 0;
        wp::int32 var_2514;
        wp::float32 var_2515;
        wp::float32 var_2516;
        const wp::int32 var_2517 = 1;
        wp::int32 var_2518;
        wp::float32 var_2519;
        wp::float32 var_2520;
        const wp::int32 var_2521 = 2;
        wp::int32 var_2522;
        wp::float32 var_2523;
        wp::float32 var_2524;
        wp::int32 var_2525;
        const wp::int32 var_2526 = 2;
        wp::int32 var_2527;
        wp::int32* var_2528;
        wp::int32* var_2529;
        wp::int32 var_2530;
        wp::int32 var_2531;
        wp::int32 var_2532;
        const wp::int32 var_2533 = 0;
        bool var_2534;
        const wp::int32 var_2535 = 0;
        wp::int32 var_2536;
        wp::float32 var_2537;
        wp::float32 var_2538;
        const wp::int32 var_2539 = 1;
        wp::int32 var_2540;
        wp::float32 var_2541;
        wp::float32 var_2542;
        const wp::int32 var_2543 = 2;
        wp::int32 var_2544;
        wp::float32 var_2545;
        wp::float32 var_2546;
        wp::int32 var_2547;
        wp::int32 var_2548;
        wp::int32 var_2549;
        wp::int32 var_2550;
        wp::mat_t<3, 3, wp::float32> var_2551;
        wp::int32 var_2552;
        wp::int32 var_2553;
        wp::int32 var_2554;
        wp::int32 var_2555;
        wp::int32 var_2556;
        wp::mat_t<3, 3, wp::float32> var_2557;
        wp::int32 var_2558;
        wp::float32 var_2559;
        wp::int32 var_2560;
        wp::int32 var_2561;
        wp::int32 var_2562;
        wp::int32 var_2563;
        wp::mat_t<3, 3, wp::float32> var_2564;
        wp::int32 var_2565;
        bool var_2566;
        wp::int32 var_2567;
        wp::mat_t<3, 3, wp::float32> var_2568;
        wp::quat_t<wp::float32> var_2569;
        wp::int32 var_2570;
        wp::int32 var_2571;
        wp::float32 var_2572;
        wp::int32 var_2573;
        wp::int32 var_2574;
        wp::int32 var_2575;
        wp::int32 var_2576;
        wp::mat_t<3, 3, wp::float32> var_2577;
        wp::int32 var_2578;
        //---------
        // forward
        // def _eff_flex_bend_stiff(                                                              <L 2133>
        // worldid, edgeid = wp.tid()                                                             <L 2164>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 2165>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // f = flex_edgeflexid[edgeid]                                                            <L 2167>
        var_10 = wp::address(var_flex_edgeflexid, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if f < 0:                                                                              <L 2169>
        var_14 = (var_11 < var_13);
        if (var_14) {
            // return                                                                             <L 2170>
            return;
        }
        // if flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] != 2 or not flex_simple[f]:       <L 2172>
        var_16 = wp::address(var_flex_interp, var_11);
        var_19 = wp::load(var_16);
        var_18 = (var_19 != var_17);
        var_15 = var_18;
        if (!var_15) {
            var_20 = wp::address(var_flex_rigid, var_11);
            var_21 = wp::load(var_20);
            var_15 = var_15 || var_21;
        }
        if (!var_15) {
            var_22 = wp::address(var_flex_dim, var_11);
            var_25 = wp::load(var_22);
            var_24 = (var_25 != var_23);
            var_15 = var_15 || var_24;
        }
        if (!var_15) {
            var_26 = wp::address(var_flex_simple, var_11);
            var_28 = wp::load(var_26);
            var_27 = wp::unot(var_28);
            var_15 = var_15 || var_27;
        }
        if (var_15) {
            // return                                                                             <L 2173>
            return;
        }
        // bendingadr = flex_bendingadr[f]                                                        <L 2175>
        var_29 = wp::address(var_flex_bendingadr, var_11);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // if bendingadr < 0:                                                                     <L 2176>
        var_33 = (var_30 < var_32);
        if (var_33) {
            // return                                                                             <L 2177>
            return;
        }
        // s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep            <L 2179>
        var_35 = wp::bit_and(var_opt_disableflags, var_34);
        if (var_35) {
        }
        if (!var_35) {
            var_37 = wp::mul(var_8, var_8);
        }
        var_38 = wp::where(var_35, var_36, var_37);
        // s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep                       <L 2180>
        var_40 = wp::bit_and(var_opt_disableflags, var_39);
        if (var_40) {
        }
        if (!var_40) {
        }
        var_42 = wp::where(var_40, var_41, var_8);
        // scale = s1 + s2 * flex_damping[f]                                                      <L 2181>
        var_43 = wp::address(var_flex_damping, var_11);
        var_45 = wp::load(var_43);
        var_44 = wp::mul(var_42, var_45);
        var_46 = wp::add(var_38, var_44);
        // if scale == 0.0:                                                                       <L 2182>
        var_48 = (var_46 == var_47);
        if (var_48) {
            // return                                                                             <L 2183>
            return;
        }
        // local_edgeid = edgeid - flex_edgeadr[f]                                                <L 2185>
        var_49 = wp::address(var_flex_edgeadr, var_11);
        var_51 = wp::load(var_49);
        var_50 = wp::sub(var_1, var_51);
        // vbase = flex_vertadr[f]                                                                <L 2186>
        var_52 = wp::address(var_flex_vertadr, var_11);
        var_54 = wp::load(var_52);
        var_53 = wp::copy(var_54);
        // edge = flex_edge[edgeid]                                                               <L 2188>
        var_55 = wp::address(var_flex_edge, var_1);
        var_57 = wp::load(var_55);
        var_56 = wp::copy(var_57);
        // flap = flex_edgeflap[edgeid]                                                           <L 2189>
        var_58 = wp::address(var_flex_edgeflap, var_1);
        var_60 = wp::load(var_58);
        var_59 = wp::copy(var_60);
        // if flap[1] == -1:                                                                      <L 2191>
        var_62 = wp::extract(var_59, var_61);
        var_64 = (var_62 == var_63);
        if (var_64) {
            // return                                                                             <L 2192>
            return;
        }
        // verts = wp.vec4i(edge[0], edge[1], flap[0], flap[1])                                   <L 2194>
        var_66 = wp::extract(var_56, var_65);
        var_68 = wp::extract(var_56, var_67);
        var_70 = wp::extract(var_59, var_69);
        var_72 = wp::extract(var_59, var_71);
        var_73 = wp::vec_t<4, wp::int32>(var_66, var_68, var_70, var_72);
        // bend_offset = bendingadr + 17 * local_edgeid                                           <L 2195>
        var_75 = wp::mul(var_74, var_50);
        var_76 = wp::add(var_30, var_75);
        // dofs = wp.vec4i(-1, -1, -1, -1)                                                        <L 2197>
        var_81 = wp::vec_t<4, wp::int32>(var_77, var_78, var_79, var_80);
        // bodies = wp.vec4i(-1, -1, -1, -1)                                                      <L 2198>
        var_86 = wp::vec_t<4, wp::int32>(var_82, var_83, var_84, var_85);
        // for k in range(4):                                                                     <L 2199>
        // bk = body_weldid[flex_vertbodyid[vbase + verts[k]]]                                    <L 2200>
        var_88 = wp::extract(var_73, var_87);
        var_89 = wp::add(var_53, var_88);
        var_90 = wp::address(var_flex_vertbodyid, var_89);
        var_91 = wp::load(var_90);
        var_92 = wp::address(var_body_weldid, var_91);
        var_94 = wp::load(var_92);
        var_93 = wp::copy(var_94);
        // bodies[k] = bk                                                                         <L 2201>
        wp::assign_inplace(var_86, var_87, var_93);
        // if body_dofnum[bk] == 3:                                                               <L 2202>
        var_95 = wp::address(var_body_dofnum, var_93);
        var_98 = wp::load(var_95);
        var_97 = (var_98 == var_96);
        if (var_97) {
            // dofs[k] = body_dofadr[bk]                                                          <L 2203>
            var_99 = wp::address(var_body_dofadr, var_93);
            var_101 = wp::load(var_99);
            var_100 = wp::copy(var_101);
            wp::assign_inplace(var_81, var_87, var_100);
        }
        // bk = body_weldid[flex_vertbodyid[vbase + verts[k]]]                                    <L 2200>
        var_103 = wp::extract(var_73, var_102);
        var_104 = wp::add(var_53, var_103);
        var_105 = wp::address(var_flex_vertbodyid, var_104);
        var_106 = wp::load(var_105);
        var_107 = wp::address(var_body_weldid, var_106);
        var_109 = wp::load(var_107);
        var_108 = wp::copy(var_109);
        // bodies[k] = bk                                                                         <L 2201>
        wp::assign_inplace(var_86, var_102, var_108);
        // if body_dofnum[bk] == 3:                                                               <L 2202>
        var_110 = wp::address(var_body_dofnum, var_108);
        var_113 = wp::load(var_110);
        var_112 = (var_113 == var_111);
        if (var_112) {
            // dofs[k] = body_dofadr[bk]                                                          <L 2203>
            var_114 = wp::address(var_body_dofadr, var_108);
            var_116 = wp::load(var_114);
            var_115 = wp::copy(var_116);
            wp::assign_inplace(var_81, var_102, var_115);
        }
        // bk = body_weldid[flex_vertbodyid[vbase + verts[k]]]                                    <L 2200>
        var_118 = wp::extract(var_73, var_117);
        var_119 = wp::add(var_53, var_118);
        var_120 = wp::address(var_flex_vertbodyid, var_119);
        var_121 = wp::load(var_120);
        var_122 = wp::address(var_body_weldid, var_121);
        var_124 = wp::load(var_122);
        var_123 = wp::copy(var_124);
        // bodies[k] = bk                                                                         <L 2201>
        wp::assign_inplace(var_86, var_117, var_123);
        // if body_dofnum[bk] == 3:                                                               <L 2202>
        var_125 = wp::address(var_body_dofnum, var_123);
        var_128 = wp::load(var_125);
        var_127 = (var_128 == var_126);
        if (var_127) {
            // dofs[k] = body_dofadr[bk]                                                          <L 2203>
            var_129 = wp::address(var_body_dofadr, var_123);
            var_131 = wp::load(var_129);
            var_130 = wp::copy(var_131);
            wp::assign_inplace(var_81, var_117, var_130);
        }
        // bk = body_weldid[flex_vertbodyid[vbase + verts[k]]]                                    <L 2200>
        var_133 = wp::extract(var_73, var_132);
        var_134 = wp::add(var_53, var_133);
        var_135 = wp::address(var_flex_vertbodyid, var_134);
        var_136 = wp::load(var_135);
        var_137 = wp::address(var_body_weldid, var_136);
        var_139 = wp::load(var_137);
        var_138 = wp::copy(var_139);
        // bodies[k] = bk                                                                         <L 2201>
        wp::assign_inplace(var_86, var_132, var_138);
        // if body_dofnum[bk] == 3:                                                               <L 2202>
        var_140 = wp::address(var_body_dofnum, var_138);
        var_143 = wp::load(var_140);
        var_142 = (var_143 == var_141);
        if (var_142) {
            // dofs[k] = body_dofadr[bk]                                                          <L 2203>
            var_144 = wp::address(var_body_dofadr, var_138);
            var_146 = wp::load(var_144);
            var_145 = wp::copy(var_146);
            wp::assign_inplace(var_81, var_132, var_145);
        }
        // for i in range(4):                                                                     <L 2205>
        // dof_i = dofs[i]                                                                        <L 2206>
        var_149 = wp::extract(var_81, var_147);
        // if dof_i < 0 or efm_K_rownnz[dof_i] == 0:                                              <L 2207>
        var_152 = (var_149 < var_151);
        var_150 = var_152;
        if (!var_150) {
            var_153 = wp::address(var_efm_K_rownnz, var_149);
            var_156 = wp::load(var_153);
            var_155 = (var_156 == var_154);
            var_150 = var_150 || var_155;
        }
        if (var_150) {
            // continue                                                                           <L 2208>
        }
        var_158 = wp::where(var_150, var_157, var_148);
        // bi = bodies[i]                                                                         <L 2209>
        if (var_158) {
            var_159 = wp::extract(var_86, var_147);
            // R_bi_T = wp.transpose(xmat_in[worldid, bi])                                        <L 2210>
            var_160 = wp::address(var_xmat_in, var_0, var_159);
            var_162 = wp::load(var_160);
            var_161 = wp::transpose(var_162);
            // qi = xquat_in[worldid, bi]                                                         <L 2211>
            var_163 = wp::address(var_xquat_in, var_0, var_159);
            var_165 = wp::load(var_163);
            var_164 = wp::copy(var_165);
            // for j in range(4):                                                                 <L 2212>
            // dof_j = dofs[j]                                                                    <L 2213>
            var_168 = wp::extract(var_81, var_166);
            // if dof_j < 0:                                                                      <L 2214>
            var_170 = (var_168 < var_169);
            if (var_170) {
                // continue                                                                       <L 2215>
            }
            var_172 = wp::where(var_170, var_171, var_167);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_172) {
                var_174 = wp::mul(var_173, var_147);
                var_175 = wp::add(var_76, var_174);
                var_176 = wp::add(var_175, var_166);
                var_177 = wp::address(var_flex_bending, var_176);
                var_179 = wp::load(var_177);
                var_178 = wp::mul(var_46, var_179);
                // if q == 0.0:                                                                   <L 2217>
                var_181 = (var_178 == var_180);
                if (var_181) {
                    // continue                                                                   <L 2218>
                }
                var_183 = wp::where(var_181, var_182, var_172);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_183) {
                    var_184 = wp::extract(var_86, var_166);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_186 = (var_159 == var_184);
                    var_185 = var_186;
                    if (!var_185) {
                        var_187 = wp::address(var_xquat_in, var_0, var_184);
                        var_189 = wp::load(var_187);
                        var_188 = (var_164 == var_189);
                        var_185 = var_185 || var_188;
                    }
                    if (var_185) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_191 = wp::add(var_149, var_190);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_192 = wp::address(var_efm_K_rowadr, var_191);
                        var_193 = wp::address(var_efm_K_rownnz, var_191);
                        var_195 = wp::load(var_192);
                        var_196 = wp::load(var_193);
                        var_194 = _find_csr_col_0(var_195, var_196, var_168, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_198 = (var_194 >= var_197);
                        if (var_198) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_199 = wp::add(var_194, var_190);
                            var_200 = wp::atomic_add(var_efm_K_val_out, var_0, var_199, var_178);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_202 = wp::add(var_149, var_201);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_203 = wp::address(var_efm_K_rowadr, var_202);
                        var_204 = wp::address(var_efm_K_rownnz, var_202);
                        var_206 = wp::load(var_203);
                        var_207 = wp::load(var_204);
                        var_205 = _find_csr_col_0(var_206, var_207, var_168, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_209 = (var_205 >= var_208);
                        if (var_209) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_210 = wp::add(var_205, var_201);
                            var_211 = wp::atomic_add(var_efm_K_val_out, var_0, var_210, var_178);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_213 = wp::add(var_149, var_212);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_214 = wp::address(var_efm_K_rowadr, var_213);
                        var_215 = wp::address(var_efm_K_rownnz, var_213);
                        var_217 = wp::load(var_214);
                        var_218 = wp::load(var_215);
                        var_216 = _find_csr_col_0(var_217, var_218, var_168, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_220 = (var_216 >= var_219);
                        if (var_220) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_221 = wp::add(var_216, var_212);
                            var_222 = wp::atomic_add(var_efm_K_val_out, var_0, var_221, var_178);
                        }
                    }
                    if (!var_185) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_223 = wp::address(var_xmat_in, var_0, var_184);
                        var_225 = wp::load(var_223);
                        var_224 = wp::mul(var_161, var_225);
                        var_226 = wp::mul(var_178, var_224);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_228 = wp::add(var_149, var_227);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_229 = wp::address(var_efm_K_rowadr, var_228);
                        var_230 = wp::address(var_efm_K_rownnz, var_228);
                        var_232 = wp::load(var_229);
                        var_233 = wp::load(var_230);
                        var_231 = _find_csr_col_0(var_232, var_233, var_168, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_235 = (var_231 >= var_234);
                        if (var_235) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_237 = wp::add(var_231, var_236);
                            var_238 = wp::extract(var_226, var_227, var_236);
                            var_239 = wp::atomic_add(var_efm_K_val_out, var_0, var_237, var_238);
                            var_241 = wp::add(var_231, var_240);
                            var_242 = wp::extract(var_226, var_227, var_240);
                            var_243 = wp::atomic_add(var_efm_K_val_out, var_0, var_241, var_242);
                            var_245 = wp::add(var_231, var_244);
                            var_246 = wp::extract(var_226, var_227, var_244);
                            var_247 = wp::atomic_add(var_efm_K_val_out, var_0, var_245, var_246);
                        }
                        // row = dof_i + r                                                        <L 2229>
                        var_249 = wp::add(var_149, var_248);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_250 = wp::address(var_efm_K_rowadr, var_249);
                        var_251 = wp::address(var_efm_K_rownnz, var_249);
                        var_253 = wp::load(var_250);
                        var_254 = wp::load(var_251);
                        var_252 = _find_csr_col_0(var_253, var_254, var_168, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_256 = (var_252 >= var_255);
                        if (var_256) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_258 = wp::add(var_252, var_257);
                            var_259 = wp::extract(var_226, var_248, var_257);
                            var_260 = wp::atomic_add(var_efm_K_val_out, var_0, var_258, var_259);
                            var_262 = wp::add(var_252, var_261);
                            var_263 = wp::extract(var_226, var_248, var_261);
                            var_264 = wp::atomic_add(var_efm_K_val_out, var_0, var_262, var_263);
                            var_266 = wp::add(var_252, var_265);
                            var_267 = wp::extract(var_226, var_248, var_265);
                            var_268 = wp::atomic_add(var_efm_K_val_out, var_0, var_266, var_267);
                        }
                        var_269 = wp::where(var_256, var_265, var_244);
                        // row = dof_i + r                                                        <L 2229>
                        var_271 = wp::add(var_149, var_270);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_272 = wp::address(var_efm_K_rowadr, var_271);
                        var_273 = wp::address(var_efm_K_rownnz, var_271);
                        var_275 = wp::load(var_272);
                        var_276 = wp::load(var_273);
                        var_274 = _find_csr_col_0(var_275, var_276, var_168, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_278 = (var_274 >= var_277);
                        if (var_278) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_280 = wp::add(var_274, var_279);
                            var_281 = wp::extract(var_226, var_270, var_279);
                            var_282 = wp::atomic_add(var_efm_K_val_out, var_0, var_280, var_281);
                            var_284 = wp::add(var_274, var_283);
                            var_285 = wp::extract(var_226, var_270, var_283);
                            var_286 = wp::atomic_add(var_efm_K_val_out, var_0, var_284, var_285);
                            var_288 = wp::add(var_274, var_287);
                            var_289 = wp::extract(var_226, var_270, var_287);
                            var_290 = wp::atomic_add(var_efm_K_val_out, var_0, var_288, var_289);
                        }
                        var_291 = wp::where(var_278, var_287, var_269);
                    }
                    var_292 = wp::where(var_185, var_212, var_270);
                    var_293 = wp::where(var_185, var_213, var_271);
                    var_294 = wp::where(var_185, var_216, var_274);
                }
            }
            var_295 = wp::where(var_172, var_183, var_172);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_298 = wp::extract(var_81, var_296);
            // if dof_j < 0:                                                                      <L 2214>
            var_300 = (var_298 < var_299);
            if (var_300) {
                // continue                                                                       <L 2215>
            }
            var_302 = wp::where(var_300, var_301, var_297);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_302) {
                var_304 = wp::mul(var_303, var_147);
                var_305 = wp::add(var_76, var_304);
                var_306 = wp::add(var_305, var_296);
                var_307 = wp::address(var_flex_bending, var_306);
                var_309 = wp::load(var_307);
                var_308 = wp::mul(var_46, var_309);
                // if q == 0.0:                                                                   <L 2217>
                var_311 = (var_308 == var_310);
                if (var_311) {
                    // continue                                                                   <L 2218>
                }
                var_313 = wp::where(var_311, var_312, var_302);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_313) {
                    var_314 = wp::extract(var_86, var_296);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_316 = (var_159 == var_314);
                    var_315 = var_316;
                    if (!var_315) {
                        var_317 = wp::address(var_xquat_in, var_0, var_314);
                        var_319 = wp::load(var_317);
                        var_318 = (var_164 == var_319);
                        var_315 = var_315 || var_318;
                    }
                    if (var_315) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_321 = wp::add(var_149, var_320);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_322 = wp::address(var_efm_K_rowadr, var_321);
                        var_323 = wp::address(var_efm_K_rownnz, var_321);
                        var_325 = wp::load(var_322);
                        var_326 = wp::load(var_323);
                        var_324 = _find_csr_col_0(var_325, var_326, var_298, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_328 = (var_324 >= var_327);
                        if (var_328) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_329 = wp::add(var_324, var_320);
                            var_330 = wp::atomic_add(var_efm_K_val_out, var_0, var_329, var_308);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_332 = wp::add(var_149, var_331);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_333 = wp::address(var_efm_K_rowadr, var_332);
                        var_334 = wp::address(var_efm_K_rownnz, var_332);
                        var_336 = wp::load(var_333);
                        var_337 = wp::load(var_334);
                        var_335 = _find_csr_col_0(var_336, var_337, var_298, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_339 = (var_335 >= var_338);
                        if (var_339) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_340 = wp::add(var_335, var_331);
                            var_341 = wp::atomic_add(var_efm_K_val_out, var_0, var_340, var_308);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_343 = wp::add(var_149, var_342);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_344 = wp::address(var_efm_K_rowadr, var_343);
                        var_345 = wp::address(var_efm_K_rownnz, var_343);
                        var_347 = wp::load(var_344);
                        var_348 = wp::load(var_345);
                        var_346 = _find_csr_col_0(var_347, var_348, var_298, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_350 = (var_346 >= var_349);
                        if (var_350) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_351 = wp::add(var_346, var_342);
                            var_352 = wp::atomic_add(var_efm_K_val_out, var_0, var_351, var_308);
                        }
                    }
                    if (!var_315) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_353 = wp::address(var_xmat_in, var_0, var_314);
                        var_355 = wp::load(var_353);
                        var_354 = wp::mul(var_161, var_355);
                        var_356 = wp::mul(var_308, var_354);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_358 = wp::add(var_149, var_357);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_359 = wp::address(var_efm_K_rowadr, var_358);
                        var_360 = wp::address(var_efm_K_rownnz, var_358);
                        var_362 = wp::load(var_359);
                        var_363 = wp::load(var_360);
                        var_361 = _find_csr_col_0(var_362, var_363, var_298, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_365 = (var_361 >= var_364);
                        if (var_365) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_367 = wp::add(var_361, var_366);
                            var_368 = wp::extract(var_356, var_357, var_366);
                            var_369 = wp::atomic_add(var_efm_K_val_out, var_0, var_367, var_368);
                            var_371 = wp::add(var_361, var_370);
                            var_372 = wp::extract(var_356, var_357, var_370);
                            var_373 = wp::atomic_add(var_efm_K_val_out, var_0, var_371, var_372);
                            var_375 = wp::add(var_361, var_374);
                            var_376 = wp::extract(var_356, var_357, var_374);
                            var_377 = wp::atomic_add(var_efm_K_val_out, var_0, var_375, var_376);
                        }
                        var_378 = wp::where(var_365, var_374, var_291);
                        // row = dof_i + r                                                        <L 2229>
                        var_380 = wp::add(var_149, var_379);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_381 = wp::address(var_efm_K_rowadr, var_380);
                        var_382 = wp::address(var_efm_K_rownnz, var_380);
                        var_384 = wp::load(var_381);
                        var_385 = wp::load(var_382);
                        var_383 = _find_csr_col_0(var_384, var_385, var_298, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_387 = (var_383 >= var_386);
                        if (var_387) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_389 = wp::add(var_383, var_388);
                            var_390 = wp::extract(var_356, var_379, var_388);
                            var_391 = wp::atomic_add(var_efm_K_val_out, var_0, var_389, var_390);
                            var_393 = wp::add(var_383, var_392);
                            var_394 = wp::extract(var_356, var_379, var_392);
                            var_395 = wp::atomic_add(var_efm_K_val_out, var_0, var_393, var_394);
                            var_397 = wp::add(var_383, var_396);
                            var_398 = wp::extract(var_356, var_379, var_396);
                            var_399 = wp::atomic_add(var_efm_K_val_out, var_0, var_397, var_398);
                        }
                        var_400 = wp::where(var_387, var_396, var_378);
                        // row = dof_i + r                                                        <L 2229>
                        var_402 = wp::add(var_149, var_401);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_403 = wp::address(var_efm_K_rowadr, var_402);
                        var_404 = wp::address(var_efm_K_rownnz, var_402);
                        var_406 = wp::load(var_403);
                        var_407 = wp::load(var_404);
                        var_405 = _find_csr_col_0(var_406, var_407, var_298, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_409 = (var_405 >= var_408);
                        if (var_409) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_411 = wp::add(var_405, var_410);
                            var_412 = wp::extract(var_356, var_401, var_410);
                            var_413 = wp::atomic_add(var_efm_K_val_out, var_0, var_411, var_412);
                            var_415 = wp::add(var_405, var_414);
                            var_416 = wp::extract(var_356, var_401, var_414);
                            var_417 = wp::atomic_add(var_efm_K_val_out, var_0, var_415, var_416);
                            var_419 = wp::add(var_405, var_418);
                            var_420 = wp::extract(var_356, var_401, var_418);
                            var_421 = wp::atomic_add(var_efm_K_val_out, var_0, var_419, var_420);
                        }
                        var_422 = wp::where(var_409, var_418, var_400);
                    }
                    var_423 = wp::where(var_315, var_342, var_401);
                    var_424 = wp::where(var_315, var_343, var_402);
                    var_425 = wp::where(var_315, var_346, var_405);
                    var_426 = wp::where(var_315, var_226, var_356);
                    var_427 = wp::where(var_315, var_291, var_422);
                }
                var_428 = wp::where(var_313, var_314, var_184);
                var_429 = wp::where(var_313, var_423, var_292);
                var_430 = wp::where(var_313, var_424, var_293);
                var_431 = wp::where(var_313, var_425, var_294);
                var_432 = wp::where(var_313, var_426, var_226);
                var_433 = wp::where(var_313, var_427, var_291);
            }
            var_434 = wp::where(var_302, var_313, var_302);
            var_435 = wp::where(var_302, var_308, var_178);
            var_436 = wp::where(var_302, var_428, var_184);
            var_437 = wp::where(var_302, var_429, var_292);
            var_438 = wp::where(var_302, var_430, var_293);
            var_439 = wp::where(var_302, var_431, var_294);
            var_440 = wp::where(var_302, var_432, var_226);
            var_441 = wp::where(var_302, var_433, var_291);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_444 = wp::extract(var_81, var_442);
            // if dof_j < 0:                                                                      <L 2214>
            var_446 = (var_444 < var_445);
            if (var_446) {
                // continue                                                                       <L 2215>
            }
            var_448 = wp::where(var_446, var_447, var_443);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_448) {
                var_450 = wp::mul(var_449, var_147);
                var_451 = wp::add(var_76, var_450);
                var_452 = wp::add(var_451, var_442);
                var_453 = wp::address(var_flex_bending, var_452);
                var_455 = wp::load(var_453);
                var_454 = wp::mul(var_46, var_455);
                // if q == 0.0:                                                                   <L 2217>
                var_457 = (var_454 == var_456);
                if (var_457) {
                    // continue                                                                   <L 2218>
                }
                var_459 = wp::where(var_457, var_458, var_448);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_459) {
                    var_460 = wp::extract(var_86, var_442);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_462 = (var_159 == var_460);
                    var_461 = var_462;
                    if (!var_461) {
                        var_463 = wp::address(var_xquat_in, var_0, var_460);
                        var_465 = wp::load(var_463);
                        var_464 = (var_164 == var_465);
                        var_461 = var_461 || var_464;
                    }
                    if (var_461) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_467 = wp::add(var_149, var_466);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_468 = wp::address(var_efm_K_rowadr, var_467);
                        var_469 = wp::address(var_efm_K_rownnz, var_467);
                        var_471 = wp::load(var_468);
                        var_472 = wp::load(var_469);
                        var_470 = _find_csr_col_0(var_471, var_472, var_444, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_474 = (var_470 >= var_473);
                        if (var_474) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_475 = wp::add(var_470, var_466);
                            var_476 = wp::atomic_add(var_efm_K_val_out, var_0, var_475, var_454);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_478 = wp::add(var_149, var_477);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_479 = wp::address(var_efm_K_rowadr, var_478);
                        var_480 = wp::address(var_efm_K_rownnz, var_478);
                        var_482 = wp::load(var_479);
                        var_483 = wp::load(var_480);
                        var_481 = _find_csr_col_0(var_482, var_483, var_444, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_485 = (var_481 >= var_484);
                        if (var_485) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_486 = wp::add(var_481, var_477);
                            var_487 = wp::atomic_add(var_efm_K_val_out, var_0, var_486, var_454);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_489 = wp::add(var_149, var_488);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_490 = wp::address(var_efm_K_rowadr, var_489);
                        var_491 = wp::address(var_efm_K_rownnz, var_489);
                        var_493 = wp::load(var_490);
                        var_494 = wp::load(var_491);
                        var_492 = _find_csr_col_0(var_493, var_494, var_444, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_496 = (var_492 >= var_495);
                        if (var_496) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_497 = wp::add(var_492, var_488);
                            var_498 = wp::atomic_add(var_efm_K_val_out, var_0, var_497, var_454);
                        }
                    }
                    if (!var_461) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_499 = wp::address(var_xmat_in, var_0, var_460);
                        var_501 = wp::load(var_499);
                        var_500 = wp::mul(var_161, var_501);
                        var_502 = wp::mul(var_454, var_500);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_504 = wp::add(var_149, var_503);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_505 = wp::address(var_efm_K_rowadr, var_504);
                        var_506 = wp::address(var_efm_K_rownnz, var_504);
                        var_508 = wp::load(var_505);
                        var_509 = wp::load(var_506);
                        var_507 = _find_csr_col_0(var_508, var_509, var_444, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_511 = (var_507 >= var_510);
                        if (var_511) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_513 = wp::add(var_507, var_512);
                            var_514 = wp::extract(var_502, var_503, var_512);
                            var_515 = wp::atomic_add(var_efm_K_val_out, var_0, var_513, var_514);
                            var_517 = wp::add(var_507, var_516);
                            var_518 = wp::extract(var_502, var_503, var_516);
                            var_519 = wp::atomic_add(var_efm_K_val_out, var_0, var_517, var_518);
                            var_521 = wp::add(var_507, var_520);
                            var_522 = wp::extract(var_502, var_503, var_520);
                            var_523 = wp::atomic_add(var_efm_K_val_out, var_0, var_521, var_522);
                        }
                        var_524 = wp::where(var_511, var_520, var_441);
                        // row = dof_i + r                                                        <L 2229>
                        var_526 = wp::add(var_149, var_525);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_527 = wp::address(var_efm_K_rowadr, var_526);
                        var_528 = wp::address(var_efm_K_rownnz, var_526);
                        var_530 = wp::load(var_527);
                        var_531 = wp::load(var_528);
                        var_529 = _find_csr_col_0(var_530, var_531, var_444, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_533 = (var_529 >= var_532);
                        if (var_533) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_535 = wp::add(var_529, var_534);
                            var_536 = wp::extract(var_502, var_525, var_534);
                            var_537 = wp::atomic_add(var_efm_K_val_out, var_0, var_535, var_536);
                            var_539 = wp::add(var_529, var_538);
                            var_540 = wp::extract(var_502, var_525, var_538);
                            var_541 = wp::atomic_add(var_efm_K_val_out, var_0, var_539, var_540);
                            var_543 = wp::add(var_529, var_542);
                            var_544 = wp::extract(var_502, var_525, var_542);
                            var_545 = wp::atomic_add(var_efm_K_val_out, var_0, var_543, var_544);
                        }
                        var_546 = wp::where(var_533, var_542, var_524);
                        // row = dof_i + r                                                        <L 2229>
                        var_548 = wp::add(var_149, var_547);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_549 = wp::address(var_efm_K_rowadr, var_548);
                        var_550 = wp::address(var_efm_K_rownnz, var_548);
                        var_552 = wp::load(var_549);
                        var_553 = wp::load(var_550);
                        var_551 = _find_csr_col_0(var_552, var_553, var_444, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_555 = (var_551 >= var_554);
                        if (var_555) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_557 = wp::add(var_551, var_556);
                            var_558 = wp::extract(var_502, var_547, var_556);
                            var_559 = wp::atomic_add(var_efm_K_val_out, var_0, var_557, var_558);
                            var_561 = wp::add(var_551, var_560);
                            var_562 = wp::extract(var_502, var_547, var_560);
                            var_563 = wp::atomic_add(var_efm_K_val_out, var_0, var_561, var_562);
                            var_565 = wp::add(var_551, var_564);
                            var_566 = wp::extract(var_502, var_547, var_564);
                            var_567 = wp::atomic_add(var_efm_K_val_out, var_0, var_565, var_566);
                        }
                        var_568 = wp::where(var_555, var_564, var_546);
                    }
                    var_569 = wp::where(var_461, var_488, var_547);
                    var_570 = wp::where(var_461, var_489, var_548);
                    var_571 = wp::where(var_461, var_492, var_551);
                    var_572 = wp::where(var_461, var_440, var_502);
                    var_573 = wp::where(var_461, var_441, var_568);
                }
                var_574 = wp::where(var_459, var_460, var_436);
                var_575 = wp::where(var_459, var_569, var_437);
                var_576 = wp::where(var_459, var_570, var_438);
                var_577 = wp::where(var_459, var_571, var_439);
                var_578 = wp::where(var_459, var_572, var_440);
                var_579 = wp::where(var_459, var_573, var_441);
            }
            var_580 = wp::where(var_448, var_459, var_448);
            var_581 = wp::where(var_448, var_454, var_435);
            var_582 = wp::where(var_448, var_574, var_436);
            var_583 = wp::where(var_448, var_575, var_437);
            var_584 = wp::where(var_448, var_576, var_438);
            var_585 = wp::where(var_448, var_577, var_439);
            var_586 = wp::where(var_448, var_578, var_440);
            var_587 = wp::where(var_448, var_579, var_441);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_590 = wp::extract(var_81, var_588);
            // if dof_j < 0:                                                                      <L 2214>
            var_592 = (var_590 < var_591);
            if (var_592) {
                // continue                                                                       <L 2215>
            }
            var_594 = wp::where(var_592, var_593, var_589);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_594) {
                var_596 = wp::mul(var_595, var_147);
                var_597 = wp::add(var_76, var_596);
                var_598 = wp::add(var_597, var_588);
                var_599 = wp::address(var_flex_bending, var_598);
                var_601 = wp::load(var_599);
                var_600 = wp::mul(var_46, var_601);
                // if q == 0.0:                                                                   <L 2217>
                var_603 = (var_600 == var_602);
                if (var_603) {
                    // continue                                                                   <L 2218>
                }
                var_605 = wp::where(var_603, var_604, var_594);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_605) {
                    var_606 = wp::extract(var_86, var_588);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_608 = (var_159 == var_606);
                    var_607 = var_608;
                    if (!var_607) {
                        var_609 = wp::address(var_xquat_in, var_0, var_606);
                        var_611 = wp::load(var_609);
                        var_610 = (var_164 == var_611);
                        var_607 = var_607 || var_610;
                    }
                    if (var_607) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_613 = wp::add(var_149, var_612);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_614 = wp::address(var_efm_K_rowadr, var_613);
                        var_615 = wp::address(var_efm_K_rownnz, var_613);
                        var_617 = wp::load(var_614);
                        var_618 = wp::load(var_615);
                        var_616 = _find_csr_col_0(var_617, var_618, var_590, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_620 = (var_616 >= var_619);
                        if (var_620) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_621 = wp::add(var_616, var_612);
                            var_622 = wp::atomic_add(var_efm_K_val_out, var_0, var_621, var_600);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_624 = wp::add(var_149, var_623);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_625 = wp::address(var_efm_K_rowadr, var_624);
                        var_626 = wp::address(var_efm_K_rownnz, var_624);
                        var_628 = wp::load(var_625);
                        var_629 = wp::load(var_626);
                        var_627 = _find_csr_col_0(var_628, var_629, var_590, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_631 = (var_627 >= var_630);
                        if (var_631) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_632 = wp::add(var_627, var_623);
                            var_633 = wp::atomic_add(var_efm_K_val_out, var_0, var_632, var_600);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_635 = wp::add(var_149, var_634);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_636 = wp::address(var_efm_K_rowadr, var_635);
                        var_637 = wp::address(var_efm_K_rownnz, var_635);
                        var_639 = wp::load(var_636);
                        var_640 = wp::load(var_637);
                        var_638 = _find_csr_col_0(var_639, var_640, var_590, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_642 = (var_638 >= var_641);
                        if (var_642) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_643 = wp::add(var_638, var_634);
                            var_644 = wp::atomic_add(var_efm_K_val_out, var_0, var_643, var_600);
                        }
                    }
                    if (!var_607) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_645 = wp::address(var_xmat_in, var_0, var_606);
                        var_647 = wp::load(var_645);
                        var_646 = wp::mul(var_161, var_647);
                        var_648 = wp::mul(var_600, var_646);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_650 = wp::add(var_149, var_649);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_651 = wp::address(var_efm_K_rowadr, var_650);
                        var_652 = wp::address(var_efm_K_rownnz, var_650);
                        var_654 = wp::load(var_651);
                        var_655 = wp::load(var_652);
                        var_653 = _find_csr_col_0(var_654, var_655, var_590, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_657 = (var_653 >= var_656);
                        if (var_657) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_659 = wp::add(var_653, var_658);
                            var_660 = wp::extract(var_648, var_649, var_658);
                            var_661 = wp::atomic_add(var_efm_K_val_out, var_0, var_659, var_660);
                            var_663 = wp::add(var_653, var_662);
                            var_664 = wp::extract(var_648, var_649, var_662);
                            var_665 = wp::atomic_add(var_efm_K_val_out, var_0, var_663, var_664);
                            var_667 = wp::add(var_653, var_666);
                            var_668 = wp::extract(var_648, var_649, var_666);
                            var_669 = wp::atomic_add(var_efm_K_val_out, var_0, var_667, var_668);
                        }
                        var_670 = wp::where(var_657, var_666, var_587);
                        // row = dof_i + r                                                        <L 2229>
                        var_672 = wp::add(var_149, var_671);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_673 = wp::address(var_efm_K_rowadr, var_672);
                        var_674 = wp::address(var_efm_K_rownnz, var_672);
                        var_676 = wp::load(var_673);
                        var_677 = wp::load(var_674);
                        var_675 = _find_csr_col_0(var_676, var_677, var_590, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_679 = (var_675 >= var_678);
                        if (var_679) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_681 = wp::add(var_675, var_680);
                            var_682 = wp::extract(var_648, var_671, var_680);
                            var_683 = wp::atomic_add(var_efm_K_val_out, var_0, var_681, var_682);
                            var_685 = wp::add(var_675, var_684);
                            var_686 = wp::extract(var_648, var_671, var_684);
                            var_687 = wp::atomic_add(var_efm_K_val_out, var_0, var_685, var_686);
                            var_689 = wp::add(var_675, var_688);
                            var_690 = wp::extract(var_648, var_671, var_688);
                            var_691 = wp::atomic_add(var_efm_K_val_out, var_0, var_689, var_690);
                        }
                        var_692 = wp::where(var_679, var_688, var_670);
                        // row = dof_i + r                                                        <L 2229>
                        var_694 = wp::add(var_149, var_693);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_695 = wp::address(var_efm_K_rowadr, var_694);
                        var_696 = wp::address(var_efm_K_rownnz, var_694);
                        var_698 = wp::load(var_695);
                        var_699 = wp::load(var_696);
                        var_697 = _find_csr_col_0(var_698, var_699, var_590, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_701 = (var_697 >= var_700);
                        if (var_701) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_703 = wp::add(var_697, var_702);
                            var_704 = wp::extract(var_648, var_693, var_702);
                            var_705 = wp::atomic_add(var_efm_K_val_out, var_0, var_703, var_704);
                            var_707 = wp::add(var_697, var_706);
                            var_708 = wp::extract(var_648, var_693, var_706);
                            var_709 = wp::atomic_add(var_efm_K_val_out, var_0, var_707, var_708);
                            var_711 = wp::add(var_697, var_710);
                            var_712 = wp::extract(var_648, var_693, var_710);
                            var_713 = wp::atomic_add(var_efm_K_val_out, var_0, var_711, var_712);
                        }
                        var_714 = wp::where(var_701, var_710, var_692);
                    }
                    var_715 = wp::where(var_607, var_634, var_693);
                    var_716 = wp::where(var_607, var_635, var_694);
                    var_717 = wp::where(var_607, var_638, var_697);
                    var_718 = wp::where(var_607, var_586, var_648);
                    var_719 = wp::where(var_607, var_587, var_714);
                }
                var_720 = wp::where(var_605, var_606, var_582);
                var_721 = wp::where(var_605, var_715, var_583);
                var_722 = wp::where(var_605, var_716, var_584);
                var_723 = wp::where(var_605, var_717, var_585);
                var_724 = wp::where(var_605, var_718, var_586);
                var_725 = wp::where(var_605, var_719, var_587);
            }
            var_726 = wp::where(var_594, var_605, var_594);
            var_727 = wp::where(var_594, var_600, var_581);
            var_728 = wp::where(var_594, var_720, var_582);
            var_729 = wp::where(var_594, var_721, var_583);
            var_730 = wp::where(var_594, var_722, var_584);
            var_731 = wp::where(var_594, var_723, var_585);
            var_732 = wp::where(var_594, var_724, var_586);
            var_733 = wp::where(var_594, var_725, var_587);
        }
        // dof_i = dofs[i]                                                                        <L 2206>
        var_736 = wp::extract(var_81, var_734);
        // if dof_i < 0 or efm_K_rownnz[dof_i] == 0:                                              <L 2207>
        var_739 = (var_736 < var_738);
        var_737 = var_739;
        if (!var_737) {
            var_740 = wp::address(var_efm_K_rownnz, var_736);
            var_743 = wp::load(var_740);
            var_742 = (var_743 == var_741);
            var_737 = var_737 || var_742;
        }
        if (var_737) {
            // continue                                                                           <L 2208>
        }
        var_745 = wp::where(var_737, var_744, var_735);
        // bi = bodies[i]                                                                         <L 2209>
        if (var_745) {
            var_746 = wp::extract(var_86, var_734);
            // R_bi_T = wp.transpose(xmat_in[worldid, bi])                                        <L 2210>
            var_747 = wp::address(var_xmat_in, var_0, var_746);
            var_749 = wp::load(var_747);
            var_748 = wp::transpose(var_749);
            // qi = xquat_in[worldid, bi]                                                         <L 2211>
            var_750 = wp::address(var_xquat_in, var_0, var_746);
            var_752 = wp::load(var_750);
            var_751 = wp::copy(var_752);
            // for j in range(4):                                                                 <L 2212>
            // dof_j = dofs[j]                                                                    <L 2213>
            var_755 = wp::extract(var_81, var_753);
            // if dof_j < 0:                                                                      <L 2214>
            var_757 = (var_755 < var_756);
            if (var_757) {
                // continue                                                                       <L 2215>
            }
            var_759 = wp::where(var_757, var_758, var_754);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_759) {
                var_761 = wp::mul(var_760, var_734);
                var_762 = wp::add(var_76, var_761);
                var_763 = wp::add(var_762, var_753);
                var_764 = wp::address(var_flex_bending, var_763);
                var_766 = wp::load(var_764);
                var_765 = wp::mul(var_46, var_766);
                // if q == 0.0:                                                                   <L 2217>
                var_768 = (var_765 == var_767);
                if (var_768) {
                    // continue                                                                   <L 2218>
                }
                var_770 = wp::where(var_768, var_769, var_759);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_770) {
                    var_771 = wp::extract(var_86, var_753);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_773 = (var_746 == var_771);
                    var_772 = var_773;
                    if (!var_772) {
                        var_774 = wp::address(var_xquat_in, var_0, var_771);
                        var_776 = wp::load(var_774);
                        var_775 = (var_751 == var_776);
                        var_772 = var_772 || var_775;
                    }
                    if (var_772) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_778 = wp::add(var_736, var_777);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_779 = wp::address(var_efm_K_rowadr, var_778);
                        var_780 = wp::address(var_efm_K_rownnz, var_778);
                        var_782 = wp::load(var_779);
                        var_783 = wp::load(var_780);
                        var_781 = _find_csr_col_0(var_782, var_783, var_755, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_785 = (var_781 >= var_784);
                        if (var_785) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_786 = wp::add(var_781, var_777);
                            var_787 = wp::atomic_add(var_efm_K_val_out, var_0, var_786, var_765);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_789 = wp::add(var_736, var_788);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_790 = wp::address(var_efm_K_rowadr, var_789);
                        var_791 = wp::address(var_efm_K_rownnz, var_789);
                        var_793 = wp::load(var_790);
                        var_794 = wp::load(var_791);
                        var_792 = _find_csr_col_0(var_793, var_794, var_755, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_796 = (var_792 >= var_795);
                        if (var_796) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_797 = wp::add(var_792, var_788);
                            var_798 = wp::atomic_add(var_efm_K_val_out, var_0, var_797, var_765);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_800 = wp::add(var_736, var_799);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_801 = wp::address(var_efm_K_rowadr, var_800);
                        var_802 = wp::address(var_efm_K_rownnz, var_800);
                        var_804 = wp::load(var_801);
                        var_805 = wp::load(var_802);
                        var_803 = _find_csr_col_0(var_804, var_805, var_755, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_807 = (var_803 >= var_806);
                        if (var_807) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_808 = wp::add(var_803, var_799);
                            var_809 = wp::atomic_add(var_efm_K_val_out, var_0, var_808, var_765);
                        }
                    }
                    if (!var_772) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_810 = wp::address(var_xmat_in, var_0, var_771);
                        var_812 = wp::load(var_810);
                        var_811 = wp::mul(var_748, var_812);
                        var_813 = wp::mul(var_765, var_811);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_815 = wp::add(var_736, var_814);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_816 = wp::address(var_efm_K_rowadr, var_815);
                        var_817 = wp::address(var_efm_K_rownnz, var_815);
                        var_819 = wp::load(var_816);
                        var_820 = wp::load(var_817);
                        var_818 = _find_csr_col_0(var_819, var_820, var_755, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_822 = (var_818 >= var_821);
                        if (var_822) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_824 = wp::add(var_818, var_823);
                            var_825 = wp::extract(var_813, var_814, var_823);
                            var_826 = wp::atomic_add(var_efm_K_val_out, var_0, var_824, var_825);
                            var_828 = wp::add(var_818, var_827);
                            var_829 = wp::extract(var_813, var_814, var_827);
                            var_830 = wp::atomic_add(var_efm_K_val_out, var_0, var_828, var_829);
                            var_832 = wp::add(var_818, var_831);
                            var_833 = wp::extract(var_813, var_814, var_831);
                            var_834 = wp::atomic_add(var_efm_K_val_out, var_0, var_832, var_833);
                        }
                        var_835 = wp::where(var_822, var_831, var_733);
                        // row = dof_i + r                                                        <L 2229>
                        var_837 = wp::add(var_736, var_836);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_838 = wp::address(var_efm_K_rowadr, var_837);
                        var_839 = wp::address(var_efm_K_rownnz, var_837);
                        var_841 = wp::load(var_838);
                        var_842 = wp::load(var_839);
                        var_840 = _find_csr_col_0(var_841, var_842, var_755, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_844 = (var_840 >= var_843);
                        if (var_844) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_846 = wp::add(var_840, var_845);
                            var_847 = wp::extract(var_813, var_836, var_845);
                            var_848 = wp::atomic_add(var_efm_K_val_out, var_0, var_846, var_847);
                            var_850 = wp::add(var_840, var_849);
                            var_851 = wp::extract(var_813, var_836, var_849);
                            var_852 = wp::atomic_add(var_efm_K_val_out, var_0, var_850, var_851);
                            var_854 = wp::add(var_840, var_853);
                            var_855 = wp::extract(var_813, var_836, var_853);
                            var_856 = wp::atomic_add(var_efm_K_val_out, var_0, var_854, var_855);
                        }
                        var_857 = wp::where(var_844, var_853, var_835);
                        // row = dof_i + r                                                        <L 2229>
                        var_859 = wp::add(var_736, var_858);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_860 = wp::address(var_efm_K_rowadr, var_859);
                        var_861 = wp::address(var_efm_K_rownnz, var_859);
                        var_863 = wp::load(var_860);
                        var_864 = wp::load(var_861);
                        var_862 = _find_csr_col_0(var_863, var_864, var_755, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_866 = (var_862 >= var_865);
                        if (var_866) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_868 = wp::add(var_862, var_867);
                            var_869 = wp::extract(var_813, var_858, var_867);
                            var_870 = wp::atomic_add(var_efm_K_val_out, var_0, var_868, var_869);
                            var_872 = wp::add(var_862, var_871);
                            var_873 = wp::extract(var_813, var_858, var_871);
                            var_874 = wp::atomic_add(var_efm_K_val_out, var_0, var_872, var_873);
                            var_876 = wp::add(var_862, var_875);
                            var_877 = wp::extract(var_813, var_858, var_875);
                            var_878 = wp::atomic_add(var_efm_K_val_out, var_0, var_876, var_877);
                        }
                        var_879 = wp::where(var_866, var_875, var_857);
                    }
                    var_880 = wp::where(var_772, var_799, var_858);
                    var_881 = wp::where(var_772, var_800, var_859);
                    var_882 = wp::where(var_772, var_803, var_862);
                    var_883 = wp::where(var_772, var_732, var_813);
                    var_884 = wp::where(var_772, var_733, var_879);
                }
                var_885 = wp::where(var_770, var_771, var_728);
                var_886 = wp::where(var_770, var_880, var_729);
                var_887 = wp::where(var_770, var_881, var_730);
                var_888 = wp::where(var_770, var_882, var_731);
                var_889 = wp::where(var_770, var_883, var_732);
                var_890 = wp::where(var_770, var_884, var_733);
            }
            var_891 = wp::where(var_759, var_765, var_727);
            var_892 = wp::where(var_759, var_885, var_728);
            var_893 = wp::where(var_759, var_886, var_729);
            var_894 = wp::where(var_759, var_887, var_730);
            var_895 = wp::where(var_759, var_888, var_731);
            var_896 = wp::where(var_759, var_889, var_732);
            var_897 = wp::where(var_759, var_890, var_733);
            var_898 = wp::where(var_759, var_770, var_759);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_901 = wp::extract(var_81, var_899);
            // if dof_j < 0:                                                                      <L 2214>
            var_903 = (var_901 < var_902);
            if (var_903) {
                // continue                                                                       <L 2215>
            }
            var_905 = wp::where(var_903, var_904, var_900);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_905) {
                var_907 = wp::mul(var_906, var_734);
                var_908 = wp::add(var_76, var_907);
                var_909 = wp::add(var_908, var_899);
                var_910 = wp::address(var_flex_bending, var_909);
                var_912 = wp::load(var_910);
                var_911 = wp::mul(var_46, var_912);
                // if q == 0.0:                                                                   <L 2217>
                var_914 = (var_911 == var_913);
                if (var_914) {
                    // continue                                                                   <L 2218>
                }
                var_916 = wp::where(var_914, var_915, var_905);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_916) {
                    var_917 = wp::extract(var_86, var_899);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_919 = (var_746 == var_917);
                    var_918 = var_919;
                    if (!var_918) {
                        var_920 = wp::address(var_xquat_in, var_0, var_917);
                        var_922 = wp::load(var_920);
                        var_921 = (var_751 == var_922);
                        var_918 = var_918 || var_921;
                    }
                    if (var_918) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_924 = wp::add(var_736, var_923);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_925 = wp::address(var_efm_K_rowadr, var_924);
                        var_926 = wp::address(var_efm_K_rownnz, var_924);
                        var_928 = wp::load(var_925);
                        var_929 = wp::load(var_926);
                        var_927 = _find_csr_col_0(var_928, var_929, var_901, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_931 = (var_927 >= var_930);
                        if (var_931) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_932 = wp::add(var_927, var_923);
                            var_933 = wp::atomic_add(var_efm_K_val_out, var_0, var_932, var_911);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_935 = wp::add(var_736, var_934);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_936 = wp::address(var_efm_K_rowadr, var_935);
                        var_937 = wp::address(var_efm_K_rownnz, var_935);
                        var_939 = wp::load(var_936);
                        var_940 = wp::load(var_937);
                        var_938 = _find_csr_col_0(var_939, var_940, var_901, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_942 = (var_938 >= var_941);
                        if (var_942) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_943 = wp::add(var_938, var_934);
                            var_944 = wp::atomic_add(var_efm_K_val_out, var_0, var_943, var_911);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_946 = wp::add(var_736, var_945);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_947 = wp::address(var_efm_K_rowadr, var_946);
                        var_948 = wp::address(var_efm_K_rownnz, var_946);
                        var_950 = wp::load(var_947);
                        var_951 = wp::load(var_948);
                        var_949 = _find_csr_col_0(var_950, var_951, var_901, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_953 = (var_949 >= var_952);
                        if (var_953) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_954 = wp::add(var_949, var_945);
                            var_955 = wp::atomic_add(var_efm_K_val_out, var_0, var_954, var_911);
                        }
                    }
                    if (!var_918) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_956 = wp::address(var_xmat_in, var_0, var_917);
                        var_958 = wp::load(var_956);
                        var_957 = wp::mul(var_748, var_958);
                        var_959 = wp::mul(var_911, var_957);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_961 = wp::add(var_736, var_960);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_962 = wp::address(var_efm_K_rowadr, var_961);
                        var_963 = wp::address(var_efm_K_rownnz, var_961);
                        var_965 = wp::load(var_962);
                        var_966 = wp::load(var_963);
                        var_964 = _find_csr_col_0(var_965, var_966, var_901, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_968 = (var_964 >= var_967);
                        if (var_968) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_970 = wp::add(var_964, var_969);
                            var_971 = wp::extract(var_959, var_960, var_969);
                            var_972 = wp::atomic_add(var_efm_K_val_out, var_0, var_970, var_971);
                            var_974 = wp::add(var_964, var_973);
                            var_975 = wp::extract(var_959, var_960, var_973);
                            var_976 = wp::atomic_add(var_efm_K_val_out, var_0, var_974, var_975);
                            var_978 = wp::add(var_964, var_977);
                            var_979 = wp::extract(var_959, var_960, var_977);
                            var_980 = wp::atomic_add(var_efm_K_val_out, var_0, var_978, var_979);
                        }
                        var_981 = wp::where(var_968, var_977, var_897);
                        // row = dof_i + r                                                        <L 2229>
                        var_983 = wp::add(var_736, var_982);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_984 = wp::address(var_efm_K_rowadr, var_983);
                        var_985 = wp::address(var_efm_K_rownnz, var_983);
                        var_987 = wp::load(var_984);
                        var_988 = wp::load(var_985);
                        var_986 = _find_csr_col_0(var_987, var_988, var_901, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_990 = (var_986 >= var_989);
                        if (var_990) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_992 = wp::add(var_986, var_991);
                            var_993 = wp::extract(var_959, var_982, var_991);
                            var_994 = wp::atomic_add(var_efm_K_val_out, var_0, var_992, var_993);
                            var_996 = wp::add(var_986, var_995);
                            var_997 = wp::extract(var_959, var_982, var_995);
                            var_998 = wp::atomic_add(var_efm_K_val_out, var_0, var_996, var_997);
                            var_1000 = wp::add(var_986, var_999);
                            var_1001 = wp::extract(var_959, var_982, var_999);
                            var_1002 = wp::atomic_add(var_efm_K_val_out, var_0, var_1000, var_1001);
                        }
                        var_1003 = wp::where(var_990, var_999, var_981);
                        // row = dof_i + r                                                        <L 2229>
                        var_1005 = wp::add(var_736, var_1004);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1006 = wp::address(var_efm_K_rowadr, var_1005);
                        var_1007 = wp::address(var_efm_K_rownnz, var_1005);
                        var_1009 = wp::load(var_1006);
                        var_1010 = wp::load(var_1007);
                        var_1008 = _find_csr_col_0(var_1009, var_1010, var_901, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1012 = (var_1008 >= var_1011);
                        if (var_1012) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1014 = wp::add(var_1008, var_1013);
                            var_1015 = wp::extract(var_959, var_1004, var_1013);
                            var_1016 = wp::atomic_add(var_efm_K_val_out, var_0, var_1014, var_1015);
                            var_1018 = wp::add(var_1008, var_1017);
                            var_1019 = wp::extract(var_959, var_1004, var_1017);
                            var_1020 = wp::atomic_add(var_efm_K_val_out, var_0, var_1018, var_1019);
                            var_1022 = wp::add(var_1008, var_1021);
                            var_1023 = wp::extract(var_959, var_1004, var_1021);
                            var_1024 = wp::atomic_add(var_efm_K_val_out, var_0, var_1022, var_1023);
                        }
                        var_1025 = wp::where(var_1012, var_1021, var_1003);
                    }
                    var_1026 = wp::where(var_918, var_945, var_1004);
                    var_1027 = wp::where(var_918, var_946, var_1005);
                    var_1028 = wp::where(var_918, var_949, var_1008);
                    var_1029 = wp::where(var_918, var_896, var_959);
                    var_1030 = wp::where(var_918, var_897, var_1025);
                }
                var_1031 = wp::where(var_916, var_917, var_892);
                var_1032 = wp::where(var_916, var_1026, var_893);
                var_1033 = wp::where(var_916, var_1027, var_894);
                var_1034 = wp::where(var_916, var_1028, var_895);
                var_1035 = wp::where(var_916, var_1029, var_896);
                var_1036 = wp::where(var_916, var_1030, var_897);
            }
            var_1037 = wp::where(var_905, var_911, var_891);
            var_1038 = wp::where(var_905, var_1031, var_892);
            var_1039 = wp::where(var_905, var_1032, var_893);
            var_1040 = wp::where(var_905, var_1033, var_894);
            var_1041 = wp::where(var_905, var_1034, var_895);
            var_1042 = wp::where(var_905, var_1035, var_896);
            var_1043 = wp::where(var_905, var_1036, var_897);
            var_1044 = wp::where(var_905, var_916, var_905);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_1047 = wp::extract(var_81, var_1045);
            // if dof_j < 0:                                                                      <L 2214>
            var_1049 = (var_1047 < var_1048);
            if (var_1049) {
                // continue                                                                       <L 2215>
            }
            var_1051 = wp::where(var_1049, var_1050, var_1046);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_1051) {
                var_1053 = wp::mul(var_1052, var_734);
                var_1054 = wp::add(var_76, var_1053);
                var_1055 = wp::add(var_1054, var_1045);
                var_1056 = wp::address(var_flex_bending, var_1055);
                var_1058 = wp::load(var_1056);
                var_1057 = wp::mul(var_46, var_1058);
                // if q == 0.0:                                                                   <L 2217>
                var_1060 = (var_1057 == var_1059);
                if (var_1060) {
                    // continue                                                                   <L 2218>
                }
                var_1062 = wp::where(var_1060, var_1061, var_1051);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_1062) {
                    var_1063 = wp::extract(var_86, var_1045);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_1065 = (var_746 == var_1063);
                    var_1064 = var_1065;
                    if (!var_1064) {
                        var_1066 = wp::address(var_xquat_in, var_0, var_1063);
                        var_1068 = wp::load(var_1066);
                        var_1067 = (var_751 == var_1068);
                        var_1064 = var_1064 || var_1067;
                    }
                    if (var_1064) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_1070 = wp::add(var_736, var_1069);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1071 = wp::address(var_efm_K_rowadr, var_1070);
                        var_1072 = wp::address(var_efm_K_rownnz, var_1070);
                        var_1074 = wp::load(var_1071);
                        var_1075 = wp::load(var_1072);
                        var_1073 = _find_csr_col_0(var_1074, var_1075, var_1047, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1077 = (var_1073 >= var_1076);
                        if (var_1077) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1078 = wp::add(var_1073, var_1069);
                            var_1079 = wp::atomic_add(var_efm_K_val_out, var_0, var_1078, var_1057);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1081 = wp::add(var_736, var_1080);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1082 = wp::address(var_efm_K_rowadr, var_1081);
                        var_1083 = wp::address(var_efm_K_rownnz, var_1081);
                        var_1085 = wp::load(var_1082);
                        var_1086 = wp::load(var_1083);
                        var_1084 = _find_csr_col_0(var_1085, var_1086, var_1047, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1088 = (var_1084 >= var_1087);
                        if (var_1088) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1089 = wp::add(var_1084, var_1080);
                            var_1090 = wp::atomic_add(var_efm_K_val_out, var_0, var_1089, var_1057);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1092 = wp::add(var_736, var_1091);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1093 = wp::address(var_efm_K_rowadr, var_1092);
                        var_1094 = wp::address(var_efm_K_rownnz, var_1092);
                        var_1096 = wp::load(var_1093);
                        var_1097 = wp::load(var_1094);
                        var_1095 = _find_csr_col_0(var_1096, var_1097, var_1047, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1099 = (var_1095 >= var_1098);
                        if (var_1099) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1100 = wp::add(var_1095, var_1091);
                            var_1101 = wp::atomic_add(var_efm_K_val_out, var_0, var_1100, var_1057);
                        }
                    }
                    if (!var_1064) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_1102 = wp::address(var_xmat_in, var_0, var_1063);
                        var_1104 = wp::load(var_1102);
                        var_1103 = wp::mul(var_748, var_1104);
                        var_1105 = wp::mul(var_1057, var_1103);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_1107 = wp::add(var_736, var_1106);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1108 = wp::address(var_efm_K_rowadr, var_1107);
                        var_1109 = wp::address(var_efm_K_rownnz, var_1107);
                        var_1111 = wp::load(var_1108);
                        var_1112 = wp::load(var_1109);
                        var_1110 = _find_csr_col_0(var_1111, var_1112, var_1047, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1114 = (var_1110 >= var_1113);
                        if (var_1114) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1116 = wp::add(var_1110, var_1115);
                            var_1117 = wp::extract(var_1105, var_1106, var_1115);
                            var_1118 = wp::atomic_add(var_efm_K_val_out, var_0, var_1116, var_1117);
                            var_1120 = wp::add(var_1110, var_1119);
                            var_1121 = wp::extract(var_1105, var_1106, var_1119);
                            var_1122 = wp::atomic_add(var_efm_K_val_out, var_0, var_1120, var_1121);
                            var_1124 = wp::add(var_1110, var_1123);
                            var_1125 = wp::extract(var_1105, var_1106, var_1123);
                            var_1126 = wp::atomic_add(var_efm_K_val_out, var_0, var_1124, var_1125);
                        }
                        var_1127 = wp::where(var_1114, var_1123, var_1043);
                        // row = dof_i + r                                                        <L 2229>
                        var_1129 = wp::add(var_736, var_1128);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1130 = wp::address(var_efm_K_rowadr, var_1129);
                        var_1131 = wp::address(var_efm_K_rownnz, var_1129);
                        var_1133 = wp::load(var_1130);
                        var_1134 = wp::load(var_1131);
                        var_1132 = _find_csr_col_0(var_1133, var_1134, var_1047, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1136 = (var_1132 >= var_1135);
                        if (var_1136) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1138 = wp::add(var_1132, var_1137);
                            var_1139 = wp::extract(var_1105, var_1128, var_1137);
                            var_1140 = wp::atomic_add(var_efm_K_val_out, var_0, var_1138, var_1139);
                            var_1142 = wp::add(var_1132, var_1141);
                            var_1143 = wp::extract(var_1105, var_1128, var_1141);
                            var_1144 = wp::atomic_add(var_efm_K_val_out, var_0, var_1142, var_1143);
                            var_1146 = wp::add(var_1132, var_1145);
                            var_1147 = wp::extract(var_1105, var_1128, var_1145);
                            var_1148 = wp::atomic_add(var_efm_K_val_out, var_0, var_1146, var_1147);
                        }
                        var_1149 = wp::where(var_1136, var_1145, var_1127);
                        // row = dof_i + r                                                        <L 2229>
                        var_1151 = wp::add(var_736, var_1150);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1152 = wp::address(var_efm_K_rowadr, var_1151);
                        var_1153 = wp::address(var_efm_K_rownnz, var_1151);
                        var_1155 = wp::load(var_1152);
                        var_1156 = wp::load(var_1153);
                        var_1154 = _find_csr_col_0(var_1155, var_1156, var_1047, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1158 = (var_1154 >= var_1157);
                        if (var_1158) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1160 = wp::add(var_1154, var_1159);
                            var_1161 = wp::extract(var_1105, var_1150, var_1159);
                            var_1162 = wp::atomic_add(var_efm_K_val_out, var_0, var_1160, var_1161);
                            var_1164 = wp::add(var_1154, var_1163);
                            var_1165 = wp::extract(var_1105, var_1150, var_1163);
                            var_1166 = wp::atomic_add(var_efm_K_val_out, var_0, var_1164, var_1165);
                            var_1168 = wp::add(var_1154, var_1167);
                            var_1169 = wp::extract(var_1105, var_1150, var_1167);
                            var_1170 = wp::atomic_add(var_efm_K_val_out, var_0, var_1168, var_1169);
                        }
                        var_1171 = wp::where(var_1158, var_1167, var_1149);
                    }
                    var_1172 = wp::where(var_1064, var_1091, var_1150);
                    var_1173 = wp::where(var_1064, var_1092, var_1151);
                    var_1174 = wp::where(var_1064, var_1095, var_1154);
                    var_1175 = wp::where(var_1064, var_1042, var_1105);
                    var_1176 = wp::where(var_1064, var_1043, var_1171);
                }
                var_1177 = wp::where(var_1062, var_1063, var_1038);
                var_1178 = wp::where(var_1062, var_1172, var_1039);
                var_1179 = wp::where(var_1062, var_1173, var_1040);
                var_1180 = wp::where(var_1062, var_1174, var_1041);
                var_1181 = wp::where(var_1062, var_1175, var_1042);
                var_1182 = wp::where(var_1062, var_1176, var_1043);
            }
            var_1183 = wp::where(var_1051, var_1057, var_1037);
            var_1184 = wp::where(var_1051, var_1177, var_1038);
            var_1185 = wp::where(var_1051, var_1178, var_1039);
            var_1186 = wp::where(var_1051, var_1179, var_1040);
            var_1187 = wp::where(var_1051, var_1180, var_1041);
            var_1188 = wp::where(var_1051, var_1181, var_1042);
            var_1189 = wp::where(var_1051, var_1182, var_1043);
            var_1190 = wp::where(var_1051, var_1062, var_1051);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_1193 = wp::extract(var_81, var_1191);
            // if dof_j < 0:                                                                      <L 2214>
            var_1195 = (var_1193 < var_1194);
            if (var_1195) {
                // continue                                                                       <L 2215>
            }
            var_1197 = wp::where(var_1195, var_1196, var_1192);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_1197) {
                var_1199 = wp::mul(var_1198, var_734);
                var_1200 = wp::add(var_76, var_1199);
                var_1201 = wp::add(var_1200, var_1191);
                var_1202 = wp::address(var_flex_bending, var_1201);
                var_1204 = wp::load(var_1202);
                var_1203 = wp::mul(var_46, var_1204);
                // if q == 0.0:                                                                   <L 2217>
                var_1206 = (var_1203 == var_1205);
                if (var_1206) {
                    // continue                                                                   <L 2218>
                }
                var_1208 = wp::where(var_1206, var_1207, var_1197);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_1208) {
                    var_1209 = wp::extract(var_86, var_1191);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_1211 = (var_746 == var_1209);
                    var_1210 = var_1211;
                    if (!var_1210) {
                        var_1212 = wp::address(var_xquat_in, var_0, var_1209);
                        var_1214 = wp::load(var_1212);
                        var_1213 = (var_751 == var_1214);
                        var_1210 = var_1210 || var_1213;
                    }
                    if (var_1210) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_1216 = wp::add(var_736, var_1215);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1217 = wp::address(var_efm_K_rowadr, var_1216);
                        var_1218 = wp::address(var_efm_K_rownnz, var_1216);
                        var_1220 = wp::load(var_1217);
                        var_1221 = wp::load(var_1218);
                        var_1219 = _find_csr_col_0(var_1220, var_1221, var_1193, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1223 = (var_1219 >= var_1222);
                        if (var_1223) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1224 = wp::add(var_1219, var_1215);
                            var_1225 = wp::atomic_add(var_efm_K_val_out, var_0, var_1224, var_1203);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1227 = wp::add(var_736, var_1226);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1228 = wp::address(var_efm_K_rowadr, var_1227);
                        var_1229 = wp::address(var_efm_K_rownnz, var_1227);
                        var_1231 = wp::load(var_1228);
                        var_1232 = wp::load(var_1229);
                        var_1230 = _find_csr_col_0(var_1231, var_1232, var_1193, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1234 = (var_1230 >= var_1233);
                        if (var_1234) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1235 = wp::add(var_1230, var_1226);
                            var_1236 = wp::atomic_add(var_efm_K_val_out, var_0, var_1235, var_1203);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1238 = wp::add(var_736, var_1237);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1239 = wp::address(var_efm_K_rowadr, var_1238);
                        var_1240 = wp::address(var_efm_K_rownnz, var_1238);
                        var_1242 = wp::load(var_1239);
                        var_1243 = wp::load(var_1240);
                        var_1241 = _find_csr_col_0(var_1242, var_1243, var_1193, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1245 = (var_1241 >= var_1244);
                        if (var_1245) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1246 = wp::add(var_1241, var_1237);
                            var_1247 = wp::atomic_add(var_efm_K_val_out, var_0, var_1246, var_1203);
                        }
                    }
                    if (!var_1210) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_1248 = wp::address(var_xmat_in, var_0, var_1209);
                        var_1250 = wp::load(var_1248);
                        var_1249 = wp::mul(var_748, var_1250);
                        var_1251 = wp::mul(var_1203, var_1249);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_1253 = wp::add(var_736, var_1252);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1254 = wp::address(var_efm_K_rowadr, var_1253);
                        var_1255 = wp::address(var_efm_K_rownnz, var_1253);
                        var_1257 = wp::load(var_1254);
                        var_1258 = wp::load(var_1255);
                        var_1256 = _find_csr_col_0(var_1257, var_1258, var_1193, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1260 = (var_1256 >= var_1259);
                        if (var_1260) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1262 = wp::add(var_1256, var_1261);
                            var_1263 = wp::extract(var_1251, var_1252, var_1261);
                            var_1264 = wp::atomic_add(var_efm_K_val_out, var_0, var_1262, var_1263);
                            var_1266 = wp::add(var_1256, var_1265);
                            var_1267 = wp::extract(var_1251, var_1252, var_1265);
                            var_1268 = wp::atomic_add(var_efm_K_val_out, var_0, var_1266, var_1267);
                            var_1270 = wp::add(var_1256, var_1269);
                            var_1271 = wp::extract(var_1251, var_1252, var_1269);
                            var_1272 = wp::atomic_add(var_efm_K_val_out, var_0, var_1270, var_1271);
                        }
                        var_1273 = wp::where(var_1260, var_1269, var_1189);
                        // row = dof_i + r                                                        <L 2229>
                        var_1275 = wp::add(var_736, var_1274);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1276 = wp::address(var_efm_K_rowadr, var_1275);
                        var_1277 = wp::address(var_efm_K_rownnz, var_1275);
                        var_1279 = wp::load(var_1276);
                        var_1280 = wp::load(var_1277);
                        var_1278 = _find_csr_col_0(var_1279, var_1280, var_1193, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1282 = (var_1278 >= var_1281);
                        if (var_1282) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1284 = wp::add(var_1278, var_1283);
                            var_1285 = wp::extract(var_1251, var_1274, var_1283);
                            var_1286 = wp::atomic_add(var_efm_K_val_out, var_0, var_1284, var_1285);
                            var_1288 = wp::add(var_1278, var_1287);
                            var_1289 = wp::extract(var_1251, var_1274, var_1287);
                            var_1290 = wp::atomic_add(var_efm_K_val_out, var_0, var_1288, var_1289);
                            var_1292 = wp::add(var_1278, var_1291);
                            var_1293 = wp::extract(var_1251, var_1274, var_1291);
                            var_1294 = wp::atomic_add(var_efm_K_val_out, var_0, var_1292, var_1293);
                        }
                        var_1295 = wp::where(var_1282, var_1291, var_1273);
                        // row = dof_i + r                                                        <L 2229>
                        var_1297 = wp::add(var_736, var_1296);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1298 = wp::address(var_efm_K_rowadr, var_1297);
                        var_1299 = wp::address(var_efm_K_rownnz, var_1297);
                        var_1301 = wp::load(var_1298);
                        var_1302 = wp::load(var_1299);
                        var_1300 = _find_csr_col_0(var_1301, var_1302, var_1193, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1304 = (var_1300 >= var_1303);
                        if (var_1304) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1306 = wp::add(var_1300, var_1305);
                            var_1307 = wp::extract(var_1251, var_1296, var_1305);
                            var_1308 = wp::atomic_add(var_efm_K_val_out, var_0, var_1306, var_1307);
                            var_1310 = wp::add(var_1300, var_1309);
                            var_1311 = wp::extract(var_1251, var_1296, var_1309);
                            var_1312 = wp::atomic_add(var_efm_K_val_out, var_0, var_1310, var_1311);
                            var_1314 = wp::add(var_1300, var_1313);
                            var_1315 = wp::extract(var_1251, var_1296, var_1313);
                            var_1316 = wp::atomic_add(var_efm_K_val_out, var_0, var_1314, var_1315);
                        }
                        var_1317 = wp::where(var_1304, var_1313, var_1295);
                    }
                    var_1318 = wp::where(var_1210, var_1237, var_1296);
                    var_1319 = wp::where(var_1210, var_1238, var_1297);
                    var_1320 = wp::where(var_1210, var_1241, var_1300);
                    var_1321 = wp::where(var_1210, var_1188, var_1251);
                    var_1322 = wp::where(var_1210, var_1189, var_1317);
                }
                var_1323 = wp::where(var_1208, var_1209, var_1184);
                var_1324 = wp::where(var_1208, var_1318, var_1185);
                var_1325 = wp::where(var_1208, var_1319, var_1186);
                var_1326 = wp::where(var_1208, var_1320, var_1187);
                var_1327 = wp::where(var_1208, var_1321, var_1188);
                var_1328 = wp::where(var_1208, var_1322, var_1189);
            }
            var_1329 = wp::where(var_1197, var_1203, var_1183);
            var_1330 = wp::where(var_1197, var_1323, var_1184);
            var_1331 = wp::where(var_1197, var_1324, var_1185);
            var_1332 = wp::where(var_1197, var_1325, var_1186);
            var_1333 = wp::where(var_1197, var_1326, var_1187);
            var_1334 = wp::where(var_1197, var_1327, var_1188);
            var_1335 = wp::where(var_1197, var_1328, var_1189);
            var_1336 = wp::where(var_1197, var_1208, var_1197);
        }
        var_1337 = wp::where(var_745, var_746, var_159);
        var_1338 = wp::where(var_745, var_748, var_161);
        var_1339 = wp::where(var_745, var_751, var_164);
        var_1340 = wp::where(var_745, var_1191, var_588);
        var_1341 = wp::where(var_745, var_1193, var_590);
        var_1342 = wp::where(var_745, var_1329, var_727);
        var_1343 = wp::where(var_745, var_1330, var_728);
        var_1344 = wp::where(var_745, var_1331, var_729);
        var_1345 = wp::where(var_745, var_1332, var_730);
        var_1346 = wp::where(var_745, var_1333, var_731);
        var_1347 = wp::where(var_745, var_1334, var_732);
        var_1348 = wp::where(var_745, var_1335, var_733);
        // dof_i = dofs[i]                                                                        <L 2206>
        var_1351 = wp::extract(var_81, var_1349);
        // if dof_i < 0 or efm_K_rownnz[dof_i] == 0:                                              <L 2207>
        var_1354 = (var_1351 < var_1353);
        var_1352 = var_1354;
        if (!var_1352) {
            var_1355 = wp::address(var_efm_K_rownnz, var_1351);
            var_1358 = wp::load(var_1355);
            var_1357 = (var_1358 == var_1356);
            var_1352 = var_1352 || var_1357;
        }
        if (var_1352) {
            // continue                                                                           <L 2208>
        }
        var_1360 = wp::where(var_1352, var_1359, var_1350);
        // bi = bodies[i]                                                                         <L 2209>
        if (var_1360) {
            var_1361 = wp::extract(var_86, var_1349);
            // R_bi_T = wp.transpose(xmat_in[worldid, bi])                                        <L 2210>
            var_1362 = wp::address(var_xmat_in, var_0, var_1361);
            var_1364 = wp::load(var_1362);
            var_1363 = wp::transpose(var_1364);
            // qi = xquat_in[worldid, bi]                                                         <L 2211>
            var_1365 = wp::address(var_xquat_in, var_0, var_1361);
            var_1367 = wp::load(var_1365);
            var_1366 = wp::copy(var_1367);
            // for j in range(4):                                                                 <L 2212>
            // dof_j = dofs[j]                                                                    <L 2213>
            var_1370 = wp::extract(var_81, var_1368);
            // if dof_j < 0:                                                                      <L 2214>
            var_1372 = (var_1370 < var_1371);
            if (var_1372) {
                // continue                                                                       <L 2215>
            }
            var_1374 = wp::where(var_1372, var_1373, var_1369);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_1374) {
                var_1376 = wp::mul(var_1375, var_1349);
                var_1377 = wp::add(var_76, var_1376);
                var_1378 = wp::add(var_1377, var_1368);
                var_1379 = wp::address(var_flex_bending, var_1378);
                var_1381 = wp::load(var_1379);
                var_1380 = wp::mul(var_46, var_1381);
                // if q == 0.0:                                                                   <L 2217>
                var_1383 = (var_1380 == var_1382);
                if (var_1383) {
                    // continue                                                                   <L 2218>
                }
                var_1385 = wp::where(var_1383, var_1384, var_1374);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_1385) {
                    var_1386 = wp::extract(var_86, var_1368);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_1388 = (var_1361 == var_1386);
                    var_1387 = var_1388;
                    if (!var_1387) {
                        var_1389 = wp::address(var_xquat_in, var_0, var_1386);
                        var_1391 = wp::load(var_1389);
                        var_1390 = (var_1366 == var_1391);
                        var_1387 = var_1387 || var_1390;
                    }
                    if (var_1387) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_1393 = wp::add(var_1351, var_1392);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1394 = wp::address(var_efm_K_rowadr, var_1393);
                        var_1395 = wp::address(var_efm_K_rownnz, var_1393);
                        var_1397 = wp::load(var_1394);
                        var_1398 = wp::load(var_1395);
                        var_1396 = _find_csr_col_0(var_1397, var_1398, var_1370, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1400 = (var_1396 >= var_1399);
                        if (var_1400) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1401 = wp::add(var_1396, var_1392);
                            var_1402 = wp::atomic_add(var_efm_K_val_out, var_0, var_1401, var_1380);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1404 = wp::add(var_1351, var_1403);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1405 = wp::address(var_efm_K_rowadr, var_1404);
                        var_1406 = wp::address(var_efm_K_rownnz, var_1404);
                        var_1408 = wp::load(var_1405);
                        var_1409 = wp::load(var_1406);
                        var_1407 = _find_csr_col_0(var_1408, var_1409, var_1370, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1411 = (var_1407 >= var_1410);
                        if (var_1411) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1412 = wp::add(var_1407, var_1403);
                            var_1413 = wp::atomic_add(var_efm_K_val_out, var_0, var_1412, var_1380);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1415 = wp::add(var_1351, var_1414);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1416 = wp::address(var_efm_K_rowadr, var_1415);
                        var_1417 = wp::address(var_efm_K_rownnz, var_1415);
                        var_1419 = wp::load(var_1416);
                        var_1420 = wp::load(var_1417);
                        var_1418 = _find_csr_col_0(var_1419, var_1420, var_1370, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1422 = (var_1418 >= var_1421);
                        if (var_1422) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1423 = wp::add(var_1418, var_1414);
                            var_1424 = wp::atomic_add(var_efm_K_val_out, var_0, var_1423, var_1380);
                        }
                    }
                    if (!var_1387) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_1425 = wp::address(var_xmat_in, var_0, var_1386);
                        var_1427 = wp::load(var_1425);
                        var_1426 = wp::mul(var_1363, var_1427);
                        var_1428 = wp::mul(var_1380, var_1426);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_1430 = wp::add(var_1351, var_1429);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1431 = wp::address(var_efm_K_rowadr, var_1430);
                        var_1432 = wp::address(var_efm_K_rownnz, var_1430);
                        var_1434 = wp::load(var_1431);
                        var_1435 = wp::load(var_1432);
                        var_1433 = _find_csr_col_0(var_1434, var_1435, var_1370, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1437 = (var_1433 >= var_1436);
                        if (var_1437) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1439 = wp::add(var_1433, var_1438);
                            var_1440 = wp::extract(var_1428, var_1429, var_1438);
                            var_1441 = wp::atomic_add(var_efm_K_val_out, var_0, var_1439, var_1440);
                            var_1443 = wp::add(var_1433, var_1442);
                            var_1444 = wp::extract(var_1428, var_1429, var_1442);
                            var_1445 = wp::atomic_add(var_efm_K_val_out, var_0, var_1443, var_1444);
                            var_1447 = wp::add(var_1433, var_1446);
                            var_1448 = wp::extract(var_1428, var_1429, var_1446);
                            var_1449 = wp::atomic_add(var_efm_K_val_out, var_0, var_1447, var_1448);
                        }
                        var_1450 = wp::where(var_1437, var_1446, var_1348);
                        // row = dof_i + r                                                        <L 2229>
                        var_1452 = wp::add(var_1351, var_1451);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1453 = wp::address(var_efm_K_rowadr, var_1452);
                        var_1454 = wp::address(var_efm_K_rownnz, var_1452);
                        var_1456 = wp::load(var_1453);
                        var_1457 = wp::load(var_1454);
                        var_1455 = _find_csr_col_0(var_1456, var_1457, var_1370, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1459 = (var_1455 >= var_1458);
                        if (var_1459) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1461 = wp::add(var_1455, var_1460);
                            var_1462 = wp::extract(var_1428, var_1451, var_1460);
                            var_1463 = wp::atomic_add(var_efm_K_val_out, var_0, var_1461, var_1462);
                            var_1465 = wp::add(var_1455, var_1464);
                            var_1466 = wp::extract(var_1428, var_1451, var_1464);
                            var_1467 = wp::atomic_add(var_efm_K_val_out, var_0, var_1465, var_1466);
                            var_1469 = wp::add(var_1455, var_1468);
                            var_1470 = wp::extract(var_1428, var_1451, var_1468);
                            var_1471 = wp::atomic_add(var_efm_K_val_out, var_0, var_1469, var_1470);
                        }
                        var_1472 = wp::where(var_1459, var_1468, var_1450);
                        // row = dof_i + r                                                        <L 2229>
                        var_1474 = wp::add(var_1351, var_1473);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1475 = wp::address(var_efm_K_rowadr, var_1474);
                        var_1476 = wp::address(var_efm_K_rownnz, var_1474);
                        var_1478 = wp::load(var_1475);
                        var_1479 = wp::load(var_1476);
                        var_1477 = _find_csr_col_0(var_1478, var_1479, var_1370, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1481 = (var_1477 >= var_1480);
                        if (var_1481) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1483 = wp::add(var_1477, var_1482);
                            var_1484 = wp::extract(var_1428, var_1473, var_1482);
                            var_1485 = wp::atomic_add(var_efm_K_val_out, var_0, var_1483, var_1484);
                            var_1487 = wp::add(var_1477, var_1486);
                            var_1488 = wp::extract(var_1428, var_1473, var_1486);
                            var_1489 = wp::atomic_add(var_efm_K_val_out, var_0, var_1487, var_1488);
                            var_1491 = wp::add(var_1477, var_1490);
                            var_1492 = wp::extract(var_1428, var_1473, var_1490);
                            var_1493 = wp::atomic_add(var_efm_K_val_out, var_0, var_1491, var_1492);
                        }
                        var_1494 = wp::where(var_1481, var_1490, var_1472);
                    }
                    var_1495 = wp::where(var_1387, var_1414, var_1473);
                    var_1496 = wp::where(var_1387, var_1415, var_1474);
                    var_1497 = wp::where(var_1387, var_1418, var_1477);
                    var_1498 = wp::where(var_1387, var_1347, var_1428);
                    var_1499 = wp::where(var_1387, var_1348, var_1494);
                }
                var_1500 = wp::where(var_1385, var_1386, var_1343);
                var_1501 = wp::where(var_1385, var_1495, var_1344);
                var_1502 = wp::where(var_1385, var_1496, var_1345);
                var_1503 = wp::where(var_1385, var_1497, var_1346);
                var_1504 = wp::where(var_1385, var_1498, var_1347);
                var_1505 = wp::where(var_1385, var_1499, var_1348);
            }
            var_1506 = wp::where(var_1374, var_1380, var_1342);
            var_1507 = wp::where(var_1374, var_1500, var_1343);
            var_1508 = wp::where(var_1374, var_1501, var_1344);
            var_1509 = wp::where(var_1374, var_1502, var_1345);
            var_1510 = wp::where(var_1374, var_1503, var_1346);
            var_1511 = wp::where(var_1374, var_1504, var_1347);
            var_1512 = wp::where(var_1374, var_1505, var_1348);
            var_1513 = wp::where(var_1374, var_1385, var_1374);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_1516 = wp::extract(var_81, var_1514);
            // if dof_j < 0:                                                                      <L 2214>
            var_1518 = (var_1516 < var_1517);
            if (var_1518) {
                // continue                                                                       <L 2215>
            }
            var_1520 = wp::where(var_1518, var_1519, var_1515);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_1520) {
                var_1522 = wp::mul(var_1521, var_1349);
                var_1523 = wp::add(var_76, var_1522);
                var_1524 = wp::add(var_1523, var_1514);
                var_1525 = wp::address(var_flex_bending, var_1524);
                var_1527 = wp::load(var_1525);
                var_1526 = wp::mul(var_46, var_1527);
                // if q == 0.0:                                                                   <L 2217>
                var_1529 = (var_1526 == var_1528);
                if (var_1529) {
                    // continue                                                                   <L 2218>
                }
                var_1531 = wp::where(var_1529, var_1530, var_1520);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_1531) {
                    var_1532 = wp::extract(var_86, var_1514);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_1534 = (var_1361 == var_1532);
                    var_1533 = var_1534;
                    if (!var_1533) {
                        var_1535 = wp::address(var_xquat_in, var_0, var_1532);
                        var_1537 = wp::load(var_1535);
                        var_1536 = (var_1366 == var_1537);
                        var_1533 = var_1533 || var_1536;
                    }
                    if (var_1533) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_1539 = wp::add(var_1351, var_1538);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1540 = wp::address(var_efm_K_rowadr, var_1539);
                        var_1541 = wp::address(var_efm_K_rownnz, var_1539);
                        var_1543 = wp::load(var_1540);
                        var_1544 = wp::load(var_1541);
                        var_1542 = _find_csr_col_0(var_1543, var_1544, var_1516, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1546 = (var_1542 >= var_1545);
                        if (var_1546) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1547 = wp::add(var_1542, var_1538);
                            var_1548 = wp::atomic_add(var_efm_K_val_out, var_0, var_1547, var_1526);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1550 = wp::add(var_1351, var_1549);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1551 = wp::address(var_efm_K_rowadr, var_1550);
                        var_1552 = wp::address(var_efm_K_rownnz, var_1550);
                        var_1554 = wp::load(var_1551);
                        var_1555 = wp::load(var_1552);
                        var_1553 = _find_csr_col_0(var_1554, var_1555, var_1516, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1557 = (var_1553 >= var_1556);
                        if (var_1557) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1558 = wp::add(var_1553, var_1549);
                            var_1559 = wp::atomic_add(var_efm_K_val_out, var_0, var_1558, var_1526);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1561 = wp::add(var_1351, var_1560);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1562 = wp::address(var_efm_K_rowadr, var_1561);
                        var_1563 = wp::address(var_efm_K_rownnz, var_1561);
                        var_1565 = wp::load(var_1562);
                        var_1566 = wp::load(var_1563);
                        var_1564 = _find_csr_col_0(var_1565, var_1566, var_1516, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1568 = (var_1564 >= var_1567);
                        if (var_1568) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1569 = wp::add(var_1564, var_1560);
                            var_1570 = wp::atomic_add(var_efm_K_val_out, var_0, var_1569, var_1526);
                        }
                    }
                    if (!var_1533) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_1571 = wp::address(var_xmat_in, var_0, var_1532);
                        var_1573 = wp::load(var_1571);
                        var_1572 = wp::mul(var_1363, var_1573);
                        var_1574 = wp::mul(var_1526, var_1572);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_1576 = wp::add(var_1351, var_1575);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1577 = wp::address(var_efm_K_rowadr, var_1576);
                        var_1578 = wp::address(var_efm_K_rownnz, var_1576);
                        var_1580 = wp::load(var_1577);
                        var_1581 = wp::load(var_1578);
                        var_1579 = _find_csr_col_0(var_1580, var_1581, var_1516, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1583 = (var_1579 >= var_1582);
                        if (var_1583) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1585 = wp::add(var_1579, var_1584);
                            var_1586 = wp::extract(var_1574, var_1575, var_1584);
                            var_1587 = wp::atomic_add(var_efm_K_val_out, var_0, var_1585, var_1586);
                            var_1589 = wp::add(var_1579, var_1588);
                            var_1590 = wp::extract(var_1574, var_1575, var_1588);
                            var_1591 = wp::atomic_add(var_efm_K_val_out, var_0, var_1589, var_1590);
                            var_1593 = wp::add(var_1579, var_1592);
                            var_1594 = wp::extract(var_1574, var_1575, var_1592);
                            var_1595 = wp::atomic_add(var_efm_K_val_out, var_0, var_1593, var_1594);
                        }
                        var_1596 = wp::where(var_1583, var_1592, var_1512);
                        // row = dof_i + r                                                        <L 2229>
                        var_1598 = wp::add(var_1351, var_1597);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1599 = wp::address(var_efm_K_rowadr, var_1598);
                        var_1600 = wp::address(var_efm_K_rownnz, var_1598);
                        var_1602 = wp::load(var_1599);
                        var_1603 = wp::load(var_1600);
                        var_1601 = _find_csr_col_0(var_1602, var_1603, var_1516, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1605 = (var_1601 >= var_1604);
                        if (var_1605) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1607 = wp::add(var_1601, var_1606);
                            var_1608 = wp::extract(var_1574, var_1597, var_1606);
                            var_1609 = wp::atomic_add(var_efm_K_val_out, var_0, var_1607, var_1608);
                            var_1611 = wp::add(var_1601, var_1610);
                            var_1612 = wp::extract(var_1574, var_1597, var_1610);
                            var_1613 = wp::atomic_add(var_efm_K_val_out, var_0, var_1611, var_1612);
                            var_1615 = wp::add(var_1601, var_1614);
                            var_1616 = wp::extract(var_1574, var_1597, var_1614);
                            var_1617 = wp::atomic_add(var_efm_K_val_out, var_0, var_1615, var_1616);
                        }
                        var_1618 = wp::where(var_1605, var_1614, var_1596);
                        // row = dof_i + r                                                        <L 2229>
                        var_1620 = wp::add(var_1351, var_1619);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1621 = wp::address(var_efm_K_rowadr, var_1620);
                        var_1622 = wp::address(var_efm_K_rownnz, var_1620);
                        var_1624 = wp::load(var_1621);
                        var_1625 = wp::load(var_1622);
                        var_1623 = _find_csr_col_0(var_1624, var_1625, var_1516, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1627 = (var_1623 >= var_1626);
                        if (var_1627) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1629 = wp::add(var_1623, var_1628);
                            var_1630 = wp::extract(var_1574, var_1619, var_1628);
                            var_1631 = wp::atomic_add(var_efm_K_val_out, var_0, var_1629, var_1630);
                            var_1633 = wp::add(var_1623, var_1632);
                            var_1634 = wp::extract(var_1574, var_1619, var_1632);
                            var_1635 = wp::atomic_add(var_efm_K_val_out, var_0, var_1633, var_1634);
                            var_1637 = wp::add(var_1623, var_1636);
                            var_1638 = wp::extract(var_1574, var_1619, var_1636);
                            var_1639 = wp::atomic_add(var_efm_K_val_out, var_0, var_1637, var_1638);
                        }
                        var_1640 = wp::where(var_1627, var_1636, var_1618);
                    }
                    var_1641 = wp::where(var_1533, var_1560, var_1619);
                    var_1642 = wp::where(var_1533, var_1561, var_1620);
                    var_1643 = wp::where(var_1533, var_1564, var_1623);
                    var_1644 = wp::where(var_1533, var_1511, var_1574);
                    var_1645 = wp::where(var_1533, var_1512, var_1640);
                }
                var_1646 = wp::where(var_1531, var_1532, var_1507);
                var_1647 = wp::where(var_1531, var_1641, var_1508);
                var_1648 = wp::where(var_1531, var_1642, var_1509);
                var_1649 = wp::where(var_1531, var_1643, var_1510);
                var_1650 = wp::where(var_1531, var_1644, var_1511);
                var_1651 = wp::where(var_1531, var_1645, var_1512);
            }
            var_1652 = wp::where(var_1520, var_1526, var_1506);
            var_1653 = wp::where(var_1520, var_1646, var_1507);
            var_1654 = wp::where(var_1520, var_1647, var_1508);
            var_1655 = wp::where(var_1520, var_1648, var_1509);
            var_1656 = wp::where(var_1520, var_1649, var_1510);
            var_1657 = wp::where(var_1520, var_1650, var_1511);
            var_1658 = wp::where(var_1520, var_1651, var_1512);
            var_1659 = wp::where(var_1520, var_1531, var_1520);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_1662 = wp::extract(var_81, var_1660);
            // if dof_j < 0:                                                                      <L 2214>
            var_1664 = (var_1662 < var_1663);
            if (var_1664) {
                // continue                                                                       <L 2215>
            }
            var_1666 = wp::where(var_1664, var_1665, var_1661);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_1666) {
                var_1668 = wp::mul(var_1667, var_1349);
                var_1669 = wp::add(var_76, var_1668);
                var_1670 = wp::add(var_1669, var_1660);
                var_1671 = wp::address(var_flex_bending, var_1670);
                var_1673 = wp::load(var_1671);
                var_1672 = wp::mul(var_46, var_1673);
                // if q == 0.0:                                                                   <L 2217>
                var_1675 = (var_1672 == var_1674);
                if (var_1675) {
                    // continue                                                                   <L 2218>
                }
                var_1677 = wp::where(var_1675, var_1676, var_1666);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_1677) {
                    var_1678 = wp::extract(var_86, var_1660);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_1680 = (var_1361 == var_1678);
                    var_1679 = var_1680;
                    if (!var_1679) {
                        var_1681 = wp::address(var_xquat_in, var_0, var_1678);
                        var_1683 = wp::load(var_1681);
                        var_1682 = (var_1366 == var_1683);
                        var_1679 = var_1679 || var_1682;
                    }
                    if (var_1679) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_1685 = wp::add(var_1351, var_1684);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1686 = wp::address(var_efm_K_rowadr, var_1685);
                        var_1687 = wp::address(var_efm_K_rownnz, var_1685);
                        var_1689 = wp::load(var_1686);
                        var_1690 = wp::load(var_1687);
                        var_1688 = _find_csr_col_0(var_1689, var_1690, var_1662, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1692 = (var_1688 >= var_1691);
                        if (var_1692) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1693 = wp::add(var_1688, var_1684);
                            var_1694 = wp::atomic_add(var_efm_K_val_out, var_0, var_1693, var_1672);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1696 = wp::add(var_1351, var_1695);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1697 = wp::address(var_efm_K_rowadr, var_1696);
                        var_1698 = wp::address(var_efm_K_rownnz, var_1696);
                        var_1700 = wp::load(var_1697);
                        var_1701 = wp::load(var_1698);
                        var_1699 = _find_csr_col_0(var_1700, var_1701, var_1662, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1703 = (var_1699 >= var_1702);
                        if (var_1703) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1704 = wp::add(var_1699, var_1695);
                            var_1705 = wp::atomic_add(var_efm_K_val_out, var_0, var_1704, var_1672);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1707 = wp::add(var_1351, var_1706);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1708 = wp::address(var_efm_K_rowadr, var_1707);
                        var_1709 = wp::address(var_efm_K_rownnz, var_1707);
                        var_1711 = wp::load(var_1708);
                        var_1712 = wp::load(var_1709);
                        var_1710 = _find_csr_col_0(var_1711, var_1712, var_1662, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1714 = (var_1710 >= var_1713);
                        if (var_1714) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1715 = wp::add(var_1710, var_1706);
                            var_1716 = wp::atomic_add(var_efm_K_val_out, var_0, var_1715, var_1672);
                        }
                    }
                    if (!var_1679) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_1717 = wp::address(var_xmat_in, var_0, var_1678);
                        var_1719 = wp::load(var_1717);
                        var_1718 = wp::mul(var_1363, var_1719);
                        var_1720 = wp::mul(var_1672, var_1718);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_1722 = wp::add(var_1351, var_1721);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1723 = wp::address(var_efm_K_rowadr, var_1722);
                        var_1724 = wp::address(var_efm_K_rownnz, var_1722);
                        var_1726 = wp::load(var_1723);
                        var_1727 = wp::load(var_1724);
                        var_1725 = _find_csr_col_0(var_1726, var_1727, var_1662, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1729 = (var_1725 >= var_1728);
                        if (var_1729) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1731 = wp::add(var_1725, var_1730);
                            var_1732 = wp::extract(var_1720, var_1721, var_1730);
                            var_1733 = wp::atomic_add(var_efm_K_val_out, var_0, var_1731, var_1732);
                            var_1735 = wp::add(var_1725, var_1734);
                            var_1736 = wp::extract(var_1720, var_1721, var_1734);
                            var_1737 = wp::atomic_add(var_efm_K_val_out, var_0, var_1735, var_1736);
                            var_1739 = wp::add(var_1725, var_1738);
                            var_1740 = wp::extract(var_1720, var_1721, var_1738);
                            var_1741 = wp::atomic_add(var_efm_K_val_out, var_0, var_1739, var_1740);
                        }
                        var_1742 = wp::where(var_1729, var_1738, var_1658);
                        // row = dof_i + r                                                        <L 2229>
                        var_1744 = wp::add(var_1351, var_1743);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1745 = wp::address(var_efm_K_rowadr, var_1744);
                        var_1746 = wp::address(var_efm_K_rownnz, var_1744);
                        var_1748 = wp::load(var_1745);
                        var_1749 = wp::load(var_1746);
                        var_1747 = _find_csr_col_0(var_1748, var_1749, var_1662, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1751 = (var_1747 >= var_1750);
                        if (var_1751) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1753 = wp::add(var_1747, var_1752);
                            var_1754 = wp::extract(var_1720, var_1743, var_1752);
                            var_1755 = wp::atomic_add(var_efm_K_val_out, var_0, var_1753, var_1754);
                            var_1757 = wp::add(var_1747, var_1756);
                            var_1758 = wp::extract(var_1720, var_1743, var_1756);
                            var_1759 = wp::atomic_add(var_efm_K_val_out, var_0, var_1757, var_1758);
                            var_1761 = wp::add(var_1747, var_1760);
                            var_1762 = wp::extract(var_1720, var_1743, var_1760);
                            var_1763 = wp::atomic_add(var_efm_K_val_out, var_0, var_1761, var_1762);
                        }
                        var_1764 = wp::where(var_1751, var_1760, var_1742);
                        // row = dof_i + r                                                        <L 2229>
                        var_1766 = wp::add(var_1351, var_1765);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1767 = wp::address(var_efm_K_rowadr, var_1766);
                        var_1768 = wp::address(var_efm_K_rownnz, var_1766);
                        var_1770 = wp::load(var_1767);
                        var_1771 = wp::load(var_1768);
                        var_1769 = _find_csr_col_0(var_1770, var_1771, var_1662, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1773 = (var_1769 >= var_1772);
                        if (var_1773) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1775 = wp::add(var_1769, var_1774);
                            var_1776 = wp::extract(var_1720, var_1765, var_1774);
                            var_1777 = wp::atomic_add(var_efm_K_val_out, var_0, var_1775, var_1776);
                            var_1779 = wp::add(var_1769, var_1778);
                            var_1780 = wp::extract(var_1720, var_1765, var_1778);
                            var_1781 = wp::atomic_add(var_efm_K_val_out, var_0, var_1779, var_1780);
                            var_1783 = wp::add(var_1769, var_1782);
                            var_1784 = wp::extract(var_1720, var_1765, var_1782);
                            var_1785 = wp::atomic_add(var_efm_K_val_out, var_0, var_1783, var_1784);
                        }
                        var_1786 = wp::where(var_1773, var_1782, var_1764);
                    }
                    var_1787 = wp::where(var_1679, var_1706, var_1765);
                    var_1788 = wp::where(var_1679, var_1707, var_1766);
                    var_1789 = wp::where(var_1679, var_1710, var_1769);
                    var_1790 = wp::where(var_1679, var_1657, var_1720);
                    var_1791 = wp::where(var_1679, var_1658, var_1786);
                }
                var_1792 = wp::where(var_1677, var_1678, var_1653);
                var_1793 = wp::where(var_1677, var_1787, var_1654);
                var_1794 = wp::where(var_1677, var_1788, var_1655);
                var_1795 = wp::where(var_1677, var_1789, var_1656);
                var_1796 = wp::where(var_1677, var_1790, var_1657);
                var_1797 = wp::where(var_1677, var_1791, var_1658);
            }
            var_1798 = wp::where(var_1666, var_1672, var_1652);
            var_1799 = wp::where(var_1666, var_1792, var_1653);
            var_1800 = wp::where(var_1666, var_1793, var_1654);
            var_1801 = wp::where(var_1666, var_1794, var_1655);
            var_1802 = wp::where(var_1666, var_1795, var_1656);
            var_1803 = wp::where(var_1666, var_1796, var_1657);
            var_1804 = wp::where(var_1666, var_1797, var_1658);
            var_1805 = wp::where(var_1666, var_1677, var_1666);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_1808 = wp::extract(var_81, var_1806);
            // if dof_j < 0:                                                                      <L 2214>
            var_1810 = (var_1808 < var_1809);
            if (var_1810) {
                // continue                                                                       <L 2215>
            }
            var_1812 = wp::where(var_1810, var_1811, var_1807);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_1812) {
                var_1814 = wp::mul(var_1813, var_1349);
                var_1815 = wp::add(var_76, var_1814);
                var_1816 = wp::add(var_1815, var_1806);
                var_1817 = wp::address(var_flex_bending, var_1816);
                var_1819 = wp::load(var_1817);
                var_1818 = wp::mul(var_46, var_1819);
                // if q == 0.0:                                                                   <L 2217>
                var_1821 = (var_1818 == var_1820);
                if (var_1821) {
                    // continue                                                                   <L 2218>
                }
                var_1823 = wp::where(var_1821, var_1822, var_1812);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_1823) {
                    var_1824 = wp::extract(var_86, var_1806);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_1826 = (var_1361 == var_1824);
                    var_1825 = var_1826;
                    if (!var_1825) {
                        var_1827 = wp::address(var_xquat_in, var_0, var_1824);
                        var_1829 = wp::load(var_1827);
                        var_1828 = (var_1366 == var_1829);
                        var_1825 = var_1825 || var_1828;
                    }
                    if (var_1825) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_1831 = wp::add(var_1351, var_1830);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1832 = wp::address(var_efm_K_rowadr, var_1831);
                        var_1833 = wp::address(var_efm_K_rownnz, var_1831);
                        var_1835 = wp::load(var_1832);
                        var_1836 = wp::load(var_1833);
                        var_1834 = _find_csr_col_0(var_1835, var_1836, var_1808, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1838 = (var_1834 >= var_1837);
                        if (var_1838) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1839 = wp::add(var_1834, var_1830);
                            var_1840 = wp::atomic_add(var_efm_K_val_out, var_0, var_1839, var_1818);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1842 = wp::add(var_1351, var_1841);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1843 = wp::address(var_efm_K_rowadr, var_1842);
                        var_1844 = wp::address(var_efm_K_rownnz, var_1842);
                        var_1846 = wp::load(var_1843);
                        var_1847 = wp::load(var_1844);
                        var_1845 = _find_csr_col_0(var_1846, var_1847, var_1808, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1849 = (var_1845 >= var_1848);
                        if (var_1849) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1850 = wp::add(var_1845, var_1841);
                            var_1851 = wp::atomic_add(var_efm_K_val_out, var_0, var_1850, var_1818);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_1853 = wp::add(var_1351, var_1852);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_1854 = wp::address(var_efm_K_rowadr, var_1853);
                        var_1855 = wp::address(var_efm_K_rownnz, var_1853);
                        var_1857 = wp::load(var_1854);
                        var_1858 = wp::load(var_1855);
                        var_1856 = _find_csr_col_0(var_1857, var_1858, var_1808, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_1860 = (var_1856 >= var_1859);
                        if (var_1860) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_1861 = wp::add(var_1856, var_1852);
                            var_1862 = wp::atomic_add(var_efm_K_val_out, var_0, var_1861, var_1818);
                        }
                    }
                    if (!var_1825) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_1863 = wp::address(var_xmat_in, var_0, var_1824);
                        var_1865 = wp::load(var_1863);
                        var_1864 = wp::mul(var_1363, var_1865);
                        var_1866 = wp::mul(var_1818, var_1864);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_1868 = wp::add(var_1351, var_1867);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1869 = wp::address(var_efm_K_rowadr, var_1868);
                        var_1870 = wp::address(var_efm_K_rownnz, var_1868);
                        var_1872 = wp::load(var_1869);
                        var_1873 = wp::load(var_1870);
                        var_1871 = _find_csr_col_0(var_1872, var_1873, var_1808, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1875 = (var_1871 >= var_1874);
                        if (var_1875) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1877 = wp::add(var_1871, var_1876);
                            var_1878 = wp::extract(var_1866, var_1867, var_1876);
                            var_1879 = wp::atomic_add(var_efm_K_val_out, var_0, var_1877, var_1878);
                            var_1881 = wp::add(var_1871, var_1880);
                            var_1882 = wp::extract(var_1866, var_1867, var_1880);
                            var_1883 = wp::atomic_add(var_efm_K_val_out, var_0, var_1881, var_1882);
                            var_1885 = wp::add(var_1871, var_1884);
                            var_1886 = wp::extract(var_1866, var_1867, var_1884);
                            var_1887 = wp::atomic_add(var_efm_K_val_out, var_0, var_1885, var_1886);
                        }
                        var_1888 = wp::where(var_1875, var_1884, var_1804);
                        // row = dof_i + r                                                        <L 2229>
                        var_1890 = wp::add(var_1351, var_1889);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1891 = wp::address(var_efm_K_rowadr, var_1890);
                        var_1892 = wp::address(var_efm_K_rownnz, var_1890);
                        var_1894 = wp::load(var_1891);
                        var_1895 = wp::load(var_1892);
                        var_1893 = _find_csr_col_0(var_1894, var_1895, var_1808, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1897 = (var_1893 >= var_1896);
                        if (var_1897) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1899 = wp::add(var_1893, var_1898);
                            var_1900 = wp::extract(var_1866, var_1889, var_1898);
                            var_1901 = wp::atomic_add(var_efm_K_val_out, var_0, var_1899, var_1900);
                            var_1903 = wp::add(var_1893, var_1902);
                            var_1904 = wp::extract(var_1866, var_1889, var_1902);
                            var_1905 = wp::atomic_add(var_efm_K_val_out, var_0, var_1903, var_1904);
                            var_1907 = wp::add(var_1893, var_1906);
                            var_1908 = wp::extract(var_1866, var_1889, var_1906);
                            var_1909 = wp::atomic_add(var_efm_K_val_out, var_0, var_1907, var_1908);
                        }
                        var_1910 = wp::where(var_1897, var_1906, var_1888);
                        // row = dof_i + r                                                        <L 2229>
                        var_1912 = wp::add(var_1351, var_1911);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_1913 = wp::address(var_efm_K_rowadr, var_1912);
                        var_1914 = wp::address(var_efm_K_rownnz, var_1912);
                        var_1916 = wp::load(var_1913);
                        var_1917 = wp::load(var_1914);
                        var_1915 = _find_csr_col_0(var_1916, var_1917, var_1808, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_1919 = (var_1915 >= var_1918);
                        if (var_1919) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_1921 = wp::add(var_1915, var_1920);
                            var_1922 = wp::extract(var_1866, var_1911, var_1920);
                            var_1923 = wp::atomic_add(var_efm_K_val_out, var_0, var_1921, var_1922);
                            var_1925 = wp::add(var_1915, var_1924);
                            var_1926 = wp::extract(var_1866, var_1911, var_1924);
                            var_1927 = wp::atomic_add(var_efm_K_val_out, var_0, var_1925, var_1926);
                            var_1929 = wp::add(var_1915, var_1928);
                            var_1930 = wp::extract(var_1866, var_1911, var_1928);
                            var_1931 = wp::atomic_add(var_efm_K_val_out, var_0, var_1929, var_1930);
                        }
                        var_1932 = wp::where(var_1919, var_1928, var_1910);
                    }
                    var_1933 = wp::where(var_1825, var_1852, var_1911);
                    var_1934 = wp::where(var_1825, var_1853, var_1912);
                    var_1935 = wp::where(var_1825, var_1856, var_1915);
                    var_1936 = wp::where(var_1825, var_1803, var_1866);
                    var_1937 = wp::where(var_1825, var_1804, var_1932);
                }
                var_1938 = wp::where(var_1823, var_1824, var_1799);
                var_1939 = wp::where(var_1823, var_1933, var_1800);
                var_1940 = wp::where(var_1823, var_1934, var_1801);
                var_1941 = wp::where(var_1823, var_1935, var_1802);
                var_1942 = wp::where(var_1823, var_1936, var_1803);
                var_1943 = wp::where(var_1823, var_1937, var_1804);
            }
            var_1944 = wp::where(var_1812, var_1818, var_1798);
            var_1945 = wp::where(var_1812, var_1938, var_1799);
            var_1946 = wp::where(var_1812, var_1939, var_1800);
            var_1947 = wp::where(var_1812, var_1940, var_1801);
            var_1948 = wp::where(var_1812, var_1941, var_1802);
            var_1949 = wp::where(var_1812, var_1942, var_1803);
            var_1950 = wp::where(var_1812, var_1943, var_1804);
            var_1951 = wp::where(var_1812, var_1823, var_1812);
        }
        var_1952 = wp::where(var_1360, var_1361, var_1337);
        var_1953 = wp::where(var_1360, var_1363, var_1338);
        var_1954 = wp::where(var_1360, var_1366, var_1339);
        var_1955 = wp::where(var_1360, var_1806, var_1340);
        var_1956 = wp::where(var_1360, var_1808, var_1341);
        var_1957 = wp::where(var_1360, var_1944, var_1342);
        var_1958 = wp::where(var_1360, var_1945, var_1343);
        var_1959 = wp::where(var_1360, var_1946, var_1344);
        var_1960 = wp::where(var_1360, var_1947, var_1345);
        var_1961 = wp::where(var_1360, var_1948, var_1346);
        var_1962 = wp::where(var_1360, var_1949, var_1347);
        var_1963 = wp::where(var_1360, var_1950, var_1348);
        // dof_i = dofs[i]                                                                        <L 2206>
        var_1966 = wp::extract(var_81, var_1964);
        // if dof_i < 0 or efm_K_rownnz[dof_i] == 0:                                              <L 2207>
        var_1969 = (var_1966 < var_1968);
        var_1967 = var_1969;
        if (!var_1967) {
            var_1970 = wp::address(var_efm_K_rownnz, var_1966);
            var_1973 = wp::load(var_1970);
            var_1972 = (var_1973 == var_1971);
            var_1967 = var_1967 || var_1972;
        }
        if (var_1967) {
            // continue                                                                           <L 2208>
        }
        var_1975 = wp::where(var_1967, var_1974, var_1965);
        // bi = bodies[i]                                                                         <L 2209>
        if (var_1975) {
            var_1976 = wp::extract(var_86, var_1964);
            // R_bi_T = wp.transpose(xmat_in[worldid, bi])                                        <L 2210>
            var_1977 = wp::address(var_xmat_in, var_0, var_1976);
            var_1979 = wp::load(var_1977);
            var_1978 = wp::transpose(var_1979);
            // qi = xquat_in[worldid, bi]                                                         <L 2211>
            var_1980 = wp::address(var_xquat_in, var_0, var_1976);
            var_1982 = wp::load(var_1980);
            var_1981 = wp::copy(var_1982);
            // for j in range(4):                                                                 <L 2212>
            // dof_j = dofs[j]                                                                    <L 2213>
            var_1985 = wp::extract(var_81, var_1983);
            // if dof_j < 0:                                                                      <L 2214>
            var_1987 = (var_1985 < var_1986);
            if (var_1987) {
                // continue                                                                       <L 2215>
            }
            var_1989 = wp::where(var_1987, var_1988, var_1984);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_1989) {
                var_1991 = wp::mul(var_1990, var_1964);
                var_1992 = wp::add(var_76, var_1991);
                var_1993 = wp::add(var_1992, var_1983);
                var_1994 = wp::address(var_flex_bending, var_1993);
                var_1996 = wp::load(var_1994);
                var_1995 = wp::mul(var_46, var_1996);
                // if q == 0.0:                                                                   <L 2217>
                var_1998 = (var_1995 == var_1997);
                if (var_1998) {
                    // continue                                                                   <L 2218>
                }
                var_2000 = wp::where(var_1998, var_1999, var_1989);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_2000) {
                    var_2001 = wp::extract(var_86, var_1983);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_2003 = (var_1976 == var_2001);
                    var_2002 = var_2003;
                    if (!var_2002) {
                        var_2004 = wp::address(var_xquat_in, var_0, var_2001);
                        var_2006 = wp::load(var_2004);
                        var_2005 = (var_1981 == var_2006);
                        var_2002 = var_2002 || var_2005;
                    }
                    if (var_2002) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_2008 = wp::add(var_1966, var_2007);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2009 = wp::address(var_efm_K_rowadr, var_2008);
                        var_2010 = wp::address(var_efm_K_rownnz, var_2008);
                        var_2012 = wp::load(var_2009);
                        var_2013 = wp::load(var_2010);
                        var_2011 = _find_csr_col_0(var_2012, var_2013, var_1985, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2015 = (var_2011 >= var_2014);
                        if (var_2015) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2016 = wp::add(var_2011, var_2007);
                            var_2017 = wp::atomic_add(var_efm_K_val_out, var_0, var_2016, var_1995);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_2019 = wp::add(var_1966, var_2018);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2020 = wp::address(var_efm_K_rowadr, var_2019);
                        var_2021 = wp::address(var_efm_K_rownnz, var_2019);
                        var_2023 = wp::load(var_2020);
                        var_2024 = wp::load(var_2021);
                        var_2022 = _find_csr_col_0(var_2023, var_2024, var_1985, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2026 = (var_2022 >= var_2025);
                        if (var_2026) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2027 = wp::add(var_2022, var_2018);
                            var_2028 = wp::atomic_add(var_efm_K_val_out, var_0, var_2027, var_1995);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_2030 = wp::add(var_1966, var_2029);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2031 = wp::address(var_efm_K_rowadr, var_2030);
                        var_2032 = wp::address(var_efm_K_rownnz, var_2030);
                        var_2034 = wp::load(var_2031);
                        var_2035 = wp::load(var_2032);
                        var_2033 = _find_csr_col_0(var_2034, var_2035, var_1985, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2037 = (var_2033 >= var_2036);
                        if (var_2037) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2038 = wp::add(var_2033, var_2029);
                            var_2039 = wp::atomic_add(var_efm_K_val_out, var_0, var_2038, var_1995);
                        }
                    }
                    if (!var_2002) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_2040 = wp::address(var_xmat_in, var_0, var_2001);
                        var_2042 = wp::load(var_2040);
                        var_2041 = wp::mul(var_1978, var_2042);
                        var_2043 = wp::mul(var_1995, var_2041);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_2045 = wp::add(var_1966, var_2044);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2046 = wp::address(var_efm_K_rowadr, var_2045);
                        var_2047 = wp::address(var_efm_K_rownnz, var_2045);
                        var_2049 = wp::load(var_2046);
                        var_2050 = wp::load(var_2047);
                        var_2048 = _find_csr_col_0(var_2049, var_2050, var_1985, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2052 = (var_2048 >= var_2051);
                        if (var_2052) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2054 = wp::add(var_2048, var_2053);
                            var_2055 = wp::extract(var_2043, var_2044, var_2053);
                            var_2056 = wp::atomic_add(var_efm_K_val_out, var_0, var_2054, var_2055);
                            var_2058 = wp::add(var_2048, var_2057);
                            var_2059 = wp::extract(var_2043, var_2044, var_2057);
                            var_2060 = wp::atomic_add(var_efm_K_val_out, var_0, var_2058, var_2059);
                            var_2062 = wp::add(var_2048, var_2061);
                            var_2063 = wp::extract(var_2043, var_2044, var_2061);
                            var_2064 = wp::atomic_add(var_efm_K_val_out, var_0, var_2062, var_2063);
                        }
                        var_2065 = wp::where(var_2052, var_2061, var_1963);
                        // row = dof_i + r                                                        <L 2229>
                        var_2067 = wp::add(var_1966, var_2066);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2068 = wp::address(var_efm_K_rowadr, var_2067);
                        var_2069 = wp::address(var_efm_K_rownnz, var_2067);
                        var_2071 = wp::load(var_2068);
                        var_2072 = wp::load(var_2069);
                        var_2070 = _find_csr_col_0(var_2071, var_2072, var_1985, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2074 = (var_2070 >= var_2073);
                        if (var_2074) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2076 = wp::add(var_2070, var_2075);
                            var_2077 = wp::extract(var_2043, var_2066, var_2075);
                            var_2078 = wp::atomic_add(var_efm_K_val_out, var_0, var_2076, var_2077);
                            var_2080 = wp::add(var_2070, var_2079);
                            var_2081 = wp::extract(var_2043, var_2066, var_2079);
                            var_2082 = wp::atomic_add(var_efm_K_val_out, var_0, var_2080, var_2081);
                            var_2084 = wp::add(var_2070, var_2083);
                            var_2085 = wp::extract(var_2043, var_2066, var_2083);
                            var_2086 = wp::atomic_add(var_efm_K_val_out, var_0, var_2084, var_2085);
                        }
                        var_2087 = wp::where(var_2074, var_2083, var_2065);
                        // row = dof_i + r                                                        <L 2229>
                        var_2089 = wp::add(var_1966, var_2088);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2090 = wp::address(var_efm_K_rowadr, var_2089);
                        var_2091 = wp::address(var_efm_K_rownnz, var_2089);
                        var_2093 = wp::load(var_2090);
                        var_2094 = wp::load(var_2091);
                        var_2092 = _find_csr_col_0(var_2093, var_2094, var_1985, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2096 = (var_2092 >= var_2095);
                        if (var_2096) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2098 = wp::add(var_2092, var_2097);
                            var_2099 = wp::extract(var_2043, var_2088, var_2097);
                            var_2100 = wp::atomic_add(var_efm_K_val_out, var_0, var_2098, var_2099);
                            var_2102 = wp::add(var_2092, var_2101);
                            var_2103 = wp::extract(var_2043, var_2088, var_2101);
                            var_2104 = wp::atomic_add(var_efm_K_val_out, var_0, var_2102, var_2103);
                            var_2106 = wp::add(var_2092, var_2105);
                            var_2107 = wp::extract(var_2043, var_2088, var_2105);
                            var_2108 = wp::atomic_add(var_efm_K_val_out, var_0, var_2106, var_2107);
                        }
                        var_2109 = wp::where(var_2096, var_2105, var_2087);
                    }
                    var_2110 = wp::where(var_2002, var_2029, var_2088);
                    var_2111 = wp::where(var_2002, var_2030, var_2089);
                    var_2112 = wp::where(var_2002, var_2033, var_2092);
                    var_2113 = wp::where(var_2002, var_1962, var_2043);
                    var_2114 = wp::where(var_2002, var_1963, var_2109);
                }
                var_2115 = wp::where(var_2000, var_2001, var_1958);
                var_2116 = wp::where(var_2000, var_2110, var_1959);
                var_2117 = wp::where(var_2000, var_2111, var_1960);
                var_2118 = wp::where(var_2000, var_2112, var_1961);
                var_2119 = wp::where(var_2000, var_2113, var_1962);
                var_2120 = wp::where(var_2000, var_2114, var_1963);
            }
            var_2121 = wp::where(var_1989, var_1995, var_1957);
            var_2122 = wp::where(var_1989, var_2115, var_1958);
            var_2123 = wp::where(var_1989, var_2116, var_1959);
            var_2124 = wp::where(var_1989, var_2117, var_1960);
            var_2125 = wp::where(var_1989, var_2118, var_1961);
            var_2126 = wp::where(var_1989, var_2119, var_1962);
            var_2127 = wp::where(var_1989, var_2120, var_1963);
            var_2128 = wp::where(var_1989, var_2000, var_1989);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_2131 = wp::extract(var_81, var_2129);
            // if dof_j < 0:                                                                      <L 2214>
            var_2133 = (var_2131 < var_2132);
            if (var_2133) {
                // continue                                                                       <L 2215>
            }
            var_2135 = wp::where(var_2133, var_2134, var_2130);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_2135) {
                var_2137 = wp::mul(var_2136, var_1964);
                var_2138 = wp::add(var_76, var_2137);
                var_2139 = wp::add(var_2138, var_2129);
                var_2140 = wp::address(var_flex_bending, var_2139);
                var_2142 = wp::load(var_2140);
                var_2141 = wp::mul(var_46, var_2142);
                // if q == 0.0:                                                                   <L 2217>
                var_2144 = (var_2141 == var_2143);
                if (var_2144) {
                    // continue                                                                   <L 2218>
                }
                var_2146 = wp::where(var_2144, var_2145, var_2135);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_2146) {
                    var_2147 = wp::extract(var_86, var_2129);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_2149 = (var_1976 == var_2147);
                    var_2148 = var_2149;
                    if (!var_2148) {
                        var_2150 = wp::address(var_xquat_in, var_0, var_2147);
                        var_2152 = wp::load(var_2150);
                        var_2151 = (var_1981 == var_2152);
                        var_2148 = var_2148 || var_2151;
                    }
                    if (var_2148) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_2154 = wp::add(var_1966, var_2153);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2155 = wp::address(var_efm_K_rowadr, var_2154);
                        var_2156 = wp::address(var_efm_K_rownnz, var_2154);
                        var_2158 = wp::load(var_2155);
                        var_2159 = wp::load(var_2156);
                        var_2157 = _find_csr_col_0(var_2158, var_2159, var_2131, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2161 = (var_2157 >= var_2160);
                        if (var_2161) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2162 = wp::add(var_2157, var_2153);
                            var_2163 = wp::atomic_add(var_efm_K_val_out, var_0, var_2162, var_2141);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_2165 = wp::add(var_1966, var_2164);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2166 = wp::address(var_efm_K_rowadr, var_2165);
                        var_2167 = wp::address(var_efm_K_rownnz, var_2165);
                        var_2169 = wp::load(var_2166);
                        var_2170 = wp::load(var_2167);
                        var_2168 = _find_csr_col_0(var_2169, var_2170, var_2131, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2172 = (var_2168 >= var_2171);
                        if (var_2172) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2173 = wp::add(var_2168, var_2164);
                            var_2174 = wp::atomic_add(var_efm_K_val_out, var_0, var_2173, var_2141);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_2176 = wp::add(var_1966, var_2175);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2177 = wp::address(var_efm_K_rowadr, var_2176);
                        var_2178 = wp::address(var_efm_K_rownnz, var_2176);
                        var_2180 = wp::load(var_2177);
                        var_2181 = wp::load(var_2178);
                        var_2179 = _find_csr_col_0(var_2180, var_2181, var_2131, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2183 = (var_2179 >= var_2182);
                        if (var_2183) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2184 = wp::add(var_2179, var_2175);
                            var_2185 = wp::atomic_add(var_efm_K_val_out, var_0, var_2184, var_2141);
                        }
                    }
                    if (!var_2148) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_2186 = wp::address(var_xmat_in, var_0, var_2147);
                        var_2188 = wp::load(var_2186);
                        var_2187 = wp::mul(var_1978, var_2188);
                        var_2189 = wp::mul(var_2141, var_2187);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_2191 = wp::add(var_1966, var_2190);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2192 = wp::address(var_efm_K_rowadr, var_2191);
                        var_2193 = wp::address(var_efm_K_rownnz, var_2191);
                        var_2195 = wp::load(var_2192);
                        var_2196 = wp::load(var_2193);
                        var_2194 = _find_csr_col_0(var_2195, var_2196, var_2131, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2198 = (var_2194 >= var_2197);
                        if (var_2198) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2200 = wp::add(var_2194, var_2199);
                            var_2201 = wp::extract(var_2189, var_2190, var_2199);
                            var_2202 = wp::atomic_add(var_efm_K_val_out, var_0, var_2200, var_2201);
                            var_2204 = wp::add(var_2194, var_2203);
                            var_2205 = wp::extract(var_2189, var_2190, var_2203);
                            var_2206 = wp::atomic_add(var_efm_K_val_out, var_0, var_2204, var_2205);
                            var_2208 = wp::add(var_2194, var_2207);
                            var_2209 = wp::extract(var_2189, var_2190, var_2207);
                            var_2210 = wp::atomic_add(var_efm_K_val_out, var_0, var_2208, var_2209);
                        }
                        var_2211 = wp::where(var_2198, var_2207, var_2127);
                        // row = dof_i + r                                                        <L 2229>
                        var_2213 = wp::add(var_1966, var_2212);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2214 = wp::address(var_efm_K_rowadr, var_2213);
                        var_2215 = wp::address(var_efm_K_rownnz, var_2213);
                        var_2217 = wp::load(var_2214);
                        var_2218 = wp::load(var_2215);
                        var_2216 = _find_csr_col_0(var_2217, var_2218, var_2131, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2220 = (var_2216 >= var_2219);
                        if (var_2220) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2222 = wp::add(var_2216, var_2221);
                            var_2223 = wp::extract(var_2189, var_2212, var_2221);
                            var_2224 = wp::atomic_add(var_efm_K_val_out, var_0, var_2222, var_2223);
                            var_2226 = wp::add(var_2216, var_2225);
                            var_2227 = wp::extract(var_2189, var_2212, var_2225);
                            var_2228 = wp::atomic_add(var_efm_K_val_out, var_0, var_2226, var_2227);
                            var_2230 = wp::add(var_2216, var_2229);
                            var_2231 = wp::extract(var_2189, var_2212, var_2229);
                            var_2232 = wp::atomic_add(var_efm_K_val_out, var_0, var_2230, var_2231);
                        }
                        var_2233 = wp::where(var_2220, var_2229, var_2211);
                        // row = dof_i + r                                                        <L 2229>
                        var_2235 = wp::add(var_1966, var_2234);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2236 = wp::address(var_efm_K_rowadr, var_2235);
                        var_2237 = wp::address(var_efm_K_rownnz, var_2235);
                        var_2239 = wp::load(var_2236);
                        var_2240 = wp::load(var_2237);
                        var_2238 = _find_csr_col_0(var_2239, var_2240, var_2131, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2242 = (var_2238 >= var_2241);
                        if (var_2242) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2244 = wp::add(var_2238, var_2243);
                            var_2245 = wp::extract(var_2189, var_2234, var_2243);
                            var_2246 = wp::atomic_add(var_efm_K_val_out, var_0, var_2244, var_2245);
                            var_2248 = wp::add(var_2238, var_2247);
                            var_2249 = wp::extract(var_2189, var_2234, var_2247);
                            var_2250 = wp::atomic_add(var_efm_K_val_out, var_0, var_2248, var_2249);
                            var_2252 = wp::add(var_2238, var_2251);
                            var_2253 = wp::extract(var_2189, var_2234, var_2251);
                            var_2254 = wp::atomic_add(var_efm_K_val_out, var_0, var_2252, var_2253);
                        }
                        var_2255 = wp::where(var_2242, var_2251, var_2233);
                    }
                    var_2256 = wp::where(var_2148, var_2175, var_2234);
                    var_2257 = wp::where(var_2148, var_2176, var_2235);
                    var_2258 = wp::where(var_2148, var_2179, var_2238);
                    var_2259 = wp::where(var_2148, var_2126, var_2189);
                    var_2260 = wp::where(var_2148, var_2127, var_2255);
                }
                var_2261 = wp::where(var_2146, var_2147, var_2122);
                var_2262 = wp::where(var_2146, var_2256, var_2123);
                var_2263 = wp::where(var_2146, var_2257, var_2124);
                var_2264 = wp::where(var_2146, var_2258, var_2125);
                var_2265 = wp::where(var_2146, var_2259, var_2126);
                var_2266 = wp::where(var_2146, var_2260, var_2127);
            }
            var_2267 = wp::where(var_2135, var_2141, var_2121);
            var_2268 = wp::where(var_2135, var_2261, var_2122);
            var_2269 = wp::where(var_2135, var_2262, var_2123);
            var_2270 = wp::where(var_2135, var_2263, var_2124);
            var_2271 = wp::where(var_2135, var_2264, var_2125);
            var_2272 = wp::where(var_2135, var_2265, var_2126);
            var_2273 = wp::where(var_2135, var_2266, var_2127);
            var_2274 = wp::where(var_2135, var_2146, var_2135);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_2277 = wp::extract(var_81, var_2275);
            // if dof_j < 0:                                                                      <L 2214>
            var_2279 = (var_2277 < var_2278);
            if (var_2279) {
                // continue                                                                       <L 2215>
            }
            var_2281 = wp::where(var_2279, var_2280, var_2276);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_2281) {
                var_2283 = wp::mul(var_2282, var_1964);
                var_2284 = wp::add(var_76, var_2283);
                var_2285 = wp::add(var_2284, var_2275);
                var_2286 = wp::address(var_flex_bending, var_2285);
                var_2288 = wp::load(var_2286);
                var_2287 = wp::mul(var_46, var_2288);
                // if q == 0.0:                                                                   <L 2217>
                var_2290 = (var_2287 == var_2289);
                if (var_2290) {
                    // continue                                                                   <L 2218>
                }
                var_2292 = wp::where(var_2290, var_2291, var_2281);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_2292) {
                    var_2293 = wp::extract(var_86, var_2275);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_2295 = (var_1976 == var_2293);
                    var_2294 = var_2295;
                    if (!var_2294) {
                        var_2296 = wp::address(var_xquat_in, var_0, var_2293);
                        var_2298 = wp::load(var_2296);
                        var_2297 = (var_1981 == var_2298);
                        var_2294 = var_2294 || var_2297;
                    }
                    if (var_2294) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_2300 = wp::add(var_1966, var_2299);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2301 = wp::address(var_efm_K_rowadr, var_2300);
                        var_2302 = wp::address(var_efm_K_rownnz, var_2300);
                        var_2304 = wp::load(var_2301);
                        var_2305 = wp::load(var_2302);
                        var_2303 = _find_csr_col_0(var_2304, var_2305, var_2277, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2307 = (var_2303 >= var_2306);
                        if (var_2307) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2308 = wp::add(var_2303, var_2299);
                            var_2309 = wp::atomic_add(var_efm_K_val_out, var_0, var_2308, var_2287);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_2311 = wp::add(var_1966, var_2310);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2312 = wp::address(var_efm_K_rowadr, var_2311);
                        var_2313 = wp::address(var_efm_K_rownnz, var_2311);
                        var_2315 = wp::load(var_2312);
                        var_2316 = wp::load(var_2313);
                        var_2314 = _find_csr_col_0(var_2315, var_2316, var_2277, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2318 = (var_2314 >= var_2317);
                        if (var_2318) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2319 = wp::add(var_2314, var_2310);
                            var_2320 = wp::atomic_add(var_efm_K_val_out, var_0, var_2319, var_2287);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_2322 = wp::add(var_1966, var_2321);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2323 = wp::address(var_efm_K_rowadr, var_2322);
                        var_2324 = wp::address(var_efm_K_rownnz, var_2322);
                        var_2326 = wp::load(var_2323);
                        var_2327 = wp::load(var_2324);
                        var_2325 = _find_csr_col_0(var_2326, var_2327, var_2277, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2329 = (var_2325 >= var_2328);
                        if (var_2329) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2330 = wp::add(var_2325, var_2321);
                            var_2331 = wp::atomic_add(var_efm_K_val_out, var_0, var_2330, var_2287);
                        }
                    }
                    if (!var_2294) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_2332 = wp::address(var_xmat_in, var_0, var_2293);
                        var_2334 = wp::load(var_2332);
                        var_2333 = wp::mul(var_1978, var_2334);
                        var_2335 = wp::mul(var_2287, var_2333);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_2337 = wp::add(var_1966, var_2336);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2338 = wp::address(var_efm_K_rowadr, var_2337);
                        var_2339 = wp::address(var_efm_K_rownnz, var_2337);
                        var_2341 = wp::load(var_2338);
                        var_2342 = wp::load(var_2339);
                        var_2340 = _find_csr_col_0(var_2341, var_2342, var_2277, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2344 = (var_2340 >= var_2343);
                        if (var_2344) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2346 = wp::add(var_2340, var_2345);
                            var_2347 = wp::extract(var_2335, var_2336, var_2345);
                            var_2348 = wp::atomic_add(var_efm_K_val_out, var_0, var_2346, var_2347);
                            var_2350 = wp::add(var_2340, var_2349);
                            var_2351 = wp::extract(var_2335, var_2336, var_2349);
                            var_2352 = wp::atomic_add(var_efm_K_val_out, var_0, var_2350, var_2351);
                            var_2354 = wp::add(var_2340, var_2353);
                            var_2355 = wp::extract(var_2335, var_2336, var_2353);
                            var_2356 = wp::atomic_add(var_efm_K_val_out, var_0, var_2354, var_2355);
                        }
                        var_2357 = wp::where(var_2344, var_2353, var_2273);
                        // row = dof_i + r                                                        <L 2229>
                        var_2359 = wp::add(var_1966, var_2358);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2360 = wp::address(var_efm_K_rowadr, var_2359);
                        var_2361 = wp::address(var_efm_K_rownnz, var_2359);
                        var_2363 = wp::load(var_2360);
                        var_2364 = wp::load(var_2361);
                        var_2362 = _find_csr_col_0(var_2363, var_2364, var_2277, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2366 = (var_2362 >= var_2365);
                        if (var_2366) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2368 = wp::add(var_2362, var_2367);
                            var_2369 = wp::extract(var_2335, var_2358, var_2367);
                            var_2370 = wp::atomic_add(var_efm_K_val_out, var_0, var_2368, var_2369);
                            var_2372 = wp::add(var_2362, var_2371);
                            var_2373 = wp::extract(var_2335, var_2358, var_2371);
                            var_2374 = wp::atomic_add(var_efm_K_val_out, var_0, var_2372, var_2373);
                            var_2376 = wp::add(var_2362, var_2375);
                            var_2377 = wp::extract(var_2335, var_2358, var_2375);
                            var_2378 = wp::atomic_add(var_efm_K_val_out, var_0, var_2376, var_2377);
                        }
                        var_2379 = wp::where(var_2366, var_2375, var_2357);
                        // row = dof_i + r                                                        <L 2229>
                        var_2381 = wp::add(var_1966, var_2380);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2382 = wp::address(var_efm_K_rowadr, var_2381);
                        var_2383 = wp::address(var_efm_K_rownnz, var_2381);
                        var_2385 = wp::load(var_2382);
                        var_2386 = wp::load(var_2383);
                        var_2384 = _find_csr_col_0(var_2385, var_2386, var_2277, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2388 = (var_2384 >= var_2387);
                        if (var_2388) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2390 = wp::add(var_2384, var_2389);
                            var_2391 = wp::extract(var_2335, var_2380, var_2389);
                            var_2392 = wp::atomic_add(var_efm_K_val_out, var_0, var_2390, var_2391);
                            var_2394 = wp::add(var_2384, var_2393);
                            var_2395 = wp::extract(var_2335, var_2380, var_2393);
                            var_2396 = wp::atomic_add(var_efm_K_val_out, var_0, var_2394, var_2395);
                            var_2398 = wp::add(var_2384, var_2397);
                            var_2399 = wp::extract(var_2335, var_2380, var_2397);
                            var_2400 = wp::atomic_add(var_efm_K_val_out, var_0, var_2398, var_2399);
                        }
                        var_2401 = wp::where(var_2388, var_2397, var_2379);
                    }
                    var_2402 = wp::where(var_2294, var_2321, var_2380);
                    var_2403 = wp::where(var_2294, var_2322, var_2381);
                    var_2404 = wp::where(var_2294, var_2325, var_2384);
                    var_2405 = wp::where(var_2294, var_2272, var_2335);
                    var_2406 = wp::where(var_2294, var_2273, var_2401);
                }
                var_2407 = wp::where(var_2292, var_2293, var_2268);
                var_2408 = wp::where(var_2292, var_2402, var_2269);
                var_2409 = wp::where(var_2292, var_2403, var_2270);
                var_2410 = wp::where(var_2292, var_2404, var_2271);
                var_2411 = wp::where(var_2292, var_2405, var_2272);
                var_2412 = wp::where(var_2292, var_2406, var_2273);
            }
            var_2413 = wp::where(var_2281, var_2287, var_2267);
            var_2414 = wp::where(var_2281, var_2407, var_2268);
            var_2415 = wp::where(var_2281, var_2408, var_2269);
            var_2416 = wp::where(var_2281, var_2409, var_2270);
            var_2417 = wp::where(var_2281, var_2410, var_2271);
            var_2418 = wp::where(var_2281, var_2411, var_2272);
            var_2419 = wp::where(var_2281, var_2412, var_2273);
            var_2420 = wp::where(var_2281, var_2292, var_2281);
            // dof_j = dofs[j]                                                                    <L 2213>
            var_2423 = wp::extract(var_81, var_2421);
            // if dof_j < 0:                                                                      <L 2214>
            var_2425 = (var_2423 < var_2424);
            if (var_2425) {
                // continue                                                                       <L 2215>
            }
            var_2427 = wp::where(var_2425, var_2426, var_2422);
            // q = scale * flex_bending[bend_offset + 4 * i + j]                                  <L 2216>
            if (var_2427) {
                var_2429 = wp::mul(var_2428, var_1964);
                var_2430 = wp::add(var_76, var_2429);
                var_2431 = wp::add(var_2430, var_2421);
                var_2432 = wp::address(var_flex_bending, var_2431);
                var_2434 = wp::load(var_2432);
                var_2433 = wp::mul(var_46, var_2434);
                // if q == 0.0:                                                                   <L 2217>
                var_2436 = (var_2433 == var_2435);
                if (var_2436) {
                    // continue                                                                   <L 2218>
                }
                var_2438 = wp::where(var_2436, var_2437, var_2427);
                // bj = bodies[j]                                                                 <L 2219>
                if (var_2438) {
                    var_2439 = wp::extract(var_86, var_2421);
                    // if bi == bj or qi == xquat_in[worldid, bj]:                                <L 2220>
                    var_2441 = (var_1976 == var_2439);
                    var_2440 = var_2441;
                    if (!var_2440) {
                        var_2442 = wp::address(var_xquat_in, var_0, var_2439);
                        var_2444 = wp::load(var_2442);
                        var_2443 = (var_1981 == var_2444);
                        var_2440 = var_2440 || var_2443;
                    }
                    if (var_2440) {
                        // for r in range(3):                                                     <L 2221>
                        // row = dof_i + r                                                        <L 2222>
                        var_2446 = wp::add(var_1966, var_2445);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2447 = wp::address(var_efm_K_rowadr, var_2446);
                        var_2448 = wp::address(var_efm_K_rownnz, var_2446);
                        var_2450 = wp::load(var_2447);
                        var_2451 = wp::load(var_2448);
                        var_2449 = _find_csr_col_0(var_2450, var_2451, var_2423, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2453 = (var_2449 >= var_2452);
                        if (var_2453) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2454 = wp::add(var_2449, var_2445);
                            var_2455 = wp::atomic_add(var_efm_K_val_out, var_0, var_2454, var_2433);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_2457 = wp::add(var_1966, var_2456);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2458 = wp::address(var_efm_K_rowadr, var_2457);
                        var_2459 = wp::address(var_efm_K_rownnz, var_2457);
                        var_2461 = wp::load(var_2458);
                        var_2462 = wp::load(var_2459);
                        var_2460 = _find_csr_col_0(var_2461, var_2462, var_2423, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2464 = (var_2460 >= var_2463);
                        if (var_2464) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2465 = wp::add(var_2460, var_2456);
                            var_2466 = wp::atomic_add(var_efm_K_val_out, var_0, var_2465, var_2433);
                        }
                        // row = dof_i + r                                                        <L 2222>
                        var_2468 = wp::add(var_1966, var_2467);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2223>
                        var_2469 = wp::address(var_efm_K_rowadr, var_2468);
                        var_2470 = wp::address(var_efm_K_rownnz, var_2468);
                        var_2472 = wp::load(var_2469);
                        var_2473 = wp::load(var_2470);
                        var_2471 = _find_csr_col_0(var_2472, var_2473, var_2423, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2224>
                        var_2475 = (var_2471 >= var_2474);
                        if (var_2475) {
                            // wp.atomic_add(efm_K_val_out, worldid, idx + r, q)                  <L 2225>
                            var_2476 = wp::add(var_2471, var_2467);
                            var_2477 = wp::atomic_add(var_efm_K_val_out, var_0, var_2476, var_2433);
                        }
                    }
                    if (!var_2440) {
                        // blkd = q * (R_bi_T * xmat_in[worldid, bj])                             <L 2227>
                        var_2478 = wp::address(var_xmat_in, var_0, var_2439);
                        var_2480 = wp::load(var_2478);
                        var_2479 = wp::mul(var_1978, var_2480);
                        var_2481 = wp::mul(var_2433, var_2479);
                        // for r in range(3):                                                     <L 2228>
                        // row = dof_i + r                                                        <L 2229>
                        var_2483 = wp::add(var_1966, var_2482);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2484 = wp::address(var_efm_K_rowadr, var_2483);
                        var_2485 = wp::address(var_efm_K_rownnz, var_2483);
                        var_2487 = wp::load(var_2484);
                        var_2488 = wp::load(var_2485);
                        var_2486 = _find_csr_col_0(var_2487, var_2488, var_2423, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2490 = (var_2486 >= var_2489);
                        if (var_2490) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2492 = wp::add(var_2486, var_2491);
                            var_2493 = wp::extract(var_2481, var_2482, var_2491);
                            var_2494 = wp::atomic_add(var_efm_K_val_out, var_0, var_2492, var_2493);
                            var_2496 = wp::add(var_2486, var_2495);
                            var_2497 = wp::extract(var_2481, var_2482, var_2495);
                            var_2498 = wp::atomic_add(var_efm_K_val_out, var_0, var_2496, var_2497);
                            var_2500 = wp::add(var_2486, var_2499);
                            var_2501 = wp::extract(var_2481, var_2482, var_2499);
                            var_2502 = wp::atomic_add(var_efm_K_val_out, var_0, var_2500, var_2501);
                        }
                        var_2503 = wp::where(var_2490, var_2499, var_2419);
                        // row = dof_i + r                                                        <L 2229>
                        var_2505 = wp::add(var_1966, var_2504);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2506 = wp::address(var_efm_K_rowadr, var_2505);
                        var_2507 = wp::address(var_efm_K_rownnz, var_2505);
                        var_2509 = wp::load(var_2506);
                        var_2510 = wp::load(var_2507);
                        var_2508 = _find_csr_col_0(var_2509, var_2510, var_2423, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2512 = (var_2508 >= var_2511);
                        if (var_2512) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2514 = wp::add(var_2508, var_2513);
                            var_2515 = wp::extract(var_2481, var_2504, var_2513);
                            var_2516 = wp::atomic_add(var_efm_K_val_out, var_0, var_2514, var_2515);
                            var_2518 = wp::add(var_2508, var_2517);
                            var_2519 = wp::extract(var_2481, var_2504, var_2517);
                            var_2520 = wp::atomic_add(var_efm_K_val_out, var_0, var_2518, var_2519);
                            var_2522 = wp::add(var_2508, var_2521);
                            var_2523 = wp::extract(var_2481, var_2504, var_2521);
                            var_2524 = wp::atomic_add(var_efm_K_val_out, var_0, var_2522, var_2523);
                        }
                        var_2525 = wp::where(var_2512, var_2521, var_2503);
                        // row = dof_i + r                                                        <L 2229>
                        var_2527 = wp::add(var_1966, var_2526);
                        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)       <L 2230>
                        var_2528 = wp::address(var_efm_K_rowadr, var_2527);
                        var_2529 = wp::address(var_efm_K_rownnz, var_2527);
                        var_2531 = wp::load(var_2528);
                        var_2532 = wp::load(var_2529);
                        var_2530 = _find_csr_col_0(var_2531, var_2532, var_2423, var_efm_K_colind);
                        // if idx >= 0:                                                           <L 2231>
                        var_2534 = (var_2530 >= var_2533);
                        if (var_2534) {
                            // for c in range(3):                                                 <L 2232>
                            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])         <L 2233>
                            var_2536 = wp::add(var_2530, var_2535);
                            var_2537 = wp::extract(var_2481, var_2526, var_2535);
                            var_2538 = wp::atomic_add(var_efm_K_val_out, var_0, var_2536, var_2537);
                            var_2540 = wp::add(var_2530, var_2539);
                            var_2541 = wp::extract(var_2481, var_2526, var_2539);
                            var_2542 = wp::atomic_add(var_efm_K_val_out, var_0, var_2540, var_2541);
                            var_2544 = wp::add(var_2530, var_2543);
                            var_2545 = wp::extract(var_2481, var_2526, var_2543);
                            var_2546 = wp::atomic_add(var_efm_K_val_out, var_0, var_2544, var_2545);
                        }
                        var_2547 = wp::where(var_2534, var_2543, var_2525);
                    }
                    var_2548 = wp::where(var_2440, var_2467, var_2526);
                    var_2549 = wp::where(var_2440, var_2468, var_2527);
                    var_2550 = wp::where(var_2440, var_2471, var_2530);
                    var_2551 = wp::where(var_2440, var_2418, var_2481);
                    var_2552 = wp::where(var_2440, var_2419, var_2547);
                }
                var_2553 = wp::where(var_2438, var_2439, var_2414);
                var_2554 = wp::where(var_2438, var_2548, var_2415);
                var_2555 = wp::where(var_2438, var_2549, var_2416);
                var_2556 = wp::where(var_2438, var_2550, var_2417);
                var_2557 = wp::where(var_2438, var_2551, var_2418);
                var_2558 = wp::where(var_2438, var_2552, var_2419);
            }
            var_2559 = wp::where(var_2427, var_2433, var_2413);
            var_2560 = wp::where(var_2427, var_2553, var_2414);
            var_2561 = wp::where(var_2427, var_2554, var_2415);
            var_2562 = wp::where(var_2427, var_2555, var_2416);
            var_2563 = wp::where(var_2427, var_2556, var_2417);
            var_2564 = wp::where(var_2427, var_2557, var_2418);
            var_2565 = wp::where(var_2427, var_2558, var_2419);
            var_2566 = wp::where(var_2427, var_2438, var_2427);
        }
        var_2567 = wp::where(var_1975, var_1976, var_1952);
        var_2568 = wp::where(var_1975, var_1978, var_1953);
        var_2569 = wp::where(var_1975, var_1981, var_1954);
        var_2570 = wp::where(var_1975, var_2421, var_1955);
        var_2571 = wp::where(var_1975, var_2423, var_1956);
        var_2572 = wp::where(var_1975, var_2559, var_1957);
        var_2573 = wp::where(var_1975, var_2560, var_1958);
        var_2574 = wp::where(var_1975, var_2561, var_1959);
        var_2575 = wp::where(var_1975, var_2562, var_1960);
        var_2576 = wp::where(var_1975, var_2563, var_1961);
        var_2577 = wp::where(var_1975, var_2564, var_1962);
        var_2578 = wp::where(var_1975, var_2565, var_1963);
}



extern "C" __global__ void _eff_flex_interp_stiff_3fe1612b_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_flex_interp,
    wp::array_t<wp::vec_t<3, wp::int32>> var_flex_cellnum,
    wp::array_t<wp::int32> var_flex_nodeadr,
    wp::array_t<wp::int32> var_flex_stiffnessadr,
    wp::array_t<wp::int32> var_flex_nodebodyid,
    wp::array_t<wp::float32> var_flex_stiffness,
    wp::array_t<wp::float32> var_flex_damping,
    wp::array_t<bool> var_flex_centered,
    wp::array_t<wp::int32> var_efm_K_rownnz,
    wp::array_t<wp::int32> var_efm_K_rowadr,
    wp::array_t<wp::int32> var_efm_K_colind,
    wp::array_t<wp::vec_t<4, wp::int32>> var_flex_cell_map,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flexnode_xpos_in,
    wp::array_t<wp::float32> var_efm_K_val_out)
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
        wp::float32 var_8;
        wp::float32 var_9;
        wp::vec_t<4, wp::int32>* var_10;
        wp::vec_t<4, wp::int32> var_11;
        wp::vec_t<4, wp::int32> var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        const wp::int32 var_17 = 2;
        wp::int32 var_18;
        const wp::int32 var_19 = 3;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        bool var_30;
        const wp::int32 var_31 = 32;
        wp::int32 var_32;
        const wp::float32 var_33 = 0.0;
        wp::float32 var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 64;
        wp::int32 var_37;
        const wp::float32 var_38 = 0.0;
        wp::float32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::float32 var_45 = 0.0;
        bool var_46;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        const wp::int32 var_49 = 1;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 1;
        wp::int32 var_53;
        wp::int32 var_54;
        const wp::int32 var_55 = 3;
        wp::int32 var_56;
        wp::vec_t<3, wp::int32>* var_57;
        wp::vec_t<3, wp::int32> var_58;
        wp::vec_t<3, wp::int32> var_59;
        const wp::int32 var_60 = 1;
        wp::int32 var_61;
        const wp::int32 var_62 = 2;
        wp::int32 var_63;
        wp::int32* var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        wp::int32 var_70;
        const wp::int32 var_71 = 1;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32 var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        wp::float32* var_81;
        const wp::float32 var_82 = 0.0;
        bool var_83;
        wp::float32 var_84;
        wp::quat_t<wp::float32> var_85;
        wp::mat_t<3, 3, wp::float32> var_86;
        wp::mat_t<3, 3, wp::float32> var_87;
        const wp::int32 var_88 = 1;
        wp::int32 var_89;
        wp::range_t var_90;
        wp::int32 var_91;
        const wp::int32 var_92 = 1;
        wp::int32 var_93;
        wp::range_t var_94;
        wp::int32 var_95;
        const wp::int32 var_96 = 1;
        wp::int32 var_97;
        wp::range_t var_98;
        wp::int32 var_99;
        const wp::int32 var_100 = 1;
        wp::int32 var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        const wp::int32 var_104 = 1;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::int32 var_107;
        wp::int32 var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::int32 var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        wp::int32* var_120;
        wp::int32 var_121;
        wp::int32 var_122;
        wp::int32* var_123;
        const wp::int32 var_124 = 3;
        bool var_125;
        wp::int32 var_126;
        wp::int32* var_127;
        wp::int32 var_128;
        wp::int32 var_129;
        wp::mat_t<3, 3, wp::float32>* var_130;
        wp::mat_t<3, 3, wp::float32> var_131;
        wp::mat_t<3, 3, wp::float32> var_132;
        wp::mat_t<3, 3, wp::float32> var_133;
        const wp::int32 var_134 = 1;
        wp::int32 var_135;
        wp::range_t var_136;
        wp::int32 var_137;
        const wp::int32 var_138 = 1;
        wp::int32 var_139;
        wp::range_t var_140;
        wp::int32 var_141;
        const wp::int32 var_142 = 1;
        wp::int32 var_143;
        wp::range_t var_144;
        wp::int32 var_145;
        const wp::int32 var_146 = 1;
        wp::int32 var_147;
        wp::int32 var_148;
        wp::int32 var_149;
        const wp::int32 var_150 = 1;
        wp::int32 var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::int32 var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        wp::int32 var_158;
        wp::int32 var_159;
        wp::int32 var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        wp::int32 var_163;
        wp::int32 var_164;
        wp::int32 var_165;
        wp::int32* var_166;
        wp::int32 var_167;
        wp::int32 var_168;
        wp::int32* var_169;
        const wp::int32 var_170 = 3;
        bool var_171;
        wp::int32 var_172;
        wp::int32* var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::mat_t<3, 3, wp::float32>* var_176;
        wp::mat_t<3, 3, wp::float32> var_177;
        wp::mat_t<3, 3, wp::float32> var_178;
        const wp::float32 var_179 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_180;
        const wp::int32 var_181 = 0;
        const wp::int32 var_182 = 0;
        const wp::int32 var_183 = 3;
        wp::int32 var_184;
        wp::int32 var_185;
        wp::int32 var_186;
        wp::int32 var_187;
        const wp::int32 var_188 = 3;
        wp::int32 var_189;
        wp::int32 var_190;
        wp::int32 var_191;
        wp::float32* var_192;
        wp::float32 var_193;
        wp::float32 var_194;
        const wp::int32 var_195 = 1;
        const wp::int32 var_196 = 3;
        wp::int32 var_197;
        wp::int32 var_198;
        wp::int32 var_199;
        wp::int32 var_200;
        const wp::int32 var_201 = 3;
        wp::int32 var_202;
        wp::int32 var_203;
        wp::int32 var_204;
        wp::float32* var_205;
        wp::float32 var_206;
        wp::float32 var_207;
        const wp::int32 var_208 = 2;
        const wp::int32 var_209 = 3;
        wp::int32 var_210;
        wp::int32 var_211;
        wp::int32 var_212;
        wp::int32 var_213;
        const wp::int32 var_214 = 3;
        wp::int32 var_215;
        wp::int32 var_216;
        wp::int32 var_217;
        wp::float32* var_218;
        wp::float32 var_219;
        wp::float32 var_220;
        const wp::int32 var_221 = 1;
        const wp::int32 var_222 = 0;
        const wp::int32 var_223 = 3;
        wp::int32 var_224;
        wp::int32 var_225;
        wp::int32 var_226;
        wp::int32 var_227;
        const wp::int32 var_228 = 3;
        wp::int32 var_229;
        wp::int32 var_230;
        wp::int32 var_231;
        wp::float32* var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        const wp::int32 var_235 = 1;
        const wp::int32 var_236 = 3;
        wp::int32 var_237;
        wp::int32 var_238;
        wp::int32 var_239;
        wp::int32 var_240;
        const wp::int32 var_241 = 3;
        wp::int32 var_242;
        wp::int32 var_243;
        wp::int32 var_244;
        wp::float32* var_245;
        wp::float32 var_246;
        wp::float32 var_247;
        const wp::int32 var_248 = 2;
        const wp::int32 var_249 = 3;
        wp::int32 var_250;
        wp::int32 var_251;
        wp::int32 var_252;
        wp::int32 var_253;
        const wp::int32 var_254 = 3;
        wp::int32 var_255;
        wp::int32 var_256;
        wp::int32 var_257;
        wp::float32* var_258;
        wp::float32 var_259;
        wp::float32 var_260;
        const wp::int32 var_261 = 2;
        const wp::int32 var_262 = 0;
        const wp::int32 var_263 = 3;
        wp::int32 var_264;
        wp::int32 var_265;
        wp::int32 var_266;
        wp::int32 var_267;
        const wp::int32 var_268 = 3;
        wp::int32 var_269;
        wp::int32 var_270;
        wp::int32 var_271;
        wp::float32* var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        const wp::int32 var_275 = 1;
        const wp::int32 var_276 = 3;
        wp::int32 var_277;
        wp::int32 var_278;
        wp::int32 var_279;
        wp::int32 var_280;
        const wp::int32 var_281 = 3;
        wp::int32 var_282;
        wp::int32 var_283;
        wp::int32 var_284;
        wp::float32* var_285;
        wp::float32 var_286;
        wp::float32 var_287;
        const wp::int32 var_288 = 2;
        const wp::int32 var_289 = 3;
        wp::int32 var_290;
        wp::int32 var_291;
        wp::int32 var_292;
        wp::int32 var_293;
        const wp::int32 var_294 = 3;
        wp::int32 var_295;
        wp::int32 var_296;
        wp::int32 var_297;
        wp::float32* var_298;
        wp::float32 var_299;
        wp::float32 var_300;
        wp::mat_t<3, 3, wp::float32> var_301;
        wp::mat_t<3, 3, wp::float32> var_302;
        wp::mat_t<3, 3, wp::float32> var_303;
        const wp::int32 var_304 = 0;
        wp::int32 var_305;
        wp::int32* var_306;
        wp::int32 var_307;
        wp::int32 var_308;
        wp::int32* var_309;
        wp::int32 var_310;
        wp::int32 var_311;
        const wp::int32 var_312 = 0;
        wp::int32 var_313;
        wp::int32 var_314;
        const wp::int32 var_315 = 0;
        bool var_316;
        wp::float32 var_317;
        wp::float32 var_318;
        const wp::int32 var_319 = 1;
        wp::int32 var_320;
        wp::int32 var_321;
        const wp::int32 var_322 = 0;
        bool var_323;
        wp::float32 var_324;
        wp::float32 var_325;
        const wp::int32 var_326 = 2;
        wp::int32 var_327;
        wp::int32 var_328;
        const wp::int32 var_329 = 0;
        bool var_330;
        wp::float32 var_331;
        wp::float32 var_332;
        const wp::int32 var_333 = 1;
        wp::int32 var_334;
        wp::int32* var_335;
        wp::int32 var_336;
        wp::int32 var_337;
        wp::int32* var_338;
        wp::int32 var_339;
        wp::int32 var_340;
        const wp::int32 var_341 = 0;
        wp::int32 var_342;
        wp::int32 var_343;
        const wp::int32 var_344 = 0;
        bool var_345;
        wp::float32 var_346;
        wp::float32 var_347;
        const wp::int32 var_348 = 1;
        wp::int32 var_349;
        wp::int32 var_350;
        const wp::int32 var_351 = 0;
        bool var_352;
        wp::float32 var_353;
        wp::float32 var_354;
        const wp::int32 var_355 = 2;
        wp::int32 var_356;
        wp::int32 var_357;
        const wp::int32 var_358 = 0;
        bool var_359;
        wp::float32 var_360;
        wp::float32 var_361;
        const wp::int32 var_362 = 2;
        wp::int32 var_363;
        wp::int32* var_364;
        wp::int32 var_365;
        wp::int32 var_366;
        wp::int32* var_367;
        wp::int32 var_368;
        wp::int32 var_369;
        const wp::int32 var_370 = 0;
        wp::int32 var_371;
        wp::int32 var_372;
        const wp::int32 var_373 = 0;
        bool var_374;
        wp::float32 var_375;
        wp::float32 var_376;
        const wp::int32 var_377 = 1;
        wp::int32 var_378;
        wp::int32 var_379;
        const wp::int32 var_380 = 0;
        bool var_381;
        wp::float32 var_382;
        wp::float32 var_383;
        const wp::int32 var_384 = 2;
        wp::int32 var_385;
        wp::int32 var_386;
        const wp::int32 var_387 = 0;
        bool var_388;
        wp::float32 var_389;
        wp::float32 var_390;
        //---------
        // forward
        // def _eff_flex_interp_stiff(                                                            <L 2237>
        // worldid, cellid = wp.tid()                                                             <L 2261>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 2262>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // mapping = flex_cell_map[cellid]                                                        <L 2264>
        var_10 = wp::address(var_flex_cell_map, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // f = mapping[0]                                                                         <L 2265>
        var_14 = wp::extract(var_11, var_13);
        // ci = mapping[1]                                                                        <L 2266>
        var_16 = wp::extract(var_11, var_15);
        // cj = mapping[2]                                                                        <L 2267>
        var_18 = wp::extract(var_11, var_17);
        // ck = mapping[3]                                                                        <L 2268>
        var_20 = wp::extract(var_11, var_19);
        // order = flex_interp[f]                                                                 <L 2270>
        var_21 = wp::address(var_flex_interp, var_14);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // if order <= 0:                                                                         <L 2271>
        var_25 = (var_22 <= var_24);
        if (var_25) {
            // return                                                                             <L 2272>
            return;
        }
        // stiffness_adr_base = flex_stiffnessadr[f]                                              <L 2274>
        var_26 = wp::address(var_flex_stiffnessadr, var_14);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // if stiffness_adr_base < 0:                                                             <L 2275>
        var_30 = (var_27 < var_29);
        if (var_30) {
            // return                                                                             <L 2276>
            return;
        }
        // s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep            <L 2278>
        var_32 = wp::bit_and(var_opt_disableflags, var_31);
        if (var_32) {
        }
        if (!var_32) {
            var_34 = wp::mul(var_8, var_8);
        }
        var_35 = wp::where(var_32, var_33, var_34);
        // s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep                       <L 2279>
        var_37 = wp::bit_and(var_opt_disableflags, var_36);
        if (var_37) {
        }
        if (!var_37) {
        }
        var_39 = wp::where(var_37, var_38, var_8);
        // iscale = -(s1 + s2 * flex_damping[f])                                                  <L 2280>
        var_40 = wp::address(var_flex_damping, var_14);
        var_42 = wp::load(var_40);
        var_41 = wp::mul(var_39, var_42);
        var_43 = wp::add(var_35, var_41);
        var_44 = wp::neg(var_43);
        // if iscale == 0.0:                                                                      <L 2281>
        var_46 = (var_44 == var_45);
        if (var_46) {
            // return                                                                             <L 2282>
            return;
        }
        // npc = (order + 1) * (order + 1) * (order + 1)                                          <L 2284>
        var_48 = wp::add(var_22, var_47);
        var_50 = wp::add(var_22, var_49);
        var_51 = wp::mul(var_48, var_50);
        var_53 = wp::add(var_22, var_52);
        var_54 = wp::mul(var_51, var_53);
        // ndof_cell = 3 * npc                                                                    <L 2285>
        var_56 = wp::mul(var_55, var_54);
        // cellnum = flex_cellnum[f]                                                              <L 2286>
        var_57 = wp::address(var_flex_cellnum, var_14);
        var_59 = wp::load(var_57);
        var_58 = wp::copy(var_59);
        // cy = cellnum[1]                                                                        <L 2287>
        var_61 = wp::extract(var_58, var_60);
        // cz = cellnum[2]                                                                        <L 2288>
        var_63 = wp::extract(var_58, var_62);
        // nstart = flex_nodeadr[f]                                                               <L 2289>
        var_64 = wp::address(var_flex_nodeadr, var_14);
        var_66 = wp::load(var_64);
        var_65 = wp::copy(var_66);
        // ny_g = cy * order + 1                                                                  <L 2290>
        var_67 = wp::mul(var_61, var_22);
        var_69 = wp::add(var_67, var_68);
        // nz_g = cz * order + 1                                                                  <L 2291>
        var_70 = wp::mul(var_63, var_22);
        var_72 = wp::add(var_70, var_71);
        // cell_idx = ci * cy * cz + cj * cz + ck                                                 <L 2293>
        var_73 = wp::mul(var_16, var_61);
        var_74 = wp::mul(var_73, var_63);
        var_75 = wp::mul(var_18, var_63);
        var_76 = wp::add(var_74, var_75);
        var_77 = wp::add(var_76, var_20);
        // k_base = stiffness_adr_base + cell_idx * ndof_cell * ndof_cell                         <L 2294>
        var_78 = wp::mul(var_77, var_56);
        var_79 = wp::mul(var_78, var_56);
        var_80 = wp::add(var_27, var_79);
        // if flex_stiffness[k_base] == 0.0:                                                      <L 2295>
        var_81 = wp::address(var_flex_stiffness, var_80);
        var_84 = wp::load(var_81);
        var_83 = (var_84 == var_82);
        if (var_83) {
            // return                                                                             <L 2296>
            return;
        }
        // cell_quat = support.compute_interp_cell_quat(flexnode_xpos_in, order, ci, cj, ck, cy, cz, ny_g, nz_g, nstart, worldid)       <L 2298>
        var_85 = compute_interp_cell_quat_0(var_flexnode_xpos_in, var_22, var_16, var_18, var_20, var_61, var_63, var_69, var_72, var_65, var_0);
        // R = wp.quat_to_matrix(cell_quat)                                                       <L 2299>
        var_86 = wp::quat_to_matrix(var_85);
        // RT = wp.transpose(R)                                                                   <L 2300>
        var_87 = wp::transpose(var_86);
        // for li_i in range(order + 1):                                                          <L 2302>
        var_89 = wp::add(var_22, var_88);
        var_90 = wp::range(var_89);
        start_for_4:;
            if (iter_cmp(var_90) == 0) goto end_for_4;
            var_91 = wp::iter_next(var_90);
            // for lj_i in range(order + 1):                                                      <L 2303>
            var_93 = wp::add(var_22, var_92);
            var_94 = wp::range(var_93);
            start_for_6:;
                if (iter_cmp(var_94) == 0) goto end_for_6;
                var_95 = wp::iter_next(var_94);
                // for lk_i in range(order + 1):                                                  <L 2304>
                var_97 = wp::add(var_22, var_96);
                var_98 = wp::range(var_97);
                start_for_8:;
                    if (iter_cmp(var_98) == 0) goto end_for_8;
                    var_99 = wp::iter_next(var_98);
                    // idx_i = (li_i * (order + 1) + lj_i) * (order + 1) + lk_i                   <L 2305>
                    var_101 = wp::add(var_22, var_100);
                    var_102 = wp::mul(var_91, var_101);
                    var_103 = wp::add(var_102, var_95);
                    var_105 = wp::add(var_22, var_104);
                    var_106 = wp::mul(var_103, var_105);
                    var_107 = wp::add(var_106, var_99);
                    // gidx_i = (ci * order + li_i) * ny_g * nz_g + (cj * order + lj_i) * nz_g + (ck * order + lk_i)       <L 2306>
                    var_108 = wp::mul(var_16, var_22);
                    var_109 = wp::add(var_108, var_91);
                    var_110 = wp::mul(var_109, var_69);
                    var_111 = wp::mul(var_110, var_72);
                    var_112 = wp::mul(var_18, var_22);
                    var_113 = wp::add(var_112, var_95);
                    var_114 = wp::mul(var_113, var_72);
                    var_115 = wp::add(var_111, var_114);
                    var_116 = wp::mul(var_20, var_22);
                    var_117 = wp::add(var_116, var_99);
                    var_118 = wp::add(var_115, var_117);
                    // bi = flex_nodebodyid[nstart + gidx_i]                                      <L 2307>
                    var_119 = wp::add(var_65, var_118);
                    var_120 = wp::address(var_flex_nodebodyid, var_119);
                    var_122 = wp::load(var_120);
                    var_121 = wp::copy(var_122);
                    // if body_dofnum[bi] != 3:                                                   <L 2308>
                    var_123 = wp::address(var_body_dofnum, var_121);
                    var_126 = wp::load(var_123);
                    var_125 = (var_126 != var_124);
                    if (var_125) {
                        // continue                                                               <L 2309>
                        goto start_for_8;
                    }
                    // dof_i = body_dofadr[bi]                                                    <L 2310>
                    var_127 = wp::address(var_body_dofadr, var_121);
                    var_129 = wp::load(var_127);
                    var_128 = wp::copy(var_129);
                    // T_i_T = wp.transpose(xmat_in[worldid, bi]) * R                             <L 2311>
                    var_130 = wp::address(var_xmat_in, var_0, var_121);
                    var_132 = wp::load(var_130);
                    var_131 = wp::transpose(var_132);
                    var_133 = wp::mul(var_131, var_86);
                    // for li_j in range(order + 1):                                              <L 2313>
                    var_135 = wp::add(var_22, var_134);
                    var_136 = wp::range(var_135);
                    start_for_10:;
                        if (iter_cmp(var_136) == 0) goto end_for_10;
                        var_137 = wp::iter_next(var_136);
                        // for lj_j in range(order + 1):                                          <L 2314>
                        var_139 = wp::add(var_22, var_138);
                        var_140 = wp::range(var_139);
                        start_for_12:;
                            if (iter_cmp(var_140) == 0) goto end_for_12;
                            var_141 = wp::iter_next(var_140);
                            // for lk_j in range(order + 1):                                      <L 2315>
                            var_143 = wp::add(var_22, var_142);
                            var_144 = wp::range(var_143);
                            start_for_14:;
                                if (iter_cmp(var_144) == 0) goto end_for_14;
                                var_145 = wp::iter_next(var_144);
                                // idx_j = (li_j * (order + 1) + lj_j) * (order + 1) + lk_j       <L 2316>
                                var_147 = wp::add(var_22, var_146);
                                var_148 = wp::mul(var_137, var_147);
                                var_149 = wp::add(var_148, var_141);
                                var_151 = wp::add(var_22, var_150);
                                var_152 = wp::mul(var_149, var_151);
                                var_153 = wp::add(var_152, var_145);
                                // gidx_j = (ci * order + li_j) * ny_g * nz_g + (cj * order + lj_j) * nz_g + (ck * order + lk_j)       <L 2317>
                                var_154 = wp::mul(var_16, var_22);
                                var_155 = wp::add(var_154, var_137);
                                var_156 = wp::mul(var_155, var_69);
                                var_157 = wp::mul(var_156, var_72);
                                var_158 = wp::mul(var_18, var_22);
                                var_159 = wp::add(var_158, var_141);
                                var_160 = wp::mul(var_159, var_72);
                                var_161 = wp::add(var_157, var_160);
                                var_162 = wp::mul(var_20, var_22);
                                var_163 = wp::add(var_162, var_145);
                                var_164 = wp::add(var_161, var_163);
                                // bj = flex_nodebodyid[nstart + gidx_j]                          <L 2318>
                                var_165 = wp::add(var_65, var_164);
                                var_166 = wp::address(var_flex_nodebodyid, var_165);
                                var_168 = wp::load(var_166);
                                var_167 = wp::copy(var_168);
                                // if body_dofnum[bj] != 3:                                       <L 2319>
                                var_169 = wp::address(var_body_dofnum, var_167);
                                var_172 = wp::load(var_169);
                                var_171 = (var_172 != var_170);
                                if (var_171) {
                                    // continue                                                   <L 2320>
                                    goto start_for_14;
                                }
                                // dof_j = body_dofadr[bj]                                        <L 2321>
                                var_173 = wp::address(var_body_dofadr, var_167);
                                var_175 = wp::load(var_173);
                                var_174 = wp::copy(var_175);
                                // T_j = RT * xmat_in[worldid, bj]                                <L 2322>
                                var_176 = wp::address(var_xmat_in, var_0, var_167);
                                var_178 = wp::load(var_176);
                                var_177 = wp::mul(var_87, var_178);
                                // K_ij = wp.mat33(0.0)                                           <L 2324>
                                var_180 = wp::mat_t<3, 3, wp::float32>(var_179);
                                // for r in range(3):                                             <L 2325>
                                // for c in range(3):                                             <L 2326>
                                // K_ij[r, c] = flex_stiffness[k_base + (3 * idx_i + r) * ndof_cell + (3 * idx_j + c)]       <L 2327>
                                var_184 = wp::mul(var_183, var_107);
                                var_185 = wp::add(var_184, var_181);
                                var_186 = wp::mul(var_185, var_56);
                                var_187 = wp::add(var_80, var_186);
                                var_189 = wp::mul(var_188, var_153);
                                var_190 = wp::add(var_189, var_182);
                                var_191 = wp::add(var_187, var_190);
                                var_192 = wp::address(var_flex_stiffness, var_191);
                                var_194 = wp::load(var_192);
                                var_193 = wp::copy(var_194);
                                wp::assign_inplace(var_180, var_181, var_182, var_193);
                                var_197 = wp::mul(var_196, var_107);
                                var_198 = wp::add(var_197, var_181);
                                var_199 = wp::mul(var_198, var_56);
                                var_200 = wp::add(var_80, var_199);
                                var_202 = wp::mul(var_201, var_153);
                                var_203 = wp::add(var_202, var_195);
                                var_204 = wp::add(var_200, var_203);
                                var_205 = wp::address(var_flex_stiffness, var_204);
                                var_207 = wp::load(var_205);
                                var_206 = wp::copy(var_207);
                                wp::assign_inplace(var_180, var_181, var_195, var_206);
                                var_210 = wp::mul(var_209, var_107);
                                var_211 = wp::add(var_210, var_181);
                                var_212 = wp::mul(var_211, var_56);
                                var_213 = wp::add(var_80, var_212);
                                var_215 = wp::mul(var_214, var_153);
                                var_216 = wp::add(var_215, var_208);
                                var_217 = wp::add(var_213, var_216);
                                var_218 = wp::address(var_flex_stiffness, var_217);
                                var_220 = wp::load(var_218);
                                var_219 = wp::copy(var_220);
                                wp::assign_inplace(var_180, var_181, var_208, var_219);
                                // for c in range(3):                                             <L 2326>
                                // K_ij[r, c] = flex_stiffness[k_base + (3 * idx_i + r) * ndof_cell + (3 * idx_j + c)]       <L 2327>
                                var_224 = wp::mul(var_223, var_107);
                                var_225 = wp::add(var_224, var_221);
                                var_226 = wp::mul(var_225, var_56);
                                var_227 = wp::add(var_80, var_226);
                                var_229 = wp::mul(var_228, var_153);
                                var_230 = wp::add(var_229, var_222);
                                var_231 = wp::add(var_227, var_230);
                                var_232 = wp::address(var_flex_stiffness, var_231);
                                var_234 = wp::load(var_232);
                                var_233 = wp::copy(var_234);
                                wp::assign_inplace(var_180, var_221, var_222, var_233);
                                var_237 = wp::mul(var_236, var_107);
                                var_238 = wp::add(var_237, var_221);
                                var_239 = wp::mul(var_238, var_56);
                                var_240 = wp::add(var_80, var_239);
                                var_242 = wp::mul(var_241, var_153);
                                var_243 = wp::add(var_242, var_235);
                                var_244 = wp::add(var_240, var_243);
                                var_245 = wp::address(var_flex_stiffness, var_244);
                                var_247 = wp::load(var_245);
                                var_246 = wp::copy(var_247);
                                wp::assign_inplace(var_180, var_221, var_235, var_246);
                                var_250 = wp::mul(var_249, var_107);
                                var_251 = wp::add(var_250, var_221);
                                var_252 = wp::mul(var_251, var_56);
                                var_253 = wp::add(var_80, var_252);
                                var_255 = wp::mul(var_254, var_153);
                                var_256 = wp::add(var_255, var_248);
                                var_257 = wp::add(var_253, var_256);
                                var_258 = wp::address(var_flex_stiffness, var_257);
                                var_260 = wp::load(var_258);
                                var_259 = wp::copy(var_260);
                                wp::assign_inplace(var_180, var_221, var_248, var_259);
                                // for c in range(3):                                             <L 2326>
                                // K_ij[r, c] = flex_stiffness[k_base + (3 * idx_i + r) * ndof_cell + (3 * idx_j + c)]       <L 2327>
                                var_264 = wp::mul(var_263, var_107);
                                var_265 = wp::add(var_264, var_261);
                                var_266 = wp::mul(var_265, var_56);
                                var_267 = wp::add(var_80, var_266);
                                var_269 = wp::mul(var_268, var_153);
                                var_270 = wp::add(var_269, var_262);
                                var_271 = wp::add(var_267, var_270);
                                var_272 = wp::address(var_flex_stiffness, var_271);
                                var_274 = wp::load(var_272);
                                var_273 = wp::copy(var_274);
                                wp::assign_inplace(var_180, var_261, var_262, var_273);
                                var_277 = wp::mul(var_276, var_107);
                                var_278 = wp::add(var_277, var_261);
                                var_279 = wp::mul(var_278, var_56);
                                var_280 = wp::add(var_80, var_279);
                                var_282 = wp::mul(var_281, var_153);
                                var_283 = wp::add(var_282, var_275);
                                var_284 = wp::add(var_280, var_283);
                                var_285 = wp::address(var_flex_stiffness, var_284);
                                var_287 = wp::load(var_285);
                                var_286 = wp::copy(var_287);
                                wp::assign_inplace(var_180, var_261, var_275, var_286);
                                var_290 = wp::mul(var_289, var_107);
                                var_291 = wp::add(var_290, var_261);
                                var_292 = wp::mul(var_291, var_56);
                                var_293 = wp::add(var_80, var_292);
                                var_295 = wp::mul(var_294, var_153);
                                var_296 = wp::add(var_295, var_288);
                                var_297 = wp::add(var_293, var_296);
                                var_298 = wp::address(var_flex_stiffness, var_297);
                                var_300 = wp::load(var_298);
                                var_299 = wp::copy(var_300);
                                wp::assign_inplace(var_180, var_261, var_288, var_299);
                                // blkd = iscale * (T_i_T * K_ij * T_j)                           <L 2329>
                                var_301 = wp::mul(var_133, var_180);
                                var_302 = wp::mul(var_301, var_177);
                                var_303 = wp::mul(var_44, var_302);
                                // for r in range(3):                                             <L 2330>
                                // row = dof_i + r                                                <L 2331>
                                var_305 = wp::add(var_128, var_304);
                                // rowadr = efm_K_rowadr[row]                                     <L 2332>
                                var_306 = wp::address(var_efm_K_rowadr, var_305);
                                var_308 = wp::load(var_306);
                                var_307 = wp::copy(var_308);
                                // rownnz = efm_K_rownnz[row]                                     <L 2333>
                                var_309 = wp::address(var_efm_K_rownnz, var_305);
                                var_311 = wp::load(var_309);
                                var_310 = wp::copy(var_311);
                                // for c in range(3):                                             <L 2334>
                                // col = dof_j + c                                                <L 2335>
                                var_313 = wp::add(var_174, var_312);
                                // idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)         <L 2336>
                                var_314 = _find_csr_col_0(var_307, var_310, var_313, var_efm_K_colind);
                                // if idx >= 0:                                                   <L 2337>
                                var_316 = (var_314 >= var_315);
                                if (var_316) {
                                    // wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])       <L 2338>
                                    var_317 = wp::extract(var_303, var_304, var_312);
                                    var_318 = wp::atomic_add(var_efm_K_val_out, var_0, var_314, var_317);
                                }
                                // col = dof_j + c                                                <L 2335>
                                var_320 = wp::add(var_174, var_319);
                                // idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)         <L 2336>
                                var_321 = _find_csr_col_0(var_307, var_310, var_320, var_efm_K_colind);
                                // if idx >= 0:                                                   <L 2337>
                                var_323 = (var_321 >= var_322);
                                if (var_323) {
                                    // wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])       <L 2338>
                                    var_324 = wp::extract(var_303, var_304, var_319);
                                    var_325 = wp::atomic_add(var_efm_K_val_out, var_0, var_321, var_324);
                                }
                                // col = dof_j + c                                                <L 2335>
                                var_327 = wp::add(var_174, var_326);
                                // idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)         <L 2336>
                                var_328 = _find_csr_col_0(var_307, var_310, var_327, var_efm_K_colind);
                                // if idx >= 0:                                                   <L 2337>
                                var_330 = (var_328 >= var_329);
                                if (var_330) {
                                    // wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])       <L 2338>
                                    var_331 = wp::extract(var_303, var_304, var_326);
                                    var_332 = wp::atomic_add(var_efm_K_val_out, var_0, var_328, var_331);
                                }
                                // row = dof_i + r                                                <L 2331>
                                var_334 = wp::add(var_128, var_333);
                                // rowadr = efm_K_rowadr[row]                                     <L 2332>
                                var_335 = wp::address(var_efm_K_rowadr, var_334);
                                var_337 = wp::load(var_335);
                                var_336 = wp::copy(var_337);
                                // rownnz = efm_K_rownnz[row]                                     <L 2333>
                                var_338 = wp::address(var_efm_K_rownnz, var_334);
                                var_340 = wp::load(var_338);
                                var_339 = wp::copy(var_340);
                                // for c in range(3):                                             <L 2334>
                                // col = dof_j + c                                                <L 2335>
                                var_342 = wp::add(var_174, var_341);
                                // idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)         <L 2336>
                                var_343 = _find_csr_col_0(var_336, var_339, var_342, var_efm_K_colind);
                                // if idx >= 0:                                                   <L 2337>
                                var_345 = (var_343 >= var_344);
                                if (var_345) {
                                    // wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])       <L 2338>
                                    var_346 = wp::extract(var_303, var_333, var_341);
                                    var_347 = wp::atomic_add(var_efm_K_val_out, var_0, var_343, var_346);
                                }
                                // col = dof_j + c                                                <L 2335>
                                var_349 = wp::add(var_174, var_348);
                                // idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)         <L 2336>
                                var_350 = _find_csr_col_0(var_336, var_339, var_349, var_efm_K_colind);
                                // if idx >= 0:                                                   <L 2337>
                                var_352 = (var_350 >= var_351);
                                if (var_352) {
                                    // wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])       <L 2338>
                                    var_353 = wp::extract(var_303, var_333, var_348);
                                    var_354 = wp::atomic_add(var_efm_K_val_out, var_0, var_350, var_353);
                                }
                                // col = dof_j + c                                                <L 2335>
                                var_356 = wp::add(var_174, var_355);
                                // idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)         <L 2336>
                                var_357 = _find_csr_col_0(var_336, var_339, var_356, var_efm_K_colind);
                                // if idx >= 0:                                                   <L 2337>
                                var_359 = (var_357 >= var_358);
                                if (var_359) {
                                    // wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])       <L 2338>
                                    var_360 = wp::extract(var_303, var_333, var_355);
                                    var_361 = wp::atomic_add(var_efm_K_val_out, var_0, var_357, var_360);
                                }
                                // row = dof_i + r                                                <L 2331>
                                var_363 = wp::add(var_128, var_362);
                                // rowadr = efm_K_rowadr[row]                                     <L 2332>
                                var_364 = wp::address(var_efm_K_rowadr, var_363);
                                var_366 = wp::load(var_364);
                                var_365 = wp::copy(var_366);
                                // rownnz = efm_K_rownnz[row]                                     <L 2333>
                                var_367 = wp::address(var_efm_K_rownnz, var_363);
                                var_369 = wp::load(var_367);
                                var_368 = wp::copy(var_369);
                                // for c in range(3):                                             <L 2334>
                                // col = dof_j + c                                                <L 2335>
                                var_371 = wp::add(var_174, var_370);
                                // idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)         <L 2336>
                                var_372 = _find_csr_col_0(var_365, var_368, var_371, var_efm_K_colind);
                                // if idx >= 0:                                                   <L 2337>
                                var_374 = (var_372 >= var_373);
                                if (var_374) {
                                    // wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])       <L 2338>
                                    var_375 = wp::extract(var_303, var_362, var_370);
                                    var_376 = wp::atomic_add(var_efm_K_val_out, var_0, var_372, var_375);
                                }
                                // col = dof_j + c                                                <L 2335>
                                var_378 = wp::add(var_174, var_377);
                                // idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)         <L 2336>
                                var_379 = _find_csr_col_0(var_365, var_368, var_378, var_efm_K_colind);
                                // if idx >= 0:                                                   <L 2337>
                                var_381 = (var_379 >= var_380);
                                if (var_381) {
                                    // wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])       <L 2338>
                                    var_382 = wp::extract(var_303, var_362, var_377);
                                    var_383 = wp::atomic_add(var_efm_K_val_out, var_0, var_379, var_382);
                                }
                                // col = dof_j + c                                                <L 2335>
                                var_385 = wp::add(var_174, var_384);
                                // idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)         <L 2336>
                                var_386 = _find_csr_col_0(var_365, var_368, var_385, var_efm_K_colind);
                                // if idx >= 0:                                                   <L 2337>
                                var_388 = (var_386 >= var_387);
                                if (var_388) {
                                    // wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])       <L 2338>
                                    var_389 = wp::extract(var_303, var_362, var_384);
                                    var_390 = wp::atomic_add(var_efm_K_val_out, var_0, var_386, var_389);
                                }
                                goto start_for_14;
                            end_for_14:;
                            goto start_for_12;
                        end_for_12:;
                        goto start_for_10;
                    end_for_10:;
                    goto start_for_8;
                end_for_8:;
                goto start_for_6;
            end_for_6:;
            goto start_for_4;
        end_for_4:;
}



extern "C" __global__ void _eff_flex_stretch_stiff_vert_40df6972_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_flex_dim,
    wp::array_t<wp::int32> var_flex_interp,
    wp::array_t<wp::int32> var_flex_stiffnessadr,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::float32> var_flex_stiffness,
    wp::array_t<wp::float32> var_flex_damping,
    wp::array_t<bool> var_flex_rigid,
    wp::array_t<wp::int32> var_efm_K_rownnz,
    wp::array_t<wp::int32> var_efm_K_rowadr,
    wp::array_t<wp::int32> var_efm_K_colind,
    wp::array_t<wp::int32> var_flex_vertflexid,
    wp::array_t<bool> var_flex_simple,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_flexvert_hessian_in,
    wp::array_t<wp::float32> var_efm_K_val_out)
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
        wp::float32 var_8;
        wp::float32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        wp::int32* var_16;
        const wp::int32 var_17 = 0;
        bool var_18;
        wp::int32 var_19;
        bool* var_20;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 2;
        bool var_24;
        wp::int32 var_25;
        bool* var_26;
        bool var_27;
        bool var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        bool var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::float32* var_35;
        const wp::float32 var_36 = 0.0;
        bool var_37;
        wp::float32 var_38;
        const wp::int32 var_39 = 32;
        wp::int32 var_40;
        const wp::float32 var_41 = 0.0;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 64;
        wp::int32 var_45;
        const wp::float32 var_46 = 0.0;
        wp::float32 var_47;
        wp::float32* var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.0;
        bool var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::int32* var_59;
        const wp::int32 var_60 = 3;
        bool var_61;
        wp::int32 var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32* var_66;
        const wp::int32 var_67 = 0;
        bool var_68;
        wp::int32 var_69;
        wp::vec_t<6, wp::float32>* var_70;
        wp::vec_t<6, wp::float32> var_71;
        wp::vec_t<6, wp::float32> var_72;
        const wp::int32 var_73 = 0;
        wp::float32 var_74;
        const wp::int32 var_75 = 1;
        wp::float32 var_76;
        const wp::int32 var_77 = 2;
        wp::float32 var_78;
        const wp::int32 var_79 = 1;
        wp::float32 var_80;
        const wp::int32 var_81 = 3;
        wp::float32 var_82;
        const wp::int32 var_83 = 4;
        wp::float32 var_84;
        const wp::int32 var_85 = 2;
        wp::float32 var_86;
        const wp::int32 var_87 = 4;
        wp::float32 var_88;
        const wp::int32 var_89 = 5;
        wp::float32 var_90;
        wp::mat_t<3, 3, wp::float32> var_91;
        wp::mat_t<3, 3, wp::float32>* var_92;
        wp::mat_t<3, 3, wp::float32> var_93;
        wp::mat_t<3, 3, wp::float32> var_94;
        wp::mat_t<3, 3, wp::float32> var_95;
        wp::mat_t<3, 3, wp::float32> var_96;
        wp::mat_t<3, 3, wp::float32> var_97;
        wp::mat_t<3, 3, wp::float32> var_98;
        const wp::int32 var_99 = 0;
        wp::int32 var_100;
        wp::int32* var_101;
        wp::int32* var_102;
        wp::int32 var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        const wp::int32 var_106 = 0;
        bool var_107;
        const wp::int32 var_108 = 0;
        wp::int32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        const wp::int32 var_112 = 1;
        wp::int32 var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        const wp::int32 var_116 = 2;
        wp::int32 var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::int32 var_120 = 1;
        wp::int32 var_121;
        wp::int32* var_122;
        wp::int32* var_123;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        const wp::int32 var_127 = 0;
        bool var_128;
        const wp::int32 var_129 = 0;
        wp::int32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        const wp::int32 var_133 = 1;
        wp::int32 var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        const wp::int32 var_137 = 2;
        wp::int32 var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::int32 var_141;
        const wp::int32 var_142 = 2;
        wp::int32 var_143;
        wp::int32* var_144;
        wp::int32* var_145;
        wp::int32 var_146;
        wp::int32 var_147;
        wp::int32 var_148;
        const wp::int32 var_149 = 0;
        bool var_150;
        const wp::int32 var_151 = 0;
        wp::int32 var_152;
        wp::float32 var_153;
        wp::float32 var_154;
        const wp::int32 var_155 = 1;
        wp::int32 var_156;
        wp::float32 var_157;
        wp::float32 var_158;
        const wp::int32 var_159 = 2;
        wp::int32 var_160;
        wp::float32 var_161;
        wp::float32 var_162;
        wp::int32 var_163;
        //---------
        // forward
        // def _eff_flex_stretch_stiff_vert(                                                      <L 1998>
        // worldid, vertid = wp.tid()                                                             <L 2023>
        builtin_tid2d(var_0, var_1);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 2024>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // f = flex_vertflexid[vertid]                                                            <L 2026>
        var_10 = wp::address(var_flex_vertflexid, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // if f < 0 or flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] < 2 or not flex_simple[f]:       <L 2027>
        var_15 = (var_11 < var_14);
        var_13 = var_15;
        if (!var_13) {
            var_16 = wp::address(var_flex_interp, var_11);
            var_19 = wp::load(var_16);
            var_18 = (var_19 != var_17);
            var_13 = var_13 || var_18;
        }
        if (!var_13) {
            var_20 = wp::address(var_flex_rigid, var_11);
            var_21 = wp::load(var_20);
            var_13 = var_13 || var_21;
        }
        if (!var_13) {
            var_22 = wp::address(var_flex_dim, var_11);
            var_25 = wp::load(var_22);
            var_24 = (var_25 < var_23);
            var_13 = var_13 || var_24;
        }
        if (!var_13) {
            var_26 = wp::address(var_flex_simple, var_11);
            var_28 = wp::load(var_26);
            var_27 = wp::unot(var_28);
            var_13 = var_13 || var_27;
        }
        if (var_13) {
            // return                                                                             <L 2028>
            return;
        }
        // stiffness_adr = flex_stiffnessadr[f]                                                   <L 2030>
        var_29 = wp::address(var_flex_stiffnessadr, var_11);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // if stiffness_adr < 0 or flex_stiffness[stiffness_adr] == 0.0:                          <L 2031>
        var_34 = (var_30 < var_33);
        var_32 = var_34;
        if (!var_32) {
            var_35 = wp::address(var_flex_stiffness, var_30);
            var_38 = wp::load(var_35);
            var_37 = (var_38 == var_36);
            var_32 = var_32 || var_37;
        }
        if (var_32) {
            // return                                                                             <L 2032>
            return;
        }
        // s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep            <L 2034>
        var_40 = wp::bit_and(var_opt_disableflags, var_39);
        if (var_40) {
        }
        if (!var_40) {
            var_42 = wp::mul(var_8, var_8);
        }
        var_43 = wp::where(var_40, var_41, var_42);
        // s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep                       <L 2035>
        var_45 = wp::bit_and(var_opt_disableflags, var_44);
        if (var_45) {
        }
        if (!var_45) {
        }
        var_47 = wp::where(var_45, var_46, var_8);
        // scale = s1 + s2 * flex_damping[f]                                                      <L 2036>
        var_48 = wp::address(var_flex_damping, var_11);
        var_50 = wp::load(var_48);
        var_49 = wp::mul(var_47, var_50);
        var_51 = wp::add(var_43, var_49);
        // if scale == 0.0:                                                                       <L 2037>
        var_53 = (var_51 == var_52);
        if (var_53) {
            // return                                                                             <L 2038>
            return;
        }
        // bi = body_weldid[flex_vertbodyid[vertid]]                                              <L 2040>
        var_54 = wp::address(var_flex_vertbodyid, var_1);
        var_55 = wp::load(var_54);
        var_56 = wp::address(var_body_weldid, var_55);
        var_58 = wp::load(var_56);
        var_57 = wp::copy(var_58);
        // if body_dofnum[bi] != 3:                                                               <L 2041>
        var_59 = wp::address(var_body_dofnum, var_57);
        var_62 = wp::load(var_59);
        var_61 = (var_62 != var_60);
        if (var_61) {
            // return                                                                             <L 2042>
            return;
        }
        // dof_i = body_dofadr[bi]                                                                <L 2043>
        var_63 = wp::address(var_body_dofadr, var_57);
        var_65 = wp::load(var_63);
        var_64 = wp::copy(var_65);
        // if efm_K_rownnz[dof_i] == 0:                                                           <L 2044>
        var_66 = wp::address(var_efm_K_rownnz, var_64);
        var_69 = wp::load(var_66);
        var_68 = (var_69 == var_67);
        if (var_68) {
            // return                                                                             <L 2045>
            return;
        }
        // h = flexvert_hessian_in[worldid, vertid]                                               <L 2047>
        var_70 = wp::address(var_flexvert_hessian_in, var_0, var_1);
        var_72 = wp::load(var_70);
        var_71 = wp::copy(var_72);
        // block = wp.mat33(h[0], h[1], h[2], h[1], h[3], h[4], h[2], h[4], h[5])                 <L 2048>
        var_74 = wp::extract(var_71, var_73);
        var_76 = wp::extract(var_71, var_75);
        var_78 = wp::extract(var_71, var_77);
        var_80 = wp::extract(var_71, var_79);
        var_82 = wp::extract(var_71, var_81);
        var_84 = wp::extract(var_71, var_83);
        var_86 = wp::extract(var_71, var_85);
        var_88 = wp::extract(var_71, var_87);
        var_90 = wp::extract(var_71, var_89);
        var_91 = wp::mat_t<3, 3, wp::float32>(var_74, var_76, var_78, var_80, var_82, var_84, var_86, var_88, var_90);
        // R_bi = xmat_in[worldid, bi]                                                            <L 2049>
        var_92 = wp::address(var_xmat_in, var_0, var_57);
        var_94 = wp::load(var_92);
        var_93 = wp::copy(var_94);
        // blkd = scale * (wp.transpose(R_bi) * block * R_bi)                                     <L 2050>
        var_95 = wp::transpose(var_93);
        var_96 = wp::mul(var_95, var_91);
        var_97 = wp::mul(var_96, var_93);
        var_98 = wp::mul(var_51, var_97);
        // for r in range(3):                                                                     <L 2052>
        // row = dof_i + r                                                                        <L 2053>
        var_100 = wp::add(var_64, var_99);
        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_i, efm_K_colind)         <L 2054>
        var_101 = wp::address(var_efm_K_rowadr, var_100);
        var_102 = wp::address(var_efm_K_rownnz, var_100);
        var_104 = wp::load(var_101);
        var_105 = wp::load(var_102);
        var_103 = _find_csr_col_0(var_104, var_105, var_64, var_efm_K_colind);
        // if idx >= 0:                                                                           <L 2055>
        var_107 = (var_103 >= var_106);
        if (var_107) {
            // for c in range(3):                                                                 <L 2056>
            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])                         <L 2057>
            var_109 = wp::add(var_103, var_108);
            var_110 = wp::extract(var_98, var_99, var_108);
            var_111 = wp::atomic_add(var_efm_K_val_out, var_0, var_109, var_110);
            var_113 = wp::add(var_103, var_112);
            var_114 = wp::extract(var_98, var_99, var_112);
            var_115 = wp::atomic_add(var_efm_K_val_out, var_0, var_113, var_114);
            var_117 = wp::add(var_103, var_116);
            var_118 = wp::extract(var_98, var_99, var_116);
            var_119 = wp::atomic_add(var_efm_K_val_out, var_0, var_117, var_118);
        }
        // row = dof_i + r                                                                        <L 2053>
        var_121 = wp::add(var_64, var_120);
        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_i, efm_K_colind)         <L 2054>
        var_122 = wp::address(var_efm_K_rowadr, var_121);
        var_123 = wp::address(var_efm_K_rownnz, var_121);
        var_125 = wp::load(var_122);
        var_126 = wp::load(var_123);
        var_124 = _find_csr_col_0(var_125, var_126, var_64, var_efm_K_colind);
        // if idx >= 0:                                                                           <L 2055>
        var_128 = (var_124 >= var_127);
        if (var_128) {
            // for c in range(3):                                                                 <L 2056>
            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])                         <L 2057>
            var_130 = wp::add(var_124, var_129);
            var_131 = wp::extract(var_98, var_120, var_129);
            var_132 = wp::atomic_add(var_efm_K_val_out, var_0, var_130, var_131);
            var_134 = wp::add(var_124, var_133);
            var_135 = wp::extract(var_98, var_120, var_133);
            var_136 = wp::atomic_add(var_efm_K_val_out, var_0, var_134, var_135);
            var_138 = wp::add(var_124, var_137);
            var_139 = wp::extract(var_98, var_120, var_137);
            var_140 = wp::atomic_add(var_efm_K_val_out, var_0, var_138, var_139);
        }
        var_141 = wp::where(var_128, var_137, var_116);
        // row = dof_i + r                                                                        <L 2053>
        var_143 = wp::add(var_64, var_142);
        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_i, efm_K_colind)         <L 2054>
        var_144 = wp::address(var_efm_K_rowadr, var_143);
        var_145 = wp::address(var_efm_K_rownnz, var_143);
        var_147 = wp::load(var_144);
        var_148 = wp::load(var_145);
        var_146 = _find_csr_col_0(var_147, var_148, var_64, var_efm_K_colind);
        // if idx >= 0:                                                                           <L 2055>
        var_150 = (var_146 >= var_149);
        if (var_150) {
            // for c in range(3):                                                                 <L 2056>
            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])                         <L 2057>
            var_152 = wp::add(var_146, var_151);
            var_153 = wp::extract(var_98, var_142, var_151);
            var_154 = wp::atomic_add(var_efm_K_val_out, var_0, var_152, var_153);
            var_156 = wp::add(var_146, var_155);
            var_157 = wp::extract(var_98, var_142, var_155);
            var_158 = wp::atomic_add(var_efm_K_val_out, var_0, var_156, var_157);
            var_160 = wp::add(var_146, var_159);
            var_161 = wp::extract(var_98, var_142, var_159);
            var_162 = wp::atomic_add(var_efm_K_val_out, var_0, var_160, var_161);
        }
        var_163 = wp::where(var_150, var_159, var_141);
}



extern "C" __global__ void _eff_fold_contact_rank1_41b2c0de_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_efm_dofid,
    wp::array_t<wp::int32> var_efm_dofblk,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_nacon_in,
    wp::array_t<wp::int32> var_efm_con_dof_in,
    wp::array_t<wp::float32> var_efm_con_val_in,
    wp::array_t<wp::float32> var_efm_con_scale_in,
    wp::array_t<wp::int32> var_efm_con_nnz_in,
    wp::array_t<wp::float32> var_epB_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::float32* var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        const wp::float32 var_8 = 0.0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::range_t var_16;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::range_t var_32;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        bool var_38;
        wp::int32 var_39;
        wp::float32* var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        const wp::int32 var_43 = 9;
        wp::int32 var_44;
        const wp::int32 var_45 = 3;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        //---------
        // forward
        // def _eff_fold_contact_rank1(                                                           <L 3695>
        // cid = wp.tid()                                                                         <L 3710>
        var_0 = builtin_tid1d();
        // if cid >= nacon_in[0]:                                                                 <L 3711>
        var_2 = wp::address(var_nacon_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 3712>
            return;
        }
        // scale = efm_con_scale_in[cid]                                                          <L 3713>
        var_5 = wp::address(var_efm_con_scale_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if scale == 0.0:                                                                       <L 3714>
        var_9 = (var_6 == var_8);
        if (var_9) {
            // return                                                                             <L 3715>
            return;
        }
        // worldid = contact_worldid_in[cid]                                                      <L 3716>
        var_10 = wp::address(var_contact_worldid_in, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // nnz = efm_con_nnz_in[cid]                                                              <L 3717>
        var_13 = wp::address(var_efm_con_nnz_in, var_0);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // for a in range(nnz):                                                                   <L 3718>
        var_16 = wp::range(var_14);
        start_for_2:;
            if (iter_cmp(var_16) == 0) goto end_for_2;
            var_17 = wp::iter_next(var_16);
            // ia = efm_con_dof_in[cid, a]                                                        <L 3719>
            var_18 = wp::address(var_efm_con_dof_in, var_0, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // k = efm_dofblk[ia]                                                                 <L 3720>
            var_21 = wp::address(var_efm_dofblk, var_19);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // if k < 0:                                                                          <L 3721>
            var_25 = (var_22 < var_24);
            if (var_25) {
                // continue                                                                       <L 3722>
                goto start_for_2;
            }
            // base = efm_dofid[k]                                                                <L 3723>
            var_26 = wp::address(var_efm_dofid, var_22);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // scale_va = scale * efm_con_val_in[cid, a]                                          <L 3724>
            var_29 = wp::address(var_efm_con_val_in, var_0, var_17);
            var_31 = wp::load(var_29);
            var_30 = wp::mul(var_6, var_31);
            // for b in range(nnz):                                                               <L 3725>
            var_32 = wp::range(var_14);
            start_for_4:;
                if (iter_cmp(var_32) == 0) goto end_for_4;
                var_33 = wp::iter_next(var_32);
                // ib = efm_con_dof_in[cid, b]                                                    <L 3726>
                var_34 = wp::address(var_efm_con_dof_in, var_0, var_33);
                var_36 = wp::load(var_34);
                var_35 = wp::copy(var_36);
                // if efm_dofblk[ib] == k:                                                        <L 3727>
                var_37 = wp::address(var_efm_dofblk, var_35);
                var_39 = wp::load(var_37);
                var_38 = (var_39 == var_22);
                if (var_38) {
                    // vb = efm_con_val_in[cid, b]                                                <L 3728>
                    var_40 = wp::address(var_efm_con_val_in, var_0, var_33);
                    var_42 = wp::load(var_40);
                    var_41 = wp::copy(var_42);
                    // wp.atomic_add(epB_out, worldid, 9 * k + 3 * (ia - base) + (ib - base), scale_va * vb)       <L 3729>
                    var_44 = wp::mul(var_43, var_22);
                    var_46 = wp::sub(var_19, var_27);
                    var_47 = wp::mul(var_45, var_46);
                    var_48 = wp::add(var_44, var_47);
                    var_49 = wp::sub(var_35, var_27);
                    var_50 = wp::add(var_48, var_49);
                    var_51 = wp::mul(var_30, var_41);
                    var_52 = wp::atomic_add(var_epB_out, var_11, var_50, var_51);
                }
                goto start_for_4;
            end_for_4:;
            goto start_for_2;
        end_for_2:;
}



extern "C" __global__ void _eff_diag_damp_shift_44d6f8a3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::float32> var_dof_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_dof_dampingpoly,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_efm_c_out,
    wp::array_t<wp::float32> var_efm_diag_out)
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
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        wp::float32 var_11;
        const wp::int32 var_12 = 64;
        wp::int32 var_13;
        bool var_14;
        wp::shape_t* var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::shape_t var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::shape_t* var_23;
        const wp::int32 var_24 = 0;
        wp::int32 var_25;
        wp::shape_t var_26;
        wp::int32 var_27;
        wp::vec_t<2, wp::float32>* var_28;
        wp::vec_t<2, wp::float32> var_29;
        wp::vec_t<2, wp::float32> var_30;
        wp::float32* var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::float32 var_34 = 0.0;
        const wp::int32 var_35 = 1;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32* var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        //---------
        // forward
        // def _eff_diag_damp_shift(                                                              <L 1440>
        // worldid, dofid = wp.tid()                                                              <L 1452>
        builtin_tid2d(var_0, var_1);
        // h = opt_timestep[worldid % opt_timestep.shape[0]]                                      <L 1453>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // damp_deriv = float(0.0)                                                                <L 1455>
        var_11 = wp::float(var_10);
        // if not (opt_disableflags & DisableBit.DAMPER):                                         <L 1456>
        var_13 = wp::bit_and(var_opt_disableflags, var_12);
        var_14 = wp::unot(var_13);
        if (var_14) {
            // damping = dof_damping[worldid % dof_damping.shape[0], dofid]                       <L 1457>
            var_15 = &(var_dof_damping.shape);
            var_18 = wp::load(var_15);
            var_17 = wp::extract(var_18, var_16);
            var_19 = wp::mod(var_0, var_17);
            var_20 = wp::address(var_dof_damping, var_19, var_1);
            var_22 = wp::load(var_20);
            var_21 = wp::copy(var_22);
            // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofid]                 <L 1458>
            var_23 = &(var_dof_dampingpoly.shape);
            var_26 = wp::load(var_23);
            var_25 = wp::extract(var_26, var_24);
            var_27 = wp::mod(var_0, var_25);
            var_28 = wp::address(var_dof_dampingpoly, var_27, var_1);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // v = qvel_in[worldid, dofid]                                                        <L 1459>
            var_31 = wp::address(var_qvel_in, var_0, var_1);
            var_33 = wp::load(var_31);
            var_32 = wp::copy(var_33);
            // damp_deriv = wp.max(0.0, util_misc._poly_force_deriv(damping, dpoly, v, 1))        <L 1460>
            var_36 = _poly_force_deriv_0(var_21, var_29, var_32, var_35);
            var_37 = wp::max(var_34, var_36);
        }
        var_38 = wp::where(var_14, var_37, var_11);
        // ck = efm_diag_out[worldid, dofid]                                                      <L 1462>
        var_39 = wp::address(var_efm_diag_out, var_0, var_1);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // efm_diag_out[worldid, dofid] = h * damp_deriv + h * ck                                 <L 1463>
        var_42 = wp::mul(var_8, var_38);
        var_43 = wp::mul(var_8, var_40);
        var_44 = wp::add(var_42, var_43);
        wp::array_store(var_efm_diag_out, var_0, var_1, var_44);
        // efm_c_out[worldid, dofid] = -ck * qvel_in[worldid, dofid]                              <L 1464>
        var_45 = wp::neg(var_40);
        var_46 = wp::address(var_qvel_in, var_0, var_1);
        var_48 = wp::load(var_46);
        var_47 = wp::mul(var_45, var_48);
        wp::array_store(var_efm_c_out, var_0, var_1, var_47);
}



extern "C" __global__ void _eff_block_solve_46d80904_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_efm_dofid,
    wp::array_t<wp::float32> var_efm_L_in,
    wp::array_t<wp::float32> var_rhs_in,
    wp::array_t<wp::float32> var_res_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 9;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        const wp::int32 var_12 = 3;
        wp::int32 var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::int32 var_17 = 4;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 6;
        wp::int32 var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::int32 var_27 = 7;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 8;
        wp::int32 var_33;
        wp::float32* var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 2;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        const wp::int32 var_70 = 2;
        wp::int32 var_71;
        //---------
        // forward
        // def _eff_block_solve(                                                                  <L 2433>
        // worldid, k = wp.tid()                                                                  <L 2443>
        builtin_tid2d(var_0, var_1);
        // i = efm_dofid[k]                                                                       <L 2444>
        var_2 = wp::address(var_efm_dofid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // base = 9 * k                                                                           <L 2445>
        var_6 = wp::mul(var_5, var_1);
        // L00 = efm_L_in[worldid, base + 0]                                                      <L 2446>
        var_8 = wp::add(var_6, var_7);
        var_9 = wp::address(var_efm_L_in, var_0, var_8);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // L10 = efm_L_in[worldid, base + 3]                                                      <L 2447>
        var_13 = wp::add(var_6, var_12);
        var_14 = wp::address(var_efm_L_in, var_0, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // L11 = efm_L_in[worldid, base + 4]                                                      <L 2448>
        var_18 = wp::add(var_6, var_17);
        var_19 = wp::address(var_efm_L_in, var_0, var_18);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // L20 = efm_L_in[worldid, base + 6]                                                      <L 2449>
        var_23 = wp::add(var_6, var_22);
        var_24 = wp::address(var_efm_L_in, var_0, var_23);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // L21 = efm_L_in[worldid, base + 7]                                                      <L 2450>
        var_28 = wp::add(var_6, var_27);
        var_29 = wp::address(var_efm_L_in, var_0, var_28);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // L22 = efm_L_in[worldid, base + 8]                                                      <L 2451>
        var_33 = wp::add(var_6, var_32);
        var_34 = wp::address(var_efm_L_in, var_0, var_33);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // b0 = rhs_in[worldid, i]                                                                <L 2453>
        var_37 = wp::address(var_rhs_in, var_0, var_3);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // b1 = rhs_in[worldid, i + 1]                                                            <L 2454>
        var_41 = wp::add(var_3, var_40);
        var_42 = wp::address(var_rhs_in, var_0, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // b2 = rhs_in[worldid, i + 2]                                                            <L 2455>
        var_46 = wp::add(var_3, var_45);
        var_47 = wp::address(var_rhs_in, var_0, var_46);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // r0 = b0 / L00                                                                          <L 2457>
        var_50 = wp::div(var_38, var_10);
        // r1 = (b1 - L10 * r0) / L11                                                             <L 2458>
        var_51 = wp::mul(var_15, var_50);
        var_52 = wp::sub(var_43, var_51);
        var_53 = wp::div(var_52, var_20);
        // r2 = (b2 - (L20 * r0 + L21 * r1)) / L22                                                <L 2459>
        var_54 = wp::mul(var_25, var_50);
        var_55 = wp::mul(var_30, var_53);
        var_56 = wp::add(var_54, var_55);
        var_57 = wp::sub(var_48, var_56);
        var_58 = wp::div(var_57, var_35);
        // x2 = r2 / L22                                                                          <L 2460>
        var_59 = wp::div(var_58, var_35);
        // x1 = (r1 - L21 * x2) / L11                                                             <L 2461>
        var_60 = wp::mul(var_30, var_59);
        var_61 = wp::sub(var_53, var_60);
        var_62 = wp::div(var_61, var_20);
        // x0 = (r0 - L10 * x1 - L20 * x2) / L00                                                  <L 2462>
        var_63 = wp::mul(var_15, var_62);
        var_64 = wp::sub(var_50, var_63);
        var_65 = wp::mul(var_25, var_59);
        var_66 = wp::sub(var_64, var_65);
        var_67 = wp::div(var_66, var_10);
        // res_out[worldid, i] = x0                                                               <L 2464>
        wp::array_store(var_res_out, var_0, var_3, var_67);
        // res_out[worldid, i + 1] = x1                                                           <L 2465>
        var_69 = wp::add(var_3, var_68);
        wp::array_store(var_res_out, var_0, var_69, var_62);
        // res_out[worldid, i + 2] = x2                                                           <L 2466>
        var_71 = wp::add(var_3, var_70);
        wp::array_store(var_res_out, var_0, var_71, var_59);
}



extern "C" __global__ void _eff_prec_efm0_solve_587f942c_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::int32 var_nefm0dof,
    wp::array_t<wp::int32> var_efm0_dofid,
    wp::array_t<wp::int32> var_efm0_L_rownnz,
    wp::array_t<wp::int32> var_efm0_L_rowadr,
    wp::array_t<wp::int32> var_efm0_L_colind,
    wp::array_t<wp::float32> var_efm0_L,
    wp::array_t<wp::float32> var_vec_in,
    wp::array_t<wp::float32> var_res_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::range_t var_1;
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        const wp::int32 var_9 = 1;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::float32* var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::float32 var_30 = 0.0;
        bool var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::range_t var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::float32* var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::range_t var_49;
        wp::int32 var_50;
        wp::int32* var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        wp::int32* var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::float32* var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        const wp::int32 var_66 = 1;
        wp::int32 var_67;
        wp::range_t var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32* var_71;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::int32* var_76;
        wp::int32 var_77;
        wp::float32* var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::float32* var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        //---------
        // forward
        // def _eff_prec_efm0_solve(                                                              <L 2485>
        // worldid = wp.tid()                                                                     <L 2498>
        var_0 = builtin_tid1d();
        // for r in range(nefm0dof):                                                              <L 2499>
        var_1 = wp::range(var_nefm0dof);
        start_for_0:;
            if (iter_cmp(var_1) == 0) goto end_for_0;
            var_2 = wp::iter_next(var_1);
            // dof_r = efm0_dofid[r]                                                              <L 2500>
            var_3 = wp::address(var_efm0_dofid, var_2);
            var_5 = wp::load(var_3);
            var_4 = wp::copy(var_5);
            // res_out[worldid, dof_r] = vec_in[worldid, dof_r]                                   <L 2501>
            var_6 = wp::address(var_vec_in, var_0, var_4);
            var_8 = wp::load(var_6);
            var_7 = wp::copy(var_8);
            wp::array_store(var_res_out, var_0, var_4, var_7);
            goto start_for_0;
        end_for_0:;
        // i = nefm0dof - 1                                                                       <L 2503>
        var_10 = wp::sub(var_nefm0dof, var_9);
        // while i >= 0:                                                                          <L 2504>
        start_while_2:;
        var_12 = (var_10 >= var_11);
        if ((var_12) == false) goto end_while_2;
            // adr = efm0_L_rowadr[i]                                                             <L 2505>
            var_13 = wp::address(var_efm0_L_rowadr, var_10);
            var_15 = wp::load(var_13);
            var_14 = wp::copy(var_15);
            // nnz = efm0_L_rownnz[i]                                                             <L 2506>
            var_16 = wp::address(var_efm0_L_rownnz, var_10);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // diag_idx = adr + nnz - 1                                                           <L 2507>
            var_19 = wp::add(var_14, var_17);
            var_21 = wp::sub(var_19, var_20);
            // dof_i = efm0_dofid[i]                                                              <L 2508>
            var_22 = wp::address(var_efm0_dofid, var_10);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // val_i = res_out[worldid, dof_i] / efm0_L[diag_idx]                                 <L 2509>
            var_25 = wp::address(var_res_out, var_0, var_23);
            var_26 = wp::address(var_efm0_L, var_21);
            var_28 = wp::load(var_25);
            var_29 = wp::load(var_26);
            var_27 = wp::div(var_28, var_29);
            // res_out[worldid, dof_i] = val_i                                                    <L 2510>
            wp::array_store(var_res_out, var_0, var_23, var_27);
            // if val_i != 0.0:                                                                   <L 2511>
            var_31 = (var_27 != var_30);
            if (var_31) {
                // for j in range(nnz - 1):                                                       <L 2512>
                var_33 = wp::sub(var_17, var_32);
                var_34 = wp::range(var_33);
                start_for_4:;
                    if (iter_cmp(var_34) == 0) goto end_for_4;
                    var_35 = wp::iter_next(var_34);
                    // c = efm0_L_colind[adr + j]                                                 <L 2513>
                    var_36 = wp::add(var_14, var_35);
                    var_37 = wp::address(var_efm0_L_colind, var_36);
                    var_39 = wp::load(var_37);
                    var_38 = wp::copy(var_39);
                    // res_out[worldid, efm0_dofid[c]] -= efm0_L[adr + j] * val_i                 <L 2514>
                    var_40 = wp::address(var_efm0_dofid, var_38);
                    var_41 = wp::add(var_14, var_35);
                    var_42 = wp::address(var_efm0_L, var_41);
                    var_44 = wp::load(var_42);
                    var_43 = wp::mul(var_44, var_27);
                    var_46 = wp::load(var_40);
                    var_45 = wp::atomic_sub(var_res_out, var_0, var_46, var_43);
                    goto start_for_4;
                end_for_4:;
            }
            // i -= 1                                                                             <L 2515>
            var_48 = wp::sub(var_10, var_47);
            wp::assign(var_10, var_48);
        goto start_while_2;
        end_while_2:;
        // for i in range(nefm0dof):                                                              <L 2517>
        var_49 = wp::range(var_nefm0dof);
        start_for_6:;
            if (iter_cmp(var_49) == 0) goto end_for_6;
            var_50 = wp::iter_next(var_49);
            // adr = efm0_L_rowadr[i]                                                             <L 2518>
            var_51 = wp::address(var_efm0_L_rowadr, var_50);
            var_53 = wp::load(var_51);
            var_52 = wp::copy(var_53);
            // nnz = efm0_L_rownnz[i]                                                             <L 2519>
            var_54 = wp::address(var_efm0_L_rownnz, var_50);
            var_56 = wp::load(var_54);
            var_55 = wp::copy(var_56);
            // diag_idx = adr + nnz - 1                                                           <L 2520>
            var_57 = wp::add(var_52, var_55);
            var_59 = wp::sub(var_57, var_58);
            // dof_i = efm0_dofid[i]                                                              <L 2521>
            var_60 = wp::address(var_efm0_dofid, var_50);
            var_62 = wp::load(var_60);
            var_61 = wp::copy(var_62);
            // acc = res_out[worldid, dof_i]                                                      <L 2522>
            var_63 = wp::address(var_res_out, var_0, var_61);
            var_65 = wp::load(var_63);
            var_64 = wp::copy(var_65);
            // for j in range(nnz - 1):                                                           <L 2523>
            var_67 = wp::sub(var_55, var_66);
            var_68 = wp::range(var_67);
            start_for_8:;
                if (iter_cmp(var_68) == 0) goto end_for_8;
                var_69 = wp::iter_next(var_68);
                // c = efm0_L_colind[adr + j]                                                     <L 2524>
                var_70 = wp::add(var_52, var_69);
                var_71 = wp::address(var_efm0_L_colind, var_70);
                var_73 = wp::load(var_71);
                var_72 = wp::copy(var_73);
                // acc -= efm0_L[adr + j] * res_out[worldid, efm0_dofid[c]]                       <L 2525>
                var_74 = wp::add(var_52, var_69);
                var_75 = wp::address(var_efm0_L, var_74);
                var_76 = wp::address(var_efm0_dofid, var_72);
                var_77 = wp::load(var_76);
                var_78 = wp::address(var_res_out, var_0, var_77);
                var_80 = wp::load(var_75);
                var_81 = wp::load(var_78);
                var_79 = wp::mul(var_80, var_81);
                var_82 = wp::sub(var_64, var_79);
                wp::assign(var_38, var_72);
                wp::assign(var_64, var_82);
                goto start_for_8;
            end_for_8:;
            // res_out[worldid, dof_i] = acc / efm0_L[diag_idx]                                   <L 2526>
            var_83 = wp::address(var_efm0_L, var_59);
            var_85 = wp::load(var_83);
            var_84 = wp::div(var_64, var_85);
            wp::array_store(var_res_out, var_0, var_61, var_84);
            wp::assign(var_14, var_52);
            wp::assign(var_17, var_55);
            wp::assign(var_21, var_59);
            wp::assign(var_23, var_61);
            wp::assign(var_35, var_69);
            goto start_for_6;
        end_for_6:;
}



extern "C" __global__ void _qderiv_ellipsoid_fluid_5899939f_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::vec_t<3, wp::float32>> var_opt_wind,
    wp::array_t<wp::float32> var_opt_density,
    wp::array_t<wp::float32> var_opt_viscosity,
    wp::int32 var_opt_integrator,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_rootid,
    wp::array_t<wp::int32> var_body_geomnum,
    wp::array_t<wp::int32> var_body_geomadr,
    wp::array_t<wp::int32> var_dof_bodyid,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::float32> var_geom_fluid,
    wp::array_t<wp::int32> var_body_fluid_ellipsoid_adr,
    wp::array_t<wp::int32> var_body_isdofancestor,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_xipos_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::vec_t<3, wp::float32>> var_subtree_com_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::int32> var_Mi,
    wp::array_t<wp::int32> var_Mj,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32 var_25;
        wp::shape_t* var_26;
        const wp::int32 var_27 = 0;
        wp::int32 var_28;
        wp::shape_t var_29;
        wp::int32 var_30;
        wp::vec_t<3, wp::float32>* var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::shape_t* var_34;
        const wp::int32 var_35 = 0;
        wp::int32 var_36;
        wp::shape_t var_37;
        wp::int32 var_38;
        wp::float32* var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::shape_t* var_42;
        const wp::int32 var_43 = 0;
        wp::int32 var_44;
        wp::shape_t var_45;
        wp::int32 var_46;
        wp::float32* var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::shape_t* var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::shape_t var_53;
        wp::int32 var_54;
        wp::float32* var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        bool var_58;
        const wp::float32 var_59 = 0.0;
        bool var_60;
        const wp::float32 var_61 = 0.0;
        bool var_62;
        wp::vec_t<6, wp::float32>* var_63;
        wp::vec_t<6, wp::float32> var_64;
        wp::vec_t<6, wp::float32> var_65;
        wp::vec_t<6, wp::float32>* var_66;
        wp::vec_t<6, wp::float32> var_67;
        wp::vec_t<6, wp::float32> var_68;
        wp::int32* var_69;
        wp::int32* var_70;
        wp::int32* var_71;
        wp::float32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        wp::float32 var_76;
        const wp::float32 var_77 = 0.0;
        bool var_78;
        wp::array_t<wp::float32> var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        //---------
        // forward
        // def _qderiv_ellipsoid_fluid(                                                           <L 899>
        // worldid, fluid_idx, elemid = wp.tid()                                                  <L 936>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid = body_fluid_ellipsoid_adr[fluid_idx]                                           <L 938>
        var_3 = wp::address(var_body_fluid_ellipsoid_adr, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // dofiid = Mi[elemid]                                                                    <L 940>
        var_6 = wp::address(var_Mi, var_2);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // dofjid = Mj[elemid]                                                                    <L 941>
        var_9 = wp::address(var_Mj, var_2);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // madr = M_elemid[dofiid, dofjid]                                                        <L 943>
        var_12 = wp::address(var_M_elemid, var_7, var_10);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if madr < 0:                                                                           <L 944>
        var_16 = (var_13 < var_15);
        if (var_16) {
            // return                                                                             <L 945>
            return;
        }
        // bodyid_i = dof_bodyid[dofiid]                                                          <L 949>
        var_17 = wp::address(var_dof_bodyid, var_7);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // if bodyid_i == 0:                                                                      <L 951>
        var_21 = (var_18 == var_20);
        if (var_21) {
            // return                                                                             <L 952>
            return;
        }
        // if body_isdofancestor[bodyid, dofiid] == 0:                                            <L 954>
        var_22 = wp::address(var_body_isdofancestor, var_4, var_7);
        var_25 = wp::load(var_22);
        var_24 = (var_25 == var_23);
        if (var_24) {
            // return                                                                             <L 955>
            return;
        }
        // wind = opt_wind[worldid % opt_wind.shape[0]]                                           <L 957>
        var_26 = &(var_opt_wind.shape);
        var_29 = wp::load(var_26);
        var_28 = wp::extract(var_29, var_27);
        var_30 = wp::mod(var_0, var_28);
        var_31 = wp::address(var_opt_wind, var_30);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // density = opt_density[worldid % opt_density.shape[0]]                                  <L 958>
        var_34 = &(var_opt_density.shape);
        var_37 = wp::load(var_34);
        var_36 = wp::extract(var_37, var_35);
        var_38 = wp::mod(var_0, var_36);
        var_39 = wp::address(var_opt_density, var_38);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]                            <L 959>
        var_42 = &(var_opt_viscosity.shape);
        var_45 = wp::load(var_42);
        var_44 = wp::extract(var_45, var_43);
        var_46 = wp::mod(var_0, var_44);
        var_47 = wp::address(var_opt_viscosity, var_46);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 960>
        var_50 = &(var_opt_timestep.shape);
        var_53 = wp::load(var_50);
        var_52 = wp::extract(var_53, var_51);
        var_54 = wp::mod(var_0, var_52);
        var_55 = wp::address(var_opt_timestep, var_54);
        var_57 = wp::load(var_55);
        var_56 = wp::copy(var_57);
        // if density <= 0.0 and viscosity <= 0.0:                                                <L 962>
        var_60 = (var_40 <= var_59);
        var_58 = var_60;
        if (var_58) {
            var_62 = (var_48 <= var_61);
            var_58 = var_58 && var_62;
        }
        if (var_58) {
            // return                                                                             <L 963>
            return;
        }
        // cdof_i = cdof_in[worldid, dofiid]                                                      <L 965>
        var_63 = wp::address(var_cdof_in, var_0, var_7);
        var_65 = wp::load(var_63);
        var_64 = wp::copy(var_65);
        // cdof_j = cdof_in[worldid, dofjid]                                                      <L 966>
        var_66 = wp::address(var_cdof_in, var_0, var_10);
        var_68 = wp::load(var_66);
        var_67 = wp::copy(var_68);
        // contrib = _deriv_ellipsoid_fluid(                                                      <L 968>
        // opt_integrator,                                                                        <L 969>
        // geom_type,                                                                             <L 970>
        // geom_size,                                                                             <L 971>
        // geom_fluid,                                                                            <L 972>
        // xipos_in,                                                                              <L 973>
        // geom_xpos_in,                                                                          <L 974>
        // geom_xmat_in,                                                                          <L 975>
        // subtree_com_in,                                                                        <L 976>
        // cvel_in,                                                                               <L 977>
        // worldid,                                                                               <L 978>
        // bodyid,                                                                                <L 979>
        // body_rootid[bodyid],                                                                   <L 980>
        var_69 = wp::address(var_body_rootid, var_4);
        // body_geomadr[bodyid],                                                                  <L 981>
        var_70 = wp::address(var_body_geomadr, var_4);
        // body_geomnum[bodyid],                                                                  <L 982>
        var_71 = wp::address(var_body_geomnum, var_4);
        // cdof_i,                                                                                <L 983>
        // cdof_j,                                                                                <L 984>
        // wind,                                                                                  <L 985>
        // density,                                                                               <L 986>
        // viscosity,                                                                             <L 987>
        var_73 = wp::load(var_69);
        var_74 = wp::load(var_70);
        var_75 = wp::load(var_71);
        var_72 = _deriv_ellipsoid_fluid_0(var_opt_integrator, var_geom_type, var_geom_size, var_geom_fluid, var_xipos_in, var_geom_xpos_in, var_geom_xmat_in, var_subtree_com_in, var_cvel_in, var_0, var_4, var_73, var_74, var_75, var_64, var_67, var_32, var_40, var_48);
        // contrib *= timestep                                                                    <L 990>
        var_76 = wp::mul(var_72, var_56);
        // if contrib != 0.0:                                                                     <L 992>
        var_78 = (var_76 != var_77);
        if (var_78) {
            // wp.atomic_add(qDeriv_out[worldid], madr, -contrib)                                 <L 993>
            var_79 = wp::view(var_qDeriv_out, var_0);
            var_80 = wp::neg(var_76);
            var_81 = wp::atomic_add(var_79, var_13, var_80);
        }
}



extern "C" __global__ void _eff_add_fluid_qH_5fd5937a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_efm_fluid_in,
    wp::array_t<wp::float32> var_qH_out)
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
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        //---------
        // forward
        // def _eff_add_fluid_qH(                                                                 <L 1783>
        // worldid, madr = wp.tid()                                                               <L 1789>
        builtin_tid2d(var_0, var_1);
        // qH_out[worldid, madr] += efm_fluid_in[worldid, madr]                                   <L 1790>
        var_2 = wp::address(var_efm_fluid_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::atomic_add(var_qH_out, var_0, var_1, var_4);
}



extern "C" __global__ void _eff_add_tendon_qH_diag_635cc37e_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_efm_ts_in,
    wp::array_t<wp::float32> var_qH_out)
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
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        const wp::float32 var_5 = 0.0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::int32* var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::float32 var_31;
        //---------
        // forward
        // def _eff_add_tendon_qH_diag(                                                           <L 1725>
        // worldid, t = wp.tid()                                                                  <L 1738>
        builtin_tid2d(var_0, var_1);
        // ts = efm_ts_in[worldid, t]                                                             <L 1739>
        var_2 = wp::address(var_efm_ts_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if ts == 0.0:                                                                          <L 1740>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 1741>
            return;
        }
        // rowadr = ten_J_rowadr[t]                                                               <L 1742>
        var_7 = wp::address(var_ten_J_rowadr, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // rownnz = ten_J_rownnz[t]                                                               <L 1743>
        var_10 = wp::address(var_ten_J_rownnz, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // for j in range(rownnz):                                                                <L 1744>
        var_13 = wp::range(var_11);
        start_for_1:;
            if (iter_cmp(var_13) == 0) goto end_for_1;
            var_14 = wp::iter_next(var_13);
            // sparseid = rowadr + j                                                              <L 1745>
            var_15 = wp::add(var_8, var_14);
            // dof = ten_J_colind[sparseid]                                                       <L 1746>
            var_16 = wp::address(var_ten_J_colind, var_15);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // val = ten_J_in[worldid, sparseid]                                                  <L 1747>
            var_19 = wp::address(var_ten_J_in, var_0, var_15);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // term = ts * val * val                                                              <L 1748>
            var_22 = wp::mul(var_3, var_20);
            var_23 = wp::mul(var_22, var_20);
            // diag_elem = M_rowadr[dof] + M_rownnz[dof] - 1                                      <L 1749>
            var_24 = wp::address(var_M_rowadr, var_17);
            var_25 = wp::address(var_M_rownnz, var_17);
            var_27 = wp::load(var_24);
            var_28 = wp::load(var_25);
            var_26 = wp::add(var_27, var_28);
            var_30 = wp::sub(var_26, var_29);
            // wp.atomic_add(qH_out, worldid, diag_elem, term)                                    <L 1750>
            var_31 = wp::atomic_add(var_qH_out, var_0, var_30, var_23);
            goto start_for_1;
        end_for_1:;
}



extern "C" __global__ void _qderiv_actuator_passive_6cbfee2a_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::float32> var_dof_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_dof_dampingpoly,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::int32> var_Mi,
    wp::array_t<wp::int32> var_Mj,
    wp::array_t<wp::float32> var_qDeriv_in,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        bool var_16;
        const wp::int32 var_17 = 64;
        wp::int32 var_18;
        bool var_19;
        bool var_20;
        wp::shape_t* var_21;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        wp::shape_t var_24;
        wp::int32 var_25;
        wp::float32* var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::shape_t* var_29;
        const wp::int32 var_30 = 0;
        wp::int32 var_31;
        wp::shape_t var_32;
        wp::int32 var_33;
        wp::vec_t<2, wp::float32>* var_34;
        wp::vec_t<2, wp::float32> var_35;
        wp::vec_t<2, wp::float32> var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 1;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::shape_t* var_44;
        const wp::int32 var_45 = 0;
        wp::int32 var_46;
        wp::shape_t var_47;
        wp::int32 var_48;
        wp::float32* var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32* var_52;
        wp::float32 var_53;
        wp::float32 var_54;
        //---------
        // forward
        // def _qderiv_actuator_passive(                                                          <L 244>
        // worldid, elemid = wp.tid()                                                             <L 261>
        builtin_tid2d(var_0, var_1);
        // dofiid = Mi[elemid]                                                                    <L 263>
        var_2 = wp::address(var_Mi, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dofjid = Mj[elemid]                                                                    <L 264>
        var_5 = wp::address(var_Mj, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // madr = M_elemid[dofiid, dofjid]                                                        <L 267>
        var_8 = wp::address(var_M_elemid, var_3, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if madr < 0:                                                                           <L 268>
        var_12 = (var_9 < var_11);
        if (var_12) {
            // return                                                                             <L 269>
            return;
        }
        // qderiv = qDeriv_in[worldid, madr]                                                      <L 271>
        var_13 = wp::address(var_qDeriv_in, var_0, var_9);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if not (opt_disableflags & DisableBit.DAMPER) and dofiid == dofjid:                    <L 273>
        var_18 = wp::bit_and(var_opt_disableflags, var_17);
        var_19 = wp::unot(var_18);
        var_16 = var_19;
        if (var_16) {
            var_20 = (var_3 == var_6);
            var_16 = var_16 && var_20;
        }
        if (var_16) {
            // damping = dof_damping[worldid % dof_damping.shape[0], dofiid]                      <L 274>
            var_21 = &(var_dof_damping.shape);
            var_24 = wp::load(var_21);
            var_23 = wp::extract(var_24, var_22);
            var_25 = wp::mod(var_0, var_23);
            var_26 = wp::address(var_dof_damping, var_25, var_3);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofiid]                <L 275>
            var_29 = &(var_dof_dampingpoly.shape);
            var_32 = wp::load(var_29);
            var_31 = wp::extract(var_32, var_30);
            var_33 = wp::mod(var_0, var_31);
            var_34 = wp::address(var_dof_dampingpoly, var_33, var_3);
            var_36 = wp::load(var_34);
            var_35 = wp::copy(var_36);
            // v = qvel_in[worldid, dofiid]                                                       <L 276>
            var_37 = wp::address(var_qvel_in, var_0, var_3);
            var_39 = wp::load(var_37);
            var_38 = wp::copy(var_39);
            // qderiv -= util_misc._poly_force_deriv(damping, dpoly, v, 1)                        <L 277>
            var_41 = _poly_force_deriv_0(var_27, var_35, var_38, var_40);
            var_42 = wp::sub(var_14, var_41);
        }
        var_43 = wp::where(var_16, var_42, var_14);
        // qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]                                <L 279>
        var_44 = &(var_opt_timestep.shape);
        var_47 = wp::load(var_44);
        var_46 = wp::extract(var_47, var_45);
        var_48 = wp::mod(var_0, var_46);
        var_49 = wp::address(var_opt_timestep, var_48);
        var_51 = wp::load(var_49);
        var_50 = wp::mul(var_43, var_51);
        // qDeriv_out[worldid, madr] = M_in[worldid, madr] - qderiv                               <L 281>
        var_52 = wp::address(var_M_in, var_0, var_9);
        var_54 = wp::load(var_52);
        var_53 = wp::sub(var_54, var_50);
        wp::array_store(var_qDeriv_out, var_0, var_9, var_53);
}



extern "C" __global__ void _eff_diag_stiff_6e912a1c_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::float32> var_qpos_spring,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::int32> var_jnt_qposadr,
    wp::array_t<wp::int32> var_jnt_dofadr,
    wp::array_t<wp::float32> var_jnt_stiffness,
    wp::array_t<wp::vec_t<2, wp::float32>> var_jnt_stiffnesspoly,
    wp::array_t<wp::float32> var_qpos_in,
    wp::array_t<wp::float32> var_efm_diag_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::shape_t* var_8;
        const wp::int32 var_9 = 0;
        wp::int32 var_10;
        wp::shape_t var_11;
        wp::int32 var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::int32 var_16 = 32;
        wp::int32 var_17;
        wp::shape_t* var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        wp::shape_t var_21;
        wp::int32 var_22;
        wp::float32* var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::shape_t* var_26;
        const wp::int32 var_27 = 0;
        wp::int32 var_28;
        wp::shape_t var_29;
        wp::int32 var_30;
        wp::vec_t<2, wp::float32>* var_31;
        wp::vec_t<2, wp::float32> var_32;
        wp::vec_t<2, wp::float32> var_33;
        bool var_34;
        const wp::float32 var_35 = 0.0;
        bool var_36;
        const wp::int32 var_37 = 0;
        wp::float32 var_38;
        const wp::float32 var_39 = 0.0;
        bool var_40;
        const wp::int32 var_41 = 1;
        wp::float32 var_42;
        const wp::float32 var_43 = 0.0;
        bool var_44;
        wp::int32* var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::shape_t* var_48;
        const wp::int32 var_49 = 0;
        wp::int32 var_50;
        wp::shape_t var_51;
        wp::int32 var_52;
        const wp::int32 var_53 = 0;
        bool var_54;
        const wp::int32 var_55 = 0;
        wp::int32 var_56;
        wp::float32* var_57;
        const wp::int32 var_58 = 0;
        wp::int32 var_59;
        wp::float32* var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 1;
        wp::int32 var_65;
        wp::float32* var_66;
        const wp::int32 var_67 = 1;
        wp::int32 var_68;
        wp::float32* var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        const wp::int32 var_73 = 2;
        wp::int32 var_74;
        wp::float32* var_75;
        const wp::int32 var_76 = 2;
        wp::int32 var_77;
        wp::float32* var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::float32 var_83;
        const wp::float32 var_84 = 0.0;
        const wp::int32 var_85 = 0;
        wp::float32 var_86;
        wp::float32 var_87;
        const wp::int32 var_88 = 0;
        wp::float32 var_89;
        wp::int32 var_90;
        const wp::int32 var_91 = 1;
        wp::float32 var_92;
        wp::int32 var_93;
        const wp::int32 var_94 = 2;
        wp::float32 var_95;
        wp::int32 var_96;
        const wp::int32 var_97 = 3;
        wp::int32 var_98;
        wp::float32* var_99;
        const wp::int32 var_100 = 4;
        wp::int32 var_101;
        wp::float32* var_102;
        const wp::int32 var_103 = 5;
        wp::int32 var_104;
        wp::float32* var_105;
        const wp::int32 var_106 = 6;
        wp::int32 var_107;
        wp::float32* var_108;
        wp::quat_t<wp::float32> var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        wp::quat_t<wp::float32> var_114;
        const wp::int32 var_115 = 3;
        wp::int32 var_116;
        wp::float32* var_117;
        const wp::int32 var_118 = 4;
        wp::int32 var_119;
        wp::float32* var_120;
        const wp::int32 var_121 = 5;
        wp::int32 var_122;
        wp::float32* var_123;
        const wp::int32 var_124 = 6;
        wp::int32 var_125;
        wp::float32* var_126;
        wp::quat_t<wp::float32> var_127;
        wp::float32 var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::float32 var_133;
        const wp::float32 var_134 = 0.0;
        const wp::int32 var_135 = 0;
        wp::float32 var_136;
        wp::float32 var_137;
        const wp::int32 var_138 = 0;
        wp::float32 var_139;
        const wp::int32 var_140 = 3;
        wp::int32 var_141;
        wp::int32 var_142;
        const wp::int32 var_143 = 1;
        wp::float32 var_144;
        const wp::int32 var_145 = 3;
        wp::int32 var_146;
        wp::int32 var_147;
        const wp::int32 var_148 = 2;
        wp::float32 var_149;
        const wp::int32 var_150 = 3;
        wp::int32 var_151;
        wp::int32 var_152;
        const wp::int32 var_153 = 1;
        bool var_154;
        const wp::int32 var_155 = 0;
        wp::int32 var_156;
        wp::float32* var_157;
        const wp::int32 var_158 = 1;
        wp::int32 var_159;
        wp::float32* var_160;
        const wp::int32 var_161 = 2;
        wp::int32 var_162;
        wp::float32* var_163;
        const wp::int32 var_164 = 3;
        wp::int32 var_165;
        wp::float32* var_166;
        wp::quat_t<wp::float32> var_167;
        wp::float32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::quat_t<wp::float32> var_172;
        const wp::int32 var_173 = 0;
        wp::int32 var_174;
        wp::float32* var_175;
        const wp::int32 var_176 = 1;
        wp::int32 var_177;
        wp::float32* var_178;
        const wp::int32 var_179 = 2;
        wp::int32 var_180;
        wp::float32* var_181;
        const wp::int32 var_182 = 3;
        wp::int32 var_183;
        wp::float32* var_184;
        wp::quat_t<wp::float32> var_185;
        wp::float32 var_186;
        wp::float32 var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::vec_t<3, wp::float32> var_190;
        wp::float32 var_191;
        const wp::float32 var_192 = 0.0;
        const wp::int32 var_193 = 0;
        wp::float32 var_194;
        wp::float32 var_195;
        const wp::int32 var_196 = 0;
        wp::float32 var_197;
        wp::int32 var_198;
        const wp::int32 var_199 = 1;
        wp::float32 var_200;
        wp::int32 var_201;
        const wp::int32 var_202 = 2;
        wp::float32 var_203;
        wp::int32 var_204;
        wp::float32* var_205;
        wp::float32* var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        wp::float32 var_209;
        const wp::float32 var_210 = 0.0;
        const wp::int32 var_211 = 0;
        wp::float32 var_212;
        wp::float32 var_213;
        wp::float32 var_214;
        wp::float32 var_215;
        wp::int32 var_216;
        wp::quat_t<wp::float32> var_217;
        wp::quat_t<wp::float32> var_218;
        wp::vec_t<3, wp::float32> var_219;
        wp::float32 var_220;
        wp::float32 var_221;
        //---------
        // forward
        // def _eff_diag_stiff(                                                                   <L 1350>
        // worldid, jntid = wp.tid()                                                              <L 1365>
        builtin_tid2d(var_0, var_1);
        // dofid = jnt_dofadr[jntid]                                                              <L 1366>
        var_2 = wp::address(var_jnt_dofadr, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // jnttype = jnt_type[jntid]                                                              <L 1367>
        var_5 = wp::address(var_jnt_type, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // h = opt_timestep[worldid % opt_timestep.shape[0]]                                      <L 1368>
        var_8 = &(var_opt_timestep.shape);
        var_11 = wp::load(var_8);
        var_10 = wp::extract(var_11, var_9);
        var_12 = wp::mod(var_0, var_10);
        var_13 = wp::address(var_opt_timestep, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if opt_disableflags & DisableBit.SPRING:                                               <L 1370>
        var_17 = wp::bit_and(var_opt_disableflags, var_16);
        if (var_17) {
            // return                                                                             <L 1371>
            return;
        }
        // stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], jntid]                     <L 1373>
        var_18 = &(var_jnt_stiffness.shape);
        var_21 = wp::load(var_18);
        var_20 = wp::extract(var_21, var_19);
        var_22 = wp::mod(var_0, var_20);
        var_23 = wp::address(var_jnt_stiffness, var_22, var_1);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], jntid]                 <L 1374>
        var_26 = &(var_jnt_stiffnesspoly.shape);
        var_29 = wp::load(var_26);
        var_28 = wp::extract(var_29, var_27);
        var_30 = wp::mod(var_0, var_28);
        var_31 = wp::address(var_jnt_stiffnesspoly, var_30, var_1);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // if stiffness == 0.0 and spoly[0] == 0.0 and spoly[1] == 0.0:                           <L 1375>
        var_36 = (var_24 == var_35);
        var_34 = var_36;
        if (var_34) {
            var_38 = wp::extract(var_32, var_37);
            var_40 = (var_38 == var_39);
            var_34 = var_34 && var_40;
        }
        if (var_34) {
            var_42 = wp::extract(var_32, var_41);
            var_44 = (var_42 == var_43);
            var_34 = var_34 && var_44;
        }
        if (var_34) {
            // return                                                                             <L 1376>
            return;
        }
        // qposid = jnt_qposadr[jntid]                                                            <L 1378>
        var_45 = wp::address(var_jnt_qposadr, var_1);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // qpos_spring_id = worldid % qpos_spring.shape[0]                                        <L 1379>
        var_48 = &(var_qpos_spring.shape);
        var_51 = wp::load(var_48);
        var_50 = wp::extract(var_51, var_49);
        var_52 = wp::mod(var_0, var_50);
        // if jnttype == JointType.FREE:                                                          <L 1381>
        var_54 = (var_6 == var_53);
        if (var_54) {
            // dif = wp.vec3(                                                                     <L 1382>
            // qpos_in[worldid, qposid + 0] - qpos_spring[qpos_spring_id, qposid + 0],            <L 1383>
            var_56 = wp::add(var_46, var_55);
            var_57 = wp::address(var_qpos_in, var_0, var_56);
            var_59 = wp::add(var_46, var_58);
            var_60 = wp::address(var_qpos_spring, var_52, var_59);
            var_62 = wp::load(var_57);
            var_63 = wp::load(var_60);
            var_61 = wp::sub(var_62, var_63);
            // qpos_in[worldid, qposid + 1] - qpos_spring[qpos_spring_id, qposid + 1],            <L 1384>
            var_65 = wp::add(var_46, var_64);
            var_66 = wp::address(var_qpos_in, var_0, var_65);
            var_68 = wp::add(var_46, var_67);
            var_69 = wp::address(var_qpos_spring, var_52, var_68);
            var_71 = wp::load(var_66);
            var_72 = wp::load(var_69);
            var_70 = wp::sub(var_71, var_72);
            // qpos_in[worldid, qposid + 2] - qpos_spring[qpos_spring_id, qposid + 2],            <L 1385>
            var_74 = wp::add(var_46, var_73);
            var_75 = wp::address(var_qpos_in, var_0, var_74);
            var_77 = wp::add(var_46, var_76);
            var_78 = wp::address(var_qpos_spring, var_52, var_77);
            var_80 = wp::load(var_75);
            var_81 = wp::load(var_78);
            var_79 = wp::sub(var_80, var_81);
            var_82 = wp::vec_t<3, wp::float32>(var_61, var_70, var_79);
            // r = wp.length(dif)                                                                 <L 1387>
            var_83 = wp::length(var_82);
            // k = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, r, 0))               <L 1388>
            var_86 = _poly_force_deriv_0(var_24, var_32, var_83, var_85);
            var_87 = wp::max(var_84, var_86);
            // for i in range(3):                                                                 <L 1389>
            // efm_diag_out[worldid, dofid + i] = h * k                                           <L 1390>
            var_89 = wp::mul(var_14, var_87);
            var_90 = wp::add(var_3, var_88);
            wp::array_store(var_efm_diag_out, var_0, var_90, var_89);
            var_92 = wp::mul(var_14, var_87);
            var_93 = wp::add(var_3, var_91);
            wp::array_store(var_efm_diag_out, var_0, var_93, var_92);
            var_95 = wp::mul(var_14, var_87);
            var_96 = wp::add(var_3, var_94);
            wp::array_store(var_efm_diag_out, var_0, var_96, var_95);
            // rot = wp.normalize(                                                                <L 1392>
            // wp.quat(                                                                           <L 1393>
            // qpos_in[worldid, qposid + 3],                                                      <L 1394>
            var_98 = wp::add(var_46, var_97);
            var_99 = wp::address(var_qpos_in, var_0, var_98);
            // qpos_in[worldid, qposid + 4],                                                      <L 1395>
            var_101 = wp::add(var_46, var_100);
            var_102 = wp::address(var_qpos_in, var_0, var_101);
            // qpos_in[worldid, qposid + 5],                                                      <L 1396>
            var_104 = wp::add(var_46, var_103);
            var_105 = wp::address(var_qpos_in, var_0, var_104);
            // qpos_in[worldid, qposid + 6],                                                      <L 1397>
            var_107 = wp::add(var_46, var_106);
            var_108 = wp::address(var_qpos_in, var_0, var_107);
            var_110 = wp::load(var_99);
            var_111 = wp::load(var_102);
            var_112 = wp::load(var_105);
            var_113 = wp::load(var_108);
            var_109 = wp::quat_t<wp::float32>(var_110, var_111, var_112, var_113);
            var_114 = wp::normalize(var_109);
            // ref = wp.quat(                                                                     <L 1400>
            // qpos_spring[qpos_spring_id, qposid + 3],                                           <L 1401>
            var_116 = wp::add(var_46, var_115);
            var_117 = wp::address(var_qpos_spring, var_52, var_116);
            // qpos_spring[qpos_spring_id, qposid + 4],                                           <L 1402>
            var_119 = wp::add(var_46, var_118);
            var_120 = wp::address(var_qpos_spring, var_52, var_119);
            // qpos_spring[qpos_spring_id, qposid + 5],                                           <L 1403>
            var_122 = wp::add(var_46, var_121);
            var_123 = wp::address(var_qpos_spring, var_52, var_122);
            // qpos_spring[qpos_spring_id, qposid + 6],                                           <L 1404>
            var_125 = wp::add(var_46, var_124);
            var_126 = wp::address(var_qpos_spring, var_52, var_125);
            var_128 = wp::load(var_117);
            var_129 = wp::load(var_120);
            var_130 = wp::load(var_123);
            var_131 = wp::load(var_126);
            var_127 = wp::quat_t<wp::float32>(var_128, var_129, var_130, var_131);
            // dif_rot = math.quat_sub(rot, ref)                                                  <L 1406>
            var_132 = quat_sub_0(var_114, var_127);
            // r_rot = wp.length(dif_rot)                                                         <L 1407>
            var_133 = wp::length(var_132);
            // k_rot = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, r_rot, 0))       <L 1408>
            var_136 = _poly_force_deriv_0(var_24, var_32, var_133, var_135);
            var_137 = wp::max(var_134, var_136);
            // for i in range(3):                                                                 <L 1409>
            // efm_diag_out[worldid, dofid + 3 + i] = h * k_rot                                   <L 1410>
            var_139 = wp::mul(var_14, var_137);
            var_141 = wp::add(var_3, var_140);
            var_142 = wp::add(var_141, var_138);
            wp::array_store(var_efm_diag_out, var_0, var_142, var_139);
            var_144 = wp::mul(var_14, var_137);
            var_146 = wp::add(var_3, var_145);
            var_147 = wp::add(var_146, var_143);
            wp::array_store(var_efm_diag_out, var_0, var_147, var_144);
            var_149 = wp::mul(var_14, var_137);
            var_151 = wp::add(var_3, var_150);
            var_152 = wp::add(var_151, var_148);
            wp::array_store(var_efm_diag_out, var_0, var_152, var_149);
        }
        if (!var_54) {
            // elif jnttype == JointType.BALL:                                                    <L 1412>
            var_154 = (var_6 == var_153);
            if (var_154) {
                // rot = wp.normalize(                                                            <L 1413>
                // wp.quat(                                                                       <L 1414>
                // qpos_in[worldid, qposid + 0],                                                  <L 1415>
                var_156 = wp::add(var_46, var_155);
                var_157 = wp::address(var_qpos_in, var_0, var_156);
                // qpos_in[worldid, qposid + 1],                                                  <L 1416>
                var_159 = wp::add(var_46, var_158);
                var_160 = wp::address(var_qpos_in, var_0, var_159);
                // qpos_in[worldid, qposid + 2],                                                  <L 1417>
                var_162 = wp::add(var_46, var_161);
                var_163 = wp::address(var_qpos_in, var_0, var_162);
                // qpos_in[worldid, qposid + 3],                                                  <L 1418>
                var_165 = wp::add(var_46, var_164);
                var_166 = wp::address(var_qpos_in, var_0, var_165);
                var_168 = wp::load(var_157);
                var_169 = wp::load(var_160);
                var_170 = wp::load(var_163);
                var_171 = wp::load(var_166);
                var_167 = wp::quat_t<wp::float32>(var_168, var_169, var_170, var_171);
                var_172 = wp::normalize(var_167);
                // ref = wp.quat(                                                                 <L 1421>
                // qpos_spring[qpos_spring_id, qposid + 0],                                       <L 1422>
                var_174 = wp::add(var_46, var_173);
                var_175 = wp::address(var_qpos_spring, var_52, var_174);
                // qpos_spring[qpos_spring_id, qposid + 1],                                       <L 1423>
                var_177 = wp::add(var_46, var_176);
                var_178 = wp::address(var_qpos_spring, var_52, var_177);
                // qpos_spring[qpos_spring_id, qposid + 2],                                       <L 1424>
                var_180 = wp::add(var_46, var_179);
                var_181 = wp::address(var_qpos_spring, var_52, var_180);
                // qpos_spring[qpos_spring_id, qposid + 3],                                       <L 1425>
                var_183 = wp::add(var_46, var_182);
                var_184 = wp::address(var_qpos_spring, var_52, var_183);
                var_186 = wp::load(var_175);
                var_187 = wp::load(var_178);
                var_188 = wp::load(var_181);
                var_189 = wp::load(var_184);
                var_185 = wp::quat_t<wp::float32>(var_186, var_187, var_188, var_189);
                // dif_rot = math.quat_sub(rot, ref)                                              <L 1427>
                var_190 = quat_sub_0(var_172, var_185);
                // r_rot = wp.length(dif_rot)                                                     <L 1428>
                var_191 = wp::length(var_190);
                // k_rot = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, r_rot, 0))       <L 1429>
                var_194 = _poly_force_deriv_0(var_24, var_32, var_191, var_193);
                var_195 = wp::max(var_192, var_194);
                // for i in range(3):                                                             <L 1430>
                // efm_diag_out[worldid, dofid + i] = h * k_rot                                   <L 1431>
                var_197 = wp::mul(var_14, var_195);
                var_198 = wp::add(var_3, var_196);
                wp::array_store(var_efm_diag_out, var_0, var_198, var_197);
                var_200 = wp::mul(var_14, var_195);
                var_201 = wp::add(var_3, var_199);
                wp::array_store(var_efm_diag_out, var_0, var_201, var_200);
                var_203 = wp::mul(var_14, var_195);
                var_204 = wp::add(var_3, var_202);
                wp::array_store(var_efm_diag_out, var_0, var_204, var_203);
            }
            if (!var_154) {
                // x = qpos_in[worldid, qposid] - qpos_spring[qpos_spring_id, qposid]             <L 1434>
                var_205 = wp::address(var_qpos_in, var_0, var_46);
                var_206 = wp::address(var_qpos_spring, var_52, var_46);
                var_208 = wp::load(var_205);
                var_209 = wp::load(var_206);
                var_207 = wp::sub(var_208, var_209);
                // k = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, x, 0))           <L 1435>
                var_212 = _poly_force_deriv_0(var_24, var_32, var_207, var_211);
                var_213 = wp::max(var_210, var_212);
                // efm_diag_out[worldid, dofid] = h * k                                           <L 1436>
                var_214 = wp::mul(var_14, var_213);
                wp::array_store(var_efm_diag_out, var_0, var_3, var_214);
            }
        }
        var_215 = wp::where(var_54, var_87, var_213);
        var_216 = wp::where(var_54, var_148, var_202);
        var_217 = wp::where(var_54, var_114, var_172);
        var_218 = wp::where(var_54, var_127, var_185);
        var_219 = wp::where(var_54, var_132, var_190);
        var_220 = wp::where(var_54, var_133, var_191);
        var_221 = wp::where(var_54, var_137, var_195);
}



extern "C" __global__ void deriv_rne_cacc_cfrcbody_forward_7f6cd6a1_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::vec_t<10, wp::float32>> var_cinert_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_dot_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcvel_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcdof_dot_in,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcacc_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcfrcbody_out)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::array_t<wp::float32> var_15;
        wp::vec_t<6, wp::float32>* var_16;
        wp::vec_t<6, wp::float32> var_17;
        wp::vec_t<6, wp::float32> var_18;
        wp::int32 var_19;
        wp::range_t var_20;
        wp::int32 var_21;
        bool var_22;
        wp::vec_t<6, wp::float32>* var_23;
        wp::vec_t<6, wp::float32> var_24;
        wp::vec_t<6, wp::float32> var_25;
        wp::vec_t<6, wp::float32> var_26;
        wp::vec_t<6, wp::float32>* var_27;
        wp::vec_t<6, wp::float32> var_28;
        wp::vec_t<6, wp::float32> var_29;
        wp::float32* var_30;
        wp::vec_t<6, wp::float32> var_31;
        wp::float32 var_32;
        wp::vec_t<6, wp::float32> var_33;
        wp::vec_t<10, wp::float32>* var_34;
        wp::vec_t<10, wp::float32> var_35;
        wp::vec_t<10, wp::float32> var_36;
        wp::vec_t<6, wp::float32>* var_37;
        wp::vec_t<6, wp::float32> var_38;
        wp::vec_t<6, wp::float32> var_39;
        wp::vec_t<6, wp::float32>* var_40;
        wp::vec_t<6, wp::float32> var_41;
        wp::vec_t<6, wp::float32> var_42;
        wp::vec_t<6, wp::float32> var_43;
        wp::vec_t<6, wp::float32> var_44;
        wp::vec_t<6, wp::float32> var_45;
        wp::vec_t<6, wp::float32> var_46;
        wp::vec_t<6, wp::float32> var_47;
        wp::vec_t<6, wp::float32> var_48;
        wp::vec_t<6, wp::float32> var_49;
        //---------
        // forward
        // def deriv_rne_cacc_cfrcbody_forward(                                                   <L 429>
        // worldid, nodeid, dofid = wp.tid()                                                      <L 448>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid = body_tree_[nodeid]                                                            <L 449>
        var_3 = wp::address(var_body_tree_, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // dofadr = body_dofadr[bodyid]                                                           <L 450>
        var_6 = wp::address(var_body_dofadr, var_4);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // dofnum = body_dofnum[bodyid]                                                           <L 451>
        var_9 = wp::address(var_body_dofnum, var_4);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // pid = body_parentid[bodyid]                                                            <L 452>
        var_12 = wp::address(var_body_parentid, var_4);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // qvel = qvel_in[worldid]                                                                <L 454>
        var_15 = wp::view(var_qvel_in, var_0);
        // dcacc = Dcacc_out[worldid, pid, dofid]                                                 <L 456>
        var_16 = wp::address(var_Dcacc_out, var_0, var_13, var_2);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // for j in range(dofadr, dofadr + dofnum):                                               <L 458>
        var_19 = wp::add(var_7, var_10);
        var_20 = wp::range(var_7, var_19);
        start_for_0:;
            if (iter_cmp(var_20) == 0) goto end_for_0;
            var_21 = wp::iter_next(var_20);
            // if j == dofid:                                                                     <L 460>
            var_22 = (var_21 == var_2);
            if (var_22) {
                // dcacc += cdof_dot_in[worldid, j]                                               <L 461>
                var_23 = wp::address(var_cdof_dot_in, var_0, var_21);
                var_25 = wp::load(var_23);
                var_24 = wp::add(var_17, var_25);
            }
            var_26 = wp::where(var_22, var_24, var_17);
            // dcdofdot = Dcdof_dot_in[worldid, j, dofid]                                         <L 464>
            var_27 = wp::address(var_Dcdof_dot_in, var_0, var_21, var_2);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // dcacc += dcdofdot * qvel[j]                                                        <L 465>
            var_30 = wp::address(var_15, var_21);
            var_32 = wp::load(var_30);
            var_31 = wp::mul(var_28, var_32);
            var_33 = wp::add(var_26, var_31);
            wp::assign(var_17, var_33);
            goto start_for_0;
        end_for_0:;
        // Dcacc_out[worldid, bodyid, dofid] = dcacc                                              <L 467>
        wp::array_store(var_Dcacc_out, var_0, var_4, var_2, var_17);
        // cinert = cinert_in[worldid, bodyid]                                                    <L 470>
        var_34 = wp::address(var_cinert_in, var_0, var_4);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // cvel = cvel_in[worldid, bodyid]                                                        <L 471>
        var_37 = wp::address(var_cvel_in, var_0, var_4);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // dcvel = Dcvel_in[worldid, bodyid, dofid]                                               <L 472>
        var_40 = wp::address(var_Dcvel_in, var_0, var_4, var_2);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // term1 = math.inert_vec(cinert, dcacc)                                                  <L 475>
        var_43 = inert_vec_0(var_35, var_17);
        // cinert_cvel = math.inert_vec(cinert, cvel)                                             <L 478>
        var_44 = inert_vec_0(var_35, var_38);
        // cinert_dcvel = math.inert_vec(cinert, dcvel)                                           <L 479>
        var_45 = inert_vec_0(var_35, var_41);
        // term2 = math.motion_cross_force(dcvel, cinert_cvel) + math.motion_cross_force(cvel, cinert_dcvel)       <L 480>
        var_46 = motion_cross_force_0(var_41, var_44);
        var_47 = motion_cross_force_0(var_38, var_45);
        var_48 = wp::add(var_46, var_47);
        // Dcfrcbody_out[worldid, bodyid, dofid] = term1 + term2                                  <L 482>
        var_49 = wp::add(var_43, var_48);
        wp::array_store(var_Dcfrcbody_out, var_0, var_4, var_2, var_49);
}



extern "C" __global__ void _eff_fold_tendon_84d3c1a3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::int32> var_efm_dofid,
    wp::array_t<wp::int32> var_efm_dofblk,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_efm_ts_in,
    wp::array_t<wp::float32> var_epB_out)
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
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        const wp::float32 var_5 = 0.0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::float32* var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::range_t var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        bool var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 9;
        wp::int32 var_45;
        const wp::int32 var_46 = 3;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        //---------
        // forward
        // def _eff_fold_tendon(                                                                  <L 3590>
        // worldid, t = wp.tid()                                                                  <L 3603>
        builtin_tid2d(var_0, var_1);
        // s = efm_ts_in[worldid, t]                                                              <L 3604>
        var_2 = wp::address(var_efm_ts_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if s == 0.0:                                                                           <L 3605>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 3606>
            return;
        }
        // rowadr = ten_J_rowadr[t]                                                               <L 3607>
        var_7 = wp::address(var_ten_J_rowadr, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // rownnz = ten_J_rownnz[t]                                                               <L 3608>
        var_10 = wp::address(var_ten_J_rownnz, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // for a in range(rownnz):                                                                <L 3609>
        var_13 = wp::range(var_11);
        start_for_1:;
            if (iter_cmp(var_13) == 0) goto end_for_1;
            var_14 = wp::iter_next(var_13);
            // ia = ten_J_colind[rowadr + a]                                                      <L 3610>
            var_15 = wp::add(var_8, var_14);
            var_16 = wp::address(var_ten_J_colind, var_15);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // k = efm_dofblk[ia]                                                                 <L 3611>
            var_19 = wp::address(var_efm_dofblk, var_17);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // if k < 0:                                                                          <L 3612>
            var_23 = (var_20 < var_22);
            if (var_23) {
                // continue                                                                       <L 3613>
                goto start_for_1;
            }
            // base = efm_dofid[k]                                                                <L 3614>
            var_24 = wp::address(var_efm_dofid, var_20);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // s_va = s * ten_J_in[worldid, rowadr + a]                                           <L 3615>
            var_27 = wp::add(var_8, var_14);
            var_28 = wp::address(var_ten_J_in, var_0, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::mul(var_3, var_30);
            // for b in range(rownnz):                                                            <L 3616>
            var_31 = wp::range(var_11);
            start_for_3:;
                if (iter_cmp(var_31) == 0) goto end_for_3;
                var_32 = wp::iter_next(var_31);
                // ib = ten_J_colind[rowadr + b]                                                  <L 3617>
                var_33 = wp::add(var_8, var_32);
                var_34 = wp::address(var_ten_J_colind, var_33);
                var_36 = wp::load(var_34);
                var_35 = wp::copy(var_36);
                // if efm_dofblk[ib] == k:                                                        <L 3618>
                var_37 = wp::address(var_efm_dofblk, var_35);
                var_39 = wp::load(var_37);
                var_38 = (var_39 == var_20);
                if (var_38) {
                    // vb = ten_J_in[worldid, rowadr + b]                                         <L 3619>
                    var_40 = wp::add(var_8, var_32);
                    var_41 = wp::address(var_ten_J_in, var_0, var_40);
                    var_43 = wp::load(var_41);
                    var_42 = wp::copy(var_43);
                    // wp.atomic_add(epB_out, worldid, 9 * k + 3 * (ia - base) + (ib - base), s_va * vb)       <L 3620>
                    var_45 = wp::mul(var_44, var_20);
                    var_47 = wp::sub(var_17, var_25);
                    var_48 = wp::mul(var_46, var_47);
                    var_49 = wp::add(var_45, var_48);
                    var_50 = wp::sub(var_35, var_25);
                    var_51 = wp::add(var_49, var_50);
                    var_52 = wp::mul(var_29, var_42);
                    var_53 = wp::atomic_add(var_epB_out, var_0, var_51, var_52);
                }
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
}



extern "C" __global__ void _qderiv_actuator_passive_vel_8dfd9dcc_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_gaintype,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::int32> var_actuator_ctrlspec,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_dynprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_gainprm,
    wp::array_t<wp::vec_t<10, wp::float32>> var_actuator_biasprm,
    wp::array_t<bool> var_actuator_actlimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_actrange,
    wp::array_t<bool> var_actuator_actearly,
    wp::array_t<bool> var_actuator_forcelimited,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_forcerange,
    wp::array_t<wp::float32> var_actuator_acc0,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_lengthrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_ctrl_in,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_vel_out)
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
        const wp::float32 var_22 = 0.0;
        wp::float32 var_23;
        const wp::float32 var_24 = 0.0;
        wp::float32 var_25;
        const wp::float32 var_26 = 0.0;
        wp::float32 var_27;
        const wp::float32 var_28 = 0.0;
        wp::float32 var_29;
        wp::int32* var_30;
        const wp::int32 var_31 = 1;
        bool var_32;
        wp::int32 var_33;
        wp::vec_t<10, wp::float32>* var_34;
        const wp::int32 var_35 = 2;
        wp::float32 var_36;
        wp::vec_t<10, wp::float32> var_37;
        wp::int32* var_38;
        const wp::int32 var_39 = 2;
        bool var_40;
        wp::int32 var_41;
        wp::float32* var_42;
        wp::float32* var_43;
        wp::shape_t* var_44;
        const wp::int32 var_45 = 0;
        wp::int32 var_46;
        wp::shape_t var_47;
        wp::int32 var_48;
        wp::vec_t<2, wp::float32>* var_49;
        wp::shape_t* var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::shape_t var_53;
        wp::int32 var_54;
        wp::float32* var_55;
        wp::vec_t<10, wp::float32>* var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::vec_t<2, wp::float32> var_60;
        wp::float32 var_61;
        wp::vec_t<10, wp::float32> var_62;
        wp::int32* var_63;
        const wp::int32 var_64 = 3;
        bool var_65;
        wp::int32 var_66;
        const wp::float32 var_67 = 0.0;
        wp::vec_t<10, wp::float32>* var_68;
        wp::vec_t<10, wp::float32> var_69;
        wp::vec_t<10, wp::float32> var_70;
        wp::vec_t<10, wp::float32>* var_71;
        wp::vec_t<10, wp::float32> var_72;
        wp::vec_t<10, wp::float32> var_73;
        const wp::int32 var_74 = 0;
        wp::float32 var_75;
        const wp::int32 var_76 = 1;
        wp::float32 var_77;
        const wp::float32 var_78 = 0.0;
        wp::int32* var_79;
        const wp::int32 var_80 = 7;
        wp::int32 var_81;
        wp::int32 var_82;
        const wp::int32 var_83 = 0;
        bool var_84;
        const wp::float32 var_85 = 1e-15;
        const wp::int32 var_86 = 0;
        wp::float32 var_87;
        wp::float32 var_88;
        const wp::int32 var_89 = 6;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::int32* var_96;
        wp::float32 var_97;
        wp::int32 var_98;
        const wp::float32 var_99 = 0.0;
        bool var_100;
        const wp::float32 var_101 = 1.0;
        wp::float32* var_102;
        wp::float32 var_103;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::float32 var_110;
        wp::float32 var_111;
        wp::float32 var_112;
        const wp::float32 var_113 = 0.0;
        bool var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::float32 var_117;
        wp::float32 var_118;
        wp::float32 var_119;
        const wp::int32 var_120 = 6;
        wp::float32 var_121;
        const wp::float32 var_122 = 0.0;
        bool var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        wp::float32 var_128;
        const wp::float32 var_129 = 0.0;
        wp::float32 var_130;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        wp::float32 var_134;
        wp::float32 var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        wp::float32 var_139;
        wp::float32 var_140;
        wp::float32 var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        wp::int32* var_145;
        const wp::int32 var_146 = 1;
        bool var_147;
        wp::int32 var_148;
        wp::vec_t<10, wp::float32>* var_149;
        const wp::int32 var_150 = 2;
        wp::float32 var_151;
        wp::vec_t<10, wp::float32> var_152;
        wp::float32 var_153;
        wp::int32* var_154;
        const wp::int32 var_155 = 3;
        bool var_156;
        wp::int32 var_157;
        const wp::float32 var_158 = 0.0;
        bool var_159;
        const wp::float32 var_160 = 0.0;
        bool var_161;
        wp::float32 var_162;
        wp::float32 var_163;
        wp::float32 var_164;
        wp::float32 var_165;
        wp::float32 var_166;
        wp::vec_t<10, wp::float32>* var_167;
        wp::vec_t<10, wp::float32> var_168;
        wp::vec_t<10, wp::float32> var_169;
        const wp::int32 var_170 = 0;
        wp::float32 var_171;
        const wp::float32 var_172 = 0.0;
        bool var_173;
        wp::vec_t<10, wp::float32>* var_174;
        wp::vec_t<10, wp::float32> var_175;
        wp::vec_t<10, wp::float32> var_176;
        const wp::int32 var_177 = 1;
        wp::float32 var_178;
        wp::int32* var_179;
        wp::float32 var_180;
        wp::int32 var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        wp::float32 var_184;
        wp::float32 var_185;
        wp::float32 var_186;
        wp::vec_t<10, wp::float32> var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        wp::float32 var_190;
        wp::vec_t<10, wp::float32> var_191;
        wp::vec_t<10, wp::float32> var_192;
        wp::float32 var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        wp::float32 var_196;
        wp::vec_t<10, wp::float32> var_197;
        wp::vec_t<10, wp::float32> var_198;
        wp::float32 var_199;
        wp::float32 var_200;
        wp::float32 var_201;
        wp::float32 var_202;
        wp::vec_t<10, wp::float32> var_203;
        wp::vec_t<10, wp::float32> var_204;
        wp::float32 var_205;
        wp::float32 var_206;
        wp::float32 var_207;
        bool var_208;
        const wp::float32 var_209 = 0.0;
        bool var_210;
        const wp::float32 var_211 = 0.0;
        bool var_212;
        const wp::float32 var_213 = 0.0;
        bool* var_214;
        bool var_215;
        wp::float32* var_216;
        wp::float32 var_217;
        wp::float32 var_218;
        wp::shape_t* var_219;
        const wp::int32 var_220 = 0;
        wp::int32 var_221;
        wp::shape_t var_222;
        wp::int32 var_223;
        wp::vec_t<2, wp::float32>* var_224;
        wp::vec_t<2, wp::float32> var_225;
        wp::vec_t<2, wp::float32> var_226;
        bool var_227;
        const wp::int32 var_228 = 0;
        wp::float32 var_229;
        bool var_230;
        const wp::int32 var_231 = 1;
        wp::float32 var_232;
        bool var_233;
        const wp::float32 var_234 = 0.0;
        bool var_235;
        wp::float32 var_236;
        wp::int32* var_237;
        const wp::int32 var_238 = 0;
        bool var_239;
        wp::int32 var_240;
        const wp::float32 var_241 = 0.0;
        bool var_242;
        wp::int32* var_243;
        wp::int32* var_244;
        wp::int32 var_245;
        wp::int32 var_246;
        wp::int32 var_247;
        const wp::int32 var_248 = 1;
        wp::int32 var_249;
        bool* var_250;
        bool var_251;
        wp::float32* var_252;
        wp::int32* var_253;
        wp::vec_t<10, wp::float32>* var_254;
        wp::shape_t* var_255;
        const wp::int32 var_256 = 0;
        wp::int32 var_257;
        wp::shape_t var_258;
        wp::int32 var_259;
        wp::vec_t<2, wp::float32>* var_260;
        wp::float32* var_261;
        wp::float32* var_262;
        const wp::float32 var_263 = 1.0;
        bool* var_264;
        wp::float32 var_265;
        wp::float32 var_266;
        wp::int32 var_267;
        wp::vec_t<10, wp::float32> var_268;
        wp::vec_t<2, wp::float32> var_269;
        wp::float32 var_270;
        wp::float32 var_271;
        bool var_272;
        bool var_273;
        bool var_274;
        wp::float32* var_275;
        wp::float32 var_276;
        wp::float32 var_277;
        bool var_278;
        wp::float32 var_279;
        bool var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        const wp::float32 var_284 = 0.0;
        bool var_285;
        wp::float32* var_286;
        wp::float32 var_287;
        wp::float32 var_288;
        wp::float32 var_289;
        wp::float32 var_290;
        wp::float32 var_291;
        //---------
        // forward
        // def _qderiv_actuator_passive_vel(                                                      <L 49>
        // worldid, actid = wp.tid()                                                              <L 78>
        builtin_tid2d(var_0, var_1);
        // actuator_gainprm_id = worldid % actuator_gainprm.shape[0]                              <L 80>
        var_2 = &(var_actuator_gainprm.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        // actuator_biasprm_id = worldid % actuator_biasprm.shape[0]                              <L 81>
        var_7 = &(var_actuator_biasprm.shape);
        var_10 = wp::load(var_7);
        var_9 = wp::extract(var_10, var_8);
        var_11 = wp::mod(var_0, var_9);
        // actuator_dynprm_id = worldid % actuator_dynprm.shape[0]                                <L 82>
        var_12 = &(var_actuator_dynprm.shape);
        var_15 = wp::load(var_12);
        var_14 = wp::extract(var_15, var_13);
        var_16 = wp::mod(var_0, var_14);
        // opt_timestep_id = worldid % opt_timestep.shape[0]                                      <L 83>
        var_17 = &(var_opt_timestep.shape);
        var_20 = wp::load(var_17);
        var_19 = wp::extract(var_20, var_18);
        var_21 = wp::mod(var_0, var_19);
        // bias = float(0.0)                                                                      <L 85>
        var_23 = wp::float(var_22);
        // R_dcmotor = float(0.0)                                                                 <L 86>
        var_25 = wp::float(var_24);
        // K_dcmotor = float(0.0)                                                                 <L 87>
        var_27 = wp::float(var_26);
        // te_dcmotor = float(0.0)                                                                <L 88>
        var_29 = wp::float(var_28);
        // if actuator_gaintype[actid] == GainType.AFFINE:                                        <L 90>
        var_30 = wp::address(var_actuator_gaintype, var_1);
        var_33 = wp::load(var_30);
        var_32 = (var_33 == var_31);
        if (var_32) {
            // gain = actuator_gainprm[actuator_gainprm_id, actid][2]                             <L 91>
            var_34 = wp::address(var_actuator_gainprm, var_6, var_1);
            var_37 = wp::load(var_34);
            var_36 = wp::extract(var_37, var_35);
        }
        if (!var_32) {
            // elif actuator_gaintype[actid] == GainType.MUSCLE:                                  <L 92>
            var_38 = wp::address(var_actuator_gaintype, var_1);
            var_41 = wp::load(var_38);
            var_40 = (var_41 == var_39);
            if (var_40) {
                // gain = util_misc.muscle_gain_vel_deriv(                                        <L 93>
                // actuator_length_in[worldid, actid],                                            <L 94>
                var_42 = wp::address(var_actuator_length_in, var_0, var_1);
                // actuator_velocity_in[worldid, actid],                                          <L 95>
                var_43 = wp::address(var_actuator_velocity_in, var_0, var_1);
                // actuator_lengthrange[worldid % actuator_lengthrange.shape[0], actid],          <L 96>
                var_44 = &(var_actuator_lengthrange.shape);
                var_47 = wp::load(var_44);
                var_46 = wp::extract(var_47, var_45);
                var_48 = wp::mod(var_0, var_46);
                var_49 = wp::address(var_actuator_lengthrange, var_48, var_1);
                // actuator_acc0[worldid % actuator_acc0.shape[0], actid],                        <L 97>
                var_50 = &(var_actuator_acc0.shape);
                var_53 = wp::load(var_50);
                var_52 = wp::extract(var_53, var_51);
                var_54 = wp::mod(var_0, var_52);
                var_55 = wp::address(var_actuator_acc0, var_54, var_1);
                // actuator_gainprm[actuator_gainprm_id, actid],                                  <L 98>
                var_56 = wp::address(var_actuator_gainprm, var_6, var_1);
                var_58 = wp::load(var_42);
                var_59 = wp::load(var_43);
                var_60 = wp::load(var_49);
                var_61 = wp::load(var_55);
                var_62 = wp::load(var_56);
                var_57 = muscle_gain_vel_deriv_0(var_58, var_59, var_60, var_61, var_62);
            }
            if (!var_40) {
                // elif actuator_gaintype[actid] == GainType.DCMOTOR:                             <L 100>
                var_63 = wp::address(var_actuator_gaintype, var_1);
                var_66 = wp::load(var_63);
                var_65 = (var_66 == var_64);
                if (var_65) {
                    // gain = 0.0                                                                 <L 101>
                    // dynprm = actuator_dynprm[actuator_dynprm_id, actid]                        <L 102>
                    var_68 = wp::address(var_actuator_dynprm, var_16, var_1);
                    var_70 = wp::load(var_68);
                    var_69 = wp::copy(var_70);
                    // gainprm = actuator_gainprm[actuator_gainprm_id, actid]                     <L 103>
                    var_71 = wp::address(var_actuator_gainprm, var_6, var_1);
                    var_73 = wp::load(var_71);
                    var_72 = wp::copy(var_73);
                    // te = dynprm[0]                                                             <L 104>
                    var_75 = wp::extract(var_69, var_74);
                    // K = gainprm[1]                                                             <L 105>
                    var_77 = wp::extract(var_72, var_76);
                    // dVdw = 0.0                                                                 <L 108>
                    // if (actuator_ctrlspec[actid] & 7) != 0:                                    <L 109>
                    var_79 = wp::address(var_actuator_ctrlspec, var_1);
                    var_82 = wp::load(var_79);
                    var_81 = wp::bit_and(var_82, var_80);
                    var_84 = (var_81 != var_83);
                    if (var_84) {
                        // R0 = wp.max(MJ_MINVAL, gainprm[0])                                     <L 110>
                        var_87 = wp::extract(var_72, var_86);
                        var_88 = wp::max(var_85, var_87);
                        // dVdw = -gainprm[6] * R0 / K + K                                        <L 111>
                        var_90 = wp::extract(var_72, var_89);
                        var_91 = wp::neg(var_90);
                        var_92 = wp::mul(var_91, var_88);
                        var_93 = wp::div(var_92, var_77);
                        var_94 = wp::add(var_93, var_77);
                    }
                    var_95 = wp::where(var_84, var_94, var_78);
                    // R = util_misc.dcmotor_resistance(act_in, worldid, actuator_actadr[actid], dynprm, gainprm)       <L 114>
                    var_96 = wp::address(var_actuator_actadr, var_1);
                    var_98 = wp::load(var_96);
                    var_97 = dcmotor_resistance_0(var_act_in, var_0, var_98, var_69, var_72);
                    // if te > 0.0:                                                               <L 116>
                    var_100 = (var_75 > var_99);
                    if (var_100) {
                        // s = 1.0 - wp.exp(-opt_timestep[opt_timestep_id] / te)                  <L 119>
                        var_102 = wp::address(var_opt_timestep, var_21);
                        var_104 = wp::load(var_102);
                        var_103 = wp::neg(var_104);
                        var_105 = wp::div(var_103, var_75);
                        var_106 = wp::exp(var_105);
                        var_107 = wp::sub(var_101, var_106);
                        // bias += K * (dVdw - K) * s / R                                         <L 120>
                        var_108 = wp::sub(var_95, var_77);
                        var_109 = wp::mul(var_77, var_108);
                        var_110 = wp::mul(var_109, var_107);
                        var_111 = wp::div(var_110, var_97);
                        var_112 = wp::add(var_23, var_111);
                    }
                    if (!var_100) {
                        // elif dVdw != 0.0:                                                      <L 121>
                        var_114 = (var_95 != var_113);
                        if (var_114) {
                            // bias += K * dVdw / R                                               <L 123>
                            var_115 = wp::mul(var_77, var_95);
                            var_116 = wp::div(var_115, var_97);
                            var_117 = wp::add(var_23, var_116);
                        }
                        var_118 = wp::where(var_114, var_117, var_23);
                    }
                    var_119 = wp::where(var_100, var_112, var_118);
                    // sigma1 = dynprm[6]                                                         <L 127>
                    var_121 = wp::extract(var_69, var_120);
                    // if sigma1 > 0.0:                                                           <L 128>
                    var_123 = (var_121 > var_122);
                    if (var_123) {
                        // bias -= sigma1                                                         <L 129>
                        var_124 = wp::sub(var_119, var_121);
                    }
                    var_125 = wp::where(var_123, var_124, var_119);
                    // R_dcmotor = R                                                              <L 131>
                    var_126 = wp::copy(var_97);
                    // K_dcmotor = K                                                              <L 132>
                    var_127 = wp::copy(var_77);
                    // te_dcmotor = te                                                            <L 133>
                    var_128 = wp::copy(var_75);
                }
                if (!var_65) {
                    // gain = 0.0                                                                 <L 135>
                }
                var_130 = wp::where(var_65, var_125, var_23);
                var_131 = wp::where(var_65, var_126, var_25);
                var_132 = wp::where(var_65, var_127, var_27);
                var_133 = wp::where(var_65, var_128, var_29);
                var_134 = wp::where(var_65, var_67, var_129);
            }
            var_135 = wp::where(var_40, var_23, var_130);
            var_136 = wp::where(var_40, var_25, var_131);
            var_137 = wp::where(var_40, var_27, var_132);
            var_138 = wp::where(var_40, var_29, var_133);
            var_139 = wp::where(var_40, var_57, var_134);
        }
        var_140 = wp::where(var_32, var_23, var_135);
        var_141 = wp::where(var_32, var_25, var_136);
        var_142 = wp::where(var_32, var_27, var_137);
        var_143 = wp::where(var_32, var_29, var_138);
        var_144 = wp::where(var_32, var_36, var_139);
        // if actuator_biastype[actid] == BiasType.AFFINE:                                        <L 137>
        var_145 = wp::address(var_actuator_biastype, var_1);
        var_148 = wp::load(var_145);
        var_147 = (var_148 == var_146);
        if (var_147) {
            // bias += actuator_biasprm[actuator_biasprm_id, actid][2]                            <L 138>
            var_149 = wp::address(var_actuator_biasprm, var_11, var_1);
            var_152 = wp::load(var_149);
            var_151 = wp::extract(var_152, var_150);
            var_153 = wp::add(var_140, var_151);
        }
        if (!var_147) {
            // elif actuator_biastype[actid] == BiasType.DCMOTOR:                                 <L 139>
            var_154 = wp::address(var_actuator_biastype, var_1);
            var_157 = wp::load(var_154);
            var_156 = (var_157 == var_155);
            if (var_156) {
                // if R_dcmotor > 0.0:                                                            <L 140>
                var_159 = (var_141 > var_158);
                if (var_159) {
                    // if te_dcmotor <= 0.0:                                                      <L 141>
                    var_161 = (var_143 <= var_160);
                    if (var_161) {
                        // bias += -K_dcmotor * K_dcmotor / R_dcmotor                             <L 142>
                        var_162 = wp::neg(var_142);
                        var_163 = wp::mul(var_162, var_142);
                        var_164 = wp::div(var_163, var_141);
                        var_165 = wp::add(var_140, var_164);
                    }
                    var_166 = wp::where(var_161, var_165, var_140);
                }
                if (!var_159) {
                    // dynprm = actuator_dynprm[actuator_dynprm_id, actid]                        <L 144>
                    var_167 = wp::address(var_actuator_dynprm, var_16, var_1);
                    var_169 = wp::load(var_167);
                    var_168 = wp::copy(var_169);
                    // te = dynprm[0]                                                             <L 145>
                    var_171 = wp::extract(var_168, var_170);
                    // if te <= 0.0:                                                              <L 146>
                    var_173 = (var_171 <= var_172);
                    if (var_173) {
                        // gainprm = actuator_gainprm[actuator_gainprm_id, actid]                 <L 147>
                        var_174 = wp::address(var_actuator_gainprm, var_6, var_1);
                        var_176 = wp::load(var_174);
                        var_175 = wp::copy(var_176);
                        // K = gainprm[1]                                                         <L 148>
                        var_178 = wp::extract(var_175, var_177);
                        // R = util_misc.dcmotor_resistance(act_in, worldid, actuator_actadr[actid], dynprm, gainprm)       <L 149>
                        var_179 = wp::address(var_actuator_actadr, var_1);
                        var_181 = wp::load(var_179);
                        var_180 = dcmotor_resistance_0(var_act_in, var_0, var_181, var_168, var_175);
                        // bias += -K * K / R                                                     <L 150>
                        var_182 = wp::neg(var_178);
                        var_183 = wp::mul(var_182, var_178);
                        var_184 = wp::div(var_183, var_180);
                        var_185 = wp::add(var_140, var_184);
                    }
                    var_186 = wp::where(var_173, var_185, var_140);
                    var_187 = wp::where(var_173, var_175, var_72);
                    var_188 = wp::where(var_173, var_178, var_77);
                    var_189 = wp::where(var_173, var_180, var_97);
                }
                var_190 = wp::where(var_159, var_166, var_186);
                var_191 = wp::where(var_159, var_69, var_168);
                var_192 = wp::where(var_159, var_72, var_187);
                var_193 = wp::where(var_159, var_75, var_171);
                var_194 = wp::where(var_159, var_77, var_188);
                var_195 = wp::where(var_159, var_97, var_189);
            }
            var_196 = wp::where(var_156, var_190, var_140);
            var_197 = wp::where(var_156, var_191, var_69);
            var_198 = wp::where(var_156, var_192, var_72);
            var_199 = wp::where(var_156, var_193, var_75);
            var_200 = wp::where(var_156, var_194, var_77);
            var_201 = wp::where(var_156, var_195, var_97);
        }
        var_202 = wp::where(var_147, var_153, var_196);
        var_203 = wp::where(var_147, var_69, var_197);
        var_204 = wp::where(var_147, var_72, var_198);
        var_205 = wp::where(var_147, var_75, var_199);
        var_206 = wp::where(var_147, var_77, var_200);
        var_207 = wp::where(var_147, var_97, var_201);
        // if bias == 0.0 and gain == 0.0:                                                        <L 152>
        var_210 = (var_202 == var_209);
        var_208 = var_210;
        if (var_208) {
            var_212 = (var_144 == var_211);
            var_208 = var_208 && var_212;
        }
        if (var_208) {
            // vel_out[worldid, actid] = 0.0                                                      <L 153>
            wp::array_store(var_vel_out, var_0, var_1, var_213);
            // return                                                                             <L 154>
            return;
        }
        // if actuator_forcelimited[actid]:                                                       <L 157>
        var_214 = wp::address(var_actuator_forcelimited, var_1);
        var_215 = wp::load(var_214);
        if (var_215) {
            // force = actuator_force_in[worldid, actid]                                          <L 158>
            var_216 = wp::address(var_actuator_force_in, var_0, var_1);
            var_218 = wp::load(var_216);
            var_217 = wp::copy(var_218);
            // forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]       <L 159>
            var_219 = &(var_actuator_forcerange.shape);
            var_222 = wp::load(var_219);
            var_221 = wp::extract(var_222, var_220);
            var_223 = wp::mod(var_0, var_221);
            var_224 = wp::address(var_actuator_forcerange, var_223, var_1);
            var_226 = wp::load(var_224);
            var_225 = wp::copy(var_226);
            // if force <= forcerange[0] or force >= forcerange[1]:                               <L 160>
            var_229 = wp::extract(var_225, var_228);
            var_230 = (var_217 <= var_229);
            var_227 = var_230;
            if (!var_227) {
                var_232 = wp::extract(var_225, var_231);
                var_233 = (var_217 >= var_232);
                var_227 = var_227 || var_233;
            }
            if (var_227) {
                // vel_out[worldid, actid] = 0.0                                                  <L 161>
                wp::array_store(var_vel_out, var_0, var_1, var_234);
                // return                                                                         <L 162>
                return;
            }
        }
        var_235 = wp::load(var_214);
        // vel = float(bias)                                                                      <L 164>
        var_236 = wp::float(var_202);
        // if actuator_dyntype[actid] != DynType.NONE:                                            <L 165>
        var_237 = wp::address(var_actuator_dyntype, var_1);
        var_240 = wp::load(var_237);
        var_239 = (var_240 != var_238);
        if (var_239) {
            // if gain != 0.0:                                                                    <L 166>
            var_242 = (var_144 != var_241);
            if (var_242) {
                // act_adr = actuator_actadr[actid] + actuator_actnum[actid] - 1                  <L 167>
                var_243 = wp::address(var_actuator_actadr, var_1);
                var_244 = wp::address(var_actuator_actnum, var_1);
                var_246 = wp::load(var_243);
                var_247 = wp::load(var_244);
                var_245 = wp::add(var_246, var_247);
                var_249 = wp::sub(var_245, var_248);
                // if actuator_actearly[actid]:                                                   <L 170>
                var_250 = wp::address(var_actuator_actearly, var_1);
                var_251 = wp::load(var_250);
                if (var_251) {
                    // act = next_act(                                                            <L 171>
                    // opt_timestep[opt_timestep_id],                                             <L 172>
                    var_252 = wp::address(var_opt_timestep, var_21);
                    // actuator_dyntype[actid],                                                   <L 173>
                    var_253 = wp::address(var_actuator_dyntype, var_1);
                    // actuator_dynprm[actuator_dynprm_id, actid],                                <L 174>
                    var_254 = wp::address(var_actuator_dynprm, var_16, var_1);
                    // actuator_actrange[worldid % actuator_actrange.shape[0], actid],            <L 175>
                    var_255 = &(var_actuator_actrange.shape);
                    var_258 = wp::load(var_255);
                    var_257 = wp::extract(var_258, var_256);
                    var_259 = wp::mod(var_0, var_257);
                    var_260 = wp::address(var_actuator_actrange, var_259, var_1);
                    // act_in[worldid, act_adr],                                                  <L 176>
                    var_261 = wp::address(var_act_in, var_0, var_249);
                    // act_dot_in[worldid, act_adr],                                              <L 177>
                    var_262 = wp::address(var_act_dot_in, var_0, var_249);
                    // 1.0,                                                                       <L 178>
                    // actuator_actlimited[actid],                                                <L 179>
                    var_264 = wp::address(var_actuator_actlimited, var_1);
                    var_266 = wp::load(var_252);
                    var_267 = wp::load(var_253);
                    var_268 = wp::load(var_254);
                    var_269 = wp::load(var_260);
                    var_270 = wp::load(var_261);
                    var_271 = wp::load(var_262);
                    var_272 = wp::load(var_264);
                    var_265 = next_act_0(var_266, var_267, var_268, var_269, var_270, var_271, var_263, var_272);
                }
                var_273 = wp::load(var_250);
                var_274 = wp::load(var_250);
                if (!var_274) {
                    // act = act_in[worldid, act_adr]                                             <L 182>
                    var_275 = wp::address(var_act_in, var_0, var_249);
                    var_277 = wp::load(var_275);
                    var_276 = wp::copy(var_277);
                }
                var_278 = wp::load(var_250);
                var_280 = wp::load(var_250);
                var_279 = wp::where(var_280, var_265, var_276);
                // vel += gain * act                                                              <L 184>
                var_281 = wp::mul(var_144, var_279);
                var_282 = wp::add(var_236, var_281);
            }
            var_283 = wp::where(var_242, var_282, var_236);
        }
        if (!var_239) {
            // if gain != 0.0:                                                                    <L 186>
            var_285 = (var_144 != var_284);
            if (var_285) {
                // vel += gain * ctrl_in[worldid, actid]                                          <L 187>
                var_286 = wp::address(var_ctrl_in, var_0, var_1);
                var_288 = wp::load(var_286);
                var_287 = wp::mul(var_144, var_288);
                var_289 = wp::add(var_236, var_287);
            }
            var_290 = wp::where(var_285, var_289, var_236);
        }
        var_291 = wp::where(var_239, var_283, var_290);
        // vel_out[worldid, actid] = vel                                                          <L 189>
        wp::array_store(var_vel_out, var_0, var_1, var_291);
}



extern "C" __global__ void deriv_rne_cvel_cdof_dot_8e31d9b1_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::int32> var_body_parentid,
    wp::array_t<wp::int32> var_body_jntnum,
    wp::array_t<wp::int32> var_body_jntadr,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_jnt_type,
    wp::array_t<wp::vec_t<6, wp::float32>> var_cdof_in,
    wp::array_t<wp::int32> var_body_tree_,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcvel_out,
    wp::array_t<wp::vec_t<6, wp::float32>> var_Dcdof_dot_out)
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
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::array_t<wp::vec_t<6, wp::float32>> var_18;
        wp::vec_t<6, wp::float32>* var_19;
        wp::vec_t<6, wp::float32> var_20;
        wp::vec_t<6, wp::float32> var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 0;
        bool var_32;
        bool var_33;
        bool var_34;
        const wp::int32 var_35 = 3;
        wp::int32 var_36;
        bool var_37;
        wp::vec_t<6, wp::float32>* var_38;
        wp::vec_t<6, wp::float32> var_39;
        wp::vec_t<6, wp::float32> var_40;
        wp::vec_t<6, wp::float32> var_41;
        const wp::int32 var_42 = 3;
        wp::int32 var_43;
        wp::vec_t<6, wp::float32>* var_44;
        wp::vec_t<6, wp::float32> var_45;
        wp::vec_t<6, wp::float32> var_46;
        const wp::int32 var_47 = 3;
        wp::int32 var_48;
        const wp::int32 var_49 = 4;
        wp::int32 var_50;
        wp::vec_t<6, wp::float32>* var_51;
        wp::vec_t<6, wp::float32> var_52;
        wp::vec_t<6, wp::float32> var_53;
        const wp::int32 var_54 = 4;
        wp::int32 var_55;
        const wp::int32 var_56 = 5;
        wp::int32 var_57;
        wp::vec_t<6, wp::float32>* var_58;
        wp::vec_t<6, wp::float32> var_59;
        wp::vec_t<6, wp::float32> var_60;
        const wp::int32 var_61 = 5;
        wp::int32 var_62;
        bool var_63;
        const wp::int32 var_64 = 3;
        wp::int32 var_65;
        bool var_66;
        const wp::int32 var_67 = 6;
        wp::int32 var_68;
        bool var_69;
        wp::vec_t<6, wp::float32>* var_70;
        wp::vec_t<6, wp::float32> var_71;
        wp::vec_t<6, wp::float32> var_72;
        wp::vec_t<6, wp::float32> var_73;
        const wp::int32 var_74 = 6;
        wp::int32 var_75;
        const wp::int32 var_76 = 1;
        bool var_77;
        const wp::int32 var_78 = 0;
        wp::int32 var_79;
        wp::vec_t<6, wp::float32>* var_80;
        wp::vec_t<6, wp::float32> var_81;
        wp::vec_t<6, wp::float32> var_82;
        const wp::int32 var_83 = 0;
        wp::int32 var_84;
        const wp::int32 var_85 = 1;
        wp::int32 var_86;
        wp::vec_t<6, wp::float32>* var_87;
        wp::vec_t<6, wp::float32> var_88;
        wp::vec_t<6, wp::float32> var_89;
        const wp::int32 var_90 = 1;
        wp::int32 var_91;
        const wp::int32 var_92 = 2;
        wp::int32 var_93;
        wp::vec_t<6, wp::float32>* var_94;
        wp::vec_t<6, wp::float32> var_95;
        wp::vec_t<6, wp::float32> var_96;
        const wp::int32 var_97 = 2;
        wp::int32 var_98;
        bool var_99;
        bool var_100;
        const wp::int32 var_101 = 3;
        wp::int32 var_102;
        bool var_103;
        wp::vec_t<6, wp::float32>* var_104;
        wp::vec_t<6, wp::float32> var_105;
        wp::vec_t<6, wp::float32> var_106;
        wp::vec_t<6, wp::float32> var_107;
        const wp::int32 var_108 = 3;
        wp::int32 var_109;
        wp::vec_t<6, wp::float32>* var_110;
        wp::vec_t<6, wp::float32> var_111;
        wp::vec_t<6, wp::float32> var_112;
        bool var_113;
        wp::vec_t<6, wp::float32>* var_114;
        wp::vec_t<6, wp::float32> var_115;
        wp::vec_t<6, wp::float32> var_116;
        wp::vec_t<6, wp::float32> var_117;
        const wp::int32 var_118 = 1;
        wp::int32 var_119;
        wp::vec_t<6, wp::float32> var_120;
        wp::int32 var_121;
        wp::vec_t<6, wp::float32> var_122;
        wp::int32 var_123;
        //---------
        // forward
        // def deriv_rne_cvel_cdof_dot(                                                           <L 345>
        // worldid, nodeid, dofid = wp.tid()                                                      <L 368>
        builtin_tid3d(var_0, var_1, var_2);
        // bodyid = body_tree_[nodeid]                                                            <L 369>
        var_3 = wp::address(var_body_tree_, var_1);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // dofadr = body_dofadr[bodyid]                                                           <L 370>
        var_6 = wp::address(var_body_dofadr, var_4);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // jntid = body_jntadr[bodyid]                                                            <L 371>
        var_9 = wp::address(var_body_jntadr, var_4);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // jntnum = body_jntnum[bodyid]                                                           <L 372>
        var_12 = wp::address(var_body_jntnum, var_4);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // pid = body_parentid[bodyid]                                                            <L 373>
        var_15 = wp::address(var_body_parentid, var_4);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // cdof = cdof_in[worldid]                                                                <L 375>
        var_18 = wp::view(var_cdof_in, var_0);
        // cvel_k = Dcvel_out[worldid, pid, dofid]                                                <L 378>
        var_19 = wp::address(var_Dcvel_out, var_0, var_16, var_2);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // if jntnum == 0:                                                                        <L 380>
        var_23 = (var_13 == var_22);
        if (var_23) {
            // Dcvel_out[worldid, bodyid, dofid] = cvel_k                                         <L 381>
            wp::array_store(var_Dcvel_out, var_0, var_4, var_2, var_20);
            // return                                                                             <L 382>
            return;
        }
        // dof_i = dofadr                                                                         <L 384>
        var_24 = wp::copy(var_7);
        // for j in range(jntid, jntid + jntnum):                                                 <L 386>
        var_25 = wp::add(var_10, var_13);
        var_26 = wp::range(var_10, var_25);
        start_for_1:;
            if (iter_cmp(var_26) == 0) goto end_for_1;
            var_27 = wp::iter_next(var_26);
            // jnttype = jnt_type[j]                                                              <L 387>
            var_28 = wp::address(var_jnt_type, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::copy(var_30);
            // if jnttype == 0:  # FREE                                                           <L 389>
            var_32 = (var_29 == var_31);
            if (var_32) {
                // if dofid >= dof_i and dofid < dof_i + 3:                                       <L 391>
                var_34 = (var_2 >= var_24);
                var_33 = var_34;
                if (var_33) {
                    var_36 = wp::add(var_24, var_35);
                    var_37 = (var_2 < var_36);
                    var_33 = var_33 && var_37;
                }
                if (var_33) {
                    // cvel_k += cdof[dofid]                                                      <L 392>
                    var_38 = wp::address(var_18, var_2);
                    var_40 = wp::load(var_38);
                    var_39 = wp::add(var_20, var_40);
                }
                var_41 = wp::where(var_33, var_39, var_20);
                // Dcdof_dot_out[worldid, dof_i + 3, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 3])       <L 398>
                var_43 = wp::add(var_24, var_42);
                var_44 = wp::address(var_18, var_43);
                var_46 = wp::load(var_44);
                var_45 = motion_cross_0(var_41, var_46);
                var_48 = wp::add(var_24, var_47);
                wp::array_store(var_Dcdof_dot_out, var_0, var_48, var_2, var_45);
                // Dcdof_dot_out[worldid, dof_i + 4, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 4])       <L 399>
                var_50 = wp::add(var_24, var_49);
                var_51 = wp::address(var_18, var_50);
                var_53 = wp::load(var_51);
                var_52 = motion_cross_0(var_41, var_53);
                var_55 = wp::add(var_24, var_54);
                wp::array_store(var_Dcdof_dot_out, var_0, var_55, var_2, var_52);
                // Dcdof_dot_out[worldid, dof_i + 5, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 5])       <L 400>
                var_57 = wp::add(var_24, var_56);
                var_58 = wp::address(var_18, var_57);
                var_60 = wp::load(var_58);
                var_59 = motion_cross_0(var_41, var_60);
                var_62 = wp::add(var_24, var_61);
                wp::array_store(var_Dcdof_dot_out, var_0, var_62, var_2, var_59);
                // if dofid >= dof_i + 3 and dofid < dof_i + 6:                                   <L 403>
                var_65 = wp::add(var_24, var_64);
                var_66 = (var_2 >= var_65);
                var_63 = var_66;
                if (var_63) {
                    var_68 = wp::add(var_24, var_67);
                    var_69 = (var_2 < var_68);
                    var_63 = var_63 && var_69;
                }
                if (var_63) {
                    // cvel_k += cdof[dofid]                                                      <L 404>
                    var_70 = wp::address(var_18, var_2);
                    var_72 = wp::load(var_70);
                    var_71 = wp::add(var_41, var_72);
                }
                var_73 = wp::where(var_63, var_71, var_41);
                // dof_i += 6                                                                     <L 406>
                var_75 = wp::add(var_24, var_74);
            }
            if (!var_32) {
                // elif jnttype == 1:  # BALL                                                     <L 408>
                var_77 = (var_29 == var_76);
                if (var_77) {
                    // Dcdof_dot_out[worldid, dof_i + 0, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 0])       <L 409>
                    var_79 = wp::add(var_24, var_78);
                    var_80 = wp::address(var_18, var_79);
                    var_82 = wp::load(var_80);
                    var_81 = motion_cross_0(var_20, var_82);
                    var_84 = wp::add(var_24, var_83);
                    wp::array_store(var_Dcdof_dot_out, var_0, var_84, var_2, var_81);
                    // Dcdof_dot_out[worldid, dof_i + 1, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 1])       <L 410>
                    var_86 = wp::add(var_24, var_85);
                    var_87 = wp::address(var_18, var_86);
                    var_89 = wp::load(var_87);
                    var_88 = motion_cross_0(var_20, var_89);
                    var_91 = wp::add(var_24, var_90);
                    wp::array_store(var_Dcdof_dot_out, var_0, var_91, var_2, var_88);
                    // Dcdof_dot_out[worldid, dof_i + 2, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 2])       <L 411>
                    var_93 = wp::add(var_24, var_92);
                    var_94 = wp::address(var_18, var_93);
                    var_96 = wp::load(var_94);
                    var_95 = motion_cross_0(var_20, var_96);
                    var_98 = wp::add(var_24, var_97);
                    wp::array_store(var_Dcdof_dot_out, var_0, var_98, var_2, var_95);
                    // if dofid >= dof_i and dofid < dof_i + 3:                                   <L 413>
                    var_100 = (var_2 >= var_24);
                    var_99 = var_100;
                    if (var_99) {
                        var_102 = wp::add(var_24, var_101);
                        var_103 = (var_2 < var_102);
                        var_99 = var_99 && var_103;
                    }
                    if (var_99) {
                        // cvel_k += cdof[dofid]                                                  <L 414>
                        var_104 = wp::address(var_18, var_2);
                        var_106 = wp::load(var_104);
                        var_105 = wp::add(var_20, var_106);
                    }
                    var_107 = wp::where(var_99, var_105, var_20);
                    // dof_i += 3                                                                 <L 416>
                    var_109 = wp::add(var_24, var_108);
                }
                if (!var_77) {
                    // Dcdof_dot_out[worldid, dof_i, dofid] = math.motion_cross(cvel_k, cdof[dof_i])       <L 418>
                    var_110 = wp::address(var_18, var_24);
                    var_112 = wp::load(var_110);
                    var_111 = motion_cross_0(var_20, var_112);
                    wp::array_store(var_Dcdof_dot_out, var_0, var_24, var_2, var_111);
                    // if dofid == dof_i:                                                         <L 420>
                    var_113 = (var_2 == var_24);
                    if (var_113) {
                        // cvel_k += cdof[dof_i]                                                  <L 421>
                        var_114 = wp::address(var_18, var_24);
                        var_116 = wp::load(var_114);
                        var_115 = wp::add(var_20, var_116);
                    }
                    var_117 = wp::where(var_113, var_115, var_20);
                    // dof_i += 1                                                                 <L 423>
                    var_119 = wp::add(var_24, var_118);
                }
                var_120 = wp::where(var_77, var_107, var_117);
                var_121 = wp::where(var_77, var_109, var_119);
            }
            var_122 = wp::where(var_32, var_73, var_120);
            var_123 = wp::where(var_32, var_75, var_121);
            wp::assign(var_20, var_122);
            wp::assign(var_24, var_123);
            goto start_for_1;
        end_for_1:;
        // Dcvel_out[worldid, bodyid, dofid] = cvel_k                                             <L 425>
        wp::array_store(var_Dcvel_out, var_0, var_4, var_2, var_20);
}



extern "C" __global__ void _qderiv_actuator_passive_actuation_sparse_9edf10e9_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_vel_in,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        const wp::float32 var_5 = 0.0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::float32* var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        const wp::float32 var_19 = 0.0;
        bool var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.0;
        bool var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        const wp::int32 var_40 = 0;
        bool var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::array_t<wp::float32> var_44;
        wp::float32 var_45;
        //---------
        // forward
        // def _qderiv_actuator_passive_actuation_sparse(                                         <L 201>
        // worldid, actid = wp.tid()                                                              <L 214>
        builtin_tid2d(var_0, var_1);
        // vel = vel_in[worldid, actid]                                                           <L 216>
        var_2 = wp::address(var_vel_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if vel == 0.0:                                                                         <L 217>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 218>
            return;
        }
        // rownnz = moment_rownnz_in[worldid, actid]                                              <L 220>
        var_7 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // rowadr = moment_rowadr_in[worldid, actid]                                              <L 221>
        var_10 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // for i in range(rownnz):                                                                <L 223>
        var_13 = wp::range(var_8);
        start_for_1:;
            if (iter_cmp(var_13) == 0) goto end_for_1;
            var_14 = wp::iter_next(var_13);
            // rowadri = rowadr + i                                                               <L 224>
            var_15 = wp::add(var_11, var_14);
            // moment_i = actuator_moment_in[worldid, rowadri]                                    <L 225>
            var_16 = wp::address(var_actuator_moment_in, var_0, var_15);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // if moment_i == 0.0:                                                                <L 226>
            var_20 = (var_17 == var_19);
            if (var_20) {
                // continue                                                                       <L 227>
                goto start_for_1;
            }
            // dofi = moment_colind_in[worldid, rowadri]                                          <L 228>
            var_21 = wp::address(var_moment_colind_in, var_0, var_15);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // for j in range(i + 1):                                                             <L 230>
            var_25 = wp::add(var_14, var_24);
            var_26 = wp::range(var_25);
            start_for_3:;
                if (iter_cmp(var_26) == 0) goto end_for_3;
                var_27 = wp::iter_next(var_26);
                // rowadrj = rowadr + j                                                           <L 231>
                var_28 = wp::add(var_11, var_27);
                // moment_j = actuator_moment_in[worldid, rowadrj]                                <L 232>
                var_29 = wp::address(var_actuator_moment_in, var_0, var_28);
                var_31 = wp::load(var_29);
                var_30 = wp::copy(var_31);
                // if moment_j == 0.0:                                                            <L 233>
                var_33 = (var_30 == var_32);
                if (var_33) {
                    // continue                                                                   <L 234>
                    goto start_for_3;
                }
                // dofj = moment_colind_in[worldid, rowadrj]                                      <L 235>
                var_34 = wp::address(var_moment_colind_in, var_0, var_28);
                var_36 = wp::load(var_34);
                var_35 = wp::copy(var_36);
                // elemid = M_elemid[dofi, dofj]                                                  <L 237>
                var_37 = wp::address(var_M_elemid, var_22, var_35);
                var_39 = wp::load(var_37);
                var_38 = wp::copy(var_39);
                // if elemid >= 0:                                                                <L 238>
                var_41 = (var_38 >= var_40);
                if (var_41) {
                    // contrib = moment_i * moment_j * vel                                        <L 239>
                    var_42 = wp::mul(var_17, var_30);
                    var_43 = wp::mul(var_42, var_3);
                    // wp.atomic_add(qDeriv_out[worldid], elemid, contrib)                        <L 240>
                    var_44 = wp::view(var_qDeriv_out, var_0);
                    var_45 = wp::atomic_add(var_44, var_38, var_43);
                }
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
}



extern "C" __global__ void _eff_copy_backbone_rhs_9f29ad70_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_nv,
    wp::array_t<wp::int32> var_efm_dofblk,
    wp::array_t<wp::float32> var_vec_in,
    wp::array_t<wp::float32> var_vec_out)
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
        bool var_2;
        wp::int32* var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::int32 var_6;
        const wp::float32 var_7 = 0.0;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        //---------
        // forward
        // def _eff_copy_backbone_rhs(                                                            <L 2470>
        // worldid, dofid = wp.tid()                                                              <L 2479>
        builtin_tid2d(var_0, var_1);
        // if dofid < nv:                                                                         <L 2480>
        var_2 = (var_1 < var_nv);
        if (var_2) {
            // vec_out[worldid, dofid] = 0.0 if efm_dofblk[dofid] >= 0 else vec_in[worldid, dofid]       <L 2481>
            var_3 = wp::address(var_efm_dofblk, var_1);
            var_6 = wp::load(var_3);
            var_5 = (var_6 >= var_4);
            if (var_5) {
            }
            if (!var_5) {
                var_8 = wp::address(var_vec_in, var_0, var_1);
                var_10 = wp::load(var_8);
                var_9 = wp::copy(var_10);
            }
            var_11 = wp::where(var_5, var_7, var_9);
            wp::array_store(var_vec_out, var_0, var_1, var_11);
        }
}



extern "C" __global__ void _eff_fold_actuator_a79b9be3_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_efm_dofid,
    wp::array_t<wp::int32> var_efm_dofblk,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_efm_as_in,
    wp::array_t<wp::float32> var_epB_out)
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
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        const wp::float32 var_5 = 0.0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::float32* var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::range_t var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        bool var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 9;
        wp::int32 var_45;
        const wp::int32 var_46 = 3;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::float32 var_52;
        wp::float32 var_53;
        //---------
        // forward
        // def _eff_fold_actuator(                                                                <L 3624>
        // worldid, u = wp.tid()                                                                  <L 3637>
        builtin_tid2d(var_0, var_1);
        // s = efm_as_in[worldid, u]                                                              <L 3638>
        var_2 = wp::address(var_efm_as_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if s == 0.0:                                                                           <L 3639>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 3640>
            return;
        }
        // rowadr = moment_rowadr_in[worldid, u]                                                  <L 3641>
        var_7 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // rownnz = moment_rownnz_in[worldid, u]                                                  <L 3642>
        var_10 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // for a in range(rownnz):                                                                <L 3643>
        var_13 = wp::range(var_11);
        start_for_1:;
            if (iter_cmp(var_13) == 0) goto end_for_1;
            var_14 = wp::iter_next(var_13);
            // ia = moment_colind_in[worldid, rowadr + a]                                         <L 3644>
            var_15 = wp::add(var_8, var_14);
            var_16 = wp::address(var_moment_colind_in, var_0, var_15);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // k = efm_dofblk[ia]                                                                 <L 3645>
            var_19 = wp::address(var_efm_dofblk, var_17);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // if k < 0:                                                                          <L 3646>
            var_23 = (var_20 < var_22);
            if (var_23) {
                // continue                                                                       <L 3647>
                goto start_for_1;
            }
            // base = efm_dofid[k]                                                                <L 3648>
            var_24 = wp::address(var_efm_dofid, var_20);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // s_va = s * actuator_moment_in[worldid, rowadr + a]                                 <L 3649>
            var_27 = wp::add(var_8, var_14);
            var_28 = wp::address(var_actuator_moment_in, var_0, var_27);
            var_30 = wp::load(var_28);
            var_29 = wp::mul(var_3, var_30);
            // for b in range(rownnz):                                                            <L 3650>
            var_31 = wp::range(var_11);
            start_for_3:;
                if (iter_cmp(var_31) == 0) goto end_for_3;
                var_32 = wp::iter_next(var_31);
                // ib = moment_colind_in[worldid, rowadr + b]                                     <L 3651>
                var_33 = wp::add(var_8, var_32);
                var_34 = wp::address(var_moment_colind_in, var_0, var_33);
                var_36 = wp::load(var_34);
                var_35 = wp::copy(var_36);
                // if efm_dofblk[ib] == k:                                                        <L 3652>
                var_37 = wp::address(var_efm_dofblk, var_35);
                var_39 = wp::load(var_37);
                var_38 = (var_39 == var_20);
                if (var_38) {
                    // vb = actuator_moment_in[worldid, rowadr + b]                               <L 3653>
                    var_40 = wp::add(var_8, var_32);
                    var_41 = wp::address(var_actuator_moment_in, var_0, var_40);
                    var_43 = wp::load(var_41);
                    var_42 = wp::copy(var_43);
                    // wp.atomic_add(epB_out, worldid, 9 * k + 3 * (ia - base) + (ib - base), s_va * vb)       <L 3654>
                    var_45 = wp::mul(var_44, var_20);
                    var_47 = wp::sub(var_17, var_25);
                    var_48 = wp::mul(var_46, var_47);
                    var_49 = wp::add(var_45, var_48);
                    var_50 = wp::sub(var_35, var_25);
                    var_51 = wp::add(var_49, var_50);
                    var_52 = wp::mul(var_29, var_42);
                    var_53 = wp::atomic_add(var_epB_out, var_0, var_51, var_52);
                }
                goto start_for_3;
            end_for_3:;
            goto start_for_1;
        end_for_1:;
}



extern "C" __global__ void _eff_add_actuator_qH_diag_ae1480da_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_efm_as_in,
    wp::array_t<wp::float32> var_qH_out)
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
        wp::float32* var_2;
        wp::float32 var_3;
        wp::float32 var_4;
        const wp::float32 var_5 = 0.0;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::range_t var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::int32* var_24;
        wp::int32* var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::float32 var_31;
        //---------
        // forward
        // def _eff_add_actuator_qH_diag(                                                         <L 1754>
        // worldid, a = wp.tid()                                                                  <L 1767>
        builtin_tid2d(var_0, var_1);
        // as_val = efm_as_in[worldid, a]                                                         <L 1768>
        var_2 = wp::address(var_efm_as_in, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if as_val == 0.0:                                                                      <L 1769>
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 1770>
            return;
        }
        // rowadr = moment_rowadr_in[worldid, a]                                                  <L 1771>
        var_7 = wp::address(var_moment_rowadr_in, var_0, var_1);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // rownnz = moment_rownnz_in[worldid, a]                                                  <L 1772>
        var_10 = wp::address(var_moment_rownnz_in, var_0, var_1);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // for j in range(rownnz):                                                                <L 1773>
        var_13 = wp::range(var_11);
        start_for_1:;
            if (iter_cmp(var_13) == 0) goto end_for_1;
            var_14 = wp::iter_next(var_13);
            // sparseid = rowadr + j                                                              <L 1774>
            var_15 = wp::add(var_8, var_14);
            // dof = moment_colind_in[worldid, sparseid]                                          <L 1775>
            var_16 = wp::address(var_moment_colind_in, var_0, var_15);
            var_18 = wp::load(var_16);
            var_17 = wp::copy(var_18);
            // val = actuator_moment_in[worldid, sparseid]                                        <L 1776>
            var_19 = wp::address(var_actuator_moment_in, var_0, var_15);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // term = as_val * val * val                                                          <L 1777>
            var_22 = wp::mul(var_3, var_20);
            var_23 = wp::mul(var_22, var_20);
            // diag_elem = M_rowadr[dof] + M_rownnz[dof] - 1                                      <L 1778>
            var_24 = wp::address(var_M_rowadr, var_17);
            var_25 = wp::address(var_M_rownnz, var_17);
            var_27 = wp::load(var_24);
            var_28 = wp::load(var_25);
            var_26 = wp::add(var_27, var_28);
            var_30 = wp::sub(var_26, var_29);
            // wp.atomic_add(qH_out, worldid, diag_elem, term)                                    <L 1779>
            var_31 = wp::atomic_add(var_qH_out, var_0, var_30, var_23);
            goto start_for_1;
        end_for_1:;
}



extern "C" __global__ void _eff_factor_blocks_b035ef86_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_M_rownnz,
    wp::array_t<wp::int32> var_M_rowadr,
    wp::array_t<wp::int32> var_M_colind,
    wp::array_t<wp::int32> var_efm_K_rownnz,
    wp::array_t<wp::int32> var_efm_K_rowadr,
    wp::array_t<wp::int32> var_efm_K_colind,
    wp::array_t<wp::int32> var_efm_dofid,
    wp::array_t<wp::float32> var_M_in,
    wp::array_t<wp::float32> var_efm_K_val_in,
    wp::array_t<wp::float32> var_efm_L_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::mat_t<3, 3, wp::float32> var_5;
        const wp::int32 var_6 = 9;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        const wp::int32 var_9 = 0;
        wp::float32 var_10;
        const wp::int32 var_11 = 1;
        const wp::int32 var_12 = 0;
        wp::float32 var_13;
        const wp::int32 var_14 = 1;
        const wp::int32 var_15 = 1;
        wp::float32 var_16;
        const wp::int32 var_17 = 2;
        const wp::int32 var_18 = 0;
        wp::float32 var_19;
        const wp::int32 var_20 = 2;
        const wp::int32 var_21 = 1;
        wp::float32 var_22;
        const wp::int32 var_23 = 2;
        const wp::int32 var_24 = 2;
        wp::float32 var_25;
        //---------
        // forward
        // def _eff_factor_blocks(                                                                <L 2409>
        // worldid, k = wp.tid()                                                                  <L 2424>
        builtin_tid2d(var_0, var_1);
        // i = efm_dofid[k]                                                                       <L 2425>
        var_2 = wp::address(var_efm_dofid, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // B = _extract_3x3_diag_block(                                                           <L 2426>
        // M_rownnz, M_rowadr, M_colind, efm_K_rownnz, efm_K_rowadr, efm_K_colind, M_in, efm_K_val_in, i, worldid       <L 2427>
        var_5 = _extract_3x3_diag_block_0(var_M_rownnz, var_M_rowadr, var_M_colind, var_efm_K_rownnz, var_efm_K_rowadr, var_efm_K_colind, var_M_in, var_efm_K_val_in, var_3, var_0);
        // _write_cholesky33(worldid, 9 * k, B[0, 0], B[1, 0], B[1, 1], B[2, 0], B[2, 1], B[2, 2], efm_L_out)       <L 2429>
        var_7 = wp::mul(var_6, var_1);
        var_10 = wp::extract(var_5, var_8, var_9);
        var_13 = wp::extract(var_5, var_11, var_12);
        var_16 = wp::extract(var_5, var_14, var_15);
        var_19 = wp::extract(var_5, var_17, var_18);
        var_22 = wp::extract(var_5, var_20, var_21);
        var_25 = wp::extract(var_5, var_23, var_24);
        _write_cholesky33_0(var_0, var_7, var_10, var_13, var_16, var_19, var_22, var_25, var_efm_L_out);
}



extern "C" __global__ void _eff_contact_shift_bf04a790_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::float32> var_qvel_in,
    wp::array_t<wp::int32> var_contact_worldid_in,
    wp::array_t<wp::int32> var_efm_con_dof_in,
    wp::array_t<wp::float32> var_efm_con_val_in,
    wp::array_t<wp::float32> var_efm_con_scale_in,
    wp::array_t<wp::int32> var_efm_con_nnz_in,
    wp::array_t<wp::float32> var_efm_c_out)
{
    wp::tile_shared_storage_t tile_mem;

    const size_t _idx = static_cast<size_t>(blockIdx.z * gridDim.y + blockIdx.y) * static_cast<size_t>(gridDim.x * blockDim.x) + static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (_idx >= dim.size) return;
            // reset shared memory allocator
    wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::float32* var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        const wp::float32 var_9 = 0.0;
        wp::float32 var_10;
        bool var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::shape_t* var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::shape_t var_18;
        wp::int32 var_19;
        wp::float32* var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::float32 var_25 = 0.0;
        wp::float32 var_26;
        wp::range_t var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::float32* var_32;
        wp::float32* var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::range_t var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        //---------
        // forward
        // def _eff_contact_shift(                                                                <L 2884>
        // cid = wp.tid()                                                                         <L 2899>
        var_0 = builtin_tid1d();
        // nnz = efm_con_nnz_in[cid]                                                              <L 2900>
        var_1 = wp::address(var_efm_con_nnz_in, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if nnz == 0:                                                                           <L 2901>
        var_5 = (var_2 == var_4);
        if (var_5) {
            // return                                                                             <L 2902>
            return;
        }
        // scale = efm_con_scale_in[cid]                                                          <L 2903>
        var_6 = wp::address(var_efm_con_scale_in, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // if scale == float(0.0):                                                                <L 2904>
        var_10 = wp::float(var_9);
        var_11 = (var_7 == var_10);
        if (var_11) {
            // return                                                                             <L 2905>
            return;
        }
        // worldid = contact_worldid_in[cid]                                                      <L 2906>
        var_12 = wp::address(var_contact_worldid_in, var_0);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 2907>
        var_15 = &(var_opt_timestep.shape);
        var_18 = wp::load(var_15);
        var_17 = wp::extract(var_18, var_16);
        var_19 = wp::mod(var_13, var_17);
        var_20 = wp::address(var_opt_timestep, var_19);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // neg_scale_dt = -scale / timestep                                                       <L 2908>
        var_23 = wp::neg(var_7);
        var_24 = wp::div(var_23, var_21);
        // dot = float(0.0)                                                                       <L 2909>
        var_26 = wp::float(var_25);
        // for a in range(nnz):                                                                   <L 2910>
        var_27 = wp::range(var_2);
        start_for_2:;
            if (iter_cmp(var_27) == 0) goto end_for_2;
            var_28 = wp::iter_next(var_27);
            // dof = efm_con_dof_in[cid, a]                                                       <L 2911>
            var_29 = wp::address(var_efm_con_dof_in, var_0, var_28);
            var_31 = wp::load(var_29);
            var_30 = wp::copy(var_31);
            // dot += efm_con_val_in[cid, a] * qvel_in[worldid, dof]                              <L 2912>
            var_32 = wp::address(var_efm_con_val_in, var_0, var_28);
            var_33 = wp::address(var_qvel_in, var_13, var_30);
            var_35 = wp::load(var_32);
            var_36 = wp::load(var_33);
            var_34 = wp::mul(var_35, var_36);
            var_37 = wp::add(var_26, var_34);
            wp::assign(var_26, var_37);
            goto start_for_2;
        end_for_2:;
        // scaled_dot = dot * neg_scale_dt                                                        <L 2913>
        var_38 = wp::mul(var_26, var_24);
        // for a in range(nnz):                                                                   <L 2914>
        var_39 = wp::range(var_2);
        start_for_4:;
            if (iter_cmp(var_39) == 0) goto end_for_4;
            var_40 = wp::iter_next(var_39);
            // dof = efm_con_dof_in[cid, a]                                                       <L 2915>
            var_41 = wp::address(var_efm_con_dof_in, var_0, var_40);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // wp.atomic_add(efm_c_out, worldid, dof, scaled_dot * efm_con_val_in[cid, a])        <L 2916>
            var_44 = wp::address(var_efm_con_val_in, var_0, var_40);
            var_46 = wp::load(var_44);
            var_45 = wp::mul(var_38, var_46);
            var_47 = wp::atomic_add(var_efm_c_out, var_13, var_42, var_45);
            wp::assign(var_30, var_42);
            goto start_for_4;
        end_for_4:;
}



extern "C" __global__ void _eff_tendon_shift_c1db54ef_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_stiffness,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_stiffnesspoly,
    wp::array_t<wp::float32> var_tendon_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_dampingpoly,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_lengthspring,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_length_in,
    wp::array_t<wp::float32> var_ten_velocity_in,
    wp::array_t<wp::float32> var_efm_c_out,
    wp::array_t<wp::float32> var_efm_ts_out)
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
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 0.0;
        wp::float32 var_11;
        const wp::int32 var_12 = 32;
        wp::int32 var_13;
        bool var_14;
        wp::float32* var_15;
        wp::float32 var_16;
        wp::float32 var_17;
        wp::shape_t* var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        wp::shape_t var_21;
        wp::int32 var_22;
        wp::vec_t<2, wp::float32>* var_23;
        wp::vec_t<2, wp::float32> var_24;
        wp::vec_t<2, wp::float32> var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        const wp::int32 var_28 = 1;
        wp::float32 var_29;
        const wp::float32 var_30 = 0.0;
        wp::float32 var_31;
        bool var_32;
        wp::float32 var_33;
        bool var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        const wp::float32 var_38 = 0.0;
        bool var_39;
        wp::shape_t* var_40;
        const wp::int32 var_41 = 0;
        wp::int32 var_42;
        wp::shape_t var_43;
        wp::int32 var_44;
        wp::float32* var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::shape_t* var_48;
        const wp::int32 var_49 = 0;
        wp::int32 var_50;
        wp::shape_t var_51;
        wp::int32 var_52;
        wp::vec_t<2, wp::float32>* var_53;
        wp::vec_t<2, wp::float32> var_54;
        wp::vec_t<2, wp::float32> var_55;
        const wp::float32 var_56 = 0.0;
        const wp::int32 var_57 = 0;
        wp::float32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        const wp::float32 var_62 = 0.0;
        wp::float32 var_63;
        const wp::int32 var_64 = 64;
        wp::int32 var_65;
        bool var_66;
        wp::shape_t* var_67;
        const wp::int32 var_68 = 0;
        wp::int32 var_69;
        wp::shape_t var_70;
        wp::int32 var_71;
        wp::float32* var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::shape_t* var_75;
        const wp::int32 var_76 = 0;
        wp::int32 var_77;
        wp::shape_t var_78;
        wp::int32 var_79;
        wp::vec_t<2, wp::float32>* var_80;
        wp::vec_t<2, wp::float32> var_81;
        wp::vec_t<2, wp::float32> var_82;
        wp::float32* var_83;
        wp::float32 var_84;
        wp::float32 var_85;
        const wp::float32 var_86 = 0.0;
        const wp::int32 var_87 = 1;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        const wp::float32 var_96 = 0.0;
        bool var_97;
        wp::float32* var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::int32* var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        wp::int32* var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::range_t var_107;
        wp::int32 var_108;
        wp::int32 var_109;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::float32* var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::float32 var_117;
        //---------
        // forward
        // def _eff_tendon_shift(                                                                 <L 1468>
        // worldid, tenid = wp.tid()                                                              <L 1488>
        builtin_tid2d(var_0, var_1);
        // h = opt_timestep[worldid % opt_timestep.shape[0]]                                      <L 1489>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // k = float(0.0)                                                                         <L 1491>
        var_11 = wp::float(var_10);
        // if not (opt_disableflags & DisableBit.SPRING):                                         <L 1492>
        var_13 = wp::bit_and(var_opt_disableflags, var_12);
        var_14 = wp::unot(var_13);
        if (var_14) {
            // length = ten_length_in[worldid, tenid]                                             <L 1493>
            var_15 = wp::address(var_ten_length_in, var_0, var_1);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            // range_spring = tendon_lengthspring[worldid % tendon_lengthspring.shape[0], tenid]       <L 1494>
            var_18 = &(var_tendon_lengthspring.shape);
            var_21 = wp::load(var_18);
            var_20 = wp::extract(var_21, var_19);
            var_22 = wp::mod(var_0, var_20);
            var_23 = wp::address(var_tendon_lengthspring, var_22, var_1);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // lower = range_spring[0]                                                            <L 1495>
            var_27 = wp::extract(var_24, var_26);
            // upper = range_spring[1]                                                            <L 1496>
            var_29 = wp::extract(var_24, var_28);
            // x = float(0.0)                                                                     <L 1497>
            var_31 = wp::float(var_30);
            // if length > upper:                                                                 <L 1498>
            var_32 = (var_16 > var_29);
            if (var_32) {
                // x = length - upper                                                             <L 1499>
                var_33 = wp::sub(var_16, var_29);
            }
            if (!var_32) {
                // elif length < lower:                                                           <L 1500>
                var_34 = (var_16 < var_27);
                if (var_34) {
                    // x = length - lower                                                         <L 1501>
                    var_35 = wp::sub(var_16, var_27);
                }
                var_36 = wp::where(var_34, var_35, var_31);
            }
            var_37 = wp::where(var_32, var_33, var_36);
            // if x != 0.0:                                                                       <L 1502>
            var_39 = (var_37 != var_38);
            if (var_39) {
                // stiff = tendon_stiffness[worldid % tendon_stiffness.shape[0], tenid]           <L 1503>
                var_40 = &(var_tendon_stiffness.shape);
                var_43 = wp::load(var_40);
                var_42 = wp::extract(var_43, var_41);
                var_44 = wp::mod(var_0, var_42);
                var_45 = wp::address(var_tendon_stiffness, var_44, var_1);
                var_47 = wp::load(var_45);
                var_46 = wp::copy(var_47);
                // spoly = tendon_stiffnesspoly[worldid % tendon_stiffnesspoly.shape[0], tenid]       <L 1504>
                var_48 = &(var_tendon_stiffnesspoly.shape);
                var_51 = wp::load(var_48);
                var_50 = wp::extract(var_51, var_49);
                var_52 = wp::mod(var_0, var_50);
                var_53 = wp::address(var_tendon_stiffnesspoly, var_52, var_1);
                var_55 = wp::load(var_53);
                var_54 = wp::copy(var_55);
                // k = wp.max(0.0, util_misc._poly_force_deriv(stiff, spoly, x, 0))               <L 1505>
                var_58 = _poly_force_deriv_0(var_46, var_54, var_37, var_57);
                var_59 = wp::max(var_56, var_58);
            }
            var_60 = wp::where(var_39, var_59, var_11);
        }
        var_61 = wp::where(var_14, var_60, var_11);
        // b = float(0.0)                                                                         <L 1507>
        var_63 = wp::float(var_62);
        // if not (opt_disableflags & DisableBit.DAMPER):                                         <L 1508>
        var_65 = wp::bit_and(var_opt_disableflags, var_64);
        var_66 = wp::unot(var_65);
        if (var_66) {
            // damping = tendon_damping[worldid % tendon_damping.shape[0], tenid]                 <L 1509>
            var_67 = &(var_tendon_damping.shape);
            var_70 = wp::load(var_67);
            var_69 = wp::extract(var_70, var_68);
            var_71 = wp::mod(var_0, var_69);
            var_72 = wp::address(var_tendon_damping, var_71, var_1);
            var_74 = wp::load(var_72);
            var_73 = wp::copy(var_74);
            // dpoly = tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]           <L 1510>
            var_75 = &(var_tendon_dampingpoly.shape);
            var_78 = wp::load(var_75);
            var_77 = wp::extract(var_78, var_76);
            var_79 = wp::mod(var_0, var_77);
            var_80 = wp::address(var_tendon_dampingpoly, var_79, var_1);
            var_82 = wp::load(var_80);
            var_81 = wp::copy(var_82);
            // v = ten_velocity_in[worldid, tenid]                                                <L 1511>
            var_83 = wp::address(var_ten_velocity_in, var_0, var_1);
            var_85 = wp::load(var_83);
            var_84 = wp::copy(var_85);
            // b = wp.max(0.0, util_misc._poly_force_deriv(damping, dpoly, v, 1))                 <L 1512>
            var_88 = _poly_force_deriv_0(var_73, var_81, var_84, var_87);
            var_89 = wp::max(var_86, var_88);
        }
        var_90 = wp::where(var_66, var_89, var_63);
        // ts = h * h * k + h * b                                                                 <L 1514>
        var_91 = wp::mul(var_8, var_8);
        var_92 = wp::mul(var_91, var_61);
        var_93 = wp::mul(var_8, var_90);
        var_94 = wp::add(var_92, var_93);
        // tk = h * k                                                                             <L 1515>
        var_95 = wp::mul(var_8, var_61);
        // efm_ts_out[worldid, tenid] = ts                                                        <L 1516>
        wp::array_store(var_efm_ts_out, var_0, var_1, var_94);
        // if k > 0.0:                                                                            <L 1518>
        var_97 = (var_61 > var_96);
        if (var_97) {
            // ckv = tk * ten_velocity_in[worldid, tenid]                                         <L 1519>
            var_98 = wp::address(var_ten_velocity_in, var_0, var_1);
            var_100 = wp::load(var_98);
            var_99 = wp::mul(var_95, var_100);
            // rowadr = ten_J_rowadr[tenid]                                                       <L 1520>
            var_101 = wp::address(var_ten_J_rowadr, var_1);
            var_103 = wp::load(var_101);
            var_102 = wp::copy(var_103);
            // rownnz = ten_J_rownnz[tenid]                                                       <L 1521>
            var_104 = wp::address(var_ten_J_rownnz, var_1);
            var_106 = wp::load(var_104);
            var_105 = wp::copy(var_106);
            // for j in range(rownnz):                                                            <L 1522>
            var_107 = wp::range(var_105);
            start_for_0:;
                if (iter_cmp(var_107) == 0) goto end_for_0;
                var_108 = wp::iter_next(var_107);
                // sparseid = rowadr + j                                                          <L 1523>
                var_109 = wp::add(var_102, var_108);
                // col = ten_J_colind[sparseid]                                                   <L 1524>
                var_110 = wp::address(var_ten_J_colind, var_109);
                var_112 = wp::load(var_110);
                var_111 = wp::copy(var_112);
                // val = ten_J_in[worldid, sparseid]                                              <L 1525>
                var_113 = wp::address(var_ten_J_in, var_0, var_109);
                var_115 = wp::load(var_113);
                var_114 = wp::copy(var_115);
                // wp.atomic_sub(efm_c_out, worldid, col, ckv * val)                              <L 1526>
                var_116 = wp::mul(var_99, var_114);
                var_117 = wp::atomic_sub(var_efm_c_out, var_0, var_111, var_116);
                goto start_for_0;
            end_for_0:;
        }
}



extern "C" __global__ void _eff_flex_stretch_stiff_edge_caf1c505_cuda_kernel_forward(
    wp::launch_bounds_t<3> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_body_weldid,
    wp::array_t<wp::int32> var_body_dofnum,
    wp::array_t<wp::int32> var_body_dofadr,
    wp::array_t<wp::int32> var_flex_dim,
    wp::array_t<wp::int32> var_flex_interp,
    wp::array_t<wp::int32> var_flex_vertadr,
    wp::array_t<wp::int32> var_flex_stiffnessadr,
    wp::array_t<wp::int32> var_flex_vertbodyid,
    wp::array_t<wp::vec_t<2, wp::int32>> var_flex_edge,
    wp::array_t<wp::float32> var_flex_stiffness,
    wp::array_t<wp::float32> var_flex_damping,
    wp::array_t<bool> var_flex_rigid,
    wp::array_t<wp::int32> var_efm_K_rownnz,
    wp::array_t<wp::int32> var_efm_K_rowadr,
    wp::array_t<wp::int32> var_efm_K_colind,
    wp::array_t<wp::int32> var_flex_edgeflexid,
    wp::array_t<bool> var_flex_simple,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_xmat_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_flexedge_hessian_in,
    wp::array_t<wp::float32> var_efm_K_val_out)
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
        wp::int32 var_2;
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        wp::int32 var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        bool var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32* var_17;
        const wp::int32 var_18 = 0;
        bool var_19;
        wp::int32 var_20;
        bool* var_21;
        bool var_22;
        wp::int32* var_23;
        const wp::int32 var_24 = 2;
        bool var_25;
        wp::int32 var_26;
        bool* var_27;
        bool var_28;
        bool var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        bool var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        wp::float32* var_36;
        const wp::float32 var_37 = 0.0;
        bool var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 32;
        wp::int32 var_41;
        const wp::float32 var_42 = 0.0;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::int32 var_45 = 64;
        wp::int32 var_46;
        const wp::float32 var_47 = 0.0;
        wp::float32 var_48;
        wp::float32* var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        const wp::float32 var_53 = 0.0;
        bool var_54;
        wp::int32* var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::vec_t<2, wp::int32>* var_58;
        wp::vec_t<2, wp::int32> var_59;
        wp::vec_t<2, wp::int32> var_60;
        wp::int32 var_61;
        const wp::int32 var_62 = 1;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        wp::int32* var_72;
        wp::int32 var_73;
        wp::int32* var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        bool var_77;
        wp::int32* var_78;
        const wp::int32 var_79 = 3;
        bool var_80;
        wp::int32 var_81;
        wp::int32* var_82;
        const wp::int32 var_83 = 3;
        bool var_84;
        wp::int32 var_85;
        wp::int32* var_86;
        wp::int32 var_87;
        wp::int32 var_88;
        wp::int32* var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        wp::int32* var_92;
        const wp::int32 var_93 = 0;
        bool var_94;
        wp::int32 var_95;
        wp::mat_t<3, 3, wp::float32>* var_96;
        wp::mat_t<3, 3, wp::float32> var_97;
        wp::mat_t<3, 3, wp::float32> var_98;
        const wp::int32 var_99 = 1;
        bool var_100;
        wp::mat_t<3, 3, wp::float32> var_101;
        wp::mat_t<3, 3, wp::float32> var_102;
        wp::mat_t<3, 3, wp::float32>* var_103;
        wp::mat_t<3, 3, wp::float32> var_104;
        wp::mat_t<3, 3, wp::float32> var_105;
        wp::mat_t<3, 3, wp::float32> var_106;
        wp::mat_t<3, 3, wp::float32>* var_107;
        wp::mat_t<3, 3, wp::float32> var_108;
        wp::mat_t<3, 3, wp::float32> var_109;
        wp::mat_t<3, 3, wp::float32> var_110;
        const wp::int32 var_111 = 0;
        wp::int32 var_112;
        wp::int32* var_113;
        wp::int32* var_114;
        wp::int32 var_115;
        wp::int32 var_116;
        wp::int32 var_117;
        const wp::int32 var_118 = 0;
        bool var_119;
        const wp::int32 var_120 = 0;
        wp::int32 var_121;
        wp::float32 var_122;
        wp::float32 var_123;
        const wp::int32 var_124 = 1;
        wp::int32 var_125;
        wp::float32 var_126;
        wp::float32 var_127;
        const wp::int32 var_128 = 2;
        wp::int32 var_129;
        wp::float32 var_130;
        wp::float32 var_131;
        const wp::int32 var_132 = 1;
        wp::int32 var_133;
        wp::int32* var_134;
        wp::int32* var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::int32 var_138;
        const wp::int32 var_139 = 0;
        bool var_140;
        const wp::int32 var_141 = 0;
        wp::int32 var_142;
        wp::float32 var_143;
        wp::float32 var_144;
        const wp::int32 var_145 = 1;
        wp::int32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        const wp::int32 var_149 = 2;
        wp::int32 var_150;
        wp::float32 var_151;
        wp::float32 var_152;
        wp::int32 var_153;
        const wp::int32 var_154 = 2;
        wp::int32 var_155;
        wp::int32* var_156;
        wp::int32* var_157;
        wp::int32 var_158;
        wp::int32 var_159;
        wp::int32 var_160;
        const wp::int32 var_161 = 0;
        bool var_162;
        const wp::int32 var_163 = 0;
        wp::int32 var_164;
        wp::float32 var_165;
        wp::float32 var_166;
        const wp::int32 var_167 = 1;
        wp::int32 var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 2;
        wp::int32 var_172;
        wp::float32 var_173;
        wp::float32 var_174;
        wp::int32 var_175;
        //---------
        // forward
        // def _eff_flex_stretch_stiff_edge(                                                      <L 2061>
        // worldid, edgeid, side = wp.tid()                                                       <L 2088>
        builtin_tid3d(var_0, var_1, var_2);
        // timestep = opt_timestep[worldid % opt_timestep.shape[0]]                               <L 2089>
        var_3 = &(var_opt_timestep.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = wp::mod(var_0, var_5);
        var_8 = wp::address(var_opt_timestep, var_7);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // f = flex_edgeflexid[edgeid]                                                            <L 2091>
        var_11 = wp::address(var_flex_edgeflexid, var_1);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // if f < 0 or flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] < 2 or not flex_simple[f]:       <L 2092>
        var_16 = (var_12 < var_15);
        var_14 = var_16;
        if (!var_14) {
            var_17 = wp::address(var_flex_interp, var_12);
            var_20 = wp::load(var_17);
            var_19 = (var_20 != var_18);
            var_14 = var_14 || var_19;
        }
        if (!var_14) {
            var_21 = wp::address(var_flex_rigid, var_12);
            var_22 = wp::load(var_21);
            var_14 = var_14 || var_22;
        }
        if (!var_14) {
            var_23 = wp::address(var_flex_dim, var_12);
            var_26 = wp::load(var_23);
            var_25 = (var_26 < var_24);
            var_14 = var_14 || var_25;
        }
        if (!var_14) {
            var_27 = wp::address(var_flex_simple, var_12);
            var_29 = wp::load(var_27);
            var_28 = wp::unot(var_29);
            var_14 = var_14 || var_28;
        }
        if (var_14) {
            // return                                                                             <L 2093>
            return;
        }
        // stiffness_adr = flex_stiffnessadr[f]                                                   <L 2095>
        var_30 = wp::address(var_flex_stiffnessadr, var_12);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // if stiffness_adr < 0 or flex_stiffness[stiffness_adr] == 0.0:                          <L 2096>
        var_35 = (var_31 < var_34);
        var_33 = var_35;
        if (!var_33) {
            var_36 = wp::address(var_flex_stiffness, var_31);
            var_39 = wp::load(var_36);
            var_38 = (var_39 == var_37);
            var_33 = var_33 || var_38;
        }
        if (var_33) {
            // return                                                                             <L 2097>
            return;
        }
        // s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep            <L 2099>
        var_41 = wp::bit_and(var_opt_disableflags, var_40);
        if (var_41) {
        }
        if (!var_41) {
            var_43 = wp::mul(var_9, var_9);
        }
        var_44 = wp::where(var_41, var_42, var_43);
        // s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep                       <L 2100>
        var_46 = wp::bit_and(var_opt_disableflags, var_45);
        if (var_46) {
        }
        if (!var_46) {
        }
        var_48 = wp::where(var_46, var_47, var_9);
        // scale = s1 + s2 * flex_damping[f]                                                      <L 2101>
        var_49 = wp::address(var_flex_damping, var_12);
        var_51 = wp::load(var_49);
        var_50 = wp::mul(var_48, var_51);
        var_52 = wp::add(var_44, var_50);
        // if scale == 0.0:                                                                       <L 2102>
        var_54 = (var_52 == var_53);
        if (var_54) {
            // return                                                                             <L 2103>
            return;
        }
        // va = flex_vertadr[f]                                                                   <L 2105>
        var_55 = wp::address(var_flex_vertadr, var_12);
        var_57 = wp::load(var_55);
        var_56 = wp::copy(var_57);
        // edge = flex_edge[edgeid]                                                               <L 2106>
        var_58 = wp::address(var_flex_edge, var_1);
        var_60 = wp::load(var_58);
        var_59 = wp::copy(var_60);
        // vi = edge[side]                                                                        <L 2107>
        var_61 = wp::extract(var_59, var_2);
        // vj = edge[1 - side]                                                                    <L 2108>
        var_63 = wp::sub(var_62, var_2);
        var_64 = wp::extract(var_59, var_63);
        // bi = body_weldid[flex_vertbodyid[va + vi]]                                             <L 2110>
        var_65 = wp::add(var_56, var_61);
        var_66 = wp::address(var_flex_vertbodyid, var_65);
        var_67 = wp::load(var_66);
        var_68 = wp::address(var_body_weldid, var_67);
        var_70 = wp::load(var_68);
        var_69 = wp::copy(var_70);
        // bj = body_weldid[flex_vertbodyid[va + vj]]                                             <L 2111>
        var_71 = wp::add(var_56, var_64);
        var_72 = wp::address(var_flex_vertbodyid, var_71);
        var_73 = wp::load(var_72);
        var_74 = wp::address(var_body_weldid, var_73);
        var_76 = wp::load(var_74);
        var_75 = wp::copy(var_76);
        // if body_dofnum[bi] != 3 or body_dofnum[bj] != 3:                                       <L 2112>
        var_78 = wp::address(var_body_dofnum, var_69);
        var_81 = wp::load(var_78);
        var_80 = (var_81 != var_79);
        var_77 = var_80;
        if (!var_77) {
            var_82 = wp::address(var_body_dofnum, var_75);
            var_85 = wp::load(var_82);
            var_84 = (var_85 != var_83);
            var_77 = var_77 || var_84;
        }
        if (var_77) {
            // return                                                                             <L 2113>
            return;
        }
        // dof_i = body_dofadr[bi]                                                                <L 2115>
        var_86 = wp::address(var_body_dofadr, var_69);
        var_88 = wp::load(var_86);
        var_87 = wp::copy(var_88);
        // dof_j = body_dofadr[bj]                                                                <L 2116>
        var_89 = wp::address(var_body_dofadr, var_75);
        var_91 = wp::load(var_89);
        var_90 = wp::copy(var_91);
        // if efm_K_rownnz[dof_i] == 0:                                                           <L 2117>
        var_92 = wp::address(var_efm_K_rownnz, var_87);
        var_95 = wp::load(var_92);
        var_94 = (var_95 == var_93);
        if (var_94) {
            // return                                                                             <L 2118>
            return;
        }
        // h = flexedge_hessian_in[worldid, edgeid]                                               <L 2120>
        var_96 = wp::address(var_flexedge_hessian_in, var_0, var_1);
        var_98 = wp::load(var_96);
        var_97 = wp::copy(var_98);
        // block = wp.transpose(h) if side == 1 else h                                            <L 2121>
        var_100 = (var_2 == var_99);
        if (var_100) {
            var_101 = wp::transpose(var_97);
        }
        if (!var_100) {
        }
        var_102 = wp::where(var_100, var_101, var_97);
        // blkd = scale * (wp.transpose(xmat_in[worldid, bi]) * block * xmat_in[worldid, bj])       <L 2122>
        var_103 = wp::address(var_xmat_in, var_0, var_69);
        var_105 = wp::load(var_103);
        var_104 = wp::transpose(var_105);
        var_106 = wp::mul(var_104, var_102);
        var_107 = wp::address(var_xmat_in, var_0, var_75);
        var_109 = wp::load(var_107);
        var_108 = wp::mul(var_106, var_109);
        var_110 = wp::mul(var_52, var_108);
        // for r in range(3):                                                                     <L 2124>
        // row = dof_i + r                                                                        <L 2125>
        var_112 = wp::add(var_87, var_111);
        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)         <L 2126>
        var_113 = wp::address(var_efm_K_rowadr, var_112);
        var_114 = wp::address(var_efm_K_rownnz, var_112);
        var_116 = wp::load(var_113);
        var_117 = wp::load(var_114);
        var_115 = _find_csr_col_0(var_116, var_117, var_90, var_efm_K_colind);
        // if idx >= 0:                                                                           <L 2127>
        var_119 = (var_115 >= var_118);
        if (var_119) {
            // for c in range(3):                                                                 <L 2128>
            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])                         <L 2129>
            var_121 = wp::add(var_115, var_120);
            var_122 = wp::extract(var_110, var_111, var_120);
            var_123 = wp::atomic_add(var_efm_K_val_out, var_0, var_121, var_122);
            var_125 = wp::add(var_115, var_124);
            var_126 = wp::extract(var_110, var_111, var_124);
            var_127 = wp::atomic_add(var_efm_K_val_out, var_0, var_125, var_126);
            var_129 = wp::add(var_115, var_128);
            var_130 = wp::extract(var_110, var_111, var_128);
            var_131 = wp::atomic_add(var_efm_K_val_out, var_0, var_129, var_130);
        }
        // row = dof_i + r                                                                        <L 2125>
        var_133 = wp::add(var_87, var_132);
        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)         <L 2126>
        var_134 = wp::address(var_efm_K_rowadr, var_133);
        var_135 = wp::address(var_efm_K_rownnz, var_133);
        var_137 = wp::load(var_134);
        var_138 = wp::load(var_135);
        var_136 = _find_csr_col_0(var_137, var_138, var_90, var_efm_K_colind);
        // if idx >= 0:                                                                           <L 2127>
        var_140 = (var_136 >= var_139);
        if (var_140) {
            // for c in range(3):                                                                 <L 2128>
            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])                         <L 2129>
            var_142 = wp::add(var_136, var_141);
            var_143 = wp::extract(var_110, var_132, var_141);
            var_144 = wp::atomic_add(var_efm_K_val_out, var_0, var_142, var_143);
            var_146 = wp::add(var_136, var_145);
            var_147 = wp::extract(var_110, var_132, var_145);
            var_148 = wp::atomic_add(var_efm_K_val_out, var_0, var_146, var_147);
            var_150 = wp::add(var_136, var_149);
            var_151 = wp::extract(var_110, var_132, var_149);
            var_152 = wp::atomic_add(var_efm_K_val_out, var_0, var_150, var_151);
        }
        var_153 = wp::where(var_140, var_149, var_128);
        // row = dof_i + r                                                                        <L 2125>
        var_155 = wp::add(var_87, var_154);
        // idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)         <L 2126>
        var_156 = wp::address(var_efm_K_rowadr, var_155);
        var_157 = wp::address(var_efm_K_rownnz, var_155);
        var_159 = wp::load(var_156);
        var_160 = wp::load(var_157);
        var_158 = _find_csr_col_0(var_159, var_160, var_90, var_efm_K_colind);
        // if idx >= 0:                                                                           <L 2127>
        var_162 = (var_158 >= var_161);
        if (var_162) {
            // for c in range(3):                                                                 <L 2128>
            // wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])                         <L 2129>
            var_164 = wp::add(var_158, var_163);
            var_165 = wp::extract(var_110, var_154, var_163);
            var_166 = wp::atomic_add(var_efm_K_val_out, var_0, var_164, var_165);
            var_168 = wp::add(var_158, var_167);
            var_169 = wp::extract(var_110, var_154, var_167);
            var_170 = wp::atomic_add(var_efm_K_val_out, var_0, var_168, var_169);
            var_172 = wp::add(var_158, var_171);
            var_173 = wp::extract(var_110, var_154, var_171);
            var_174 = wp::atomic_add(var_efm_K_val_out, var_0, var_172, var_173);
        }
        var_175 = wp::where(var_162, var_171, var_153);
}



extern "C" __global__ void _eff_rhs_de934589_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_opt_enableflags,
    wp::array_t<wp::int32> var_dof_treeid,
    wp::array_t<wp::int32> var_tree_awake_in,
    wp::array_t<wp::float32> var_qfrc_smooth_in,
    wp::array_t<wp::float32> var_efm_c_in,
    wp::array_t<wp::float32> var_efm_ca_in,
    wp::array_t<wp::float32> var_rhs_out)
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
        bool var_2;
        const wp::int32 var_3 = 16;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32 var_10;
        const wp::float32 var_11 = 0.0;
        wp::float32* var_12;
        wp::float32* var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::float32* var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        //---------
        // forward
        // def _eff_rhs(                                                                          <L 1962>
        // worldid, dofid = wp.tid()                                                              <L 1974>
        builtin_tid2d(var_0, var_1);
        // if (opt_enableflags & EnableBit.SLEEP) and tree_awake_in[worldid, dof_treeid[dofid]] == 0:       <L 1975>
        var_4 = wp::bit_and(var_opt_enableflags, var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::address(var_dof_treeid, var_1);
            var_6 = wp::load(var_5);
            var_7 = wp::address(var_tree_awake_in, var_0, var_6);
            var_10 = wp::load(var_7);
            var_9 = (var_10 == var_8);
            var_2 = var_2 && var_9;
        }
        if (var_2) {
            // rhs_out[worldid, dofid] = 0.0                                                      <L 1976>
            wp::array_store(var_rhs_out, var_0, var_1, var_11);
            // return                                                                             <L 1977>
            return;
        }
        // rhs_out[worldid, dofid] = qfrc_smooth_in[worldid, dofid] + efm_c_in[worldid, dofid] + efm_ca_in[worldid, dofid]       <L 1978>
        var_12 = wp::address(var_qfrc_smooth_in, var_0, var_1);
        var_13 = wp::address(var_efm_c_in, var_0, var_1);
        var_15 = wp::load(var_12);
        var_16 = wp::load(var_13);
        var_14 = wp::add(var_15, var_16);
        var_17 = wp::address(var_efm_ca_in, var_0, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::add(var_14, var_19);
        wp::array_store(var_rhs_out, var_0, var_1, var_18);
}



extern "C" __global__ void _qderiv_tendon_damping_ec990995_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::int32 var_ntendon,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::array_t<wp::int32> var_ten_J_rownnz,
    wp::array_t<wp::int32> var_ten_J_rowadr,
    wp::array_t<wp::int32> var_ten_J_colind,
    wp::array_t<wp::float32> var_tendon_damping,
    wp::array_t<wp::vec_t<2, wp::float32>> var_tendon_dampingpoly,
    wp::array_t<wp::int32> var_M_elemid,
    wp::array_t<wp::float32> var_ten_J_in,
    wp::array_t<wp::float32> var_ten_velocity_in,
    wp::array_t<wp::int32> var_Mi,
    wp::array_t<wp::int32> var_Mj,
    wp::array_t<wp::float32> var_qDeriv_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        const wp::float32 var_13 = 0.0;
        wp::float32 var_14;
        wp::shape_t* var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::shape_t var_18;
        wp::int32 var_19;
        wp::range_t var_20;
        wp::int32 var_21;
        wp::float32* var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        wp::shape_t* var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::shape_t var_28;
        wp::int32 var_29;
        wp::vec_t<2, wp::float32>* var_30;
        wp::vec_t<2, wp::float32> var_31;
        wp::vec_t<2, wp::float32> var_32;
        bool var_33;
        const wp::float32 var_34 = 0.0;
        bool var_35;
        const wp::int32 var_36 = 0;
        wp::float32 var_37;
        const wp::float32 var_38 = 0.0;
        bool var_39;
        const wp::int32 var_40 = 1;
        wp::float32 var_41;
        const wp::float32 var_42 = 0.0;
        bool var_43;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32* var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        const wp::float32 var_50 = 0.0;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.0;
        wp::float32 var_53;
        wp::range_t var_54;
        wp::int32 var_55;
        bool var_56;
        const wp::float32 var_57 = 0.0;
        bool var_58;
        const wp::float32 var_59 = 0.0;
        bool var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        bool var_65;
        wp::float32* var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        bool var_70;
        wp::float32* var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32* var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        const wp::int32 var_79 = 1;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::shape_t* var_83;
        const wp::int32 var_84 = 0;
        wp::int32 var_85;
        wp::shape_t var_86;
        wp::int32 var_87;
        wp::float32* var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        wp::float32 var_91;
        //---------
        // forward
        // def _qderiv_tendon_damping(                                                            <L 286>
        // worldid, elemid = wp.tid()                                                             <L 305>
        builtin_tid2d(var_0, var_1);
        // dofiid = Mi[elemid]                                                                    <L 306>
        var_2 = wp::address(var_Mi, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // dofjid = Mj[elemid]                                                                    <L 307>
        var_5 = wp::address(var_Mj, var_1);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // madr = M_elemid[dofiid, dofjid]                                                        <L 310>
        var_8 = wp::address(var_M_elemid, var_3, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if madr < 0:                                                                           <L 311>
        var_12 = (var_9 < var_11);
        if (var_12) {
            // return                                                                             <L 312>
            return;
        }
        // qderiv = float(0.0)                                                                    <L 314>
        var_14 = wp::float(var_13);
        // tendon_damping_id = worldid % tendon_damping.shape[0]                                  <L 315>
        var_15 = &(var_tendon_damping.shape);
        var_18 = wp::load(var_15);
        var_17 = wp::extract(var_18, var_16);
        var_19 = wp::mod(var_0, var_17);
        // for tenid in range(ntendon):                                                           <L 316>
        var_20 = wp::range(var_ntendon);
        start_for_1:;
            if (iter_cmp(var_20) == 0) goto end_for_1;
            var_21 = wp::iter_next(var_20);
            // damping = tendon_damping[tendon_damping_id, tenid]                                 <L 317>
            var_22 = wp::address(var_tendon_damping, var_19, var_21);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // dpoly = tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]           <L 318>
            var_25 = &(var_tendon_dampingpoly.shape);
            var_28 = wp::load(var_25);
            var_27 = wp::extract(var_28, var_26);
            var_29 = wp::mod(var_0, var_27);
            var_30 = wp::address(var_tendon_dampingpoly, var_29, var_21);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            // if damping == 0.0 and dpoly[0] == 0.0 and dpoly[1] == 0.0:                         <L 319>
            var_35 = (var_23 == var_34);
            var_33 = var_35;
            if (var_33) {
                var_37 = wp::extract(var_31, var_36);
                var_39 = (var_37 == var_38);
                var_33 = var_33 && var_39;
            }
            if (var_33) {
                var_41 = wp::extract(var_31, var_40);
                var_43 = (var_41 == var_42);
                var_33 = var_33 && var_43;
            }
            if (var_33) {
                // continue                                                                       <L 320>
                goto start_for_1;
            }
            // rownnz = ten_J_rownnz[tenid]                                                       <L 322>
            var_44 = wp::address(var_ten_J_rownnz, var_21);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // rowadr = ten_J_rowadr[tenid]                                                       <L 323>
            var_47 = wp::address(var_ten_J_rowadr, var_21);
            var_49 = wp::load(var_47);
            var_48 = wp::copy(var_49);
            // Ji = float(0.0)                                                                    <L 324>
            var_51 = wp::float(var_50);
            // Jj = float(0.0)                                                                    <L 325>
            var_53 = wp::float(var_52);
            // for k in range(rownnz):                                                            <L 326>
            var_54 = wp::range(var_45);
            start_for_3:;
                if (iter_cmp(var_54) == 0) goto end_for_3;
                var_55 = wp::iter_next(var_54);
                // if Ji != 0.0 and Jj != 0.0:                                                    <L 327>
                var_58 = (var_51 != var_57);
                var_56 = var_58;
                if (var_56) {
                    var_60 = (var_53 != var_59);
                    var_56 = var_56 && var_60;
                }
                if (var_56) {
                    // break                                                                      <L 328>
                    goto end_for_3;
                }
                // sparseid = rowadr + k                                                          <L 329>
                var_61 = wp::add(var_48, var_55);
                // colind = ten_J_colind[sparseid]                                                <L 330>
                var_62 = wp::address(var_ten_J_colind, var_61);
                var_64 = wp::load(var_62);
                var_63 = wp::copy(var_64);
                // if colind == dofiid:                                                           <L 331>
                var_65 = (var_63 == var_3);
                if (var_65) {
                    // Ji = ten_J_in[worldid, sparseid]                                           <L 332>
                    var_66 = wp::address(var_ten_J_in, var_0, var_61);
                    var_68 = wp::load(var_66);
                    var_67 = wp::copy(var_68);
                }
                var_69 = wp::where(var_65, var_67, var_51);
                // if colind == dofjid:                                                           <L 333>
                var_70 = (var_63 == var_6);
                if (var_70) {
                    // Jj = ten_J_in[worldid, sparseid]                                           <L 334>
                    var_71 = wp::address(var_ten_J_in, var_0, var_61);
                    var_73 = wp::load(var_71);
                    var_72 = wp::copy(var_73);
                }
                var_74 = wp::where(var_70, var_72, var_53);
                wp::assign(var_51, var_69);
                wp::assign(var_53, var_74);
                goto start_for_3;
            end_for_3:;
            // v = ten_velocity_in[worldid, tenid]                                                <L 336>
            var_75 = wp::address(var_ten_velocity_in, var_0, var_21);
            var_77 = wp::load(var_75);
            var_76 = wp::copy(var_77);
            // qderiv -= Ji * Jj * util_misc._poly_force_deriv(damping, dpoly, v, 1)              <L 337>
            var_78 = wp::mul(var_51, var_53);
            var_80 = _poly_force_deriv_0(var_23, var_31, var_76, var_79);
            var_81 = wp::mul(var_78, var_80);
            var_82 = wp::sub(var_14, var_81);
            wp::assign(var_14, var_82);
            goto start_for_1;
        end_for_1:;
        // qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]                                <L 339>
        var_83 = &(var_opt_timestep.shape);
        var_86 = wp::load(var_83);
        var_85 = wp::extract(var_86, var_84);
        var_87 = wp::mod(var_0, var_85);
        var_88 = wp::address(var_opt_timestep, var_87);
        var_90 = wp::load(var_88);
        var_89 = wp::mul(var_14, var_90);
        // qDeriv_out[worldid, madr] -= qderiv                                                    <L 341>
        var_91 = wp::atomic_sub(var_qDeriv_out, var_0, var_9, var_89);
}



extern "C" __global__ void _zero_sleeping_dofs_ecb9f7af_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_dof_treeid,
    wp::array_t<wp::int32> var_tree_awake_in,
    wp::array_t<wp::float32> var_vec_out)
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
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32* var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::int32 var_7;
        const wp::float32 var_8 = 0.0;
        //---------
        // forward
        // def _zero_sleeping_dofs(                                                               <L 2530>
        // worldid, dofid = wp.tid()                                                              <L 2538>
        builtin_tid2d(var_0, var_1);
        // if tree_awake_in[worldid, dof_treeid[dofid]] == 0:                                     <L 2539>
        var_2 = wp::address(var_dof_treeid, var_1);
        var_3 = wp::load(var_2);
        var_4 = wp::address(var_tree_awake_in, var_0, var_3);
        var_7 = wp::load(var_4);
        var_6 = (var_7 == var_5);
        if (var_6) {
            // vec_out[worldid, dofid] = 0.0                                                      <L 2540>
            wp::array_store(var_vec_out, var_0, var_1, var_8);
        }
}



extern "C" __global__ void _eff_actuator_actuation_f6d96889_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::float32> var_opt_timestep,
    wp::int32 var_opt_disableflags,
    wp::array_t<wp::int32> var_actuator_dyntype,
    wp::array_t<wp::int32> var_actuator_gaintype,
    wp::array_t<wp::int32> var_actuator_biastype,
    wp::array_t<wp::int32> var_actuator_ctrladr,
    wp::array_t<wp::int32> var_actuator_ctrlspec,
    wp::array_t<wp::int32> var_actuator_actadr,
    wp::array_t<wp::int32> var_actuator_actnum,
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
    wp::array_t<wp::float32> var_actuator_acc0,
    wp::array_t<wp::vec_t<2, wp::float32>> var_actuator_lengthrange,
    wp::array_t<wp::float32> var_act_in,
    wp::array_t<wp::float32> var_ctrl_in,
    wp::array_t<wp::float32> var_act_dot_in,
    wp::array_t<wp::float32> var_actuator_length_in,
    wp::array_t<wp::int32> var_moment_rownnz_in,
    wp::array_t<wp::int32> var_moment_rowadr_in,
    wp::array_t<wp::int32> var_moment_colind_in,
    wp::array_t<wp::float32> var_actuator_moment_in,
    wp::array_t<wp::float32> var_actuator_velocity_in,
    wp::array_t<wp::float32> var_actuator_force_in,
    wp::array_t<wp::float32> var_efm_ca_out,
    wp::array_t<wp::float32> var_efm_as_out)
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
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::int32 var_10 = 2048;
        wp::int32 var_11;
        bool* var_12;
        bool var_13;
        wp::float32* var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        wp::shape_t* var_17;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        wp::shape_t var_20;
        wp::int32 var_21;
        wp::vec_t<2, wp::float32>* var_22;
        wp::vec_t<2, wp::float32> var_23;
        wp::vec_t<2, wp::float32> var_24;
        bool var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        bool var_28;
        const wp::int32 var_29 = 1;
        wp::float32 var_30;
        bool var_31;
        const wp::float32 var_32 = 0.0;
        bool var_33;
        wp::shape_t* var_34;
        const wp::int32 var_35 = 0;
        wp::int32 var_36;
        wp::shape_t var_37;
        wp::int32 var_38;
        wp::vec_t<10, wp::float32>* var_39;
        wp::vec_t<10, wp::float32> var_40;
        wp::vec_t<10, wp::float32> var_41;
        wp::shape_t* var_42;
        const wp::int32 var_43 = 0;
        wp::int32 var_44;
        wp::shape_t var_45;
        wp::int32 var_46;
        wp::vec_t<10, wp::float32>* var_47;
        wp::vec_t<10, wp::float32> var_48;
        wp::vec_t<10, wp::float32> var_49;
        wp::shape_t* var_50;
        const wp::int32 var_51 = 0;
        wp::int32 var_52;
        wp::shape_t var_53;
        wp::int32 var_54;
        wp::vec_t<10, wp::float32>* var_55;
        wp::vec_t<10, wp::float32> var_56;
        wp::vec_t<10, wp::float32> var_57;
        const wp::float32 var_58 = 0.0;
        wp::float32 var_59;
        const wp::float32 var_60 = 0.0;
        wp::float32 var_61;
        const wp::float32 var_62 = 0.0;
        wp::float32 var_63;
        const wp::float32 var_64 = 0.0;
        wp::float32 var_65;
        wp::int32* var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        const wp::int32 var_69 = 1;
        bool var_70;
        const wp::int32 var_71 = 1;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::int32 var_74 = 2;
        wp::float32 var_75;
        wp::float32 var_76;
        const wp::int32 var_77 = 2;
        bool var_78;
        wp::float32* var_79;
        wp::shape_t* var_80;
        const wp::int32 var_81 = 0;
        wp::int32 var_82;
        wp::shape_t var_83;
        wp::int32 var_84;
        wp::vec_t<2, wp::float32>* var_85;
        wp::shape_t* var_86;
        const wp::int32 var_87 = 0;
        wp::int32 var_88;
        wp::shape_t var_89;
        wp::int32 var_90;
        wp::float32* var_91;
        wp::float32 var_92;
        wp::float32 var_93;
        wp::vec_t<2, wp::float32> var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        const wp::int32 var_97 = 3;
        bool var_98;
        const wp::int32 var_99 = 0;
        wp::float32 var_100;
        const wp::float32 var_101 = 0.0;
        bool var_102;
        const wp::int32 var_103 = 0;
        wp::float32 var_104;
        const wp::int32 var_105 = 1;
        wp::float32 var_106;
        wp::vec_t<6, wp::int32> var_107;
        const wp::int32 var_108 = 2;
        wp::int32 var_109;
        const wp::int32 var_110 = 0;
        bool var_111;
        wp::int32* var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::float32* var_115;
        wp::float32 var_116;
        wp::float32 var_117;
        const wp::float32 var_118 = 1.0;
        const wp::int32 var_119 = 2;
        wp::float32 var_120;
        const wp::int32 var_121 = 4;
        wp::float32 var_122;
        wp::float32 var_123;
        const wp::int32 var_124 = 3;
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
        wp::float32 var_139;
        wp::int32* var_140;
        wp::int32 var_141;
        wp::int32 var_142;
        const wp::int32 var_143 = 1;
        bool var_144;
        const wp::int32 var_145 = 1;
        wp::float32 var_146;
        const wp::int32 var_147 = 2;
        wp::float32 var_148;
        const wp::int32 var_149 = 2;
        bool var_150;
        wp::shape_t* var_151;
        const wp::int32 var_152 = 0;
        wp::int32 var_153;
        wp::shape_t var_154;
        wp::int32 var_155;
        wp::vec_t<2, wp::float32>* var_156;
        wp::vec_t<2, wp::float32> var_157;
        wp::vec_t<2, wp::float32> var_158;
        wp::shape_t* var_159;
        const wp::int32 var_160 = 0;
        wp::int32 var_161;
        wp::shape_t var_162;
        wp::int32 var_163;
        wp::float32* var_164;
        wp::float32 var_165;
        wp::float32 var_166;
        wp::float32* var_167;
        wp::float32* var_168;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::float32* var_172;
        wp::float32* var_173;
        wp::float32 var_174;
        wp::float32 var_175;
        wp::float32 var_176;
        const wp::int32 var_177 = 3;
        bool var_178;
        const wp::int32 var_179 = 0;
        wp::float32 var_180;
        const wp::int32 var_181 = 0;
        wp::float32 var_182;
        const wp::int32 var_183 = 1;
        wp::float32 var_184;
        const wp::int32 var_185 = 0;
        wp::float32 var_186;
        wp::vec_t<6, wp::int32> var_187;
        const wp::int32 var_188 = 2;
        wp::int32 var_189;
        const wp::int32 var_190 = 0;
        bool var_191;
        wp::int32* var_192;
        wp::int32 var_193;
        wp::int32 var_194;
        wp::float32* var_195;
        wp::float32 var_196;
        wp::float32 var_197;
        const wp::float32 var_198 = 1.0;
        const wp::int32 var_199 = 2;
        wp::float32 var_200;
        const wp::int32 var_201 = 4;
        wp::float32 var_202;
        wp::float32 var_203;
        const wp::int32 var_204 = 3;
        wp::float32 var_205;
        wp::float32 var_206;
        wp::float32 var_207;
        wp::float32 var_208;
        wp::float32 var_209;
        wp::float32 var_210;
        wp::float32 var_211;
        const wp::float32 var_212 = 0.0;
        wp::float32 var_213;
        const wp::float32 var_214 = 0.0;
        wp::float32 var_215;
        wp::int32* var_216;
        const wp::int32 var_217 = 7;
        wp::int32 var_218;
        wp::int32 var_219;
        const wp::int32 var_220 = 0;
        bool var_221;
        const wp::int32 var_222 = 4;
        wp::float32 var_223;
        wp::float32 var_224;
        wp::float32 var_225;
        wp::float32 var_226;
        const wp::int32 var_227 = 6;
        wp::float32 var_228;
        wp::float32 var_229;
        wp::float32 var_230;
        wp::float32 var_231;
        wp::float32 var_232;
        wp::float32 var_233;
        wp::float32 var_234;
        const wp::float32 var_235 = 0.0;
        bool var_236;
        const wp::float32 var_237 = 1.0;
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
        const wp::float32 var_251 = 0.0;
        bool var_252;
        wp::float32 var_253;
        wp::float32 var_254;
        wp::float32 var_255;
        wp::float32 var_256;
        const wp::float32 var_257 = 0.0;
        bool var_258;
        wp::float32 var_259;
        wp::float32 var_260;
        wp::float32 var_261;
        wp::float32 var_262;
        wp::float32 var_263;
        wp::float32 var_264;
        const wp::int32 var_265 = 6;
        wp::float32 var_266;
        const wp::float32 var_267 = 0.0;
        bool var_268;
        wp::float32 var_269;
        wp::float32 var_270;
        wp::float32 var_271;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::float32 var_274;
        wp::vec_t<6, wp::int32> var_275;
        wp::int32 var_276;
        wp::float32 var_277;
        wp::float32 var_278;
        wp::float32 var_279;
        wp::float32 var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        wp::float32 var_283;
        wp::vec_t<6, wp::int32> var_284;
        wp::int32 var_285;
        wp::float32 var_286;
        wp::float32 var_287;
        wp::float32 var_288;
        wp::float32 var_289;
        wp::float32 var_290;
        wp::float32 var_291;
        wp::float32 var_292;
        wp::vec_t<6, wp::int32> var_293;
        wp::int32 var_294;
        wp::float32 var_295;
        bool var_296;
        const wp::float32 var_297 = 0.0;
        bool var_298;
        const wp::float32 var_299 = 0.0;
        bool var_300;
        const wp::float32 var_301 = 0.0;
        wp::float32 var_302;
        wp::int32* var_303;
        const wp::int32 var_304 = 0;
        bool var_305;
        wp::int32 var_306;
        wp::int32* var_307;
        wp::int32* var_308;
        wp::int32 var_309;
        wp::int32 var_310;
        wp::int32 var_311;
        const wp::int32 var_312 = 1;
        wp::int32 var_313;
        bool* var_314;
        bool var_315;
        wp::int32* var_316;
        wp::shape_t* var_317;
        const wp::int32 var_318 = 0;
        wp::int32 var_319;
        wp::shape_t var_320;
        wp::int32 var_321;
        wp::vec_t<2, wp::float32>* var_322;
        wp::float32* var_323;
        wp::float32* var_324;
        const wp::float32 var_325 = 1.0;
        bool* var_326;
        wp::float32 var_327;
        wp::int32 var_328;
        wp::vec_t<2, wp::float32> var_329;
        wp::float32 var_330;
        wp::float32 var_331;
        bool var_332;
        bool var_333;
        bool var_334;
        wp::float32* var_335;
        wp::float32 var_336;
        wp::float32 var_337;
        bool var_338;
        wp::float32 var_339;
        bool var_340;
        wp::int32* var_341;
        wp::int32 var_342;
        wp::int32 var_343;
        const wp::int32 var_344 = 0;
        bool var_345;
        wp::float32* var_346;
        wp::float32 var_347;
        wp::float32 var_348;
        bool var_349;
        const wp::int32 var_350 = 256;
        wp::int32 var_351;
        bool var_352;
        bool* var_353;
        bool var_354;
        wp::shape_t* var_355;
        const wp::int32 var_356 = 0;
        wp::int32 var_357;
        wp::shape_t var_358;
        wp::int32 var_359;
        wp::vec_t<2, wp::float32>* var_360;
        wp::vec_t<2, wp::float32> var_361;
        wp::vec_t<2, wp::float32> var_362;
        const wp::int32 var_363 = 0;
        wp::float32 var_364;
        const wp::int32 var_365 = 1;
        wp::float32 var_366;
        wp::float32 var_367;
        wp::float32 var_368;
        wp::float32 var_369;
        wp::float32 var_370;
        wp::float32 var_371;
        wp::float32 var_372;
        wp::float32 var_373;
        wp::float32 var_374;
        wp::float32 var_375;
        wp::float32 var_376;
        const wp::float32 var_377 = 0.0;
        wp::float32 var_378;
        wp::float32 var_379;
        const wp::float32 var_380 = 0.0;
        wp::float32 var_381;
        wp::float32 var_382;
        wp::float32 var_383;
        wp::float32 var_384;
        wp::float32 var_385;
        wp::float32 var_386;
        wp::float32 var_387;
        const wp::float32 var_388 = 0.0;
        bool var_389;
        wp::float32* var_390;
        wp::float32 var_391;
        wp::float32 var_392;
        wp::int32* var_393;
        wp::int32 var_394;
        wp::int32 var_395;
        wp::int32* var_396;
        wp::int32 var_397;
        wp::int32 var_398;
        wp::range_t var_399;
        wp::int32 var_400;
        wp::int32 var_401;
        wp::int32* var_402;
        wp::int32 var_403;
        wp::int32 var_404;
        wp::float32* var_405;
        wp::float32 var_406;
        wp::float32 var_407;
        wp::float32 var_408;
        wp::float32 var_409;
        //---------
        // forward
        // def _eff_actuator_actuation(                                                           <L 1530>
        // worldid, actid = wp.tid()                                                              <L 1568>
        builtin_tid2d(var_0, var_1);
        // h = opt_timestep[worldid % opt_timestep.shape[0]]                                      <L 1569>
        var_2 = &(var_opt_timestep.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_0, var_4);
        var_7 = wp::address(var_opt_timestep, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if opt_disableflags & DisableBit.ACTUATION:                                            <L 1571>
        var_11 = wp::bit_and(var_opt_disableflags, var_10);
        if (var_11) {
            // return                                                                             <L 1572>
            return;
        }
        // if actuator_forcelimited[actid]:                                                       <L 1574>
        var_12 = wp::address(var_actuator_forcelimited, var_1);
        var_13 = wp::load(var_12);
        if (var_13) {
            // force = actuator_force_in[worldid, actid]                                          <L 1575>
            var_14 = wp::address(var_actuator_force_in, var_0, var_1);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]       <L 1576>
            var_17 = &(var_actuator_forcerange.shape);
            var_20 = wp::load(var_17);
            var_19 = wp::extract(var_20, var_18);
            var_21 = wp::mod(var_0, var_19);
            var_22 = wp::address(var_actuator_forcerange, var_21, var_1);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // if force <= forcerange[0] or force >= forcerange[1]:                               <L 1577>
            var_27 = wp::extract(var_23, var_26);
            var_28 = (var_15 <= var_27);
            var_25 = var_28;
            if (!var_25) {
                var_30 = wp::extract(var_23, var_29);
                var_31 = (var_15 >= var_30);
                var_25 = var_25 || var_31;
            }
            if (var_25) {
                // efm_as_out[worldid, actid] = 0.0                                               <L 1578>
                wp::array_store(var_efm_as_out, var_0, var_1, var_32);
                // return                                                                         <L 1579>
                return;
            }
        }
        var_33 = wp::load(var_12);
        // gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], actid]                 <L 1581>
        var_34 = &(var_actuator_gainprm.shape);
        var_37 = wp::load(var_34);
        var_36 = wp::extract(var_37, var_35);
        var_38 = wp::mod(var_0, var_36);
        var_39 = wp::address(var_actuator_gainprm, var_38, var_1);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], actid]                 <L 1582>
        var_42 = &(var_actuator_biasprm.shape);
        var_45 = wp::load(var_42);
        var_44 = wp::extract(var_45, var_43);
        var_46 = wp::mod(var_0, var_44);
        var_47 = wp::address(var_actuator_biasprm, var_46, var_1);
        var_49 = wp::load(var_47);
        var_48 = wp::copy(var_49);
        // dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], actid]                    <L 1583>
        var_50 = &(var_actuator_dynprm.shape);
        var_53 = wp::load(var_50);
        var_52 = wp::extract(var_53, var_51);
        var_54 = wp::mod(var_0, var_52);
        var_55 = wp::address(var_actuator_dynprm, var_54, var_1);
        var_57 = wp::load(var_55);
        var_56 = wp::copy(var_57);
        // bias_len = float(0.0)                                                                  <L 1585>
        var_59 = wp::float(var_58);
        // gain_len = float(0.0)                                                                  <L 1586>
        var_61 = wp::float(var_60);
        // bias_vel = float(0.0)                                                                  <L 1587>
        var_63 = wp::float(var_62);
        // gain_vel = float(0.0)                                                                  <L 1588>
        var_65 = wp::float(var_64);
        // biastype = actuator_biastype[actid]                                                    <L 1590>
        var_66 = wp::address(var_actuator_biastype, var_1);
        var_68 = wp::load(var_66);
        var_67 = wp::copy(var_68);
        // if biastype == BiasType.AFFINE:                                                        <L 1591>
        var_70 = (var_67 == var_69);
        if (var_70) {
            // bias_len += biasprm[1]                                                             <L 1592>
            var_72 = wp::extract(var_48, var_71);
            var_73 = wp::add(var_59, var_72);
            // bias_vel += biasprm[2]                                                             <L 1593>
            var_75 = wp::extract(var_48, var_74);
            var_76 = wp::add(var_63, var_75);
        }
        if (!var_70) {
            // elif biastype == BiasType.MUSCLE:                                                  <L 1594>
            var_78 = (var_67 == var_77);
            if (var_78) {
                // bias_len += util_misc.muscle_bias_len_deriv(                                   <L 1595>
                // actuator_length_in[worldid, actid],                                            <L 1596>
                var_79 = wp::address(var_actuator_length_in, var_0, var_1);
                // actuator_lengthrange[worldid % actuator_lengthrange.shape[0], actid],          <L 1597>
                var_80 = &(var_actuator_lengthrange.shape);
                var_83 = wp::load(var_80);
                var_82 = wp::extract(var_83, var_81);
                var_84 = wp::mod(var_0, var_82);
                var_85 = wp::address(var_actuator_lengthrange, var_84, var_1);
                // actuator_acc0[worldid % actuator_acc0.shape[0], actid],                        <L 1598>
                var_86 = &(var_actuator_acc0.shape);
                var_89 = wp::load(var_86);
                var_88 = wp::extract(var_89, var_87);
                var_90 = wp::mod(var_0, var_88);
                var_91 = wp::address(var_actuator_acc0, var_90, var_1);
                // biasprm,                                                                       <L 1599>
                var_93 = wp::load(var_79);
                var_94 = wp::load(var_85);
                var_95 = wp::load(var_91);
                var_92 = muscle_bias_len_deriv_0(var_93, var_94, var_95, var_48);
                // bias_len += util_misc.muscle_bias_len_deriv(                                   <L 1595>
                var_96 = wp::add(var_59, var_92);
            }
            if (!var_78) {
                // elif biastype == BiasType.DCMOTOR:                                             <L 1601>
                var_98 = (var_67 == var_97);
                if (var_98) {
                    // if dynprm[0] <= 0.0:                                                       <L 1602>
                    var_100 = wp::extract(var_56, var_99);
                    var_102 = (var_100 <= var_101);
                    if (var_102) {
                        // R = gainprm[0]                                                         <L 1603>
                        var_104 = wp::extract(var_40, var_103);
                        // K = gainprm[1]                                                         <L 1604>
                        var_106 = wp::extract(var_40, var_105);
                        // slots = util_misc.dcmotor_slots(dynprm, gainprm)                       <L 1605>
                        var_107 = dcmotor_slots_0(var_56, var_40);
                        // slot_Ta = slots[2]                                                     <L 1606>
                        var_109 = wp::extract(var_107, var_108);
                        // if slot_Ta >= 0:                                                       <L 1607>
                        var_111 = (var_109 >= var_110);
                        if (var_111) {
                            // T = act_in[worldid, actuator_actadr[actid] + slot_Ta]              <L 1608>
                            var_112 = wp::address(var_actuator_actadr, var_1);
                            var_114 = wp::load(var_112);
                            var_113 = wp::add(var_114, var_109);
                            var_115 = wp::address(var_act_in, var_0, var_113);
                            var_117 = wp::load(var_115);
                            var_116 = wp::copy(var_117);
                            // R *= 1.0 + gainprm[2] * (T + dynprm[4] - gainprm[3])               <L 1609>
                            var_120 = wp::extract(var_40, var_119);
                            var_122 = wp::extract(var_56, var_121);
                            var_123 = wp::add(var_116, var_122);
                            var_125 = wp::extract(var_40, var_124);
                            var_126 = wp::sub(var_123, var_125);
                            var_127 = wp::mul(var_120, var_126);
                            var_128 = wp::add(var_118, var_127);
                            var_129 = wp::mul(var_104, var_128);
                        }
                        var_130 = wp::where(var_111, var_129, var_104);
                        // bias_vel -= math.safe_div(K * K, R)                                    <L 1610>
                        var_131 = wp::mul(var_106, var_106);
                        var_132 = safe_div_0(var_131, var_130);
                        var_133 = wp::sub(var_63, var_132);
                    }
                    var_134 = wp::where(var_102, var_133, var_63);
                }
                var_135 = wp::where(var_98, var_134, var_63);
            }
            var_136 = wp::where(var_78, var_96, var_59);
            var_137 = wp::where(var_78, var_63, var_135);
        }
        var_138 = wp::where(var_70, var_73, var_136);
        var_139 = wp::where(var_70, var_76, var_137);
        // gaintype = actuator_gaintype[actid]                                                    <L 1612>
        var_140 = wp::address(var_actuator_gaintype, var_1);
        var_142 = wp::load(var_140);
        var_141 = wp::copy(var_142);
        // if gaintype == GainType.AFFINE:                                                        <L 1613>
        var_144 = (var_141 == var_143);
        if (var_144) {
            // gain_len = gainprm[1]                                                              <L 1614>
            var_146 = wp::extract(var_40, var_145);
            // gain_vel = gainprm[2]                                                              <L 1615>
            var_148 = wp::extract(var_40, var_147);
        }
        if (!var_144) {
            // elif gaintype == GainType.MUSCLE:                                                  <L 1616>
            var_150 = (var_141 == var_149);
            if (var_150) {
                // lenrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], actid]       <L 1617>
                var_151 = &(var_actuator_lengthrange.shape);
                var_154 = wp::load(var_151);
                var_153 = wp::extract(var_154, var_152);
                var_155 = wp::mod(var_0, var_153);
                var_156 = wp::address(var_actuator_lengthrange, var_155, var_1);
                var_158 = wp::load(var_156);
                var_157 = wp::copy(var_158);
                // acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], actid]                  <L 1618>
                var_159 = &(var_actuator_acc0.shape);
                var_162 = wp::load(var_159);
                var_161 = wp::extract(var_162, var_160);
                var_163 = wp::mod(var_0, var_161);
                var_164 = wp::address(var_actuator_acc0, var_163, var_1);
                var_166 = wp::load(var_164);
                var_165 = wp::copy(var_166);
                // gain_len = util_misc.muscle_gain_len_deriv(                                    <L 1619>
                // actuator_length_in[worldid, actid],                                            <L 1620>
                var_167 = wp::address(var_actuator_length_in, var_0, var_1);
                // actuator_velocity_in[worldid, actid],                                          <L 1621>
                var_168 = wp::address(var_actuator_velocity_in, var_0, var_1);
                // lenrange,                                                                      <L 1622>
                // acc0,                                                                          <L 1623>
                // gainprm,                                                                       <L 1624>
                var_170 = wp::load(var_167);
                var_171 = wp::load(var_168);
                var_169 = muscle_gain_len_deriv_0(var_170, var_171, var_157, var_165, var_40);
                // gain_vel = util_misc.muscle_gain_vel_deriv(                                    <L 1626>
                // actuator_length_in[worldid, actid],                                            <L 1627>
                var_172 = wp::address(var_actuator_length_in, var_0, var_1);
                // actuator_velocity_in[worldid, actid],                                          <L 1628>
                var_173 = wp::address(var_actuator_velocity_in, var_0, var_1);
                // lenrange,                                                                      <L 1629>
                // acc0,                                                                          <L 1630>
                // gainprm,                                                                       <L 1631>
                var_175 = wp::load(var_172);
                var_176 = wp::load(var_173);
                var_174 = muscle_gain_vel_deriv_0(var_175, var_176, var_157, var_165, var_40);
            }
            if (!var_150) {
                // elif gaintype == GainType.DCMOTOR:                                             <L 1633>
                var_178 = (var_141 == var_177);
                if (var_178) {
                    // te = dynprm[0]                                                             <L 1634>
                    var_180 = wp::extract(var_56, var_179);
                    // R0 = gainprm[0]                                                            <L 1635>
                    var_182 = wp::extract(var_40, var_181);
                    // K = gainprm[1]                                                             <L 1636>
                    var_184 = wp::extract(var_40, var_183);
                    // R = gainprm[0]                                                             <L 1637>
                    var_186 = wp::extract(var_40, var_185);
                    // slots = util_misc.dcmotor_slots(dynprm, gainprm)                           <L 1638>
                    var_187 = dcmotor_slots_0(var_56, var_40);
                    // slot_Ta = slots[2]                                                         <L 1639>
                    var_189 = wp::extract(var_187, var_188);
                    // if slot_Ta >= 0:                                                           <L 1640>
                    var_191 = (var_189 >= var_190);
                    if (var_191) {
                        // T = act_in[worldid, actuator_actadr[actid] + slot_Ta]                  <L 1641>
                        var_192 = wp::address(var_actuator_actadr, var_1);
                        var_194 = wp::load(var_192);
                        var_193 = wp::add(var_194, var_189);
                        var_195 = wp::address(var_act_in, var_0, var_193);
                        var_197 = wp::load(var_195);
                        var_196 = wp::copy(var_197);
                        // R *= 1.0 + gainprm[2] * (T + dynprm[4] - gainprm[3])                   <L 1642>
                        var_200 = wp::extract(var_40, var_199);
                        var_202 = wp::extract(var_56, var_201);
                        var_203 = wp::add(var_196, var_202);
                        var_205 = wp::extract(var_40, var_204);
                        var_206 = wp::sub(var_203, var_205);
                        var_207 = wp::mul(var_200, var_206);
                        var_208 = wp::add(var_198, var_207);
                        var_209 = wp::mul(var_186, var_208);
                    }
                    var_210 = wp::where(var_191, var_209, var_186);
                    var_211 = wp::where(var_191, var_196, var_116);
                    // dVdl = float(0.0)                                                          <L 1644>
                    var_213 = wp::float(var_212);
                    // dVdw = float(0.0)                                                          <L 1645>
                    var_215 = wp::float(var_214);
                    // if (actuator_ctrlspec[actid] & 7) != 0:                                    <L 1646>
                    var_216 = wp::address(var_actuator_ctrlspec, var_1);
                    var_219 = wp::load(var_216);
                    var_218 = wp::bit_and(var_219, var_217);
                    var_221 = (var_218 != var_220);
                    if (var_221) {
                        // dVdl = math.safe_div(-gainprm[4] * R0, K)                              <L 1647>
                        var_223 = wp::extract(var_40, var_222);
                        var_224 = wp::neg(var_223);
                        var_225 = wp::mul(var_224, var_182);
                        var_226 = safe_div_0(var_225, var_184);
                        // dVdw = math.safe_div(-gainprm[6] * R0, K) + K                          <L 1648>
                        var_228 = wp::extract(var_40, var_227);
                        var_229 = wp::neg(var_228);
                        var_230 = wp::mul(var_229, var_182);
                        var_231 = safe_div_0(var_230, var_184);
                        var_232 = wp::add(var_231, var_184);
                    }
                    var_233 = wp::where(var_221, var_226, var_213);
                    var_234 = wp::where(var_221, var_232, var_215);
                    // if te > 0.0:                                                               <L 1650>
                    var_236 = (var_180 > var_235);
                    if (var_236) {
                        // s = 1.0 - wp.exp(-h / te)                                              <L 1651>
                        var_238 = wp::neg(var_8);
                        var_239 = wp::div(var_238, var_180);
                        var_240 = wp::exp(var_239);
                        var_241 = wp::sub(var_237, var_240);
                        // bias_len += math.safe_div(K * dVdl * s, R0)                            <L 1652>
                        var_242 = wp::mul(var_184, var_233);
                        var_243 = wp::mul(var_242, var_241);
                        var_244 = safe_div_0(var_243, var_182);
                        var_245 = wp::add(var_138, var_244);
                        // bias_vel += math.safe_div(K * (dVdw - K) * s, R)                       <L 1653>
                        var_246 = wp::sub(var_234, var_184);
                        var_247 = wp::mul(var_184, var_246);
                        var_248 = wp::mul(var_247, var_241);
                        var_249 = safe_div_0(var_248, var_210);
                        var_250 = wp::add(var_139, var_249);
                    }
                    if (!var_236) {
                        // if dVdl != 0.0:                                                        <L 1655>
                        var_252 = (var_233 != var_251);
                        if (var_252) {
                            // bias_len += math.safe_div(K * dVdl, R0)                            <L 1656>
                            var_253 = wp::mul(var_184, var_233);
                            var_254 = safe_div_0(var_253, var_182);
                            var_255 = wp::add(var_138, var_254);
                        }
                        var_256 = wp::where(var_252, var_255, var_138);
                        // if dVdw != 0.0:                                                        <L 1657>
                        var_258 = (var_234 != var_257);
                        if (var_258) {
                            // bias_vel += math.safe_div(K * dVdw, R)                             <L 1658>
                            var_259 = wp::mul(var_184, var_234);
                            var_260 = safe_div_0(var_259, var_210);
                            var_261 = wp::add(var_139, var_260);
                        }
                        var_262 = wp::where(var_258, var_261, var_139);
                    }
                    var_263 = wp::where(var_236, var_245, var_256);
                    var_264 = wp::where(var_236, var_250, var_262);
                    // sigma1 = dynprm[6]                                                         <L 1660>
                    var_266 = wp::extract(var_56, var_265);
                    // if sigma1 > 0.0:                                                           <L 1661>
                    var_268 = (var_266 > var_267);
                    if (var_268) {
                        // bias_vel -= sigma1                                                     <L 1662>
                        var_269 = wp::sub(var_264, var_266);
                    }
                    var_270 = wp::where(var_268, var_269, var_264);
                }
                var_271 = wp::where(var_178, var_263, var_138);
                var_272 = wp::where(var_178, var_270, var_139);
                var_273 = wp::where(var_178, var_210, var_130);
                var_274 = wp::where(var_178, var_184, var_106);
                var_275 = wp::where(var_178, var_187, var_107);
                var_276 = wp::where(var_178, var_189, var_109);
                var_277 = wp::where(var_178, var_211, var_116);
            }
            var_278 = wp::where(var_150, var_138, var_271);
            var_279 = wp::where(var_150, var_169, var_61);
            var_280 = wp::where(var_150, var_139, var_272);
            var_281 = wp::where(var_150, var_174, var_65);
            var_282 = wp::where(var_150, var_130, var_273);
            var_283 = wp::where(var_150, var_106, var_274);
            var_284 = wp::where(var_150, var_107, var_275);
            var_285 = wp::where(var_150, var_109, var_276);
            var_286 = wp::where(var_150, var_116, var_277);
        }
        var_287 = wp::where(var_144, var_138, var_278);
        var_288 = wp::where(var_144, var_146, var_279);
        var_289 = wp::where(var_144, var_139, var_280);
        var_290 = wp::where(var_144, var_148, var_281);
        var_291 = wp::where(var_144, var_130, var_282);
        var_292 = wp::where(var_144, var_106, var_283);
        var_293 = wp::where(var_144, var_107, var_284);
        var_294 = wp::where(var_144, var_109, var_285);
        var_295 = wp::where(var_144, var_116, var_286);
        // if gain_len != 0.0 or gain_vel != 0.0:                                                 <L 1664>
        var_298 = (var_288 != var_297);
        var_296 = var_298;
        if (!var_296) {
            var_300 = (var_290 != var_299);
            var_296 = var_296 || var_300;
        }
        if (var_296) {
            // u_val = float(0.0)                                                                 <L 1665>
            var_302 = wp::float(var_301);
            // if actuator_dyntype[actid] != DynType.NONE:                                        <L 1666>
            var_303 = wp::address(var_actuator_dyntype, var_1);
            var_306 = wp::load(var_303);
            var_305 = (var_306 != var_304);
            if (var_305) {
                // act_adr = actuator_actadr[actid] + actuator_actnum[actid] - 1                  <L 1667>
                var_307 = wp::address(var_actuator_actadr, var_1);
                var_308 = wp::address(var_actuator_actnum, var_1);
                var_310 = wp::load(var_307);
                var_311 = wp::load(var_308);
                var_309 = wp::add(var_310, var_311);
                var_313 = wp::sub(var_309, var_312);
                // if actuator_actearly[actid]:                                                   <L 1668>
                var_314 = wp::address(var_actuator_actearly, var_1);
                var_315 = wp::load(var_314);
                if (var_315) {
                    // u_val = next_act(                                                          <L 1669>
                    // h,                                                                         <L 1670>
                    // actuator_dyntype[actid],                                                   <L 1671>
                    var_316 = wp::address(var_actuator_dyntype, var_1);
                    // dynprm,                                                                    <L 1672>
                    // actuator_actrange[worldid % actuator_actrange.shape[0], actid],            <L 1673>
                    var_317 = &(var_actuator_actrange.shape);
                    var_320 = wp::load(var_317);
                    var_319 = wp::extract(var_320, var_318);
                    var_321 = wp::mod(var_0, var_319);
                    var_322 = wp::address(var_actuator_actrange, var_321, var_1);
                    // act_in[worldid, act_adr],                                                  <L 1674>
                    var_323 = wp::address(var_act_in, var_0, var_313);
                    // act_dot_in[worldid, act_adr],                                              <L 1675>
                    var_324 = wp::address(var_act_dot_in, var_0, var_313);
                    // 1.0,                                                                       <L 1676>
                    // actuator_actlimited[actid],                                                <L 1677>
                    var_326 = wp::address(var_actuator_actlimited, var_1);
                    var_328 = wp::load(var_316);
                    var_329 = wp::load(var_322);
                    var_330 = wp::load(var_323);
                    var_331 = wp::load(var_324);
                    var_332 = wp::load(var_326);
                    var_327 = next_act_0(var_8, var_328, var_56, var_329, var_330, var_331, var_325, var_332);
                }
                var_333 = wp::load(var_314);
                var_334 = wp::load(var_314);
                if (!var_334) {
                    // u_val = act_in[worldid, act_adr]                                           <L 1680>
                    var_335 = wp::address(var_act_in, var_0, var_313);
                    var_337 = wp::load(var_335);
                    var_336 = wp::copy(var_337);
                }
                var_338 = wp::load(var_314);
                var_340 = wp::load(var_314);
                var_339 = wp::where(var_340, var_327, var_336);
            }
            if (!var_305) {
                // cadr = actuator_ctrladr[actid]                                                 <L 1682>
                var_341 = wp::address(var_actuator_ctrladr, var_1);
                var_343 = wp::load(var_341);
                var_342 = wp::copy(var_343);
                // if cadr >= 0:                                                                  <L 1683>
                var_345 = (var_342 >= var_344);
                if (var_345) {
                    // u_val = ctrl_in[worldid, cadr]                                             <L 1684>
                    var_346 = wp::address(var_ctrl_in, var_0, var_342);
                    var_348 = wp::load(var_346);
                    var_347 = wp::copy(var_348);
                    // if not (opt_disableflags & DisableBit.CLAMPCTRL) and actuator_ctrllimited[cadr]:       <L 1685>
                    var_351 = wp::bit_and(var_opt_disableflags, var_350);
                    var_352 = wp::unot(var_351);
                    var_349 = var_352;
                    if (var_349) {
                        var_353 = wp::address(var_actuator_ctrllimited, var_342);
                        var_354 = wp::load(var_353);
                        var_349 = var_349 && var_354;
                    }
                    if (var_349) {
                        // crange = actuator_ctrlrange[worldid % actuator_ctrlrange.shape[0], cadr]       <L 1686>
                        var_355 = &(var_actuator_ctrlrange.shape);
                        var_358 = wp::load(var_355);
                        var_357 = wp::extract(var_358, var_356);
                        var_359 = wp::mod(var_0, var_357);
                        var_360 = wp::address(var_actuator_ctrlrange, var_359, var_342);
                        var_362 = wp::load(var_360);
                        var_361 = wp::copy(var_362);
                        // u_val = wp.clamp(u_val, crange[0], crange[1])                          <L 1687>
                        var_364 = wp::extract(var_361, var_363);
                        var_366 = wp::extract(var_361, var_365);
                        var_367 = wp::clamp(var_347, var_364, var_366);
                    }
                    var_368 = wp::where(var_349, var_367, var_347);
                }
                var_369 = wp::where(var_345, var_368, var_302);
            }
            var_370 = wp::where(var_305, var_339, var_369);
            // bias_len += gain_len * u_val                                                       <L 1688>
            var_371 = wp::mul(var_288, var_370);
            var_372 = wp::add(var_287, var_371);
            // bias_vel += gain_vel * u_val                                                       <L 1689>
            var_373 = wp::mul(var_290, var_370);
            var_374 = wp::add(var_289, var_373);
        }
        var_375 = wp::where(var_296, var_372, var_287);
        var_376 = wp::where(var_296, var_374, var_289);
        // gp = wp.max(0.0, -bias_len)                                                            <L 1691>
        var_378 = wp::neg(var_375);
        var_379 = wp::max(var_377, var_378);
        // gv = wp.max(0.0, -bias_vel)                                                            <L 1692>
        var_381 = wp::neg(var_376);
        var_382 = wp::max(var_380, var_381);
        // as_val = h * h * gp + h * gv                                                           <L 1694>
        var_383 = wp::mul(var_8, var_8);
        var_384 = wp::mul(var_383, var_379);
        var_385 = wp::mul(var_8, var_382);
        var_386 = wp::add(var_384, var_385);
        // ak_val = h * gp                                                                        <L 1695>
        var_387 = wp::mul(var_8, var_379);
        // efm_as_out[worldid, actid] = as_val                                                    <L 1696>
        wp::array_store(var_efm_as_out, var_0, var_1, var_386);
        // if gp > 0.0:                                                                           <L 1698>
        var_389 = (var_379 > var_388);
        if (var_389) {
            // ckv = ak_val * actuator_velocity_in[worldid, actid]                                <L 1699>
            var_390 = wp::address(var_actuator_velocity_in, var_0, var_1);
            var_392 = wp::load(var_390);
            var_391 = wp::mul(var_387, var_392);
            // rowadr = moment_rowadr_in[worldid, actid]                                          <L 1700>
            var_393 = wp::address(var_moment_rowadr_in, var_0, var_1);
            var_395 = wp::load(var_393);
            var_394 = wp::copy(var_395);
            // rownnz = moment_rownnz_in[worldid, actid]                                          <L 1701>
            var_396 = wp::address(var_moment_rownnz_in, var_0, var_1);
            var_398 = wp::load(var_396);
            var_397 = wp::copy(var_398);
            // for j in range(rownnz):                                                            <L 1702>
            var_399 = wp::range(var_397);
            start_for_2:;
                if (iter_cmp(var_399) == 0) goto end_for_2;
                var_400 = wp::iter_next(var_399);
                // sparseid = rowadr + j                                                          <L 1703>
                var_401 = wp::add(var_394, var_400);
                // col = moment_colind_in[worldid, sparseid]                                      <L 1704>
                var_402 = wp::address(var_moment_colind_in, var_0, var_401);
                var_404 = wp::load(var_402);
                var_403 = wp::copy(var_404);
                // val = actuator_moment_in[worldid, sparseid]                                    <L 1705>
                var_405 = wp::address(var_actuator_moment_in, var_0, var_401);
                var_407 = wp::load(var_405);
                var_406 = wp::copy(var_407);
                // wp.atomic_sub(efm_ca_out, worldid, col, ckv * val)                             <L 1706>
                var_408 = wp::mul(var_391, var_406);
                var_409 = wp::atomic_sub(var_efm_ca_out, var_0, var_403, var_408);
                goto start_for_2;
            end_for_2:;
        }
}

