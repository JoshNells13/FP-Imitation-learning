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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_driver.py:98
static CUDA_CALLABLE bool _plane_filter_0(
    wp::float32 var_size1,
    wp::float32 var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2,
    wp::mat_t<3, 3, wp::float32> var_xmat1,
    wp::mat_t<3, 3, wp::float32> var_xmat2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 2;
    wp::float32 var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    const wp::int32 var_9 = 2;
    const wp::int32 var_10 = 2;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    bool var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    wp::vec_t<3, wp::float32> var_19;
    const wp::int32 var_20 = 0;
    const wp::int32 var_21 = 2;
    wp::float32 var_22;
    const wp::int32 var_23 = 1;
    const wp::int32 var_24 = 2;
    wp::float32 var_25;
    const wp::int32 var_26 = 2;
    const wp::int32 var_27 = 2;
    wp::float32 var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    bool var_33;
    wp::float32 var_34;
    const bool var_35 = true;
    //---------
    // forward
    // def _plane_filter(                                                                     <L 99>
    // if size1 == 0.0:                                                                       <L 102>
    var_1 = (var_size1 == var_0);
    if (var_1) {
        // dist = wp.dot(xpos2 - xpos1, wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2]))       <L 104>
        var_2 = wp::sub(var_xpos2, var_xpos1);
        var_5 = wp::extract(var_xmat1, var_3, var_4);
        var_8 = wp::extract(var_xmat1, var_6, var_7);
        var_11 = wp::extract(var_xmat1, var_9, var_10);
        var_12 = wp::vec_t<3, wp::float32>(var_5, var_8, var_11);
        var_13 = wp::dot(var_2, var_12);
        // return dist <= size2 + margin1 + margin2                                           <L 105>
        var_14 = wp::add(var_size2, var_margin1);
        var_15 = wp::add(var_14, var_margin2);
        var_16 = (var_13 <= var_15);
        return var_16;
    }
    if (!var_1) {
        // elif size2 == 0.0:                                                                 <L 106>
        var_18 = (var_size2 == var_17);
        if (var_18) {
            // dist = wp.dot(xpos1 - xpos2, wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2]))       <L 108>
            var_19 = wp::sub(var_xpos1, var_xpos2);
            var_22 = wp::extract(var_xmat2, var_20, var_21);
            var_25 = wp::extract(var_xmat2, var_23, var_24);
            var_28 = wp::extract(var_xmat2, var_26, var_27);
            var_29 = wp::vec_t<3, wp::float32>(var_22, var_25, var_28);
            var_30 = wp::dot(var_19, var_29);
            // return dist <= size1 + margin1 + margin2                                       <L 109>
            var_31 = wp::add(var_size1, var_margin1);
            var_32 = wp::add(var_31, var_margin2);
            var_33 = (var_30 <= var_32);
            return var_33;
        }
    }
    var_34 = wp::where(var_1, var_13, var_30);
    // return True                                                                            <L 111>
    return var_35;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_driver.py:114
static CUDA_CALLABLE bool _sphere_filter_0(
    wp::float32 var_size1,
    wp::float32 var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    //---------
    // forward
    // def _sphere_filter(size1: float, size2: float, margin1: float, margin2: float, xpos1: wp.vec3, xpos2: wp.vec3) -> bool:       <L 115>
    // bound = size1 + size2 + margin1 + margin2                                              <L 116>
    var_0 = wp::add(var_size1, var_size2);
    var_1 = wp::add(var_0, var_margin1);
    var_2 = wp::add(var_1, var_margin2);
    // dif = xpos2 - xpos1                                                                    <L 117>
    var_3 = wp::sub(var_xpos2, var_xpos1);
    // dist_sq = wp.dot(dif, dif)                                                             <L 118>
    var_4 = wp::dot(var_3, var_3);
    // return dist_sq <= bound * bound                                                        <L 119>
    var_5 = wp::mul(var_2, var_2);
    var_6 = (var_4 <= var_5);
    return var_6;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_driver.py:225
static CUDA_CALLABLE bool _obb_filter_0(
    wp::vec_t<3, wp::float32> var_center1,
    wp::vec_t<3, wp::float32> var_center2,
    wp::vec_t<3, wp::float32> var_size1,
    wp::vec_t<3, wp::float32> var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2,
    wp::mat_t<3, 3, wp::float32> var_xmat1,
    wp::mat_t<3, 3, wp::float32> var_xmat2)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::mat_t<2, 3, wp::float32> var_1;
    wp::mat_t<6, 3, wp::float32> var_2;
    wp::vec_t<2, wp::float32> var_3;
    wp::vec_t<2, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 1;
    const wp::int32 var_11 = 0;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 0;
    const wp::int32 var_22 = 0;
    const wp::int32 var_23 = 1;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    const wp::int32 var_26 = 1;
    wp::float32 var_27;
    const wp::int32 var_28 = 2;
    const wp::int32 var_29 = 1;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    const wp::int32 var_32 = 1;
    const wp::int32 var_33 = 0;
    const wp::int32 var_34 = 2;
    wp::float32 var_35;
    const wp::int32 var_36 = 1;
    const wp::int32 var_37 = 2;
    wp::float32 var_38;
    const wp::int32 var_39 = 2;
    const wp::int32 var_40 = 2;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    const wp::int32 var_43 = 2;
    const wp::int32 var_44 = 0;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    const wp::int32 var_48 = 0;
    wp::float32 var_49;
    const wp::int32 var_50 = 2;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    wp::vec_t<3, wp::float32> var_53;
    const wp::int32 var_54 = 3;
    const wp::int32 var_55 = 0;
    const wp::int32 var_56 = 1;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    const wp::int32 var_62 = 1;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    const wp::int32 var_65 = 4;
    const wp::int32 var_66 = 0;
    const wp::int32 var_67 = 2;
    wp::float32 var_68;
    const wp::int32 var_69 = 1;
    const wp::int32 var_70 = 2;
    wp::float32 var_71;
    const wp::int32 var_72 = 2;
    const wp::int32 var_73 = 2;
    wp::float32 var_74;
    wp::vec_t<3, wp::float32> var_75;
    const wp::int32 var_76 = 5;
    const wp::int32 var_77 = 0;
    const wp::int32 var_78 = 0;
    const wp::int32 var_79 = 0;
    wp::vec_t<3, wp::float32> var_80;
    const wp::int32 var_81 = 3;
    wp::int32 var_82;
    wp::int32 var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::float32 var_85;
    const wp::int32 var_86 = 0;
    bool var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    const wp::int32 var_91 = 0;
    wp::float32 var_92;
    const wp::int32 var_93 = 3;
    wp::int32 var_94;
    const wp::int32 var_95 = 0;
    wp::int32 var_96;
    wp::vec_t<3, wp::float32> var_97;
    const wp::int32 var_98 = 3;
    wp::int32 var_99;
    wp::int32 var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    const wp::int32 var_105 = 1;
    wp::float32 var_106;
    const wp::int32 var_107 = 3;
    wp::int32 var_108;
    const wp::int32 var_109 = 1;
    wp::int32 var_110;
    wp::vec_t<3, wp::float32> var_111;
    const wp::int32 var_112 = 3;
    wp::int32 var_113;
    wp::int32 var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    wp::float32 var_119;
    const wp::int32 var_120 = 2;
    wp::float32 var_121;
    const wp::int32 var_122 = 3;
    wp::int32 var_123;
    const wp::int32 var_124 = 2;
    wp::int32 var_125;
    wp::vec_t<3, wp::float32> var_126;
    const wp::int32 var_127 = 3;
    wp::int32 var_128;
    wp::int32 var_129;
    wp::vec_t<3, wp::float32> var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    wp::float32 var_133;
    wp::float32 var_134;
    const wp::int32 var_135 = 1;
    wp::vec_t<3, wp::float32> var_136;
    const wp::int32 var_137 = 3;
    wp::int32 var_138;
    wp::int32 var_139;
    wp::vec_t<3, wp::float32> var_140;
    wp::float32 var_141;
    const wp::int32 var_142 = 0;
    bool var_143;
    wp::vec_t<3, wp::float32> var_144;
    wp::vec_t<3, wp::float32> var_145;
    wp::vec_t<3, wp::float32> var_146;
    const wp::int32 var_147 = 0;
    wp::float32 var_148;
    const wp::int32 var_149 = 3;
    wp::int32 var_150;
    const wp::int32 var_151 = 0;
    wp::int32 var_152;
    wp::vec_t<3, wp::float32> var_153;
    const wp::int32 var_154 = 3;
    wp::int32 var_155;
    wp::int32 var_156;
    wp::vec_t<3, wp::float32> var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::float32 var_160;
    const wp::int32 var_161 = 1;
    wp::float32 var_162;
    const wp::int32 var_163 = 3;
    wp::int32 var_164;
    const wp::int32 var_165 = 1;
    wp::int32 var_166;
    wp::vec_t<3, wp::float32> var_167;
    const wp::int32 var_168 = 3;
    wp::int32 var_169;
    wp::int32 var_170;
    wp::vec_t<3, wp::float32> var_171;
    wp::float32 var_172;
    wp::float32 var_173;
    wp::float32 var_174;
    wp::float32 var_175;
    const wp::int32 var_176 = 2;
    wp::float32 var_177;
    const wp::int32 var_178 = 3;
    wp::int32 var_179;
    const wp::int32 var_180 = 2;
    wp::int32 var_181;
    wp::vec_t<3, wp::float32> var_182;
    const wp::int32 var_183 = 3;
    wp::int32 var_184;
    wp::int32 var_185;
    wp::vec_t<3, wp::float32> var_186;
    wp::float32 var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::float32 var_190;
    const wp::int32 var_191 = 0;
    wp::float32 var_192;
    const wp::int32 var_193 = 1;
    wp::float32 var_194;
    wp::float32 var_195;
    wp::float32 var_196;
    const wp::int32 var_197 = 1;
    wp::float32 var_198;
    const wp::int32 var_199 = 0;
    wp::float32 var_200;
    wp::float32 var_201;
    wp::float32 var_202;
    bool var_203;
    const bool var_204 = false;
    const wp::int32 var_205 = 1;
    const wp::int32 var_206 = 0;
    wp::vec_t<3, wp::float32> var_207;
    const wp::int32 var_208 = 3;
    wp::int32 var_209;
    wp::int32 var_210;
    wp::vec_t<3, wp::float32> var_211;
    wp::float32 var_212;
    const wp::int32 var_213 = 0;
    bool var_214;
    wp::vec_t<3, wp::float32> var_215;
    wp::vec_t<3, wp::float32> var_216;
    wp::vec_t<3, wp::float32> var_217;
    const wp::int32 var_218 = 0;
    wp::float32 var_219;
    const wp::int32 var_220 = 3;
    wp::int32 var_221;
    const wp::int32 var_222 = 0;
    wp::int32 var_223;
    wp::vec_t<3, wp::float32> var_224;
    const wp::int32 var_225 = 3;
    wp::int32 var_226;
    wp::int32 var_227;
    wp::vec_t<3, wp::float32> var_228;
    wp::float32 var_229;
    wp::float32 var_230;
    wp::float32 var_231;
    const wp::int32 var_232 = 1;
    wp::float32 var_233;
    const wp::int32 var_234 = 3;
    wp::int32 var_235;
    const wp::int32 var_236 = 1;
    wp::int32 var_237;
    wp::vec_t<3, wp::float32> var_238;
    const wp::int32 var_239 = 3;
    wp::int32 var_240;
    wp::int32 var_241;
    wp::vec_t<3, wp::float32> var_242;
    wp::float32 var_243;
    wp::float32 var_244;
    wp::float32 var_245;
    wp::float32 var_246;
    const wp::int32 var_247 = 2;
    wp::float32 var_248;
    const wp::int32 var_249 = 3;
    wp::int32 var_250;
    const wp::int32 var_251 = 2;
    wp::int32 var_252;
    wp::vec_t<3, wp::float32> var_253;
    const wp::int32 var_254 = 3;
    wp::int32 var_255;
    wp::int32 var_256;
    wp::vec_t<3, wp::float32> var_257;
    wp::float32 var_258;
    wp::float32 var_259;
    wp::float32 var_260;
    wp::float32 var_261;
    const wp::int32 var_262 = 1;
    wp::vec_t<3, wp::float32> var_263;
    const wp::int32 var_264 = 3;
    wp::int32 var_265;
    wp::int32 var_266;
    wp::vec_t<3, wp::float32> var_267;
    wp::float32 var_268;
    const wp::int32 var_269 = 0;
    bool var_270;
    wp::vec_t<3, wp::float32> var_271;
    wp::vec_t<3, wp::float32> var_272;
    wp::vec_t<3, wp::float32> var_273;
    const wp::int32 var_274 = 0;
    wp::float32 var_275;
    const wp::int32 var_276 = 3;
    wp::int32 var_277;
    const wp::int32 var_278 = 0;
    wp::int32 var_279;
    wp::vec_t<3, wp::float32> var_280;
    const wp::int32 var_281 = 3;
    wp::int32 var_282;
    wp::int32 var_283;
    wp::vec_t<3, wp::float32> var_284;
    wp::float32 var_285;
    wp::float32 var_286;
    wp::float32 var_287;
    const wp::int32 var_288 = 1;
    wp::float32 var_289;
    const wp::int32 var_290 = 3;
    wp::int32 var_291;
    const wp::int32 var_292 = 1;
    wp::int32 var_293;
    wp::vec_t<3, wp::float32> var_294;
    const wp::int32 var_295 = 3;
    wp::int32 var_296;
    wp::int32 var_297;
    wp::vec_t<3, wp::float32> var_298;
    wp::float32 var_299;
    wp::float32 var_300;
    wp::float32 var_301;
    wp::float32 var_302;
    const wp::int32 var_303 = 2;
    wp::float32 var_304;
    const wp::int32 var_305 = 3;
    wp::int32 var_306;
    const wp::int32 var_307 = 2;
    wp::int32 var_308;
    wp::vec_t<3, wp::float32> var_309;
    const wp::int32 var_310 = 3;
    wp::int32 var_311;
    wp::int32 var_312;
    wp::vec_t<3, wp::float32> var_313;
    wp::float32 var_314;
    wp::float32 var_315;
    wp::float32 var_316;
    wp::float32 var_317;
    const wp::int32 var_318 = 0;
    wp::float32 var_319;
    const wp::int32 var_320 = 1;
    wp::float32 var_321;
    wp::float32 var_322;
    wp::float32 var_323;
    const wp::int32 var_324 = 1;
    wp::float32 var_325;
    const wp::int32 var_326 = 0;
    wp::float32 var_327;
    wp::float32 var_328;
    wp::float32 var_329;
    bool var_330;
    const bool var_331 = false;
    const wp::int32 var_332 = 2;
    const wp::int32 var_333 = 0;
    wp::vec_t<3, wp::float32> var_334;
    const wp::int32 var_335 = 3;
    wp::int32 var_336;
    wp::int32 var_337;
    wp::vec_t<3, wp::float32> var_338;
    wp::float32 var_339;
    const wp::int32 var_340 = 0;
    bool var_341;
    wp::vec_t<3, wp::float32> var_342;
    wp::vec_t<3, wp::float32> var_343;
    wp::vec_t<3, wp::float32> var_344;
    const wp::int32 var_345 = 0;
    wp::float32 var_346;
    const wp::int32 var_347 = 3;
    wp::int32 var_348;
    const wp::int32 var_349 = 0;
    wp::int32 var_350;
    wp::vec_t<3, wp::float32> var_351;
    const wp::int32 var_352 = 3;
    wp::int32 var_353;
    wp::int32 var_354;
    wp::vec_t<3, wp::float32> var_355;
    wp::float32 var_356;
    wp::float32 var_357;
    wp::float32 var_358;
    const wp::int32 var_359 = 1;
    wp::float32 var_360;
    const wp::int32 var_361 = 3;
    wp::int32 var_362;
    const wp::int32 var_363 = 1;
    wp::int32 var_364;
    wp::vec_t<3, wp::float32> var_365;
    const wp::int32 var_366 = 3;
    wp::int32 var_367;
    wp::int32 var_368;
    wp::vec_t<3, wp::float32> var_369;
    wp::float32 var_370;
    wp::float32 var_371;
    wp::float32 var_372;
    wp::float32 var_373;
    const wp::int32 var_374 = 2;
    wp::float32 var_375;
    const wp::int32 var_376 = 3;
    wp::int32 var_377;
    const wp::int32 var_378 = 2;
    wp::int32 var_379;
    wp::vec_t<3, wp::float32> var_380;
    const wp::int32 var_381 = 3;
    wp::int32 var_382;
    wp::int32 var_383;
    wp::vec_t<3, wp::float32> var_384;
    wp::float32 var_385;
    wp::float32 var_386;
    wp::float32 var_387;
    wp::float32 var_388;
    const wp::int32 var_389 = 1;
    wp::vec_t<3, wp::float32> var_390;
    const wp::int32 var_391 = 3;
    wp::int32 var_392;
    wp::int32 var_393;
    wp::vec_t<3, wp::float32> var_394;
    wp::float32 var_395;
    const wp::int32 var_396 = 0;
    bool var_397;
    wp::vec_t<3, wp::float32> var_398;
    wp::vec_t<3, wp::float32> var_399;
    wp::vec_t<3, wp::float32> var_400;
    const wp::int32 var_401 = 0;
    wp::float32 var_402;
    const wp::int32 var_403 = 3;
    wp::int32 var_404;
    const wp::int32 var_405 = 0;
    wp::int32 var_406;
    wp::vec_t<3, wp::float32> var_407;
    const wp::int32 var_408 = 3;
    wp::int32 var_409;
    wp::int32 var_410;
    wp::vec_t<3, wp::float32> var_411;
    wp::float32 var_412;
    wp::float32 var_413;
    wp::float32 var_414;
    const wp::int32 var_415 = 1;
    wp::float32 var_416;
    const wp::int32 var_417 = 3;
    wp::int32 var_418;
    const wp::int32 var_419 = 1;
    wp::int32 var_420;
    wp::vec_t<3, wp::float32> var_421;
    const wp::int32 var_422 = 3;
    wp::int32 var_423;
    wp::int32 var_424;
    wp::vec_t<3, wp::float32> var_425;
    wp::float32 var_426;
    wp::float32 var_427;
    wp::float32 var_428;
    wp::float32 var_429;
    const wp::int32 var_430 = 2;
    wp::float32 var_431;
    const wp::int32 var_432 = 3;
    wp::int32 var_433;
    const wp::int32 var_434 = 2;
    wp::int32 var_435;
    wp::vec_t<3, wp::float32> var_436;
    const wp::int32 var_437 = 3;
    wp::int32 var_438;
    wp::int32 var_439;
    wp::vec_t<3, wp::float32> var_440;
    wp::float32 var_441;
    wp::float32 var_442;
    wp::float32 var_443;
    wp::float32 var_444;
    const wp::int32 var_445 = 0;
    wp::float32 var_446;
    const wp::int32 var_447 = 1;
    wp::float32 var_448;
    wp::float32 var_449;
    wp::float32 var_450;
    const wp::int32 var_451 = 1;
    wp::float32 var_452;
    const wp::int32 var_453 = 0;
    wp::float32 var_454;
    wp::float32 var_455;
    wp::float32 var_456;
    bool var_457;
    const bool var_458 = false;
    const wp::int32 var_459 = 1;
    const wp::int32 var_460 = 0;
    const wp::int32 var_461 = 0;
    wp::vec_t<3, wp::float32> var_462;
    const wp::int32 var_463 = 3;
    wp::int32 var_464;
    wp::int32 var_465;
    wp::vec_t<3, wp::float32> var_466;
    wp::float32 var_467;
    const wp::int32 var_468 = 0;
    bool var_469;
    wp::vec_t<3, wp::float32> var_470;
    wp::vec_t<3, wp::float32> var_471;
    wp::vec_t<3, wp::float32> var_472;
    const wp::int32 var_473 = 0;
    wp::float32 var_474;
    const wp::int32 var_475 = 3;
    wp::int32 var_476;
    const wp::int32 var_477 = 0;
    wp::int32 var_478;
    wp::vec_t<3, wp::float32> var_479;
    const wp::int32 var_480 = 3;
    wp::int32 var_481;
    wp::int32 var_482;
    wp::vec_t<3, wp::float32> var_483;
    wp::float32 var_484;
    wp::float32 var_485;
    wp::float32 var_486;
    const wp::int32 var_487 = 1;
    wp::float32 var_488;
    const wp::int32 var_489 = 3;
    wp::int32 var_490;
    const wp::int32 var_491 = 1;
    wp::int32 var_492;
    wp::vec_t<3, wp::float32> var_493;
    const wp::int32 var_494 = 3;
    wp::int32 var_495;
    wp::int32 var_496;
    wp::vec_t<3, wp::float32> var_497;
    wp::float32 var_498;
    wp::float32 var_499;
    wp::float32 var_500;
    wp::float32 var_501;
    const wp::int32 var_502 = 2;
    wp::float32 var_503;
    const wp::int32 var_504 = 3;
    wp::int32 var_505;
    const wp::int32 var_506 = 2;
    wp::int32 var_507;
    wp::vec_t<3, wp::float32> var_508;
    const wp::int32 var_509 = 3;
    wp::int32 var_510;
    wp::int32 var_511;
    wp::vec_t<3, wp::float32> var_512;
    wp::float32 var_513;
    wp::float32 var_514;
    wp::float32 var_515;
    wp::float32 var_516;
    const wp::int32 var_517 = 1;
    wp::vec_t<3, wp::float32> var_518;
    const wp::int32 var_519 = 3;
    wp::int32 var_520;
    wp::int32 var_521;
    wp::vec_t<3, wp::float32> var_522;
    wp::float32 var_523;
    const wp::int32 var_524 = 0;
    bool var_525;
    wp::vec_t<3, wp::float32> var_526;
    wp::vec_t<3, wp::float32> var_527;
    wp::vec_t<3, wp::float32> var_528;
    const wp::int32 var_529 = 0;
    wp::float32 var_530;
    const wp::int32 var_531 = 3;
    wp::int32 var_532;
    const wp::int32 var_533 = 0;
    wp::int32 var_534;
    wp::vec_t<3, wp::float32> var_535;
    const wp::int32 var_536 = 3;
    wp::int32 var_537;
    wp::int32 var_538;
    wp::vec_t<3, wp::float32> var_539;
    wp::float32 var_540;
    wp::float32 var_541;
    wp::float32 var_542;
    const wp::int32 var_543 = 1;
    wp::float32 var_544;
    const wp::int32 var_545 = 3;
    wp::int32 var_546;
    const wp::int32 var_547 = 1;
    wp::int32 var_548;
    wp::vec_t<3, wp::float32> var_549;
    const wp::int32 var_550 = 3;
    wp::int32 var_551;
    wp::int32 var_552;
    wp::vec_t<3, wp::float32> var_553;
    wp::float32 var_554;
    wp::float32 var_555;
    wp::float32 var_556;
    wp::float32 var_557;
    const wp::int32 var_558 = 2;
    wp::float32 var_559;
    const wp::int32 var_560 = 3;
    wp::int32 var_561;
    const wp::int32 var_562 = 2;
    wp::int32 var_563;
    wp::vec_t<3, wp::float32> var_564;
    const wp::int32 var_565 = 3;
    wp::int32 var_566;
    wp::int32 var_567;
    wp::vec_t<3, wp::float32> var_568;
    wp::float32 var_569;
    wp::float32 var_570;
    wp::float32 var_571;
    wp::float32 var_572;
    const wp::int32 var_573 = 0;
    wp::float32 var_574;
    const wp::int32 var_575 = 1;
    wp::float32 var_576;
    wp::float32 var_577;
    wp::float32 var_578;
    const wp::int32 var_579 = 1;
    wp::float32 var_580;
    const wp::int32 var_581 = 0;
    wp::float32 var_582;
    wp::float32 var_583;
    wp::float32 var_584;
    bool var_585;
    const bool var_586 = false;
    const wp::int32 var_587 = 1;
    const wp::int32 var_588 = 0;
    wp::vec_t<3, wp::float32> var_589;
    const wp::int32 var_590 = 3;
    wp::int32 var_591;
    wp::int32 var_592;
    wp::vec_t<3, wp::float32> var_593;
    wp::float32 var_594;
    const wp::int32 var_595 = 0;
    bool var_596;
    wp::vec_t<3, wp::float32> var_597;
    wp::vec_t<3, wp::float32> var_598;
    wp::vec_t<3, wp::float32> var_599;
    const wp::int32 var_600 = 0;
    wp::float32 var_601;
    const wp::int32 var_602 = 3;
    wp::int32 var_603;
    const wp::int32 var_604 = 0;
    wp::int32 var_605;
    wp::vec_t<3, wp::float32> var_606;
    const wp::int32 var_607 = 3;
    wp::int32 var_608;
    wp::int32 var_609;
    wp::vec_t<3, wp::float32> var_610;
    wp::float32 var_611;
    wp::float32 var_612;
    wp::float32 var_613;
    const wp::int32 var_614 = 1;
    wp::float32 var_615;
    const wp::int32 var_616 = 3;
    wp::int32 var_617;
    const wp::int32 var_618 = 1;
    wp::int32 var_619;
    wp::vec_t<3, wp::float32> var_620;
    const wp::int32 var_621 = 3;
    wp::int32 var_622;
    wp::int32 var_623;
    wp::vec_t<3, wp::float32> var_624;
    wp::float32 var_625;
    wp::float32 var_626;
    wp::float32 var_627;
    wp::float32 var_628;
    const wp::int32 var_629 = 2;
    wp::float32 var_630;
    const wp::int32 var_631 = 3;
    wp::int32 var_632;
    const wp::int32 var_633 = 2;
    wp::int32 var_634;
    wp::vec_t<3, wp::float32> var_635;
    const wp::int32 var_636 = 3;
    wp::int32 var_637;
    wp::int32 var_638;
    wp::vec_t<3, wp::float32> var_639;
    wp::float32 var_640;
    wp::float32 var_641;
    wp::float32 var_642;
    wp::float32 var_643;
    const wp::int32 var_644 = 1;
    wp::vec_t<3, wp::float32> var_645;
    const wp::int32 var_646 = 3;
    wp::int32 var_647;
    wp::int32 var_648;
    wp::vec_t<3, wp::float32> var_649;
    wp::float32 var_650;
    const wp::int32 var_651 = 0;
    bool var_652;
    wp::vec_t<3, wp::float32> var_653;
    wp::vec_t<3, wp::float32> var_654;
    wp::vec_t<3, wp::float32> var_655;
    const wp::int32 var_656 = 0;
    wp::float32 var_657;
    const wp::int32 var_658 = 3;
    wp::int32 var_659;
    const wp::int32 var_660 = 0;
    wp::int32 var_661;
    wp::vec_t<3, wp::float32> var_662;
    const wp::int32 var_663 = 3;
    wp::int32 var_664;
    wp::int32 var_665;
    wp::vec_t<3, wp::float32> var_666;
    wp::float32 var_667;
    wp::float32 var_668;
    wp::float32 var_669;
    const wp::int32 var_670 = 1;
    wp::float32 var_671;
    const wp::int32 var_672 = 3;
    wp::int32 var_673;
    const wp::int32 var_674 = 1;
    wp::int32 var_675;
    wp::vec_t<3, wp::float32> var_676;
    const wp::int32 var_677 = 3;
    wp::int32 var_678;
    wp::int32 var_679;
    wp::vec_t<3, wp::float32> var_680;
    wp::float32 var_681;
    wp::float32 var_682;
    wp::float32 var_683;
    wp::float32 var_684;
    const wp::int32 var_685 = 2;
    wp::float32 var_686;
    const wp::int32 var_687 = 3;
    wp::int32 var_688;
    const wp::int32 var_689 = 2;
    wp::int32 var_690;
    wp::vec_t<3, wp::float32> var_691;
    const wp::int32 var_692 = 3;
    wp::int32 var_693;
    wp::int32 var_694;
    wp::vec_t<3, wp::float32> var_695;
    wp::float32 var_696;
    wp::float32 var_697;
    wp::float32 var_698;
    wp::float32 var_699;
    const wp::int32 var_700 = 0;
    wp::float32 var_701;
    const wp::int32 var_702 = 1;
    wp::float32 var_703;
    wp::float32 var_704;
    wp::float32 var_705;
    const wp::int32 var_706 = 1;
    wp::float32 var_707;
    const wp::int32 var_708 = 0;
    wp::float32 var_709;
    wp::float32 var_710;
    wp::float32 var_711;
    bool var_712;
    const bool var_713 = false;
    const wp::int32 var_714 = 2;
    const wp::int32 var_715 = 0;
    wp::vec_t<3, wp::float32> var_716;
    const wp::int32 var_717 = 3;
    wp::int32 var_718;
    wp::int32 var_719;
    wp::vec_t<3, wp::float32> var_720;
    wp::float32 var_721;
    const wp::int32 var_722 = 0;
    bool var_723;
    wp::vec_t<3, wp::float32> var_724;
    wp::vec_t<3, wp::float32> var_725;
    wp::vec_t<3, wp::float32> var_726;
    const wp::int32 var_727 = 0;
    wp::float32 var_728;
    const wp::int32 var_729 = 3;
    wp::int32 var_730;
    const wp::int32 var_731 = 0;
    wp::int32 var_732;
    wp::vec_t<3, wp::float32> var_733;
    const wp::int32 var_734 = 3;
    wp::int32 var_735;
    wp::int32 var_736;
    wp::vec_t<3, wp::float32> var_737;
    wp::float32 var_738;
    wp::float32 var_739;
    wp::float32 var_740;
    const wp::int32 var_741 = 1;
    wp::float32 var_742;
    const wp::int32 var_743 = 3;
    wp::int32 var_744;
    const wp::int32 var_745 = 1;
    wp::int32 var_746;
    wp::vec_t<3, wp::float32> var_747;
    const wp::int32 var_748 = 3;
    wp::int32 var_749;
    wp::int32 var_750;
    wp::vec_t<3, wp::float32> var_751;
    wp::float32 var_752;
    wp::float32 var_753;
    wp::float32 var_754;
    wp::float32 var_755;
    const wp::int32 var_756 = 2;
    wp::float32 var_757;
    const wp::int32 var_758 = 3;
    wp::int32 var_759;
    const wp::int32 var_760 = 2;
    wp::int32 var_761;
    wp::vec_t<3, wp::float32> var_762;
    const wp::int32 var_763 = 3;
    wp::int32 var_764;
    wp::int32 var_765;
    wp::vec_t<3, wp::float32> var_766;
    wp::float32 var_767;
    wp::float32 var_768;
    wp::float32 var_769;
    wp::float32 var_770;
    const wp::int32 var_771 = 1;
    wp::vec_t<3, wp::float32> var_772;
    const wp::int32 var_773 = 3;
    wp::int32 var_774;
    wp::int32 var_775;
    wp::vec_t<3, wp::float32> var_776;
    wp::float32 var_777;
    const wp::int32 var_778 = 0;
    bool var_779;
    wp::vec_t<3, wp::float32> var_780;
    wp::vec_t<3, wp::float32> var_781;
    wp::vec_t<3, wp::float32> var_782;
    const wp::int32 var_783 = 0;
    wp::float32 var_784;
    const wp::int32 var_785 = 3;
    wp::int32 var_786;
    const wp::int32 var_787 = 0;
    wp::int32 var_788;
    wp::vec_t<3, wp::float32> var_789;
    const wp::int32 var_790 = 3;
    wp::int32 var_791;
    wp::int32 var_792;
    wp::vec_t<3, wp::float32> var_793;
    wp::float32 var_794;
    wp::float32 var_795;
    wp::float32 var_796;
    const wp::int32 var_797 = 1;
    wp::float32 var_798;
    const wp::int32 var_799 = 3;
    wp::int32 var_800;
    const wp::int32 var_801 = 1;
    wp::int32 var_802;
    wp::vec_t<3, wp::float32> var_803;
    const wp::int32 var_804 = 3;
    wp::int32 var_805;
    wp::int32 var_806;
    wp::vec_t<3, wp::float32> var_807;
    wp::float32 var_808;
    wp::float32 var_809;
    wp::float32 var_810;
    wp::float32 var_811;
    const wp::int32 var_812 = 2;
    wp::float32 var_813;
    const wp::int32 var_814 = 3;
    wp::int32 var_815;
    const wp::int32 var_816 = 2;
    wp::int32 var_817;
    wp::vec_t<3, wp::float32> var_818;
    const wp::int32 var_819 = 3;
    wp::int32 var_820;
    wp::int32 var_821;
    wp::vec_t<3, wp::float32> var_822;
    wp::float32 var_823;
    wp::float32 var_824;
    wp::float32 var_825;
    wp::float32 var_826;
    const wp::int32 var_827 = 0;
    wp::float32 var_828;
    const wp::int32 var_829 = 1;
    wp::float32 var_830;
    wp::float32 var_831;
    wp::float32 var_832;
    const wp::int32 var_833 = 1;
    wp::float32 var_834;
    const wp::int32 var_835 = 0;
    wp::float32 var_836;
    wp::float32 var_837;
    wp::float32 var_838;
    bool var_839;
    const bool var_840 = false;
    const bool var_841 = true;
    //---------
    // forward
    // def _obb_filter(                                                                       <L 226>
    // margin = margin1 + margin2                                                             <L 240>
    var_0 = wp::add(var_margin1, var_margin2);
    // xcenter = mat23()                                                                      <L 242>
    var_1 = wp::mat_t<2, 3, wp::float32>();
    // normal = mat63()                                                                       <L 243>
    var_2 = wp::mat_t<6, 3, wp::float32>();
    // proj = wp.vec2()                                                                       <L 244>
    var_3 = wp::vec_t<2, wp::float32>();
    // radius = wp.vec2()                                                                     <L 245>
    var_4 = wp::vec_t<2, wp::float32>();
    // xcenter[0] = xmat1 @ center1 + xpos1                                                   <L 248>
    var_5 = wp::mul(var_xmat1, var_center1);
    var_6 = wp::add(var_5, var_xpos1);
    wp::assign_inplace(var_1, var_7, var_6);
    // xcenter[1] = xmat2 @ center2 + xpos2                                                   <L 249>
    var_8 = wp::mul(var_xmat2, var_center2);
    var_9 = wp::add(var_8, var_xpos2);
    wp::assign_inplace(var_1, var_10, var_9);
    // normal[0] = wp.vec3(xmat1[0, 0], xmat1[1, 0], xmat1[2, 0])                             <L 252>
    var_13 = wp::extract(var_xmat1, var_11, var_12);
    var_16 = wp::extract(var_xmat1, var_14, var_15);
    var_19 = wp::extract(var_xmat1, var_17, var_18);
    var_20 = wp::vec_t<3, wp::float32>(var_13, var_16, var_19);
    wp::assign_inplace(var_2, var_21, var_20);
    // normal[1] = wp.vec3(xmat1[0, 1], xmat1[1, 1], xmat1[2, 1])                             <L 253>
    var_24 = wp::extract(var_xmat1, var_22, var_23);
    var_27 = wp::extract(var_xmat1, var_25, var_26);
    var_30 = wp::extract(var_xmat1, var_28, var_29);
    var_31 = wp::vec_t<3, wp::float32>(var_24, var_27, var_30);
    wp::assign_inplace(var_2, var_32, var_31);
    // normal[2] = wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2])                             <L 254>
    var_35 = wp::extract(var_xmat1, var_33, var_34);
    var_38 = wp::extract(var_xmat1, var_36, var_37);
    var_41 = wp::extract(var_xmat1, var_39, var_40);
    var_42 = wp::vec_t<3, wp::float32>(var_35, var_38, var_41);
    wp::assign_inplace(var_2, var_43, var_42);
    // normal[3] = wp.vec3(xmat2[0, 0], xmat2[1, 0], xmat2[2, 0])                             <L 255>
    var_46 = wp::extract(var_xmat2, var_44, var_45);
    var_49 = wp::extract(var_xmat2, var_47, var_48);
    var_52 = wp::extract(var_xmat2, var_50, var_51);
    var_53 = wp::vec_t<3, wp::float32>(var_46, var_49, var_52);
    wp::assign_inplace(var_2, var_54, var_53);
    // normal[4] = wp.vec3(xmat2[0, 1], xmat2[1, 1], xmat2[2, 1])                             <L 256>
    var_57 = wp::extract(var_xmat2, var_55, var_56);
    var_60 = wp::extract(var_xmat2, var_58, var_59);
    var_63 = wp::extract(var_xmat2, var_61, var_62);
    var_64 = wp::vec_t<3, wp::float32>(var_57, var_60, var_63);
    wp::assign_inplace(var_2, var_65, var_64);
    // normal[5] = wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2])                             <L 257>
    var_68 = wp::extract(var_xmat2, var_66, var_67);
    var_71 = wp::extract(var_xmat2, var_69, var_70);
    var_74 = wp::extract(var_xmat2, var_72, var_73);
    var_75 = wp::vec_t<3, wp::float32>(var_68, var_71, var_74);
    wp::assign_inplace(var_2, var_76, var_75);
    // for j in range(2):                                                                     <L 260>
    // for k in range(3):                                                                     <L 261>
    // for i in range(2):                                                                     <L 262>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_80 = wp::extract(var_1, var_79);
    var_82 = wp::mul(var_81, var_77);
    var_83 = wp::add(var_82, var_78);
    var_84 = wp::extract(var_2, var_83);
    var_85 = wp::dot(var_80, var_84);
    wp::assign_inplace(var_3, var_79, var_85);
    // if i == 0:                                                                             <L 264>
    var_87 = (var_79 == var_86);
    if (var_87) {
        // size = size1                                                                       <L 265>
        var_88 = wp::copy(var_size1);
    }
    if (!var_87) {
        // size = size2                                                                       <L 267>
        var_89 = wp::copy(var_size2);
    }
    var_90 = wp::where(var_87, var_88, var_89);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_92 = wp::extract(var_90, var_91);
    var_94 = wp::mul(var_93, var_79);
    var_96 = wp::add(var_94, var_95);
    var_97 = wp::extract(var_2, var_96);
    var_99 = wp::mul(var_98, var_77);
    var_100 = wp::add(var_99, var_78);
    var_101 = wp::extract(var_2, var_100);
    var_102 = wp::dot(var_97, var_101);
    var_103 = wp::mul(var_92, var_102);
    var_104 = wp::abs(var_103);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_106 = wp::extract(var_90, var_105);
    var_108 = wp::mul(var_107, var_79);
    var_110 = wp::add(var_108, var_109);
    var_111 = wp::extract(var_2, var_110);
    var_113 = wp::mul(var_112, var_77);
    var_114 = wp::add(var_113, var_78);
    var_115 = wp::extract(var_2, var_114);
    var_116 = wp::dot(var_111, var_115);
    var_117 = wp::mul(var_106, var_116);
    var_118 = wp::abs(var_117);
    var_119 = wp::add(var_104, var_118);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_121 = wp::extract(var_90, var_120);
    var_123 = wp::mul(var_122, var_79);
    var_125 = wp::add(var_123, var_124);
    var_126 = wp::extract(var_2, var_125);
    var_128 = wp::mul(var_127, var_77);
    var_129 = wp::add(var_128, var_78);
    var_130 = wp::extract(var_2, var_129);
    var_131 = wp::dot(var_126, var_130);
    var_132 = wp::mul(var_121, var_131);
    var_133 = wp::abs(var_132);
    var_134 = wp::add(var_119, var_133);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_79, var_134);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_136 = wp::extract(var_1, var_135);
    var_138 = wp::mul(var_137, var_77);
    var_139 = wp::add(var_138, var_78);
    var_140 = wp::extract(var_2, var_139);
    var_141 = wp::dot(var_136, var_140);
    wp::assign_inplace(var_3, var_135, var_141);
    // if i == 0:                                                                             <L 264>
    var_143 = (var_135 == var_142);
    if (var_143) {
        // size = size1                                                                       <L 265>
        var_144 = wp::copy(var_size1);
    }
    if (!var_143) {
        // size = size2                                                                       <L 267>
        var_145 = wp::copy(var_size2);
    }
    var_146 = wp::where(var_143, var_144, var_145);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_148 = wp::extract(var_146, var_147);
    var_150 = wp::mul(var_149, var_135);
    var_152 = wp::add(var_150, var_151);
    var_153 = wp::extract(var_2, var_152);
    var_155 = wp::mul(var_154, var_77);
    var_156 = wp::add(var_155, var_78);
    var_157 = wp::extract(var_2, var_156);
    var_158 = wp::dot(var_153, var_157);
    var_159 = wp::mul(var_148, var_158);
    var_160 = wp::abs(var_159);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_162 = wp::extract(var_146, var_161);
    var_164 = wp::mul(var_163, var_135);
    var_166 = wp::add(var_164, var_165);
    var_167 = wp::extract(var_2, var_166);
    var_169 = wp::mul(var_168, var_77);
    var_170 = wp::add(var_169, var_78);
    var_171 = wp::extract(var_2, var_170);
    var_172 = wp::dot(var_167, var_171);
    var_173 = wp::mul(var_162, var_172);
    var_174 = wp::abs(var_173);
    var_175 = wp::add(var_160, var_174);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_177 = wp::extract(var_146, var_176);
    var_179 = wp::mul(var_178, var_135);
    var_181 = wp::add(var_179, var_180);
    var_182 = wp::extract(var_2, var_181);
    var_184 = wp::mul(var_183, var_77);
    var_185 = wp::add(var_184, var_78);
    var_186 = wp::extract(var_2, var_185);
    var_187 = wp::dot(var_182, var_186);
    var_188 = wp::mul(var_177, var_187);
    var_189 = wp::abs(var_188);
    var_190 = wp::add(var_175, var_189);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_135, var_190);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 276>
    var_192 = wp::extract(var_4, var_191);
    var_194 = wp::extract(var_4, var_193);
    var_195 = wp::add(var_192, var_194);
    var_196 = wp::add(var_195, var_0);
    var_198 = wp::extract(var_3, var_197);
    var_200 = wp::extract(var_3, var_199);
    var_201 = wp::sub(var_198, var_200);
    var_202 = wp::abs(var_201);
    var_203 = (var_196 < var_202);
    if (var_203) {
        // return False                                                                       <L 277>
        return var_204;
    }
    // for i in range(2):                                                                     <L 262>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_207 = wp::extract(var_1, var_206);
    var_209 = wp::mul(var_208, var_77);
    var_210 = wp::add(var_209, var_205);
    var_211 = wp::extract(var_2, var_210);
    var_212 = wp::dot(var_207, var_211);
    wp::assign_inplace(var_3, var_206, var_212);
    // if i == 0:                                                                             <L 264>
    var_214 = (var_206 == var_213);
    if (var_214) {
        // size = size1                                                                       <L 265>
        var_215 = wp::copy(var_size1);
    }
    if (!var_214) {
        // size = size2                                                                       <L 267>
        var_216 = wp::copy(var_size2);
    }
    var_217 = wp::where(var_214, var_215, var_216);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_219 = wp::extract(var_217, var_218);
    var_221 = wp::mul(var_220, var_206);
    var_223 = wp::add(var_221, var_222);
    var_224 = wp::extract(var_2, var_223);
    var_226 = wp::mul(var_225, var_77);
    var_227 = wp::add(var_226, var_205);
    var_228 = wp::extract(var_2, var_227);
    var_229 = wp::dot(var_224, var_228);
    var_230 = wp::mul(var_219, var_229);
    var_231 = wp::abs(var_230);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_233 = wp::extract(var_217, var_232);
    var_235 = wp::mul(var_234, var_206);
    var_237 = wp::add(var_235, var_236);
    var_238 = wp::extract(var_2, var_237);
    var_240 = wp::mul(var_239, var_77);
    var_241 = wp::add(var_240, var_205);
    var_242 = wp::extract(var_2, var_241);
    var_243 = wp::dot(var_238, var_242);
    var_244 = wp::mul(var_233, var_243);
    var_245 = wp::abs(var_244);
    var_246 = wp::add(var_231, var_245);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_248 = wp::extract(var_217, var_247);
    var_250 = wp::mul(var_249, var_206);
    var_252 = wp::add(var_250, var_251);
    var_253 = wp::extract(var_2, var_252);
    var_255 = wp::mul(var_254, var_77);
    var_256 = wp::add(var_255, var_205);
    var_257 = wp::extract(var_2, var_256);
    var_258 = wp::dot(var_253, var_257);
    var_259 = wp::mul(var_248, var_258);
    var_260 = wp::abs(var_259);
    var_261 = wp::add(var_246, var_260);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_206, var_261);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_263 = wp::extract(var_1, var_262);
    var_265 = wp::mul(var_264, var_77);
    var_266 = wp::add(var_265, var_205);
    var_267 = wp::extract(var_2, var_266);
    var_268 = wp::dot(var_263, var_267);
    wp::assign_inplace(var_3, var_262, var_268);
    // if i == 0:                                                                             <L 264>
    var_270 = (var_262 == var_269);
    if (var_270) {
        // size = size1                                                                       <L 265>
        var_271 = wp::copy(var_size1);
    }
    if (!var_270) {
        // size = size2                                                                       <L 267>
        var_272 = wp::copy(var_size2);
    }
    var_273 = wp::where(var_270, var_271, var_272);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_275 = wp::extract(var_273, var_274);
    var_277 = wp::mul(var_276, var_262);
    var_279 = wp::add(var_277, var_278);
    var_280 = wp::extract(var_2, var_279);
    var_282 = wp::mul(var_281, var_77);
    var_283 = wp::add(var_282, var_205);
    var_284 = wp::extract(var_2, var_283);
    var_285 = wp::dot(var_280, var_284);
    var_286 = wp::mul(var_275, var_285);
    var_287 = wp::abs(var_286);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_289 = wp::extract(var_273, var_288);
    var_291 = wp::mul(var_290, var_262);
    var_293 = wp::add(var_291, var_292);
    var_294 = wp::extract(var_2, var_293);
    var_296 = wp::mul(var_295, var_77);
    var_297 = wp::add(var_296, var_205);
    var_298 = wp::extract(var_2, var_297);
    var_299 = wp::dot(var_294, var_298);
    var_300 = wp::mul(var_289, var_299);
    var_301 = wp::abs(var_300);
    var_302 = wp::add(var_287, var_301);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_304 = wp::extract(var_273, var_303);
    var_306 = wp::mul(var_305, var_262);
    var_308 = wp::add(var_306, var_307);
    var_309 = wp::extract(var_2, var_308);
    var_311 = wp::mul(var_310, var_77);
    var_312 = wp::add(var_311, var_205);
    var_313 = wp::extract(var_2, var_312);
    var_314 = wp::dot(var_309, var_313);
    var_315 = wp::mul(var_304, var_314);
    var_316 = wp::abs(var_315);
    var_317 = wp::add(var_302, var_316);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_262, var_317);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 276>
    var_319 = wp::extract(var_4, var_318);
    var_321 = wp::extract(var_4, var_320);
    var_322 = wp::add(var_319, var_321);
    var_323 = wp::add(var_322, var_0);
    var_325 = wp::extract(var_3, var_324);
    var_327 = wp::extract(var_3, var_326);
    var_328 = wp::sub(var_325, var_327);
    var_329 = wp::abs(var_328);
    var_330 = (var_323 < var_329);
    if (var_330) {
        // return False                                                                       <L 277>
        return var_331;
    }
    // for i in range(2):                                                                     <L 262>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_334 = wp::extract(var_1, var_333);
    var_336 = wp::mul(var_335, var_77);
    var_337 = wp::add(var_336, var_332);
    var_338 = wp::extract(var_2, var_337);
    var_339 = wp::dot(var_334, var_338);
    wp::assign_inplace(var_3, var_333, var_339);
    // if i == 0:                                                                             <L 264>
    var_341 = (var_333 == var_340);
    if (var_341) {
        // size = size1                                                                       <L 265>
        var_342 = wp::copy(var_size1);
    }
    if (!var_341) {
        // size = size2                                                                       <L 267>
        var_343 = wp::copy(var_size2);
    }
    var_344 = wp::where(var_341, var_342, var_343);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_346 = wp::extract(var_344, var_345);
    var_348 = wp::mul(var_347, var_333);
    var_350 = wp::add(var_348, var_349);
    var_351 = wp::extract(var_2, var_350);
    var_353 = wp::mul(var_352, var_77);
    var_354 = wp::add(var_353, var_332);
    var_355 = wp::extract(var_2, var_354);
    var_356 = wp::dot(var_351, var_355);
    var_357 = wp::mul(var_346, var_356);
    var_358 = wp::abs(var_357);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_360 = wp::extract(var_344, var_359);
    var_362 = wp::mul(var_361, var_333);
    var_364 = wp::add(var_362, var_363);
    var_365 = wp::extract(var_2, var_364);
    var_367 = wp::mul(var_366, var_77);
    var_368 = wp::add(var_367, var_332);
    var_369 = wp::extract(var_2, var_368);
    var_370 = wp::dot(var_365, var_369);
    var_371 = wp::mul(var_360, var_370);
    var_372 = wp::abs(var_371);
    var_373 = wp::add(var_358, var_372);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_375 = wp::extract(var_344, var_374);
    var_377 = wp::mul(var_376, var_333);
    var_379 = wp::add(var_377, var_378);
    var_380 = wp::extract(var_2, var_379);
    var_382 = wp::mul(var_381, var_77);
    var_383 = wp::add(var_382, var_332);
    var_384 = wp::extract(var_2, var_383);
    var_385 = wp::dot(var_380, var_384);
    var_386 = wp::mul(var_375, var_385);
    var_387 = wp::abs(var_386);
    var_388 = wp::add(var_373, var_387);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_333, var_388);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_390 = wp::extract(var_1, var_389);
    var_392 = wp::mul(var_391, var_77);
    var_393 = wp::add(var_392, var_332);
    var_394 = wp::extract(var_2, var_393);
    var_395 = wp::dot(var_390, var_394);
    wp::assign_inplace(var_3, var_389, var_395);
    // if i == 0:                                                                             <L 264>
    var_397 = (var_389 == var_396);
    if (var_397) {
        // size = size1                                                                       <L 265>
        var_398 = wp::copy(var_size1);
    }
    if (!var_397) {
        // size = size2                                                                       <L 267>
        var_399 = wp::copy(var_size2);
    }
    var_400 = wp::where(var_397, var_398, var_399);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_402 = wp::extract(var_400, var_401);
    var_404 = wp::mul(var_403, var_389);
    var_406 = wp::add(var_404, var_405);
    var_407 = wp::extract(var_2, var_406);
    var_409 = wp::mul(var_408, var_77);
    var_410 = wp::add(var_409, var_332);
    var_411 = wp::extract(var_2, var_410);
    var_412 = wp::dot(var_407, var_411);
    var_413 = wp::mul(var_402, var_412);
    var_414 = wp::abs(var_413);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_416 = wp::extract(var_400, var_415);
    var_418 = wp::mul(var_417, var_389);
    var_420 = wp::add(var_418, var_419);
    var_421 = wp::extract(var_2, var_420);
    var_423 = wp::mul(var_422, var_77);
    var_424 = wp::add(var_423, var_332);
    var_425 = wp::extract(var_2, var_424);
    var_426 = wp::dot(var_421, var_425);
    var_427 = wp::mul(var_416, var_426);
    var_428 = wp::abs(var_427);
    var_429 = wp::add(var_414, var_428);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_431 = wp::extract(var_400, var_430);
    var_433 = wp::mul(var_432, var_389);
    var_435 = wp::add(var_433, var_434);
    var_436 = wp::extract(var_2, var_435);
    var_438 = wp::mul(var_437, var_77);
    var_439 = wp::add(var_438, var_332);
    var_440 = wp::extract(var_2, var_439);
    var_441 = wp::dot(var_436, var_440);
    var_442 = wp::mul(var_431, var_441);
    var_443 = wp::abs(var_442);
    var_444 = wp::add(var_429, var_443);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_389, var_444);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 276>
    var_446 = wp::extract(var_4, var_445);
    var_448 = wp::extract(var_4, var_447);
    var_449 = wp::add(var_446, var_448);
    var_450 = wp::add(var_449, var_0);
    var_452 = wp::extract(var_3, var_451);
    var_454 = wp::extract(var_3, var_453);
    var_455 = wp::sub(var_452, var_454);
    var_456 = wp::abs(var_455);
    var_457 = (var_450 < var_456);
    if (var_457) {
        // return False                                                                       <L 277>
        return var_458;
    }
    // for k in range(3):                                                                     <L 261>
    // for i in range(2):                                                                     <L 262>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_462 = wp::extract(var_1, var_461);
    var_464 = wp::mul(var_463, var_459);
    var_465 = wp::add(var_464, var_460);
    var_466 = wp::extract(var_2, var_465);
    var_467 = wp::dot(var_462, var_466);
    wp::assign_inplace(var_3, var_461, var_467);
    // if i == 0:                                                                             <L 264>
    var_469 = (var_461 == var_468);
    if (var_469) {
        // size = size1                                                                       <L 265>
        var_470 = wp::copy(var_size1);
    }
    if (!var_469) {
        // size = size2                                                                       <L 267>
        var_471 = wp::copy(var_size2);
    }
    var_472 = wp::where(var_469, var_470, var_471);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_474 = wp::extract(var_472, var_473);
    var_476 = wp::mul(var_475, var_461);
    var_478 = wp::add(var_476, var_477);
    var_479 = wp::extract(var_2, var_478);
    var_481 = wp::mul(var_480, var_459);
    var_482 = wp::add(var_481, var_460);
    var_483 = wp::extract(var_2, var_482);
    var_484 = wp::dot(var_479, var_483);
    var_485 = wp::mul(var_474, var_484);
    var_486 = wp::abs(var_485);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_488 = wp::extract(var_472, var_487);
    var_490 = wp::mul(var_489, var_461);
    var_492 = wp::add(var_490, var_491);
    var_493 = wp::extract(var_2, var_492);
    var_495 = wp::mul(var_494, var_459);
    var_496 = wp::add(var_495, var_460);
    var_497 = wp::extract(var_2, var_496);
    var_498 = wp::dot(var_493, var_497);
    var_499 = wp::mul(var_488, var_498);
    var_500 = wp::abs(var_499);
    var_501 = wp::add(var_486, var_500);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_503 = wp::extract(var_472, var_502);
    var_505 = wp::mul(var_504, var_461);
    var_507 = wp::add(var_505, var_506);
    var_508 = wp::extract(var_2, var_507);
    var_510 = wp::mul(var_509, var_459);
    var_511 = wp::add(var_510, var_460);
    var_512 = wp::extract(var_2, var_511);
    var_513 = wp::dot(var_508, var_512);
    var_514 = wp::mul(var_503, var_513);
    var_515 = wp::abs(var_514);
    var_516 = wp::add(var_501, var_515);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_461, var_516);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_518 = wp::extract(var_1, var_517);
    var_520 = wp::mul(var_519, var_459);
    var_521 = wp::add(var_520, var_460);
    var_522 = wp::extract(var_2, var_521);
    var_523 = wp::dot(var_518, var_522);
    wp::assign_inplace(var_3, var_517, var_523);
    // if i == 0:                                                                             <L 264>
    var_525 = (var_517 == var_524);
    if (var_525) {
        // size = size1                                                                       <L 265>
        var_526 = wp::copy(var_size1);
    }
    if (!var_525) {
        // size = size2                                                                       <L 267>
        var_527 = wp::copy(var_size2);
    }
    var_528 = wp::where(var_525, var_526, var_527);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_530 = wp::extract(var_528, var_529);
    var_532 = wp::mul(var_531, var_517);
    var_534 = wp::add(var_532, var_533);
    var_535 = wp::extract(var_2, var_534);
    var_537 = wp::mul(var_536, var_459);
    var_538 = wp::add(var_537, var_460);
    var_539 = wp::extract(var_2, var_538);
    var_540 = wp::dot(var_535, var_539);
    var_541 = wp::mul(var_530, var_540);
    var_542 = wp::abs(var_541);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_544 = wp::extract(var_528, var_543);
    var_546 = wp::mul(var_545, var_517);
    var_548 = wp::add(var_546, var_547);
    var_549 = wp::extract(var_2, var_548);
    var_551 = wp::mul(var_550, var_459);
    var_552 = wp::add(var_551, var_460);
    var_553 = wp::extract(var_2, var_552);
    var_554 = wp::dot(var_549, var_553);
    var_555 = wp::mul(var_544, var_554);
    var_556 = wp::abs(var_555);
    var_557 = wp::add(var_542, var_556);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_559 = wp::extract(var_528, var_558);
    var_561 = wp::mul(var_560, var_517);
    var_563 = wp::add(var_561, var_562);
    var_564 = wp::extract(var_2, var_563);
    var_566 = wp::mul(var_565, var_459);
    var_567 = wp::add(var_566, var_460);
    var_568 = wp::extract(var_2, var_567);
    var_569 = wp::dot(var_564, var_568);
    var_570 = wp::mul(var_559, var_569);
    var_571 = wp::abs(var_570);
    var_572 = wp::add(var_557, var_571);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_517, var_572);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 276>
    var_574 = wp::extract(var_4, var_573);
    var_576 = wp::extract(var_4, var_575);
    var_577 = wp::add(var_574, var_576);
    var_578 = wp::add(var_577, var_0);
    var_580 = wp::extract(var_3, var_579);
    var_582 = wp::extract(var_3, var_581);
    var_583 = wp::sub(var_580, var_582);
    var_584 = wp::abs(var_583);
    var_585 = (var_578 < var_584);
    if (var_585) {
        // return False                                                                       <L 277>
        return var_586;
    }
    // for i in range(2):                                                                     <L 262>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_589 = wp::extract(var_1, var_588);
    var_591 = wp::mul(var_590, var_459);
    var_592 = wp::add(var_591, var_587);
    var_593 = wp::extract(var_2, var_592);
    var_594 = wp::dot(var_589, var_593);
    wp::assign_inplace(var_3, var_588, var_594);
    // if i == 0:                                                                             <L 264>
    var_596 = (var_588 == var_595);
    if (var_596) {
        // size = size1                                                                       <L 265>
        var_597 = wp::copy(var_size1);
    }
    if (!var_596) {
        // size = size2                                                                       <L 267>
        var_598 = wp::copy(var_size2);
    }
    var_599 = wp::where(var_596, var_597, var_598);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_601 = wp::extract(var_599, var_600);
    var_603 = wp::mul(var_602, var_588);
    var_605 = wp::add(var_603, var_604);
    var_606 = wp::extract(var_2, var_605);
    var_608 = wp::mul(var_607, var_459);
    var_609 = wp::add(var_608, var_587);
    var_610 = wp::extract(var_2, var_609);
    var_611 = wp::dot(var_606, var_610);
    var_612 = wp::mul(var_601, var_611);
    var_613 = wp::abs(var_612);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_615 = wp::extract(var_599, var_614);
    var_617 = wp::mul(var_616, var_588);
    var_619 = wp::add(var_617, var_618);
    var_620 = wp::extract(var_2, var_619);
    var_622 = wp::mul(var_621, var_459);
    var_623 = wp::add(var_622, var_587);
    var_624 = wp::extract(var_2, var_623);
    var_625 = wp::dot(var_620, var_624);
    var_626 = wp::mul(var_615, var_625);
    var_627 = wp::abs(var_626);
    var_628 = wp::add(var_613, var_627);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_630 = wp::extract(var_599, var_629);
    var_632 = wp::mul(var_631, var_588);
    var_634 = wp::add(var_632, var_633);
    var_635 = wp::extract(var_2, var_634);
    var_637 = wp::mul(var_636, var_459);
    var_638 = wp::add(var_637, var_587);
    var_639 = wp::extract(var_2, var_638);
    var_640 = wp::dot(var_635, var_639);
    var_641 = wp::mul(var_630, var_640);
    var_642 = wp::abs(var_641);
    var_643 = wp::add(var_628, var_642);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_588, var_643);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_645 = wp::extract(var_1, var_644);
    var_647 = wp::mul(var_646, var_459);
    var_648 = wp::add(var_647, var_587);
    var_649 = wp::extract(var_2, var_648);
    var_650 = wp::dot(var_645, var_649);
    wp::assign_inplace(var_3, var_644, var_650);
    // if i == 0:                                                                             <L 264>
    var_652 = (var_644 == var_651);
    if (var_652) {
        // size = size1                                                                       <L 265>
        var_653 = wp::copy(var_size1);
    }
    if (!var_652) {
        // size = size2                                                                       <L 267>
        var_654 = wp::copy(var_size2);
    }
    var_655 = wp::where(var_652, var_653, var_654);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_657 = wp::extract(var_655, var_656);
    var_659 = wp::mul(var_658, var_644);
    var_661 = wp::add(var_659, var_660);
    var_662 = wp::extract(var_2, var_661);
    var_664 = wp::mul(var_663, var_459);
    var_665 = wp::add(var_664, var_587);
    var_666 = wp::extract(var_2, var_665);
    var_667 = wp::dot(var_662, var_666);
    var_668 = wp::mul(var_657, var_667);
    var_669 = wp::abs(var_668);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_671 = wp::extract(var_655, var_670);
    var_673 = wp::mul(var_672, var_644);
    var_675 = wp::add(var_673, var_674);
    var_676 = wp::extract(var_2, var_675);
    var_678 = wp::mul(var_677, var_459);
    var_679 = wp::add(var_678, var_587);
    var_680 = wp::extract(var_2, var_679);
    var_681 = wp::dot(var_676, var_680);
    var_682 = wp::mul(var_671, var_681);
    var_683 = wp::abs(var_682);
    var_684 = wp::add(var_669, var_683);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_686 = wp::extract(var_655, var_685);
    var_688 = wp::mul(var_687, var_644);
    var_690 = wp::add(var_688, var_689);
    var_691 = wp::extract(var_2, var_690);
    var_693 = wp::mul(var_692, var_459);
    var_694 = wp::add(var_693, var_587);
    var_695 = wp::extract(var_2, var_694);
    var_696 = wp::dot(var_691, var_695);
    var_697 = wp::mul(var_686, var_696);
    var_698 = wp::abs(var_697);
    var_699 = wp::add(var_684, var_698);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_644, var_699);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 276>
    var_701 = wp::extract(var_4, var_700);
    var_703 = wp::extract(var_4, var_702);
    var_704 = wp::add(var_701, var_703);
    var_705 = wp::add(var_704, var_0);
    var_707 = wp::extract(var_3, var_706);
    var_709 = wp::extract(var_3, var_708);
    var_710 = wp::sub(var_707, var_709);
    var_711 = wp::abs(var_710);
    var_712 = (var_705 < var_711);
    if (var_712) {
        // return False                                                                       <L 277>
        return var_713;
    }
    // for i in range(2):                                                                     <L 262>
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_716 = wp::extract(var_1, var_715);
    var_718 = wp::mul(var_717, var_459);
    var_719 = wp::add(var_718, var_714);
    var_720 = wp::extract(var_2, var_719);
    var_721 = wp::dot(var_716, var_720);
    wp::assign_inplace(var_3, var_715, var_721);
    // if i == 0:                                                                             <L 264>
    var_723 = (var_715 == var_722);
    if (var_723) {
        // size = size1                                                                       <L 265>
        var_724 = wp::copy(var_size1);
    }
    if (!var_723) {
        // size = size2                                                                       <L 267>
        var_725 = wp::copy(var_size2);
    }
    var_726 = wp::where(var_723, var_724, var_725);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_728 = wp::extract(var_726, var_727);
    var_730 = wp::mul(var_729, var_715);
    var_732 = wp::add(var_730, var_731);
    var_733 = wp::extract(var_2, var_732);
    var_735 = wp::mul(var_734, var_459);
    var_736 = wp::add(var_735, var_714);
    var_737 = wp::extract(var_2, var_736);
    var_738 = wp::dot(var_733, var_737);
    var_739 = wp::mul(var_728, var_738);
    var_740 = wp::abs(var_739);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_742 = wp::extract(var_726, var_741);
    var_744 = wp::mul(var_743, var_715);
    var_746 = wp::add(var_744, var_745);
    var_747 = wp::extract(var_2, var_746);
    var_749 = wp::mul(var_748, var_459);
    var_750 = wp::add(var_749, var_714);
    var_751 = wp::extract(var_2, var_750);
    var_752 = wp::dot(var_747, var_751);
    var_753 = wp::mul(var_742, var_752);
    var_754 = wp::abs(var_753);
    var_755 = wp::add(var_740, var_754);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_757 = wp::extract(var_726, var_756);
    var_759 = wp::mul(var_758, var_715);
    var_761 = wp::add(var_759, var_760);
    var_762 = wp::extract(var_2, var_761);
    var_764 = wp::mul(var_763, var_459);
    var_765 = wp::add(var_764, var_714);
    var_766 = wp::extract(var_2, var_765);
    var_767 = wp::dot(var_762, var_766);
    var_768 = wp::mul(var_757, var_767);
    var_769 = wp::abs(var_768);
    var_770 = wp::add(var_755, var_769);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_715, var_770);
    // proj[i] = wp.dot(xcenter[i], normal[3 * j + k])                                        <L 263>
    var_772 = wp::extract(var_1, var_771);
    var_774 = wp::mul(var_773, var_459);
    var_775 = wp::add(var_774, var_714);
    var_776 = wp::extract(var_2, var_775);
    var_777 = wp::dot(var_772, var_776);
    wp::assign_inplace(var_3, var_771, var_777);
    // if i == 0:                                                                             <L 264>
    var_779 = (var_771 == var_778);
    if (var_779) {
        // size = size1                                                                       <L 265>
        var_780 = wp::copy(var_size1);
    }
    if (!var_779) {
        // size = size2                                                                       <L 267>
        var_781 = wp::copy(var_size2);
    }
    var_782 = wp::where(var_779, var_780, var_781);
    // radius[i] = (                                                                          <L 270>
    // wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))                         <L 271>
    var_784 = wp::extract(var_782, var_783);
    var_786 = wp::mul(var_785, var_771);
    var_788 = wp::add(var_786, var_787);
    var_789 = wp::extract(var_2, var_788);
    var_791 = wp::mul(var_790, var_459);
    var_792 = wp::add(var_791, var_714);
    var_793 = wp::extract(var_2, var_792);
    var_794 = wp::dot(var_789, var_793);
    var_795 = wp::mul(var_784, var_794);
    var_796 = wp::abs(var_795);
    // + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))                       <L 272>
    var_798 = wp::extract(var_782, var_797);
    var_800 = wp::mul(var_799, var_771);
    var_802 = wp::add(var_800, var_801);
    var_803 = wp::extract(var_2, var_802);
    var_805 = wp::mul(var_804, var_459);
    var_806 = wp::add(var_805, var_714);
    var_807 = wp::extract(var_2, var_806);
    var_808 = wp::dot(var_803, var_807);
    var_809 = wp::mul(var_798, var_808);
    var_810 = wp::abs(var_809);
    var_811 = wp::add(var_796, var_810);
    // + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))                       <L 273>
    var_813 = wp::extract(var_782, var_812);
    var_815 = wp::mul(var_814, var_771);
    var_817 = wp::add(var_815, var_816);
    var_818 = wp::extract(var_2, var_817);
    var_820 = wp::mul(var_819, var_459);
    var_821 = wp::add(var_820, var_714);
    var_822 = wp::extract(var_2, var_821);
    var_823 = wp::dot(var_818, var_822);
    var_824 = wp::mul(var_813, var_823);
    var_825 = wp::abs(var_824);
    var_826 = wp::add(var_811, var_825);
    // radius[i] = (                                                                          <L 270>
    wp::assign_inplace(var_4, var_771, var_826);
    // if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):                         <L 276>
    var_828 = wp::extract(var_4, var_827);
    var_830 = wp::extract(var_4, var_829);
    var_831 = wp::add(var_828, var_830);
    var_832 = wp::add(var_831, var_0);
    var_834 = wp::extract(var_3, var_833);
    var_836 = wp::extract(var_3, var_835);
    var_837 = wp::sub(var_834, var_836);
    var_838 = wp::abs(var_837);
    var_839 = (var_832 < var_838);
    if (var_839) {
        // return False                                                                       <L 277>
        return var_840;
    }
    // return True                                                                            <L 279>
    return var_841;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_driver.py:283
static CUDA_CALLABLE bool _broadphase_filter__locals__func_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_aabb,
    wp::array_t<wp::float32> var_geom_rbound,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_geom1,
    wp::int32 var_geom2,
    wp::int32 var_worldid)
{
    //---------
    // primal vars
    const bool var_0 = false;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 0;
    wp::vec_t<3, wp::float32>* var_3;
    const wp::int32 var_4 = 0;
    wp::vec_t<3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    wp::vec_t<3, wp::float32>* var_7;
    const wp::int32 var_8 = 1;
    wp::vec_t<3, wp::float32>* var_9;
    const bool var_10 = false;
    const wp::int32 var_11 = 0;
    wp::float32* var_12;
    wp::float32* var_13;
    const bool var_14 = false;
    const wp::int32 var_15 = 0;
    wp::float32* var_16;
    wp::float32* var_17;
    const bool var_18 = false;
    const wp::int32 var_19 = 0;
    wp::float32* var_20;
    wp::float32* var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::vec_t<3, wp::float32>* var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::mat_t<3, 3, wp::float32>* var_30;
    wp::mat_t<3, 3, wp::float32>* var_31;
    bool var_32;
    const wp::float32 var_33 = 0.0;
    bool var_34;
    wp::float32 var_35;
    const wp::float32 var_36 = 0.0;
    bool var_37;
    wp::float32 var_38;
    const wp::int32 var_39 = 1;
    bool var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::mat_t<3, 3, wp::float32> var_45;
    wp::mat_t<3, 3, wp::float32> var_46;
    const wp::int32 var_47 = 2;
    bool var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    bool var_53;
    const bool var_54 = false;
    const wp::int32 var_55 = 0;
    const wp::int32 var_56 = 8;
    bool var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::mat_t<3, 3, wp::float32> var_64;
    wp::mat_t<3, 3, wp::float32> var_65;
    bool var_66;
    const bool var_67 = false;
    const bool var_68 = true;
    //---------
    // forward
    // def func(                                                                              <L 284>
    // aabb_id = worldid % ngeom_aabb if wp.static(ngeom_aabb > 1) else 0                     <L 303>
    // center1, center2 = geom_aabb[aabb_id, geom1, 0], geom_aabb[aabb_id, geom2, 0]  # kernel_analyzer: ignore       <L 304>
    var_3 = wp::address(var_geom_aabb, var_1, var_geom1, var_2);
    var_5 = wp::address(var_geom_aabb, var_1, var_geom2, var_4);
    // size1, size2 = geom_aabb[aabb_id, geom1, 1], geom_aabb[aabb_id, geom2, 1]  # kernel_analyzer: ignore       <L 305>
    var_7 = wp::address(var_geom_aabb, var_1, var_geom1, var_6);
    var_9 = wp::address(var_geom_aabb, var_1, var_geom2, var_8);
    // rbound_id = worldid % ngeom_rbound if wp.static(ngeom_rbound > 1) else 0               <L 307>
    // rbound1, rbound2 = geom_rbound[rbound_id, geom1], geom_rbound[rbound_id, geom2]  # kernel_analyzer: ignore       <L 308>
    var_12 = wp::address(var_geom_rbound, var_11, var_geom1);
    var_13 = wp::address(var_geom_rbound, var_11, var_geom2);
    // margin_id = worldid % ngeom_margin if wp.static(ngeom_margin > 1) else 0               <L 309>
    // margin1, margin2 = geom_margin[margin_id, geom1], geom_margin[margin_id, geom2]  # kernel_analyzer: ignore       <L 310>
    var_16 = wp::address(var_geom_margin, var_15, var_geom1);
    var_17 = wp::address(var_geom_margin, var_15, var_geom2);
    // gap_id = worldid % ngeom_gap if wp.static(ngeom_gap > 1) else 0                        <L 311>
    // gap1, gap2 = geom_gap[gap_id, geom1], geom_gap[gap_id, geom2]  # kernel_analyzer: ignore       <L 312>
    var_20 = wp::address(var_geom_gap, var_19, var_geom1);
    var_21 = wp::address(var_geom_gap, var_19, var_geom2);
    // effective_margin1 = margin1 + gap1                                                     <L 313>
    var_23 = wp::load(var_16);
    var_24 = wp::load(var_20);
    var_22 = wp::add(var_23, var_24);
    // effective_margin2 = margin2 + gap2                                                     <L 314>
    var_26 = wp::load(var_17);
    var_27 = wp::load(var_21);
    var_25 = wp::add(var_26, var_27);
    // xpos1, xpos2 = geom_xpos_in[worldid, geom1], geom_xpos_in[worldid, geom2]              <L 315>
    var_28 = wp::address(var_geom_xpos_in, var_worldid, var_geom1);
    var_29 = wp::address(var_geom_xpos_in, var_worldid, var_geom2);
    // xmat1, xmat2 = geom_xmat_in[worldid, geom1], geom_xmat_in[worldid, geom2]              <L 316>
    var_30 = wp::address(var_geom_xmat_in, var_worldid, var_geom1);
    var_31 = wp::address(var_geom_xmat_in, var_worldid, var_geom2);
    // if rbound1 == 0.0 or rbound2 == 0.0:                                                   <L 318>
    var_35 = wp::load(var_12);
    var_34 = (var_35 == var_33);
    var_32 = var_34;
    if (!var_32) {
        var_38 = wp::load(var_13);
        var_37 = (var_38 == var_36);
        var_32 = var_32 || var_37;
    }
    if (var_32) {
        // if wp.static(opt_broadphase_filter & BroadphaseFilter.PLANE):                      <L 319>
        // return _plane_filter(rbound1, rbound2, effective_margin1, effective_margin2, xpos1, xpos2, xmat1, xmat2)       <L 320>
        var_41 = wp::load(var_12);
        var_42 = wp::load(var_13);
        var_43 = wp::load(var_28);
        var_44 = wp::load(var_29);
        var_45 = wp::load(var_30);
        var_46 = wp::load(var_31);
        var_40 = _plane_filter_0(var_41, var_42, var_22, var_25, var_43, var_44, var_45, var_46);
        return var_40;
    }
    if (!var_32) {
        // if wp.static(opt_broadphase_filter & BroadphaseFilter.SPHERE):                     <L 322>
        // if not _sphere_filter(rbound1, rbound2, effective_margin1, effective_margin2, xpos1, xpos2):       <L 323>
        var_49 = wp::load(var_12);
        var_50 = wp::load(var_13);
        var_51 = wp::load(var_28);
        var_52 = wp::load(var_29);
        var_48 = _sphere_filter_0(var_49, var_50, var_22, var_25, var_51, var_52);
        var_53 = wp::unot(var_48);
        if (var_53) {
            // return False                                                                   <L 324>
            return var_54;
        }
        // if wp.static(opt_broadphase_filter & BroadphaseFilter.AABB):                       <L 325>
        // if wp.static(opt_broadphase_filter & BroadphaseFilter.OBB):                        <L 328>
        // if not _obb_filter(center1, center2, size1, size2, effective_margin1, effective_margin2, xpos1, xpos2, xmat1, xmat2):       <L 329>
        var_58 = wp::load(var_3);
        var_59 = wp::load(var_5);
        var_60 = wp::load(var_7);
        var_61 = wp::load(var_9);
        var_62 = wp::load(var_28);
        var_63 = wp::load(var_29);
        var_64 = wp::load(var_30);
        var_65 = wp::load(var_31);
        var_57 = _obb_filter_0(var_58, var_59, var_60, var_61, var_22, var_25, var_62, var_63, var_64, var_65);
        var_66 = wp::unot(var_57);
        if (var_66) {
            // return False                                                                   <L 330>
            return var_67;
        }
    }
    // return True                                                                            <L 332>
    return var_68;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_driver.py:350
static CUDA_CALLABLE void _add_geom_pair_0(
    wp::int32 var_naconmax_in,
    wp::int32 var_geom1,
    wp::int32 var_geom2,
    wp::int32 var_type1,
    wp::int32 var_type2,
    wp::int32 var_worldid,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::int32> var_ncollision_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_out,
    wp::array_t<wp::int32> var_collision_worldid_out)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    const wp::int32 var_1 = 1;
    wp::int32 var_2;
    bool var_3;
    bool var_4;
    wp::vec_t<2, wp::int32> var_5;
    wp::vec_t<2, wp::int32> var_6;
    wp::vec_t<2, wp::int32> var_7;
    //---------
    // forward
    // def _add_geom_pair(                                                                    <L 351>
    // cid = wp.atomic_add(ncollision_out, 0, 1)                                              <L 368>
    var_2 = wp::atomic_add(var_ncollision_out, var_0, var_1);
    // if cid >= naconmax_in:                                                                 <L 370>
    var_3 = (var_2 >= var_naconmax_in);
    if (var_3) {
        // return                                                                             <L 371>
        return;
    }
    // if type1 > type2:                                                                      <L 373>
    var_4 = (var_type1 > var_type2);
    if (var_4) {
        // pair = wp.vec2i(geom2, geom1)                                                      <L 374>
        var_5 = wp::vec_t<2, wp::int32>(var_geom2, var_geom1);
    }
    if (!var_4) {
        // pair = wp.vec2i(geom1, geom2)                                                      <L 376>
        var_6 = wp::vec_t<2, wp::int32>(var_geom1, var_geom2);
    }
    var_7 = wp::where(var_4, var_5, var_6);
    // collision_pair_out[cid] = pair                                                         <L 378>
    wp::array_store(var_collision_pair_out, var_2, var_7);
    // collision_pairid_out[cid] = pairid                                                     <L 379>
    wp::array_store(var_collision_pairid_out, var_2, var_pairid);
    // collision_worldid_out[cid] = worldid                                                   <L 380>
    wp::array_store(var_collision_worldid_out, var_2, var_worldid);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_driver.py:98
static CUDA_CALLABLE void adj__plane_filter_0(
    wp::float32 var_size1,
    wp::float32 var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2,
    wp::mat_t<3, 3, wp::float32> var_xmat1,
    wp::mat_t<3, 3, wp::float32> var_xmat2,
    wp::float32 & adj_size1,
    wp::float32 & adj_size2,
    wp::float32 & adj_margin1,
    wp::float32 & adj_margin2,
    wp::vec_t<3, wp::float32> & adj_xpos1,
    wp::vec_t<3, wp::float32> & adj_xpos2,
    wp::mat_t<3, 3, wp::float32> & adj_xmat1,
    wp::mat_t<3, 3, wp::float32> & adj_xmat2,
    bool & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_driver.py:114
static CUDA_CALLABLE void adj__sphere_filter_0(
    wp::float32 var_size1,
    wp::float32 var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2,
    wp::float32 & adj_size1,
    wp::float32 & adj_size2,
    wp::float32 & adj_margin1,
    wp::float32 & adj_margin2,
    wp::vec_t<3, wp::float32> & adj_xpos1,
    wp::vec_t<3, wp::float32> & adj_xpos2,
    bool & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_driver.py:225
static CUDA_CALLABLE void adj__obb_filter_0(
    wp::vec_t<3, wp::float32> var_center1,
    wp::vec_t<3, wp::float32> var_center2,
    wp::vec_t<3, wp::float32> var_size1,
    wp::vec_t<3, wp::float32> var_size2,
    wp::float32 var_margin1,
    wp::float32 var_margin2,
    wp::vec_t<3, wp::float32> var_xpos1,
    wp::vec_t<3, wp::float32> var_xpos2,
    wp::mat_t<3, 3, wp::float32> var_xmat1,
    wp::mat_t<3, 3, wp::float32> var_xmat2,
    wp::vec_t<3, wp::float32> & adj_center1,
    wp::vec_t<3, wp::float32> & adj_center2,
    wp::vec_t<3, wp::float32> & adj_size1,
    wp::vec_t<3, wp::float32> & adj_size2,
    wp::float32 & adj_margin1,
    wp::float32 & adj_margin2,
    wp::vec_t<3, wp::float32> & adj_xpos1,
    wp::vec_t<3, wp::float32> & adj_xpos2,
    wp::mat_t<3, 3, wp::float32> & adj_xmat1,
    wp::mat_t<3, 3, wp::float32> & adj_xmat2,
    bool & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_driver.py:283
static CUDA_CALLABLE void adj__broadphase_filter__locals__func_0(
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_aabb,
    wp::array_t<wp::float32> var_geom_rbound,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_geom1,
    wp::int32 var_geom2,
    wp::int32 var_worldid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_aabb,
    wp::array_t<wp::float32> & adj_geom_rbound,
    wp::array_t<wp::float32> & adj_geom_margin,
    wp::array_t<wp::float32> & adj_geom_gap,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::int32 & adj_geom1,
    wp::int32 & adj_geom2,
    wp::int32 & adj_worldid,
    bool & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_driver.py:350
static CUDA_CALLABLE void adj__add_geom_pair_0(
    wp::int32 var_naconmax_in,
    wp::int32 var_geom1,
    wp::int32 var_geom2,
    wp::int32 var_type1,
    wp::int32 var_type2,
    wp::int32 var_worldid,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::int32> var_ncollision_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_out,
    wp::array_t<wp::int32> var_collision_worldid_out,
    wp::int32 & adj_naconmax_in,
    wp::int32 & adj_geom1,
    wp::int32 & adj_geom2,
    wp::int32 & adj_type1,
    wp::int32 & adj_type2,
    wp::int32 & adj_worldid,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::int32> & adj_ncollision_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_collision_pair_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_collision_pairid_out,
    wp::array_t<wp::int32> & adj_collision_worldid_out)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _nxn_broadphase__locals__kernel_0059d455_cuda_kernel_forward(
    wp::launch_bounds_t<2> dim,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_bodyid,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_aabb,
    wp::array_t<wp::float32> var_geom_rbound,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_geom_pair,
    wp::array_t<wp::vec_t<2, wp::int32>> var_nxn_pairid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::array_t<wp::int32> var_body_awake_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_body_awake_prev_in,
    wp::array_t<wp::int32> var_ncollision_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_out,
    wp::array_t<wp::int32> var_collision_worldid_out)
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
        wp::vec_t<2, wp::int32>* var_2;
        wp::vec_t<2, wp::int32> var_3;
        wp::vec_t<2, wp::int32> var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const bool var_15 = false;
        const bool var_16 = false;
        wp::vec_t<2, wp::int32>* var_17;
        wp::vec_t<2, wp::int32> var_18;
        wp::vec_t<2, wp::int32> var_19;
        bool var_20;
        bool var_21;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        //---------
        // forward
        // def kernel(                                                                            <L 726>
        // worldid, elementid = wp.tid()                                                          <L 751>
        builtin_tid2d(var_0, var_1);
        // geom = nxn_geom_pair[elementid]                                                        <L 753>
        var_2 = wp::address(var_nxn_geom_pair, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // geom1 = geom[0]                                                                        <L 754>
        var_6 = wp::extract(var_3, var_5);
        // geom2 = geom[1]                                                                        <L 755>
        var_8 = wp::extract(var_3, var_7);
        // type1 = geom_type[geom1]                                                               <L 756>
        var_9 = wp::address(var_geom_type, var_6);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // type2 = geom_type[geom2]                                                               <L 757>
        var_12 = wp::address(var_geom_type, var_8);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if wp.static(ngeom_dataid > 1):                                                        <L 759>
        // if wp.static(enable_sleep):                                                            <L 764>
        // pairid = nxn_pairid[elementid]                                                         <L 789>
        var_17 = wp::address(var_nxn_pairid, var_1);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // if (                                                                                   <L 790>
        // wp.static(_broadphase_filter(opt_broadphase_filter, ngeom_aabb, ngeom_rbound, ngeom_margin, ngeom_gap))(       <L 791>
        // geom_aabb, geom_rbound, geom_margin, geom_gap, geom_xpos_in, geom_xmat_in, geom1, geom2, worldid       <L 792>
        var_21 = _broadphase_filter__locals__func_0(var_geom_aabb, var_geom_rbound, var_geom_margin, var_geom_gap, var_geom_xpos_in, var_geom_xmat_in, var_6, var_8, var_0);
        var_20 = var_21;
        if (!var_20) {
            // or pairid[1] >= 0                                                                  <L 794>
            var_23 = wp::extract(var_18, var_22);
            var_25 = (var_23 >= var_24);
            var_20 = var_20 || var_25;
        }
        if (var_20) {
            // _add_geom_pair(                                                                    <L 796>
            // naconmax_in,                                                                       <L 797>
            // geom1,                                                                             <L 798>
            // geom2,                                                                             <L 799>
            // type1,                                                                             <L 800>
            // type2,                                                                             <L 801>
            // worldid,                                                                           <L 802>
            // pairid,                                                                            <L 803>
            // ncollision_out,                                                                    <L 804>
            // collision_pair_out,                                                                <L 805>
            // collision_pairid_out,                                                              <L 806>
            // collision_worldid_out,                                                             <L 807>
            _add_geom_pair_0(var_naconmax_in, var_6, var_8, var_10, var_13, var_0, var_18, var_ncollision_out, var_collision_pair_out, var_collision_pairid_out, var_collision_worldid_out);
        }
    }
}

