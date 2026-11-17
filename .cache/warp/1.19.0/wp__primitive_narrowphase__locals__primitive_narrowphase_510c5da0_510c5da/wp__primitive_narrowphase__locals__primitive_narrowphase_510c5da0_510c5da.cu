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


struct Geom_4022bfa7
{
    wp::vec_t<3, wp::float32> pos;
    wp::mat_t<3, 3, wp::float32> rot;
    wp::vec_t<3, wp::float32> normal;
    wp::vec_t<3, wp::float32> size;
    wp::float32 margin;
    wp::mat_t<6, 3, wp::float32> polyvert;
    wp::int32 vertadr;
    wp::int32 vertnum;
    wp::array_t<wp::vec_t<3, wp::float32>> vert;
    wp::int32 graphadr;
    wp::array_t<wp::int32> graph;
    wp::int32 mesh_polynum;
    wp::int32 mesh_polyadr;
    wp::array_t<wp::vec_t<3, wp::float32>> mesh_polynormal;
    wp::array_t<wp::int32> mesh_polyvertadr;
    wp::array_t<wp::int32> mesh_polyvertnum;
    wp::array_t<wp::int32> mesh_polyvert;
    wp::array_t<wp::int32> mesh_polymapadr;
    wp::array_t<wp::int32> mesh_polymapnum;
    wp::array_t<wp::int32> mesh_polymap;
    wp::int32 index;


    Geom_4022bfa7() = default;
    CUDA_CALLABLE Geom_4022bfa7(wp::vec_t<3, wp::float32> const& pos,
    wp::mat_t<3, 3, wp::float32> const& rot = {},
    wp::vec_t<3, wp::float32> const& normal = {},
    wp::vec_t<3, wp::float32> const& size = {},
    wp::float32 const& margin = {},
    wp::mat_t<6, 3, wp::float32> const& polyvert = {},
    wp::int32 const& vertadr = {},
    wp::int32 const& vertnum = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& vert = {},
    wp::int32 const& graphadr = {},
    wp::array_t<wp::int32> const& graph = {},
    wp::int32 const& mesh_polynum = {},
    wp::int32 const& mesh_polyadr = {},
    wp::array_t<wp::vec_t<3, wp::float32>> const& mesh_polynormal = {},
    wp::array_t<wp::int32> const& mesh_polyvertadr = {},
    wp::array_t<wp::int32> const& mesh_polyvertnum = {},
    wp::array_t<wp::int32> const& mesh_polyvert = {},
    wp::array_t<wp::int32> const& mesh_polymapadr = {},
    wp::array_t<wp::int32> const& mesh_polymapnum = {},
    wp::array_t<wp::int32> const& mesh_polymap = {},
    wp::int32 const& index = {})
        : pos{pos}
        , rot{rot}
        , normal{normal}
        , size{size}
        , margin{margin}
        , polyvert{polyvert}
        , vertadr{vertadr}
        , vertnum{vertnum}
        , vert{vert}
        , graphadr{graphadr}
        , graph{graph}
        , mesh_polynum{mesh_polynum}
        , mesh_polyadr{mesh_polyadr}
        , mesh_polynormal{mesh_polynormal}
        , mesh_polyvertadr{mesh_polyvertadr}
        , mesh_polyvertnum{mesh_polyvertnum}
        , mesh_polyvert{mesh_polyvert}
        , mesh_polymapadr{mesh_polymapadr}
        , mesh_polymapnum{mesh_polymapnum}
        , mesh_polymap{mesh_polymap}
        , index{index}

    {
    }

    CUDA_CALLABLE Geom_4022bfa7& operator += (const Geom_4022bfa7& rhs)
    {    pos += rhs.pos;
    rot += rhs.rot;
    normal += rhs.normal;
    size += rhs.size;
    margin += rhs.margin;
    polyvert += rhs.polyvert;
    vertadr += rhs.vertadr;
    vertnum += rhs.vertnum;
    graphadr += rhs.graphadr;
    mesh_polynum += rhs.mesh_polynum;
    mesh_polyadr += rhs.mesh_polyadr;
    index += rhs.index;

        return *this;}

};

static CUDA_CALLABLE void adj_Geom_4022bfa7(wp::vec_t<3, wp::float32> const&,
    wp::mat_t<3, 3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::float32 const&,
    wp::mat_t<6, 3, wp::float32> const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::int32 const&,
    wp::array_t<wp::int32> const&,
    wp::int32 const&,
    wp::int32 const&,
    wp::array_t<wp::vec_t<3, wp::float32>> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::array_t<wp::int32> const&,
    wp::int32 const&,
    wp::vec_t<3, wp::float32> & adj_pos,
    wp::mat_t<3, 3, wp::float32> & adj_rot,
    wp::vec_t<3, wp::float32> & adj_normal,
    wp::vec_t<3, wp::float32> & adj_size,
    wp::float32 & adj_margin,
    wp::mat_t<6, 3, wp::float32> & adj_polyvert,
    wp::int32 & adj_vertadr,
    wp::int32 & adj_vertnum,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_vert,
    wp::int32 & adj_graphadr,
    wp::array_t<wp::int32> & adj_graph,
    wp::int32 & adj_mesh_polynum,
    wp::int32 & adj_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_polynormal,
    wp::array_t<wp::int32> & adj_mesh_polyvertadr,
    wp::array_t<wp::int32> & adj_mesh_polyvertnum,
    wp::array_t<wp::int32> & adj_mesh_polyvert,
    wp::array_t<wp::int32> & adj_mesh_polymapadr,
    wp::array_t<wp::int32> & adj_mesh_polymapnum,
    wp::array_t<wp::int32> & adj_mesh_polymap,
    wp::int32 & adj_index,
    Geom_4022bfa7 & adj_ret)
{
    adj_pos += adj_ret.pos;
    adj_rot += adj_ret.rot;
    adj_normal += adj_ret.normal;
    adj_size += adj_ret.size;
    adj_margin += adj_ret.margin;
    adj_polyvert += adj_ret.polyvert;
    adj_vertadr += adj_ret.vertadr;
    adj_vertnum += adj_ret.vertnum;
    adj_vert = adj_ret.vert;
    adj_graphadr += adj_ret.graphadr;
    adj_graph = adj_ret.graph;
    adj_mesh_polynum += adj_ret.mesh_polynum;
    adj_mesh_polyadr += adj_ret.mesh_polyadr;
    adj_mesh_polynormal = adj_ret.mesh_polynormal;
    adj_mesh_polyvertadr = adj_ret.mesh_polyvertadr;
    adj_mesh_polyvertnum = adj_ret.mesh_polyvertnum;
    adj_mesh_polyvert = adj_ret.mesh_polyvert;
    adj_mesh_polymapadr = adj_ret.mesh_polymapadr;
    adj_mesh_polymapnum = adj_ret.mesh_polymapnum;
    adj_mesh_polymap = adj_ret.mesh_polymap;
    adj_index += adj_ret.index;
}

// Required when compiling adjoints.
CUDA_CALLABLE Geom_4022bfa7 add(const Geom_4022bfa7& a, const Geom_4022bfa7& b)
{
    return Geom_4022bfa7();
}

CUDA_CALLABLE void adj_atomic_add(Geom_4022bfa7* p, Geom_4022bfa7 t)
{
    wp::adj_atomic_add(&p->pos, t.pos);
    wp::adj_atomic_add(&p->rot, t.rot);
    wp::adj_atomic_add(&p->normal, t.normal);
    wp::adj_atomic_add(&p->size, t.size);
    wp::adj_atomic_add(&p->margin, t.margin);
    wp::adj_atomic_add(&p->polyvert, t.polyvert);
    wp::adj_atomic_add(&p->vertadr, t.vertadr);
    wp::adj_atomic_add(&p->vertnum, t.vertnum);
    wp::adj_atomic_add(&p->vert, t.vert);
    wp::adj_atomic_add(&p->graphadr, t.graphadr);
    wp::adj_atomic_add(&p->graph, t.graph);
    wp::adj_atomic_add(&p->mesh_polynum, t.mesh_polynum);
    wp::adj_atomic_add(&p->mesh_polyadr, t.mesh_polyadr);
    wp::adj_atomic_add(&p->mesh_polynormal, t.mesh_polynormal);
    wp::adj_atomic_add(&p->mesh_polyvertadr, t.mesh_polyvertadr);
    wp::adj_atomic_add(&p->mesh_polyvertnum, t.mesh_polyvertnum);
    wp::adj_atomic_add(&p->mesh_polyvert, t.mesh_polyvert);
    wp::adj_atomic_add(&p->mesh_polymapadr, t.mesh_polymapadr);
    wp::adj_atomic_add(&p->mesh_polymapnum, t.mesh_polymapnum);
    wp::adj_atomic_add(&p->mesh_polymap, t.mesh_polymap);
    wp::adj_atomic_add(&p->index, t.index);
}




// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:294
static CUDA_CALLABLE void contact_margin_gap_0(
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_pairid,
    wp::int32 var_worldid,
    wp::float32 & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = -1;
    bool var_1;
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
    wp::float32* var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::int32 var_18 = 0;
    wp::int32 var_19;
    const wp::int32 var_20 = 1;
    wp::int32 var_21;
    wp::shape_t* var_22;
    const wp::int32 var_23 = 0;
    wp::int32 var_24;
    wp::shape_t var_25;
    wp::int32 var_26;
    wp::shape_t* var_27;
    const wp::int32 var_28 = 0;
    wp::int32 var_29;
    wp::shape_t var_30;
    wp::int32 var_31;
    wp::float32* var_32;
    wp::float32* var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32* var_37;
    wp::float32* var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    //---------
    // forward
    // def contact_margin_gap(                                                                <L 295>
    // if pairid > -1:                                                                        <L 306>
    var_1 = (var_pairid > var_0);
    if (var_1) {
        // margin = pair_margin[worldid % pair_margin.shape[0], pairid]                       <L 307>
        var_2 = &(var_pair_margin.shape);
        var_5 = wp::load(var_2);
        var_4 = wp::extract(var_5, var_3);
        var_6 = wp::mod(var_worldid, var_4);
        var_7 = wp::address(var_pair_margin, var_6, var_pairid);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // gap = pair_gap[worldid % pair_gap.shape[0], pairid]                                <L 308>
        var_10 = &(var_pair_gap.shape);
        var_13 = wp::load(var_10);
        var_12 = wp::extract(var_13, var_11);
        var_14 = wp::mod(var_worldid, var_12);
        var_15 = wp::address(var_pair_gap, var_14, var_pairid);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
    }
    if (!var_1) {
        // g1 = geoms[0]                                                                      <L 310>
        var_19 = wp::extract(var_geoms, var_18);
        // g2 = geoms[1]                                                                      <L 311>
        var_21 = wp::extract(var_geoms, var_20);
        // margin_id = worldid % geom_margin.shape[0]                                         <L 312>
        var_22 = &(var_geom_margin.shape);
        var_25 = wp::load(var_22);
        var_24 = wp::extract(var_25, var_23);
        var_26 = wp::mod(var_worldid, var_24);
        // gap_id = worldid % geom_gap.shape[0]                                               <L 313>
        var_27 = &(var_geom_gap.shape);
        var_30 = wp::load(var_27);
        var_29 = wp::extract(var_30, var_28);
        var_31 = wp::mod(var_worldid, var_29);
        // margin = geom_margin[margin_id, g1] + geom_margin[margin_id, g2]                   <L 314>
        var_32 = wp::address(var_geom_margin, var_26, var_19);
        var_33 = wp::address(var_geom_margin, var_26, var_21);
        var_35 = wp::load(var_32);
        var_36 = wp::load(var_33);
        var_34 = wp::add(var_35, var_36);
        // gap = geom_gap[gap_id, g1] + geom_gap[gap_id, g2]                                  <L 315>
        var_37 = wp::address(var_geom_gap, var_31, var_19);
        var_38 = wp::address(var_geom_gap, var_31, var_21);
        var_40 = wp::load(var_37);
        var_41 = wp::load(var_38);
        var_39 = wp::add(var_40, var_41);
    }
    var_42 = wp::where(var_1, var_8, var_34);
    var_43 = wp::where(var_1, var_16, var_39);
    // return margin, gap                                                                     <L 317>
    ret_0 = var_42;
    ret_1 = var_43;
    return;
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:320
static CUDA_CALLABLE void contact_material_params_0(
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_adhesion,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_adhesion,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_pairid,
    wp::int32 var_worldid,
    wp::int32 & ret_0,
    wp::vec_t<5, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2,
    wp::vec_t<2, wp::float32> & ret_3,
    wp::vec_t<5, wp::float32> & ret_4,
    wp::float32 & ret_5)
{
    //---------
    // primal vars
    const wp::int32 var_0 = -1;
    bool var_1;
    wp::int32* var_2;
    wp::int32 var_3;
    wp::int32 var_4;
    wp::shape_t* var_5;
    const wp::int32 var_6 = 0;
    wp::int32 var_7;
    wp::shape_t var_8;
    wp::int32 var_9;
    wp::vec_t<5, wp::float32>* var_10;
    wp::vec_t<5, wp::float32> var_11;
    wp::vec_t<5, wp::float32> var_12;
    wp::shape_t* var_13;
    const wp::int32 var_14 = 0;
    wp::int32 var_15;
    wp::shape_t var_16;
    wp::int32 var_17;
    wp::vec_t<2, wp::float32>* var_18;
    wp::vec_t<2, wp::float32> var_19;
    wp::vec_t<2, wp::float32> var_20;
    wp::shape_t* var_21;
    const wp::int32 var_22 = 0;
    wp::int32 var_23;
    wp::shape_t var_24;
    wp::int32 var_25;
    wp::vec_t<2, wp::float32>* var_26;
    wp::vec_t<2, wp::float32> var_27;
    wp::vec_t<2, wp::float32> var_28;
    wp::shape_t* var_29;
    const wp::int32 var_30 = 0;
    wp::int32 var_31;
    wp::shape_t var_32;
    wp::int32 var_33;
    wp::vec_t<5, wp::float32>* var_34;
    wp::vec_t<5, wp::float32> var_35;
    wp::vec_t<5, wp::float32> var_36;
    wp::shape_t* var_37;
    const wp::int32 var_38 = 0;
    wp::int32 var_39;
    wp::shape_t var_40;
    wp::int32 var_41;
    wp::float32* var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 0;
    wp::int32 var_46;
    const wp::int32 var_47 = 1;
    wp::int32 var_48;
    wp::shape_t* var_49;
    const wp::int32 var_50 = 0;
    wp::int32 var_51;
    wp::shape_t var_52;
    wp::int32 var_53;
    wp::shape_t* var_54;
    const wp::int32 var_55 = 0;
    wp::int32 var_56;
    wp::shape_t var_57;
    wp::int32 var_58;
    wp::shape_t* var_59;
    const wp::int32 var_60 = 0;
    wp::int32 var_61;
    wp::shape_t var_62;
    wp::int32 var_63;
    wp::shape_t* var_64;
    const wp::int32 var_65 = 0;
    wp::int32 var_66;
    wp::shape_t var_67;
    wp::int32 var_68;
    wp::shape_t* var_69;
    const wp::int32 var_70 = 0;
    wp::int32 var_71;
    wp::shape_t var_72;
    wp::int32 var_73;
    wp::float32* var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32* var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    wp::int32* var_80;
    wp::int32 var_81;
    wp::int32 var_82;
    wp::int32* var_83;
    wp::int32 var_84;
    wp::int32 var_85;
    wp::int32* var_86;
    wp::int32 var_87;
    wp::int32 var_88;
    wp::int32* var_89;
    wp::int32 var_90;
    wp::int32 var_91;
    wp::float32* var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32* var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    bool var_98;
    const wp::float32 var_99 = 1.0;
    wp::int32 var_100;
    wp::vec_t<3, wp::float32>* var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::float32 var_104;
    bool var_105;
    const wp::float32 var_106 = 0.0;
    wp::int32 var_107;
    wp::vec_t<3, wp::float32>* var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    bool var_114;
    const wp::float32 var_115 = 1e-15;
    bool var_116;
    bool var_117;
    const wp::float32 var_118 = 0.5;
    wp::float32 var_119;
    bool var_120;
    bool var_121;
    bool var_122;
    const wp::float32 var_123 = 0.0;
    wp::float32 var_124;
    bool var_125;
    bool var_126;
    bool var_127;
    const wp::float32 var_128 = 1.0;
    wp::float32 var_129;
    wp::int32 var_130;
    wp::vec_t<3, wp::float32>* var_131;
    wp::vec_t<3, wp::float32>* var_132;
    wp::vec_t<3, wp::float32> var_133;
    wp::vec_t<3, wp::float32> var_134;
    wp::vec_t<3, wp::float32> var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::int32 var_138;
    wp::vec_t<3, wp::float32> var_139;
    wp::float32 var_140;
    wp::float32 var_141;
    wp::int32 var_142;
    wp::vec_t<3, wp::float32> var_143;
    wp::float32 var_144;
    const wp::int32 var_145 = 0;
    wp::float32 var_146;
    const wp::int32 var_147 = 0;
    wp::float32 var_148;
    const wp::int32 var_149 = 1;
    wp::float32 var_150;
    const wp::int32 var_151 = 2;
    wp::float32 var_152;
    const wp::int32 var_153 = 2;
    wp::float32 var_154;
    wp::vec_t<5, wp::float32> var_155;
    bool var_156;
    wp::vec_t<2, wp::float32>* var_157;
    const wp::int32 var_158 = 0;
    wp::float32 var_159;
    wp::vec_t<2, wp::float32> var_160;
    const wp::float32 var_161 = 0.0;
    bool var_162;
    wp::vec_t<2, wp::float32>* var_163;
    const wp::int32 var_164 = 0;
    wp::float32 var_165;
    wp::vec_t<2, wp::float32> var_166;
    const wp::float32 var_167 = 0.0;
    bool var_168;
    wp::vec_t<2, wp::float32>* var_169;
    wp::vec_t<2, wp::float32> var_170;
    wp::vec_t<2, wp::float32> var_171;
    const wp::float32 var_172 = 1.0;
    wp::float32 var_173;
    wp::vec_t<2, wp::float32>* var_174;
    wp::vec_t<2, wp::float32> var_175;
    wp::vec_t<2, wp::float32> var_176;
    wp::vec_t<2, wp::float32> var_177;
    wp::vec_t<2, wp::float32>* var_178;
    wp::vec_t<2, wp::float32>* var_179;
    wp::vec_t<2, wp::float32> var_180;
    wp::vec_t<2, wp::float32> var_181;
    wp::vec_t<2, wp::float32> var_182;
    wp::vec_t<2, wp::float32> var_183;
    const wp::float32 var_184 = 0.0;
    const wp::float32 var_185 = 0.0;
    wp::vec_t<2, wp::float32> var_186;
    wp::vec_t<5, wp::float32>* var_187;
    wp::vec_t<5, wp::float32> var_188;
    wp::vec_t<5, wp::float32> var_189;
    const wp::float32 var_190 = 1.0;
    wp::float32 var_191;
    wp::vec_t<5, wp::float32>* var_192;
    wp::vec_t<5, wp::float32> var_193;
    wp::vec_t<5, wp::float32> var_194;
    wp::vec_t<5, wp::float32> var_195;
    wp::int32 var_196;
    wp::vec_t<5, wp::float32> var_197;
    wp::vec_t<2, wp::float32> var_198;
    wp::vec_t<2, wp::float32> var_199;
    wp::vec_t<5, wp::float32> var_200;
    wp::float32 var_201;
    const wp::float32 var_202 = 1e-05;
    const wp::int32 var_203 = 0;
    wp::float32 var_204;
    wp::float32 var_205;
    const wp::int32 var_206 = 1;
    wp::float32 var_207;
    wp::float32 var_208;
    const wp::int32 var_209 = 2;
    wp::float32 var_210;
    wp::float32 var_211;
    const wp::int32 var_212 = 3;
    wp::float32 var_213;
    wp::float32 var_214;
    const wp::int32 var_215 = 4;
    wp::float32 var_216;
    wp::float32 var_217;
    wp::vec_t<5, wp::float32> var_218;
    //---------
    // forward
    // def contact_material_params(                                                           <L 321>
    // if pairid > -1:                                                                        <L 341>
    var_1 = (var_pairid > var_0);
    if (var_1) {
        // condim = pair_dim[pairid]                                                          <L 342>
        var_2 = wp::address(var_pair_dim, var_pairid);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // friction = pair_friction[worldid % pair_friction.shape[0], pairid]                 <L 343>
        var_5 = &(var_pair_friction.shape);
        var_8 = wp::load(var_5);
        var_7 = wp::extract(var_8, var_6);
        var_9 = wp::mod(var_worldid, var_7);
        var_10 = wp::address(var_pair_friction, var_9, var_pairid);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // solref = pair_solref[worldid % pair_solref.shape[0], pairid]                       <L 344>
        var_13 = &(var_pair_solref.shape);
        var_16 = wp::load(var_13);
        var_15 = wp::extract(var_16, var_14);
        var_17 = wp::mod(var_worldid, var_15);
        var_18 = wp::address(var_pair_solref, var_17, var_pairid);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // solreffriction = pair_solreffriction[worldid % pair_solreffriction.shape[0], pairid]       <L 345>
        var_21 = &(var_pair_solreffriction.shape);
        var_24 = wp::load(var_21);
        var_23 = wp::extract(var_24, var_22);
        var_25 = wp::mod(var_worldid, var_23);
        var_26 = wp::address(var_pair_solreffriction, var_25, var_pairid);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // solimp = pair_solimp[worldid % pair_solimp.shape[0], pairid]                       <L 346>
        var_29 = &(var_pair_solimp.shape);
        var_32 = wp::load(var_29);
        var_31 = wp::extract(var_32, var_30);
        var_33 = wp::mod(var_worldid, var_31);
        var_34 = wp::address(var_pair_solimp, var_33, var_pairid);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // adhesion = pair_adhesion[worldid % pair_adhesion.shape[0], pairid]                 <L 347>
        var_37 = &(var_pair_adhesion.shape);
        var_40 = wp::load(var_37);
        var_39 = wp::extract(var_40, var_38);
        var_41 = wp::mod(var_worldid, var_39);
        var_42 = wp::address(var_pair_adhesion, var_41, var_pairid);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
    }
    if (!var_1) {
        // g1 = geoms[0]                                                                      <L 349>
        var_46 = wp::extract(var_geoms, var_45);
        // g2 = geoms[1]                                                                      <L 350>
        var_48 = wp::extract(var_geoms, var_47);
        // solmix_id = worldid % geom_solmix.shape[0]                                         <L 351>
        var_49 = &(var_geom_solmix.shape);
        var_52 = wp::load(var_49);
        var_51 = wp::extract(var_52, var_50);
        var_53 = wp::mod(var_worldid, var_51);
        // friction_id = worldid % geom_friction.shape[0]                                     <L 352>
        var_54 = &(var_geom_friction.shape);
        var_57 = wp::load(var_54);
        var_56 = wp::extract(var_57, var_55);
        var_58 = wp::mod(var_worldid, var_56);
        // solref_id = worldid % geom_solref.shape[0]                                         <L 353>
        var_59 = &(var_geom_solref.shape);
        var_62 = wp::load(var_59);
        var_61 = wp::extract(var_62, var_60);
        var_63 = wp::mod(var_worldid, var_61);
        // solimp_id = worldid % geom_solimp.shape[0]                                         <L 354>
        var_64 = &(var_geom_solimp.shape);
        var_67 = wp::load(var_64);
        var_66 = wp::extract(var_67, var_65);
        var_68 = wp::mod(var_worldid, var_66);
        // adhesion_id = worldid % geom_adhesion.shape[0]                                     <L 355>
        var_69 = &(var_geom_adhesion.shape);
        var_72 = wp::load(var_69);
        var_71 = wp::extract(var_72, var_70);
        var_73 = wp::mod(var_worldid, var_71);
        // solmix1 = geom_solmix[solmix_id, g1]                                               <L 357>
        var_74 = wp::address(var_geom_solmix, var_53, var_46);
        var_76 = wp::load(var_74);
        var_75 = wp::copy(var_76);
        // solmix2 = geom_solmix[solmix_id, g2]                                               <L 358>
        var_77 = wp::address(var_geom_solmix, var_53, var_48);
        var_79 = wp::load(var_77);
        var_78 = wp::copy(var_79);
        // condim1 = geom_condim[g1]                                                          <L 360>
        var_80 = wp::address(var_geom_condim, var_46);
        var_82 = wp::load(var_80);
        var_81 = wp::copy(var_82);
        // condim2 = geom_condim[g2]                                                          <L 361>
        var_83 = wp::address(var_geom_condim, var_48);
        var_85 = wp::load(var_83);
        var_84 = wp::copy(var_85);
        // p1 = geom_priority[g1]                                                             <L 364>
        var_86 = wp::address(var_geom_priority, var_46);
        var_88 = wp::load(var_86);
        var_87 = wp::copy(var_88);
        // p2 = geom_priority[g2]                                                             <L 365>
        var_89 = wp::address(var_geom_priority, var_48);
        var_91 = wp::load(var_89);
        var_90 = wp::copy(var_91);
        // a1 = geom_adhesion[adhesion_id, g1]                                                <L 366>
        var_92 = wp::address(var_geom_adhesion, var_73, var_46);
        var_94 = wp::load(var_92);
        var_93 = wp::copy(var_94);
        // a2 = geom_adhesion[adhesion_id, g2]                                                <L 367>
        var_95 = wp::address(var_geom_adhesion, var_73, var_48);
        var_97 = wp::load(var_95);
        var_96 = wp::copy(var_97);
        // if p1 > p2:                                                                        <L 369>
        var_98 = (var_87 > var_90);
        if (var_98) {
            // mix = 1.0                                                                      <L 370>
            // condim = condim1                                                               <L 371>
            var_100 = wp::copy(var_81);
            // max_geom_friction = geom_friction[friction_id, g1]                             <L 372>
            var_101 = wp::address(var_geom_friction, var_58, var_46);
            var_103 = wp::load(var_101);
            var_102 = wp::copy(var_103);
            // adhesion = a1                                                                  <L 373>
            var_104 = wp::copy(var_93);
        }
        if (!var_98) {
            // elif p2 > p1:                                                                  <L 374>
            var_105 = (var_90 > var_87);
            if (var_105) {
                // mix = 0.0                                                                  <L 375>
                // condim = condim2                                                           <L 376>
                var_107 = wp::copy(var_84);
                // max_geom_friction = geom_friction[friction_id, g2]                         <L 377>
                var_108 = wp::address(var_geom_friction, var_58, var_48);
                var_110 = wp::load(var_108);
                var_109 = wp::copy(var_110);
                // adhesion = a2                                                              <L 378>
                var_111 = wp::copy(var_96);
            }
            if (!var_105) {
                // mix = safe_div(solmix1, solmix1 + solmix2)                                 <L 380>
                var_112 = wp::add(var_75, var_78);
                var_113 = safe_div_0(var_75, var_112);
                // mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)       <L 381>
                var_116 = (var_75 < var_115);
                var_114 = var_116;
                if (var_114) {
                    var_117 = (var_78 < var_115);
                    var_114 = var_114 && var_117;
                }
                var_119 = wp::where(var_114, var_118, var_113);
                // mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)       <L 382>
                var_121 = (var_75 < var_115);
                var_120 = var_121;
                if (var_120) {
                    var_122 = (var_78 >= var_115);
                    var_120 = var_120 && var_122;
                }
                var_124 = wp::where(var_120, var_123, var_119);
                // mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)       <L 383>
                var_126 = (var_75 >= var_115);
                var_125 = var_126;
                if (var_125) {
                    var_127 = (var_78 < var_115);
                    var_125 = var_125 && var_127;
                }
                var_129 = wp::where(var_125, var_128, var_124);
                // condim = wp.max(condim1, condim2)                                          <L 384>
                var_130 = wp::max(var_81, var_84);
                // max_geom_friction = wp.max(geom_friction[friction_id, g1], geom_friction[friction_id, g2])       <L 385>
                var_131 = wp::address(var_geom_friction, var_58, var_46);
                var_132 = wp::address(var_geom_friction, var_58, var_48);
                var_134 = wp::load(var_131);
                var_135 = wp::load(var_132);
                var_133 = wp::max(var_134, var_135);
                // adhesion = a1 + a2                                                         <L 386>
                var_136 = wp::add(var_93, var_96);
            }
            var_137 = wp::where(var_105, var_106, var_129);
            var_138 = wp::where(var_105, var_107, var_130);
            var_139 = wp::where(var_105, var_109, var_133);
            var_140 = wp::where(var_105, var_111, var_136);
        }
        var_141 = wp::where(var_98, var_99, var_137);
        var_142 = wp::where(var_98, var_100, var_138);
        var_143 = wp::where(var_98, var_102, var_139);
        var_144 = wp::where(var_98, var_104, var_140);
        // friction = vec5(                                                                   <L 388>
        // max_geom_friction[0],                                                              <L 389>
        var_146 = wp::extract(var_143, var_145);
        // max_geom_friction[0],                                                              <L 390>
        var_148 = wp::extract(var_143, var_147);
        // max_geom_friction[1],                                                              <L 391>
        var_150 = wp::extract(var_143, var_149);
        // max_geom_friction[2],                                                              <L 392>
        var_152 = wp::extract(var_143, var_151);
        // max_geom_friction[2],                                                              <L 393>
        var_154 = wp::extract(var_143, var_153);
        var_155 = wp::vec_t<5, wp::float32>({var_146, var_148, var_150, var_152, var_154});
        // if geom_solref[solref_id, g1][0] > 0.0 and geom_solref[solref_id, g2][0] > 0.0:       <L 396>
        var_157 = wp::address(var_geom_solref, var_63, var_46);
        var_160 = wp::load(var_157);
        var_159 = wp::extract(var_160, var_158);
        var_162 = (var_159 > var_161);
        var_156 = var_162;
        if (var_156) {
            var_163 = wp::address(var_geom_solref, var_63, var_48);
            var_166 = wp::load(var_163);
            var_165 = wp::extract(var_166, var_164);
            var_168 = (var_165 > var_167);
            var_156 = var_156 && var_168;
        }
        if (var_156) {
            // solref = mix * geom_solref[solref_id, g1] + (1.0 - mix) * geom_solref[solref_id, g2]       <L 397>
            var_169 = wp::address(var_geom_solref, var_63, var_46);
            var_171 = wp::load(var_169);
            var_170 = wp::mul(var_141, var_171);
            var_173 = wp::sub(var_172, var_141);
            var_174 = wp::address(var_geom_solref, var_63, var_48);
            var_176 = wp::load(var_174);
            var_175 = wp::mul(var_173, var_176);
            var_177 = wp::add(var_170, var_175);
        }
        if (!var_156) {
            // solref = wp.min(geom_solref[solref_id, g1], geom_solref[solref_id, g2])        <L 399>
            var_178 = wp::address(var_geom_solref, var_63, var_46);
            var_179 = wp::address(var_geom_solref, var_63, var_48);
            var_181 = wp::load(var_178);
            var_182 = wp::load(var_179);
            var_180 = wp::min(var_181, var_182);
        }
        var_183 = wp::where(var_156, var_177, var_180);
        // solreffriction = wp.vec2(0.0, 0.0)                                                 <L 401>
        var_186 = wp::vec_t<2, wp::float32>(var_184, var_185);
        // solimp = mix * geom_solimp[solimp_id, g1] + (1.0 - mix) * geom_solimp[solimp_id, g2]       <L 402>
        var_187 = wp::address(var_geom_solimp, var_68, var_46);
        var_189 = wp::load(var_187);
        var_188 = wp::mul(var_141, var_189);
        var_191 = wp::sub(var_190, var_141);
        var_192 = wp::address(var_geom_solimp, var_68, var_48);
        var_194 = wp::load(var_192);
        var_193 = wp::mul(var_191, var_194);
        var_195 = wp::add(var_188, var_193);
    }
    var_196 = wp::where(var_1, var_3, var_142);
    var_197 = wp::where(var_1, var_11, var_155);
    var_198 = wp::where(var_1, var_19, var_183);
    var_199 = wp::where(var_1, var_27, var_186);
    var_200 = wp::where(var_1, var_35, var_195);
    var_201 = wp::where(var_1, var_43, var_144);
    // friction = vec5(                                                                       <L 404>
    // wp.max(MJ_MINMU, friction[0]),                                                         <L 405>
    var_204 = wp::extract(var_197, var_203);
    var_205 = wp::max(var_202, var_204);
    // wp.max(MJ_MINMU, friction[1]),                                                         <L 406>
    var_207 = wp::extract(var_197, var_206);
    var_208 = wp::max(var_202, var_207);
    // wp.max(MJ_MINMU, friction[2]),                                                         <L 407>
    var_210 = wp::extract(var_197, var_209);
    var_211 = wp::max(var_202, var_210);
    // wp.max(MJ_MINMU, friction[3]),                                                         <L 408>
    var_213 = wp::extract(var_197, var_212);
    var_214 = wp::max(var_202, var_213);
    // wp.max(MJ_MINMU, friction[4]),                                                         <L 409>
    var_216 = wp::extract(var_197, var_215);
    var_217 = wp::max(var_202, var_216);
    var_218 = wp::vec_t<5, wp::float32>({var_205, var_208, var_211, var_214, var_217});
    // return condim, friction, solref, solreffriction, solimp, adhesion                      <L 412>
    ret_0 = var_196;
    ret_1 = var_218;
    ret_2 = var_198;
    ret_3 = var_199;
    ret_4 = var_200;
    ret_5 = var_201;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:416
static CUDA_CALLABLE void contact_params_0(
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::float32> var_geom_adhesion,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::array_t<wp::float32> var_pair_adhesion,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::int32 var_cid,
    wp::int32 var_worldid,
    wp::vec_t<2, wp::int32> & ret_0,
    wp::float32 & ret_1,
    wp::float32 & ret_2,
    wp::int32 & ret_3,
    wp::vec_t<5, wp::float32> & ret_4,
    wp::vec_t<2, wp::float32> & ret_5,
    wp::vec_t<2, wp::float32> & ret_6,
    wp::vec_t<5, wp::float32> & ret_7,
    wp::float32 & ret_8)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::int32>* var_0;
    wp::vec_t<2, wp::int32> var_1;
    wp::vec_t<2, wp::int32> var_2;
    wp::vec_t<2, wp::int32>* var_3;
    const wp::int32 var_4 = 0;
    wp::int32 var_5;
    wp::vec_t<2, wp::int32> var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::int32 var_9;
    wp::vec_t<5, wp::float32> var_10;
    wp::vec_t<2, wp::float32> var_11;
    wp::vec_t<2, wp::float32> var_12;
    wp::vec_t<5, wp::float32> var_13;
    wp::float32 var_14;
    //---------
    // forward
    // def contact_params(                                                                    <L 417>
    // geoms = collision_pair_in[cid]                                                         <L 447>
    var_0 = wp::address(var_collision_pair_in, var_cid);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // pairid = collision_pairid_in[cid][0]                                                   <L 448>
    var_3 = wp::address(var_collision_pairid_in, var_cid);
    var_6 = wp::load(var_3);
    var_5 = wp::extract(var_6, var_4);
    // margin, gap = contact_margin_gap(geom_margin, geom_gap, pair_margin, pair_gap, geoms, pairid, worldid)       <L 450>
    contact_margin_gap_0(var_geom_margin, var_geom_gap, var_pair_margin, var_pair_gap, var_1, var_5, var_worldid, var_7, var_8);
    // condim, friction, solref, solreffriction, solimp, adhesion = contact_material_params(       <L 451>
    // geom_condim,                                                                           <L 452>
    // geom_priority,                                                                         <L 453>
    // geom_solmix,                                                                           <L 454>
    // geom_solref,                                                                           <L 455>
    // geom_solimp,                                                                           <L 456>
    // geom_friction,                                                                         <L 457>
    // geom_adhesion,                                                                         <L 458>
    // pair_dim,                                                                              <L 459>
    // pair_solref,                                                                           <L 460>
    // pair_solreffriction,                                                                   <L 461>
    // pair_solimp,                                                                           <L 462>
    // pair_adhesion,                                                                         <L 463>
    // pair_friction,                                                                         <L 464>
    // geoms,                                                                                 <L 465>
    // pairid,                                                                                <L 466>
    // worldid,                                                                               <L 467>
    contact_material_params_0(var_geom_condim, var_geom_priority, var_geom_solmix, var_geom_solref, var_geom_solimp, var_geom_friction, var_geom_adhesion, var_pair_dim, var_pair_solref, var_pair_solreffriction, var_pair_solimp, var_pair_adhesion, var_pair_friction, var_1, var_5, var_worldid, var_9, var_10, var_11, var_12, var_13, var_14);
    // return geoms, margin, gap, condim, friction, solref, solreffriction, solimp, adhesion       <L 470>
    ret_0 = var_1;
    ret_1 = var_7;
    ret_2 = var_8;
    ret_3 = var_9;
    ret_4 = var_10;
    ret_5 = var_11;
    ret_6 = var_12;
    ret_7 = var_13;
    ret_8 = var_14;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:65
static CUDA_CALLABLE void geom_collision_pair_from_types_0(
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_geom_type1,
    wp::int32 var_geom_type2,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    Geom_4022bfa7 & ret_0,
    Geom_4022bfa7 & ret_1)
{
    //---------
    // primal vars
    Geom_4022bfa7 var_0;
    Geom_4022bfa7 var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::mat_t<3, 3, wp::float32>* var_9;
    wp::mat_t<3, 3, wp::float32> var_10;
    wp::mat_t<3, 3, wp::float32> var_11;
    wp::shape_t* var_12;
    const wp::int32 var_13 = 0;
    wp::int32 var_14;
    wp::shape_t var_15;
    wp::int32 var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::mat_t<3, 3, wp::float32>* var_20;
    const wp::int32 var_21 = 0;
    const wp::int32 var_22 = 2;
    wp::float32 var_23;
    wp::mat_t<3, 3, wp::float32> var_24;
    wp::mat_t<3, 3, wp::float32>* var_25;
    const wp::int32 var_26 = 1;
    const wp::int32 var_27 = 2;
    wp::float32 var_28;
    wp::mat_t<3, 3, wp::float32> var_29;
    wp::mat_t<3, 3, wp::float32>* var_30;
    const wp::int32 var_31 = 2;
    const wp::int32 var_32 = 2;
    wp::float32 var_33;
    wp::mat_t<3, 3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32>* var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::mat_t<3, 3, wp::float32>* var_39;
    wp::mat_t<3, 3, wp::float32> var_40;
    wp::mat_t<3, 3, wp::float32> var_41;
    wp::shape_t* var_42;
    const wp::int32 var_43 = 0;
    wp::int32 var_44;
    wp::shape_t var_45;
    wp::int32 var_46;
    wp::vec_t<3, wp::float32>* var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::mat_t<3, 3, wp::float32>* var_50;
    const wp::int32 var_51 = 0;
    const wp::int32 var_52 = 2;
    wp::float32 var_53;
    wp::mat_t<3, 3, wp::float32> var_54;
    wp::mat_t<3, 3, wp::float32>* var_55;
    const wp::int32 var_56 = 1;
    const wp::int32 var_57 = 2;
    wp::float32 var_58;
    wp::mat_t<3, 3, wp::float32> var_59;
    wp::mat_t<3, 3, wp::float32>* var_60;
    const wp::int32 var_61 = 2;
    const wp::int32 var_62 = 2;
    wp::float32 var_63;
    wp::mat_t<3, 3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::shape_t* var_66;
    const wp::int32 var_67 = 0;
    wp::int32 var_68;
    wp::shape_t var_69;
    wp::int32 var_70;
    const wp::int32 var_71 = 7;
    bool var_72;
    wp::int32* var_73;
    wp::int32 var_74;
    wp::int32 var_75;
    const wp::int32 var_76 = 0;
    bool var_77;
    wp::int32* var_78;
    const wp::int32 var_79 = -1;
    wp::int32 var_80;
    wp::int32 var_81;
    const wp::int32 var_82 = 0;
    bool var_83;
    wp::int32* var_84;
    const wp::int32 var_85 = -1;
    wp::int32 var_86;
    wp::int32 var_87;
    const wp::int32 var_88 = 0;
    bool var_89;
    wp::int32* var_90;
    const wp::int32 var_91 = -1;
    wp::int32 var_92;
    wp::int32 var_93;
    const wp::int32 var_94 = 0;
    bool var_95;
    wp::int32* var_96;
    const wp::int32 var_97 = -1;
    wp::int32 var_98;
    wp::int32 var_99;
    const wp::int32 var_100 = 0;
    bool var_101;
    wp::int32* var_102;
    const wp::int32 var_103 = -1;
    wp::int32 var_104;
    wp::int32 var_105;
    const wp::int32 var_106 = 7;
    bool var_107;
    wp::int32* var_108;
    wp::int32 var_109;
    wp::int32 var_110;
    const wp::int32 var_111 = 0;
    bool var_112;
    wp::int32* var_113;
    const wp::int32 var_114 = -1;
    wp::int32 var_115;
    wp::int32 var_116;
    const wp::int32 var_117 = 0;
    bool var_118;
    wp::int32* var_119;
    const wp::int32 var_120 = -1;
    wp::int32 var_121;
    wp::int32 var_122;
    const wp::int32 var_123 = 0;
    bool var_124;
    wp::int32* var_125;
    const wp::int32 var_126 = -1;
    wp::int32 var_127;
    wp::int32 var_128;
    const wp::int32 var_129 = 0;
    bool var_130;
    wp::int32* var_131;
    const wp::int32 var_132 = -1;
    wp::int32 var_133;
    wp::int32 var_134;
    const wp::int32 var_135 = 0;
    bool var_136;
    wp::int32* var_137;
    const wp::int32 var_138 = -1;
    wp::int32 var_139;
    wp::int32 var_140;
    wp::int32 var_141;
    const wp::int32 var_142 = -1;
    const wp::float32 var_143 = 0.0;
    const wp::int32 var_144 = -1;
    const wp::float32 var_145 = 0.0;
    //---------
    // forward
    // def geom_collision_pair_from_types(                                                    <L 66>
    // geom1 = Geom()                                                                         <L 93>
    var_0 = Geom_4022bfa7();
    // geom2 = Geom()                                                                         <L 94>
    var_1 = Geom_4022bfa7();
    // g1 = geoms[0]                                                                          <L 96>
    var_3 = wp::extract(var_geoms, var_2);
    // g2 = geoms[1]                                                                          <L 97>
    var_5 = wp::extract(var_geoms, var_4);
    // geom1.pos = geom_xpos_in[worldid, g1]                                                  <L 99>
    var_6 = wp::address(var_geom_xpos_in, var_worldid, var_3);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    var_0.pos = var_7;
    // geom1.rot = geom_xmat_in[worldid, g1]                                                  <L 100>
    var_9 = wp::address(var_geom_xmat_in, var_worldid, var_3);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    var_0.rot = var_10;
    // geom1.size = geom_size[worldid % geom_size.shape[0], g1]                               <L 101>
    var_12 = &(var_geom_size.shape);
    var_15 = wp::load(var_12);
    var_14 = wp::extract(var_15, var_13);
    var_16 = wp::mod(var_worldid, var_14);
    var_17 = wp::address(var_geom_size, var_16, var_3);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    var_0.size = var_18;
    // geom1.normal = wp.vec3(geom1.rot[0, 2], geom1.rot[1, 2], geom1.rot[2, 2])              <L 103>
    var_20 = &((var_0).rot);
    var_24 = wp::load(var_20);
    var_23 = wp::extract(var_24, var_21, var_22);
    var_25 = &((var_0).rot);
    var_29 = wp::load(var_25);
    var_28 = wp::extract(var_29, var_26, var_27);
    var_30 = &((var_0).rot);
    var_34 = wp::load(var_30);
    var_33 = wp::extract(var_34, var_31, var_32);
    var_35 = wp::vec_t<3, wp::float32>(var_23, var_28, var_33);
    var_0.normal = var_35;
    // geom2.pos = geom_xpos_in[worldid, g2]                                                  <L 105>
    var_36 = wp::address(var_geom_xpos_in, var_worldid, var_5);
    var_38 = wp::load(var_36);
    var_37 = wp::copy(var_38);
    var_1.pos = var_37;
    // geom2.rot = geom_xmat_in[worldid, g2]                                                  <L 106>
    var_39 = wp::address(var_geom_xmat_in, var_worldid, var_5);
    var_41 = wp::load(var_39);
    var_40 = wp::copy(var_41);
    var_1.rot = var_40;
    // geom2.size = geom_size[worldid % geom_size.shape[0], g2]                               <L 107>
    var_42 = &(var_geom_size.shape);
    var_45 = wp::load(var_42);
    var_44 = wp::extract(var_45, var_43);
    var_46 = wp::mod(var_worldid, var_44);
    var_47 = wp::address(var_geom_size, var_46, var_5);
    var_49 = wp::load(var_47);
    var_48 = wp::copy(var_49);
    var_1.size = var_48;
    // geom2.normal = wp.vec3(geom2.rot[0, 2], geom2.rot[1, 2], geom2.rot[2, 2])              <L 109>
    var_50 = &((var_1).rot);
    var_54 = wp::load(var_50);
    var_53 = wp::extract(var_54, var_51, var_52);
    var_55 = &((var_1).rot);
    var_59 = wp::load(var_55);
    var_58 = wp::extract(var_59, var_56, var_57);
    var_60 = &((var_1).rot);
    var_64 = wp::load(var_60);
    var_63 = wp::extract(var_64, var_61, var_62);
    var_65 = wp::vec_t<3, wp::float32>(var_53, var_58, var_63);
    var_1.normal = var_65;
    // dataid_setid = worldid % geom_dataid.shape[0]                                          <L 111>
    var_66 = &(var_geom_dataid.shape);
    var_69 = wp::load(var_66);
    var_68 = wp::extract(var_69, var_67);
    var_70 = wp::mod(var_worldid, var_68);
    // if geom_type1 == GeomType.MESH:                                                        <L 113>
    var_72 = (var_geom_type1 == var_71);
    if (var_72) {
        // dataid = geom_dataid[dataid_setid, g1]                                             <L 114>
        var_73 = wp::address(var_geom_dataid, var_70, var_3);
        var_75 = wp::load(var_73);
        var_74 = wp::copy(var_75);
        // geom1.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)                    <L 115>
        var_77 = (var_74 >= var_76);
        var_78 = wp::address(var_mesh_vertadr, var_74);
        var_81 = wp::load(var_78);
        var_80 = wp::where(var_77, var_81, var_79);
        var_0.vertadr = var_80;
        // geom1.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)                    <L 116>
        var_83 = (var_74 >= var_82);
        var_84 = wp::address(var_mesh_vertnum, var_74);
        var_87 = wp::load(var_84);
        var_86 = wp::where(var_83, var_87, var_85);
        var_0.vertnum = var_86;
        // geom1.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)                  <L 117>
        var_89 = (var_74 >= var_88);
        var_90 = wp::address(var_mesh_graphadr, var_74);
        var_93 = wp::load(var_90);
        var_92 = wp::where(var_89, var_93, var_91);
        var_0.graphadr = var_92;
        // geom1.mesh_polynum = wp.where(dataid >= 0, mesh_polynum[dataid], -1)               <L 118>
        var_95 = (var_74 >= var_94);
        var_96 = wp::address(var_mesh_polynum, var_74);
        var_99 = wp::load(var_96);
        var_98 = wp::where(var_95, var_99, var_97);
        var_0.mesh_polynum = var_98;
        // geom1.mesh_polyadr = wp.where(dataid >= 0, mesh_polyadr[dataid], -1)               <L 119>
        var_101 = (var_74 >= var_100);
        var_102 = wp::address(var_mesh_polyadr, var_74);
        var_105 = wp::load(var_102);
        var_104 = wp::where(var_101, var_105, var_103);
        var_0.mesh_polyadr = var_104;
        // geom1.vert = mesh_vert                                                             <L 121>
        var_0.vert = var_mesh_vert;
        // geom1.graph = mesh_graph                                                           <L 122>
        var_0.graph = var_mesh_graph;
        // geom1.mesh_polynormal = mesh_polynormal                                            <L 123>
        var_0.mesh_polynormal = var_mesh_polynormal;
        // geom1.mesh_polyvertadr = mesh_polyvertadr                                          <L 124>
        var_0.mesh_polyvertadr = var_mesh_polyvertadr;
        // geom1.mesh_polyvertnum = mesh_polyvertnum                                          <L 125>
        var_0.mesh_polyvertnum = var_mesh_polyvertnum;
        // geom1.mesh_polyvert = mesh_polyvert                                                <L 126>
        var_0.mesh_polyvert = var_mesh_polyvert;
        // geom1.mesh_polymapadr = mesh_polymapadr                                            <L 127>
        var_0.mesh_polymapadr = var_mesh_polymapadr;
        // geom1.mesh_polymapnum = mesh_polymapnum                                            <L 128>
        var_0.mesh_polymapnum = var_mesh_polymapnum;
        // geom1.mesh_polymap = mesh_polymap                                                  <L 129>
        var_0.mesh_polymap = var_mesh_polymap;
    }
    // if geom_type2 == GeomType.MESH:                                                        <L 131>
    var_107 = (var_geom_type2 == var_106);
    if (var_107) {
        // dataid = geom_dataid[dataid_setid, g2]                                             <L 132>
        var_108 = wp::address(var_geom_dataid, var_70, var_5);
        var_110 = wp::load(var_108);
        var_109 = wp::copy(var_110);
        // geom2.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)                    <L 133>
        var_112 = (var_109 >= var_111);
        var_113 = wp::address(var_mesh_vertadr, var_109);
        var_116 = wp::load(var_113);
        var_115 = wp::where(var_112, var_116, var_114);
        var_1.vertadr = var_115;
        // geom2.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)                    <L 134>
        var_118 = (var_109 >= var_117);
        var_119 = wp::address(var_mesh_vertnum, var_109);
        var_122 = wp::load(var_119);
        var_121 = wp::where(var_118, var_122, var_120);
        var_1.vertnum = var_121;
        // geom2.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)                  <L 135>
        var_124 = (var_109 >= var_123);
        var_125 = wp::address(var_mesh_graphadr, var_109);
        var_128 = wp::load(var_125);
        var_127 = wp::where(var_124, var_128, var_126);
        var_1.graphadr = var_127;
        // geom2.mesh_polynum = wp.where(dataid >= 0, mesh_polynum[dataid], -1)               <L 136>
        var_130 = (var_109 >= var_129);
        var_131 = wp::address(var_mesh_polynum, var_109);
        var_134 = wp::load(var_131);
        var_133 = wp::where(var_130, var_134, var_132);
        var_1.mesh_polynum = var_133;
        // geom2.mesh_polyadr = wp.where(dataid >= 0, mesh_polyadr[dataid], -1)               <L 137>
        var_136 = (var_109 >= var_135);
        var_137 = wp::address(var_mesh_polyadr, var_109);
        var_140 = wp::load(var_137);
        var_139 = wp::where(var_136, var_140, var_138);
        var_1.mesh_polyadr = var_139;
        // geom2.vert = mesh_vert                                                             <L 139>
        var_1.vert = var_mesh_vert;
        // geom2.graph = mesh_graph                                                           <L 140>
        var_1.graph = var_mesh_graph;
        // geom2.mesh_polynormal = mesh_polynormal                                            <L 141>
        var_1.mesh_polynormal = var_mesh_polynormal;
        // geom2.mesh_polyvertadr = mesh_polyvertadr                                          <L 142>
        var_1.mesh_polyvertadr = var_mesh_polyvertadr;
        // geom2.mesh_polyvertnum = mesh_polyvertnum                                          <L 143>
        var_1.mesh_polyvertnum = var_mesh_polyvertnum;
        // geom2.mesh_polyvert = mesh_polyvert                                                <L 144>
        var_1.mesh_polyvert = var_mesh_polyvert;
        // geom2.mesh_polymapadr = mesh_polymapadr                                            <L 145>
        var_1.mesh_polymapadr = var_mesh_polymapadr;
        // geom2.mesh_polymapnum = mesh_polymapnum                                            <L 146>
        var_1.mesh_polymapnum = var_mesh_polymapnum;
        // geom2.mesh_polymap = mesh_polymap                                                  <L 147>
        var_1.mesh_polymap = var_mesh_polymap;
    }
    var_141 = wp::where(var_107, var_109, var_74);
    // geom1.index = -1                                                                       <L 149>
    var_0.index = var_142;
    // geom1.margin = 0.0                                                                     <L 150>
    var_0.margin = var_143;
    // geom2.index = -1                                                                       <L 152>
    var_1.index = var_144;
    // geom2.margin = 0.0                                                                     <L 153>
    var_1.margin = var_145;
    // return geom1, geom2                                                                    <L 155>
    ret_0 = var_0;
    ret_1 = var_1;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:158
static CUDA_CALLABLE void geom_collision_pair_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    Geom_4022bfa7 & ret_0,
    Geom_4022bfa7 & ret_1)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    wp::int32* var_2;
    wp::int32 var_3;
    wp::int32 var_4;
    const wp::int32 var_5 = 1;
    wp::int32 var_6;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    Geom_4022bfa7 var_10;
    Geom_4022bfa7 var_11;
    //---------
    // forward
    // def geom_collision_pair(                                                               <L 159>
    // geom_type1 = geom_type[geoms[0]]                                                       <L 185>
    var_1 = wp::extract(var_geoms, var_0);
    var_2 = wp::address(var_geom_type, var_1);
    var_4 = wp::load(var_2);
    var_3 = wp::copy(var_4);
    // geom_type2 = geom_type[geoms[1]]                                                       <L 186>
    var_6 = wp::extract(var_geoms, var_5);
    var_7 = wp::address(var_geom_type, var_6);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // return geom_collision_pair_from_types(                                                 <L 187>
    // geom_dataid,                                                                           <L 188>
    // geom_size,                                                                             <L 189>
    // mesh_vertadr,                                                                          <L 190>
    // mesh_vertnum,                                                                          <L 191>
    // mesh_graphadr,                                                                         <L 192>
    // mesh_vert,                                                                             <L 193>
    // mesh_graph,                                                                            <L 194>
    // mesh_polynum,                                                                          <L 195>
    // mesh_polyadr,                                                                          <L 196>
    // mesh_polynormal,                                                                       <L 197>
    // mesh_polyvertadr,                                                                      <L 198>
    // mesh_polyvertnum,                                                                      <L 199>
    // mesh_polyvert,                                                                         <L 200>
    // mesh_polymapadr,                                                                       <L 201>
    // mesh_polymapnum,                                                                       <L 202>
    // mesh_polymap,                                                                          <L 203>
    // geom_xpos_in,                                                                          <L 204>
    // geom_xmat_in,                                                                          <L 205>
    // geom_type1,                                                                            <L 206>
    // geom_type2,                                                                            <L 207>
    // geoms,                                                                                 <L 208>
    // worldid,                                                                               <L 209>
    geom_collision_pair_from_types_0(var_geom_dataid, var_geom_size, var_mesh_vertadr, var_mesh_vertnum, var_mesh_graphadr, var_mesh_vert, var_mesh_graph, var_mesh_polynum, var_mesh_polyadr, var_mesh_polynormal, var_mesh_polyvertadr, var_mesh_polyvertnum, var_mesh_polyvert, var_mesh_polymapadr, var_mesh_polymapnum, var_mesh_polymap, var_geom_xpos_in, var_geom_xmat_in, var_3, var_8, var_geoms, var_worldid, var_10, var_11);
    ret_0 = var_10;
    ret_1 = var_11;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:50
static CUDA_CALLABLE void plane_sphere_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    //---------
    // forward
    // def plane_sphere(plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float) -> Tuple[float, wp.vec3]:       <L 51>
    // dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius                    <L 53>
    var_0 = wp::sub(var_sphere_pos, var_plane_pos);
    var_1 = wp::dot(var_0, var_plane_normal);
    var_2 = wp::sub(var_1, var_sphere_radius);
    // pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)                         <L 54>
    var_4 = wp::mul(var_3, var_2);
    var_5 = wp::add(var_sphere_radius, var_4);
    var_6 = wp::mul(var_plane_normal, var_5);
    var_7 = wp::sub(var_sphere_pos, var_6);
    // return dist, pos                                                                       <L 55>
    ret_0 = var_2;
    ret_1 = var_7;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:202
static CUDA_CALLABLE void orthogonals_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 1.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 1.0;
    wp::vec_t<3, wp::float32> var_7;
    bool var_8;
    const wp::float32 var_9 = -0.5;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    bool var_12;
    const wp::int32 var_13 = 1;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.5;
    bool var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::float32 var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::float32 var_22;
    const wp::float32 var_23 = 0.0;
    bool var_24;
    const wp::float32 var_25 = 0.0;
    const wp::float32 var_26 = 0.0;
    const wp::float32 var_27 = 0.0;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    //---------
    // forward
    // def orthogonals(a: wp.vec3):                                                           <L 203>
    // y = wp.vec3(0.0, 1.0, 0.0)                                                             <L 204>
    var_3 = wp::vec_t<3, wp::float32>(var_0, var_1, var_2);
    // z = wp.vec3(0.0, 0.0, 1.0)                                                             <L 205>
    var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    // b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)                                     <L 206>
    var_11 = wp::extract(var_a, var_10);
    var_12 = (var_9 < var_11);
    var_8 = var_12;
    if (var_8) {
        var_14 = wp::extract(var_a, var_13);
        var_16 = (var_14 < var_15);
        var_8 = var_8 && var_16;
    }
    var_17 = wp::where(var_8, var_3, var_7);
    // b = b - a * wp.dot(a, b)                                                               <L 207>
    var_18 = wp::dot(var_a, var_17);
    var_19 = wp::mul(var_a, var_18);
    var_20 = wp::sub(var_17, var_19);
    // b = wp.normalize(b)                                                                    <L 208>
    var_21 = wp::normalize(var_20);
    // if wp.length(a) == 0.0:                                                                <L 209>
    var_22 = wp::length(var_a);
    var_24 = (var_22 == var_23);
    if (var_24) {
        // b = wp.vec3(0.0, 0.0, 0.0)                                                         <L 210>
        var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
    }
    var_29 = wp::where(var_24, var_28, var_21);
    // c = wp.cross(a, b)                                                                     <L 211>
    var_30 = wp::cross(var_a, var_29);
    // return b, c                                                                            <L 213>
    ret_0 = var_29;
    ret_1 = var_30;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:246
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> make_frame_0(
    wp::vec_t<3, wp::float32> var_a)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    const wp::int32 var_9 = 0;
    wp::float32 var_10;
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    const wp::int32 var_15 = 0;
    wp::float32 var_16;
    const wp::int32 var_17 = 1;
    wp::float32 var_18;
    const wp::int32 var_19 = 2;
    wp::float32 var_20;
    wp::mat_t<3, 3, wp::float32> var_21;
    //---------
    // forward
    // def make_frame(a: wp.vec3):                                                            <L 247>
    // a = wp.normalize(a)                                                                    <L 248>
    var_0 = wp::normalize(var_a);
    // b, c = orthogonals(a)                                                                  <L 249>
    orthogonals_0(var_0, var_1, var_2);
    // return wp.mat33(                                                                       <L 252>
    // a.x, a.y, a.z,                                                                         <L 253>
    var_4 = wp::extract(var_0, var_3);
    var_6 = wp::extract(var_0, var_5);
    var_8 = wp::extract(var_0, var_7);
    // b.x, b.y, b.z,                                                                         <L 254>
    var_10 = wp::extract(var_1, var_9);
    var_12 = wp::extract(var_1, var_11);
    var_14 = wp::extract(var_1, var_13);
    // c.x, c.y, c.z                                                                          <L 255>
    var_16 = wp::extract(var_2, var_15);
    var_18 = wp::extract(var_2, var_17);
    var_20 = wp::extract(var_2, var_19);
    var_21 = wp::mat_t<3, 3, wp::float32>(var_4, var_6, var_8, var_10, var_12, var_14, var_16, var_18, var_20);
    return var_21;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:213
static CUDA_CALLABLE wp::int32 write_contact_0(
    wp::int32 var_naconmax_in,
    wp::int32 var_id_,
    wp::float32 var_dist_in,
    wp::vec_t<3, wp::float32> var_pos_in,
    wp::mat_t<3, 3, wp::float32> var_frame_in,
    wp::float32 var_margin_in,
    wp::float32 var_gap_in,
    wp::int32 var_condim_in,
    wp::vec_t<5, wp::float32> var_friction_in,
    wp::vec_t<2, wp::float32> var_solref_in,
    wp::vec_t<2, wp::float32> var_solreffriction_in,
    wp::vec_t<5, wp::float32> var_solimp_in,
    wp::float32 var_adhesion_in,
    wp::vec_t<2, wp::int32> var_geoms_in,
    wp::vec_t<2, wp::int32> var_pairid_in,
    wp::int32 var_worldid_in,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    bool var_0;
    bool var_1;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    bool var_4;
    wp::float32 var_5;
    bool var_6;
    wp::float32 var_7;
    bool var_8;
    bool var_9;
    const wp::float32 var_10 = 0.0;
    bool var_11;
    bool var_12;
    const wp::int32 var_13 = 1;
    wp::int32 var_14;
    bool var_15;
    bool var_16;
    const wp::int32 var_17 = 0;
    wp::int32 var_18;
    const wp::int32 var_19 = -2;
    bool var_20;
    bool var_21;
    const wp::int32 var_22 = 1;
    wp::int32 var_23;
    const wp::int32 var_24 = -1;
    bool var_25;
    const wp::int32 var_26 = 0;
    const wp::int32 var_27 = 0;
    bool var_28;
    const wp::int32 var_29 = 0;
    wp::int32 var_30;
    const wp::int32 var_31 = -1;
    bool var_32;
    const wp::int32 var_33 = 1;
    wp::int32 var_34;
    wp::int32 var_35;
    const wp::int32 var_36 = 1;
    wp::int32 var_37;
    const wp::int32 var_38 = 0;
    bool var_39;
    const wp::int32 var_40 = 2;
    wp::int32 var_41;
    wp::int32 var_42;
    const wp::int32 var_43 = 0;
    const wp::int32 var_44 = 1;
    wp::int32 var_45;
    bool var_46;
    wp::float32 var_47;
    wp::shape_t* var_48;
    const wp::int32 var_49 = 1;
    wp::int32 var_50;
    wp::shape_t var_51;
    wp::range_t var_52;
    wp::int32 var_53;
    const wp::int32 var_54 = -1;
    wp::int32 var_55;
    const wp::int32 var_56 = 0;
    //---------
    // forward
    // def write_contact(                                                                     <L 214>
    // active = (dist_in < margin_in) or ((adhesion_in != 0.0) and (dist_in < margin_in + gap_in))       <L 255>
    var_1 = (var_dist_in < var_margin_in);
    var_0 = var_1;
    if (!var_0) {
        var_4 = (var_adhesion_in != var_3);
        var_2 = var_4;
        if (var_2) {
            var_5 = wp::add(var_margin_in, var_gap_in);
            var_6 = (var_dist_in < var_5);
            var_2 = var_2 && var_6;
        }
        var_0 = var_0 || var_2;
    }
    // detected = dist_in < margin_in + gap_in                                                <L 256>
    var_7 = wp::add(var_margin_in, var_gap_in);
    var_8 = (var_dist_in < var_7);
    // condim = wp.where((adhesion_in != 0.0) and (dist_in >= margin_in), 1, condim_in)       <L 257>
    var_11 = (var_adhesion_in != var_10);
    var_9 = var_11;
    if (var_9) {
        var_12 = (var_dist_in >= var_margin_in);
        var_9 = var_9 && var_12;
    }
    var_14 = wp::where(var_9, var_13, var_condim_in);
    // if (pairid_in[0] == -2 or not detected) and pairid_in[1] == -1:                        <L 260>
    var_18 = wp::extract(var_pairid_in, var_17);
    var_20 = (var_18 == var_19);
    var_16 = var_20;
    if (!var_16) {
        var_21 = wp::unot(var_8);
        var_16 = var_16 || var_21;
    }
    var_15 = var_16;
    if (var_15) {
        var_23 = wp::extract(var_pairid_in, var_22);
        var_25 = (var_23 == var_24);
        var_15 = var_15 && var_25;
    }
    if (var_15) {
        // return 0                                                                           <L 261>
        return var_26;
    }
    // contact_type = 0                                                                       <L 263>
    // if pairid_in[0] >= -1 and detected:                                                    <L 265>
    var_30 = wp::extract(var_pairid_in, var_29);
    var_32 = (var_30 >= var_31);
    var_28 = var_32;
    if (var_28) {
        var_28 = var_28 && var_8;
    }
    if (var_28) {
        // contact_type |= ContactType.CONSTRAINT                                             <L 266>
        var_34 = wp::bit_or(var_27, var_33);
    }
    var_35 = wp::where(var_28, var_34, var_27);
    // if pairid_in[1] >= 0:                                                                  <L 268>
    var_37 = wp::extract(var_pairid_in, var_36);
    var_39 = (var_37 >= var_38);
    if (var_39) {
        // contact_type |= ContactType.SENSOR                                                 <L 269>
        var_41 = wp::bit_or(var_35, var_40);
    }
    var_42 = wp::where(var_39, var_41, var_35);
    // cid = wp.atomic_add(nacon_out, 0, 1)                                                   <L 271>
    var_45 = wp::atomic_add(var_nacon_out, var_43, var_44);
    // if cid < naconmax_in:                                                                  <L 272>
    var_46 = (var_45 < var_naconmax_in);
    if (var_46) {
        // contact_dist_out[cid] = dist_in                                                    <L 273>
        wp::array_store(var_contact_dist_out, var_45, var_dist_in);
        // contact_pos_out[cid] = pos_in                                                      <L 274>
        wp::array_store(var_contact_pos_out, var_45, var_pos_in);
        // contact_frame_out[cid] = frame_in                                                  <L 275>
        wp::array_store(var_contact_frame_out, var_45, var_frame_in);
        // contact_geom_out[cid] = geoms_in                                                   <L 276>
        wp::array_store(var_contact_geom_out, var_45, var_geoms_in);
        // contact_worldid_out[cid] = worldid_in                                              <L 277>
        wp::array_store(var_contact_worldid_out, var_45, var_worldid_in);
        // includemargin = margin_in                                                          <L 278>
        var_47 = wp::copy(var_margin_in);
        // contact_includemargin_out[cid] = includemargin                                     <L 279>
        wp::array_store(var_contact_includemargin_out, var_45, var_47);
        // contact_dim_out[cid] = condim                                                      <L 280>
        wp::array_store(var_contact_dim_out, var_45, var_14);
        // contact_friction_out[cid] = friction_in                                            <L 281>
        wp::array_store(var_contact_friction_out, var_45, var_friction_in);
        // contact_solref_out[cid] = solref_in                                                <L 282>
        wp::array_store(var_contact_solref_out, var_45, var_solref_in);
        // contact_solreffriction_out[cid] = solreffriction_in                                <L 283>
        wp::array_store(var_contact_solreffriction_out, var_45, var_solreffriction_in);
        // contact_solimp_out[cid] = solimp_in                                                <L 284>
        wp::array_store(var_contact_solimp_out, var_45, var_solimp_in);
        // contact_adhesion_out[cid] = adhesion_in                                            <L 285>
        wp::array_store(var_contact_adhesion_out, var_45, var_adhesion_in);
        // contact_type_out[cid] = contact_type                                               <L 286>
        wp::array_store(var_contact_type_out, var_45, var_42);
        // contact_geomcollisionid_out[cid] = id_                                             <L 287>
        wp::array_store(var_contact_geomcollisionid_out, var_45, var_id_);
        // for i in range(contact_efc_address_out.shape[1]):                                  <L 288>
        var_48 = &(var_contact_efc_address_out.shape);
        var_51 = wp::load(var_48);
        var_50 = wp::extract(var_51, var_49);
        var_52 = wp::range(var_50);
        start_for_1:;
            if (iter_cmp(var_52) == 0) goto end_for_1;
            var_53 = wp::iter_next(var_52);
            // contact_efc_address_out[cid, i] = -1                                           <L 289>
            wp::array_store(var_contact_efc_address_out, var_45, var_53, var_54);
            goto start_for_1;
        end_for_1:;
        // return int(active)                                                                 <L 290>
        var_55 = wp::int(var_0);
        return var_55;
    }
    // return 0                                                                               <L 291>
    return var_56;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:212
static CUDA_CALLABLE void plane_sphere_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_plane,
    Geom_4022bfa7 var_sphere,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_adhesion,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32>* var_4;
    wp::vec_t<3, wp::float32>* var_5;
    const wp::int32 var_6 = 0;
    wp::float32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    const wp::int32 var_13 = 0;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::int32 var_15;
    //---------
    // forward
    // def plane_sphere_wrapper(                                                              <L 213>
    // normal = plane.normal                                                                  <L 249>
    var_0 = &((var_plane).normal);
    var_2 = wp::load(var_0);
    var_1 = wp::copy(var_2);
    // dist, pos = plane_sphere(normal, plane.pos, sphere.pos, sphere.size[0])                <L 250>
    var_3 = &((var_plane).pos);
    var_4 = &((var_sphere).pos);
    var_5 = &((var_sphere).size);
    var_8 = wp::load(var_5);
    var_7 = wp::extract(var_8, var_6);
    var_11 = wp::load(var_3);
    var_12 = wp::load(var_4);
    plane_sphere_0(var_1, var_11, var_12, var_7, var_9, var_10);
    // write_contact(                                                                         <L 252>
    // naconmax_in,                                                                           <L 253>
    // 0,                                                                                     <L 254>
    // dist,                                                                                  <L 255>
    // pos,                                                                                   <L 256>
    // make_frame(normal),                                                                    <L 257>
    var_14 = make_frame_0(var_1);
    // margin,                                                                                <L 258>
    // gap,                                                                                   <L 259>
    // condim,                                                                                <L 260>
    // friction,                                                                              <L 261>
    // solref,                                                                                <L 262>
    // solreffriction,                                                                        <L 263>
    // solimp,                                                                                <L 264>
    // adhesion,                                                                              <L 265>
    // geoms,                                                                                 <L 266>
    // pairid,                                                                                <L 267>
    // worldid,                                                                               <L 268>
    // contact_dist_out,                                                                      <L 269>
    // contact_pos_out,                                                                       <L 270>
    // contact_frame_out,                                                                     <L 271>
    // contact_includemargin_out,                                                             <L 272>
    // contact_friction_out,                                                                  <L 273>
    // contact_solref_out,                                                                    <L 274>
    // contact_solreffriction_out,                                                            <L 275>
    // contact_solimp_out,                                                                    <L 276>
    // contact_dim_out,                                                                       <L 277>
    // contact_geom_out,                                                                      <L 278>
    // contact_efc_address_out,                                                               <L 279>
    // contact_worldid_out,                                                                   <L 280>
    // contact_type_out,                                                                      <L 281>
    // contact_geomcollisionid_out,                                                           <L 282>
    // contact_adhesion_out,                                                                  <L 283>
    // nacon_out,                                                                             <L 284>
    var_15 = write_contact_0(var_naconmax_in, var_13, var_9, var_10, var_14, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE void normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 0.0;
    bool var_2;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    //---------
    // forward
    // def normalize_with_norm(x: Any):                                                       <L 1>
    // norm = wp.length(x)                                                                    <L 2>
    var_0 = wp::length(var_x);
    // if norm == 0.0:                                                                        <L 3>
    var_2 = (var_0 == var_1);
    if (var_2) {
        // return x, 0.0                                                                      <L 4>
        ret_0 = var_x;
        ret_1 = var_3;
        return;
    }
    // return x / norm, norm                                                                  <L 5>
    var_4 = wp::div(var_x, var_0);
    ret_0 = var_4;
    ret_1 = var_0;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:258
static CUDA_CALLABLE void plane_capsule_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<3, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.5;
    bool var_8;
    bool var_9;
    const wp::float32 var_10 = -0.5;
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    bool var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.5;
    bool var_17;
    const wp::float32 var_18 = 0.0;
    const wp::float32 var_19 = 1.0;
    const wp::float32 var_20 = 0.0;
    wp::vec_t<3, wp::float32> var_21;
    const wp::float32 var_22 = 0.0;
    const wp::float32 var_23 = 0.0;
    const wp::float32 var_24 = 1.0;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    const wp::int32 var_29 = 0;
    wp::float32 var_30;
    const wp::int32 var_31 = 1;
    wp::float32 var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    const wp::int32 var_39 = 2;
    wp::float32 var_40;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    const wp::int32 var_43 = 1;
    wp::float32 var_44;
    const wp::int32 var_45 = 2;
    wp::float32 var_46;
    wp::mat_t<3, 3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::float32 var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::float32 var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::vec_t<2, wp::float32> var_55;
    const wp::int32 var_56 = 0;
    wp::float32 var_57;
    const wp::int32 var_58 = 1;
    wp::float32 var_59;
    const wp::int32 var_60 = 2;
    wp::float32 var_61;
    const wp::int32 var_62 = 0;
    wp::float32 var_63;
    const wp::int32 var_64 = 1;
    wp::float32 var_65;
    const wp::int32 var_66 = 2;
    wp::float32 var_67;
    wp::mat_t<2, 3, wp::float32> var_68;
    //---------
    // forward
    // def plane_capsule(                                                                     <L 259>
    // n = plane_normal                                                                       <L 283>
    var_0 = wp::copy(var_plane_normal);
    // axis = capsule_axis                                                                    <L 284>
    var_1 = wp::copy(var_capsule_axis);
    // b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))                            <L 287>
    var_2 = wp::dot(var_0, var_1);
    var_3 = wp::mul(var_0, var_2);
    var_4 = wp::sub(var_1, var_3);
    normalize_with_norm_0(var_4, var_5, var_6);
    // if b_norm < 0.5:                                                                       <L 289>
    var_8 = (var_6 < var_7);
    if (var_8) {
        // if -0.5 < n[1] and n[1] < 0.5:                                                     <L 290>
        var_12 = wp::extract(var_0, var_11);
        var_13 = (var_10 < var_12);
        var_9 = var_13;
        if (var_9) {
            var_15 = wp::extract(var_0, var_14);
            var_17 = (var_15 < var_16);
            var_9 = var_9 && var_17;
        }
        if (var_9) {
            // b = wp.vec3(0.0, 1.0, 0.0)                                                     <L 291>
            var_21 = wp::vec_t<3, wp::float32>(var_18, var_19, var_20);
        }
        if (!var_9) {
            // b = wp.vec3(0.0, 0.0, 1.0)                                                     <L 293>
            var_25 = wp::vec_t<3, wp::float32>(var_22, var_23, var_24);
        }
        var_26 = wp::where(var_9, var_21, var_25);
    }
    var_27 = wp::where(var_8, var_26, var_5);
    // c = wp.cross(n, b)                                                                     <L 295>
    var_28 = wp::cross(var_0, var_27);
    // frame = wp.mat33(n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2])                 <L 296>
    var_30 = wp::extract(var_0, var_29);
    var_32 = wp::extract(var_0, var_31);
    var_34 = wp::extract(var_0, var_33);
    var_36 = wp::extract(var_27, var_35);
    var_38 = wp::extract(var_27, var_37);
    var_40 = wp::extract(var_27, var_39);
    var_42 = wp::extract(var_28, var_41);
    var_44 = wp::extract(var_28, var_43);
    var_46 = wp::extract(var_28, var_45);
    var_47 = wp::mat_t<3, 3, wp::float32>(var_30, var_32, var_34, var_36, var_38, var_40, var_42, var_44, var_46);
    // segment = axis * capsule_half_length                                                   <L 297>
    var_48 = wp::mul(var_1, var_capsule_half_length);
    // dist1, pos1 = plane_sphere(n, plane_pos, capsule_pos + segment, capsule_radius)        <L 300>
    var_49 = wp::add(var_capsule_pos, var_48);
    plane_sphere_0(var_0, var_plane_pos, var_49, var_capsule_radius, var_50, var_51);
    // dist2, pos2 = plane_sphere(n, plane_pos, capsule_pos - segment, capsule_radius)        <L 303>
    var_52 = wp::sub(var_capsule_pos, var_48);
    plane_sphere_0(var_0, var_plane_pos, var_52, var_capsule_radius, var_53, var_54);
    // dist = wp.vec2(dist1, dist2)                                                           <L 305>
    var_55 = wp::vec_t<2, wp::float32>(var_50, var_53);
    // pos = mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2])                     <L 306>
    var_57 = wp::extract(var_51, var_56);
    var_59 = wp::extract(var_51, var_58);
    var_61 = wp::extract(var_51, var_60);
    var_63 = wp::extract(var_54, var_62);
    var_65 = wp::extract(var_54, var_64);
    var_67 = wp::extract(var_54, var_66);
    var_68 = wp::mat_t<2, 3, wp::float32>({var_57, var_59, var_61, var_63, var_65, var_67});
    // return dist, pos, frame                                                                <L 308>
    ret_0 = var_55;
    ret_1 = var_68;
    ret_2 = var_47;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:531
static CUDA_CALLABLE void plane_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_plane,
    Geom_4022bfa7 var_cap,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_adhesion,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32>* var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::vec_t<3, wp::float32>* var_19;
    const wp::int32 var_20 = 0;
    wp::float32 var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32>* var_23;
    const wp::int32 var_24 = 1;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<2, wp::float32> var_27;
    wp::mat_t<2, 3, wp::float32> var_28;
    wp::mat_t<3, 3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    const wp::int32 var_33 = 0;
    wp::float32 var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::int32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::int32 var_40;
    //---------
    // forward
    // def plane_capsule_wrapper(                                                             <L 532>
    // capsule_axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])                    <L 569>
    var_0 = &((var_cap).rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &((var_cap).rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &((var_cap).rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // dist, pos, frame = plane_capsule(                                                      <L 571>
    // plane.normal,                                                                          <L 572>
    var_16 = &((var_plane).normal);
    // plane.pos,                                                                             <L 573>
    var_17 = &((var_plane).pos);
    // cap.pos,                                                                               <L 574>
    var_18 = &((var_cap).pos);
    // capsule_axis,                                                                          <L 575>
    // cap.size[0],  # radius                                                                 <L 576>
    var_19 = &((var_cap).size);
    var_22 = wp::load(var_19);
    var_21 = wp::extract(var_22, var_20);
    // cap.size[1],  # half_length                                                            <L 577>
    var_23 = &((var_cap).size);
    var_26 = wp::load(var_23);
    var_25 = wp::extract(var_26, var_24);
    var_30 = wp::load(var_16);
    var_31 = wp::load(var_17);
    var_32 = wp::load(var_18);
    plane_capsule_0(var_30, var_31, var_32, var_15, var_21, var_25, var_27, var_28, var_29);
    // for i in range(2):                                                                     <L 580>
    // write_contact(                                                                         <L 581>
    // naconmax_in,                                                                           <L 582>
    // i,                                                                                     <L 583>
    // dist[i],                                                                               <L 584>
    var_34 = wp::extract(var_27, var_33);
    // pos[i],                                                                                <L 585>
    var_35 = wp::extract(var_28, var_33);
    // frame,                                                                                 <L 586>
    // margin,                                                                                <L 587>
    // gap,                                                                                   <L 588>
    // condim,                                                                                <L 589>
    // friction,                                                                              <L 590>
    // solref,                                                                                <L 591>
    // solreffriction,                                                                        <L 592>
    // solimp,                                                                                <L 593>
    // adhesion,                                                                              <L 594>
    // geoms,                                                                                 <L 595>
    // pairid,                                                                                <L 596>
    // worldid,                                                                               <L 597>
    // contact_dist_out,                                                                      <L 598>
    // contact_pos_out,                                                                       <L 599>
    // contact_frame_out,                                                                     <L 600>
    // contact_includemargin_out,                                                             <L 601>
    // contact_friction_out,                                                                  <L 602>
    // contact_solref_out,                                                                    <L 603>
    // contact_solreffriction_out,                                                            <L 604>
    // contact_solimp_out,                                                                    <L 605>
    // contact_dim_out,                                                                       <L 606>
    // contact_geom_out,                                                                      <L 607>
    // contact_efc_address_out,                                                               <L 608>
    // contact_worldid_out,                                                                   <L 609>
    // contact_type_out,                                                                      <L 610>
    // contact_geomcollisionid_out,                                                           <L 611>
    // contact_adhesion_out,                                                                  <L 612>
    // nacon_out,                                                                             <L 613>
    var_36 = write_contact_0(var_naconmax_in, var_33, var_34, var_35, var_29, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 581>
    // naconmax_in,                                                                           <L 582>
    // i,                                                                                     <L 583>
    // dist[i],                                                                               <L 584>
    var_38 = wp::extract(var_27, var_37);
    // pos[i],                                                                                <L 585>
    var_39 = wp::extract(var_28, var_37);
    // frame,                                                                                 <L 586>
    // margin,                                                                                <L 587>
    // gap,                                                                                   <L 588>
    // condim,                                                                                <L 589>
    // friction,                                                                              <L 590>
    // solref,                                                                                <L 591>
    // solreffriction,                                                                        <L 592>
    // solimp,                                                                                <L 593>
    // adhesion,                                                                              <L 594>
    // geoms,                                                                                 <L 595>
    // pairid,                                                                                <L 596>
    // worldid,                                                                               <L 597>
    // contact_dist_out,                                                                      <L 598>
    // contact_pos_out,                                                                       <L 599>
    // contact_frame_out,                                                                     <L 600>
    // contact_includemargin_out,                                                             <L 601>
    // contact_friction_out,                                                                  <L 602>
    // contact_solref_out,                                                                    <L 603>
    // contact_solreffriction_out,                                                            <L 604>
    // contact_solimp_out,                                                                    <L 605>
    // contact_dim_out,                                                                       <L 606>
    // contact_geom_out,                                                                      <L 607>
    // contact_efc_address_out,                                                               <L 608>
    // contact_worldid_out,                                                                   <L 609>
    // contact_type_out,                                                                      <L 610>
    // contact_geomcollisionid_out,                                                           <L 611>
    // contact_adhesion_out,                                                                  <L 612>
    // nacon_out,                                                                             <L 613>
    var_40 = write_contact_0(var_naconmax_in, var_37, var_38, var_39, var_29, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:58
static CUDA_CALLABLE void sphere_sphere_0(
    wp::vec_t<3, wp::float32> var_pos1,
    wp::float32 var_radius1,
    wp::vec_t<3, wp::float32> var_pos2,
    wp::float32 var_radius2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    const wp::float32 var_4 = 1.0;
    const wp::float32 var_5 = 0.0;
    const wp::float32 var_6 = 0.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::float32 var_12 = 0.5;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    //---------
    // forward
    // def sphere_sphere(                                                                     <L 59>
    // dir = pos2 - pos1                                                                      <L 79>
    var_0 = wp::sub(var_pos2, var_pos1);
    // dist = wp.length(dir)                                                                  <L 80>
    var_1 = wp::length(var_0);
    // if dist == 0.0:                                                                        <L 81>
    var_3 = (var_1 == var_2);
    if (var_3) {
        // n = wp.vec3(1.0, 0.0, 0.0)                                                         <L 82>
        var_7 = wp::vec_t<3, wp::float32>(var_4, var_5, var_6);
    }
    if (!var_3) {
        // n = dir / dist                                                                     <L 84>
        var_8 = wp::div(var_0, var_1);
    }
    var_9 = wp::where(var_3, var_7, var_8);
    // dist = dist - (radius1 + radius2)                                                      <L 85>
    var_10 = wp::add(var_radius1, var_radius2);
    var_11 = wp::sub(var_1, var_10);
    // pos = pos1 + n * (radius1 + 0.5 * dist)                                                <L 86>
    var_13 = wp::mul(var_12, var_11);
    var_14 = wp::add(var_radius1, var_13);
    var_15 = wp::mul(var_9, var_14);
    var_16 = wp::add(var_pos1, var_15);
    // return dist, pos, n                                                                    <L 87>
    ret_0 = var_11;
    ret_1 = var_16;
    ret_2 = var_9;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:288
static CUDA_CALLABLE void sphere_sphere_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_sphere1,
    Geom_4022bfa7 var_sphere2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_adhesion,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32>* var_0;
    wp::vec_t<3, wp::float32>* var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::vec_t<3, wp::float32>* var_6;
    const wp::int32 var_7 = 0;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 0;
    wp::mat_t<3, 3, wp::float32> var_16;
    wp::int32 var_17;
    //---------
    // forward
    // def sphere_sphere_wrapper(                                                             <L 289>
    // dist, pos, normal = sphere_sphere(sphere1.pos, sphere1.size[0], sphere2.pos, sphere2.size[0])       <L 325>
    var_0 = &((var_sphere1).pos);
    var_1 = &((var_sphere1).size);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = &((var_sphere2).pos);
    var_6 = &((var_sphere2).size);
    var_9 = wp::load(var_6);
    var_8 = wp::extract(var_9, var_7);
    var_13 = wp::load(var_0);
    var_14 = wp::load(var_5);
    sphere_sphere_0(var_13, var_3, var_14, var_8, var_10, var_11, var_12);
    // write_contact(                                                                         <L 327>
    // naconmax_in,                                                                           <L 328>
    // 0,                                                                                     <L 329>
    // dist,                                                                                  <L 330>
    // pos,                                                                                   <L 331>
    // make_frame(normal),                                                                    <L 332>
    var_16 = make_frame_0(var_12);
    // margin,                                                                                <L 333>
    // gap,                                                                                   <L 334>
    // condim,                                                                                <L 335>
    // friction,                                                                              <L 336>
    // solref,                                                                                <L 337>
    // solreffriction,                                                                        <L 338>
    // solimp,                                                                                <L 339>
    // adhesion,                                                                              <L 340>
    // geoms,                                                                                 <L 341>
    // pairid,                                                                                <L 342>
    // worldid,                                                                               <L 343>
    // contact_dist_out,                                                                      <L 344>
    // contact_pos_out,                                                                       <L 345>
    // contact_frame_out,                                                                     <L 346>
    // contact_includemargin_out,                                                             <L 347>
    // contact_friction_out,                                                                  <L 348>
    // contact_solref_out,                                                                    <L 349>
    // contact_solreffriction_out,                                                            <L 350>
    // contact_solimp_out,                                                                    <L 351>
    // contact_dim_out,                                                                       <L 352>
    // contact_geom_out,                                                                      <L 353>
    // contact_efc_address_out,                                                               <L 354>
    // contact_worldid_out,                                                                   <L 355>
    // contact_type_out,                                                                      <L 356>
    // contact_geomcollisionid_out,                                                           <L 357>
    // contact_adhesion_out,                                                                  <L 358>
    // nacon_out,                                                                             <L 359>
    var_17 = write_contact_0(var_naconmax_in, var_15, var_10, var_11, var_16, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:90
static CUDA_CALLABLE void sphere_capsule_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    //---------
    // forward
    // def sphere_capsule(                                                                    <L 91>
    // offset = sphere_pos - capsule_pos                                                      <L 116>
    var_0 = wp::sub(var_sphere_pos, var_capsule_pos);
    // projection = wp.clamp(                                                                 <L 117>
    // wp.dot(capsule_axis, offset),                                                          <L 118>
    var_1 = wp::dot(var_capsule_axis, var_0);
    // -capsule_half_length,                                                                  <L 119>
    var_2 = wp::neg(var_capsule_half_length);
    // capsule_half_length,                                                                   <L 120>
    var_3 = wp::clamp(var_1, var_2, var_capsule_half_length);
    // pt = capsule_pos + capsule_axis * projection                                           <L 122>
    var_4 = wp::mul(var_capsule_axis, var_3);
    var_5 = wp::add(var_capsule_pos, var_4);
    // return sphere_sphere(sphere_pos, sphere_radius, pt, capsule_radius)                    <L 125>
    sphere_sphere_0(var_sphere_pos, var_sphere_radius, var_5, var_capsule_radius, var_6, var_7, var_8);
    ret_0 = var_6;
    ret_1 = var_7;
    ret_2 = var_8;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:363
static CUDA_CALLABLE void sphere_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_sphere,
    Geom_4022bfa7 var_cap,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_adhesion,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32>* var_16;
    wp::vec_t<3, wp::float32>* var_17;
    const wp::int32 var_18 = 0;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32>* var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32>* var_26;
    const wp::int32 var_27 = 1;
    wp::float32 var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::float32 var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 0;
    wp::mat_t<3, 3, wp::float32> var_36;
    wp::int32 var_37;
    //---------
    // forward
    // def sphere_capsule_wrapper(                                                            <L 364>
    // axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])                            <L 401>
    var_0 = &((var_cap).rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &((var_cap).rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &((var_cap).rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // dist, pos, normal = sphere_capsule(sphere.pos, sphere.size[0], cap.pos, axis, cap.size[0], cap.size[1])       <L 403>
    var_16 = &((var_sphere).pos);
    var_17 = &((var_sphere).size);
    var_20 = wp::load(var_17);
    var_19 = wp::extract(var_20, var_18);
    var_21 = &((var_cap).pos);
    var_22 = &((var_cap).size);
    var_25 = wp::load(var_22);
    var_24 = wp::extract(var_25, var_23);
    var_26 = &((var_cap).size);
    var_29 = wp::load(var_26);
    var_28 = wp::extract(var_29, var_27);
    var_33 = wp::load(var_16);
    var_34 = wp::load(var_21);
    sphere_capsule_0(var_33, var_19, var_34, var_15, var_24, var_28, var_30, var_31, var_32);
    // write_contact(                                                                         <L 405>
    // naconmax_in,                                                                           <L 406>
    // 0,                                                                                     <L 407>
    // dist,                                                                                  <L 408>
    // pos,                                                                                   <L 409>
    // make_frame(normal),                                                                    <L 410>
    var_36 = make_frame_0(var_32);
    // margin,                                                                                <L 411>
    // gap,                                                                                   <L 412>
    // condim,                                                                                <L 413>
    // friction,                                                                              <L 414>
    // solref,                                                                                <L 415>
    // solreffriction,                                                                        <L 416>
    // solimp,                                                                                <L 417>
    // adhesion,                                                                              <L 418>
    // geoms,                                                                                 <L 419>
    // pairid,                                                                                <L 420>
    // worldid,                                                                               <L 421>
    // contact_dist_out,                                                                      <L 422>
    // contact_pos_out,                                                                       <L 423>
    // contact_frame_out,                                                                     <L 424>
    // contact_includemargin_out,                                                             <L 425>
    // contact_friction_out,                                                                  <L 426>
    // contact_solref_out,                                                                    <L 427>
    // contact_solreffriction_out,                                                            <L 428>
    // contact_solimp_out,                                                                    <L 429>
    // contact_dim_out,                                                                       <L 430>
    // contact_geom_out,                                                                      <L 431>
    // contact_efc_address_out,                                                               <L 432>
    // contact_worldid_out,                                                                   <L 433>
    // contact_type_out,                                                                      <L 434>
    // contact_geomcollisionid_out,                                                           <L 435>
    // contact_adhesion_out,                                                                  <L 436>
    // nacon_out,                                                                             <L 437>
    var_37 = write_contact_0(var_naconmax_in, var_35, var_30, var_31, var_36, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:128
static CUDA_CALLABLE void capsule_capsule_0(
    wp::vec_t<3, wp::float32> var_cap1_pos,
    wp::vec_t<3, wp::float32> var_cap1_axis,
    wp::float32 var_cap1_radius,
    wp::float32 var_cap1_half_length,
    wp::vec_t<3, wp::float32> var_cap2_pos,
    wp::vec_t<3, wp::float32> var_cap2_axis,
    wp::float32 var_cap2_radius,
    wp::float32 var_cap2_half_length,
    wp::float32 var_margin,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = INFINITY;
    const wp::float32 var_1 = INFINITY;
    const wp::float32 var_2 = INFINITY;
    const wp::float32 var_3 = INFINITY;
    wp::vec_t<2, wp::float32> var_4;
    wp::mat_t<2, 3, wp::float32> var_5;
    wp::mat_t<2, 3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 1e-15;
    bool var_22;
    const wp::float32 var_23 = 1.0;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 1.0;
    bool var_34;
    const wp::float32 var_35 = 1.0;
    wp::float32 var_36;
    wp::float32 var_37;
    const wp::float32 var_38 = -1.0;
    bool var_39;
    const wp::float32 var_40 = -1.0;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::float32 var_47 = 1.0;
    bool var_48;
    const wp::float32 var_49 = 1.0;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = -1.0;
    const wp::float32 var_53 = 1.0;
    wp::float32 var_54;
    const wp::float32 var_55 = -1.0;
    bool var_56;
    const wp::float32 var_57 = -1.0;
    wp::float32 var_58;
    wp::float32 var_59;
    const wp::float32 var_60 = -1.0;
    const wp::float32 var_61 = 1.0;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::float32 var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    bool var_74;
    const wp::int32 var_75 = 0;
    const wp::int32 var_76 = 0;
    const wp::int32 var_77 = 0;
    const wp::int32 var_78 = 0;
    wp::vec_t<3, wp::float32> var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    const wp::float32 var_82 = -1.0;
    const wp::float32 var_83 = 1.0;
    wp::float32 var_84;
    wp::vec_t<3, wp::float32> var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::float32 var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    bool var_90;
    const wp::int32 var_91 = 1;
    wp::int32 var_92;
    wp::int32 var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    const wp::float32 var_97 = -1.0;
    const wp::float32 var_98 = 1.0;
    wp::float32 var_99;
    wp::vec_t<3, wp::float32> var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::float32 var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::vec_t<3, wp::float32> var_104;
    bool var_105;
    const wp::int32 var_106 = 1;
    wp::int32 var_107;
    wp::int32 var_108;
    const wp::int32 var_109 = 2;
    bool var_110;
    wp::vec_t<3, wp::float32> var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    const wp::float32 var_114 = -1.0;
    const wp::float32 var_115 = 1.0;
    wp::float32 var_116;
    wp::vec_t<3, wp::float32> var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::float32 var_119;
    wp::vec_t<3, wp::float32> var_120;
    wp::vec_t<3, wp::float32> var_121;
    bool var_122;
    const wp::int32 var_123 = 1;
    wp::int32 var_124;
    wp::int32 var_125;
    wp::int32 var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::vec_t<3, wp::float32> var_128;
    wp::float32 var_129;
    wp::vec_t<3, wp::float32> var_130;
    wp::vec_t<3, wp::float32> var_131;
    const wp::int32 var_132 = 2;
    bool var_133;
    wp::vec_t<3, wp::float32> var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    const wp::float32 var_137 = -1.0;
    const wp::float32 var_138 = 1.0;
    wp::float32 var_139;
    wp::vec_t<3, wp::float32> var_140;
    wp::vec_t<3, wp::float32> var_141;
    wp::float32 var_142;
    wp::vec_t<3, wp::float32> var_143;
    wp::vec_t<3, wp::float32> var_144;
    bool var_145;
    wp::vec_t<3, wp::float32> var_146;
    wp::vec_t<3, wp::float32> var_147;
    wp::float32 var_148;
    wp::vec_t<3, wp::float32> var_149;
    wp::vec_t<3, wp::float32> var_150;
    wp::float32 var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    wp::vec_t<3, wp::float32> var_154;
    wp::vec_t<3, wp::float32> var_155;
    wp::float32 var_156;
    wp::vec_t<3, wp::float32> var_157;
    wp::vec_t<3, wp::float32> var_158;
    //---------
    // forward
    // def capsule_capsule(                                                                   <L 129>
    // contact_dist = wp.vec2(wp.inf, wp.inf)                                                 <L 159>
    var_4 = wp::vec_t<2, wp::float32>(var_1, var_3);
    // contact_pos = mat23f()                                                                 <L 160>
    var_5 = wp::mat_t<2, 3, wp::float32>();
    // contact_normal = mat23f()                                                              <L 161>
    var_6 = wp::mat_t<2, 3, wp::float32>();
    // axis1 = cap1_axis * cap1_half_length                                                   <L 164>
    var_7 = wp::mul(var_cap1_axis, var_cap1_half_length);
    // axis2 = cap2_axis * cap2_half_length                                                   <L 165>
    var_8 = wp::mul(var_cap2_axis, var_cap2_half_length);
    // dif = cap1_pos - cap2_pos                                                              <L 166>
    var_9 = wp::sub(var_cap1_pos, var_cap2_pos);
    // ma = wp.dot(axis1, axis1)                                                              <L 169>
    var_10 = wp::dot(var_7, var_7);
    // mb = -wp.dot(axis1, axis2)                                                             <L 170>
    var_11 = wp::dot(var_7, var_8);
    var_12 = wp::neg(var_11);
    // mc = wp.dot(axis2, axis2)                                                              <L 171>
    var_13 = wp::dot(var_8, var_8);
    // u = -wp.dot(axis1, dif)                                                                <L 172>
    var_14 = wp::dot(var_7, var_9);
    var_15 = wp::neg(var_14);
    // v = wp.dot(axis2, dif)                                                                 <L 173>
    var_16 = wp::dot(var_8, var_9);
    // det = ma * mc - mb * mb                                                                <L 174>
    var_17 = wp::mul(var_10, var_13);
    var_18 = wp::mul(var_12, var_12);
    var_19 = wp::sub(var_17, var_18);
    // if wp.abs(det) >= MJ_MINVAL:                                                           <L 177>
    var_20 = wp::abs(var_19);
    var_22 = (var_20 >= var_21);
    if (var_22) {
        // inv_det = 1.0 / det                                                                <L 178>
        var_24 = wp::div(var_23, var_19);
        // x1 = (mc * u - mb * v) * inv_det                                                   <L 179>
        var_25 = wp::mul(var_13, var_15);
        var_26 = wp::mul(var_12, var_16);
        var_27 = wp::sub(var_25, var_26);
        var_28 = wp::mul(var_27, var_24);
        // x2 = (ma * v - mb * u) * inv_det                                                   <L 180>
        var_29 = wp::mul(var_10, var_16);
        var_30 = wp::mul(var_12, var_15);
        var_31 = wp::sub(var_29, var_30);
        var_32 = wp::mul(var_31, var_24);
        // if x1 > 1.0:                                                                       <L 182>
        var_34 = (var_28 > var_33);
        if (var_34) {
            // x1 = 1.0                                                                       <L 183>
            // x2 = (v - mb) / mc                                                             <L 184>
            var_36 = wp::sub(var_16, var_12);
            var_37 = wp::div(var_36, var_13);
        }
        if (!var_34) {
            // elif x1 < -1.0:                                                                <L 185>
            var_39 = (var_28 < var_38);
            if (var_39) {
                // x1 = -1.0                                                                  <L 186>
                // x2 = (v + mb) / mc                                                         <L 187>
                var_41 = wp::add(var_16, var_12);
                var_42 = wp::div(var_41, var_13);
            }
            var_43 = wp::where(var_39, var_40, var_28);
            var_44 = wp::where(var_39, var_42, var_32);
        }
        var_45 = wp::where(var_34, var_35, var_43);
        var_46 = wp::where(var_34, var_37, var_44);
        // if x2 > 1.0:                                                                       <L 189>
        var_48 = (var_46 > var_47);
        if (var_48) {
            // x2 = 1.0                                                                       <L 190>
            // x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)                                        <L 191>
            var_50 = wp::sub(var_15, var_12);
            var_51 = wp::div(var_50, var_10);
            var_54 = wp::clamp(var_51, var_52, var_53);
        }
        if (!var_48) {
            // elif x2 < -1.0:                                                                <L 192>
            var_56 = (var_46 < var_55);
            if (var_56) {
                // x2 = -1.0                                                                  <L 193>
                // x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)                                    <L 194>
                var_58 = wp::add(var_15, var_12);
                var_59 = wp::div(var_58, var_10);
                var_62 = wp::clamp(var_59, var_60, var_61);
            }
            var_63 = wp::where(var_56, var_62, var_45);
            var_64 = wp::where(var_56, var_57, var_46);
        }
        var_65 = wp::where(var_48, var_54, var_63);
        var_66 = wp::where(var_48, var_49, var_64);
        // vec1 = cap1_pos + axis1 * x1                                                       <L 197>
        var_67 = wp::mul(var_7, var_65);
        var_68 = wp::add(var_cap1_pos, var_67);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 198>
        var_69 = wp::mul(var_8, var_66);
        var_70 = wp::add(var_cap2_pos, var_69);
        // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)            <L 200>
        sphere_sphere_0(var_68, var_cap1_radius, var_70, var_cap2_radius, var_71, var_72, var_73);
        // if dist <= margin:                                                                 <L 201>
        var_74 = (var_71 <= var_margin);
        if (var_74) {
            // contact_dist[0] = dist                                                         <L 202>
            wp::assign_inplace(var_4, var_75, var_71);
            // contact_pos[0] = pos                                                           <L 203>
            wp::assign_inplace(var_5, var_76, var_72);
            // contact_normal[0] = normal                                                     <L 204>
            wp::assign_inplace(var_6, var_77, var_73);
        }
    }
    if (!var_22) {
        // contact_count = 0                                                                  <L 208>
        // vec1 = cap1_pos + axis1                                                            <L 211>
        var_79 = wp::add(var_cap1_pos, var_7);
        // x2 = wp.clamp((v - mb) / mc, -1.0, 1.0)                                            <L 212>
        var_80 = wp::sub(var_16, var_12);
        var_81 = wp::div(var_80, var_13);
        var_84 = wp::clamp(var_81, var_82, var_83);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 213>
        var_85 = wp::mul(var_8, var_84);
        var_86 = wp::add(var_cap2_pos, var_85);
        // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)            <L 214>
        sphere_sphere_0(var_79, var_cap1_radius, var_86, var_cap2_radius, var_87, var_88, var_89);
        // if dist <= margin:                                                                 <L 215>
        var_90 = (var_87 <= var_margin);
        if (var_90) {
            // contact_dist[contact_count] = dist                                             <L 216>
            wp::assign_inplace(var_4, var_78, var_87);
            // contact_pos[contact_count] = pos                                               <L 217>
            wp::assign_inplace(var_5, var_78, var_88);
            // contact_normal[contact_count] = normal                                         <L 218>
            wp::assign_inplace(var_6, var_78, var_89);
            // contact_count += 1                                                             <L 219>
            var_92 = wp::add(var_78, var_91);
        }
        var_93 = wp::where(var_90, var_92, var_78);
        // vec1 = cap1_pos - axis1                                                            <L 222>
        var_94 = wp::sub(var_cap1_pos, var_7);
        // x2 = wp.clamp((v + mb) / mc, -1.0, 1.0)                                            <L 223>
        var_95 = wp::add(var_16, var_12);
        var_96 = wp::div(var_95, var_13);
        var_99 = wp::clamp(var_96, var_97, var_98);
        // vec2 = cap2_pos + axis2 * x2                                                       <L 224>
        var_100 = wp::mul(var_8, var_99);
        var_101 = wp::add(var_cap2_pos, var_100);
        // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)            <L 225>
        sphere_sphere_0(var_94, var_cap1_radius, var_101, var_cap2_radius, var_102, var_103, var_104);
        // if dist <= margin:                                                                 <L 226>
        var_105 = (var_102 <= var_margin);
        if (var_105) {
            // contact_dist[contact_count] = dist                                             <L 227>
            wp::assign_inplace(var_4, var_93, var_102);
            // contact_pos[contact_count] = pos                                               <L 228>
            wp::assign_inplace(var_5, var_93, var_103);
            // contact_normal[contact_count] = normal                                         <L 229>
            wp::assign_inplace(var_6, var_93, var_104);
            // contact_count += 1                                                             <L 230>
            var_107 = wp::add(var_93, var_106);
        }
        var_108 = wp::where(var_105, var_107, var_93);
        // if contact_count < 2:                                                              <L 233>
        var_110 = (var_108 < var_109);
        if (var_110) {
            // vec2 = cap2_pos + axis2                                                        <L 234>
            var_111 = wp::add(var_cap2_pos, var_8);
            // x1 = wp.clamp((u - mb) / ma, -1.0, 1.0)                                        <L 235>
            var_112 = wp::sub(var_15, var_12);
            var_113 = wp::div(var_112, var_10);
            var_116 = wp::clamp(var_113, var_114, var_115);
            // vec1 = cap1_pos + axis1 * x1                                                   <L 236>
            var_117 = wp::mul(var_7, var_116);
            var_118 = wp::add(var_cap1_pos, var_117);
            // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)        <L 237>
            sphere_sphere_0(var_118, var_cap1_radius, var_111, var_cap2_radius, var_119, var_120, var_121);
            // if dist <= margin:                                                             <L 238>
            var_122 = (var_119 <= var_margin);
            if (var_122) {
                // contact_dist[contact_count] = dist                                         <L 239>
                wp::assign_inplace(var_4, var_108, var_119);
                // contact_pos[contact_count] = pos                                           <L 240>
                wp::assign_inplace(var_5, var_108, var_120);
                // contact_normal[contact_count] = normal                                     <L 241>
                wp::assign_inplace(var_6, var_108, var_121);
                // contact_count += 1                                                         <L 242>
                var_124 = wp::add(var_108, var_123);
            }
            var_125 = wp::where(var_122, var_124, var_108);
        }
        var_126 = wp::where(var_110, var_125, var_108);
        var_127 = wp::where(var_110, var_118, var_94);
        var_128 = wp::where(var_110, var_111, var_101);
        var_129 = wp::where(var_110, var_119, var_102);
        var_130 = wp::where(var_110, var_120, var_103);
        var_131 = wp::where(var_110, var_121, var_104);
        // if contact_count < 2:                                                              <L 245>
        var_133 = (var_126 < var_132);
        if (var_133) {
            // vec2 = cap2_pos - axis2                                                        <L 246>
            var_134 = wp::sub(var_cap2_pos, var_8);
            // x1 = wp.clamp((u + mb) / ma, -1.0, 1.0)                                        <L 247>
            var_135 = wp::add(var_15, var_12);
            var_136 = wp::div(var_135, var_10);
            var_139 = wp::clamp(var_136, var_137, var_138);
            // vec1 = cap1_pos + axis1 * x1                                                   <L 248>
            var_140 = wp::mul(var_7, var_139);
            var_141 = wp::add(var_cap1_pos, var_140);
            // dist, pos, normal = sphere_sphere(vec1, cap1_radius, vec2, cap2_radius)        <L 249>
            sphere_sphere_0(var_141, var_cap1_radius, var_134, var_cap2_radius, var_142, var_143, var_144);
            // if dist <= margin:                                                             <L 250>
            var_145 = (var_142 <= var_margin);
            if (var_145) {
                // contact_dist[contact_count] = dist                                         <L 251>
                wp::assign_inplace(var_4, var_126, var_142);
                // contact_pos[contact_count] = pos                                           <L 252>
                wp::assign_inplace(var_5, var_126, var_143);
                // contact_normal[contact_count] = normal                                     <L 253>
                wp::assign_inplace(var_6, var_126, var_144);
            }
        }
        var_146 = wp::where(var_133, var_141, var_127);
        var_147 = wp::where(var_133, var_134, var_128);
        var_148 = wp::where(var_133, var_142, var_129);
        var_149 = wp::where(var_133, var_143, var_130);
        var_150 = wp::where(var_133, var_144, var_131);
        var_151 = wp::where(var_133, var_139, var_116);
    }
    var_152 = wp::where(var_22, var_65, var_151);
    var_153 = wp::where(var_22, var_66, var_99);
    var_154 = wp::where(var_22, var_68, var_146);
    var_155 = wp::where(var_22, var_70, var_147);
    var_156 = wp::where(var_22, var_71, var_148);
    var_157 = wp::where(var_22, var_72, var_149);
    var_158 = wp::where(var_22, var_73, var_150);
    // return contact_dist, contact_pos, contact_normal                                       <L 255>
    ret_0 = var_4;
    ret_1 = var_5;
    ret_2 = var_6;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:441
static CUDA_CALLABLE void capsule_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_cap1,
    Geom_4022bfa7 var_cap2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_adhesion,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32>* var_0;
    const wp::int32 var_1 = 0;
    const wp::int32 var_2 = 2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32> var_4;
    wp::mat_t<3, 3, wp::float32>* var_5;
    const wp::int32 var_6 = 1;
    const wp::int32 var_7 = 2;
    wp::float32 var_8;
    wp::mat_t<3, 3, wp::float32> var_9;
    wp::mat_t<3, 3, wp::float32>* var_10;
    const wp::int32 var_11 = 2;
    const wp::int32 var_12 = 2;
    wp::float32 var_13;
    wp::mat_t<3, 3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::mat_t<3, 3, wp::float32>* var_16;
    const wp::int32 var_17 = 0;
    const wp::int32 var_18 = 2;
    wp::float32 var_19;
    wp::mat_t<3, 3, wp::float32> var_20;
    wp::mat_t<3, 3, wp::float32>* var_21;
    const wp::int32 var_22 = 1;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    wp::mat_t<3, 3, wp::float32> var_25;
    wp::mat_t<3, 3, wp::float32>* var_26;
    const wp::int32 var_27 = 2;
    const wp::int32 var_28 = 2;
    wp::float32 var_29;
    wp::mat_t<3, 3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32>* var_32;
    wp::vec_t<3, wp::float32>* var_33;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32>* var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32>* var_41;
    wp::vec_t<3, wp::float32>* var_42;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32>* var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<2, wp::float32> var_50;
    wp::mat_t<2, 3, wp::float32> var_51;
    wp::mat_t<2, 3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    const wp::int32 var_55 = 0;
    wp::float32 var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    const wp::int32 var_64 = 0;
    wp::float32 var_65;
    const wp::int32 var_66 = 1;
    wp::float32 var_67;
    const wp::int32 var_68 = 2;
    wp::float32 var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::mat_t<3, 3, wp::float32> var_71;
    wp::int32 var_72;
    const wp::int32 var_73 = 1;
    wp::float32 var_74;
    const wp::int32 var_75 = 0;
    wp::float32 var_76;
    const wp::int32 var_77 = 1;
    wp::float32 var_78;
    const wp::int32 var_79 = 2;
    wp::float32 var_80;
    wp::vec_t<3, wp::float32> var_81;
    const wp::int32 var_82 = 0;
    wp::float32 var_83;
    const wp::int32 var_84 = 1;
    wp::float32 var_85;
    const wp::int32 var_86 = 2;
    wp::float32 var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::mat_t<3, 3, wp::float32> var_89;
    wp::int32 var_90;
    //---------
    // forward
    // def capsule_capsule_wrapper(                                                           <L 442>
    // cap1_axis = wp.vec3(cap1.rot[0, 2], cap1.rot[1, 2], cap1.rot[2, 2])                    <L 479>
    var_0 = &((var_cap1).rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &((var_cap1).rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &((var_cap1).rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // cap2_axis = wp.vec3(cap2.rot[0, 2], cap2.rot[1, 2], cap2.rot[2, 2])                    <L 480>
    var_16 = &((var_cap2).rot);
    var_20 = wp::load(var_16);
    var_19 = wp::extract(var_20, var_17, var_18);
    var_21 = &((var_cap2).rot);
    var_25 = wp::load(var_21);
    var_24 = wp::extract(var_25, var_22, var_23);
    var_26 = &((var_cap2).rot);
    var_30 = wp::load(var_26);
    var_29 = wp::extract(var_30, var_27, var_28);
    var_31 = wp::vec_t<3, wp::float32>(var_19, var_24, var_29);
    // dist, pos, normal = capsule_capsule(                                                   <L 482>
    // cap1.pos,                                                                              <L 483>
    var_32 = &((var_cap1).pos);
    // cap1_axis,                                                                             <L 484>
    // cap1.size[0],  # radius1                                                               <L 485>
    var_33 = &((var_cap1).size);
    var_36 = wp::load(var_33);
    var_35 = wp::extract(var_36, var_34);
    // cap1.size[1],  # half_length1                                                          <L 486>
    var_37 = &((var_cap1).size);
    var_40 = wp::load(var_37);
    var_39 = wp::extract(var_40, var_38);
    // cap2.pos,                                                                              <L 487>
    var_41 = &((var_cap2).pos);
    // cap2_axis,                                                                             <L 488>
    // cap2.size[0],  # radius2                                                               <L 489>
    var_42 = &((var_cap2).size);
    var_45 = wp::load(var_42);
    var_44 = wp::extract(var_45, var_43);
    // cap2.size[1],  # half_length2                                                          <L 490>
    var_46 = &((var_cap2).size);
    var_49 = wp::load(var_46);
    var_48 = wp::extract(var_49, var_47);
    // margin,                                                                                <L 491>
    var_53 = wp::load(var_32);
    var_54 = wp::load(var_41);
    capsule_capsule_0(var_53, var_15, var_35, var_39, var_54, var_31, var_44, var_48, var_margin, var_50, var_51, var_52);
    // for i in range(2):                                                                     <L 494>
    // write_contact(                                                                         <L 495>
    // naconmax_in,                                                                           <L 496>
    // i,                                                                                     <L 497>
    // dist[i],                                                                               <L 498>
    var_56 = wp::extract(var_50, var_55);
    // wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2]),                                              <L 499>
    var_58 = wp::extract(var_51, var_55, var_57);
    var_60 = wp::extract(var_51, var_55, var_59);
    var_62 = wp::extract(var_51, var_55, var_61);
    var_63 = wp::vec_t<3, wp::float32>(var_58, var_60, var_62);
    // make_frame(wp.vec3(normal[i, 0], normal[i, 1], normal[i, 2])),                         <L 500>
    var_65 = wp::extract(var_52, var_55, var_64);
    var_67 = wp::extract(var_52, var_55, var_66);
    var_69 = wp::extract(var_52, var_55, var_68);
    var_70 = wp::vec_t<3, wp::float32>(var_65, var_67, var_69);
    var_71 = make_frame_0(var_70);
    // margin,                                                                                <L 501>
    // gap,                                                                                   <L 502>
    // condim,                                                                                <L 503>
    // friction,                                                                              <L 504>
    // solref,                                                                                <L 505>
    // solreffriction,                                                                        <L 506>
    // solimp,                                                                                <L 507>
    // adhesion,                                                                              <L 508>
    // geoms,                                                                                 <L 509>
    // pairid,                                                                                <L 510>
    // worldid,                                                                               <L 511>
    // contact_dist_out,                                                                      <L 512>
    // contact_pos_out,                                                                       <L 513>
    // contact_frame_out,                                                                     <L 514>
    // contact_includemargin_out,                                                             <L 515>
    // contact_friction_out,                                                                  <L 516>
    // contact_solref_out,                                                                    <L 517>
    // contact_solreffriction_out,                                                            <L 518>
    // contact_solimp_out,                                                                    <L 519>
    // contact_dim_out,                                                                       <L 520>
    // contact_geom_out,                                                                      <L 521>
    // contact_efc_address_out,                                                               <L 522>
    // contact_worldid_out,                                                                   <L 523>
    // contact_type_out,                                                                      <L 524>
    // contact_geomcollisionid_out,                                                           <L 525>
    // contact_adhesion_out,                                                                  <L 526>
    // nacon_out,                                                                             <L 527>
    var_72 = write_contact_0(var_naconmax_in, var_55, var_56, var_63, var_71, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 495>
    // naconmax_in,                                                                           <L 496>
    // i,                                                                                     <L 497>
    // dist[i],                                                                               <L 498>
    var_74 = wp::extract(var_50, var_73);
    // wp.vec3(pos[i, 0], pos[i, 1], pos[i, 2]),                                              <L 499>
    var_76 = wp::extract(var_51, var_73, var_75);
    var_78 = wp::extract(var_51, var_73, var_77);
    var_80 = wp::extract(var_51, var_73, var_79);
    var_81 = wp::vec_t<3, wp::float32>(var_76, var_78, var_80);
    // make_frame(wp.vec3(normal[i, 0], normal[i, 1], normal[i, 2])),                         <L 500>
    var_83 = wp::extract(var_52, var_73, var_82);
    var_85 = wp::extract(var_52, var_73, var_84);
    var_87 = wp::extract(var_52, var_73, var_86);
    var_88 = wp::vec_t<3, wp::float32>(var_83, var_85, var_87);
    var_89 = make_frame_0(var_88);
    // margin,                                                                                <L 501>
    // gap,                                                                                   <L 502>
    // condim,                                                                                <L 503>
    // friction,                                                                              <L 504>
    // solref,                                                                                <L 505>
    // solreffriction,                                                                        <L 506>
    // solimp,                                                                                <L 507>
    // adhesion,                                                                              <L 508>
    // geoms,                                                                                 <L 509>
    // pairid,                                                                                <L 510>
    // worldid,                                                                               <L 511>
    // contact_dist_out,                                                                      <L 512>
    // contact_pos_out,                                                                       <L 513>
    // contact_frame_out,                                                                     <L 514>
    // contact_includemargin_out,                                                             <L 515>
    // contact_friction_out,                                                                  <L 516>
    // contact_solref_out,                                                                    <L 517>
    // contact_solreffriction_out,                                                            <L 518>
    // contact_solimp_out,                                                                    <L 519>
    // contact_dim_out,                                                                       <L 520>
    // contact_geom_out,                                                                      <L 521>
    // contact_efc_address_out,                                                               <L 522>
    // contact_worldid_out,                                                                   <L 523>
    // contact_type_out,                                                                      <L 524>
    // contact_geomcollisionid_out,                                                           <L 525>
    // contact_adhesion_out,                                                                  <L 526>
    // nacon_out,                                                                             <L 527>
    var_90 = write_contact_0(var_naconmax_in, var_73, var_74, var_81, var_89, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:294
static CUDA_CALLABLE void adj_contact_margin_gap_0(
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_pairid,
    wp::int32 var_worldid,
    wp::float32 & ret_0,
    wp::float32 & ret_1,
    wp::array_t<wp::float32> & adj_geom_margin,
    wp::array_t<wp::float32> & adj_geom_gap,
    wp::array_t<wp::float32> & adj_pair_margin,
    wp::array_t<wp::float32> & adj_pair_gap,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_pairid,
    wp::int32 & adj_worldid,
    wp::float32 & adj_ret_0,
    wp::float32 & adj_ret_1)
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:320
static CUDA_CALLABLE void adj_contact_material_params_0(
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_adhesion,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_adhesion,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_pairid,
    wp::int32 var_worldid,
    wp::int32 & ret_0,
    wp::vec_t<5, wp::float32> & ret_1,
    wp::vec_t<2, wp::float32> & ret_2,
    wp::vec_t<2, wp::float32> & ret_3,
    wp::vec_t<5, wp::float32> & ret_4,
    wp::float32 & ret_5,
    wp::array_t<wp::int32> & adj_geom_condim,
    wp::array_t<wp::int32> & adj_geom_priority,
    wp::array_t<wp::float32> & adj_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_friction,
    wp::array_t<wp::float32> & adj_geom_adhesion,
    wp::array_t<wp::int32> & adj_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_pair_solimp,
    wp::array_t<wp::float32> & adj_pair_adhesion,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_pair_friction,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_pairid,
    wp::int32 & adj_worldid,
    wp::int32 & adj_ret_0,
    wp::vec_t<5, wp::float32> & adj_ret_1,
    wp::vec_t<2, wp::float32> & adj_ret_2,
    wp::vec_t<2, wp::float32> & adj_ret_3,
    wp::vec_t<5, wp::float32> & adj_ret_4,
    wp::float32 & adj_ret_5)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:416
static CUDA_CALLABLE void adj_contact_params_0(
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::float32> var_geom_adhesion,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::array_t<wp::float32> var_pair_adhesion,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::int32 var_cid,
    wp::int32 var_worldid,
    wp::vec_t<2, wp::int32> & ret_0,
    wp::float32 & ret_1,
    wp::float32 & ret_2,
    wp::int32 & ret_3,
    wp::vec_t<5, wp::float32> & ret_4,
    wp::vec_t<2, wp::float32> & ret_5,
    wp::vec_t<2, wp::float32> & ret_6,
    wp::vec_t<5, wp::float32> & ret_7,
    wp::float32 & ret_8,
    wp::array_t<wp::int32> & adj_geom_condim,
    wp::array_t<wp::int32> & adj_geom_priority,
    wp::array_t<wp::float32> & adj_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_friction,
    wp::array_t<wp::float32> & adj_geom_margin,
    wp::array_t<wp::float32> & adj_geom_gap,
    wp::array_t<wp::float32> & adj_geom_adhesion,
    wp::array_t<wp::int32> & adj_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_pair_solimp,
    wp::array_t<wp::float32> & adj_pair_margin,
    wp::array_t<wp::float32> & adj_pair_gap,
    wp::array_t<wp::float32> & adj_pair_adhesion,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_pair_friction,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_collision_pairid_in,
    wp::int32 & adj_cid,
    wp::int32 & adj_worldid,
    wp::vec_t<2, wp::int32> & adj_ret_0,
    wp::float32 & adj_ret_1,
    wp::float32 & adj_ret_2,
    wp::int32 & adj_ret_3,
    wp::vec_t<5, wp::float32> & adj_ret_4,
    wp::vec_t<2, wp::float32> & adj_ret_5,
    wp::vec_t<2, wp::float32> & adj_ret_6,
    wp::vec_t<5, wp::float32> & adj_ret_7,
    wp::float32 & adj_ret_8)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:65
static CUDA_CALLABLE void adj_geom_collision_pair_from_types_0(
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_geom_type1,
    wp::int32 var_geom_type2,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    Geom_4022bfa7 & ret_0,
    Geom_4022bfa7 & ret_1,
    wp::array_t<wp::int32> & adj_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_size,
    wp::array_t<wp::int32> & adj_mesh_vertadr,
    wp::array_t<wp::int32> & adj_mesh_vertnum,
    wp::array_t<wp::int32> & adj_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_vert,
    wp::array_t<wp::int32> & adj_mesh_graph,
    wp::array_t<wp::int32> & adj_mesh_polynum,
    wp::array_t<wp::int32> & adj_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_polynormal,
    wp::array_t<wp::int32> & adj_mesh_polyvertadr,
    wp::array_t<wp::int32> & adj_mesh_polyvertnum,
    wp::array_t<wp::int32> & adj_mesh_polyvert,
    wp::array_t<wp::int32> & adj_mesh_polymapadr,
    wp::array_t<wp::int32> & adj_mesh_polymapnum,
    wp::array_t<wp::int32> & adj_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::int32 & adj_geom_type1,
    wp::int32 & adj_geom_type2,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_worldid,
    Geom_4022bfa7 & adj_ret_0,
    Geom_4022bfa7 & adj_ret_1)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:158
static CUDA_CALLABLE void adj_geom_collision_pair_0(
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::int32 var_worldid,
    Geom_4022bfa7 & ret_0,
    Geom_4022bfa7 & ret_1,
    wp::array_t<wp::int32> & adj_geom_type,
    wp::array_t<wp::int32> & adj_geom_dataid,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_size,
    wp::array_t<wp::int32> & adj_mesh_vertadr,
    wp::array_t<wp::int32> & adj_mesh_vertnum,
    wp::array_t<wp::int32> & adj_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_vert,
    wp::array_t<wp::int32> & adj_mesh_graph,
    wp::array_t<wp::int32> & adj_mesh_polynum,
    wp::array_t<wp::int32> & adj_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_mesh_polynormal,
    wp::array_t<wp::int32> & adj_mesh_polyvertadr,
    wp::array_t<wp::int32> & adj_mesh_polyvertnum,
    wp::array_t<wp::int32> & adj_mesh_polyvert,
    wp::array_t<wp::int32> & adj_mesh_polymapadr,
    wp::array_t<wp::int32> & adj_mesh_polymapnum,
    wp::array_t<wp::int32> & adj_mesh_polymap,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_geom_xmat_in,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::int32 & adj_worldid,
    Geom_4022bfa7 & adj_ret_0,
    Geom_4022bfa7 & adj_ret_1)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:50
static CUDA_CALLABLE void adj_plane_sphere_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:202
static CUDA_CALLABLE void adj_orthogonals_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:246
static CUDA_CALLABLE void adj_make_frame_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_core.py:213
static CUDA_CALLABLE void adj_write_contact_0(
    wp::int32 var_naconmax_in,
    wp::int32 var_id_,
    wp::float32 var_dist_in,
    wp::vec_t<3, wp::float32> var_pos_in,
    wp::mat_t<3, 3, wp::float32> var_frame_in,
    wp::float32 var_margin_in,
    wp::float32 var_gap_in,
    wp::int32 var_condim_in,
    wp::vec_t<5, wp::float32> var_friction_in,
    wp::vec_t<2, wp::float32> var_solref_in,
    wp::vec_t<2, wp::float32> var_solreffriction_in,
    wp::vec_t<5, wp::float32> var_solimp_in,
    wp::float32 var_adhesion_in,
    wp::vec_t<2, wp::int32> var_geoms_in,
    wp::vec_t<2, wp::int32> var_pairid_in,
    wp::int32 var_worldid_in,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    wp::int32 & adj_id_,
    wp::float32 & adj_dist_in,
    wp::vec_t<3, wp::float32> & adj_pos_in,
    wp::mat_t<3, 3, wp::float32> & adj_frame_in,
    wp::float32 & adj_margin_in,
    wp::float32 & adj_gap_in,
    wp::int32 & adj_condim_in,
    wp::vec_t<5, wp::float32> & adj_friction_in,
    wp::vec_t<2, wp::float32> & adj_solref_in,
    wp::vec_t<2, wp::float32> & adj_solreffriction_in,
    wp::vec_t<5, wp::float32> & adj_solimp_in,
    wp::float32 & adj_adhesion_in,
    wp::vec_t<2, wp::int32> & adj_geoms_in,
    wp::vec_t<2, wp::int32> & adj_pairid_in,
    wp::int32 & adj_worldid_in,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::float32> & adj_contact_adhesion_out,
    wp::array_t<wp::int32> & adj_nacon_out,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:212
static CUDA_CALLABLE void adj_plane_sphere_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_plane,
    Geom_4022bfa7 var_sphere,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_adhesion,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_4022bfa7 & adj_plane,
    Geom_4022bfa7 & adj_sphere,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::float32 & adj_adhesion,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::float32> & adj_contact_adhesion_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\math.py:0
static CUDA_CALLABLE void adj_normalize_with_norm_0(
    wp::vec_t<3, wp::float32> var_x,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::float32 & ret_1,
    wp::vec_t<3, wp::float32> & adj_x,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::float32 & adj_ret_1)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:258
static CUDA_CALLABLE void adj_plane_capsule_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<3, 3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_axis,
    wp::float32 & adj_capsule_radius,
    wp::float32 & adj_capsule_half_length,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::mat_t<2, 3, wp::float32> & adj_ret_1,
    wp::mat_t<3, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:531
static CUDA_CALLABLE void adj_plane_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_plane,
    Geom_4022bfa7 var_cap,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_adhesion,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_4022bfa7 & adj_plane,
    Geom_4022bfa7 & adj_cap,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::float32 & adj_adhesion,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::float32> & adj_contact_adhesion_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:58
static CUDA_CALLABLE void adj_sphere_sphere_0(
    wp::vec_t<3, wp::float32> var_pos1,
    wp::float32 var_radius1,
    wp::vec_t<3, wp::float32> var_pos2,
    wp::float32 var_radius2,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_pos1,
    wp::float32 & adj_radius1,
    wp::vec_t<3, wp::float32> & adj_pos2,
    wp::float32 & adj_radius2,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:288
static CUDA_CALLABLE void adj_sphere_sphere_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_sphere1,
    Geom_4022bfa7 var_sphere2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_adhesion,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_4022bfa7 & adj_sphere1,
    Geom_4022bfa7 & adj_sphere2,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::float32 & adj_adhesion,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::float32> & adj_contact_adhesion_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:90
static CUDA_CALLABLE void adj_sphere_capsule_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_capsule_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_axis,
    wp::float32 & adj_capsule_radius,
    wp::float32 & adj_capsule_half_length,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:363
static CUDA_CALLABLE void adj_sphere_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_sphere,
    Geom_4022bfa7 var_cap,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_adhesion,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_4022bfa7 & adj_sphere,
    Geom_4022bfa7 & adj_cap,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::float32 & adj_adhesion,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::float32> & adj_contact_adhesion_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:128
static CUDA_CALLABLE void adj_capsule_capsule_0(
    wp::vec_t<3, wp::float32> var_cap1_pos,
    wp::vec_t<3, wp::float32> var_cap1_axis,
    wp::float32 var_cap1_radius,
    wp::float32 var_cap1_half_length,
    wp::vec_t<3, wp::float32> var_cap2_pos,
    wp::vec_t<3, wp::float32> var_cap2_axis,
    wp::float32 var_cap2_radius,
    wp::float32 var_cap2_half_length,
    wp::float32 var_margin,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_cap1_pos,
    wp::vec_t<3, wp::float32> & adj_cap1_axis,
    wp::float32 & adj_cap1_radius,
    wp::float32 & adj_cap1_half_length,
    wp::vec_t<3, wp::float32> & adj_cap2_pos,
    wp::vec_t<3, wp::float32> & adj_cap2_axis,
    wp::float32 & adj_cap2_radius,
    wp::float32 & adj_cap2_half_length,
    wp::float32 & adj_margin,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::mat_t<2, 3, wp::float32> & adj_ret_1,
    wp::mat_t<2, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:441
static CUDA_CALLABLE void adj_capsule_capsule_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_cap1,
    Geom_4022bfa7 var_cap2,
    wp::int32 var_worldid,
    wp::float32 var_margin,
    wp::float32 var_gap,
    wp::int32 var_condim,
    wp::vec_t<5, wp::float32> var_friction,
    wp::vec_t<2, wp::float32> var_solref,
    wp::vec_t<2, wp::float32> var_solreffriction,
    wp::vec_t<5, wp::float32> var_solimp,
    wp::float32 var_adhesion,
    wp::vec_t<2, wp::int32> var_geoms,
    wp::vec_t<2, wp::int32> var_pairid,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out,
    wp::int32 & adj_naconmax_in,
    Geom_4022bfa7 & adj_cap1,
    Geom_4022bfa7 & adj_cap2,
    wp::int32 & adj_worldid,
    wp::float32 & adj_margin,
    wp::float32 & adj_gap,
    wp::int32 & adj_condim,
    wp::vec_t<5, wp::float32> & adj_friction,
    wp::vec_t<2, wp::float32> & adj_solref,
    wp::vec_t<2, wp::float32> & adj_solreffriction,
    wp::vec_t<5, wp::float32> & adj_solimp,
    wp::float32 & adj_adhesion,
    wp::vec_t<2, wp::int32> & adj_geoms,
    wp::vec_t<2, wp::int32> & adj_pairid,
    wp::array_t<wp::float32> & adj_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> & adj_contact_frame_out,
    wp::array_t<wp::float32> & adj_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> & adj_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> & adj_contact_solimp_out,
    wp::array_t<wp::int32> & adj_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> & adj_contact_geom_out,
    wp::array_t<wp::int32> & adj_contact_efc_address_out,
    wp::array_t<wp::int32> & adj_contact_worldid_out,
    wp::array_t<wp::int32> & adj_contact_type_out,
    wp::array_t<wp::int32> & adj_contact_geomcollisionid_out,
    wp::array_t<wp::float32> & adj_contact_adhesion_out,
    wp::array_t<wp::int32> & adj_nacon_out)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}



extern "C" __global__ void _primitive_narrowphase__locals__primitive_narrowphase_5d3fb36d_cuda_kernel_forward(
    wp::launch_bounds_t<1> dim,
    wp::array_t<wp::int32> var_geom_type,
    wp::array_t<wp::int32> var_geom_condim,
    wp::array_t<wp::int32> var_geom_dataid,
    wp::array_t<wp::int32> var_geom_priority,
    wp::array_t<wp::float32> var_geom_solmix,
    wp::array_t<wp::vec_t<2, wp::float32>> var_geom_solref,
    wp::array_t<wp::vec_t<5, wp::float32>> var_geom_solimp,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_size,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_friction,
    wp::array_t<wp::float32> var_geom_margin,
    wp::array_t<wp::float32> var_geom_gap,
    wp::array_t<wp::float32> var_geom_adhesion,
    wp::array_t<wp::int32> var_mesh_vertadr,
    wp::array_t<wp::int32> var_mesh_vertnum,
    wp::array_t<wp::int32> var_mesh_graphadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_vert,
    wp::array_t<wp::int32> var_mesh_graph,
    wp::array_t<wp::int32> var_mesh_polynum,
    wp::array_t<wp::int32> var_mesh_polyadr,
    wp::array_t<wp::vec_t<3, wp::float32>> var_mesh_polynormal,
    wp::array_t<wp::int32> var_mesh_polyvertadr,
    wp::array_t<wp::int32> var_mesh_polyvertnum,
    wp::array_t<wp::int32> var_mesh_polyvert,
    wp::array_t<wp::int32> var_mesh_polymapadr,
    wp::array_t<wp::int32> var_mesh_polymapnum,
    wp::array_t<wp::int32> var_mesh_polymap,
    wp::array_t<wp::int32> var_pair_dim,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solref,
    wp::array_t<wp::vec_t<2, wp::float32>> var_pair_solreffriction,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_solimp,
    wp::array_t<wp::float32> var_pair_margin,
    wp::array_t<wp::float32> var_pair_gap,
    wp::array_t<wp::float32> var_pair_adhesion,
    wp::array_t<wp::vec_t<5, wp::float32>> var_pair_friction,
    wp::array_t<wp::vec_t<3, wp::float32>> var_geom_xpos_in,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_geom_xmat_in,
    wp::int32 var_naconmax_in,
    wp::array_t<wp::int32> var_ncollision_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pair_in,
    wp::array_t<wp::vec_t<2, wp::int32>> var_collision_pairid_in,
    wp::array_t<wp::int32> var_collision_worldid_in,
    wp::array_t<wp::float32> var_contact_dist_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_contact_pos_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_contact_frame_out,
    wp::array_t<wp::float32> var_contact_includemargin_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_friction_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solref_out,
    wp::array_t<wp::vec_t<2, wp::float32>> var_contact_solreffriction_out,
    wp::array_t<wp::vec_t<5, wp::float32>> var_contact_solimp_out,
    wp::array_t<wp::int32> var_contact_dim_out,
    wp::array_t<wp::vec_t<2, wp::int32>> var_contact_geom_out,
    wp::array_t<wp::int32> var_contact_efc_address_out,
    wp::array_t<wp::int32> var_contact_worldid_out,
    wp::array_t<wp::int32> var_contact_type_out,
    wp::array_t<wp::int32> var_contact_geomcollisionid_out,
    wp::array_t<wp::float32> var_contact_adhesion_out,
    wp::array_t<wp::int32> var_nacon_out)
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
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::vec_t<2, wp::int32>* var_5;
        wp::vec_t<2, wp::int32> var_6;
        wp::vec_t<2, wp::int32> var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::vec_t<2, wp::int32> var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        wp::int32 var_14;
        wp::vec_t<5, wp::float32> var_15;
        wp::vec_t<2, wp::float32> var_16;
        wp::vec_t<2, wp::float32> var_17;
        wp::vec_t<5, wp::float32> var_18;
        wp::float32 var_19;
        Geom_4022bfa7 var_20;
        Geom_4022bfa7 var_21;
        const wp::int32 var_22 = 0;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 2;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        bool var_35;
        bool var_36;
        bool var_37;
        wp::vec_t<2, wp::int32>* var_38;
        wp::vec_t<2, wp::int32> var_39;
        const wp::int32 var_40 = 1;
        const wp::int32 var_41 = 0;
        const wp::int32 var_42 = 3;
        const wp::int32 var_43 = 0;
        wp::int32 var_44;
        wp::int32* var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        bool var_53;
        bool var_54;
        bool var_55;
        wp::vec_t<2, wp::int32>* var_56;
        wp::vec_t<2, wp::int32> var_57;
        const wp::int32 var_58 = 2;
        const wp::int32 var_59 = 2;
        const wp::int32 var_60 = 2;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        wp::int32* var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        const wp::int32 var_66 = 1;
        wp::int32 var_67;
        wp::int32* var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        bool var_71;
        bool var_72;
        bool var_73;
        wp::vec_t<2, wp::int32>* var_74;
        wp::vec_t<2, wp::int32> var_75;
        const wp::int32 var_76 = 3;
        const wp::int32 var_77 = 2;
        const wp::int32 var_78 = 3;
        const wp::int32 var_79 = 0;
        wp::int32 var_80;
        wp::int32* var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        const wp::int32 var_84 = 1;
        wp::int32 var_85;
        wp::int32* var_86;
        wp::int32 var_87;
        wp::int32 var_88;
        bool var_89;
        bool var_90;
        bool var_91;
        wp::vec_t<2, wp::int32>* var_92;
        wp::vec_t<2, wp::int32> var_93;
        const wp::int32 var_94 = 4;
        const wp::int32 var_95 = 3;
        const wp::int32 var_96 = 3;
        const wp::int32 var_97 = 0;
        wp::int32 var_98;
        wp::int32* var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        const wp::int32 var_102 = 1;
        wp::int32 var_103;
        wp::int32* var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        bool var_107;
        bool var_108;
        bool var_109;
        wp::vec_t<2, wp::int32>* var_110;
        wp::vec_t<2, wp::int32> var_111;
        //---------
        // forward
        // def primitive_narrowphase(                                                             <L 1287>
        // tid = wp.tid()                                                                         <L 1350>
        var_0 = builtin_tid1d();
        // if tid >= ncollision_in[0]:                                                            <L 1352>
        var_2 = wp::address(var_ncollision_in, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 1353>
            continue;
        }
        // geoms = collision_pair_in[tid]                                                         <L 1355>
        var_5 = wp::address(var_collision_pair_in, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // worldid = collision_worldid_in[tid]                                                    <L 1356>
        var_8 = wp::address(var_collision_worldid_in, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // _, margin, gap, condim, friction, solref, solreffriction, solimp, adhesion = contact_params(       <L 1358>
        // geom_condim,                                                                           <L 1359>
        // geom_priority,                                                                         <L 1360>
        // geom_solmix,                                                                           <L 1361>
        // geom_solref,                                                                           <L 1362>
        // geom_solimp,                                                                           <L 1363>
        // geom_friction,                                                                         <L 1364>
        // geom_margin,                                                                           <L 1365>
        // geom_gap,                                                                              <L 1366>
        // geom_adhesion,                                                                         <L 1367>
        // pair_dim,                                                                              <L 1368>
        // pair_solref,                                                                           <L 1369>
        // pair_solreffriction,                                                                   <L 1370>
        // pair_solimp,                                                                           <L 1371>
        // pair_margin,                                                                           <L 1372>
        // pair_gap,                                                                              <L 1373>
        // pair_adhesion,                                                                         <L 1374>
        // pair_friction,                                                                         <L 1375>
        // collision_pair_in,                                                                     <L 1376>
        // collision_pairid_in,                                                                   <L 1377>
        // tid,                                                                                   <L 1378>
        // worldid,                                                                               <L 1379>
        contact_params_0(var_geom_condim, var_geom_priority, var_geom_solmix, var_geom_solref, var_geom_solimp, var_geom_friction, var_geom_margin, var_geom_gap, var_geom_adhesion, var_pair_dim, var_pair_solref, var_pair_solreffriction, var_pair_solimp, var_pair_margin, var_pair_gap, var_pair_adhesion, var_pair_friction, var_collision_pair_in, var_collision_pairid_in, var_0, var_9, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19);
        // geom1, geom2 = geom_collision_pair(                                                    <L 1382>
        // geom_type,                                                                             <L 1383>
        // geom_dataid,                                                                           <L 1384>
        // geom_size,                                                                             <L 1385>
        // mesh_vertadr,                                                                          <L 1386>
        // mesh_vertnum,                                                                          <L 1387>
        // mesh_graphadr,                                                                         <L 1388>
        // mesh_vert,                                                                             <L 1389>
        // mesh_graph,                                                                            <L 1390>
        // mesh_polynum,                                                                          <L 1391>
        // mesh_polyadr,                                                                          <L 1392>
        // mesh_polynormal,                                                                       <L 1393>
        // mesh_polyvertadr,                                                                      <L 1394>
        // mesh_polyvertnum,                                                                      <L 1395>
        // mesh_polyvert,                                                                         <L 1396>
        // mesh_polymapadr,                                                                       <L 1397>
        // mesh_polymapnum,                                                                       <L 1398>
        // mesh_polymap,                                                                          <L 1399>
        // geom_xpos_in,                                                                          <L 1400>
        // geom_xmat_in,                                                                          <L 1401>
        // geoms,                                                                                 <L 1402>
        // worldid,                                                                               <L 1403>
        geom_collision_pair_0(var_geom_type, var_geom_dataid, var_geom_size, var_mesh_vertadr, var_mesh_vertnum, var_mesh_graphadr, var_mesh_vert, var_mesh_graph, var_mesh_polynum, var_mesh_polyadr, var_mesh_polynormal, var_mesh_polyvertadr, var_mesh_polyvertnum, var_mesh_polyvert, var_mesh_polymapadr, var_mesh_polymapnum, var_mesh_polymap, var_geom_xpos_in, var_geom_xmat_in, var_6, var_9, var_20, var_21);
        // for i in range(wp.static(len(primitive_collisions_func))):                             <L 1406>
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1407>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1408>
        // type1 = geom_type[geoms[0]]                                                            <L 1409>
        var_26 = wp::extract(var_6, var_25);
        var_27 = wp::address(var_geom_type, var_26);
        var_29 = wp::load(var_27);
        var_28 = wp::copy(var_29);
        // type2 = geom_type[geoms[1]]                                                            <L 1410>
        var_31 = wp::extract(var_6, var_30);
        var_32 = wp::address(var_geom_type, var_31);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1411>
        var_36 = (var_23 == var_28);
        var_35 = var_36;
        if (var_35) {
            var_37 = (var_24 == var_33);
            var_35 = var_35 && var_37;
        }
        if (var_35) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1412>
            // naconmax_in,                                                                       <L 1413>
            // geom1,                                                                             <L 1414>
            // geom2,                                                                             <L 1415>
            // worldid,                                                                           <L 1416>
            // margin,                                                                            <L 1417>
            // gap,                                                                               <L 1418>
            // condim,                                                                            <L 1419>
            // friction,                                                                          <L 1420>
            // solref,                                                                            <L 1421>
            // solreffriction,                                                                    <L 1422>
            // solimp,                                                                            <L 1423>
            // adhesion,                                                                          <L 1424>
            // geoms,                                                                             <L 1425>
            // collision_pairid_in[tid],                                                          <L 1426>
            var_38 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1427>
            // contact_pos_out,                                                                   <L 1428>
            // contact_frame_out,                                                                 <L 1429>
            // contact_includemargin_out,                                                         <L 1430>
            // contact_friction_out,                                                              <L 1431>
            // contact_solref_out,                                                                <L 1432>
            // contact_solreffriction_out,                                                        <L 1433>
            // contact_solimp_out,                                                                <L 1434>
            // contact_dim_out,                                                                   <L 1435>
            // contact_geom_out,                                                                  <L 1436>
            // contact_efc_address_out,                                                           <L 1437>
            // contact_worldid_out,                                                               <L 1438>
            // contact_type_out,                                                                  <L 1439>
            // contact_geomcollisionid_out,                                                       <L 1440>
            // contact_adhesion_out,                                                              <L 1441>
            // nacon_out,                                                                         <L 1442>
            var_39 = wp::load(var_38);
            plane_sphere_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_39, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1407>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1408>
        // type1 = geom_type[geoms[0]]                                                            <L 1409>
        var_44 = wp::extract(var_6, var_43);
        var_45 = wp::address(var_geom_type, var_44);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // type2 = geom_type[geoms[1]]                                                            <L 1410>
        var_49 = wp::extract(var_6, var_48);
        var_50 = wp::address(var_geom_type, var_49);
        var_52 = wp::load(var_50);
        var_51 = wp::copy(var_52);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1411>
        var_54 = (var_41 == var_46);
        var_53 = var_54;
        if (var_53) {
            var_55 = (var_42 == var_51);
            var_53 = var_53 && var_55;
        }
        if (var_53) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1412>
            // naconmax_in,                                                                       <L 1413>
            // geom1,                                                                             <L 1414>
            // geom2,                                                                             <L 1415>
            // worldid,                                                                           <L 1416>
            // margin,                                                                            <L 1417>
            // gap,                                                                               <L 1418>
            // condim,                                                                            <L 1419>
            // friction,                                                                          <L 1420>
            // solref,                                                                            <L 1421>
            // solreffriction,                                                                    <L 1422>
            // solimp,                                                                            <L 1423>
            // adhesion,                                                                          <L 1424>
            // geoms,                                                                             <L 1425>
            // collision_pairid_in[tid],                                                          <L 1426>
            var_56 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1427>
            // contact_pos_out,                                                                   <L 1428>
            // contact_frame_out,                                                                 <L 1429>
            // contact_includemargin_out,                                                         <L 1430>
            // contact_friction_out,                                                              <L 1431>
            // contact_solref_out,                                                                <L 1432>
            // contact_solreffriction_out,                                                        <L 1433>
            // contact_solimp_out,                                                                <L 1434>
            // contact_dim_out,                                                                   <L 1435>
            // contact_geom_out,                                                                  <L 1436>
            // contact_efc_address_out,                                                           <L 1437>
            // contact_worldid_out,                                                               <L 1438>
            // contact_type_out,                                                                  <L 1439>
            // contact_geomcollisionid_out,                                                       <L 1440>
            // contact_adhesion_out,                                                              <L 1441>
            // nacon_out,                                                                         <L 1442>
            var_57 = wp::load(var_56);
            plane_capsule_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_57, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1407>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1408>
        // type1 = geom_type[geoms[0]]                                                            <L 1409>
        var_62 = wp::extract(var_6, var_61);
        var_63 = wp::address(var_geom_type, var_62);
        var_65 = wp::load(var_63);
        var_64 = wp::copy(var_65);
        // type2 = geom_type[geoms[1]]                                                            <L 1410>
        var_67 = wp::extract(var_6, var_66);
        var_68 = wp::address(var_geom_type, var_67);
        var_70 = wp::load(var_68);
        var_69 = wp::copy(var_70);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1411>
        var_72 = (var_59 == var_64);
        var_71 = var_72;
        if (var_71) {
            var_73 = (var_60 == var_69);
            var_71 = var_71 && var_73;
        }
        if (var_71) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1412>
            // naconmax_in,                                                                       <L 1413>
            // geom1,                                                                             <L 1414>
            // geom2,                                                                             <L 1415>
            // worldid,                                                                           <L 1416>
            // margin,                                                                            <L 1417>
            // gap,                                                                               <L 1418>
            // condim,                                                                            <L 1419>
            // friction,                                                                          <L 1420>
            // solref,                                                                            <L 1421>
            // solreffriction,                                                                    <L 1422>
            // solimp,                                                                            <L 1423>
            // adhesion,                                                                          <L 1424>
            // geoms,                                                                             <L 1425>
            // collision_pairid_in[tid],                                                          <L 1426>
            var_74 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1427>
            // contact_pos_out,                                                                   <L 1428>
            // contact_frame_out,                                                                 <L 1429>
            // contact_includemargin_out,                                                         <L 1430>
            // contact_friction_out,                                                              <L 1431>
            // contact_solref_out,                                                                <L 1432>
            // contact_solreffriction_out,                                                        <L 1433>
            // contact_solimp_out,                                                                <L 1434>
            // contact_dim_out,                                                                   <L 1435>
            // contact_geom_out,                                                                  <L 1436>
            // contact_efc_address_out,                                                           <L 1437>
            // contact_worldid_out,                                                               <L 1438>
            // contact_type_out,                                                                  <L 1439>
            // contact_geomcollisionid_out,                                                       <L 1440>
            // contact_adhesion_out,                                                              <L 1441>
            // nacon_out,                                                                         <L 1442>
            var_75 = wp::load(var_74);
            sphere_sphere_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_75, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1407>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1408>
        // type1 = geom_type[geoms[0]]                                                            <L 1409>
        var_80 = wp::extract(var_6, var_79);
        var_81 = wp::address(var_geom_type, var_80);
        var_83 = wp::load(var_81);
        var_82 = wp::copy(var_83);
        // type2 = geom_type[geoms[1]]                                                            <L 1410>
        var_85 = wp::extract(var_6, var_84);
        var_86 = wp::address(var_geom_type, var_85);
        var_88 = wp::load(var_86);
        var_87 = wp::copy(var_88);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1411>
        var_90 = (var_77 == var_82);
        var_89 = var_90;
        if (var_89) {
            var_91 = (var_78 == var_87);
            var_89 = var_89 && var_91;
        }
        if (var_89) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1412>
            // naconmax_in,                                                                       <L 1413>
            // geom1,                                                                             <L 1414>
            // geom2,                                                                             <L 1415>
            // worldid,                                                                           <L 1416>
            // margin,                                                                            <L 1417>
            // gap,                                                                               <L 1418>
            // condim,                                                                            <L 1419>
            // friction,                                                                          <L 1420>
            // solref,                                                                            <L 1421>
            // solreffriction,                                                                    <L 1422>
            // solimp,                                                                            <L 1423>
            // adhesion,                                                                          <L 1424>
            // geoms,                                                                             <L 1425>
            // collision_pairid_in[tid],                                                          <L 1426>
            var_92 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1427>
            // contact_pos_out,                                                                   <L 1428>
            // contact_frame_out,                                                                 <L 1429>
            // contact_includemargin_out,                                                         <L 1430>
            // contact_friction_out,                                                              <L 1431>
            // contact_solref_out,                                                                <L 1432>
            // contact_solreffriction_out,                                                        <L 1433>
            // contact_solimp_out,                                                                <L 1434>
            // contact_dim_out,                                                                   <L 1435>
            // contact_geom_out,                                                                  <L 1436>
            // contact_efc_address_out,                                                           <L 1437>
            // contact_worldid_out,                                                               <L 1438>
            // contact_type_out,                                                                  <L 1439>
            // contact_geomcollisionid_out,                                                       <L 1440>
            // contact_adhesion_out,                                                              <L 1441>
            // nacon_out,                                                                         <L 1442>
            var_93 = wp::load(var_92);
            sphere_capsule_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_93, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1407>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1408>
        // type1 = geom_type[geoms[0]]                                                            <L 1409>
        var_98 = wp::extract(var_6, var_97);
        var_99 = wp::address(var_geom_type, var_98);
        var_101 = wp::load(var_99);
        var_100 = wp::copy(var_101);
        // type2 = geom_type[geoms[1]]                                                            <L 1410>
        var_103 = wp::extract(var_6, var_102);
        var_104 = wp::address(var_geom_type, var_103);
        var_106 = wp::load(var_104);
        var_105 = wp::copy(var_106);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1411>
        var_108 = (var_95 == var_100);
        var_107 = var_108;
        if (var_107) {
            var_109 = (var_96 == var_105);
            var_107 = var_107 && var_109;
        }
        if (var_107) {
            // wp.static(primitive_collisions_func[i])(                                           <L 1412>
            // naconmax_in,                                                                       <L 1413>
            // geom1,                                                                             <L 1414>
            // geom2,                                                                             <L 1415>
            // worldid,                                                                           <L 1416>
            // margin,                                                                            <L 1417>
            // gap,                                                                               <L 1418>
            // condim,                                                                            <L 1419>
            // friction,                                                                          <L 1420>
            // solref,                                                                            <L 1421>
            // solreffriction,                                                                    <L 1422>
            // solimp,                                                                            <L 1423>
            // adhesion,                                                                          <L 1424>
            // geoms,                                                                             <L 1425>
            // collision_pairid_in[tid],                                                          <L 1426>
            var_110 = wp::address(var_collision_pairid_in, var_0);
            // contact_dist_out,                                                                  <L 1427>
            // contact_pos_out,                                                                   <L 1428>
            // contact_frame_out,                                                                 <L 1429>
            // contact_includemargin_out,                                                         <L 1430>
            // contact_friction_out,                                                              <L 1431>
            // contact_solref_out,                                                                <L 1432>
            // contact_solreffriction_out,                                                        <L 1433>
            // contact_solimp_out,                                                                <L 1434>
            // contact_dim_out,                                                                   <L 1435>
            // contact_geom_out,                                                                  <L 1436>
            // contact_efc_address_out,                                                           <L 1437>
            // contact_worldid_out,                                                               <L 1438>
            // contact_type_out,                                                                  <L 1439>
            // contact_geomcollisionid_out,                                                       <L 1440>
            // contact_adhesion_out,                                                              <L 1441>
            // nacon_out,                                                                         <L 1442>
            var_111 = wp::load(var_110);
            capsule_capsule_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_111, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
        }
    }
}

