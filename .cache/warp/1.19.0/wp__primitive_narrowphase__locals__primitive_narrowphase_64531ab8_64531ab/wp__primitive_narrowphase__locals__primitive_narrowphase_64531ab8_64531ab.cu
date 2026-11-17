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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:465
static CUDA_CALLABLE void plane_cylinder_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_cylinder_center,
    wp::vec_t<3, wp::float32> var_cylinder_axis,
    wp::float32 var_cylinder_radius,
    wp::float32 var_cylinder_half_height,
    wp::vec_t<4, wp::float32> & ret_0,
    wp::mat_t<4, 3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 10000000000.0;
    wp::vec_t<4, wp::float32> var_1;
    wp::mat_t<4, 3, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    bool var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::float32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::float32 var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 1e-12;
    bool var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::vec_t<3, wp::float32> var_22;
    const wp::float32 var_23 = 1.0;
    const wp::float32 var_24 = 0.0;
    const wp::float32 var_25 = 0.0;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    const wp::float32 var_36 = 0.5;
    wp::float32 var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    const wp::int32 var_40 = 1;
    wp::int32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    const wp::float32 var_46 = 0.5;
    wp::float32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    const wp::int32 var_50 = 1;
    wp::int32 var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 0.5;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::vec_t<3, wp::float32> var_57;
    wp::vec_t<3, wp::float32> var_58;
    const wp::float32 var_59 = 3.0;
    wp::float32 var_60;
    wp::float32 var_61;
    const wp::float32 var_62 = 0.5;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    const wp::float32 var_67 = 0.5;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    const wp::float32 var_70 = 0.5;
    wp::float32 var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    const wp::int32 var_74 = 1;
    wp::int32 var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    const wp::float32 var_78 = 0.5;
    wp::vec_t<3, wp::float32> var_79;
    wp::vec_t<3, wp::float32> var_80;
    const wp::float32 var_81 = 0.5;
    wp::float32 var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32> var_84;
    const wp::int32 var_85 = 1;
    wp::int32 var_86;
    //---------
    // forward
    // def plane_cylinder(                                                                    <L 466>
    // contact_dist = wp.vec4(MJ_MAXVAL)                                                      <L 491>
    var_1 = wp::vec_t<4, wp::float32>(var_0);
    // contact_pos = mat43f()                                                                 <L 492>
    var_2 = wp::mat_t<4, 3, wp::float32>();
    // contact_count = 0                                                                      <L 493>
    // n = plane_normal                                                                       <L 495>
    var_4 = wp::copy(var_plane_normal);
    // axis = cylinder_axis                                                                   <L 496>
    var_5 = wp::copy(var_cylinder_axis);
    // prjaxis = wp.dot(n, axis)                                                              <L 499>
    var_6 = wp::dot(var_4, var_5);
    // if prjaxis > 0:                                                                        <L 500>
    var_8 = (var_6 > var_7);
    if (var_8) {
        // axis = -axis                                                                       <L 501>
        var_9 = wp::neg(var_5);
        // prjaxis = -prjaxis                                                                 <L 502>
        var_10 = wp::neg(var_6);
    }
    var_11 = wp::where(var_8, var_9, var_5);
    var_12 = wp::where(var_8, var_10, var_6);
    // dist0 = wp.dot(cylinder_center - plane_pos, n)                                         <L 505>
    var_13 = wp::sub(var_cylinder_center, var_plane_pos);
    var_14 = wp::dot(var_13, var_4);
    // vec = axis * prjaxis - n                                                               <L 508>
    var_15 = wp::mul(var_11, var_12);
    var_16 = wp::sub(var_15, var_4);
    // len_sqr = wp.dot(vec, vec)                                                             <L 509>
    var_17 = wp::dot(var_16, var_16);
    // vec = wp.where(                                                                        <L 513>
    // len_sqr >= 1e-12,                                                                      <L 514>
    var_19 = (var_17 >= var_18);
    // vec * safe_div(cylinder_radius, wp.sqrt(len_sqr)),                                     <L 515>
    var_20 = wp::sqrt(var_17);
    var_21 = safe_div_0(var_cylinder_radius, var_20);
    var_22 = wp::mul(var_16, var_21);
    // wp.vec3(1.0, 0.0, 0.0) * cylinder_radius,  # Default x-axis when degenerate            <L 516>
    var_26 = wp::vec_t<3, wp::float32>(var_23, var_24, var_25);
    var_27 = wp::mul(var_26, var_cylinder_radius);
    var_28 = wp::where(var_19, var_22, var_27);
    // prjvec = wp.dot(vec, n)                                                                <L 520>
    var_29 = wp::dot(var_28, var_4);
    // axis = axis * cylinder_half_height                                                     <L 523>
    var_30 = wp::mul(var_11, var_cylinder_half_height);
    // prjaxis = prjaxis * cylinder_half_height                                               <L 524>
    var_31 = wp::mul(var_12, var_cylinder_half_height);
    // dist1 = dist0 + prjaxis + prjvec                                                       <L 527>
    var_32 = wp::add(var_14, var_31);
    var_33 = wp::add(var_32, var_29);
    // pos1 = cylinder_center + vec + axis - n * (dist1 * 0.5)                                <L 528>
    var_34 = wp::add(var_cylinder_center, var_28);
    var_35 = wp::add(var_34, var_30);
    var_37 = wp::mul(var_33, var_36);
    var_38 = wp::mul(var_4, var_37);
    var_39 = wp::sub(var_35, var_38);
    // contact_dist[contact_count] = dist1                                                    <L 529>
    wp::assign_inplace(var_1, var_3, var_33);
    // contact_pos[contact_count] = pos1                                                      <L 530>
    wp::assign_inplace(var_2, var_3, var_39);
    // contact_count = contact_count + 1                                                      <L 531>
    var_41 = wp::add(var_3, var_40);
    // dist2 = dist0 - prjaxis + prjvec                                                       <L 534>
    var_42 = wp::sub(var_14, var_31);
    var_43 = wp::add(var_42, var_29);
    // pos2 = cylinder_center + vec - axis - n * (dist2 * 0.5)                                <L 535>
    var_44 = wp::add(var_cylinder_center, var_28);
    var_45 = wp::sub(var_44, var_30);
    var_47 = wp::mul(var_43, var_46);
    var_48 = wp::mul(var_4, var_47);
    var_49 = wp::sub(var_45, var_48);
    // contact_dist[contact_count] = dist2                                                    <L 536>
    wp::assign_inplace(var_1, var_41, var_43);
    // contact_pos[contact_count] = pos2                                                      <L 537>
    wp::assign_inplace(var_2, var_41, var_49);
    // contact_count = contact_count + 1                                                      <L 538>
    var_51 = wp::add(var_41, var_50);
    // prjvec1 = -prjvec * 0.5                                                                <L 541>
    var_52 = wp::neg(var_29);
    var_54 = wp::mul(var_52, var_53);
    // dist3 = dist0 + prjaxis + prjvec1                                                      <L 542>
    var_55 = wp::add(var_14, var_31);
    var_56 = wp::add(var_55, var_54);
    // vec1 = wp.cross(vec, axis)                                                             <L 544>
    var_57 = wp::cross(var_28, var_30);
    // vec1 = wp.normalize(vec1) * (cylinder_radius * wp.sqrt(3.0) * 0.5)                     <L 545>
    var_58 = wp::normalize(var_57);
    var_60 = wp::sqrt(var_59);
    var_61 = wp::mul(var_cylinder_radius, var_60);
    var_63 = wp::mul(var_61, var_62);
    var_64 = wp::mul(var_58, var_63);
    // pos3 = cylinder_center + vec1 + axis - vec * 0.5 - n * (dist3 * 0.5)                   <L 548>
    var_65 = wp::add(var_cylinder_center, var_64);
    var_66 = wp::add(var_65, var_30);
    var_68 = wp::mul(var_28, var_67);
    var_69 = wp::sub(var_66, var_68);
    var_71 = wp::mul(var_56, var_70);
    var_72 = wp::mul(var_4, var_71);
    var_73 = wp::sub(var_69, var_72);
    // contact_dist[contact_count] = dist3                                                    <L 549>
    wp::assign_inplace(var_1, var_51, var_56);
    // contact_pos[contact_count] = pos3                                                      <L 550>
    wp::assign_inplace(var_2, var_51, var_73);
    // contact_count = contact_count + 1                                                      <L 551>
    var_75 = wp::add(var_51, var_74);
    // pos4 = cylinder_center - vec1 + axis - vec * 0.5 - n * (dist3 * 0.5)                   <L 554>
    var_76 = wp::sub(var_cylinder_center, var_64);
    var_77 = wp::add(var_76, var_30);
    var_79 = wp::mul(var_28, var_78);
    var_80 = wp::sub(var_77, var_79);
    var_82 = wp::mul(var_56, var_81);
    var_83 = wp::mul(var_4, var_82);
    var_84 = wp::sub(var_80, var_83);
    // contact_dist[contact_count] = dist3                                                    <L 555>
    wp::assign_inplace(var_1, var_75, var_56);
    // contact_pos[contact_count] = pos4                                                      <L 556>
    wp::assign_inplace(var_2, var_75, var_84);
    // contact_count = contact_count + 1                                                      <L 557>
    var_86 = wp::add(var_75, var_85);
    // return contact_dist, contact_pos, n                                                    <L 559>
    ret_0 = var_1;
    ret_1 = var_2;
    ret_2 = var_4;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:931
static CUDA_CALLABLE void plane_cylinder_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_plane,
    Geom_4022bfa7 var_cylinder,
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
    wp::vec_t<4, wp::float32> var_27;
    wp::mat_t<4, 3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::mat_t<3, 3, wp::float32> var_33;
    const wp::int32 var_34 = 0;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::int32 var_37;
    const wp::int32 var_38 = 1;
    wp::float32 var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::int32 var_41;
    const wp::int32 var_42 = 2;
    wp::float32 var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::int32 var_45;
    const wp::int32 var_46 = 3;
    wp::float32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::int32 var_49;
    //---------
    // forward
    // def plane_cylinder_wrapper(                                                            <L 932>
    // cylinder_axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2], cylinder.rot[2, 2])       <L 969>
    var_0 = &((var_cylinder).rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &((var_cylinder).rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &((var_cylinder).rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // dist, pos, normal = plane_cylinder(                                                    <L 971>
    // plane.normal,                                                                          <L 972>
    var_16 = &((var_plane).normal);
    // plane.pos,                                                                             <L 973>
    var_17 = &((var_plane).pos);
    // cylinder.pos,                                                                          <L 974>
    var_18 = &((var_cylinder).pos);
    // cylinder_axis,                                                                         <L 975>
    // cylinder.size[0],  # radius                                                            <L 976>
    var_19 = &((var_cylinder).size);
    var_22 = wp::load(var_19);
    var_21 = wp::extract(var_22, var_20);
    // cylinder.size[1],  # half_height                                                       <L 977>
    var_23 = &((var_cylinder).size);
    var_26 = wp::load(var_23);
    var_25 = wp::extract(var_26, var_24);
    var_30 = wp::load(var_16);
    var_31 = wp::load(var_17);
    var_32 = wp::load(var_18);
    plane_cylinder_0(var_30, var_31, var_32, var_15, var_21, var_25, var_27, var_28, var_29);
    // frame = make_frame(normal)                                                             <L 980>
    var_33 = make_frame_0(var_29);
    // for i in range(4):                                                                     <L 981>
    // write_contact(                                                                         <L 982>
    // naconmax_in,                                                                           <L 983>
    // i,                                                                                     <L 984>
    // dist[i],                                                                               <L 985>
    var_35 = wp::extract(var_27, var_34);
    // pos[i],                                                                                <L 986>
    var_36 = wp::extract(var_28, var_34);
    // frame,                                                                                 <L 987>
    // margin,                                                                                <L 988>
    // gap,                                                                                   <L 989>
    // condim,                                                                                <L 990>
    // friction,                                                                              <L 991>
    // solref,                                                                                <L 992>
    // solreffriction,                                                                        <L 993>
    // solimp,                                                                                <L 994>
    // adhesion,                                                                              <L 995>
    // geoms,                                                                                 <L 996>
    // pairid,                                                                                <L 997>
    // worldid,                                                                               <L 998>
    // contact_dist_out,                                                                      <L 999>
    // contact_pos_out,                                                                       <L 1000>
    // contact_frame_out,                                                                     <L 1001>
    // contact_includemargin_out,                                                             <L 1002>
    // contact_friction_out,                                                                  <L 1003>
    // contact_solref_out,                                                                    <L 1004>
    // contact_solreffriction_out,                                                            <L 1005>
    // contact_solimp_out,                                                                    <L 1006>
    // contact_dim_out,                                                                       <L 1007>
    // contact_geom_out,                                                                      <L 1008>
    // contact_efc_address_out,                                                               <L 1009>
    // contact_worldid_out,                                                                   <L 1010>
    // contact_type_out,                                                                      <L 1011>
    // contact_geomcollisionid_out,                                                           <L 1012>
    // contact_adhesion_out,                                                                  <L 1013>
    // nacon_out,                                                                             <L 1014>
    var_37 = write_contact_0(var_naconmax_in, var_34, var_35, var_36, var_33, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 982>
    // naconmax_in,                                                                           <L 983>
    // i,                                                                                     <L 984>
    // dist[i],                                                                               <L 985>
    var_39 = wp::extract(var_27, var_38);
    // pos[i],                                                                                <L 986>
    var_40 = wp::extract(var_28, var_38);
    // frame,                                                                                 <L 987>
    // margin,                                                                                <L 988>
    // gap,                                                                                   <L 989>
    // condim,                                                                                <L 990>
    // friction,                                                                              <L 991>
    // solref,                                                                                <L 992>
    // solreffriction,                                                                        <L 993>
    // solimp,                                                                                <L 994>
    // adhesion,                                                                              <L 995>
    // geoms,                                                                                 <L 996>
    // pairid,                                                                                <L 997>
    // worldid,                                                                               <L 998>
    // contact_dist_out,                                                                      <L 999>
    // contact_pos_out,                                                                       <L 1000>
    // contact_frame_out,                                                                     <L 1001>
    // contact_includemargin_out,                                                             <L 1002>
    // contact_friction_out,                                                                  <L 1003>
    // contact_solref_out,                                                                    <L 1004>
    // contact_solreffriction_out,                                                            <L 1005>
    // contact_solimp_out,                                                                    <L 1006>
    // contact_dim_out,                                                                       <L 1007>
    // contact_geom_out,                                                                      <L 1008>
    // contact_efc_address_out,                                                               <L 1009>
    // contact_worldid_out,                                                                   <L 1010>
    // contact_type_out,                                                                      <L 1011>
    // contact_geomcollisionid_out,                                                           <L 1012>
    // contact_adhesion_out,                                                                  <L 1013>
    // nacon_out,                                                                             <L 1014>
    var_41 = write_contact_0(var_naconmax_in, var_38, var_39, var_40, var_33, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 982>
    // naconmax_in,                                                                           <L 983>
    // i,                                                                                     <L 984>
    // dist[i],                                                                               <L 985>
    var_43 = wp::extract(var_27, var_42);
    // pos[i],                                                                                <L 986>
    var_44 = wp::extract(var_28, var_42);
    // frame,                                                                                 <L 987>
    // margin,                                                                                <L 988>
    // gap,                                                                                   <L 989>
    // condim,                                                                                <L 990>
    // friction,                                                                              <L 991>
    // solref,                                                                                <L 992>
    // solreffriction,                                                                        <L 993>
    // solimp,                                                                                <L 994>
    // adhesion,                                                                              <L 995>
    // geoms,                                                                                 <L 996>
    // pairid,                                                                                <L 997>
    // worldid,                                                                               <L 998>
    // contact_dist_out,                                                                      <L 999>
    // contact_pos_out,                                                                       <L 1000>
    // contact_frame_out,                                                                     <L 1001>
    // contact_includemargin_out,                                                             <L 1002>
    // contact_friction_out,                                                                  <L 1003>
    // contact_solref_out,                                                                    <L 1004>
    // contact_solreffriction_out,                                                            <L 1005>
    // contact_solimp_out,                                                                    <L 1006>
    // contact_dim_out,                                                                       <L 1007>
    // contact_geom_out,                                                                      <L 1008>
    // contact_efc_address_out,                                                               <L 1009>
    // contact_worldid_out,                                                                   <L 1010>
    // contact_type_out,                                                                      <L 1011>
    // contact_geomcollisionid_out,                                                           <L 1012>
    // contact_adhesion_out,                                                                  <L 1013>
    // nacon_out,                                                                             <L 1014>
    var_45 = write_contact_0(var_naconmax_in, var_42, var_43, var_44, var_33, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 982>
    // naconmax_in,                                                                           <L 983>
    // i,                                                                                     <L 984>
    // dist[i],                                                                               <L 985>
    var_47 = wp::extract(var_27, var_46);
    // pos[i],                                                                                <L 986>
    var_48 = wp::extract(var_28, var_46);
    // frame,                                                                                 <L 987>
    // margin,                                                                                <L 988>
    // gap,                                                                                   <L 989>
    // condim,                                                                                <L 990>
    // friction,                                                                              <L 991>
    // solref,                                                                                <L 992>
    // solreffriction,                                                                        <L 993>
    // solimp,                                                                                <L 994>
    // adhesion,                                                                              <L 995>
    // geoms,                                                                                 <L 996>
    // pairid,                                                                                <L 997>
    // worldid,                                                                               <L 998>
    // contact_dist_out,                                                                      <L 999>
    // contact_pos_out,                                                                       <L 1000>
    // contact_frame_out,                                                                     <L 1001>
    // contact_includemargin_out,                                                             <L 1002>
    // contact_friction_out,                                                                  <L 1003>
    // contact_solref_out,                                                                    <L 1004>
    // contact_solreffriction_out,                                                            <L 1005>
    // contact_solimp_out,                                                                    <L 1006>
    // contact_dim_out,                                                                       <L 1007>
    // contact_geom_out,                                                                      <L 1008>
    // contact_efc_address_out,                                                               <L 1009>
    // contact_worldid_out,                                                                   <L 1010>
    // contact_type_out,                                                                      <L 1011>
    // contact_geomcollisionid_out,                                                           <L 1012>
    // contact_adhesion_out,                                                                  <L 1013>
    // nacon_out,                                                                             <L 1014>
    var_49 = write_contact_0(var_naconmax_in, var_46, var_47, var_48, var_33, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:342
static CUDA_CALLABLE void plane_box_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::vec_t<8, wp::float32> & ret_0,
    wp::mat_t<8, 3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 10000000000.0;
    wp::vec_t<8, wp::float32> var_3;
    wp::mat_t<8, 3, wp::float32> var_4;
    const wp::int32 var_5 = 0;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::int32 var_14 = 2;
    wp::int32 var_15;
    const wp::int32 var_16 = 1;
    wp::float32 var_17;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    const wp::int32 var_22 = 4;
    wp::int32 var_23;
    const wp::int32 var_24 = 2;
    wp::float32 var_25;
    const wp::int32 var_26 = 2;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::float32 var_35 = 0.5;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    const wp::int32 var_39 = 1;
    const wp::int32 var_40 = 1;
    wp::int32 var_41;
    const wp::int32 var_42 = 0;
    wp::float32 var_43;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    const wp::int32 var_48 = 2;
    wp::int32 var_49;
    const wp::int32 var_50 = 1;
    wp::float32 var_51;
    const wp::int32 var_52 = 1;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    const wp::int32 var_56 = 4;
    wp::int32 var_57;
    const wp::int32 var_58 = 2;
    wp::float32 var_59;
    const wp::int32 var_60 = 2;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::vec_t<3, wp::float32> var_68;
    const wp::float32 var_69 = 0.5;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    const wp::int32 var_73 = 2;
    const wp::int32 var_74 = 1;
    wp::int32 var_75;
    const wp::int32 var_76 = 0;
    wp::float32 var_77;
    const wp::int32 var_78 = 0;
    wp::float32 var_79;
    wp::float32 var_80;
    wp::float32 var_81;
    const wp::int32 var_82 = 2;
    wp::int32 var_83;
    const wp::int32 var_84 = 1;
    wp::float32 var_85;
    const wp::int32 var_86 = 1;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    const wp::int32 var_90 = 4;
    wp::int32 var_91;
    const wp::int32 var_92 = 2;
    wp::float32 var_93;
    const wp::int32 var_94 = 2;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::vec_t<3, wp::float32> var_102;
    const wp::float32 var_103 = 0.5;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    const wp::int32 var_107 = 3;
    const wp::int32 var_108 = 1;
    wp::int32 var_109;
    const wp::int32 var_110 = 0;
    wp::float32 var_111;
    const wp::int32 var_112 = 0;
    wp::float32 var_113;
    wp::float32 var_114;
    wp::float32 var_115;
    const wp::int32 var_116 = 2;
    wp::int32 var_117;
    const wp::int32 var_118 = 1;
    wp::float32 var_119;
    const wp::int32 var_120 = 1;
    wp::float32 var_121;
    wp::float32 var_122;
    wp::float32 var_123;
    const wp::int32 var_124 = 4;
    wp::int32 var_125;
    const wp::int32 var_126 = 2;
    wp::float32 var_127;
    const wp::int32 var_128 = 2;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::float32 var_131;
    wp::vec_t<3, wp::float32> var_132;
    wp::vec_t<3, wp::float32> var_133;
    wp::float32 var_134;
    wp::float32 var_135;
    wp::vec_t<3, wp::float32> var_136;
    const wp::float32 var_137 = 0.5;
    wp::vec_t<3, wp::float32> var_138;
    wp::vec_t<3, wp::float32> var_139;
    wp::vec_t<3, wp::float32> var_140;
    const wp::int32 var_141 = 4;
    const wp::int32 var_142 = 1;
    wp::int32 var_143;
    const wp::int32 var_144 = 0;
    wp::float32 var_145;
    const wp::int32 var_146 = 0;
    wp::float32 var_147;
    wp::float32 var_148;
    wp::float32 var_149;
    const wp::int32 var_150 = 2;
    wp::int32 var_151;
    const wp::int32 var_152 = 1;
    wp::float32 var_153;
    const wp::int32 var_154 = 1;
    wp::float32 var_155;
    wp::float32 var_156;
    wp::float32 var_157;
    const wp::int32 var_158 = 4;
    wp::int32 var_159;
    const wp::int32 var_160 = 2;
    wp::float32 var_161;
    const wp::int32 var_162 = 2;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::float32 var_165;
    wp::vec_t<3, wp::float32> var_166;
    wp::vec_t<3, wp::float32> var_167;
    wp::float32 var_168;
    wp::float32 var_169;
    wp::vec_t<3, wp::float32> var_170;
    const wp::float32 var_171 = 0.5;
    wp::vec_t<3, wp::float32> var_172;
    wp::vec_t<3, wp::float32> var_173;
    wp::vec_t<3, wp::float32> var_174;
    const wp::int32 var_175 = 5;
    const wp::int32 var_176 = 1;
    wp::int32 var_177;
    const wp::int32 var_178 = 0;
    wp::float32 var_179;
    const wp::int32 var_180 = 0;
    wp::float32 var_181;
    wp::float32 var_182;
    wp::float32 var_183;
    const wp::int32 var_184 = 2;
    wp::int32 var_185;
    const wp::int32 var_186 = 1;
    wp::float32 var_187;
    const wp::int32 var_188 = 1;
    wp::float32 var_189;
    wp::float32 var_190;
    wp::float32 var_191;
    const wp::int32 var_192 = 4;
    wp::int32 var_193;
    const wp::int32 var_194 = 2;
    wp::float32 var_195;
    const wp::int32 var_196 = 2;
    wp::float32 var_197;
    wp::float32 var_198;
    wp::float32 var_199;
    wp::vec_t<3, wp::float32> var_200;
    wp::vec_t<3, wp::float32> var_201;
    wp::float32 var_202;
    wp::float32 var_203;
    wp::vec_t<3, wp::float32> var_204;
    const wp::float32 var_205 = 0.5;
    wp::vec_t<3, wp::float32> var_206;
    wp::vec_t<3, wp::float32> var_207;
    wp::vec_t<3, wp::float32> var_208;
    const wp::int32 var_209 = 6;
    const wp::int32 var_210 = 1;
    wp::int32 var_211;
    const wp::int32 var_212 = 0;
    wp::float32 var_213;
    const wp::int32 var_214 = 0;
    wp::float32 var_215;
    wp::float32 var_216;
    wp::float32 var_217;
    const wp::int32 var_218 = 2;
    wp::int32 var_219;
    const wp::int32 var_220 = 1;
    wp::float32 var_221;
    const wp::int32 var_222 = 1;
    wp::float32 var_223;
    wp::float32 var_224;
    wp::float32 var_225;
    const wp::int32 var_226 = 4;
    wp::int32 var_227;
    const wp::int32 var_228 = 2;
    wp::float32 var_229;
    const wp::int32 var_230 = 2;
    wp::float32 var_231;
    wp::float32 var_232;
    wp::float32 var_233;
    wp::vec_t<3, wp::float32> var_234;
    wp::vec_t<3, wp::float32> var_235;
    wp::float32 var_236;
    wp::float32 var_237;
    wp::vec_t<3, wp::float32> var_238;
    const wp::float32 var_239 = 0.5;
    wp::vec_t<3, wp::float32> var_240;
    wp::vec_t<3, wp::float32> var_241;
    wp::vec_t<3, wp::float32> var_242;
    const wp::int32 var_243 = 7;
    const wp::int32 var_244 = 1;
    wp::int32 var_245;
    const wp::int32 var_246 = 0;
    wp::float32 var_247;
    const wp::int32 var_248 = 0;
    wp::float32 var_249;
    wp::float32 var_250;
    wp::float32 var_251;
    const wp::int32 var_252 = 2;
    wp::int32 var_253;
    const wp::int32 var_254 = 1;
    wp::float32 var_255;
    const wp::int32 var_256 = 1;
    wp::float32 var_257;
    wp::float32 var_258;
    wp::float32 var_259;
    const wp::int32 var_260 = 4;
    wp::int32 var_261;
    const wp::int32 var_262 = 2;
    wp::float32 var_263;
    const wp::int32 var_264 = 2;
    wp::float32 var_265;
    wp::float32 var_266;
    wp::float32 var_267;
    wp::vec_t<3, wp::float32> var_268;
    wp::vec_t<3, wp::float32> var_269;
    wp::float32 var_270;
    wp::float32 var_271;
    wp::vec_t<3, wp::float32> var_272;
    const wp::float32 var_273 = 0.5;
    wp::vec_t<3, wp::float32> var_274;
    wp::vec_t<3, wp::float32> var_275;
    wp::vec_t<3, wp::float32> var_276;
    //---------
    // forward
    // def plane_box(                                                                         <L 343>
    // center_dist = wp.dot(box_pos - plane_pos, plane_normal)                                <L 366>
    var_0 = wp::sub(var_box_pos, var_plane_pos);
    var_1 = wp::dot(var_0, var_plane_normal);
    // dist = vec8f(MJ_MAXVAL)                                                                <L 368>
    var_3 = wp::vec_t<8, wp::float32>(var_2);
    // pos = mat83f()                                                                         <L 369>
    var_4 = wp::mat_t<8, 3, wp::float32>();
    // for i in range(8):                                                                     <L 372>
    // corner = wp.vec3(                                                                      <L 374>
    // wp.where(i & 1, box_size[0], -box_size[0]),                                            <L 375>
    var_7 = wp::bit_and(var_5, var_6);
    var_9 = wp::extract(var_box_size, var_8);
    var_11 = wp::extract(var_box_size, var_10);
    var_12 = wp::neg(var_11);
    var_13 = wp::where(var_7, var_9, var_12);
    // wp.where(i & 2, box_size[1], -box_size[1]),                                            <L 376>
    var_15 = wp::bit_and(var_5, var_14);
    var_17 = wp::extract(var_box_size, var_16);
    var_19 = wp::extract(var_box_size, var_18);
    var_20 = wp::neg(var_19);
    var_21 = wp::where(var_15, var_17, var_20);
    // wp.where(i & 4, box_size[2], -box_size[2]),                                            <L 377>
    var_23 = wp::bit_and(var_5, var_22);
    var_25 = wp::extract(var_box_size, var_24);
    var_27 = wp::extract(var_box_size, var_26);
    var_28 = wp::neg(var_27);
    var_29 = wp::where(var_23, var_25, var_28);
    var_30 = wp::vec_t<3, wp::float32>(var_13, var_21, var_29);
    // corner = box_rot * corner                                                              <L 381>
    var_31 = wp::mul(var_box_rot, var_30);
    // ldist = wp.dot(plane_normal, corner)                                                   <L 384>
    var_32 = wp::dot(var_plane_normal, var_31);
    // cdist = center_dist + ldist                                                            <L 385>
    var_33 = wp::add(var_1, var_32);
    // dist[i] = cdist                                                                        <L 387>
    wp::assign_inplace(var_3, var_5, var_33);
    // pos[i] = corner + box_pos - 0.5 * plane_normal * cdist                                 <L 388>
    var_34 = wp::add(var_31, var_box_pos);
    var_36 = wp::mul(var_35, var_plane_normal);
    var_37 = wp::mul(var_36, var_33);
    var_38 = wp::sub(var_34, var_37);
    wp::assign_inplace(var_4, var_5, var_38);
    // corner = wp.vec3(                                                                      <L 374>
    // wp.where(i & 1, box_size[0], -box_size[0]),                                            <L 375>
    var_41 = wp::bit_and(var_39, var_40);
    var_43 = wp::extract(var_box_size, var_42);
    var_45 = wp::extract(var_box_size, var_44);
    var_46 = wp::neg(var_45);
    var_47 = wp::where(var_41, var_43, var_46);
    // wp.where(i & 2, box_size[1], -box_size[1]),                                            <L 376>
    var_49 = wp::bit_and(var_39, var_48);
    var_51 = wp::extract(var_box_size, var_50);
    var_53 = wp::extract(var_box_size, var_52);
    var_54 = wp::neg(var_53);
    var_55 = wp::where(var_49, var_51, var_54);
    // wp.where(i & 4, box_size[2], -box_size[2]),                                            <L 377>
    var_57 = wp::bit_and(var_39, var_56);
    var_59 = wp::extract(var_box_size, var_58);
    var_61 = wp::extract(var_box_size, var_60);
    var_62 = wp::neg(var_61);
    var_63 = wp::where(var_57, var_59, var_62);
    var_64 = wp::vec_t<3, wp::float32>(var_47, var_55, var_63);
    // corner = box_rot * corner                                                              <L 381>
    var_65 = wp::mul(var_box_rot, var_64);
    // ldist = wp.dot(plane_normal, corner)                                                   <L 384>
    var_66 = wp::dot(var_plane_normal, var_65);
    // cdist = center_dist + ldist                                                            <L 385>
    var_67 = wp::add(var_1, var_66);
    // dist[i] = cdist                                                                        <L 387>
    wp::assign_inplace(var_3, var_39, var_67);
    // pos[i] = corner + box_pos - 0.5 * plane_normal * cdist                                 <L 388>
    var_68 = wp::add(var_65, var_box_pos);
    var_70 = wp::mul(var_69, var_plane_normal);
    var_71 = wp::mul(var_70, var_67);
    var_72 = wp::sub(var_68, var_71);
    wp::assign_inplace(var_4, var_39, var_72);
    // corner = wp.vec3(                                                                      <L 374>
    // wp.where(i & 1, box_size[0], -box_size[0]),                                            <L 375>
    var_75 = wp::bit_and(var_73, var_74);
    var_77 = wp::extract(var_box_size, var_76);
    var_79 = wp::extract(var_box_size, var_78);
    var_80 = wp::neg(var_79);
    var_81 = wp::where(var_75, var_77, var_80);
    // wp.where(i & 2, box_size[1], -box_size[1]),                                            <L 376>
    var_83 = wp::bit_and(var_73, var_82);
    var_85 = wp::extract(var_box_size, var_84);
    var_87 = wp::extract(var_box_size, var_86);
    var_88 = wp::neg(var_87);
    var_89 = wp::where(var_83, var_85, var_88);
    // wp.where(i & 4, box_size[2], -box_size[2]),                                            <L 377>
    var_91 = wp::bit_and(var_73, var_90);
    var_93 = wp::extract(var_box_size, var_92);
    var_95 = wp::extract(var_box_size, var_94);
    var_96 = wp::neg(var_95);
    var_97 = wp::where(var_91, var_93, var_96);
    var_98 = wp::vec_t<3, wp::float32>(var_81, var_89, var_97);
    // corner = box_rot * corner                                                              <L 381>
    var_99 = wp::mul(var_box_rot, var_98);
    // ldist = wp.dot(plane_normal, corner)                                                   <L 384>
    var_100 = wp::dot(var_plane_normal, var_99);
    // cdist = center_dist + ldist                                                            <L 385>
    var_101 = wp::add(var_1, var_100);
    // dist[i] = cdist                                                                        <L 387>
    wp::assign_inplace(var_3, var_73, var_101);
    // pos[i] = corner + box_pos - 0.5 * plane_normal * cdist                                 <L 388>
    var_102 = wp::add(var_99, var_box_pos);
    var_104 = wp::mul(var_103, var_plane_normal);
    var_105 = wp::mul(var_104, var_101);
    var_106 = wp::sub(var_102, var_105);
    wp::assign_inplace(var_4, var_73, var_106);
    // corner = wp.vec3(                                                                      <L 374>
    // wp.where(i & 1, box_size[0], -box_size[0]),                                            <L 375>
    var_109 = wp::bit_and(var_107, var_108);
    var_111 = wp::extract(var_box_size, var_110);
    var_113 = wp::extract(var_box_size, var_112);
    var_114 = wp::neg(var_113);
    var_115 = wp::where(var_109, var_111, var_114);
    // wp.where(i & 2, box_size[1], -box_size[1]),                                            <L 376>
    var_117 = wp::bit_and(var_107, var_116);
    var_119 = wp::extract(var_box_size, var_118);
    var_121 = wp::extract(var_box_size, var_120);
    var_122 = wp::neg(var_121);
    var_123 = wp::where(var_117, var_119, var_122);
    // wp.where(i & 4, box_size[2], -box_size[2]),                                            <L 377>
    var_125 = wp::bit_and(var_107, var_124);
    var_127 = wp::extract(var_box_size, var_126);
    var_129 = wp::extract(var_box_size, var_128);
    var_130 = wp::neg(var_129);
    var_131 = wp::where(var_125, var_127, var_130);
    var_132 = wp::vec_t<3, wp::float32>(var_115, var_123, var_131);
    // corner = box_rot * corner                                                              <L 381>
    var_133 = wp::mul(var_box_rot, var_132);
    // ldist = wp.dot(plane_normal, corner)                                                   <L 384>
    var_134 = wp::dot(var_plane_normal, var_133);
    // cdist = center_dist + ldist                                                            <L 385>
    var_135 = wp::add(var_1, var_134);
    // dist[i] = cdist                                                                        <L 387>
    wp::assign_inplace(var_3, var_107, var_135);
    // pos[i] = corner + box_pos - 0.5 * plane_normal * cdist                                 <L 388>
    var_136 = wp::add(var_133, var_box_pos);
    var_138 = wp::mul(var_137, var_plane_normal);
    var_139 = wp::mul(var_138, var_135);
    var_140 = wp::sub(var_136, var_139);
    wp::assign_inplace(var_4, var_107, var_140);
    // corner = wp.vec3(                                                                      <L 374>
    // wp.where(i & 1, box_size[0], -box_size[0]),                                            <L 375>
    var_143 = wp::bit_and(var_141, var_142);
    var_145 = wp::extract(var_box_size, var_144);
    var_147 = wp::extract(var_box_size, var_146);
    var_148 = wp::neg(var_147);
    var_149 = wp::where(var_143, var_145, var_148);
    // wp.where(i & 2, box_size[1], -box_size[1]),                                            <L 376>
    var_151 = wp::bit_and(var_141, var_150);
    var_153 = wp::extract(var_box_size, var_152);
    var_155 = wp::extract(var_box_size, var_154);
    var_156 = wp::neg(var_155);
    var_157 = wp::where(var_151, var_153, var_156);
    // wp.where(i & 4, box_size[2], -box_size[2]),                                            <L 377>
    var_159 = wp::bit_and(var_141, var_158);
    var_161 = wp::extract(var_box_size, var_160);
    var_163 = wp::extract(var_box_size, var_162);
    var_164 = wp::neg(var_163);
    var_165 = wp::where(var_159, var_161, var_164);
    var_166 = wp::vec_t<3, wp::float32>(var_149, var_157, var_165);
    // corner = box_rot * corner                                                              <L 381>
    var_167 = wp::mul(var_box_rot, var_166);
    // ldist = wp.dot(plane_normal, corner)                                                   <L 384>
    var_168 = wp::dot(var_plane_normal, var_167);
    // cdist = center_dist + ldist                                                            <L 385>
    var_169 = wp::add(var_1, var_168);
    // dist[i] = cdist                                                                        <L 387>
    wp::assign_inplace(var_3, var_141, var_169);
    // pos[i] = corner + box_pos - 0.5 * plane_normal * cdist                                 <L 388>
    var_170 = wp::add(var_167, var_box_pos);
    var_172 = wp::mul(var_171, var_plane_normal);
    var_173 = wp::mul(var_172, var_169);
    var_174 = wp::sub(var_170, var_173);
    wp::assign_inplace(var_4, var_141, var_174);
    // corner = wp.vec3(                                                                      <L 374>
    // wp.where(i & 1, box_size[0], -box_size[0]),                                            <L 375>
    var_177 = wp::bit_and(var_175, var_176);
    var_179 = wp::extract(var_box_size, var_178);
    var_181 = wp::extract(var_box_size, var_180);
    var_182 = wp::neg(var_181);
    var_183 = wp::where(var_177, var_179, var_182);
    // wp.where(i & 2, box_size[1], -box_size[1]),                                            <L 376>
    var_185 = wp::bit_and(var_175, var_184);
    var_187 = wp::extract(var_box_size, var_186);
    var_189 = wp::extract(var_box_size, var_188);
    var_190 = wp::neg(var_189);
    var_191 = wp::where(var_185, var_187, var_190);
    // wp.where(i & 4, box_size[2], -box_size[2]),                                            <L 377>
    var_193 = wp::bit_and(var_175, var_192);
    var_195 = wp::extract(var_box_size, var_194);
    var_197 = wp::extract(var_box_size, var_196);
    var_198 = wp::neg(var_197);
    var_199 = wp::where(var_193, var_195, var_198);
    var_200 = wp::vec_t<3, wp::float32>(var_183, var_191, var_199);
    // corner = box_rot * corner                                                              <L 381>
    var_201 = wp::mul(var_box_rot, var_200);
    // ldist = wp.dot(plane_normal, corner)                                                   <L 384>
    var_202 = wp::dot(var_plane_normal, var_201);
    // cdist = center_dist + ldist                                                            <L 385>
    var_203 = wp::add(var_1, var_202);
    // dist[i] = cdist                                                                        <L 387>
    wp::assign_inplace(var_3, var_175, var_203);
    // pos[i] = corner + box_pos - 0.5 * plane_normal * cdist                                 <L 388>
    var_204 = wp::add(var_201, var_box_pos);
    var_206 = wp::mul(var_205, var_plane_normal);
    var_207 = wp::mul(var_206, var_203);
    var_208 = wp::sub(var_204, var_207);
    wp::assign_inplace(var_4, var_175, var_208);
    // corner = wp.vec3(                                                                      <L 374>
    // wp.where(i & 1, box_size[0], -box_size[0]),                                            <L 375>
    var_211 = wp::bit_and(var_209, var_210);
    var_213 = wp::extract(var_box_size, var_212);
    var_215 = wp::extract(var_box_size, var_214);
    var_216 = wp::neg(var_215);
    var_217 = wp::where(var_211, var_213, var_216);
    // wp.where(i & 2, box_size[1], -box_size[1]),                                            <L 376>
    var_219 = wp::bit_and(var_209, var_218);
    var_221 = wp::extract(var_box_size, var_220);
    var_223 = wp::extract(var_box_size, var_222);
    var_224 = wp::neg(var_223);
    var_225 = wp::where(var_219, var_221, var_224);
    // wp.where(i & 4, box_size[2], -box_size[2]),                                            <L 377>
    var_227 = wp::bit_and(var_209, var_226);
    var_229 = wp::extract(var_box_size, var_228);
    var_231 = wp::extract(var_box_size, var_230);
    var_232 = wp::neg(var_231);
    var_233 = wp::where(var_227, var_229, var_232);
    var_234 = wp::vec_t<3, wp::float32>(var_217, var_225, var_233);
    // corner = box_rot * corner                                                              <L 381>
    var_235 = wp::mul(var_box_rot, var_234);
    // ldist = wp.dot(plane_normal, corner)                                                   <L 384>
    var_236 = wp::dot(var_plane_normal, var_235);
    // cdist = center_dist + ldist                                                            <L 385>
    var_237 = wp::add(var_1, var_236);
    // dist[i] = cdist                                                                        <L 387>
    wp::assign_inplace(var_3, var_209, var_237);
    // pos[i] = corner + box_pos - 0.5 * plane_normal * cdist                                 <L 388>
    var_238 = wp::add(var_235, var_box_pos);
    var_240 = wp::mul(var_239, var_plane_normal);
    var_241 = wp::mul(var_240, var_237);
    var_242 = wp::sub(var_238, var_241);
    wp::assign_inplace(var_4, var_209, var_242);
    // corner = wp.vec3(                                                                      <L 374>
    // wp.where(i & 1, box_size[0], -box_size[0]),                                            <L 375>
    var_245 = wp::bit_and(var_243, var_244);
    var_247 = wp::extract(var_box_size, var_246);
    var_249 = wp::extract(var_box_size, var_248);
    var_250 = wp::neg(var_249);
    var_251 = wp::where(var_245, var_247, var_250);
    // wp.where(i & 2, box_size[1], -box_size[1]),                                            <L 376>
    var_253 = wp::bit_and(var_243, var_252);
    var_255 = wp::extract(var_box_size, var_254);
    var_257 = wp::extract(var_box_size, var_256);
    var_258 = wp::neg(var_257);
    var_259 = wp::where(var_253, var_255, var_258);
    // wp.where(i & 4, box_size[2], -box_size[2]),                                            <L 377>
    var_261 = wp::bit_and(var_243, var_260);
    var_263 = wp::extract(var_box_size, var_262);
    var_265 = wp::extract(var_box_size, var_264);
    var_266 = wp::neg(var_265);
    var_267 = wp::where(var_261, var_263, var_266);
    var_268 = wp::vec_t<3, wp::float32>(var_251, var_259, var_267);
    // corner = box_rot * corner                                                              <L 381>
    var_269 = wp::mul(var_box_rot, var_268);
    // ldist = wp.dot(plane_normal, corner)                                                   <L 384>
    var_270 = wp::dot(var_plane_normal, var_269);
    // cdist = center_dist + ldist                                                            <L 385>
    var_271 = wp::add(var_1, var_270);
    // dist[i] = cdist                                                                        <L 387>
    wp::assign_inplace(var_3, var_243, var_271);
    // pos[i] = corner + box_pos - 0.5 * plane_normal * cdist                                 <L 388>
    var_272 = wp::add(var_269, var_box_pos);
    var_274 = wp::mul(var_273, var_plane_normal);
    var_275 = wp::mul(var_274, var_271);
    var_276 = wp::sub(var_272, var_275);
    wp::assign_inplace(var_4, var_243, var_276);
    // return dist, pos, plane_normal                                                         <L 390>
    ret_0 = var_3;
    ret_1 = var_4;
    ret_2 = var_plane_normal;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:692
static CUDA_CALLABLE void plane_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_plane,
    Geom_4022bfa7 var_box,
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
    wp::vec_t<3, wp::float32>* var_2;
    wp::mat_t<3, 3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32>* var_4;
    wp::vec_t<8, wp::float32> var_5;
    wp::mat_t<8, 3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::mat_t<3, 3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    const wp::int32 var_14 = 0;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::int32 var_17;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::int32 var_21;
    const wp::int32 var_22 = 2;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::int32 var_25;
    const wp::int32 var_26 = 3;
    wp::float32 var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::int32 var_29;
    const wp::int32 var_30 = 4;
    wp::float32 var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::int32 var_33;
    const wp::int32 var_34 = 5;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::int32 var_37;
    const wp::int32 var_38 = 6;
    wp::float32 var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::int32 var_41;
    const wp::int32 var_42 = 7;
    wp::float32 var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::int32 var_45;
    //---------
    // forward
    // def plane_box_wrapper(                                                                 <L 693>
    // dist, pos, normal = plane_box(plane.normal, plane.pos, box.pos, box.rot, box.size)       <L 729>
    var_0 = &((var_plane).normal);
    var_1 = &((var_plane).pos);
    var_2 = &((var_box).pos);
    var_3 = &((var_box).rot);
    var_4 = &((var_box).size);
    var_8 = wp::load(var_0);
    var_9 = wp::load(var_1);
    var_10 = wp::load(var_2);
    var_11 = wp::load(var_3);
    var_12 = wp::load(var_4);
    plane_box_0(var_8, var_9, var_10, var_11, var_12, var_5, var_6, var_7);
    // frame = make_frame(normal)                                                             <L 730>
    var_13 = make_frame_0(var_7);
    // for i in range(8):                                                                     <L 732>
    // write_contact(                                                                         <L 733>
    // naconmax_in,                                                                           <L 734>
    // i,                                                                                     <L 735>
    // dist[i],                                                                               <L 736>
    var_15 = wp::extract(var_5, var_14);
    // pos[i],                                                                                <L 737>
    var_16 = wp::extract(var_6, var_14);
    // frame,                                                                                 <L 738>
    // margin,                                                                                <L 739>
    // gap,                                                                                   <L 740>
    // condim,                                                                                <L 741>
    // friction,                                                                              <L 742>
    // solref,                                                                                <L 743>
    // solreffriction,                                                                        <L 744>
    // solimp,                                                                                <L 745>
    // adhesion,                                                                              <L 746>
    // geoms,                                                                                 <L 747>
    // pairid,                                                                                <L 748>
    // worldid,                                                                               <L 749>
    // contact_dist_out,                                                                      <L 750>
    // contact_pos_out,                                                                       <L 751>
    // contact_frame_out,                                                                     <L 752>
    // contact_includemargin_out,                                                             <L 753>
    // contact_friction_out,                                                                  <L 754>
    // contact_solref_out,                                                                    <L 755>
    // contact_solreffriction_out,                                                            <L 756>
    // contact_solimp_out,                                                                    <L 757>
    // contact_dim_out,                                                                       <L 758>
    // contact_geom_out,                                                                      <L 759>
    // contact_efc_address_out,                                                               <L 760>
    // contact_worldid_out,                                                                   <L 761>
    // contact_type_out,                                                                      <L 762>
    // contact_geomcollisionid_out,                                                           <L 763>
    // contact_adhesion_out,                                                                  <L 764>
    // nacon_out,                                                                             <L 765>
    var_17 = write_contact_0(var_naconmax_in, var_14, var_15, var_16, var_13, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 733>
    // naconmax_in,                                                                           <L 734>
    // i,                                                                                     <L 735>
    // dist[i],                                                                               <L 736>
    var_19 = wp::extract(var_5, var_18);
    // pos[i],                                                                                <L 737>
    var_20 = wp::extract(var_6, var_18);
    // frame,                                                                                 <L 738>
    // margin,                                                                                <L 739>
    // gap,                                                                                   <L 740>
    // condim,                                                                                <L 741>
    // friction,                                                                              <L 742>
    // solref,                                                                                <L 743>
    // solreffriction,                                                                        <L 744>
    // solimp,                                                                                <L 745>
    // adhesion,                                                                              <L 746>
    // geoms,                                                                                 <L 747>
    // pairid,                                                                                <L 748>
    // worldid,                                                                               <L 749>
    // contact_dist_out,                                                                      <L 750>
    // contact_pos_out,                                                                       <L 751>
    // contact_frame_out,                                                                     <L 752>
    // contact_includemargin_out,                                                             <L 753>
    // contact_friction_out,                                                                  <L 754>
    // contact_solref_out,                                                                    <L 755>
    // contact_solreffriction_out,                                                            <L 756>
    // contact_solimp_out,                                                                    <L 757>
    // contact_dim_out,                                                                       <L 758>
    // contact_geom_out,                                                                      <L 759>
    // contact_efc_address_out,                                                               <L 760>
    // contact_worldid_out,                                                                   <L 761>
    // contact_type_out,                                                                      <L 762>
    // contact_geomcollisionid_out,                                                           <L 763>
    // contact_adhesion_out,                                                                  <L 764>
    // nacon_out,                                                                             <L 765>
    var_21 = write_contact_0(var_naconmax_in, var_18, var_19, var_20, var_13, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 733>
    // naconmax_in,                                                                           <L 734>
    // i,                                                                                     <L 735>
    // dist[i],                                                                               <L 736>
    var_23 = wp::extract(var_5, var_22);
    // pos[i],                                                                                <L 737>
    var_24 = wp::extract(var_6, var_22);
    // frame,                                                                                 <L 738>
    // margin,                                                                                <L 739>
    // gap,                                                                                   <L 740>
    // condim,                                                                                <L 741>
    // friction,                                                                              <L 742>
    // solref,                                                                                <L 743>
    // solreffriction,                                                                        <L 744>
    // solimp,                                                                                <L 745>
    // adhesion,                                                                              <L 746>
    // geoms,                                                                                 <L 747>
    // pairid,                                                                                <L 748>
    // worldid,                                                                               <L 749>
    // contact_dist_out,                                                                      <L 750>
    // contact_pos_out,                                                                       <L 751>
    // contact_frame_out,                                                                     <L 752>
    // contact_includemargin_out,                                                             <L 753>
    // contact_friction_out,                                                                  <L 754>
    // contact_solref_out,                                                                    <L 755>
    // contact_solreffriction_out,                                                            <L 756>
    // contact_solimp_out,                                                                    <L 757>
    // contact_dim_out,                                                                       <L 758>
    // contact_geom_out,                                                                      <L 759>
    // contact_efc_address_out,                                                               <L 760>
    // contact_worldid_out,                                                                   <L 761>
    // contact_type_out,                                                                      <L 762>
    // contact_geomcollisionid_out,                                                           <L 763>
    // contact_adhesion_out,                                                                  <L 764>
    // nacon_out,                                                                             <L 765>
    var_25 = write_contact_0(var_naconmax_in, var_22, var_23, var_24, var_13, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 733>
    // naconmax_in,                                                                           <L 734>
    // i,                                                                                     <L 735>
    // dist[i],                                                                               <L 736>
    var_27 = wp::extract(var_5, var_26);
    // pos[i],                                                                                <L 737>
    var_28 = wp::extract(var_6, var_26);
    // frame,                                                                                 <L 738>
    // margin,                                                                                <L 739>
    // gap,                                                                                   <L 740>
    // condim,                                                                                <L 741>
    // friction,                                                                              <L 742>
    // solref,                                                                                <L 743>
    // solreffriction,                                                                        <L 744>
    // solimp,                                                                                <L 745>
    // adhesion,                                                                              <L 746>
    // geoms,                                                                                 <L 747>
    // pairid,                                                                                <L 748>
    // worldid,                                                                               <L 749>
    // contact_dist_out,                                                                      <L 750>
    // contact_pos_out,                                                                       <L 751>
    // contact_frame_out,                                                                     <L 752>
    // contact_includemargin_out,                                                             <L 753>
    // contact_friction_out,                                                                  <L 754>
    // contact_solref_out,                                                                    <L 755>
    // contact_solreffriction_out,                                                            <L 756>
    // contact_solimp_out,                                                                    <L 757>
    // contact_dim_out,                                                                       <L 758>
    // contact_geom_out,                                                                      <L 759>
    // contact_efc_address_out,                                                               <L 760>
    // contact_worldid_out,                                                                   <L 761>
    // contact_type_out,                                                                      <L 762>
    // contact_geomcollisionid_out,                                                           <L 763>
    // contact_adhesion_out,                                                                  <L 764>
    // nacon_out,                                                                             <L 765>
    var_29 = write_contact_0(var_naconmax_in, var_26, var_27, var_28, var_13, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 733>
    // naconmax_in,                                                                           <L 734>
    // i,                                                                                     <L 735>
    // dist[i],                                                                               <L 736>
    var_31 = wp::extract(var_5, var_30);
    // pos[i],                                                                                <L 737>
    var_32 = wp::extract(var_6, var_30);
    // frame,                                                                                 <L 738>
    // margin,                                                                                <L 739>
    // gap,                                                                                   <L 740>
    // condim,                                                                                <L 741>
    // friction,                                                                              <L 742>
    // solref,                                                                                <L 743>
    // solreffriction,                                                                        <L 744>
    // solimp,                                                                                <L 745>
    // adhesion,                                                                              <L 746>
    // geoms,                                                                                 <L 747>
    // pairid,                                                                                <L 748>
    // worldid,                                                                               <L 749>
    // contact_dist_out,                                                                      <L 750>
    // contact_pos_out,                                                                       <L 751>
    // contact_frame_out,                                                                     <L 752>
    // contact_includemargin_out,                                                             <L 753>
    // contact_friction_out,                                                                  <L 754>
    // contact_solref_out,                                                                    <L 755>
    // contact_solreffriction_out,                                                            <L 756>
    // contact_solimp_out,                                                                    <L 757>
    // contact_dim_out,                                                                       <L 758>
    // contact_geom_out,                                                                      <L 759>
    // contact_efc_address_out,                                                               <L 760>
    // contact_worldid_out,                                                                   <L 761>
    // contact_type_out,                                                                      <L 762>
    // contact_geomcollisionid_out,                                                           <L 763>
    // contact_adhesion_out,                                                                  <L 764>
    // nacon_out,                                                                             <L 765>
    var_33 = write_contact_0(var_naconmax_in, var_30, var_31, var_32, var_13, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 733>
    // naconmax_in,                                                                           <L 734>
    // i,                                                                                     <L 735>
    // dist[i],                                                                               <L 736>
    var_35 = wp::extract(var_5, var_34);
    // pos[i],                                                                                <L 737>
    var_36 = wp::extract(var_6, var_34);
    // frame,                                                                                 <L 738>
    // margin,                                                                                <L 739>
    // gap,                                                                                   <L 740>
    // condim,                                                                                <L 741>
    // friction,                                                                              <L 742>
    // solref,                                                                                <L 743>
    // solreffriction,                                                                        <L 744>
    // solimp,                                                                                <L 745>
    // adhesion,                                                                              <L 746>
    // geoms,                                                                                 <L 747>
    // pairid,                                                                                <L 748>
    // worldid,                                                                               <L 749>
    // contact_dist_out,                                                                      <L 750>
    // contact_pos_out,                                                                       <L 751>
    // contact_frame_out,                                                                     <L 752>
    // contact_includemargin_out,                                                             <L 753>
    // contact_friction_out,                                                                  <L 754>
    // contact_solref_out,                                                                    <L 755>
    // contact_solreffriction_out,                                                            <L 756>
    // contact_solimp_out,                                                                    <L 757>
    // contact_dim_out,                                                                       <L 758>
    // contact_geom_out,                                                                      <L 759>
    // contact_efc_address_out,                                                               <L 760>
    // contact_worldid_out,                                                                   <L 761>
    // contact_type_out,                                                                      <L 762>
    // contact_geomcollisionid_out,                                                           <L 763>
    // contact_adhesion_out,                                                                  <L 764>
    // nacon_out,                                                                             <L 765>
    var_37 = write_contact_0(var_naconmax_in, var_34, var_35, var_36, var_13, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 733>
    // naconmax_in,                                                                           <L 734>
    // i,                                                                                     <L 735>
    // dist[i],                                                                               <L 736>
    var_39 = wp::extract(var_5, var_38);
    // pos[i],                                                                                <L 737>
    var_40 = wp::extract(var_6, var_38);
    // frame,                                                                                 <L 738>
    // margin,                                                                                <L 739>
    // gap,                                                                                   <L 740>
    // condim,                                                                                <L 741>
    // friction,                                                                              <L 742>
    // solref,                                                                                <L 743>
    // solreffriction,                                                                        <L 744>
    // solimp,                                                                                <L 745>
    // adhesion,                                                                              <L 746>
    // geoms,                                                                                 <L 747>
    // pairid,                                                                                <L 748>
    // worldid,                                                                               <L 749>
    // contact_dist_out,                                                                      <L 750>
    // contact_pos_out,                                                                       <L 751>
    // contact_frame_out,                                                                     <L 752>
    // contact_includemargin_out,                                                             <L 753>
    // contact_friction_out,                                                                  <L 754>
    // contact_solref_out,                                                                    <L 755>
    // contact_solreffriction_out,                                                            <L 756>
    // contact_solimp_out,                                                                    <L 757>
    // contact_dim_out,                                                                       <L 758>
    // contact_geom_out,                                                                      <L 759>
    // contact_efc_address_out,                                                               <L 760>
    // contact_worldid_out,                                                                   <L 761>
    // contact_type_out,                                                                      <L 762>
    // contact_geomcollisionid_out,                                                           <L 763>
    // contact_adhesion_out,                                                                  <L 764>
    // nacon_out,                                                                             <L 765>
    var_41 = write_contact_0(var_naconmax_in, var_38, var_39, var_40, var_13, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 733>
    // naconmax_in,                                                                           <L 734>
    // i,                                                                                     <L 735>
    // dist[i],                                                                               <L 736>
    var_43 = wp::extract(var_5, var_42);
    // pos[i],                                                                                <L 737>
    var_44 = wp::extract(var_6, var_42);
    // frame,                                                                                 <L 738>
    // margin,                                                                                <L 739>
    // gap,                                                                                   <L 740>
    // condim,                                                                                <L 741>
    // friction,                                                                              <L 742>
    // solref,                                                                                <L 743>
    // solreffriction,                                                                        <L 744>
    // solimp,                                                                                <L 745>
    // adhesion,                                                                              <L 746>
    // geoms,                                                                                 <L 747>
    // pairid,                                                                                <L 748>
    // worldid,                                                                               <L 749>
    // contact_dist_out,                                                                      <L 750>
    // contact_pos_out,                                                                       <L 751>
    // contact_frame_out,                                                                     <L 752>
    // contact_includemargin_out,                                                             <L 753>
    // contact_friction_out,                                                                  <L 754>
    // contact_solref_out,                                                                    <L 755>
    // contact_solreffriction_out,                                                            <L 756>
    // contact_solimp_out,                                                                    <L 757>
    // contact_dim_out,                                                                       <L 758>
    // contact_geom_out,                                                                      <L 759>
    // contact_efc_address_out,                                                               <L 760>
    // contact_worldid_out,                                                                   <L 761>
    // contact_type_out,                                                                      <L 762>
    // contact_geomcollisionid_out,                                                           <L 763>
    // contact_adhesion_out,                                                                  <L 764>
    // nacon_out,                                                                             <L 765>
    var_45 = write_contact_0(var_naconmax_in, var_42, var_43, var_44, var_13, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:393
static CUDA_CALLABLE void sphere_cylinder_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_cylinder_pos,
    wp::vec_t<3, wp::float32> var_cylinder_axis,
    wp::float32 var_cylinder_radius,
    wp::float32 var_cylinder_half_height,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    bool var_6;
    wp::float32 var_7;
    bool var_8;
    bool var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    bool var_14;
    const bool var_15 = false;
    const bool var_16 = false;
    bool var_17;
    bool var_18;
    bool var_19;
    bool var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.0;
    bool var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    const wp::float32 var_38 = 1.0;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    const wp::float32 var_48 = 0.0;
    wp::float32 var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    //---------
    // forward
    // def sphere_cylinder(                                                                   <L 394>
    // vec = sphere_pos - cylinder_pos                                                        <L 418>
    var_0 = wp::sub(var_sphere_pos, var_cylinder_pos);
    // x = wp.dot(vec, cylinder_axis)                                                         <L 419>
    var_1 = wp::dot(var_0, var_cylinder_axis);
    // a_proj = cylinder_axis * x                                                             <L 421>
    var_2 = wp::mul(var_cylinder_axis, var_1);
    // p_proj = vec - a_proj                                                                  <L 422>
    var_3 = wp::sub(var_0, var_2);
    // p_proj_sqr = wp.dot(p_proj, p_proj)                                                    <L 423>
    var_4 = wp::dot(var_3, var_3);
    // collide_side = wp.abs(x) < cylinder_half_height                                        <L 425>
    var_5 = wp::abs(var_1);
    var_6 = (var_5 < var_cylinder_half_height);
    // collide_cap = p_proj_sqr < (cylinder_radius * cylinder_radius)                         <L 426>
    var_7 = wp::mul(var_cylinder_radius, var_cylinder_radius);
    var_8 = (var_4 < var_7);
    // if collide_side and collide_cap:                                                       <L 428>
    var_9 = var_6;
    if (var_9) {
        var_9 = var_9 && var_8;
    }
    if (var_9) {
        // dist_cap = cylinder_half_height - wp.abs(x)                                        <L 429>
        var_10 = wp::abs(var_1);
        var_11 = wp::sub(var_cylinder_half_height, var_10);
        // dist_radius = cylinder_radius - wp.sqrt(p_proj_sqr)                                <L 430>
        var_12 = wp::sqrt(var_4);
        var_13 = wp::sub(var_cylinder_radius, var_12);
        // if dist_cap < dist_radius:                                                         <L 432>
        var_14 = (var_11 < var_13);
        if (var_14) {
            // collide_side = False                                                           <L 433>
        }
        if (!var_14) {
            // collide_cap = False                                                            <L 435>
        }
        var_17 = wp::where(var_14, var_15, var_6);
        var_18 = wp::where(var_14, var_8, var_16);
    }
    var_19 = wp::where(var_9, var_17, var_6);
    var_20 = wp::where(var_9, var_18, var_8);
    // if collide_side:                                                                       <L 438>
    if (var_19) {
        // pos_target = cylinder_pos + a_proj                                                 <L 439>
        var_21 = wp::add(var_cylinder_pos, var_2);
        // return sphere_sphere(sphere_pos, sphere_radius, pos_target, cylinder_radius)       <L 440>
        sphere_sphere_0(var_sphere_pos, var_sphere_radius, var_21, var_cylinder_radius, var_22, var_23, var_24);
        ret_0 = var_22;
        ret_1 = var_23;
        ret_2 = var_24;
        return;
    }
    if (!var_19) {
        // elif collide_cap:                                                                  <L 442>
        if (var_20) {
            // if x > 0.0:                                                                    <L 443>
            var_26 = (var_1 > var_25);
            if (var_26) {
                // pos_cap = cylinder_pos + cylinder_axis * cylinder_half_height              <L 445>
                var_27 = wp::mul(var_cylinder_axis, var_cylinder_half_height);
                var_28 = wp::add(var_cylinder_pos, var_27);
                // plane_normal = cylinder_axis                                               <L 446>
                var_29 = wp::copy(var_cylinder_axis);
            }
            if (!var_26) {
                // pos_cap = cylinder_pos - cylinder_axis * cylinder_half_height              <L 449>
                var_30 = wp::mul(var_cylinder_axis, var_cylinder_half_height);
                var_31 = wp::sub(var_cylinder_pos, var_30);
                // plane_normal = -cylinder_axis                                              <L 450>
                var_32 = wp::neg(var_cylinder_axis);
            }
            var_33 = wp::where(var_26, var_28, var_31);
            var_34 = wp::where(var_26, var_29, var_32);
            // dist, pos = plane_sphere(plane_normal, pos_cap, sphere_pos, sphere_radius)       <L 452>
            plane_sphere_0(var_34, var_33, var_sphere_pos, var_sphere_radius, var_35, var_36);
            // return dist, pos, -plane_normal  # flip normal after position calculation       <L 453>
            var_37 = wp::neg(var_34);
            ret_0 = var_35;
            ret_1 = var_36;
            ret_2 = var_37;
            return;
        }
        if (!var_20) {
            // inv_len = safe_div(1.0, wp.sqrt(p_proj_sqr))                                   <L 456>
            var_39 = wp::sqrt(var_4);
            var_40 = safe_div_0(var_38, var_39);
            // p_proj = p_proj * (cylinder_radius * inv_len)                                  <L 457>
            var_41 = wp::mul(var_cylinder_radius, var_40);
            var_42 = wp::mul(var_3, var_41);
            // cap_offset = cylinder_axis * (wp.sign(x) * cylinder_half_height)               <L 459>
            var_43 = wp::sign(var_1);
            var_44 = wp::mul(var_43, var_cylinder_half_height);
            var_45 = wp::mul(var_cylinder_axis, var_44);
            // pos_corner = cylinder_pos + cap_offset + p_proj                                <L 460>
            var_46 = wp::add(var_cylinder_pos, var_45);
            var_47 = wp::add(var_46, var_42);
            // return sphere_sphere(sphere_pos, sphere_radius, pos_corner, 0.0)               <L 462>
            sphere_sphere_0(var_sphere_pos, var_sphere_radius, var_47, var_48, var_49, var_50, var_51);
            ret_0 = var_49;
            ret_1 = var_50;
            ret_2 = var_51;
            return;
        }
        var_52 = wp::where(var_20, var_3, var_42);
    }
    var_53 = wp::where(var_19, var_3, var_52);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:846
static CUDA_CALLABLE void sphere_cylinder_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_sphere,
    Geom_4022bfa7 var_cylinder,
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
    // def sphere_cylinder_wrapper(                                                           <L 847>
    // cylinder_axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2], cylinder.rot[2, 2])       <L 884>
    var_0 = &((var_cylinder).rot);
    var_4 = wp::load(var_0);
    var_3 = wp::extract(var_4, var_1, var_2);
    var_5 = &((var_cylinder).rot);
    var_9 = wp::load(var_5);
    var_8 = wp::extract(var_9, var_6, var_7);
    var_10 = &((var_cylinder).rot);
    var_14 = wp::load(var_10);
    var_13 = wp::extract(var_14, var_11, var_12);
    var_15 = wp::vec_t<3, wp::float32>(var_3, var_8, var_13);
    // dist, pos, normal = sphere_cylinder(                                                   <L 886>
    // sphere.pos,                                                                            <L 887>
    var_16 = &((var_sphere).pos);
    // sphere.size[0],  # sphere radius                                                       <L 888>
    var_17 = &((var_sphere).size);
    var_20 = wp::load(var_17);
    var_19 = wp::extract(var_20, var_18);
    // cylinder.pos,                                                                          <L 889>
    var_21 = &((var_cylinder).pos);
    // cylinder_axis,                                                                         <L 890>
    // cylinder.size[0],  # cylinder radius                                                   <L 891>
    var_22 = &((var_cylinder).size);
    var_25 = wp::load(var_22);
    var_24 = wp::extract(var_25, var_23);
    // cylinder.size[1],  # cylinder half_height                                              <L 892>
    var_26 = &((var_cylinder).size);
    var_29 = wp::load(var_26);
    var_28 = wp::extract(var_29, var_27);
    var_33 = wp::load(var_16);
    var_34 = wp::load(var_21);
    sphere_cylinder_0(var_33, var_19, var_34, var_15, var_24, var_28, var_30, var_31, var_32);
    // write_contact(                                                                         <L 895>
    // naconmax_in,                                                                           <L 896>
    // 0,                                                                                     <L 897>
    // dist,                                                                                  <L 898>
    // pos,                                                                                   <L 899>
    // make_frame(normal),                                                                    <L 900>
    var_36 = make_frame_0(var_32);
    // margin,                                                                                <L 901>
    // gap,                                                                                   <L 902>
    // condim,                                                                                <L 903>
    // friction,                                                                              <L 904>
    // solref,                                                                                <L 905>
    // solreffriction,                                                                        <L 906>
    // solimp,                                                                                <L 907>
    // adhesion,                                                                              <L 908>
    // geoms,                                                                                 <L 909>
    // pairid,                                                                                <L 910>
    // worldid,                                                                               <L 911>
    // contact_dist_out,                                                                      <L 912>
    // contact_pos_out,                                                                       <L 913>
    // contact_frame_out,                                                                     <L 914>
    // contact_includemargin_out,                                                             <L 915>
    // contact_friction_out,                                                                  <L 916>
    // contact_solref_out,                                                                    <L 917>
    // contact_solreffriction_out,                                                            <L 918>
    // contact_solimp_out,                                                                    <L 919>
    // contact_dim_out,                                                                       <L 920>
    // contact_geom_out,                                                                      <L 921>
    // contact_efc_address_out,                                                               <L 922>
    // contact_worldid_out,                                                                   <L 923>
    // contact_type_out,                                                                      <L 924>
    // contact_geomcollisionid_out,                                                           <L 925>
    // contact_adhesion_out,                                                                  <L 926>
    // nacon_out,                                                                             <L 927>
    var_37 = write_contact_0(var_naconmax_in, var_35, var_30, var_31, var_36, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:1049
static CUDA_CALLABLE void sphere_box_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 1e-15;
    bool var_10;
    const wp::float32 var_11 = 2.0;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    const wp::int32 var_14 = 1;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 2;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::int32 var_22;
    const wp::int32 var_23 = 0;
    const wp::int32 var_24 = 2;
    wp::int32 var_25;
    const wp::float32 var_26 = 1.0;
    const wp::float32 var_27 = -1.0;
    wp::float32 var_28;
    const wp::int32 var_29 = 2;
    wp::int32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    const wp::int32 var_33 = 2;
    wp::int32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    bool var_38;
    wp::float32 var_39;
    wp::int32 var_40;
    wp::float32 var_41;
    wp::int32 var_42;
    const wp::int32 var_43 = 1;
    const wp::int32 var_44 = 2;
    wp::int32 var_45;
    const wp::float32 var_46 = 1.0;
    const wp::float32 var_47 = -1.0;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    wp::int32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::int32 var_53 = 2;
    wp::int32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    bool var_58;
    wp::float32 var_59;
    wp::int32 var_60;
    wp::float32 var_61;
    wp::int32 var_62;
    const wp::int32 var_63 = 2;
    const wp::int32 var_64 = 2;
    wp::int32 var_65;
    const wp::float32 var_66 = 1.0;
    const wp::float32 var_67 = -1.0;
    wp::float32 var_68;
    const wp::int32 var_69 = 2;
    wp::int32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    const wp::int32 var_73 = 2;
    wp::int32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    bool var_78;
    wp::float32 var_79;
    wp::int32 var_80;
    wp::float32 var_81;
    wp::int32 var_82;
    const wp::int32 var_83 = 3;
    const wp::int32 var_84 = 2;
    wp::int32 var_85;
    const wp::float32 var_86 = 1.0;
    const wp::float32 var_87 = -1.0;
    wp::float32 var_88;
    const wp::int32 var_89 = 2;
    wp::int32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::int32 var_93 = 2;
    wp::int32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    bool var_98;
    wp::float32 var_99;
    wp::int32 var_100;
    wp::float32 var_101;
    wp::int32 var_102;
    const wp::int32 var_103 = 4;
    const wp::int32 var_104 = 2;
    wp::int32 var_105;
    const wp::float32 var_106 = 1.0;
    const wp::float32 var_107 = -1.0;
    wp::float32 var_108;
    const wp::int32 var_109 = 2;
    wp::int32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::int32 var_113 = 2;
    wp::int32 var_114;
    wp::float32 var_115;
    wp::float32 var_116;
    wp::float32 var_117;
    bool var_118;
    wp::float32 var_119;
    wp::int32 var_120;
    wp::float32 var_121;
    wp::int32 var_122;
    const wp::int32 var_123 = 5;
    const wp::int32 var_124 = 2;
    wp::int32 var_125;
    const wp::float32 var_126 = 1.0;
    const wp::float32 var_127 = -1.0;
    wp::float32 var_128;
    const wp::int32 var_129 = 2;
    wp::int32 var_130;
    wp::float32 var_131;
    wp::float32 var_132;
    const wp::int32 var_133 = 2;
    wp::int32 var_134;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    bool var_138;
    wp::float32 var_139;
    wp::int32 var_140;
    wp::float32 var_141;
    wp::int32 var_142;
    const wp::float32 var_143 = 0.0;
    wp::vec_t<3, wp::float32> var_144;
    const wp::int32 var_145 = 2;
    wp::int32 var_146;
    const wp::float32 var_147 = -1.0;
    const wp::float32 var_148 = 1.0;
    wp::float32 var_149;
    const wp::int32 var_150 = 2;
    wp::int32 var_151;
    wp::float32 var_152;
    wp::vec_t<3, wp::float32> var_153;
    const wp::float32 var_154 = 2.0;
    wp::vec_t<3, wp::float32> var_155;
    wp::vec_t<3, wp::float32> var_156;
    wp::vec_t<3, wp::float32> var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    wp::vec_t<3, wp::float32> var_160;
    wp::vec_t<3, wp::float32> var_161;
    const wp::float32 var_162 = 0.5;
    wp::vec_t<3, wp::float32> var_163;
    wp::vec_t<3, wp::float32> var_164;
    wp::vec_t<3, wp::float32> var_165;
    wp::float32 var_166;
    wp::vec_t<3, wp::float32> var_167;
    wp::vec_t<3, wp::float32> var_168;
    wp::float32 var_169;
    wp::vec_t<3, wp::float32> var_170;
    wp::vec_t<3, wp::float32> var_171;
    //---------
    // forward
    // def sphere_box(                                                                        <L 1050>
    // center = wp.transpose(box_rot) @ (sphere_pos - box_pos)                                <L 1072>
    var_0 = wp::transpose(var_box_rot);
    var_1 = wp::sub(var_sphere_pos, var_box_pos);
    var_2 = wp::mul(var_0, var_1);
    // clamped = wp.max(-box_size, wp.min(box_size, center))                                  <L 1074>
    var_3 = wp::neg(var_box_size);
    var_4 = wp::min(var_box_size, var_2);
    var_5 = wp::max(var_3, var_4);
    // clamped_dir, dist = normalize_with_norm(clamped - center)                              <L 1075>
    var_6 = wp::sub(var_5, var_2);
    normalize_with_norm_0(var_6, var_7, var_8);
    // if dist <= MJ_MINVAL:                                                                  <L 1078>
    var_10 = (var_8 <= var_9);
    if (var_10) {
        // closest = 2.0 * (box_size[0] + box_size[1] + box_size[2])                          <L 1079>
        var_13 = wp::extract(var_box_size, var_12);
        var_15 = wp::extract(var_box_size, var_14);
        var_16 = wp::add(var_13, var_15);
        var_18 = wp::extract(var_box_size, var_17);
        var_19 = wp::add(var_16, var_18);
        var_20 = wp::mul(var_11, var_19);
        // k = wp.int32(0)                                                                    <L 1080>
        var_22 = wp::int32(var_21);
        // for i in range(6):                                                                 <L 1081>
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1082>
        var_25 = wp::mod(var_23, var_24);
        var_28 = wp::where(var_25, var_26, var_27);
        var_30 = wp::floordiv(var_23, var_29);
        var_31 = wp::extract(var_box_size, var_30);
        var_32 = wp::mul(var_28, var_31);
        var_34 = wp::floordiv(var_23, var_33);
        var_35 = wp::extract(var_2, var_34);
        var_36 = wp::sub(var_32, var_35);
        var_37 = wp::abs(var_36);
        // if closest > face_dist:                                                            <L 1083>
        var_38 = (var_20 > var_37);
        if (var_38) {
            // closest = face_dist                                                            <L 1084>
            var_39 = wp::copy(var_37);
            // k = i                                                                          <L 1085>
            var_40 = wp::copy(var_23);
        }
        var_41 = wp::where(var_38, var_39, var_20);
        var_42 = wp::where(var_38, var_40, var_22);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1082>
        var_45 = wp::mod(var_43, var_44);
        var_48 = wp::where(var_45, var_46, var_47);
        var_50 = wp::floordiv(var_43, var_49);
        var_51 = wp::extract(var_box_size, var_50);
        var_52 = wp::mul(var_48, var_51);
        var_54 = wp::floordiv(var_43, var_53);
        var_55 = wp::extract(var_2, var_54);
        var_56 = wp::sub(var_52, var_55);
        var_57 = wp::abs(var_56);
        // if closest > face_dist:                                                            <L 1083>
        var_58 = (var_41 > var_57);
        if (var_58) {
            // closest = face_dist                                                            <L 1084>
            var_59 = wp::copy(var_57);
            // k = i                                                                          <L 1085>
            var_60 = wp::copy(var_43);
        }
        var_61 = wp::where(var_58, var_59, var_41);
        var_62 = wp::where(var_58, var_60, var_42);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1082>
        var_65 = wp::mod(var_63, var_64);
        var_68 = wp::where(var_65, var_66, var_67);
        var_70 = wp::floordiv(var_63, var_69);
        var_71 = wp::extract(var_box_size, var_70);
        var_72 = wp::mul(var_68, var_71);
        var_74 = wp::floordiv(var_63, var_73);
        var_75 = wp::extract(var_2, var_74);
        var_76 = wp::sub(var_72, var_75);
        var_77 = wp::abs(var_76);
        // if closest > face_dist:                                                            <L 1083>
        var_78 = (var_61 > var_77);
        if (var_78) {
            // closest = face_dist                                                            <L 1084>
            var_79 = wp::copy(var_77);
            // k = i                                                                          <L 1085>
            var_80 = wp::copy(var_63);
        }
        var_81 = wp::where(var_78, var_79, var_61);
        var_82 = wp::where(var_78, var_80, var_62);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1082>
        var_85 = wp::mod(var_83, var_84);
        var_88 = wp::where(var_85, var_86, var_87);
        var_90 = wp::floordiv(var_83, var_89);
        var_91 = wp::extract(var_box_size, var_90);
        var_92 = wp::mul(var_88, var_91);
        var_94 = wp::floordiv(var_83, var_93);
        var_95 = wp::extract(var_2, var_94);
        var_96 = wp::sub(var_92, var_95);
        var_97 = wp::abs(var_96);
        // if closest > face_dist:                                                            <L 1083>
        var_98 = (var_81 > var_97);
        if (var_98) {
            // closest = face_dist                                                            <L 1084>
            var_99 = wp::copy(var_97);
            // k = i                                                                          <L 1085>
            var_100 = wp::copy(var_83);
        }
        var_101 = wp::where(var_98, var_99, var_81);
        var_102 = wp::where(var_98, var_100, var_82);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1082>
        var_105 = wp::mod(var_103, var_104);
        var_108 = wp::where(var_105, var_106, var_107);
        var_110 = wp::floordiv(var_103, var_109);
        var_111 = wp::extract(var_box_size, var_110);
        var_112 = wp::mul(var_108, var_111);
        var_114 = wp::floordiv(var_103, var_113);
        var_115 = wp::extract(var_2, var_114);
        var_116 = wp::sub(var_112, var_115);
        var_117 = wp::abs(var_116);
        // if closest > face_dist:                                                            <L 1083>
        var_118 = (var_101 > var_117);
        if (var_118) {
            // closest = face_dist                                                            <L 1084>
            var_119 = wp::copy(var_117);
            // k = i                                                                          <L 1085>
            var_120 = wp::copy(var_103);
        }
        var_121 = wp::where(var_118, var_119, var_101);
        var_122 = wp::where(var_118, var_120, var_102);
        // face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])       <L 1082>
        var_125 = wp::mod(var_123, var_124);
        var_128 = wp::where(var_125, var_126, var_127);
        var_130 = wp::floordiv(var_123, var_129);
        var_131 = wp::extract(var_box_size, var_130);
        var_132 = wp::mul(var_128, var_131);
        var_134 = wp::floordiv(var_123, var_133);
        var_135 = wp::extract(var_2, var_134);
        var_136 = wp::sub(var_132, var_135);
        var_137 = wp::abs(var_136);
        // if closest > face_dist:                                                            <L 1083>
        var_138 = (var_121 > var_137);
        if (var_138) {
            // closest = face_dist                                                            <L 1084>
            var_139 = wp::copy(var_137);
            // k = i                                                                          <L 1085>
            var_140 = wp::copy(var_123);
        }
        var_141 = wp::where(var_138, var_139, var_121);
        var_142 = wp::where(var_138, var_140, var_122);
        // nearest = wp.vec3(0.0)                                                             <L 1087>
        var_144 = wp::vec_t<3, wp::float32>(var_143);
        // nearest[k // 2] = wp.where(k % 2, -1.0, 1.0)                                       <L 1088>
        var_146 = wp::mod(var_142, var_145);
        var_149 = wp::where(var_146, var_147, var_148);
        var_151 = wp::floordiv(var_142, var_150);
        wp::assign_inplace(var_144, var_151, var_149);
        // pos = center + nearest * (sphere_radius - closest) / 2.0                           <L 1089>
        var_152 = wp::sub(var_sphere_radius, var_141);
        var_153 = wp::mul(var_144, var_152);
        var_155 = wp::div(var_153, var_154);
        var_156 = wp::add(var_2, var_155);
        // contact_normal = box_rot @ nearest                                                 <L 1090>
        var_157 = wp::mul(var_box_rot, var_144);
        // contact_distance = -closest - sphere_radius                                        <L 1091>
        var_158 = wp::neg(var_141);
        var_159 = wp::sub(var_158, var_sphere_radius);
    }
    if (!var_10) {
        // deepest = center + clamped_dir * sphere_radius                                     <L 1094>
        var_160 = wp::mul(var_7, var_sphere_radius);
        var_161 = wp::add(var_2, var_160);
        // pos = 0.5 * (clamped + deepest)                                                    <L 1095>
        var_163 = wp::add(var_5, var_161);
        var_164 = wp::mul(var_162, var_163);
        // contact_normal = box_rot @ clamped_dir                                             <L 1096>
        var_165 = wp::mul(var_box_rot, var_7);
        // contact_distance = dist - sphere_radius                                            <L 1097>
        var_166 = wp::sub(var_8, var_sphere_radius);
    }
    var_167 = wp::where(var_10, var_156, var_164);
    var_168 = wp::where(var_10, var_157, var_165);
    var_169 = wp::where(var_10, var_159, var_166);
    // contact_position = box_pos + box_rot @ pos                                             <L 1099>
    var_170 = wp::mul(var_box_rot, var_167);
    var_171 = wp::add(var_box_pos, var_170);
    // return contact_distance, contact_position, contact_normal                              <L 1101>
    ret_0 = var_169;
    ret_1 = var_171;
    ret_2 = var_168;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:1018
static CUDA_CALLABLE void sphere_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_sphere,
    Geom_4022bfa7 var_box,
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
    wp::mat_t<3, 3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32>* var_7;
    wp::float32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::mat_t<3, 3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    const wp::int32 var_15 = 0;
    wp::mat_t<3, 3, wp::float32> var_16;
    wp::int32 var_17;
    //---------
    // forward
    // def sphere_box_wrapper(                                                                <L 1019>
    // dist, pos, normal = sphere_box(sphere.pos, sphere.size[0], box.pos, box.rot, box.size)       <L 1054>
    var_0 = &((var_sphere).pos);
    var_1 = &((var_sphere).size);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = &((var_box).pos);
    var_6 = &((var_box).rot);
    var_7 = &((var_box).size);
    var_11 = wp::load(var_0);
    var_12 = wp::load(var_5);
    var_13 = wp::load(var_6);
    var_14 = wp::load(var_7);
    sphere_box_0(var_11, var_3, var_12, var_13, var_14, var_8, var_9, var_10);
    // write_contact(                                                                         <L 1056>
    // naconmax_in,                                                                           <L 1057>
    // 0,                                                                                     <L 1058>
    // dist,                                                                                  <L 1059>
    // pos,                                                                                   <L 1060>
    // make_frame(normal),                                                                    <L 1061>
    var_16 = make_frame_0(var_10);
    // margin,                                                                                <L 1062>
    // gap,                                                                                   <L 1063>
    // condim,                                                                                <L 1064>
    // friction,                                                                              <L 1065>
    // solref,                                                                                <L 1066>
    // solreffriction,                                                                        <L 1067>
    // solimp,                                                                                <L 1068>
    // adhesion,                                                                              <L 1069>
    // geoms,                                                                                 <L 1070>
    // pairid,                                                                                <L 1071>
    // worldid,                                                                               <L 1072>
    // contact_dist_out,                                                                      <L 1073>
    // contact_pos_out,                                                                       <L 1074>
    // contact_frame_out,                                                                     <L 1075>
    // contact_includemargin_out,                                                             <L 1076>
    // contact_friction_out,                                                                  <L 1077>
    // contact_solref_out,                                                                    <L 1078>
    // contact_solreffriction_out,                                                            <L 1079>
    // contact_solimp_out,                                                                    <L 1080>
    // contact_dim_out,                                                                       <L 1081>
    // contact_geom_out,                                                                      <L 1082>
    // contact_efc_address_out,                                                               <L 1083>
    // contact_worldid_out,                                                                   <L 1084>
    // contact_type_out,                                                                      <L 1085>
    // contact_geomcollisionid_out,                                                           <L 1086>
    // contact_adhesion_out,                                                                  <L 1087>
    // nacon_out,                                                                             <L 1088>
    var_17 = write_contact_0(var_naconmax_in, var_15, var_8, var_9, var_16, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:1104
static CUDA_CALLABLE void capsule_box_0(
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2)
{
    //---------
    // primal vars
    wp::mat_t<3, 3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    const wp::int32 var_5 = 0;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 2;
    const wp::int32 var_11 = 1;
    wp::float32 var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    const wp::int32 var_18 = 4;
    const wp::int32 var_19 = 2;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    wp::int32 var_25;
    const wp::float32 var_26 = 1e+32;
    wp::float32 var_27;
    const wp::int32 var_28 = -12;
    wp::float32 var_29;
    const wp::int32 var_30 = -4;
    wp::int32 var_31;
    const wp::int32 var_32 = -12;
    wp::int32 var_33;
    const wp::int32 var_34 = -1;
    const bool var_35 = true;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    const wp::int32 var_40 = 0;
    wp::int32 var_41;
    const wp::int32 var_42 = -1;
    wp::int32 var_43;
    const wp::int32 var_44 = 0;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    bool var_48;
    const wp::int32 var_49 = 1;
    wp::int32 var_50;
    wp::int32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    bool var_56;
    const wp::int32 var_57 = 1;
    wp::int32 var_58;
    wp::int32 var_59;
    wp::float32 var_60;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::int32 var_63;
    wp::int32 var_64;
    const wp::int32 var_65 = 1;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    bool var_69;
    const wp::int32 var_70 = 1;
    wp::int32 var_71;
    wp::int32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    bool var_77;
    const wp::int32 var_78 = 1;
    wp::int32 var_79;
    wp::int32 var_80;
    wp::float32 var_81;
    wp::int32 var_82;
    wp::int32 var_83;
    wp::int32 var_84;
    wp::int32 var_85;
    const wp::int32 var_86 = 2;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    bool var_90;
    const wp::int32 var_91 = 1;
    wp::int32 var_92;
    wp::int32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    bool var_98;
    const wp::int32 var_99 = 1;
    wp::int32 var_100;
    wp::int32 var_101;
    wp::float32 var_102;
    wp::int32 var_103;
    wp::int32 var_104;
    wp::int32 var_105;
    wp::int32 var_106;
    const wp::int32 var_107 = 1;
    bool var_108;
    const bool var_109 = false;
    bool var_110;
    wp::vec_t<3, wp::float32> var_111;
    wp::float32 var_112;
    bool var_113;
    wp::float32 var_114;
    wp::float32 var_115;
    const wp::int32 var_116 = -2;
    wp::int32 var_117;
    wp::int32 var_118;
    wp::float32 var_119;
    wp::float32 var_120;
    wp::int32 var_121;
    wp::int32 var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::int32 var_125;
    wp::int32 var_126;
    const wp::int32 var_127 = 1;
    const bool var_128 = true;
    wp::float32 var_129;
    wp::vec_t<3, wp::float32> var_130;
    wp::vec_t<3, wp::float32> var_131;
    wp::vec_t<3, wp::float32> var_132;
    const wp::int32 var_133 = 0;
    wp::int32 var_134;
    const wp::int32 var_135 = -1;
    wp::int32 var_136;
    const wp::int32 var_137 = 0;
    wp::float32 var_138;
    wp::float32 var_139;
    wp::float32 var_140;
    bool var_141;
    const wp::int32 var_142 = 1;
    wp::int32 var_143;
    wp::int32 var_144;
    wp::float32 var_145;
    wp::float32 var_146;
    wp::float32 var_147;
    wp::float32 var_148;
    bool var_149;
    const wp::int32 var_150 = 1;
    wp::int32 var_151;
    wp::int32 var_152;
    wp::float32 var_153;
    wp::int32 var_154;
    wp::int32 var_155;
    wp::int32 var_156;
    wp::int32 var_157;
    const wp::int32 var_158 = 1;
    wp::float32 var_159;
    wp::float32 var_160;
    wp::float32 var_161;
    bool var_162;
    const wp::int32 var_163 = 1;
    wp::int32 var_164;
    wp::int32 var_165;
    wp::float32 var_166;
    wp::float32 var_167;
    wp::float32 var_168;
    wp::float32 var_169;
    bool var_170;
    const wp::int32 var_171 = 1;
    wp::int32 var_172;
    wp::int32 var_173;
    wp::float32 var_174;
    wp::int32 var_175;
    wp::int32 var_176;
    wp::int32 var_177;
    wp::int32 var_178;
    const wp::int32 var_179 = 2;
    wp::float32 var_180;
    wp::float32 var_181;
    wp::float32 var_182;
    bool var_183;
    const wp::int32 var_184 = 1;
    wp::int32 var_185;
    wp::int32 var_186;
    wp::float32 var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    wp::float32 var_190;
    bool var_191;
    const wp::int32 var_192 = 1;
    wp::int32 var_193;
    wp::int32 var_194;
    wp::float32 var_195;
    wp::int32 var_196;
    wp::int32 var_197;
    wp::int32 var_198;
    wp::int32 var_199;
    const wp::int32 var_200 = 1;
    bool var_201;
    const bool var_202 = false;
    bool var_203;
    wp::vec_t<3, wp::float32> var_204;
    wp::float32 var_205;
    bool var_206;
    wp::float32 var_207;
    wp::float32 var_208;
    const wp::int32 var_209 = -2;
    wp::int32 var_210;
    wp::int32 var_211;
    wp::float32 var_212;
    wp::float32 var_213;
    wp::int32 var_214;
    wp::int32 var_215;
    wp::float32 var_216;
    wp::float32 var_217;
    wp::int32 var_218;
    wp::int32 var_219;
    wp::float32 var_220;
    const wp::int32 var_221 = -123;
    wp::int32 var_222;
    const wp::int32 var_223 = -123;
    wp::int32 var_224;
    const wp::float32 var_225 = 0.0;
    wp::float32 var_226;
    const wp::int32 var_227 = 0;
    const wp::int32 var_228 = 0;
    const bool var_229 = true;
    const wp::int32 var_230 = 1;
    wp::int32 var_231;
    wp::int32 var_232;
    const wp::int32 var_233 = 0;
    bool var_234;
    const bool var_235 = false;
    bool var_236;
    const wp::int32 var_237 = -123;
    wp::int32 var_238;
    const wp::int32 var_239 = 1;
    wp::int32 var_240;
    const wp::float32 var_241 = 1.0;
    const wp::float32 var_242 = -1.0;
    wp::float32 var_243;
    const wp::int32 var_244 = 2;
    wp::int32 var_245;
    const wp::float32 var_246 = 1.0;
    const wp::float32 var_247 = -1.0;
    wp::float32 var_248;
    const wp::int32 var_249 = 4;
    wp::int32 var_250;
    const wp::float32 var_251 = 1.0;
    const wp::float32 var_252 = -1.0;
    wp::float32 var_253;
    wp::vec_t<3, wp::float32> var_254;
    wp::vec_t<3, wp::float32> var_255;
    const wp::float32 var_256 = 0.0;
    wp::vec_t<3, wp::float32> var_257;
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
    wp::float32 var_272;
    wp::float32 var_273;
    wp::float32 var_274;
    const wp::float32 var_275 = 1e-15;
    bool var_276;
    const bool var_277 = false;
    bool var_278;
    const wp::float32 var_279 = 1.0;
    wp::float32 var_280;
    wp::float32 var_281;
    wp::float32 var_282;
    wp::float32 var_283;
    wp::float32 var_284;
    wp::float32 var_285;
    wp::float32 var_286;
    wp::float32 var_287;
    wp::float32 var_288;
    wp::float32 var_289;
    wp::float32 var_290;
    const wp::int32 var_291 = 1;
    wp::int32 var_292;
    const wp::int32 var_293 = 1;
    wp::int32 var_294;
    const wp::int32 var_295 = 1;
    bool var_296;
    const wp::float32 var_297 = 1.0;
    const wp::int32 var_298 = 2;
    wp::float32 var_299;
    wp::float32 var_300;
    const wp::int32 var_301 = -1;
    bool var_302;
    const wp::float32 var_303 = -1.0;
    const wp::int32 var_304 = 0;
    wp::float32 var_305;
    wp::float32 var_306;
    wp::float32 var_307;
    wp::float32 var_308;
    wp::int32 var_309;
    wp::float32 var_310;
    wp::float32 var_311;
    wp::int32 var_312;
    const wp::float32 var_313 = 1.0;
    bool var_314;
    bool var_315;
    const wp::float32 var_316 = -1.0;
    bool var_317;
    const wp::float32 var_318 = 1.0;
    const wp::int32 var_319 = 2;
    wp::float32 var_320;
    wp::float32 var_321;
    const wp::float32 var_322 = -1.0;
    const wp::int32 var_323 = 0;
    wp::float32 var_324;
    wp::float32 var_325;
    wp::float32 var_326;
    wp::float32 var_327;
    wp::int32 var_328;
    const wp::int32 var_329 = 1;
    bool var_330;
    const wp::float32 var_331 = 1.0;
    const wp::int32 var_332 = 2;
    const wp::int32 var_333 = -1;
    bool var_334;
    const wp::float32 var_335 = -1.0;
    const wp::int32 var_336 = 0;
    wp::float32 var_337;
    wp::int32 var_338;
    wp::float32 var_339;
    wp::int32 var_340;
    wp::float32 var_341;
    wp::float32 var_342;
    wp::int32 var_343;
    wp::int32 var_344;
    wp::vec_t<3, wp::float32> var_345;
    wp::vec_t<3, wp::float32> var_346;
    wp::float32 var_347;
    wp::float32 var_348;
    const wp::int32 var_349 = 3;
    wp::int32 var_350;
    wp::int32 var_351;
    wp::float32 var_352;
    wp::float32 var_353;
    bool var_354;
    wp::float32 var_355;
    wp::float32 var_356;
    wp::float32 var_357;
    const wp::int32 var_358 = 6;
    wp::int32 var_359;
    const wp::int32 var_360 = 1;
    wp::int32 var_361;
    wp::int32 var_362;
    wp::int32 var_363;
    wp::int32 var_364;
    wp::int32 var_365;
    wp::float32 var_366;
    wp::float32 var_367;
    wp::int32 var_368;
    wp::int32 var_369;
    wp::int32 var_370;
    wp::float32 var_371;
    wp::int32 var_372;
    wp::float32 var_373;
    wp::float32 var_374;
    wp::int32 var_375;
    wp::int32 var_376;
    wp::int32 var_377;
    wp::float32 var_378;
    wp::int32 var_379;
    wp::vec_t<3, wp::float32> var_380;
    wp::float32 var_381;
    wp::float32 var_382;
    wp::int32 var_383;
    wp::int32 var_384;
    wp::int32 var_385;
    wp::float32 var_386;
    bool var_387;
    const wp::int32 var_388 = 1;
    const bool var_389 = true;
    const wp::int32 var_390 = 1;
    wp::int32 var_391;
    wp::int32 var_392;
    const wp::int32 var_393 = 0;
    bool var_394;
    const bool var_395 = false;
    bool var_396;
    const wp::int32 var_397 = -123;
    wp::int32 var_398;
    const wp::int32 var_399 = 1;
    wp::int32 var_400;
    const wp::float32 var_401 = 1.0;
    const wp::float32 var_402 = -1.0;
    wp::float32 var_403;
    const wp::int32 var_404 = 2;
    wp::int32 var_405;
    const wp::float32 var_406 = 1.0;
    const wp::float32 var_407 = -1.0;
    wp::float32 var_408;
    const wp::int32 var_409 = 4;
    wp::int32 var_410;
    const wp::float32 var_411 = 1.0;
    const wp::float32 var_412 = -1.0;
    wp::float32 var_413;
    wp::vec_t<3, wp::float32> var_414;
    wp::vec_t<3, wp::float32> var_415;
    const wp::float32 var_416 = 0.0;
    wp::vec_t<3, wp::float32> var_417;
    wp::float32 var_418;
    wp::float32 var_419;
    wp::float32 var_420;
    wp::float32 var_421;
    wp::float32 var_422;
    wp::float32 var_423;
    wp::float32 var_424;
    wp::float32 var_425;
    wp::float32 var_426;
    wp::float32 var_427;
    wp::float32 var_428;
    wp::float32 var_429;
    wp::float32 var_430;
    wp::float32 var_431;
    wp::float32 var_432;
    wp::float32 var_433;
    wp::float32 var_434;
    bool var_435;
    const bool var_436 = false;
    bool var_437;
    const wp::float32 var_438 = 1.0;
    wp::float32 var_439;
    wp::float32 var_440;
    wp::float32 var_441;
    wp::float32 var_442;
    wp::float32 var_443;
    wp::float32 var_444;
    wp::float32 var_445;
    wp::float32 var_446;
    wp::float32 var_447;
    wp::float32 var_448;
    wp::float32 var_449;
    const wp::int32 var_450 = 1;
    wp::int32 var_451;
    const wp::int32 var_452 = 1;
    wp::int32 var_453;
    const wp::int32 var_454 = 1;
    bool var_455;
    const wp::float32 var_456 = 1.0;
    const wp::int32 var_457 = 2;
    wp::float32 var_458;
    wp::float32 var_459;
    const wp::int32 var_460 = -1;
    bool var_461;
    const wp::float32 var_462 = -1.0;
    const wp::int32 var_463 = 0;
    wp::float32 var_464;
    wp::float32 var_465;
    wp::float32 var_466;
    wp::float32 var_467;
    wp::int32 var_468;
    wp::float32 var_469;
    wp::float32 var_470;
    wp::int32 var_471;
    const wp::float32 var_472 = 1.0;
    bool var_473;
    bool var_474;
    const wp::float32 var_475 = -1.0;
    bool var_476;
    const wp::float32 var_477 = 1.0;
    const wp::int32 var_478 = 2;
    wp::float32 var_479;
    wp::float32 var_480;
    const wp::float32 var_481 = -1.0;
    const wp::int32 var_482 = 0;
    wp::float32 var_483;
    wp::float32 var_484;
    wp::float32 var_485;
    wp::float32 var_486;
    wp::int32 var_487;
    const wp::int32 var_488 = 1;
    bool var_489;
    const wp::float32 var_490 = 1.0;
    const wp::int32 var_491 = 2;
    const wp::int32 var_492 = -1;
    bool var_493;
    const wp::float32 var_494 = -1.0;
    const wp::int32 var_495 = 0;
    wp::float32 var_496;
    wp::int32 var_497;
    wp::float32 var_498;
    wp::int32 var_499;
    wp::float32 var_500;
    wp::float32 var_501;
    wp::int32 var_502;
    wp::int32 var_503;
    wp::vec_t<3, wp::float32> var_504;
    wp::vec_t<3, wp::float32> var_505;
    wp::float32 var_506;
    wp::float32 var_507;
    const wp::int32 var_508 = 3;
    wp::int32 var_509;
    wp::int32 var_510;
    wp::float32 var_511;
    wp::float32 var_512;
    bool var_513;
    wp::float32 var_514;
    wp::float32 var_515;
    wp::float32 var_516;
    const wp::int32 var_517 = 6;
    wp::int32 var_518;
    const wp::int32 var_519 = 1;
    wp::int32 var_520;
    wp::int32 var_521;
    wp::int32 var_522;
    wp::int32 var_523;
    wp::int32 var_524;
    wp::float32 var_525;
    wp::float32 var_526;
    wp::int32 var_527;
    wp::int32 var_528;
    wp::int32 var_529;
    wp::float32 var_530;
    wp::int32 var_531;
    wp::float32 var_532;
    wp::float32 var_533;
    wp::int32 var_534;
    wp::int32 var_535;
    wp::int32 var_536;
    wp::float32 var_537;
    wp::int32 var_538;
    wp::vec_t<3, wp::float32> var_539;
    wp::float32 var_540;
    wp::float32 var_541;
    wp::float32 var_542;
    wp::int32 var_543;
    wp::int32 var_544;
    bool var_545;
    wp::int32 var_546;
    wp::float32 var_547;
    wp::float32 var_548;
    wp::float32 var_549;
    wp::int32 var_550;
    wp::int32 var_551;
    wp::int32 var_552;
    wp::float32 var_553;
    bool var_554;
    wp::int32 var_555;
    wp::vec_t<3, wp::float32> var_556;
    wp::vec_t<3, wp::float32> var_557;
    wp::float32 var_558;
    wp::float32 var_559;
    wp::float32 var_560;
    wp::float32 var_561;
    wp::float32 var_562;
    wp::float32 var_563;
    wp::float32 var_564;
    wp::float32 var_565;
    wp::float32 var_566;
    wp::int32 var_567;
    wp::int32 var_568;
    bool var_569;
    wp::int32 var_570;
    wp::float32 var_571;
    const wp::int32 var_572 = 2;
    const bool var_573 = true;
    const wp::int32 var_574 = 1;
    wp::int32 var_575;
    wp::int32 var_576;
    const wp::int32 var_577 = 0;
    bool var_578;
    const bool var_579 = false;
    bool var_580;
    const wp::int32 var_581 = -123;
    wp::int32 var_582;
    const wp::int32 var_583 = 1;
    wp::int32 var_584;
    const wp::float32 var_585 = 1.0;
    const wp::float32 var_586 = -1.0;
    wp::float32 var_587;
    const wp::int32 var_588 = 2;
    wp::int32 var_589;
    const wp::float32 var_590 = 1.0;
    const wp::float32 var_591 = -1.0;
    wp::float32 var_592;
    const wp::int32 var_593 = 4;
    wp::int32 var_594;
    const wp::float32 var_595 = 1.0;
    const wp::float32 var_596 = -1.0;
    wp::float32 var_597;
    wp::vec_t<3, wp::float32> var_598;
    wp::vec_t<3, wp::float32> var_599;
    const wp::float32 var_600 = 0.0;
    wp::vec_t<3, wp::float32> var_601;
    wp::float32 var_602;
    wp::float32 var_603;
    wp::float32 var_604;
    wp::float32 var_605;
    wp::float32 var_606;
    wp::float32 var_607;
    wp::float32 var_608;
    wp::float32 var_609;
    wp::float32 var_610;
    wp::float32 var_611;
    wp::float32 var_612;
    wp::float32 var_613;
    wp::float32 var_614;
    wp::float32 var_615;
    wp::float32 var_616;
    wp::float32 var_617;
    wp::float32 var_618;
    bool var_619;
    const bool var_620 = false;
    bool var_621;
    const wp::float32 var_622 = 1.0;
    wp::float32 var_623;
    wp::float32 var_624;
    wp::float32 var_625;
    wp::float32 var_626;
    wp::float32 var_627;
    wp::float32 var_628;
    wp::float32 var_629;
    wp::float32 var_630;
    wp::float32 var_631;
    wp::float32 var_632;
    wp::float32 var_633;
    const wp::int32 var_634 = 1;
    wp::int32 var_635;
    const wp::int32 var_636 = 1;
    wp::int32 var_637;
    const wp::int32 var_638 = 1;
    bool var_639;
    const wp::float32 var_640 = 1.0;
    const wp::int32 var_641 = 2;
    wp::float32 var_642;
    wp::float32 var_643;
    const wp::int32 var_644 = -1;
    bool var_645;
    const wp::float32 var_646 = -1.0;
    const wp::int32 var_647 = 0;
    wp::float32 var_648;
    wp::float32 var_649;
    wp::float32 var_650;
    wp::float32 var_651;
    wp::int32 var_652;
    wp::float32 var_653;
    wp::float32 var_654;
    wp::int32 var_655;
    const wp::float32 var_656 = 1.0;
    bool var_657;
    bool var_658;
    const wp::float32 var_659 = -1.0;
    bool var_660;
    const wp::float32 var_661 = 1.0;
    const wp::int32 var_662 = 2;
    wp::float32 var_663;
    wp::float32 var_664;
    const wp::float32 var_665 = -1.0;
    const wp::int32 var_666 = 0;
    wp::float32 var_667;
    wp::float32 var_668;
    wp::float32 var_669;
    wp::float32 var_670;
    wp::int32 var_671;
    const wp::int32 var_672 = 1;
    bool var_673;
    const wp::float32 var_674 = 1.0;
    const wp::int32 var_675 = 2;
    const wp::int32 var_676 = -1;
    bool var_677;
    const wp::float32 var_678 = -1.0;
    const wp::int32 var_679 = 0;
    wp::float32 var_680;
    wp::int32 var_681;
    wp::float32 var_682;
    wp::int32 var_683;
    wp::float32 var_684;
    wp::float32 var_685;
    wp::int32 var_686;
    wp::int32 var_687;
    wp::vec_t<3, wp::float32> var_688;
    wp::vec_t<3, wp::float32> var_689;
    wp::float32 var_690;
    wp::float32 var_691;
    const wp::int32 var_692 = 3;
    wp::int32 var_693;
    wp::int32 var_694;
    wp::float32 var_695;
    wp::float32 var_696;
    bool var_697;
    wp::float32 var_698;
    wp::float32 var_699;
    wp::float32 var_700;
    const wp::int32 var_701 = 6;
    wp::int32 var_702;
    const wp::int32 var_703 = 1;
    wp::int32 var_704;
    wp::int32 var_705;
    wp::int32 var_706;
    wp::int32 var_707;
    wp::int32 var_708;
    wp::float32 var_709;
    wp::float32 var_710;
    wp::int32 var_711;
    wp::int32 var_712;
    wp::int32 var_713;
    wp::float32 var_714;
    wp::int32 var_715;
    wp::float32 var_716;
    wp::float32 var_717;
    wp::int32 var_718;
    wp::int32 var_719;
    wp::int32 var_720;
    wp::float32 var_721;
    wp::int32 var_722;
    wp::vec_t<3, wp::float32> var_723;
    wp::float32 var_724;
    wp::float32 var_725;
    wp::float32 var_726;
    wp::int32 var_727;
    wp::int32 var_728;
    bool var_729;
    wp::int32 var_730;
    wp::float32 var_731;
    wp::float32 var_732;
    wp::float32 var_733;
    wp::int32 var_734;
    wp::int32 var_735;
    wp::int32 var_736;
    wp::float32 var_737;
    bool var_738;
    wp::int32 var_739;
    wp::vec_t<3, wp::float32> var_740;
    wp::vec_t<3, wp::float32> var_741;
    wp::float32 var_742;
    wp::float32 var_743;
    wp::float32 var_744;
    wp::float32 var_745;
    wp::float32 var_746;
    wp::float32 var_747;
    wp::float32 var_748;
    wp::float32 var_749;
    wp::float32 var_750;
    wp::int32 var_751;
    wp::int32 var_752;
    bool var_753;
    wp::int32 var_754;
    wp::float32 var_755;
    const wp::int32 var_756 = 1;
    const wp::int32 var_757 = 0;
    const bool var_758 = true;
    const wp::int32 var_759 = 1;
    wp::int32 var_760;
    wp::int32 var_761;
    const wp::int32 var_762 = 0;
    bool var_763;
    const bool var_764 = false;
    bool var_765;
    const wp::int32 var_766 = -123;
    wp::int32 var_767;
    const wp::int32 var_768 = 1;
    wp::int32 var_769;
    const wp::float32 var_770 = 1.0;
    const wp::float32 var_771 = -1.0;
    wp::float32 var_772;
    const wp::int32 var_773 = 2;
    wp::int32 var_774;
    const wp::float32 var_775 = 1.0;
    const wp::float32 var_776 = -1.0;
    wp::float32 var_777;
    const wp::int32 var_778 = 4;
    wp::int32 var_779;
    const wp::float32 var_780 = 1.0;
    const wp::float32 var_781 = -1.0;
    wp::float32 var_782;
    wp::vec_t<3, wp::float32> var_783;
    wp::vec_t<3, wp::float32> var_784;
    const wp::float32 var_785 = 0.0;
    wp::vec_t<3, wp::float32> var_786;
    wp::float32 var_787;
    wp::float32 var_788;
    wp::float32 var_789;
    wp::float32 var_790;
    wp::float32 var_791;
    wp::float32 var_792;
    wp::float32 var_793;
    wp::float32 var_794;
    wp::float32 var_795;
    wp::float32 var_796;
    wp::float32 var_797;
    wp::float32 var_798;
    wp::float32 var_799;
    wp::float32 var_800;
    wp::float32 var_801;
    wp::float32 var_802;
    wp::float32 var_803;
    bool var_804;
    const bool var_805 = false;
    bool var_806;
    const wp::float32 var_807 = 1.0;
    wp::float32 var_808;
    wp::float32 var_809;
    wp::float32 var_810;
    wp::float32 var_811;
    wp::float32 var_812;
    wp::float32 var_813;
    wp::float32 var_814;
    wp::float32 var_815;
    wp::float32 var_816;
    wp::float32 var_817;
    wp::float32 var_818;
    const wp::int32 var_819 = 1;
    wp::int32 var_820;
    const wp::int32 var_821 = 1;
    wp::int32 var_822;
    const wp::int32 var_823 = 1;
    bool var_824;
    const wp::float32 var_825 = 1.0;
    const wp::int32 var_826 = 2;
    wp::float32 var_827;
    wp::float32 var_828;
    const wp::int32 var_829 = -1;
    bool var_830;
    const wp::float32 var_831 = -1.0;
    const wp::int32 var_832 = 0;
    wp::float32 var_833;
    wp::float32 var_834;
    wp::float32 var_835;
    wp::float32 var_836;
    wp::int32 var_837;
    wp::float32 var_838;
    wp::float32 var_839;
    wp::int32 var_840;
    const wp::float32 var_841 = 1.0;
    bool var_842;
    bool var_843;
    const wp::float32 var_844 = -1.0;
    bool var_845;
    const wp::float32 var_846 = 1.0;
    const wp::int32 var_847 = 2;
    wp::float32 var_848;
    wp::float32 var_849;
    const wp::float32 var_850 = -1.0;
    const wp::int32 var_851 = 0;
    wp::float32 var_852;
    wp::float32 var_853;
    wp::float32 var_854;
    wp::float32 var_855;
    wp::int32 var_856;
    const wp::int32 var_857 = 1;
    bool var_858;
    const wp::float32 var_859 = 1.0;
    const wp::int32 var_860 = 2;
    const wp::int32 var_861 = -1;
    bool var_862;
    const wp::float32 var_863 = -1.0;
    const wp::int32 var_864 = 0;
    wp::float32 var_865;
    wp::int32 var_866;
    wp::float32 var_867;
    wp::int32 var_868;
    wp::float32 var_869;
    wp::float32 var_870;
    wp::int32 var_871;
    wp::int32 var_872;
    wp::vec_t<3, wp::float32> var_873;
    wp::vec_t<3, wp::float32> var_874;
    wp::float32 var_875;
    wp::float32 var_876;
    const wp::int32 var_877 = 3;
    wp::int32 var_878;
    wp::int32 var_879;
    wp::float32 var_880;
    wp::float32 var_881;
    bool var_882;
    wp::float32 var_883;
    wp::float32 var_884;
    wp::float32 var_885;
    const wp::int32 var_886 = 6;
    wp::int32 var_887;
    const wp::int32 var_888 = 1;
    wp::int32 var_889;
    wp::int32 var_890;
    wp::int32 var_891;
    wp::int32 var_892;
    wp::int32 var_893;
    wp::float32 var_894;
    wp::float32 var_895;
    wp::int32 var_896;
    wp::int32 var_897;
    wp::int32 var_898;
    wp::float32 var_899;
    wp::int32 var_900;
    wp::float32 var_901;
    wp::float32 var_902;
    wp::int32 var_903;
    wp::int32 var_904;
    wp::int32 var_905;
    wp::float32 var_906;
    wp::int32 var_907;
    wp::vec_t<3, wp::float32> var_908;
    wp::float32 var_909;
    wp::float32 var_910;
    wp::float32 var_911;
    wp::int32 var_912;
    wp::int32 var_913;
    bool var_914;
    wp::int32 var_915;
    wp::float32 var_916;
    wp::float32 var_917;
    wp::float32 var_918;
    wp::int32 var_919;
    wp::int32 var_920;
    wp::int32 var_921;
    wp::float32 var_922;
    wp::int32 var_923;
    wp::vec_t<3, wp::float32> var_924;
    wp::vec_t<3, wp::float32> var_925;
    wp::float32 var_926;
    wp::float32 var_927;
    wp::float32 var_928;
    wp::float32 var_929;
    wp::float32 var_930;
    wp::float32 var_931;
    wp::float32 var_932;
    wp::float32 var_933;
    wp::float32 var_934;
    wp::int32 var_935;
    wp::int32 var_936;
    bool var_937;
    wp::int32 var_938;
    wp::float32 var_939;
    bool var_940;
    const wp::int32 var_941 = 1;
    const bool var_942 = true;
    const wp::int32 var_943 = 1;
    wp::int32 var_944;
    wp::int32 var_945;
    const wp::int32 var_946 = 0;
    bool var_947;
    const bool var_948 = false;
    bool var_949;
    const wp::int32 var_950 = -123;
    wp::int32 var_951;
    const wp::int32 var_952 = 1;
    wp::int32 var_953;
    const wp::float32 var_954 = 1.0;
    const wp::float32 var_955 = -1.0;
    wp::float32 var_956;
    const wp::int32 var_957 = 2;
    wp::int32 var_958;
    const wp::float32 var_959 = 1.0;
    const wp::float32 var_960 = -1.0;
    wp::float32 var_961;
    const wp::int32 var_962 = 4;
    wp::int32 var_963;
    const wp::float32 var_964 = 1.0;
    const wp::float32 var_965 = -1.0;
    wp::float32 var_966;
    wp::vec_t<3, wp::float32> var_967;
    wp::vec_t<3, wp::float32> var_968;
    const wp::float32 var_969 = 0.0;
    wp::vec_t<3, wp::float32> var_970;
    wp::float32 var_971;
    wp::float32 var_972;
    wp::float32 var_973;
    wp::float32 var_974;
    wp::float32 var_975;
    wp::float32 var_976;
    wp::float32 var_977;
    wp::float32 var_978;
    wp::float32 var_979;
    wp::float32 var_980;
    wp::float32 var_981;
    wp::float32 var_982;
    wp::float32 var_983;
    wp::float32 var_984;
    wp::float32 var_985;
    wp::float32 var_986;
    wp::float32 var_987;
    bool var_988;
    const bool var_989 = false;
    bool var_990;
    const wp::float32 var_991 = 1.0;
    wp::float32 var_992;
    wp::float32 var_993;
    wp::float32 var_994;
    wp::float32 var_995;
    wp::float32 var_996;
    wp::float32 var_997;
    wp::float32 var_998;
    wp::float32 var_999;
    wp::float32 var_1000;
    wp::float32 var_1001;
    wp::float32 var_1002;
    const wp::int32 var_1003 = 1;
    wp::int32 var_1004;
    const wp::int32 var_1005 = 1;
    wp::int32 var_1006;
    const wp::int32 var_1007 = 1;
    bool var_1008;
    const wp::float32 var_1009 = 1.0;
    const wp::int32 var_1010 = 2;
    wp::float32 var_1011;
    wp::float32 var_1012;
    const wp::int32 var_1013 = -1;
    bool var_1014;
    const wp::float32 var_1015 = -1.0;
    const wp::int32 var_1016 = 0;
    wp::float32 var_1017;
    wp::float32 var_1018;
    wp::float32 var_1019;
    wp::float32 var_1020;
    wp::int32 var_1021;
    wp::float32 var_1022;
    wp::float32 var_1023;
    wp::int32 var_1024;
    const wp::float32 var_1025 = 1.0;
    bool var_1026;
    bool var_1027;
    const wp::float32 var_1028 = -1.0;
    bool var_1029;
    const wp::float32 var_1030 = 1.0;
    const wp::int32 var_1031 = 2;
    wp::float32 var_1032;
    wp::float32 var_1033;
    const wp::float32 var_1034 = -1.0;
    const wp::int32 var_1035 = 0;
    wp::float32 var_1036;
    wp::float32 var_1037;
    wp::float32 var_1038;
    wp::float32 var_1039;
    wp::int32 var_1040;
    const wp::int32 var_1041 = 1;
    bool var_1042;
    const wp::float32 var_1043 = 1.0;
    const wp::int32 var_1044 = 2;
    const wp::int32 var_1045 = -1;
    bool var_1046;
    const wp::float32 var_1047 = -1.0;
    const wp::int32 var_1048 = 0;
    wp::float32 var_1049;
    wp::int32 var_1050;
    wp::float32 var_1051;
    wp::int32 var_1052;
    wp::float32 var_1053;
    wp::float32 var_1054;
    wp::int32 var_1055;
    wp::int32 var_1056;
    wp::vec_t<3, wp::float32> var_1057;
    wp::vec_t<3, wp::float32> var_1058;
    wp::float32 var_1059;
    wp::float32 var_1060;
    const wp::int32 var_1061 = 3;
    wp::int32 var_1062;
    wp::int32 var_1063;
    wp::float32 var_1064;
    wp::float32 var_1065;
    bool var_1066;
    wp::float32 var_1067;
    wp::float32 var_1068;
    wp::float32 var_1069;
    const wp::int32 var_1070 = 6;
    wp::int32 var_1071;
    const wp::int32 var_1072 = 1;
    wp::int32 var_1073;
    wp::int32 var_1074;
    wp::int32 var_1075;
    wp::int32 var_1076;
    wp::int32 var_1077;
    wp::float32 var_1078;
    wp::float32 var_1079;
    wp::int32 var_1080;
    wp::int32 var_1081;
    wp::int32 var_1082;
    wp::float32 var_1083;
    wp::int32 var_1084;
    wp::float32 var_1085;
    wp::float32 var_1086;
    wp::int32 var_1087;
    wp::int32 var_1088;
    wp::int32 var_1089;
    wp::float32 var_1090;
    wp::int32 var_1091;
    wp::vec_t<3, wp::float32> var_1092;
    wp::float32 var_1093;
    wp::float32 var_1094;
    wp::float32 var_1095;
    wp::int32 var_1096;
    wp::int32 var_1097;
    bool var_1098;
    wp::int32 var_1099;
    wp::float32 var_1100;
    wp::float32 var_1101;
    wp::float32 var_1102;
    wp::int32 var_1103;
    wp::int32 var_1104;
    wp::int32 var_1105;
    wp::float32 var_1106;
    wp::int32 var_1107;
    wp::vec_t<3, wp::float32> var_1108;
    wp::vec_t<3, wp::float32> var_1109;
    wp::float32 var_1110;
    wp::float32 var_1111;
    wp::float32 var_1112;
    wp::float32 var_1113;
    wp::float32 var_1114;
    wp::float32 var_1115;
    wp::float32 var_1116;
    wp::float32 var_1117;
    wp::float32 var_1118;
    wp::int32 var_1119;
    wp::int32 var_1120;
    bool var_1121;
    wp::int32 var_1122;
    wp::float32 var_1123;
    bool var_1124;
    const wp::int32 var_1125 = 2;
    const bool var_1126 = true;
    const wp::int32 var_1127 = 1;
    wp::int32 var_1128;
    wp::int32 var_1129;
    const wp::int32 var_1130 = 0;
    bool var_1131;
    const bool var_1132 = false;
    bool var_1133;
    const wp::int32 var_1134 = -123;
    wp::int32 var_1135;
    const wp::int32 var_1136 = 1;
    wp::int32 var_1137;
    const wp::float32 var_1138 = 1.0;
    const wp::float32 var_1139 = -1.0;
    wp::float32 var_1140;
    const wp::int32 var_1141 = 2;
    wp::int32 var_1142;
    const wp::float32 var_1143 = 1.0;
    const wp::float32 var_1144 = -1.0;
    wp::float32 var_1145;
    const wp::int32 var_1146 = 4;
    wp::int32 var_1147;
    const wp::float32 var_1148 = 1.0;
    const wp::float32 var_1149 = -1.0;
    wp::float32 var_1150;
    wp::vec_t<3, wp::float32> var_1151;
    wp::vec_t<3, wp::float32> var_1152;
    const wp::float32 var_1153 = 0.0;
    wp::vec_t<3, wp::float32> var_1154;
    wp::float32 var_1155;
    wp::float32 var_1156;
    wp::float32 var_1157;
    wp::float32 var_1158;
    wp::float32 var_1159;
    wp::float32 var_1160;
    wp::float32 var_1161;
    wp::float32 var_1162;
    wp::float32 var_1163;
    wp::float32 var_1164;
    wp::float32 var_1165;
    wp::float32 var_1166;
    wp::float32 var_1167;
    wp::float32 var_1168;
    wp::float32 var_1169;
    wp::float32 var_1170;
    wp::float32 var_1171;
    bool var_1172;
    const bool var_1173 = false;
    bool var_1174;
    const wp::float32 var_1175 = 1.0;
    wp::float32 var_1176;
    wp::float32 var_1177;
    wp::float32 var_1178;
    wp::float32 var_1179;
    wp::float32 var_1180;
    wp::float32 var_1181;
    wp::float32 var_1182;
    wp::float32 var_1183;
    wp::float32 var_1184;
    wp::float32 var_1185;
    wp::float32 var_1186;
    const wp::int32 var_1187 = 1;
    wp::int32 var_1188;
    const wp::int32 var_1189 = 1;
    wp::int32 var_1190;
    const wp::int32 var_1191 = 1;
    bool var_1192;
    const wp::float32 var_1193 = 1.0;
    const wp::int32 var_1194 = 2;
    wp::float32 var_1195;
    wp::float32 var_1196;
    const wp::int32 var_1197 = -1;
    bool var_1198;
    const wp::float32 var_1199 = -1.0;
    const wp::int32 var_1200 = 0;
    wp::float32 var_1201;
    wp::float32 var_1202;
    wp::float32 var_1203;
    wp::float32 var_1204;
    wp::int32 var_1205;
    wp::float32 var_1206;
    wp::float32 var_1207;
    wp::int32 var_1208;
    const wp::float32 var_1209 = 1.0;
    bool var_1210;
    bool var_1211;
    const wp::float32 var_1212 = -1.0;
    bool var_1213;
    const wp::float32 var_1214 = 1.0;
    const wp::int32 var_1215 = 2;
    wp::float32 var_1216;
    wp::float32 var_1217;
    const wp::float32 var_1218 = -1.0;
    const wp::int32 var_1219 = 0;
    wp::float32 var_1220;
    wp::float32 var_1221;
    wp::float32 var_1222;
    wp::float32 var_1223;
    wp::int32 var_1224;
    const wp::int32 var_1225 = 1;
    bool var_1226;
    const wp::float32 var_1227 = 1.0;
    const wp::int32 var_1228 = 2;
    const wp::int32 var_1229 = -1;
    bool var_1230;
    const wp::float32 var_1231 = -1.0;
    const wp::int32 var_1232 = 0;
    wp::float32 var_1233;
    wp::int32 var_1234;
    wp::float32 var_1235;
    wp::int32 var_1236;
    wp::float32 var_1237;
    wp::float32 var_1238;
    wp::int32 var_1239;
    wp::int32 var_1240;
    wp::vec_t<3, wp::float32> var_1241;
    wp::vec_t<3, wp::float32> var_1242;
    wp::float32 var_1243;
    wp::float32 var_1244;
    const wp::int32 var_1245 = 3;
    wp::int32 var_1246;
    wp::int32 var_1247;
    wp::float32 var_1248;
    wp::float32 var_1249;
    bool var_1250;
    wp::float32 var_1251;
    wp::float32 var_1252;
    wp::float32 var_1253;
    const wp::int32 var_1254 = 6;
    wp::int32 var_1255;
    const wp::int32 var_1256 = 1;
    wp::int32 var_1257;
    wp::int32 var_1258;
    wp::int32 var_1259;
    wp::int32 var_1260;
    wp::int32 var_1261;
    wp::float32 var_1262;
    wp::float32 var_1263;
    wp::int32 var_1264;
    wp::int32 var_1265;
    wp::int32 var_1266;
    wp::float32 var_1267;
    wp::int32 var_1268;
    wp::float32 var_1269;
    wp::float32 var_1270;
    wp::int32 var_1271;
    wp::int32 var_1272;
    wp::int32 var_1273;
    wp::float32 var_1274;
    wp::int32 var_1275;
    wp::vec_t<3, wp::float32> var_1276;
    wp::float32 var_1277;
    wp::float32 var_1278;
    wp::float32 var_1279;
    wp::int32 var_1280;
    wp::int32 var_1281;
    bool var_1282;
    wp::int32 var_1283;
    wp::float32 var_1284;
    wp::float32 var_1285;
    wp::float32 var_1286;
    wp::int32 var_1287;
    wp::int32 var_1288;
    wp::int32 var_1289;
    wp::float32 var_1290;
    wp::int32 var_1291;
    wp::vec_t<3, wp::float32> var_1292;
    wp::vec_t<3, wp::float32> var_1293;
    wp::float32 var_1294;
    wp::float32 var_1295;
    wp::float32 var_1296;
    wp::float32 var_1297;
    wp::float32 var_1298;
    wp::float32 var_1299;
    wp::float32 var_1300;
    wp::float32 var_1301;
    wp::float32 var_1302;
    wp::int32 var_1303;
    wp::int32 var_1304;
    bool var_1305;
    wp::int32 var_1306;
    wp::float32 var_1307;
    bool var_1308;
    const wp::int32 var_1309 = 2;
    const wp::int32 var_1310 = 0;
    const bool var_1311 = true;
    const wp::int32 var_1312 = 1;
    wp::int32 var_1313;
    wp::int32 var_1314;
    const wp::int32 var_1315 = 0;
    bool var_1316;
    const bool var_1317 = false;
    bool var_1318;
    const wp::int32 var_1319 = -123;
    wp::int32 var_1320;
    const wp::int32 var_1321 = 1;
    wp::int32 var_1322;
    const wp::float32 var_1323 = 1.0;
    const wp::float32 var_1324 = -1.0;
    wp::float32 var_1325;
    const wp::int32 var_1326 = 2;
    wp::int32 var_1327;
    const wp::float32 var_1328 = 1.0;
    const wp::float32 var_1329 = -1.0;
    wp::float32 var_1330;
    const wp::int32 var_1331 = 4;
    wp::int32 var_1332;
    const wp::float32 var_1333 = 1.0;
    const wp::float32 var_1334 = -1.0;
    wp::float32 var_1335;
    wp::vec_t<3, wp::float32> var_1336;
    wp::vec_t<3, wp::float32> var_1337;
    const wp::float32 var_1338 = 0.0;
    wp::vec_t<3, wp::float32> var_1339;
    wp::float32 var_1340;
    wp::float32 var_1341;
    wp::float32 var_1342;
    wp::float32 var_1343;
    wp::float32 var_1344;
    wp::float32 var_1345;
    wp::float32 var_1346;
    wp::float32 var_1347;
    wp::float32 var_1348;
    wp::float32 var_1349;
    wp::float32 var_1350;
    wp::float32 var_1351;
    wp::float32 var_1352;
    wp::float32 var_1353;
    wp::float32 var_1354;
    wp::float32 var_1355;
    wp::float32 var_1356;
    bool var_1357;
    const bool var_1358 = false;
    bool var_1359;
    const wp::float32 var_1360 = 1.0;
    wp::float32 var_1361;
    wp::float32 var_1362;
    wp::float32 var_1363;
    wp::float32 var_1364;
    wp::float32 var_1365;
    wp::float32 var_1366;
    wp::float32 var_1367;
    wp::float32 var_1368;
    wp::float32 var_1369;
    wp::float32 var_1370;
    wp::float32 var_1371;
    const wp::int32 var_1372 = 1;
    wp::int32 var_1373;
    const wp::int32 var_1374 = 1;
    wp::int32 var_1375;
    const wp::int32 var_1376 = 1;
    bool var_1377;
    const wp::float32 var_1378 = 1.0;
    const wp::int32 var_1379 = 2;
    wp::float32 var_1380;
    wp::float32 var_1381;
    const wp::int32 var_1382 = -1;
    bool var_1383;
    const wp::float32 var_1384 = -1.0;
    const wp::int32 var_1385 = 0;
    wp::float32 var_1386;
    wp::float32 var_1387;
    wp::float32 var_1388;
    wp::float32 var_1389;
    wp::int32 var_1390;
    wp::float32 var_1391;
    wp::float32 var_1392;
    wp::int32 var_1393;
    const wp::float32 var_1394 = 1.0;
    bool var_1395;
    bool var_1396;
    const wp::float32 var_1397 = -1.0;
    bool var_1398;
    const wp::float32 var_1399 = 1.0;
    const wp::int32 var_1400 = 2;
    wp::float32 var_1401;
    wp::float32 var_1402;
    const wp::float32 var_1403 = -1.0;
    const wp::int32 var_1404 = 0;
    wp::float32 var_1405;
    wp::float32 var_1406;
    wp::float32 var_1407;
    wp::float32 var_1408;
    wp::int32 var_1409;
    const wp::int32 var_1410 = 1;
    bool var_1411;
    const wp::float32 var_1412 = 1.0;
    const wp::int32 var_1413 = 2;
    const wp::int32 var_1414 = -1;
    bool var_1415;
    const wp::float32 var_1416 = -1.0;
    const wp::int32 var_1417 = 0;
    wp::float32 var_1418;
    wp::int32 var_1419;
    wp::float32 var_1420;
    wp::int32 var_1421;
    wp::float32 var_1422;
    wp::float32 var_1423;
    wp::int32 var_1424;
    wp::int32 var_1425;
    wp::vec_t<3, wp::float32> var_1426;
    wp::vec_t<3, wp::float32> var_1427;
    wp::float32 var_1428;
    wp::float32 var_1429;
    const wp::int32 var_1430 = 3;
    wp::int32 var_1431;
    wp::int32 var_1432;
    wp::float32 var_1433;
    wp::float32 var_1434;
    bool var_1435;
    wp::float32 var_1436;
    wp::float32 var_1437;
    wp::float32 var_1438;
    const wp::int32 var_1439 = 6;
    wp::int32 var_1440;
    const wp::int32 var_1441 = 1;
    wp::int32 var_1442;
    wp::int32 var_1443;
    wp::int32 var_1444;
    wp::int32 var_1445;
    wp::int32 var_1446;
    wp::float32 var_1447;
    wp::float32 var_1448;
    wp::int32 var_1449;
    wp::int32 var_1450;
    wp::int32 var_1451;
    wp::float32 var_1452;
    wp::int32 var_1453;
    wp::float32 var_1454;
    wp::float32 var_1455;
    wp::int32 var_1456;
    wp::int32 var_1457;
    wp::int32 var_1458;
    wp::float32 var_1459;
    wp::int32 var_1460;
    wp::vec_t<3, wp::float32> var_1461;
    wp::float32 var_1462;
    wp::float32 var_1463;
    wp::float32 var_1464;
    wp::int32 var_1465;
    wp::int32 var_1466;
    bool var_1467;
    wp::int32 var_1468;
    wp::float32 var_1469;
    wp::float32 var_1470;
    wp::float32 var_1471;
    wp::int32 var_1472;
    wp::int32 var_1473;
    wp::int32 var_1474;
    wp::float32 var_1475;
    wp::int32 var_1476;
    wp::vec_t<3, wp::float32> var_1477;
    wp::vec_t<3, wp::float32> var_1478;
    wp::float32 var_1479;
    wp::float32 var_1480;
    wp::float32 var_1481;
    wp::float32 var_1482;
    wp::float32 var_1483;
    wp::float32 var_1484;
    wp::float32 var_1485;
    wp::float32 var_1486;
    wp::float32 var_1487;
    wp::int32 var_1488;
    wp::int32 var_1489;
    bool var_1490;
    wp::int32 var_1491;
    wp::float32 var_1492;
    bool var_1493;
    const wp::int32 var_1494 = 1;
    const bool var_1495 = true;
    const wp::int32 var_1496 = 1;
    wp::int32 var_1497;
    wp::int32 var_1498;
    const wp::int32 var_1499 = 0;
    bool var_1500;
    const bool var_1501 = false;
    bool var_1502;
    const wp::int32 var_1503 = -123;
    wp::int32 var_1504;
    const wp::int32 var_1505 = 1;
    wp::int32 var_1506;
    const wp::float32 var_1507 = 1.0;
    const wp::float32 var_1508 = -1.0;
    wp::float32 var_1509;
    const wp::int32 var_1510 = 2;
    wp::int32 var_1511;
    const wp::float32 var_1512 = 1.0;
    const wp::float32 var_1513 = -1.0;
    wp::float32 var_1514;
    const wp::int32 var_1515 = 4;
    wp::int32 var_1516;
    const wp::float32 var_1517 = 1.0;
    const wp::float32 var_1518 = -1.0;
    wp::float32 var_1519;
    wp::vec_t<3, wp::float32> var_1520;
    wp::vec_t<3, wp::float32> var_1521;
    const wp::float32 var_1522 = 0.0;
    wp::vec_t<3, wp::float32> var_1523;
    wp::float32 var_1524;
    wp::float32 var_1525;
    wp::float32 var_1526;
    wp::float32 var_1527;
    wp::float32 var_1528;
    wp::float32 var_1529;
    wp::float32 var_1530;
    wp::float32 var_1531;
    wp::float32 var_1532;
    wp::float32 var_1533;
    wp::float32 var_1534;
    wp::float32 var_1535;
    wp::float32 var_1536;
    wp::float32 var_1537;
    wp::float32 var_1538;
    wp::float32 var_1539;
    wp::float32 var_1540;
    bool var_1541;
    const bool var_1542 = false;
    bool var_1543;
    const wp::float32 var_1544 = 1.0;
    wp::float32 var_1545;
    wp::float32 var_1546;
    wp::float32 var_1547;
    wp::float32 var_1548;
    wp::float32 var_1549;
    wp::float32 var_1550;
    wp::float32 var_1551;
    wp::float32 var_1552;
    wp::float32 var_1553;
    wp::float32 var_1554;
    wp::float32 var_1555;
    const wp::int32 var_1556 = 1;
    wp::int32 var_1557;
    const wp::int32 var_1558 = 1;
    wp::int32 var_1559;
    const wp::int32 var_1560 = 1;
    bool var_1561;
    const wp::float32 var_1562 = 1.0;
    const wp::int32 var_1563 = 2;
    wp::float32 var_1564;
    wp::float32 var_1565;
    const wp::int32 var_1566 = -1;
    bool var_1567;
    const wp::float32 var_1568 = -1.0;
    const wp::int32 var_1569 = 0;
    wp::float32 var_1570;
    wp::float32 var_1571;
    wp::float32 var_1572;
    wp::float32 var_1573;
    wp::int32 var_1574;
    wp::float32 var_1575;
    wp::float32 var_1576;
    wp::int32 var_1577;
    const wp::float32 var_1578 = 1.0;
    bool var_1579;
    bool var_1580;
    const wp::float32 var_1581 = -1.0;
    bool var_1582;
    const wp::float32 var_1583 = 1.0;
    const wp::int32 var_1584 = 2;
    wp::float32 var_1585;
    wp::float32 var_1586;
    const wp::float32 var_1587 = -1.0;
    const wp::int32 var_1588 = 0;
    wp::float32 var_1589;
    wp::float32 var_1590;
    wp::float32 var_1591;
    wp::float32 var_1592;
    wp::int32 var_1593;
    const wp::int32 var_1594 = 1;
    bool var_1595;
    const wp::float32 var_1596 = 1.0;
    const wp::int32 var_1597 = 2;
    const wp::int32 var_1598 = -1;
    bool var_1599;
    const wp::float32 var_1600 = -1.0;
    const wp::int32 var_1601 = 0;
    wp::float32 var_1602;
    wp::int32 var_1603;
    wp::float32 var_1604;
    wp::int32 var_1605;
    wp::float32 var_1606;
    wp::float32 var_1607;
    wp::int32 var_1608;
    wp::int32 var_1609;
    wp::vec_t<3, wp::float32> var_1610;
    wp::vec_t<3, wp::float32> var_1611;
    wp::float32 var_1612;
    wp::float32 var_1613;
    const wp::int32 var_1614 = 3;
    wp::int32 var_1615;
    wp::int32 var_1616;
    wp::float32 var_1617;
    wp::float32 var_1618;
    bool var_1619;
    wp::float32 var_1620;
    wp::float32 var_1621;
    wp::float32 var_1622;
    const wp::int32 var_1623 = 6;
    wp::int32 var_1624;
    const wp::int32 var_1625 = 1;
    wp::int32 var_1626;
    wp::int32 var_1627;
    wp::int32 var_1628;
    wp::int32 var_1629;
    wp::int32 var_1630;
    wp::float32 var_1631;
    wp::float32 var_1632;
    wp::int32 var_1633;
    wp::int32 var_1634;
    wp::int32 var_1635;
    wp::float32 var_1636;
    wp::int32 var_1637;
    wp::float32 var_1638;
    wp::float32 var_1639;
    wp::int32 var_1640;
    wp::int32 var_1641;
    wp::int32 var_1642;
    wp::float32 var_1643;
    wp::int32 var_1644;
    wp::vec_t<3, wp::float32> var_1645;
    wp::float32 var_1646;
    wp::float32 var_1647;
    wp::float32 var_1648;
    wp::int32 var_1649;
    wp::int32 var_1650;
    bool var_1651;
    wp::int32 var_1652;
    wp::float32 var_1653;
    wp::float32 var_1654;
    wp::float32 var_1655;
    wp::int32 var_1656;
    wp::int32 var_1657;
    wp::int32 var_1658;
    wp::float32 var_1659;
    wp::int32 var_1660;
    wp::vec_t<3, wp::float32> var_1661;
    wp::vec_t<3, wp::float32> var_1662;
    wp::float32 var_1663;
    wp::float32 var_1664;
    wp::float32 var_1665;
    wp::float32 var_1666;
    wp::float32 var_1667;
    wp::float32 var_1668;
    wp::float32 var_1669;
    wp::float32 var_1670;
    wp::float32 var_1671;
    wp::int32 var_1672;
    wp::int32 var_1673;
    bool var_1674;
    wp::int32 var_1675;
    wp::float32 var_1676;
    bool var_1677;
    const wp::int32 var_1678 = 2;
    const bool var_1679 = true;
    const wp::int32 var_1680 = 1;
    wp::int32 var_1681;
    wp::int32 var_1682;
    const wp::int32 var_1683 = 0;
    bool var_1684;
    const bool var_1685 = false;
    bool var_1686;
    const wp::int32 var_1687 = -123;
    wp::int32 var_1688;
    const wp::int32 var_1689 = 1;
    wp::int32 var_1690;
    const wp::float32 var_1691 = 1.0;
    const wp::float32 var_1692 = -1.0;
    wp::float32 var_1693;
    const wp::int32 var_1694 = 2;
    wp::int32 var_1695;
    const wp::float32 var_1696 = 1.0;
    const wp::float32 var_1697 = -1.0;
    wp::float32 var_1698;
    const wp::int32 var_1699 = 4;
    wp::int32 var_1700;
    const wp::float32 var_1701 = 1.0;
    const wp::float32 var_1702 = -1.0;
    wp::float32 var_1703;
    wp::vec_t<3, wp::float32> var_1704;
    wp::vec_t<3, wp::float32> var_1705;
    const wp::float32 var_1706 = 0.0;
    wp::vec_t<3, wp::float32> var_1707;
    wp::float32 var_1708;
    wp::float32 var_1709;
    wp::float32 var_1710;
    wp::float32 var_1711;
    wp::float32 var_1712;
    wp::float32 var_1713;
    wp::float32 var_1714;
    wp::float32 var_1715;
    wp::float32 var_1716;
    wp::float32 var_1717;
    wp::float32 var_1718;
    wp::float32 var_1719;
    wp::float32 var_1720;
    wp::float32 var_1721;
    wp::float32 var_1722;
    wp::float32 var_1723;
    wp::float32 var_1724;
    bool var_1725;
    const bool var_1726 = false;
    bool var_1727;
    const wp::float32 var_1728 = 1.0;
    wp::float32 var_1729;
    wp::float32 var_1730;
    wp::float32 var_1731;
    wp::float32 var_1732;
    wp::float32 var_1733;
    wp::float32 var_1734;
    wp::float32 var_1735;
    wp::float32 var_1736;
    wp::float32 var_1737;
    wp::float32 var_1738;
    wp::float32 var_1739;
    const wp::int32 var_1740 = 1;
    wp::int32 var_1741;
    const wp::int32 var_1742 = 1;
    wp::int32 var_1743;
    const wp::int32 var_1744 = 1;
    bool var_1745;
    const wp::float32 var_1746 = 1.0;
    const wp::int32 var_1747 = 2;
    wp::float32 var_1748;
    wp::float32 var_1749;
    const wp::int32 var_1750 = -1;
    bool var_1751;
    const wp::float32 var_1752 = -1.0;
    const wp::int32 var_1753 = 0;
    wp::float32 var_1754;
    wp::float32 var_1755;
    wp::float32 var_1756;
    wp::float32 var_1757;
    wp::int32 var_1758;
    wp::float32 var_1759;
    wp::float32 var_1760;
    wp::int32 var_1761;
    const wp::float32 var_1762 = 1.0;
    bool var_1763;
    bool var_1764;
    const wp::float32 var_1765 = -1.0;
    bool var_1766;
    const wp::float32 var_1767 = 1.0;
    const wp::int32 var_1768 = 2;
    wp::float32 var_1769;
    wp::float32 var_1770;
    const wp::float32 var_1771 = -1.0;
    const wp::int32 var_1772 = 0;
    wp::float32 var_1773;
    wp::float32 var_1774;
    wp::float32 var_1775;
    wp::float32 var_1776;
    wp::int32 var_1777;
    const wp::int32 var_1778 = 1;
    bool var_1779;
    const wp::float32 var_1780 = 1.0;
    const wp::int32 var_1781 = 2;
    const wp::int32 var_1782 = -1;
    bool var_1783;
    const wp::float32 var_1784 = -1.0;
    const wp::int32 var_1785 = 0;
    wp::float32 var_1786;
    wp::int32 var_1787;
    wp::float32 var_1788;
    wp::int32 var_1789;
    wp::float32 var_1790;
    wp::float32 var_1791;
    wp::int32 var_1792;
    wp::int32 var_1793;
    wp::vec_t<3, wp::float32> var_1794;
    wp::vec_t<3, wp::float32> var_1795;
    wp::float32 var_1796;
    wp::float32 var_1797;
    const wp::int32 var_1798 = 3;
    wp::int32 var_1799;
    wp::int32 var_1800;
    wp::float32 var_1801;
    wp::float32 var_1802;
    bool var_1803;
    wp::float32 var_1804;
    wp::float32 var_1805;
    wp::float32 var_1806;
    const wp::int32 var_1807 = 6;
    wp::int32 var_1808;
    const wp::int32 var_1809 = 1;
    wp::int32 var_1810;
    wp::int32 var_1811;
    wp::int32 var_1812;
    wp::int32 var_1813;
    wp::int32 var_1814;
    wp::float32 var_1815;
    wp::float32 var_1816;
    wp::int32 var_1817;
    wp::int32 var_1818;
    wp::int32 var_1819;
    wp::float32 var_1820;
    wp::int32 var_1821;
    wp::float32 var_1822;
    wp::float32 var_1823;
    wp::int32 var_1824;
    wp::int32 var_1825;
    wp::int32 var_1826;
    wp::float32 var_1827;
    wp::int32 var_1828;
    wp::vec_t<3, wp::float32> var_1829;
    wp::float32 var_1830;
    wp::float32 var_1831;
    wp::float32 var_1832;
    wp::int32 var_1833;
    wp::int32 var_1834;
    bool var_1835;
    wp::int32 var_1836;
    wp::float32 var_1837;
    wp::float32 var_1838;
    wp::float32 var_1839;
    wp::int32 var_1840;
    wp::int32 var_1841;
    wp::int32 var_1842;
    wp::float32 var_1843;
    wp::int32 var_1844;
    wp::vec_t<3, wp::float32> var_1845;
    wp::vec_t<3, wp::float32> var_1846;
    wp::float32 var_1847;
    wp::float32 var_1848;
    wp::float32 var_1849;
    wp::float32 var_1850;
    wp::float32 var_1851;
    wp::float32 var_1852;
    wp::float32 var_1853;
    wp::float32 var_1854;
    wp::float32 var_1855;
    wp::int32 var_1856;
    wp::int32 var_1857;
    bool var_1858;
    wp::int32 var_1859;
    wp::float32 var_1860;
    bool var_1861;
    const wp::int32 var_1862 = 3;
    const wp::int32 var_1863 = 0;
    const bool var_1864 = true;
    const wp::int32 var_1865 = 1;
    wp::int32 var_1866;
    wp::int32 var_1867;
    const wp::int32 var_1868 = 0;
    bool var_1869;
    const bool var_1870 = false;
    bool var_1871;
    const wp::int32 var_1872 = -123;
    wp::int32 var_1873;
    const wp::int32 var_1874 = 1;
    wp::int32 var_1875;
    const wp::float32 var_1876 = 1.0;
    const wp::float32 var_1877 = -1.0;
    wp::float32 var_1878;
    const wp::int32 var_1879 = 2;
    wp::int32 var_1880;
    const wp::float32 var_1881 = 1.0;
    const wp::float32 var_1882 = -1.0;
    wp::float32 var_1883;
    const wp::int32 var_1884 = 4;
    wp::int32 var_1885;
    const wp::float32 var_1886 = 1.0;
    const wp::float32 var_1887 = -1.0;
    wp::float32 var_1888;
    wp::vec_t<3, wp::float32> var_1889;
    wp::vec_t<3, wp::float32> var_1890;
    const wp::float32 var_1891 = 0.0;
    wp::vec_t<3, wp::float32> var_1892;
    wp::float32 var_1893;
    wp::float32 var_1894;
    wp::float32 var_1895;
    wp::float32 var_1896;
    wp::float32 var_1897;
    wp::float32 var_1898;
    wp::float32 var_1899;
    wp::float32 var_1900;
    wp::float32 var_1901;
    wp::float32 var_1902;
    wp::float32 var_1903;
    wp::float32 var_1904;
    wp::float32 var_1905;
    wp::float32 var_1906;
    wp::float32 var_1907;
    wp::float32 var_1908;
    wp::float32 var_1909;
    bool var_1910;
    const bool var_1911 = false;
    bool var_1912;
    const wp::float32 var_1913 = 1.0;
    wp::float32 var_1914;
    wp::float32 var_1915;
    wp::float32 var_1916;
    wp::float32 var_1917;
    wp::float32 var_1918;
    wp::float32 var_1919;
    wp::float32 var_1920;
    wp::float32 var_1921;
    wp::float32 var_1922;
    wp::float32 var_1923;
    wp::float32 var_1924;
    const wp::int32 var_1925 = 1;
    wp::int32 var_1926;
    const wp::int32 var_1927 = 1;
    wp::int32 var_1928;
    const wp::int32 var_1929 = 1;
    bool var_1930;
    const wp::float32 var_1931 = 1.0;
    const wp::int32 var_1932 = 2;
    wp::float32 var_1933;
    wp::float32 var_1934;
    const wp::int32 var_1935 = -1;
    bool var_1936;
    const wp::float32 var_1937 = -1.0;
    const wp::int32 var_1938 = 0;
    wp::float32 var_1939;
    wp::float32 var_1940;
    wp::float32 var_1941;
    wp::float32 var_1942;
    wp::int32 var_1943;
    wp::float32 var_1944;
    wp::float32 var_1945;
    wp::int32 var_1946;
    const wp::float32 var_1947 = 1.0;
    bool var_1948;
    bool var_1949;
    const wp::float32 var_1950 = -1.0;
    bool var_1951;
    const wp::float32 var_1952 = 1.0;
    const wp::int32 var_1953 = 2;
    wp::float32 var_1954;
    wp::float32 var_1955;
    const wp::float32 var_1956 = -1.0;
    const wp::int32 var_1957 = 0;
    wp::float32 var_1958;
    wp::float32 var_1959;
    wp::float32 var_1960;
    wp::float32 var_1961;
    wp::int32 var_1962;
    const wp::int32 var_1963 = 1;
    bool var_1964;
    const wp::float32 var_1965 = 1.0;
    const wp::int32 var_1966 = 2;
    const wp::int32 var_1967 = -1;
    bool var_1968;
    const wp::float32 var_1969 = -1.0;
    const wp::int32 var_1970 = 0;
    wp::float32 var_1971;
    wp::int32 var_1972;
    wp::float32 var_1973;
    wp::int32 var_1974;
    wp::float32 var_1975;
    wp::float32 var_1976;
    wp::int32 var_1977;
    wp::int32 var_1978;
    wp::vec_t<3, wp::float32> var_1979;
    wp::vec_t<3, wp::float32> var_1980;
    wp::float32 var_1981;
    wp::float32 var_1982;
    const wp::int32 var_1983 = 3;
    wp::int32 var_1984;
    wp::int32 var_1985;
    wp::float32 var_1986;
    wp::float32 var_1987;
    bool var_1988;
    wp::float32 var_1989;
    wp::float32 var_1990;
    wp::float32 var_1991;
    const wp::int32 var_1992 = 6;
    wp::int32 var_1993;
    const wp::int32 var_1994 = 1;
    wp::int32 var_1995;
    wp::int32 var_1996;
    wp::int32 var_1997;
    wp::int32 var_1998;
    wp::int32 var_1999;
    wp::float32 var_2000;
    wp::float32 var_2001;
    wp::int32 var_2002;
    wp::int32 var_2003;
    wp::int32 var_2004;
    wp::float32 var_2005;
    wp::int32 var_2006;
    wp::float32 var_2007;
    wp::float32 var_2008;
    wp::int32 var_2009;
    wp::int32 var_2010;
    wp::int32 var_2011;
    wp::float32 var_2012;
    wp::int32 var_2013;
    wp::vec_t<3, wp::float32> var_2014;
    wp::float32 var_2015;
    wp::float32 var_2016;
    wp::float32 var_2017;
    wp::int32 var_2018;
    wp::int32 var_2019;
    bool var_2020;
    wp::int32 var_2021;
    wp::float32 var_2022;
    wp::float32 var_2023;
    wp::float32 var_2024;
    wp::int32 var_2025;
    wp::int32 var_2026;
    wp::int32 var_2027;
    wp::float32 var_2028;
    wp::int32 var_2029;
    wp::vec_t<3, wp::float32> var_2030;
    wp::vec_t<3, wp::float32> var_2031;
    wp::float32 var_2032;
    wp::float32 var_2033;
    wp::float32 var_2034;
    wp::float32 var_2035;
    wp::float32 var_2036;
    wp::float32 var_2037;
    wp::float32 var_2038;
    wp::float32 var_2039;
    wp::float32 var_2040;
    wp::int32 var_2041;
    wp::int32 var_2042;
    bool var_2043;
    wp::int32 var_2044;
    wp::float32 var_2045;
    bool var_2046;
    const wp::int32 var_2047 = 1;
    const bool var_2048 = true;
    const wp::int32 var_2049 = 1;
    wp::int32 var_2050;
    wp::int32 var_2051;
    const wp::int32 var_2052 = 0;
    bool var_2053;
    const bool var_2054 = false;
    bool var_2055;
    const wp::int32 var_2056 = -123;
    wp::int32 var_2057;
    const wp::int32 var_2058 = 1;
    wp::int32 var_2059;
    const wp::float32 var_2060 = 1.0;
    const wp::float32 var_2061 = -1.0;
    wp::float32 var_2062;
    const wp::int32 var_2063 = 2;
    wp::int32 var_2064;
    const wp::float32 var_2065 = 1.0;
    const wp::float32 var_2066 = -1.0;
    wp::float32 var_2067;
    const wp::int32 var_2068 = 4;
    wp::int32 var_2069;
    const wp::float32 var_2070 = 1.0;
    const wp::float32 var_2071 = -1.0;
    wp::float32 var_2072;
    wp::vec_t<3, wp::float32> var_2073;
    wp::vec_t<3, wp::float32> var_2074;
    const wp::float32 var_2075 = 0.0;
    wp::vec_t<3, wp::float32> var_2076;
    wp::float32 var_2077;
    wp::float32 var_2078;
    wp::float32 var_2079;
    wp::float32 var_2080;
    wp::float32 var_2081;
    wp::float32 var_2082;
    wp::float32 var_2083;
    wp::float32 var_2084;
    wp::float32 var_2085;
    wp::float32 var_2086;
    wp::float32 var_2087;
    wp::float32 var_2088;
    wp::float32 var_2089;
    wp::float32 var_2090;
    wp::float32 var_2091;
    wp::float32 var_2092;
    wp::float32 var_2093;
    bool var_2094;
    const bool var_2095 = false;
    bool var_2096;
    const wp::float32 var_2097 = 1.0;
    wp::float32 var_2098;
    wp::float32 var_2099;
    wp::float32 var_2100;
    wp::float32 var_2101;
    wp::float32 var_2102;
    wp::float32 var_2103;
    wp::float32 var_2104;
    wp::float32 var_2105;
    wp::float32 var_2106;
    wp::float32 var_2107;
    wp::float32 var_2108;
    const wp::int32 var_2109 = 1;
    wp::int32 var_2110;
    const wp::int32 var_2111 = 1;
    wp::int32 var_2112;
    const wp::int32 var_2113 = 1;
    bool var_2114;
    const wp::float32 var_2115 = 1.0;
    const wp::int32 var_2116 = 2;
    wp::float32 var_2117;
    wp::float32 var_2118;
    const wp::int32 var_2119 = -1;
    bool var_2120;
    const wp::float32 var_2121 = -1.0;
    const wp::int32 var_2122 = 0;
    wp::float32 var_2123;
    wp::float32 var_2124;
    wp::float32 var_2125;
    wp::float32 var_2126;
    wp::int32 var_2127;
    wp::float32 var_2128;
    wp::float32 var_2129;
    wp::int32 var_2130;
    const wp::float32 var_2131 = 1.0;
    bool var_2132;
    bool var_2133;
    const wp::float32 var_2134 = -1.0;
    bool var_2135;
    const wp::float32 var_2136 = 1.0;
    const wp::int32 var_2137 = 2;
    wp::float32 var_2138;
    wp::float32 var_2139;
    const wp::float32 var_2140 = -1.0;
    const wp::int32 var_2141 = 0;
    wp::float32 var_2142;
    wp::float32 var_2143;
    wp::float32 var_2144;
    wp::float32 var_2145;
    wp::int32 var_2146;
    const wp::int32 var_2147 = 1;
    bool var_2148;
    const wp::float32 var_2149 = 1.0;
    const wp::int32 var_2150 = 2;
    const wp::int32 var_2151 = -1;
    bool var_2152;
    const wp::float32 var_2153 = -1.0;
    const wp::int32 var_2154 = 0;
    wp::float32 var_2155;
    wp::int32 var_2156;
    wp::float32 var_2157;
    wp::int32 var_2158;
    wp::float32 var_2159;
    wp::float32 var_2160;
    wp::int32 var_2161;
    wp::int32 var_2162;
    wp::vec_t<3, wp::float32> var_2163;
    wp::vec_t<3, wp::float32> var_2164;
    wp::float32 var_2165;
    wp::float32 var_2166;
    const wp::int32 var_2167 = 3;
    wp::int32 var_2168;
    wp::int32 var_2169;
    wp::float32 var_2170;
    wp::float32 var_2171;
    bool var_2172;
    wp::float32 var_2173;
    wp::float32 var_2174;
    wp::float32 var_2175;
    const wp::int32 var_2176 = 6;
    wp::int32 var_2177;
    const wp::int32 var_2178 = 1;
    wp::int32 var_2179;
    wp::int32 var_2180;
    wp::int32 var_2181;
    wp::int32 var_2182;
    wp::int32 var_2183;
    wp::float32 var_2184;
    wp::float32 var_2185;
    wp::int32 var_2186;
    wp::int32 var_2187;
    wp::int32 var_2188;
    wp::float32 var_2189;
    wp::int32 var_2190;
    wp::float32 var_2191;
    wp::float32 var_2192;
    wp::int32 var_2193;
    wp::int32 var_2194;
    wp::int32 var_2195;
    wp::float32 var_2196;
    wp::int32 var_2197;
    wp::vec_t<3, wp::float32> var_2198;
    wp::float32 var_2199;
    wp::float32 var_2200;
    wp::float32 var_2201;
    wp::int32 var_2202;
    wp::int32 var_2203;
    bool var_2204;
    wp::int32 var_2205;
    wp::float32 var_2206;
    wp::float32 var_2207;
    wp::float32 var_2208;
    wp::int32 var_2209;
    wp::int32 var_2210;
    wp::int32 var_2211;
    wp::float32 var_2212;
    wp::int32 var_2213;
    wp::vec_t<3, wp::float32> var_2214;
    wp::vec_t<3, wp::float32> var_2215;
    wp::float32 var_2216;
    wp::float32 var_2217;
    wp::float32 var_2218;
    wp::float32 var_2219;
    wp::float32 var_2220;
    wp::float32 var_2221;
    wp::float32 var_2222;
    wp::float32 var_2223;
    wp::float32 var_2224;
    wp::int32 var_2225;
    wp::int32 var_2226;
    bool var_2227;
    wp::int32 var_2228;
    wp::float32 var_2229;
    bool var_2230;
    const wp::int32 var_2231 = 2;
    const bool var_2232 = true;
    const wp::int32 var_2233 = 1;
    wp::int32 var_2234;
    wp::int32 var_2235;
    const wp::int32 var_2236 = 0;
    bool var_2237;
    const bool var_2238 = false;
    bool var_2239;
    const wp::int32 var_2240 = -123;
    wp::int32 var_2241;
    const wp::int32 var_2242 = 1;
    wp::int32 var_2243;
    const wp::float32 var_2244 = 1.0;
    const wp::float32 var_2245 = -1.0;
    wp::float32 var_2246;
    const wp::int32 var_2247 = 2;
    wp::int32 var_2248;
    const wp::float32 var_2249 = 1.0;
    const wp::float32 var_2250 = -1.0;
    wp::float32 var_2251;
    const wp::int32 var_2252 = 4;
    wp::int32 var_2253;
    const wp::float32 var_2254 = 1.0;
    const wp::float32 var_2255 = -1.0;
    wp::float32 var_2256;
    wp::vec_t<3, wp::float32> var_2257;
    wp::vec_t<3, wp::float32> var_2258;
    const wp::float32 var_2259 = 0.0;
    wp::vec_t<3, wp::float32> var_2260;
    wp::float32 var_2261;
    wp::float32 var_2262;
    wp::float32 var_2263;
    wp::float32 var_2264;
    wp::float32 var_2265;
    wp::float32 var_2266;
    wp::float32 var_2267;
    wp::float32 var_2268;
    wp::float32 var_2269;
    wp::float32 var_2270;
    wp::float32 var_2271;
    wp::float32 var_2272;
    wp::float32 var_2273;
    wp::float32 var_2274;
    wp::float32 var_2275;
    wp::float32 var_2276;
    wp::float32 var_2277;
    bool var_2278;
    const bool var_2279 = false;
    bool var_2280;
    const wp::float32 var_2281 = 1.0;
    wp::float32 var_2282;
    wp::float32 var_2283;
    wp::float32 var_2284;
    wp::float32 var_2285;
    wp::float32 var_2286;
    wp::float32 var_2287;
    wp::float32 var_2288;
    wp::float32 var_2289;
    wp::float32 var_2290;
    wp::float32 var_2291;
    wp::float32 var_2292;
    const wp::int32 var_2293 = 1;
    wp::int32 var_2294;
    const wp::int32 var_2295 = 1;
    wp::int32 var_2296;
    const wp::int32 var_2297 = 1;
    bool var_2298;
    const wp::float32 var_2299 = 1.0;
    const wp::int32 var_2300 = 2;
    wp::float32 var_2301;
    wp::float32 var_2302;
    const wp::int32 var_2303 = -1;
    bool var_2304;
    const wp::float32 var_2305 = -1.0;
    const wp::int32 var_2306 = 0;
    wp::float32 var_2307;
    wp::float32 var_2308;
    wp::float32 var_2309;
    wp::float32 var_2310;
    wp::int32 var_2311;
    wp::float32 var_2312;
    wp::float32 var_2313;
    wp::int32 var_2314;
    const wp::float32 var_2315 = 1.0;
    bool var_2316;
    bool var_2317;
    const wp::float32 var_2318 = -1.0;
    bool var_2319;
    const wp::float32 var_2320 = 1.0;
    const wp::int32 var_2321 = 2;
    wp::float32 var_2322;
    wp::float32 var_2323;
    const wp::float32 var_2324 = -1.0;
    const wp::int32 var_2325 = 0;
    wp::float32 var_2326;
    wp::float32 var_2327;
    wp::float32 var_2328;
    wp::float32 var_2329;
    wp::int32 var_2330;
    const wp::int32 var_2331 = 1;
    bool var_2332;
    const wp::float32 var_2333 = 1.0;
    const wp::int32 var_2334 = 2;
    const wp::int32 var_2335 = -1;
    bool var_2336;
    const wp::float32 var_2337 = -1.0;
    const wp::int32 var_2338 = 0;
    wp::float32 var_2339;
    wp::int32 var_2340;
    wp::float32 var_2341;
    wp::int32 var_2342;
    wp::float32 var_2343;
    wp::float32 var_2344;
    wp::int32 var_2345;
    wp::int32 var_2346;
    wp::vec_t<3, wp::float32> var_2347;
    wp::vec_t<3, wp::float32> var_2348;
    wp::float32 var_2349;
    wp::float32 var_2350;
    const wp::int32 var_2351 = 3;
    wp::int32 var_2352;
    wp::int32 var_2353;
    wp::float32 var_2354;
    wp::float32 var_2355;
    bool var_2356;
    wp::float32 var_2357;
    wp::float32 var_2358;
    wp::float32 var_2359;
    const wp::int32 var_2360 = 6;
    wp::int32 var_2361;
    const wp::int32 var_2362 = 1;
    wp::int32 var_2363;
    wp::int32 var_2364;
    wp::int32 var_2365;
    wp::int32 var_2366;
    wp::int32 var_2367;
    wp::float32 var_2368;
    wp::float32 var_2369;
    wp::int32 var_2370;
    wp::int32 var_2371;
    wp::int32 var_2372;
    wp::float32 var_2373;
    wp::int32 var_2374;
    wp::float32 var_2375;
    wp::float32 var_2376;
    wp::int32 var_2377;
    wp::int32 var_2378;
    wp::int32 var_2379;
    wp::float32 var_2380;
    wp::int32 var_2381;
    wp::vec_t<3, wp::float32> var_2382;
    wp::float32 var_2383;
    wp::float32 var_2384;
    wp::float32 var_2385;
    wp::int32 var_2386;
    wp::int32 var_2387;
    bool var_2388;
    wp::int32 var_2389;
    wp::float32 var_2390;
    wp::float32 var_2391;
    wp::float32 var_2392;
    wp::int32 var_2393;
    wp::int32 var_2394;
    wp::int32 var_2395;
    wp::float32 var_2396;
    wp::int32 var_2397;
    wp::vec_t<3, wp::float32> var_2398;
    wp::vec_t<3, wp::float32> var_2399;
    wp::float32 var_2400;
    wp::float32 var_2401;
    wp::float32 var_2402;
    wp::float32 var_2403;
    wp::float32 var_2404;
    wp::float32 var_2405;
    wp::float32 var_2406;
    wp::float32 var_2407;
    wp::float32 var_2408;
    wp::int32 var_2409;
    wp::int32 var_2410;
    bool var_2411;
    wp::int32 var_2412;
    wp::float32 var_2413;
    bool var_2414;
    const wp::int32 var_2415 = 4;
    const wp::int32 var_2416 = 0;
    const bool var_2417 = true;
    const wp::int32 var_2418 = 1;
    wp::int32 var_2419;
    wp::int32 var_2420;
    const wp::int32 var_2421 = 0;
    bool var_2422;
    const bool var_2423 = false;
    bool var_2424;
    const wp::int32 var_2425 = -123;
    wp::int32 var_2426;
    const wp::int32 var_2427 = 1;
    wp::int32 var_2428;
    const wp::float32 var_2429 = 1.0;
    const wp::float32 var_2430 = -1.0;
    wp::float32 var_2431;
    const wp::int32 var_2432 = 2;
    wp::int32 var_2433;
    const wp::float32 var_2434 = 1.0;
    const wp::float32 var_2435 = -1.0;
    wp::float32 var_2436;
    const wp::int32 var_2437 = 4;
    wp::int32 var_2438;
    const wp::float32 var_2439 = 1.0;
    const wp::float32 var_2440 = -1.0;
    wp::float32 var_2441;
    wp::vec_t<3, wp::float32> var_2442;
    wp::vec_t<3, wp::float32> var_2443;
    const wp::float32 var_2444 = 0.0;
    wp::vec_t<3, wp::float32> var_2445;
    wp::float32 var_2446;
    wp::float32 var_2447;
    wp::float32 var_2448;
    wp::float32 var_2449;
    wp::float32 var_2450;
    wp::float32 var_2451;
    wp::float32 var_2452;
    wp::float32 var_2453;
    wp::float32 var_2454;
    wp::float32 var_2455;
    wp::float32 var_2456;
    wp::float32 var_2457;
    wp::float32 var_2458;
    wp::float32 var_2459;
    wp::float32 var_2460;
    wp::float32 var_2461;
    wp::float32 var_2462;
    bool var_2463;
    const bool var_2464 = false;
    bool var_2465;
    const wp::float32 var_2466 = 1.0;
    wp::float32 var_2467;
    wp::float32 var_2468;
    wp::float32 var_2469;
    wp::float32 var_2470;
    wp::float32 var_2471;
    wp::float32 var_2472;
    wp::float32 var_2473;
    wp::float32 var_2474;
    wp::float32 var_2475;
    wp::float32 var_2476;
    wp::float32 var_2477;
    const wp::int32 var_2478 = 1;
    wp::int32 var_2479;
    const wp::int32 var_2480 = 1;
    wp::int32 var_2481;
    const wp::int32 var_2482 = 1;
    bool var_2483;
    const wp::float32 var_2484 = 1.0;
    const wp::int32 var_2485 = 2;
    wp::float32 var_2486;
    wp::float32 var_2487;
    const wp::int32 var_2488 = -1;
    bool var_2489;
    const wp::float32 var_2490 = -1.0;
    const wp::int32 var_2491 = 0;
    wp::float32 var_2492;
    wp::float32 var_2493;
    wp::float32 var_2494;
    wp::float32 var_2495;
    wp::int32 var_2496;
    wp::float32 var_2497;
    wp::float32 var_2498;
    wp::int32 var_2499;
    const wp::float32 var_2500 = 1.0;
    bool var_2501;
    bool var_2502;
    const wp::float32 var_2503 = -1.0;
    bool var_2504;
    const wp::float32 var_2505 = 1.0;
    const wp::int32 var_2506 = 2;
    wp::float32 var_2507;
    wp::float32 var_2508;
    const wp::float32 var_2509 = -1.0;
    const wp::int32 var_2510 = 0;
    wp::float32 var_2511;
    wp::float32 var_2512;
    wp::float32 var_2513;
    wp::float32 var_2514;
    wp::int32 var_2515;
    const wp::int32 var_2516 = 1;
    bool var_2517;
    const wp::float32 var_2518 = 1.0;
    const wp::int32 var_2519 = 2;
    const wp::int32 var_2520 = -1;
    bool var_2521;
    const wp::float32 var_2522 = -1.0;
    const wp::int32 var_2523 = 0;
    wp::float32 var_2524;
    wp::int32 var_2525;
    wp::float32 var_2526;
    wp::int32 var_2527;
    wp::float32 var_2528;
    wp::float32 var_2529;
    wp::int32 var_2530;
    wp::int32 var_2531;
    wp::vec_t<3, wp::float32> var_2532;
    wp::vec_t<3, wp::float32> var_2533;
    wp::float32 var_2534;
    wp::float32 var_2535;
    const wp::int32 var_2536 = 3;
    wp::int32 var_2537;
    wp::int32 var_2538;
    wp::float32 var_2539;
    wp::float32 var_2540;
    bool var_2541;
    wp::float32 var_2542;
    wp::float32 var_2543;
    wp::float32 var_2544;
    const wp::int32 var_2545 = 6;
    wp::int32 var_2546;
    const wp::int32 var_2547 = 1;
    wp::int32 var_2548;
    wp::int32 var_2549;
    wp::int32 var_2550;
    wp::int32 var_2551;
    wp::int32 var_2552;
    wp::float32 var_2553;
    wp::float32 var_2554;
    wp::int32 var_2555;
    wp::int32 var_2556;
    wp::int32 var_2557;
    wp::float32 var_2558;
    wp::int32 var_2559;
    wp::float32 var_2560;
    wp::float32 var_2561;
    wp::int32 var_2562;
    wp::int32 var_2563;
    wp::int32 var_2564;
    wp::float32 var_2565;
    wp::int32 var_2566;
    wp::vec_t<3, wp::float32> var_2567;
    wp::float32 var_2568;
    wp::float32 var_2569;
    wp::float32 var_2570;
    wp::int32 var_2571;
    wp::int32 var_2572;
    bool var_2573;
    wp::int32 var_2574;
    wp::float32 var_2575;
    wp::float32 var_2576;
    wp::float32 var_2577;
    wp::int32 var_2578;
    wp::int32 var_2579;
    wp::int32 var_2580;
    wp::float32 var_2581;
    wp::int32 var_2582;
    wp::vec_t<3, wp::float32> var_2583;
    wp::vec_t<3, wp::float32> var_2584;
    wp::float32 var_2585;
    wp::float32 var_2586;
    wp::float32 var_2587;
    wp::float32 var_2588;
    wp::float32 var_2589;
    wp::float32 var_2590;
    wp::float32 var_2591;
    wp::float32 var_2592;
    wp::float32 var_2593;
    wp::int32 var_2594;
    wp::int32 var_2595;
    bool var_2596;
    wp::int32 var_2597;
    wp::float32 var_2598;
    bool var_2599;
    const wp::int32 var_2600 = 1;
    const bool var_2601 = true;
    const wp::int32 var_2602 = 1;
    wp::int32 var_2603;
    wp::int32 var_2604;
    const wp::int32 var_2605 = 0;
    bool var_2606;
    const bool var_2607 = false;
    bool var_2608;
    const wp::int32 var_2609 = -123;
    wp::int32 var_2610;
    const wp::int32 var_2611 = 1;
    wp::int32 var_2612;
    const wp::float32 var_2613 = 1.0;
    const wp::float32 var_2614 = -1.0;
    wp::float32 var_2615;
    const wp::int32 var_2616 = 2;
    wp::int32 var_2617;
    const wp::float32 var_2618 = 1.0;
    const wp::float32 var_2619 = -1.0;
    wp::float32 var_2620;
    const wp::int32 var_2621 = 4;
    wp::int32 var_2622;
    const wp::float32 var_2623 = 1.0;
    const wp::float32 var_2624 = -1.0;
    wp::float32 var_2625;
    wp::vec_t<3, wp::float32> var_2626;
    wp::vec_t<3, wp::float32> var_2627;
    const wp::float32 var_2628 = 0.0;
    wp::vec_t<3, wp::float32> var_2629;
    wp::float32 var_2630;
    wp::float32 var_2631;
    wp::float32 var_2632;
    wp::float32 var_2633;
    wp::float32 var_2634;
    wp::float32 var_2635;
    wp::float32 var_2636;
    wp::float32 var_2637;
    wp::float32 var_2638;
    wp::float32 var_2639;
    wp::float32 var_2640;
    wp::float32 var_2641;
    wp::float32 var_2642;
    wp::float32 var_2643;
    wp::float32 var_2644;
    wp::float32 var_2645;
    wp::float32 var_2646;
    bool var_2647;
    const bool var_2648 = false;
    bool var_2649;
    const wp::float32 var_2650 = 1.0;
    wp::float32 var_2651;
    wp::float32 var_2652;
    wp::float32 var_2653;
    wp::float32 var_2654;
    wp::float32 var_2655;
    wp::float32 var_2656;
    wp::float32 var_2657;
    wp::float32 var_2658;
    wp::float32 var_2659;
    wp::float32 var_2660;
    wp::float32 var_2661;
    const wp::int32 var_2662 = 1;
    wp::int32 var_2663;
    const wp::int32 var_2664 = 1;
    wp::int32 var_2665;
    const wp::int32 var_2666 = 1;
    bool var_2667;
    const wp::float32 var_2668 = 1.0;
    const wp::int32 var_2669 = 2;
    wp::float32 var_2670;
    wp::float32 var_2671;
    const wp::int32 var_2672 = -1;
    bool var_2673;
    const wp::float32 var_2674 = -1.0;
    const wp::int32 var_2675 = 0;
    wp::float32 var_2676;
    wp::float32 var_2677;
    wp::float32 var_2678;
    wp::float32 var_2679;
    wp::int32 var_2680;
    wp::float32 var_2681;
    wp::float32 var_2682;
    wp::int32 var_2683;
    const wp::float32 var_2684 = 1.0;
    bool var_2685;
    bool var_2686;
    const wp::float32 var_2687 = -1.0;
    bool var_2688;
    const wp::float32 var_2689 = 1.0;
    const wp::int32 var_2690 = 2;
    wp::float32 var_2691;
    wp::float32 var_2692;
    const wp::float32 var_2693 = -1.0;
    const wp::int32 var_2694 = 0;
    wp::float32 var_2695;
    wp::float32 var_2696;
    wp::float32 var_2697;
    wp::float32 var_2698;
    wp::int32 var_2699;
    const wp::int32 var_2700 = 1;
    bool var_2701;
    const wp::float32 var_2702 = 1.0;
    const wp::int32 var_2703 = 2;
    const wp::int32 var_2704 = -1;
    bool var_2705;
    const wp::float32 var_2706 = -1.0;
    const wp::int32 var_2707 = 0;
    wp::float32 var_2708;
    wp::int32 var_2709;
    wp::float32 var_2710;
    wp::int32 var_2711;
    wp::float32 var_2712;
    wp::float32 var_2713;
    wp::int32 var_2714;
    wp::int32 var_2715;
    wp::vec_t<3, wp::float32> var_2716;
    wp::vec_t<3, wp::float32> var_2717;
    wp::float32 var_2718;
    wp::float32 var_2719;
    const wp::int32 var_2720 = 3;
    wp::int32 var_2721;
    wp::int32 var_2722;
    wp::float32 var_2723;
    wp::float32 var_2724;
    bool var_2725;
    wp::float32 var_2726;
    wp::float32 var_2727;
    wp::float32 var_2728;
    const wp::int32 var_2729 = 6;
    wp::int32 var_2730;
    const wp::int32 var_2731 = 1;
    wp::int32 var_2732;
    wp::int32 var_2733;
    wp::int32 var_2734;
    wp::int32 var_2735;
    wp::int32 var_2736;
    wp::float32 var_2737;
    wp::float32 var_2738;
    wp::int32 var_2739;
    wp::int32 var_2740;
    wp::int32 var_2741;
    wp::float32 var_2742;
    wp::int32 var_2743;
    wp::float32 var_2744;
    wp::float32 var_2745;
    wp::int32 var_2746;
    wp::int32 var_2747;
    wp::int32 var_2748;
    wp::float32 var_2749;
    wp::int32 var_2750;
    wp::vec_t<3, wp::float32> var_2751;
    wp::float32 var_2752;
    wp::float32 var_2753;
    wp::float32 var_2754;
    wp::int32 var_2755;
    wp::int32 var_2756;
    bool var_2757;
    wp::int32 var_2758;
    wp::float32 var_2759;
    wp::float32 var_2760;
    wp::float32 var_2761;
    wp::int32 var_2762;
    wp::int32 var_2763;
    wp::int32 var_2764;
    wp::float32 var_2765;
    wp::int32 var_2766;
    wp::vec_t<3, wp::float32> var_2767;
    wp::vec_t<3, wp::float32> var_2768;
    wp::float32 var_2769;
    wp::float32 var_2770;
    wp::float32 var_2771;
    wp::float32 var_2772;
    wp::float32 var_2773;
    wp::float32 var_2774;
    wp::float32 var_2775;
    wp::float32 var_2776;
    wp::float32 var_2777;
    wp::int32 var_2778;
    wp::int32 var_2779;
    bool var_2780;
    wp::int32 var_2781;
    wp::float32 var_2782;
    bool var_2783;
    const wp::int32 var_2784 = 2;
    const bool var_2785 = true;
    const wp::int32 var_2786 = 1;
    wp::int32 var_2787;
    wp::int32 var_2788;
    const wp::int32 var_2789 = 0;
    bool var_2790;
    const bool var_2791 = false;
    bool var_2792;
    const wp::int32 var_2793 = -123;
    wp::int32 var_2794;
    const wp::int32 var_2795 = 1;
    wp::int32 var_2796;
    const wp::float32 var_2797 = 1.0;
    const wp::float32 var_2798 = -1.0;
    wp::float32 var_2799;
    const wp::int32 var_2800 = 2;
    wp::int32 var_2801;
    const wp::float32 var_2802 = 1.0;
    const wp::float32 var_2803 = -1.0;
    wp::float32 var_2804;
    const wp::int32 var_2805 = 4;
    wp::int32 var_2806;
    const wp::float32 var_2807 = 1.0;
    const wp::float32 var_2808 = -1.0;
    wp::float32 var_2809;
    wp::vec_t<3, wp::float32> var_2810;
    wp::vec_t<3, wp::float32> var_2811;
    const wp::float32 var_2812 = 0.0;
    wp::vec_t<3, wp::float32> var_2813;
    wp::float32 var_2814;
    wp::float32 var_2815;
    wp::float32 var_2816;
    wp::float32 var_2817;
    wp::float32 var_2818;
    wp::float32 var_2819;
    wp::float32 var_2820;
    wp::float32 var_2821;
    wp::float32 var_2822;
    wp::float32 var_2823;
    wp::float32 var_2824;
    wp::float32 var_2825;
    wp::float32 var_2826;
    wp::float32 var_2827;
    wp::float32 var_2828;
    wp::float32 var_2829;
    wp::float32 var_2830;
    bool var_2831;
    const bool var_2832 = false;
    bool var_2833;
    const wp::float32 var_2834 = 1.0;
    wp::float32 var_2835;
    wp::float32 var_2836;
    wp::float32 var_2837;
    wp::float32 var_2838;
    wp::float32 var_2839;
    wp::float32 var_2840;
    wp::float32 var_2841;
    wp::float32 var_2842;
    wp::float32 var_2843;
    wp::float32 var_2844;
    wp::float32 var_2845;
    const wp::int32 var_2846 = 1;
    wp::int32 var_2847;
    const wp::int32 var_2848 = 1;
    wp::int32 var_2849;
    const wp::int32 var_2850 = 1;
    bool var_2851;
    const wp::float32 var_2852 = 1.0;
    const wp::int32 var_2853 = 2;
    wp::float32 var_2854;
    wp::float32 var_2855;
    const wp::int32 var_2856 = -1;
    bool var_2857;
    const wp::float32 var_2858 = -1.0;
    const wp::int32 var_2859 = 0;
    wp::float32 var_2860;
    wp::float32 var_2861;
    wp::float32 var_2862;
    wp::float32 var_2863;
    wp::int32 var_2864;
    wp::float32 var_2865;
    wp::float32 var_2866;
    wp::int32 var_2867;
    const wp::float32 var_2868 = 1.0;
    bool var_2869;
    bool var_2870;
    const wp::float32 var_2871 = -1.0;
    bool var_2872;
    const wp::float32 var_2873 = 1.0;
    const wp::int32 var_2874 = 2;
    wp::float32 var_2875;
    wp::float32 var_2876;
    const wp::float32 var_2877 = -1.0;
    const wp::int32 var_2878 = 0;
    wp::float32 var_2879;
    wp::float32 var_2880;
    wp::float32 var_2881;
    wp::float32 var_2882;
    wp::int32 var_2883;
    const wp::int32 var_2884 = 1;
    bool var_2885;
    const wp::float32 var_2886 = 1.0;
    const wp::int32 var_2887 = 2;
    const wp::int32 var_2888 = -1;
    bool var_2889;
    const wp::float32 var_2890 = -1.0;
    const wp::int32 var_2891 = 0;
    wp::float32 var_2892;
    wp::int32 var_2893;
    wp::float32 var_2894;
    wp::int32 var_2895;
    wp::float32 var_2896;
    wp::float32 var_2897;
    wp::int32 var_2898;
    wp::int32 var_2899;
    wp::vec_t<3, wp::float32> var_2900;
    wp::vec_t<3, wp::float32> var_2901;
    wp::float32 var_2902;
    wp::float32 var_2903;
    const wp::int32 var_2904 = 3;
    wp::int32 var_2905;
    wp::int32 var_2906;
    wp::float32 var_2907;
    wp::float32 var_2908;
    bool var_2909;
    wp::float32 var_2910;
    wp::float32 var_2911;
    wp::float32 var_2912;
    const wp::int32 var_2913 = 6;
    wp::int32 var_2914;
    const wp::int32 var_2915 = 1;
    wp::int32 var_2916;
    wp::int32 var_2917;
    wp::int32 var_2918;
    wp::int32 var_2919;
    wp::int32 var_2920;
    wp::float32 var_2921;
    wp::float32 var_2922;
    wp::int32 var_2923;
    wp::int32 var_2924;
    wp::int32 var_2925;
    wp::float32 var_2926;
    wp::int32 var_2927;
    wp::float32 var_2928;
    wp::float32 var_2929;
    wp::int32 var_2930;
    wp::int32 var_2931;
    wp::int32 var_2932;
    wp::float32 var_2933;
    wp::int32 var_2934;
    wp::vec_t<3, wp::float32> var_2935;
    wp::float32 var_2936;
    wp::float32 var_2937;
    wp::float32 var_2938;
    wp::int32 var_2939;
    wp::int32 var_2940;
    bool var_2941;
    wp::int32 var_2942;
    wp::float32 var_2943;
    wp::float32 var_2944;
    wp::float32 var_2945;
    wp::int32 var_2946;
    wp::int32 var_2947;
    wp::int32 var_2948;
    wp::float32 var_2949;
    wp::int32 var_2950;
    wp::vec_t<3, wp::float32> var_2951;
    wp::vec_t<3, wp::float32> var_2952;
    wp::float32 var_2953;
    wp::float32 var_2954;
    wp::float32 var_2955;
    wp::float32 var_2956;
    wp::float32 var_2957;
    wp::float32 var_2958;
    wp::float32 var_2959;
    wp::float32 var_2960;
    wp::float32 var_2961;
    wp::int32 var_2962;
    wp::int32 var_2963;
    bool var_2964;
    wp::int32 var_2965;
    wp::float32 var_2966;
    bool var_2967;
    const wp::int32 var_2968 = 5;
    const wp::int32 var_2969 = 0;
    const bool var_2970 = true;
    const wp::int32 var_2971 = 1;
    wp::int32 var_2972;
    wp::int32 var_2973;
    const wp::int32 var_2974 = 0;
    bool var_2975;
    const bool var_2976 = false;
    bool var_2977;
    const wp::int32 var_2978 = -123;
    wp::int32 var_2979;
    const wp::int32 var_2980 = 1;
    wp::int32 var_2981;
    const wp::float32 var_2982 = 1.0;
    const wp::float32 var_2983 = -1.0;
    wp::float32 var_2984;
    const wp::int32 var_2985 = 2;
    wp::int32 var_2986;
    const wp::float32 var_2987 = 1.0;
    const wp::float32 var_2988 = -1.0;
    wp::float32 var_2989;
    const wp::int32 var_2990 = 4;
    wp::int32 var_2991;
    const wp::float32 var_2992 = 1.0;
    const wp::float32 var_2993 = -1.0;
    wp::float32 var_2994;
    wp::vec_t<3, wp::float32> var_2995;
    wp::vec_t<3, wp::float32> var_2996;
    const wp::float32 var_2997 = 0.0;
    wp::vec_t<3, wp::float32> var_2998;
    wp::float32 var_2999;
    wp::float32 var_3000;
    wp::float32 var_3001;
    wp::float32 var_3002;
    wp::float32 var_3003;
    wp::float32 var_3004;
    wp::float32 var_3005;
    wp::float32 var_3006;
    wp::float32 var_3007;
    wp::float32 var_3008;
    wp::float32 var_3009;
    wp::float32 var_3010;
    wp::float32 var_3011;
    wp::float32 var_3012;
    wp::float32 var_3013;
    wp::float32 var_3014;
    wp::float32 var_3015;
    bool var_3016;
    const bool var_3017 = false;
    bool var_3018;
    const wp::float32 var_3019 = 1.0;
    wp::float32 var_3020;
    wp::float32 var_3021;
    wp::float32 var_3022;
    wp::float32 var_3023;
    wp::float32 var_3024;
    wp::float32 var_3025;
    wp::float32 var_3026;
    wp::float32 var_3027;
    wp::float32 var_3028;
    wp::float32 var_3029;
    wp::float32 var_3030;
    const wp::int32 var_3031 = 1;
    wp::int32 var_3032;
    const wp::int32 var_3033 = 1;
    wp::int32 var_3034;
    const wp::int32 var_3035 = 1;
    bool var_3036;
    const wp::float32 var_3037 = 1.0;
    const wp::int32 var_3038 = 2;
    wp::float32 var_3039;
    wp::float32 var_3040;
    const wp::int32 var_3041 = -1;
    bool var_3042;
    const wp::float32 var_3043 = -1.0;
    const wp::int32 var_3044 = 0;
    wp::float32 var_3045;
    wp::float32 var_3046;
    wp::float32 var_3047;
    wp::float32 var_3048;
    wp::int32 var_3049;
    wp::float32 var_3050;
    wp::float32 var_3051;
    wp::int32 var_3052;
    const wp::float32 var_3053 = 1.0;
    bool var_3054;
    bool var_3055;
    const wp::float32 var_3056 = -1.0;
    bool var_3057;
    const wp::float32 var_3058 = 1.0;
    const wp::int32 var_3059 = 2;
    wp::float32 var_3060;
    wp::float32 var_3061;
    const wp::float32 var_3062 = -1.0;
    const wp::int32 var_3063 = 0;
    wp::float32 var_3064;
    wp::float32 var_3065;
    wp::float32 var_3066;
    wp::float32 var_3067;
    wp::int32 var_3068;
    const wp::int32 var_3069 = 1;
    bool var_3070;
    const wp::float32 var_3071 = 1.0;
    const wp::int32 var_3072 = 2;
    const wp::int32 var_3073 = -1;
    bool var_3074;
    const wp::float32 var_3075 = -1.0;
    const wp::int32 var_3076 = 0;
    wp::float32 var_3077;
    wp::int32 var_3078;
    wp::float32 var_3079;
    wp::int32 var_3080;
    wp::float32 var_3081;
    wp::float32 var_3082;
    wp::int32 var_3083;
    wp::int32 var_3084;
    wp::vec_t<3, wp::float32> var_3085;
    wp::vec_t<3, wp::float32> var_3086;
    wp::float32 var_3087;
    wp::float32 var_3088;
    const wp::int32 var_3089 = 3;
    wp::int32 var_3090;
    wp::int32 var_3091;
    wp::float32 var_3092;
    wp::float32 var_3093;
    bool var_3094;
    wp::float32 var_3095;
    wp::float32 var_3096;
    wp::float32 var_3097;
    const wp::int32 var_3098 = 6;
    wp::int32 var_3099;
    const wp::int32 var_3100 = 1;
    wp::int32 var_3101;
    wp::int32 var_3102;
    wp::int32 var_3103;
    wp::int32 var_3104;
    wp::int32 var_3105;
    wp::float32 var_3106;
    wp::float32 var_3107;
    wp::int32 var_3108;
    wp::int32 var_3109;
    wp::int32 var_3110;
    wp::float32 var_3111;
    wp::int32 var_3112;
    wp::float32 var_3113;
    wp::float32 var_3114;
    wp::int32 var_3115;
    wp::int32 var_3116;
    wp::int32 var_3117;
    wp::float32 var_3118;
    wp::int32 var_3119;
    wp::vec_t<3, wp::float32> var_3120;
    wp::float32 var_3121;
    wp::float32 var_3122;
    wp::float32 var_3123;
    wp::int32 var_3124;
    wp::int32 var_3125;
    bool var_3126;
    wp::int32 var_3127;
    wp::float32 var_3128;
    wp::float32 var_3129;
    wp::float32 var_3130;
    wp::int32 var_3131;
    wp::int32 var_3132;
    wp::int32 var_3133;
    wp::float32 var_3134;
    wp::int32 var_3135;
    wp::vec_t<3, wp::float32> var_3136;
    wp::vec_t<3, wp::float32> var_3137;
    wp::float32 var_3138;
    wp::float32 var_3139;
    wp::float32 var_3140;
    wp::float32 var_3141;
    wp::float32 var_3142;
    wp::float32 var_3143;
    wp::float32 var_3144;
    wp::float32 var_3145;
    wp::float32 var_3146;
    wp::int32 var_3147;
    wp::int32 var_3148;
    bool var_3149;
    wp::int32 var_3150;
    wp::float32 var_3151;
    bool var_3152;
    const wp::int32 var_3153 = 1;
    const bool var_3154 = true;
    const wp::int32 var_3155 = 1;
    wp::int32 var_3156;
    wp::int32 var_3157;
    const wp::int32 var_3158 = 0;
    bool var_3159;
    const bool var_3160 = false;
    bool var_3161;
    const wp::int32 var_3162 = -123;
    wp::int32 var_3163;
    const wp::int32 var_3164 = 1;
    wp::int32 var_3165;
    const wp::float32 var_3166 = 1.0;
    const wp::float32 var_3167 = -1.0;
    wp::float32 var_3168;
    const wp::int32 var_3169 = 2;
    wp::int32 var_3170;
    const wp::float32 var_3171 = 1.0;
    const wp::float32 var_3172 = -1.0;
    wp::float32 var_3173;
    const wp::int32 var_3174 = 4;
    wp::int32 var_3175;
    const wp::float32 var_3176 = 1.0;
    const wp::float32 var_3177 = -1.0;
    wp::float32 var_3178;
    wp::vec_t<3, wp::float32> var_3179;
    wp::vec_t<3, wp::float32> var_3180;
    const wp::float32 var_3181 = 0.0;
    wp::vec_t<3, wp::float32> var_3182;
    wp::float32 var_3183;
    wp::float32 var_3184;
    wp::float32 var_3185;
    wp::float32 var_3186;
    wp::float32 var_3187;
    wp::float32 var_3188;
    wp::float32 var_3189;
    wp::float32 var_3190;
    wp::float32 var_3191;
    wp::float32 var_3192;
    wp::float32 var_3193;
    wp::float32 var_3194;
    wp::float32 var_3195;
    wp::float32 var_3196;
    wp::float32 var_3197;
    wp::float32 var_3198;
    wp::float32 var_3199;
    bool var_3200;
    const bool var_3201 = false;
    bool var_3202;
    const wp::float32 var_3203 = 1.0;
    wp::float32 var_3204;
    wp::float32 var_3205;
    wp::float32 var_3206;
    wp::float32 var_3207;
    wp::float32 var_3208;
    wp::float32 var_3209;
    wp::float32 var_3210;
    wp::float32 var_3211;
    wp::float32 var_3212;
    wp::float32 var_3213;
    wp::float32 var_3214;
    const wp::int32 var_3215 = 1;
    wp::int32 var_3216;
    const wp::int32 var_3217 = 1;
    wp::int32 var_3218;
    const wp::int32 var_3219 = 1;
    bool var_3220;
    const wp::float32 var_3221 = 1.0;
    const wp::int32 var_3222 = 2;
    wp::float32 var_3223;
    wp::float32 var_3224;
    const wp::int32 var_3225 = -1;
    bool var_3226;
    const wp::float32 var_3227 = -1.0;
    const wp::int32 var_3228 = 0;
    wp::float32 var_3229;
    wp::float32 var_3230;
    wp::float32 var_3231;
    wp::float32 var_3232;
    wp::int32 var_3233;
    wp::float32 var_3234;
    wp::float32 var_3235;
    wp::int32 var_3236;
    const wp::float32 var_3237 = 1.0;
    bool var_3238;
    bool var_3239;
    const wp::float32 var_3240 = -1.0;
    bool var_3241;
    const wp::float32 var_3242 = 1.0;
    const wp::int32 var_3243 = 2;
    wp::float32 var_3244;
    wp::float32 var_3245;
    const wp::float32 var_3246 = -1.0;
    const wp::int32 var_3247 = 0;
    wp::float32 var_3248;
    wp::float32 var_3249;
    wp::float32 var_3250;
    wp::float32 var_3251;
    wp::int32 var_3252;
    const wp::int32 var_3253 = 1;
    bool var_3254;
    const wp::float32 var_3255 = 1.0;
    const wp::int32 var_3256 = 2;
    const wp::int32 var_3257 = -1;
    bool var_3258;
    const wp::float32 var_3259 = -1.0;
    const wp::int32 var_3260 = 0;
    wp::float32 var_3261;
    wp::int32 var_3262;
    wp::float32 var_3263;
    wp::int32 var_3264;
    wp::float32 var_3265;
    wp::float32 var_3266;
    wp::int32 var_3267;
    wp::int32 var_3268;
    wp::vec_t<3, wp::float32> var_3269;
    wp::vec_t<3, wp::float32> var_3270;
    wp::float32 var_3271;
    wp::float32 var_3272;
    const wp::int32 var_3273 = 3;
    wp::int32 var_3274;
    wp::int32 var_3275;
    wp::float32 var_3276;
    wp::float32 var_3277;
    bool var_3278;
    wp::float32 var_3279;
    wp::float32 var_3280;
    wp::float32 var_3281;
    const wp::int32 var_3282 = 6;
    wp::int32 var_3283;
    const wp::int32 var_3284 = 1;
    wp::int32 var_3285;
    wp::int32 var_3286;
    wp::int32 var_3287;
    wp::int32 var_3288;
    wp::int32 var_3289;
    wp::float32 var_3290;
    wp::float32 var_3291;
    wp::int32 var_3292;
    wp::int32 var_3293;
    wp::int32 var_3294;
    wp::float32 var_3295;
    wp::int32 var_3296;
    wp::float32 var_3297;
    wp::float32 var_3298;
    wp::int32 var_3299;
    wp::int32 var_3300;
    wp::int32 var_3301;
    wp::float32 var_3302;
    wp::int32 var_3303;
    wp::vec_t<3, wp::float32> var_3304;
    wp::float32 var_3305;
    wp::float32 var_3306;
    wp::float32 var_3307;
    wp::int32 var_3308;
    wp::int32 var_3309;
    bool var_3310;
    wp::int32 var_3311;
    wp::float32 var_3312;
    wp::float32 var_3313;
    wp::float32 var_3314;
    wp::int32 var_3315;
    wp::int32 var_3316;
    wp::int32 var_3317;
    wp::float32 var_3318;
    wp::int32 var_3319;
    wp::vec_t<3, wp::float32> var_3320;
    wp::vec_t<3, wp::float32> var_3321;
    wp::float32 var_3322;
    wp::float32 var_3323;
    wp::float32 var_3324;
    wp::float32 var_3325;
    wp::float32 var_3326;
    wp::float32 var_3327;
    wp::float32 var_3328;
    wp::float32 var_3329;
    wp::float32 var_3330;
    wp::int32 var_3331;
    wp::int32 var_3332;
    bool var_3333;
    wp::int32 var_3334;
    wp::float32 var_3335;
    bool var_3336;
    const wp::int32 var_3337 = 2;
    const bool var_3338 = true;
    const wp::int32 var_3339 = 1;
    wp::int32 var_3340;
    wp::int32 var_3341;
    const wp::int32 var_3342 = 0;
    bool var_3343;
    const bool var_3344 = false;
    bool var_3345;
    const wp::int32 var_3346 = -123;
    wp::int32 var_3347;
    const wp::int32 var_3348 = 1;
    wp::int32 var_3349;
    const wp::float32 var_3350 = 1.0;
    const wp::float32 var_3351 = -1.0;
    wp::float32 var_3352;
    const wp::int32 var_3353 = 2;
    wp::int32 var_3354;
    const wp::float32 var_3355 = 1.0;
    const wp::float32 var_3356 = -1.0;
    wp::float32 var_3357;
    const wp::int32 var_3358 = 4;
    wp::int32 var_3359;
    const wp::float32 var_3360 = 1.0;
    const wp::float32 var_3361 = -1.0;
    wp::float32 var_3362;
    wp::vec_t<3, wp::float32> var_3363;
    wp::vec_t<3, wp::float32> var_3364;
    const wp::float32 var_3365 = 0.0;
    wp::vec_t<3, wp::float32> var_3366;
    wp::float32 var_3367;
    wp::float32 var_3368;
    wp::float32 var_3369;
    wp::float32 var_3370;
    wp::float32 var_3371;
    wp::float32 var_3372;
    wp::float32 var_3373;
    wp::float32 var_3374;
    wp::float32 var_3375;
    wp::float32 var_3376;
    wp::float32 var_3377;
    wp::float32 var_3378;
    wp::float32 var_3379;
    wp::float32 var_3380;
    wp::float32 var_3381;
    wp::float32 var_3382;
    wp::float32 var_3383;
    bool var_3384;
    const bool var_3385 = false;
    bool var_3386;
    const wp::float32 var_3387 = 1.0;
    wp::float32 var_3388;
    wp::float32 var_3389;
    wp::float32 var_3390;
    wp::float32 var_3391;
    wp::float32 var_3392;
    wp::float32 var_3393;
    wp::float32 var_3394;
    wp::float32 var_3395;
    wp::float32 var_3396;
    wp::float32 var_3397;
    wp::float32 var_3398;
    const wp::int32 var_3399 = 1;
    wp::int32 var_3400;
    const wp::int32 var_3401 = 1;
    wp::int32 var_3402;
    const wp::int32 var_3403 = 1;
    bool var_3404;
    const wp::float32 var_3405 = 1.0;
    const wp::int32 var_3406 = 2;
    wp::float32 var_3407;
    wp::float32 var_3408;
    const wp::int32 var_3409 = -1;
    bool var_3410;
    const wp::float32 var_3411 = -1.0;
    const wp::int32 var_3412 = 0;
    wp::float32 var_3413;
    wp::float32 var_3414;
    wp::float32 var_3415;
    wp::float32 var_3416;
    wp::int32 var_3417;
    wp::float32 var_3418;
    wp::float32 var_3419;
    wp::int32 var_3420;
    const wp::float32 var_3421 = 1.0;
    bool var_3422;
    bool var_3423;
    const wp::float32 var_3424 = -1.0;
    bool var_3425;
    const wp::float32 var_3426 = 1.0;
    const wp::int32 var_3427 = 2;
    wp::float32 var_3428;
    wp::float32 var_3429;
    const wp::float32 var_3430 = -1.0;
    const wp::int32 var_3431 = 0;
    wp::float32 var_3432;
    wp::float32 var_3433;
    wp::float32 var_3434;
    wp::float32 var_3435;
    wp::int32 var_3436;
    const wp::int32 var_3437 = 1;
    bool var_3438;
    const wp::float32 var_3439 = 1.0;
    const wp::int32 var_3440 = 2;
    const wp::int32 var_3441 = -1;
    bool var_3442;
    const wp::float32 var_3443 = -1.0;
    const wp::int32 var_3444 = 0;
    wp::float32 var_3445;
    wp::int32 var_3446;
    wp::float32 var_3447;
    wp::int32 var_3448;
    wp::float32 var_3449;
    wp::float32 var_3450;
    wp::int32 var_3451;
    wp::int32 var_3452;
    wp::vec_t<3, wp::float32> var_3453;
    wp::vec_t<3, wp::float32> var_3454;
    wp::float32 var_3455;
    wp::float32 var_3456;
    const wp::int32 var_3457 = 3;
    wp::int32 var_3458;
    wp::int32 var_3459;
    wp::float32 var_3460;
    wp::float32 var_3461;
    bool var_3462;
    wp::float32 var_3463;
    wp::float32 var_3464;
    wp::float32 var_3465;
    const wp::int32 var_3466 = 6;
    wp::int32 var_3467;
    const wp::int32 var_3468 = 1;
    wp::int32 var_3469;
    wp::int32 var_3470;
    wp::int32 var_3471;
    wp::int32 var_3472;
    wp::int32 var_3473;
    wp::float32 var_3474;
    wp::float32 var_3475;
    wp::int32 var_3476;
    wp::int32 var_3477;
    wp::int32 var_3478;
    wp::float32 var_3479;
    wp::int32 var_3480;
    wp::float32 var_3481;
    wp::float32 var_3482;
    wp::int32 var_3483;
    wp::int32 var_3484;
    wp::int32 var_3485;
    wp::float32 var_3486;
    wp::int32 var_3487;
    wp::vec_t<3, wp::float32> var_3488;
    wp::float32 var_3489;
    wp::float32 var_3490;
    wp::float32 var_3491;
    wp::int32 var_3492;
    wp::int32 var_3493;
    bool var_3494;
    wp::int32 var_3495;
    wp::float32 var_3496;
    wp::float32 var_3497;
    wp::float32 var_3498;
    wp::int32 var_3499;
    wp::int32 var_3500;
    wp::int32 var_3501;
    wp::float32 var_3502;
    wp::int32 var_3503;
    wp::vec_t<3, wp::float32> var_3504;
    wp::vec_t<3, wp::float32> var_3505;
    wp::float32 var_3506;
    wp::float32 var_3507;
    wp::float32 var_3508;
    wp::float32 var_3509;
    wp::float32 var_3510;
    wp::float32 var_3511;
    wp::float32 var_3512;
    wp::float32 var_3513;
    wp::float32 var_3514;
    wp::int32 var_3515;
    wp::int32 var_3516;
    bool var_3517;
    wp::int32 var_3518;
    wp::float32 var_3519;
    bool var_3520;
    const wp::int32 var_3521 = 6;
    const wp::int32 var_3522 = 0;
    const bool var_3523 = true;
    const wp::int32 var_3524 = 1;
    wp::int32 var_3525;
    wp::int32 var_3526;
    const wp::int32 var_3527 = 0;
    bool var_3528;
    const bool var_3529 = false;
    bool var_3530;
    const wp::int32 var_3531 = -123;
    wp::int32 var_3532;
    const wp::int32 var_3533 = 1;
    wp::int32 var_3534;
    const wp::float32 var_3535 = 1.0;
    const wp::float32 var_3536 = -1.0;
    wp::float32 var_3537;
    const wp::int32 var_3538 = 2;
    wp::int32 var_3539;
    const wp::float32 var_3540 = 1.0;
    const wp::float32 var_3541 = -1.0;
    wp::float32 var_3542;
    const wp::int32 var_3543 = 4;
    wp::int32 var_3544;
    const wp::float32 var_3545 = 1.0;
    const wp::float32 var_3546 = -1.0;
    wp::float32 var_3547;
    wp::vec_t<3, wp::float32> var_3548;
    wp::vec_t<3, wp::float32> var_3549;
    const wp::float32 var_3550 = 0.0;
    wp::vec_t<3, wp::float32> var_3551;
    wp::float32 var_3552;
    wp::float32 var_3553;
    wp::float32 var_3554;
    wp::float32 var_3555;
    wp::float32 var_3556;
    wp::float32 var_3557;
    wp::float32 var_3558;
    wp::float32 var_3559;
    wp::float32 var_3560;
    wp::float32 var_3561;
    wp::float32 var_3562;
    wp::float32 var_3563;
    wp::float32 var_3564;
    wp::float32 var_3565;
    wp::float32 var_3566;
    wp::float32 var_3567;
    wp::float32 var_3568;
    bool var_3569;
    const bool var_3570 = false;
    bool var_3571;
    const wp::float32 var_3572 = 1.0;
    wp::float32 var_3573;
    wp::float32 var_3574;
    wp::float32 var_3575;
    wp::float32 var_3576;
    wp::float32 var_3577;
    wp::float32 var_3578;
    wp::float32 var_3579;
    wp::float32 var_3580;
    wp::float32 var_3581;
    wp::float32 var_3582;
    wp::float32 var_3583;
    const wp::int32 var_3584 = 1;
    wp::int32 var_3585;
    const wp::int32 var_3586 = 1;
    wp::int32 var_3587;
    const wp::int32 var_3588 = 1;
    bool var_3589;
    const wp::float32 var_3590 = 1.0;
    const wp::int32 var_3591 = 2;
    wp::float32 var_3592;
    wp::float32 var_3593;
    const wp::int32 var_3594 = -1;
    bool var_3595;
    const wp::float32 var_3596 = -1.0;
    const wp::int32 var_3597 = 0;
    wp::float32 var_3598;
    wp::float32 var_3599;
    wp::float32 var_3600;
    wp::float32 var_3601;
    wp::int32 var_3602;
    wp::float32 var_3603;
    wp::float32 var_3604;
    wp::int32 var_3605;
    const wp::float32 var_3606 = 1.0;
    bool var_3607;
    bool var_3608;
    const wp::float32 var_3609 = -1.0;
    bool var_3610;
    const wp::float32 var_3611 = 1.0;
    const wp::int32 var_3612 = 2;
    wp::float32 var_3613;
    wp::float32 var_3614;
    const wp::float32 var_3615 = -1.0;
    const wp::int32 var_3616 = 0;
    wp::float32 var_3617;
    wp::float32 var_3618;
    wp::float32 var_3619;
    wp::float32 var_3620;
    wp::int32 var_3621;
    const wp::int32 var_3622 = 1;
    bool var_3623;
    const wp::float32 var_3624 = 1.0;
    const wp::int32 var_3625 = 2;
    const wp::int32 var_3626 = -1;
    bool var_3627;
    const wp::float32 var_3628 = -1.0;
    const wp::int32 var_3629 = 0;
    wp::float32 var_3630;
    wp::int32 var_3631;
    wp::float32 var_3632;
    wp::int32 var_3633;
    wp::float32 var_3634;
    wp::float32 var_3635;
    wp::int32 var_3636;
    wp::int32 var_3637;
    wp::vec_t<3, wp::float32> var_3638;
    wp::vec_t<3, wp::float32> var_3639;
    wp::float32 var_3640;
    wp::float32 var_3641;
    const wp::int32 var_3642 = 3;
    wp::int32 var_3643;
    wp::int32 var_3644;
    wp::float32 var_3645;
    wp::float32 var_3646;
    bool var_3647;
    wp::float32 var_3648;
    wp::float32 var_3649;
    wp::float32 var_3650;
    const wp::int32 var_3651 = 6;
    wp::int32 var_3652;
    const wp::int32 var_3653 = 1;
    wp::int32 var_3654;
    wp::int32 var_3655;
    wp::int32 var_3656;
    wp::int32 var_3657;
    wp::int32 var_3658;
    wp::float32 var_3659;
    wp::float32 var_3660;
    wp::int32 var_3661;
    wp::int32 var_3662;
    wp::int32 var_3663;
    wp::float32 var_3664;
    wp::int32 var_3665;
    wp::float32 var_3666;
    wp::float32 var_3667;
    wp::int32 var_3668;
    wp::int32 var_3669;
    wp::int32 var_3670;
    wp::float32 var_3671;
    wp::int32 var_3672;
    wp::vec_t<3, wp::float32> var_3673;
    wp::float32 var_3674;
    wp::float32 var_3675;
    wp::float32 var_3676;
    wp::int32 var_3677;
    wp::int32 var_3678;
    bool var_3679;
    wp::int32 var_3680;
    wp::float32 var_3681;
    wp::float32 var_3682;
    wp::float32 var_3683;
    wp::int32 var_3684;
    wp::int32 var_3685;
    wp::int32 var_3686;
    wp::float32 var_3687;
    wp::int32 var_3688;
    wp::vec_t<3, wp::float32> var_3689;
    wp::vec_t<3, wp::float32> var_3690;
    wp::float32 var_3691;
    wp::float32 var_3692;
    wp::float32 var_3693;
    wp::float32 var_3694;
    wp::float32 var_3695;
    wp::float32 var_3696;
    wp::float32 var_3697;
    wp::float32 var_3698;
    wp::float32 var_3699;
    wp::int32 var_3700;
    wp::int32 var_3701;
    bool var_3702;
    wp::int32 var_3703;
    wp::float32 var_3704;
    bool var_3705;
    const wp::int32 var_3706 = 1;
    const bool var_3707 = true;
    const wp::int32 var_3708 = 1;
    wp::int32 var_3709;
    wp::int32 var_3710;
    const wp::int32 var_3711 = 0;
    bool var_3712;
    const bool var_3713 = false;
    bool var_3714;
    const wp::int32 var_3715 = -123;
    wp::int32 var_3716;
    const wp::int32 var_3717 = 1;
    wp::int32 var_3718;
    const wp::float32 var_3719 = 1.0;
    const wp::float32 var_3720 = -1.0;
    wp::float32 var_3721;
    const wp::int32 var_3722 = 2;
    wp::int32 var_3723;
    const wp::float32 var_3724 = 1.0;
    const wp::float32 var_3725 = -1.0;
    wp::float32 var_3726;
    const wp::int32 var_3727 = 4;
    wp::int32 var_3728;
    const wp::float32 var_3729 = 1.0;
    const wp::float32 var_3730 = -1.0;
    wp::float32 var_3731;
    wp::vec_t<3, wp::float32> var_3732;
    wp::vec_t<3, wp::float32> var_3733;
    const wp::float32 var_3734 = 0.0;
    wp::vec_t<3, wp::float32> var_3735;
    wp::float32 var_3736;
    wp::float32 var_3737;
    wp::float32 var_3738;
    wp::float32 var_3739;
    wp::float32 var_3740;
    wp::float32 var_3741;
    wp::float32 var_3742;
    wp::float32 var_3743;
    wp::float32 var_3744;
    wp::float32 var_3745;
    wp::float32 var_3746;
    wp::float32 var_3747;
    wp::float32 var_3748;
    wp::float32 var_3749;
    wp::float32 var_3750;
    wp::float32 var_3751;
    wp::float32 var_3752;
    bool var_3753;
    const bool var_3754 = false;
    bool var_3755;
    const wp::float32 var_3756 = 1.0;
    wp::float32 var_3757;
    wp::float32 var_3758;
    wp::float32 var_3759;
    wp::float32 var_3760;
    wp::float32 var_3761;
    wp::float32 var_3762;
    wp::float32 var_3763;
    wp::float32 var_3764;
    wp::float32 var_3765;
    wp::float32 var_3766;
    wp::float32 var_3767;
    const wp::int32 var_3768 = 1;
    wp::int32 var_3769;
    const wp::int32 var_3770 = 1;
    wp::int32 var_3771;
    const wp::int32 var_3772 = 1;
    bool var_3773;
    const wp::float32 var_3774 = 1.0;
    const wp::int32 var_3775 = 2;
    wp::float32 var_3776;
    wp::float32 var_3777;
    const wp::int32 var_3778 = -1;
    bool var_3779;
    const wp::float32 var_3780 = -1.0;
    const wp::int32 var_3781 = 0;
    wp::float32 var_3782;
    wp::float32 var_3783;
    wp::float32 var_3784;
    wp::float32 var_3785;
    wp::int32 var_3786;
    wp::float32 var_3787;
    wp::float32 var_3788;
    wp::int32 var_3789;
    const wp::float32 var_3790 = 1.0;
    bool var_3791;
    bool var_3792;
    const wp::float32 var_3793 = -1.0;
    bool var_3794;
    const wp::float32 var_3795 = 1.0;
    const wp::int32 var_3796 = 2;
    wp::float32 var_3797;
    wp::float32 var_3798;
    const wp::float32 var_3799 = -1.0;
    const wp::int32 var_3800 = 0;
    wp::float32 var_3801;
    wp::float32 var_3802;
    wp::float32 var_3803;
    wp::float32 var_3804;
    wp::int32 var_3805;
    const wp::int32 var_3806 = 1;
    bool var_3807;
    const wp::float32 var_3808 = 1.0;
    const wp::int32 var_3809 = 2;
    const wp::int32 var_3810 = -1;
    bool var_3811;
    const wp::float32 var_3812 = -1.0;
    const wp::int32 var_3813 = 0;
    wp::float32 var_3814;
    wp::int32 var_3815;
    wp::float32 var_3816;
    wp::int32 var_3817;
    wp::float32 var_3818;
    wp::float32 var_3819;
    wp::int32 var_3820;
    wp::int32 var_3821;
    wp::vec_t<3, wp::float32> var_3822;
    wp::vec_t<3, wp::float32> var_3823;
    wp::float32 var_3824;
    wp::float32 var_3825;
    const wp::int32 var_3826 = 3;
    wp::int32 var_3827;
    wp::int32 var_3828;
    wp::float32 var_3829;
    wp::float32 var_3830;
    bool var_3831;
    wp::float32 var_3832;
    wp::float32 var_3833;
    wp::float32 var_3834;
    const wp::int32 var_3835 = 6;
    wp::int32 var_3836;
    const wp::int32 var_3837 = 1;
    wp::int32 var_3838;
    wp::int32 var_3839;
    wp::int32 var_3840;
    wp::int32 var_3841;
    wp::int32 var_3842;
    wp::float32 var_3843;
    wp::float32 var_3844;
    wp::int32 var_3845;
    wp::int32 var_3846;
    wp::int32 var_3847;
    wp::float32 var_3848;
    wp::int32 var_3849;
    wp::float32 var_3850;
    wp::float32 var_3851;
    wp::int32 var_3852;
    wp::int32 var_3853;
    wp::int32 var_3854;
    wp::float32 var_3855;
    wp::int32 var_3856;
    wp::vec_t<3, wp::float32> var_3857;
    wp::float32 var_3858;
    wp::float32 var_3859;
    wp::float32 var_3860;
    wp::int32 var_3861;
    wp::int32 var_3862;
    bool var_3863;
    wp::int32 var_3864;
    wp::float32 var_3865;
    wp::float32 var_3866;
    wp::float32 var_3867;
    wp::int32 var_3868;
    wp::int32 var_3869;
    wp::int32 var_3870;
    wp::float32 var_3871;
    wp::int32 var_3872;
    wp::vec_t<3, wp::float32> var_3873;
    wp::vec_t<3, wp::float32> var_3874;
    wp::float32 var_3875;
    wp::float32 var_3876;
    wp::float32 var_3877;
    wp::float32 var_3878;
    wp::float32 var_3879;
    wp::float32 var_3880;
    wp::float32 var_3881;
    wp::float32 var_3882;
    wp::float32 var_3883;
    wp::int32 var_3884;
    wp::int32 var_3885;
    bool var_3886;
    wp::int32 var_3887;
    wp::float32 var_3888;
    bool var_3889;
    const wp::int32 var_3890 = 2;
    const bool var_3891 = true;
    const wp::int32 var_3892 = 1;
    wp::int32 var_3893;
    wp::int32 var_3894;
    const wp::int32 var_3895 = 0;
    bool var_3896;
    const bool var_3897 = false;
    bool var_3898;
    const wp::int32 var_3899 = -123;
    wp::int32 var_3900;
    const wp::int32 var_3901 = 1;
    wp::int32 var_3902;
    const wp::float32 var_3903 = 1.0;
    const wp::float32 var_3904 = -1.0;
    wp::float32 var_3905;
    const wp::int32 var_3906 = 2;
    wp::int32 var_3907;
    const wp::float32 var_3908 = 1.0;
    const wp::float32 var_3909 = -1.0;
    wp::float32 var_3910;
    const wp::int32 var_3911 = 4;
    wp::int32 var_3912;
    const wp::float32 var_3913 = 1.0;
    const wp::float32 var_3914 = -1.0;
    wp::float32 var_3915;
    wp::vec_t<3, wp::float32> var_3916;
    wp::vec_t<3, wp::float32> var_3917;
    const wp::float32 var_3918 = 0.0;
    wp::vec_t<3, wp::float32> var_3919;
    wp::float32 var_3920;
    wp::float32 var_3921;
    wp::float32 var_3922;
    wp::float32 var_3923;
    wp::float32 var_3924;
    wp::float32 var_3925;
    wp::float32 var_3926;
    wp::float32 var_3927;
    wp::float32 var_3928;
    wp::float32 var_3929;
    wp::float32 var_3930;
    wp::float32 var_3931;
    wp::float32 var_3932;
    wp::float32 var_3933;
    wp::float32 var_3934;
    wp::float32 var_3935;
    wp::float32 var_3936;
    bool var_3937;
    const bool var_3938 = false;
    bool var_3939;
    const wp::float32 var_3940 = 1.0;
    wp::float32 var_3941;
    wp::float32 var_3942;
    wp::float32 var_3943;
    wp::float32 var_3944;
    wp::float32 var_3945;
    wp::float32 var_3946;
    wp::float32 var_3947;
    wp::float32 var_3948;
    wp::float32 var_3949;
    wp::float32 var_3950;
    wp::float32 var_3951;
    const wp::int32 var_3952 = 1;
    wp::int32 var_3953;
    const wp::int32 var_3954 = 1;
    wp::int32 var_3955;
    const wp::int32 var_3956 = 1;
    bool var_3957;
    const wp::float32 var_3958 = 1.0;
    const wp::int32 var_3959 = 2;
    wp::float32 var_3960;
    wp::float32 var_3961;
    const wp::int32 var_3962 = -1;
    bool var_3963;
    const wp::float32 var_3964 = -1.0;
    const wp::int32 var_3965 = 0;
    wp::float32 var_3966;
    wp::float32 var_3967;
    wp::float32 var_3968;
    wp::float32 var_3969;
    wp::int32 var_3970;
    wp::float32 var_3971;
    wp::float32 var_3972;
    wp::int32 var_3973;
    const wp::float32 var_3974 = 1.0;
    bool var_3975;
    bool var_3976;
    const wp::float32 var_3977 = -1.0;
    bool var_3978;
    const wp::float32 var_3979 = 1.0;
    const wp::int32 var_3980 = 2;
    wp::float32 var_3981;
    wp::float32 var_3982;
    const wp::float32 var_3983 = -1.0;
    const wp::int32 var_3984 = 0;
    wp::float32 var_3985;
    wp::float32 var_3986;
    wp::float32 var_3987;
    wp::float32 var_3988;
    wp::int32 var_3989;
    const wp::int32 var_3990 = 1;
    bool var_3991;
    const wp::float32 var_3992 = 1.0;
    const wp::int32 var_3993 = 2;
    const wp::int32 var_3994 = -1;
    bool var_3995;
    const wp::float32 var_3996 = -1.0;
    const wp::int32 var_3997 = 0;
    wp::float32 var_3998;
    wp::int32 var_3999;
    wp::float32 var_4000;
    wp::int32 var_4001;
    wp::float32 var_4002;
    wp::float32 var_4003;
    wp::int32 var_4004;
    wp::int32 var_4005;
    wp::vec_t<3, wp::float32> var_4006;
    wp::vec_t<3, wp::float32> var_4007;
    wp::float32 var_4008;
    wp::float32 var_4009;
    const wp::int32 var_4010 = 3;
    wp::int32 var_4011;
    wp::int32 var_4012;
    wp::float32 var_4013;
    wp::float32 var_4014;
    bool var_4015;
    wp::float32 var_4016;
    wp::float32 var_4017;
    wp::float32 var_4018;
    const wp::int32 var_4019 = 6;
    wp::int32 var_4020;
    const wp::int32 var_4021 = 1;
    wp::int32 var_4022;
    wp::int32 var_4023;
    wp::int32 var_4024;
    wp::int32 var_4025;
    wp::int32 var_4026;
    wp::float32 var_4027;
    wp::float32 var_4028;
    wp::int32 var_4029;
    wp::int32 var_4030;
    wp::int32 var_4031;
    wp::float32 var_4032;
    wp::int32 var_4033;
    wp::float32 var_4034;
    wp::float32 var_4035;
    wp::int32 var_4036;
    wp::int32 var_4037;
    wp::int32 var_4038;
    wp::float32 var_4039;
    wp::int32 var_4040;
    wp::vec_t<3, wp::float32> var_4041;
    wp::float32 var_4042;
    wp::float32 var_4043;
    wp::float32 var_4044;
    wp::int32 var_4045;
    wp::int32 var_4046;
    bool var_4047;
    wp::int32 var_4048;
    wp::float32 var_4049;
    wp::float32 var_4050;
    wp::float32 var_4051;
    wp::int32 var_4052;
    wp::int32 var_4053;
    wp::int32 var_4054;
    wp::float32 var_4055;
    wp::int32 var_4056;
    wp::vec_t<3, wp::float32> var_4057;
    wp::vec_t<3, wp::float32> var_4058;
    wp::float32 var_4059;
    wp::float32 var_4060;
    wp::float32 var_4061;
    wp::float32 var_4062;
    wp::float32 var_4063;
    wp::float32 var_4064;
    wp::float32 var_4065;
    wp::float32 var_4066;
    wp::float32 var_4067;
    wp::int32 var_4068;
    wp::int32 var_4069;
    bool var_4070;
    wp::int32 var_4071;
    wp::float32 var_4072;
    bool var_4073;
    const wp::int32 var_4074 = 7;
    const wp::int32 var_4075 = 0;
    const bool var_4076 = true;
    const wp::int32 var_4077 = 1;
    wp::int32 var_4078;
    wp::int32 var_4079;
    const wp::int32 var_4080 = 0;
    bool var_4081;
    const bool var_4082 = false;
    bool var_4083;
    const wp::int32 var_4084 = -123;
    wp::int32 var_4085;
    const wp::int32 var_4086 = 1;
    wp::int32 var_4087;
    const wp::float32 var_4088 = 1.0;
    const wp::float32 var_4089 = -1.0;
    wp::float32 var_4090;
    const wp::int32 var_4091 = 2;
    wp::int32 var_4092;
    const wp::float32 var_4093 = 1.0;
    const wp::float32 var_4094 = -1.0;
    wp::float32 var_4095;
    const wp::int32 var_4096 = 4;
    wp::int32 var_4097;
    const wp::float32 var_4098 = 1.0;
    const wp::float32 var_4099 = -1.0;
    wp::float32 var_4100;
    wp::vec_t<3, wp::float32> var_4101;
    wp::vec_t<3, wp::float32> var_4102;
    const wp::float32 var_4103 = 0.0;
    wp::vec_t<3, wp::float32> var_4104;
    wp::float32 var_4105;
    wp::float32 var_4106;
    wp::float32 var_4107;
    wp::float32 var_4108;
    wp::float32 var_4109;
    wp::float32 var_4110;
    wp::float32 var_4111;
    wp::float32 var_4112;
    wp::float32 var_4113;
    wp::float32 var_4114;
    wp::float32 var_4115;
    wp::float32 var_4116;
    wp::float32 var_4117;
    wp::float32 var_4118;
    wp::float32 var_4119;
    wp::float32 var_4120;
    wp::float32 var_4121;
    bool var_4122;
    const bool var_4123 = false;
    bool var_4124;
    const wp::float32 var_4125 = 1.0;
    wp::float32 var_4126;
    wp::float32 var_4127;
    wp::float32 var_4128;
    wp::float32 var_4129;
    wp::float32 var_4130;
    wp::float32 var_4131;
    wp::float32 var_4132;
    wp::float32 var_4133;
    wp::float32 var_4134;
    wp::float32 var_4135;
    wp::float32 var_4136;
    const wp::int32 var_4137 = 1;
    wp::int32 var_4138;
    const wp::int32 var_4139 = 1;
    wp::int32 var_4140;
    const wp::int32 var_4141 = 1;
    bool var_4142;
    const wp::float32 var_4143 = 1.0;
    const wp::int32 var_4144 = 2;
    wp::float32 var_4145;
    wp::float32 var_4146;
    const wp::int32 var_4147 = -1;
    bool var_4148;
    const wp::float32 var_4149 = -1.0;
    const wp::int32 var_4150 = 0;
    wp::float32 var_4151;
    wp::float32 var_4152;
    wp::float32 var_4153;
    wp::float32 var_4154;
    wp::int32 var_4155;
    wp::float32 var_4156;
    wp::float32 var_4157;
    wp::int32 var_4158;
    const wp::float32 var_4159 = 1.0;
    bool var_4160;
    bool var_4161;
    const wp::float32 var_4162 = -1.0;
    bool var_4163;
    const wp::float32 var_4164 = 1.0;
    const wp::int32 var_4165 = 2;
    wp::float32 var_4166;
    wp::float32 var_4167;
    const wp::float32 var_4168 = -1.0;
    const wp::int32 var_4169 = 0;
    wp::float32 var_4170;
    wp::float32 var_4171;
    wp::float32 var_4172;
    wp::float32 var_4173;
    wp::int32 var_4174;
    const wp::int32 var_4175 = 1;
    bool var_4176;
    const wp::float32 var_4177 = 1.0;
    const wp::int32 var_4178 = 2;
    const wp::int32 var_4179 = -1;
    bool var_4180;
    const wp::float32 var_4181 = -1.0;
    const wp::int32 var_4182 = 0;
    wp::float32 var_4183;
    wp::int32 var_4184;
    wp::float32 var_4185;
    wp::int32 var_4186;
    wp::float32 var_4187;
    wp::float32 var_4188;
    wp::int32 var_4189;
    wp::int32 var_4190;
    wp::vec_t<3, wp::float32> var_4191;
    wp::vec_t<3, wp::float32> var_4192;
    wp::float32 var_4193;
    wp::float32 var_4194;
    const wp::int32 var_4195 = 3;
    wp::int32 var_4196;
    wp::int32 var_4197;
    wp::float32 var_4198;
    wp::float32 var_4199;
    bool var_4200;
    wp::float32 var_4201;
    wp::float32 var_4202;
    wp::float32 var_4203;
    const wp::int32 var_4204 = 6;
    wp::int32 var_4205;
    const wp::int32 var_4206 = 1;
    wp::int32 var_4207;
    wp::int32 var_4208;
    wp::int32 var_4209;
    wp::int32 var_4210;
    wp::int32 var_4211;
    wp::float32 var_4212;
    wp::float32 var_4213;
    wp::int32 var_4214;
    wp::int32 var_4215;
    wp::int32 var_4216;
    wp::float32 var_4217;
    wp::int32 var_4218;
    wp::float32 var_4219;
    wp::float32 var_4220;
    wp::int32 var_4221;
    wp::int32 var_4222;
    wp::int32 var_4223;
    wp::float32 var_4224;
    wp::int32 var_4225;
    wp::vec_t<3, wp::float32> var_4226;
    wp::float32 var_4227;
    wp::float32 var_4228;
    wp::float32 var_4229;
    wp::int32 var_4230;
    wp::int32 var_4231;
    bool var_4232;
    wp::int32 var_4233;
    wp::float32 var_4234;
    wp::float32 var_4235;
    wp::float32 var_4236;
    wp::int32 var_4237;
    wp::int32 var_4238;
    wp::int32 var_4239;
    wp::float32 var_4240;
    wp::int32 var_4241;
    wp::vec_t<3, wp::float32> var_4242;
    wp::vec_t<3, wp::float32> var_4243;
    wp::float32 var_4244;
    wp::float32 var_4245;
    wp::float32 var_4246;
    wp::float32 var_4247;
    wp::float32 var_4248;
    wp::float32 var_4249;
    wp::float32 var_4250;
    wp::float32 var_4251;
    wp::float32 var_4252;
    wp::int32 var_4253;
    wp::int32 var_4254;
    bool var_4255;
    wp::int32 var_4256;
    wp::float32 var_4257;
    bool var_4258;
    const wp::int32 var_4259 = 1;
    const bool var_4260 = true;
    const wp::int32 var_4261 = 1;
    wp::int32 var_4262;
    wp::int32 var_4263;
    const wp::int32 var_4264 = 0;
    bool var_4265;
    const bool var_4266 = false;
    bool var_4267;
    const wp::int32 var_4268 = -123;
    wp::int32 var_4269;
    const wp::int32 var_4270 = 1;
    wp::int32 var_4271;
    const wp::float32 var_4272 = 1.0;
    const wp::float32 var_4273 = -1.0;
    wp::float32 var_4274;
    const wp::int32 var_4275 = 2;
    wp::int32 var_4276;
    const wp::float32 var_4277 = 1.0;
    const wp::float32 var_4278 = -1.0;
    wp::float32 var_4279;
    const wp::int32 var_4280 = 4;
    wp::int32 var_4281;
    const wp::float32 var_4282 = 1.0;
    const wp::float32 var_4283 = -1.0;
    wp::float32 var_4284;
    wp::vec_t<3, wp::float32> var_4285;
    wp::vec_t<3, wp::float32> var_4286;
    const wp::float32 var_4287 = 0.0;
    wp::vec_t<3, wp::float32> var_4288;
    wp::float32 var_4289;
    wp::float32 var_4290;
    wp::float32 var_4291;
    wp::float32 var_4292;
    wp::float32 var_4293;
    wp::float32 var_4294;
    wp::float32 var_4295;
    wp::float32 var_4296;
    wp::float32 var_4297;
    wp::float32 var_4298;
    wp::float32 var_4299;
    wp::float32 var_4300;
    wp::float32 var_4301;
    wp::float32 var_4302;
    wp::float32 var_4303;
    wp::float32 var_4304;
    wp::float32 var_4305;
    bool var_4306;
    const bool var_4307 = false;
    bool var_4308;
    const wp::float32 var_4309 = 1.0;
    wp::float32 var_4310;
    wp::float32 var_4311;
    wp::float32 var_4312;
    wp::float32 var_4313;
    wp::float32 var_4314;
    wp::float32 var_4315;
    wp::float32 var_4316;
    wp::float32 var_4317;
    wp::float32 var_4318;
    wp::float32 var_4319;
    wp::float32 var_4320;
    const wp::int32 var_4321 = 1;
    wp::int32 var_4322;
    const wp::int32 var_4323 = 1;
    wp::int32 var_4324;
    const wp::int32 var_4325 = 1;
    bool var_4326;
    const wp::float32 var_4327 = 1.0;
    const wp::int32 var_4328 = 2;
    wp::float32 var_4329;
    wp::float32 var_4330;
    const wp::int32 var_4331 = -1;
    bool var_4332;
    const wp::float32 var_4333 = -1.0;
    const wp::int32 var_4334 = 0;
    wp::float32 var_4335;
    wp::float32 var_4336;
    wp::float32 var_4337;
    wp::float32 var_4338;
    wp::int32 var_4339;
    wp::float32 var_4340;
    wp::float32 var_4341;
    wp::int32 var_4342;
    const wp::float32 var_4343 = 1.0;
    bool var_4344;
    bool var_4345;
    const wp::float32 var_4346 = -1.0;
    bool var_4347;
    const wp::float32 var_4348 = 1.0;
    const wp::int32 var_4349 = 2;
    wp::float32 var_4350;
    wp::float32 var_4351;
    const wp::float32 var_4352 = -1.0;
    const wp::int32 var_4353 = 0;
    wp::float32 var_4354;
    wp::float32 var_4355;
    wp::float32 var_4356;
    wp::float32 var_4357;
    wp::int32 var_4358;
    const wp::int32 var_4359 = 1;
    bool var_4360;
    const wp::float32 var_4361 = 1.0;
    const wp::int32 var_4362 = 2;
    const wp::int32 var_4363 = -1;
    bool var_4364;
    const wp::float32 var_4365 = -1.0;
    const wp::int32 var_4366 = 0;
    wp::float32 var_4367;
    wp::int32 var_4368;
    wp::float32 var_4369;
    wp::int32 var_4370;
    wp::float32 var_4371;
    wp::float32 var_4372;
    wp::int32 var_4373;
    wp::int32 var_4374;
    wp::vec_t<3, wp::float32> var_4375;
    wp::vec_t<3, wp::float32> var_4376;
    wp::float32 var_4377;
    wp::float32 var_4378;
    const wp::int32 var_4379 = 3;
    wp::int32 var_4380;
    wp::int32 var_4381;
    wp::float32 var_4382;
    wp::float32 var_4383;
    bool var_4384;
    wp::float32 var_4385;
    wp::float32 var_4386;
    wp::float32 var_4387;
    const wp::int32 var_4388 = 6;
    wp::int32 var_4389;
    const wp::int32 var_4390 = 1;
    wp::int32 var_4391;
    wp::int32 var_4392;
    wp::int32 var_4393;
    wp::int32 var_4394;
    wp::int32 var_4395;
    wp::float32 var_4396;
    wp::float32 var_4397;
    wp::int32 var_4398;
    wp::int32 var_4399;
    wp::int32 var_4400;
    wp::float32 var_4401;
    wp::int32 var_4402;
    wp::float32 var_4403;
    wp::float32 var_4404;
    wp::int32 var_4405;
    wp::int32 var_4406;
    wp::int32 var_4407;
    wp::float32 var_4408;
    wp::int32 var_4409;
    wp::vec_t<3, wp::float32> var_4410;
    wp::float32 var_4411;
    wp::float32 var_4412;
    wp::float32 var_4413;
    wp::int32 var_4414;
    wp::int32 var_4415;
    bool var_4416;
    wp::int32 var_4417;
    wp::float32 var_4418;
    wp::float32 var_4419;
    wp::float32 var_4420;
    wp::int32 var_4421;
    wp::int32 var_4422;
    wp::int32 var_4423;
    wp::float32 var_4424;
    wp::int32 var_4425;
    wp::vec_t<3, wp::float32> var_4426;
    wp::vec_t<3, wp::float32> var_4427;
    wp::float32 var_4428;
    wp::float32 var_4429;
    wp::float32 var_4430;
    wp::float32 var_4431;
    wp::float32 var_4432;
    wp::float32 var_4433;
    wp::float32 var_4434;
    wp::float32 var_4435;
    wp::float32 var_4436;
    wp::int32 var_4437;
    wp::int32 var_4438;
    bool var_4439;
    wp::int32 var_4440;
    wp::float32 var_4441;
    bool var_4442;
    const wp::int32 var_4443 = 2;
    const bool var_4444 = true;
    const wp::int32 var_4445 = 1;
    wp::int32 var_4446;
    wp::int32 var_4447;
    const wp::int32 var_4448 = 0;
    bool var_4449;
    const bool var_4450 = false;
    bool var_4451;
    const wp::int32 var_4452 = -123;
    wp::int32 var_4453;
    const wp::int32 var_4454 = 1;
    wp::int32 var_4455;
    const wp::float32 var_4456 = 1.0;
    const wp::float32 var_4457 = -1.0;
    wp::float32 var_4458;
    const wp::int32 var_4459 = 2;
    wp::int32 var_4460;
    const wp::float32 var_4461 = 1.0;
    const wp::float32 var_4462 = -1.0;
    wp::float32 var_4463;
    const wp::int32 var_4464 = 4;
    wp::int32 var_4465;
    const wp::float32 var_4466 = 1.0;
    const wp::float32 var_4467 = -1.0;
    wp::float32 var_4468;
    wp::vec_t<3, wp::float32> var_4469;
    wp::vec_t<3, wp::float32> var_4470;
    const wp::float32 var_4471 = 0.0;
    wp::vec_t<3, wp::float32> var_4472;
    wp::float32 var_4473;
    wp::float32 var_4474;
    wp::float32 var_4475;
    wp::float32 var_4476;
    wp::float32 var_4477;
    wp::float32 var_4478;
    wp::float32 var_4479;
    wp::float32 var_4480;
    wp::float32 var_4481;
    wp::float32 var_4482;
    wp::float32 var_4483;
    wp::float32 var_4484;
    wp::float32 var_4485;
    wp::float32 var_4486;
    wp::float32 var_4487;
    wp::float32 var_4488;
    wp::float32 var_4489;
    bool var_4490;
    const bool var_4491 = false;
    bool var_4492;
    const wp::float32 var_4493 = 1.0;
    wp::float32 var_4494;
    wp::float32 var_4495;
    wp::float32 var_4496;
    wp::float32 var_4497;
    wp::float32 var_4498;
    wp::float32 var_4499;
    wp::float32 var_4500;
    wp::float32 var_4501;
    wp::float32 var_4502;
    wp::float32 var_4503;
    wp::float32 var_4504;
    const wp::int32 var_4505 = 1;
    wp::int32 var_4506;
    const wp::int32 var_4507 = 1;
    wp::int32 var_4508;
    const wp::int32 var_4509 = 1;
    bool var_4510;
    const wp::float32 var_4511 = 1.0;
    const wp::int32 var_4512 = 2;
    wp::float32 var_4513;
    wp::float32 var_4514;
    const wp::int32 var_4515 = -1;
    bool var_4516;
    const wp::float32 var_4517 = -1.0;
    const wp::int32 var_4518 = 0;
    wp::float32 var_4519;
    wp::float32 var_4520;
    wp::float32 var_4521;
    wp::float32 var_4522;
    wp::int32 var_4523;
    wp::float32 var_4524;
    wp::float32 var_4525;
    wp::int32 var_4526;
    const wp::float32 var_4527 = 1.0;
    bool var_4528;
    bool var_4529;
    const wp::float32 var_4530 = -1.0;
    bool var_4531;
    const wp::float32 var_4532 = 1.0;
    const wp::int32 var_4533 = 2;
    wp::float32 var_4534;
    wp::float32 var_4535;
    const wp::float32 var_4536 = -1.0;
    const wp::int32 var_4537 = 0;
    wp::float32 var_4538;
    wp::float32 var_4539;
    wp::float32 var_4540;
    wp::float32 var_4541;
    wp::int32 var_4542;
    const wp::int32 var_4543 = 1;
    bool var_4544;
    const wp::float32 var_4545 = 1.0;
    const wp::int32 var_4546 = 2;
    const wp::int32 var_4547 = -1;
    bool var_4548;
    const wp::float32 var_4549 = -1.0;
    const wp::int32 var_4550 = 0;
    wp::float32 var_4551;
    wp::int32 var_4552;
    wp::float32 var_4553;
    wp::int32 var_4554;
    wp::float32 var_4555;
    wp::float32 var_4556;
    wp::int32 var_4557;
    wp::int32 var_4558;
    wp::vec_t<3, wp::float32> var_4559;
    wp::vec_t<3, wp::float32> var_4560;
    wp::float32 var_4561;
    wp::float32 var_4562;
    const wp::int32 var_4563 = 3;
    wp::int32 var_4564;
    wp::int32 var_4565;
    wp::float32 var_4566;
    wp::float32 var_4567;
    bool var_4568;
    wp::float32 var_4569;
    wp::float32 var_4570;
    wp::float32 var_4571;
    const wp::int32 var_4572 = 6;
    wp::int32 var_4573;
    const wp::int32 var_4574 = 1;
    wp::int32 var_4575;
    wp::int32 var_4576;
    wp::int32 var_4577;
    wp::int32 var_4578;
    wp::int32 var_4579;
    wp::float32 var_4580;
    wp::float32 var_4581;
    wp::int32 var_4582;
    wp::int32 var_4583;
    wp::int32 var_4584;
    wp::float32 var_4585;
    wp::int32 var_4586;
    wp::float32 var_4587;
    wp::float32 var_4588;
    wp::int32 var_4589;
    wp::int32 var_4590;
    wp::int32 var_4591;
    wp::float32 var_4592;
    wp::int32 var_4593;
    wp::vec_t<3, wp::float32> var_4594;
    wp::float32 var_4595;
    wp::float32 var_4596;
    wp::float32 var_4597;
    wp::int32 var_4598;
    wp::int32 var_4599;
    bool var_4600;
    wp::int32 var_4601;
    wp::float32 var_4602;
    wp::float32 var_4603;
    wp::float32 var_4604;
    wp::int32 var_4605;
    wp::int32 var_4606;
    wp::int32 var_4607;
    wp::float32 var_4608;
    wp::int32 var_4609;
    wp::vec_t<3, wp::float32> var_4610;
    wp::vec_t<3, wp::float32> var_4611;
    wp::float32 var_4612;
    wp::float32 var_4613;
    wp::float32 var_4614;
    wp::float32 var_4615;
    wp::float32 var_4616;
    wp::float32 var_4617;
    wp::float32 var_4618;
    wp::float32 var_4619;
    wp::float32 var_4620;
    wp::int32 var_4621;
    wp::int32 var_4622;
    bool var_4623;
    wp::int32 var_4624;
    wp::float32 var_4625;
    bool var_4626;
    const wp::float32 var_4627 = 0.0;
    wp::float32 var_4628;
    const wp::int32 var_4629 = 0;
    wp::float32 var_4630;
    const wp::int32 var_4631 = 1;
    wp::float32 var_4632;
    wp::vec_t<2, wp::float32> var_4633;
    const wp::int32 var_4634 = 0;
    wp::float32 var_4635;
    const wp::int32 var_4636 = 1;
    wp::float32 var_4637;
    wp::vec_t<2, wp::float32> var_4638;
    const wp::int32 var_4639 = 0;
    wp::float32 var_4640;
    const wp::int32 var_4641 = 1;
    wp::float32 var_4642;
    wp::vec_t<2, wp::float32> var_4643;
    const wp::float32 var_4644 = -4.0;
    wp::float32 var_4645;
    const wp::int32 var_4646 = 0;
    wp::float32 var_4647;
    const wp::int32 var_4648 = 1;
    wp::float32 var_4649;
    wp::float32 var_4650;
    const wp::int32 var_4651 = 1;
    wp::float32 var_4652;
    const wp::int32 var_4653 = 0;
    wp::float32 var_4654;
    wp::float32 var_4655;
    const wp::int32 var_4656 = 0;
    wp::float32 var_4657;
    const wp::int32 var_4658 = 1;
    wp::float32 var_4659;
    wp::float32 var_4660;
    const wp::int32 var_4661 = 1;
    wp::float32 var_4662;
    const wp::int32 var_4663 = 0;
    wp::float32 var_4664;
    wp::float32 var_4665;
    wp::float32 var_4666;
    const wp::int32 var_4667 = 0;
    bool var_4668;
    const wp::float32 var_4669 = -1.0;
    wp::float32 var_4670;
    wp::float32 var_4671;
    wp::float32 var_4672;
    wp::float32 var_4673;
    bool var_4674;
    wp::float32 var_4675;
    const wp::int32 var_4676 = 0;
    bool var_4677;
    bool var_4678;
    const wp::int32 var_4679 = 0;
    const wp::int32 var_4680 = 3;
    wp::int32 var_4681;
    wp::float32 var_4682;
    wp::float32 var_4683;
    bool var_4684;
    wp::float32 var_4685;
    const wp::int32 var_4686 = 0;
    bool var_4687;
    bool var_4688;
    const wp::int32 var_4689 = 1;
    const wp::int32 var_4690 = 2;
    wp::int32 var_4691;
    wp::float32 var_4692;
    wp::int32 var_4693;
    const wp::int32 var_4694 = -4;
    bool var_4695;
    const wp::float32 var_4696 = 10000000000.0;
    wp::vec_t<2, wp::float32> var_4697;
    wp::mat_t<2, 3, wp::float32> var_4698;
    wp::mat_t<2, 3, wp::float32> var_4699;
    bool var_4700;
    const wp::int32 var_4701 = 0;
    bool var_4702;
    const wp::int32 var_4703 = 3;
    wp::int32 var_4704;
    const wp::int32 var_4705 = 1;
    bool var_4706;
    wp::int32 var_4707;
    bool var_4708;
    const wp::int32 var_4709 = 0;
    bool var_4710;
    const wp::int32 var_4711 = 7;
    bool var_4712;
    bool var_4713;
    const wp::int32 var_4714 = 1;
    bool var_4715;
    const wp::int32 var_4716 = 2;
    bool var_4717;
    const wp::int32 var_4718 = 4;
    bool var_4719;
    const wp::int32 var_4720 = 1;
    const wp::int32 var_4721 = -1;
    const wp::int32 var_4722 = 7;
    wp::int32 var_4723;
    wp::int32 var_4724;
    wp::int32 var_4725;
    const wp::int32 var_4726 = 1;
    bool var_4727;
    const wp::int32 var_4728 = 0;
    const wp::int32 var_4729 = 1;
    const wp::int32 var_4730 = 2;
    const wp::int32 var_4731 = 2;
    bool var_4732;
    const wp::int32 var_4733 = 1;
    const wp::int32 var_4734 = 2;
    const wp::int32 var_4735 = 0;
    const wp::int32 var_4736 = 4;
    bool var_4737;
    const wp::int32 var_4738 = 2;
    const wp::int32 var_4739 = 0;
    const wp::int32 var_4740 = 1;
    wp::int32 var_4741;
    wp::int32 var_4742;
    wp::int32 var_4743;
    wp::int32 var_4744;
    wp::int32 var_4745;
    wp::int32 var_4746;
    wp::float32 var_4747;
    wp::float32 var_4748;
    wp::float32 var_4749;
    const wp::float32 var_4750 = 0.5;
    bool var_4751;
    const wp::float32 var_4752 = 2.0;
    wp::float32 var_4753;
    wp::float32 var_4754;
    wp::float32 var_4755;
    wp::float32 var_4756;
    wp::float32 var_4757;
    const wp::float32 var_4758 = 1.0;
    wp::float32 var_4759;
    wp::float32 var_4760;
    wp::float32 var_4761;
    wp::float32 var_4762;
    const wp::float32 var_4763 = 2.0;
    wp::float32 var_4764;
    wp::float32 var_4765;
    wp::float32 var_4766;
    wp::float32 var_4767;
    wp::float32 var_4768;
    wp::float32 var_4769;
    wp::float32 var_4770;
    wp::float32 var_4771;
    wp::float32 var_4772;
    wp::float32 var_4773;
    const wp::float32 var_4774 = 1.0;
    wp::float32 var_4775;
    wp::float32 var_4776;
    wp::float32 var_4777;
    wp::float32 var_4778;
    wp::float32 var_4779;
    wp::float32 var_4780;
    wp::float32 var_4781;
    wp::float32 var_4782;
    wp::float32 var_4783;
    wp::float32 var_4784;
    wp::int32 var_4785;
    bool var_4786;
    const wp::int32 var_4787 = 0;
    bool var_4788;
    const wp::int32 var_4789 = 3;
    wp::int32 var_4790;
    const wp::int32 var_4791 = 1;
    bool var_4792;
    wp::int32 var_4793;
    const wp::int32 var_4794 = 7;
    const wp::int32 var_4795 = 1;
    wp::int32 var_4796;
    wp::int32 var_4797;
    wp::int32 var_4798;
    bool var_4799;
    const wp::int32 var_4800 = 1;
    bool var_4801;
    const wp::int32 var_4802 = 2;
    bool var_4803;
    const wp::int32 var_4804 = 4;
    bool var_4805;
    const wp::int32 var_4806 = 0;
    bool var_4807;
    const wp::int32 var_4808 = 1;
    const wp::int32 var_4809 = 2;
    const wp::int32 var_4810 = 1;
    bool var_4811;
    const wp::int32 var_4812 = 2;
    const wp::int32 var_4813 = 0;
    wp::int32 var_4814;
    wp::int32 var_4815;
    const wp::int32 var_4816 = 2;
    bool var_4817;
    const wp::int32 var_4818 = 0;
    const wp::int32 var_4819 = 1;
    wp::int32 var_4820;
    wp::int32 var_4821;
    wp::int32 var_4822;
    wp::float32 var_4823;
    wp::float32 var_4824;
    wp::float32 var_4825;
    wp::float32 var_4826;
    bool var_4827;
    wp::int32 var_4828;
    wp::int32 var_4829;
    const wp::int32 var_4830 = 3;
    wp::int32 var_4831;
    wp::int32 var_4832;
    const wp::int32 var_4833 = 1;
    wp::int32 var_4834;
    wp::int32 var_4835;
    const wp::int32 var_4836 = 1;
    const wp::float32 var_4837 = 1.0;
    wp::float32 var_4838;
    const wp::int32 var_4839 = -1;
    const wp::float32 var_4840 = 1.0;
    wp::float32 var_4841;
    wp::float32 var_4842;
    wp::int32 var_4843;
    const wp::float32 var_4844 = 2.0;
    wp::float32 var_4845;
    wp::float32 var_4846;
    wp::float32 var_4847;
    wp::float32 var_4848;
    wp::float32 var_4849;
    wp::float32 var_4850;
    const wp::int32 var_4851 = 1;
    wp::int32 var_4852;
    wp::int32 var_4853;
    const wp::int32 var_4854 = 0;
    bool var_4855;
    const wp::int32 var_4856 = 1;
    wp::int32 var_4857;
    wp::int32 var_4858;
    const wp::int32 var_4859 = 0;
    bool var_4860;
    bool var_4861;
    const wp::float32 var_4862 = 1.0;
    wp::float32 var_4863;
    const wp::float32 var_4864 = 1.0;
    wp::float32 var_4865;
    wp::float32 var_4866;
    wp::float32 var_4867;
    wp::float32 var_4868;
    wp::float32 var_4869;
    wp::float32 var_4870;
    wp::float32 var_4871;
    wp::float32 var_4872;
    wp::float32 var_4873;
    wp::float32 var_4874;
    wp::float32 var_4875;
    const wp::int32 var_4876 = 0;
    bool var_4877;
    const wp::int32 var_4878 = -1;
    bool var_4879;
    const wp::int32 var_4880 = -3;
    bool var_4881;
    const wp::int32 var_4882 = 1;
    const wp::int32 var_4883 = -1;
    wp::int32 var_4884;
    const wp::float32 var_4885 = 2.0;
    wp::float32 var_4886;
    wp::vec_t<3, wp::float32> var_4887;
    wp::vec_t<3, wp::float32> var_4888;
    const wp::int32 var_4889 = 0;
    bool var_4890;
    wp::float32 var_4891;
    wp::float32 var_4892;
    wp::float32 var_4893;
    wp::float32 var_4894;
    wp::float32 var_4895;
    wp::float32 var_4896;
    wp::float32 var_4897;
    bool var_4898;
    const wp::int32 var_4899 = 0;
    bool var_4900;
    bool var_4901;
    wp::float32 var_4902;
    wp::float32 var_4903;
    wp::float32 var_4904;
    wp::float32 var_4905;
    wp::float32 var_4906;
    wp::float32 var_4907;
    wp::float32 var_4908;
    bool var_4909;
    const wp::int32 var_4910 = 0;
    bool var_4911;
    bool var_4912;
    wp::float32 var_4913;
    wp::float32 var_4914;
    wp::float32 var_4915;
    const wp::int32 var_4916 = 1;
    bool var_4917;
    wp::float32 var_4918;
    wp::float32 var_4919;
    wp::float32 var_4920;
    wp::float32 var_4921;
    wp::float32 var_4922;
    wp::float32 var_4923;
    wp::float32 var_4924;
    bool var_4925;
    const wp::int32 var_4926 = 0;
    bool var_4927;
    bool var_4928;
    wp::float32 var_4929;
    wp::float32 var_4930;
    wp::float32 var_4931;
    wp::float32 var_4932;
    wp::float32 var_4933;
    wp::float32 var_4934;
    wp::float32 var_4935;
    bool var_4936;
    const wp::int32 var_4937 = 0;
    bool var_4938;
    bool var_4939;
    wp::float32 var_4940;
    wp::float32 var_4941;
    wp::float32 var_4942;
    wp::float32 var_4943;
    wp::float32 var_4944;
    const wp::int32 var_4945 = 2;
    bool var_4946;
    wp::float32 var_4947;
    wp::float32 var_4948;
    wp::float32 var_4949;
    wp::float32 var_4950;
    wp::float32 var_4951;
    wp::float32 var_4952;
    wp::float32 var_4953;
    bool var_4954;
    const wp::int32 var_4955 = 0;
    bool var_4956;
    bool var_4957;
    wp::float32 var_4958;
    wp::float32 var_4959;
    wp::float32 var_4960;
    wp::float32 var_4961;
    wp::float32 var_4962;
    wp::float32 var_4963;
    wp::float32 var_4964;
    bool var_4965;
    const wp::int32 var_4966 = 0;
    bool var_4967;
    bool var_4968;
    wp::float32 var_4969;
    wp::float32 var_4970;
    wp::float32 var_4971;
    wp::float32 var_4972;
    wp::float32 var_4973;
    wp::float32 var_4974;
    wp::float32 var_4975;
    wp::int32 var_4976;
    wp::float32 var_4977;
    wp::int32 var_4978;
    wp::float32 var_4979;
    wp::int32 var_4980;
    wp::float32 var_4981;
    wp::int32 var_4982;
    wp::int32 var_4983;
    wp::float32 var_4984;
    wp::int32 var_4985;
    wp::float32 var_4986;
    wp::int32 var_4987;
    wp::int32 var_4988;
    wp::int32 var_4989;
    wp::int32 var_4990;
    wp::int32 var_4991;
    wp::vec_t<3, wp::float32> var_4992;
    wp::vec_t<3, wp::float32> var_4993;
    wp::vec_t<3, wp::float32> var_4994;
    wp::vec_t<3, wp::float32> var_4995;
    wp::float32 var_4996;
    wp::vec_t<3, wp::float32> var_4997;
    wp::vec_t<3, wp::float32> var_4998;
    const wp::int32 var_4999 = -3;
    bool var_5000;
    wp::float32 var_5001;
    wp::vec_t<3, wp::float32> var_5002;
    wp::vec_t<3, wp::float32> var_5003;
    wp::vec_t<3, wp::float32> var_5004;
    wp::vec_t<3, wp::float32> var_5005;
    wp::float32 var_5006;
    wp::vec_t<3, wp::float32> var_5007;
    wp::vec_t<3, wp::float32> var_5008;
    wp::float32 var_5009;
    wp::vec_t<3, wp::float32> var_5010;
    wp::vec_t<3, wp::float32> var_5011;
    wp::float32 var_5012;
    wp::vec_t<3, wp::float32> var_5013;
    wp::vec_t<3, wp::float32> var_5014;
    wp::vec_t<2, wp::float32> var_5015;
    const wp::int32 var_5016 = 0;
    wp::float32 var_5017;
    const wp::int32 var_5018 = 1;
    wp::float32 var_5019;
    const wp::int32 var_5020 = 2;
    wp::float32 var_5021;
    const wp::int32 var_5022 = 0;
    wp::float32 var_5023;
    const wp::int32 var_5024 = 1;
    wp::float32 var_5025;
    const wp::int32 var_5026 = 2;
    wp::float32 var_5027;
    wp::mat_t<2, 3, wp::float32> var_5028;
    const wp::int32 var_5029 = 0;
    wp::float32 var_5030;
    const wp::int32 var_5031 = 1;
    wp::float32 var_5032;
    const wp::int32 var_5033 = 2;
    wp::float32 var_5034;
    const wp::int32 var_5035 = 0;
    wp::float32 var_5036;
    const wp::int32 var_5037 = 1;
    wp::float32 var_5038;
    const wp::int32 var_5039 = 2;
    wp::float32 var_5040;
    wp::mat_t<2, 3, wp::float32> var_5041;
    //---------
    // forward
    // def capsule_box(                                                                       <L 1105>
    // boxmatT = wp.transpose(box_rot)                                                        <L 1132>
    var_0 = wp::transpose(var_box_rot);
    // pos = boxmatT @ (capsule_pos - box_pos)                                                <L 1133>
    var_1 = wp::sub(var_capsule_pos, var_box_pos);
    var_2 = wp::mul(var_0, var_1);
    // axis = boxmatT @ capsule_axis                                                          <L 1134>
    var_3 = wp::mul(var_0, var_capsule_axis);
    // halfaxis = axis * capsule_half_length  # halfaxis is the capsule direction             <L 1135>
    var_4 = wp::mul(var_3, var_capsule_half_length);
    // axisdir = wp.int32(halfaxis[0] > 0.0) + 2 * wp.int32(halfaxis[1] > 0.0) + 4 * wp.int32(halfaxis[2] > 0.0)       <L 1136>
    var_6 = wp::extract(var_4, var_5);
    var_8 = (var_6 > var_7);
    var_9 = wp::int32(var_8);
    var_12 = wp::extract(var_4, var_11);
    var_14 = (var_12 > var_13);
    var_15 = wp::int32(var_14);
    var_16 = wp::mul(var_10, var_15);
    var_17 = wp::add(var_9, var_16);
    var_20 = wp::extract(var_4, var_19);
    var_22 = (var_20 > var_21);
    var_23 = wp::int32(var_22);
    var_24 = wp::mul(var_18, var_23);
    var_25 = wp::add(var_17, var_24);
    // bestdist = wp.float32(1.0e32)                                                          <L 1139>
    var_27 = wp::float32(var_26);
    // bestsegmentpos = wp.float32(-12)                                                       <L 1140>
    var_29 = wp::float32(var_28);
    // cltype = wp.int32(-4)                                                                  <L 1149>
    var_31 = wp::int32(var_30);
    // clface = wp.int32(-12)                                                                 <L 1154>
    var_33 = wp::int32(var_32);
    // for i in range(-1, 2, 2):                                                              <L 1157>
    // axisTip = pos + wp.float32(i) * halfaxis                                               <L 1158>
    var_36 = wp::float32(var_34);
    var_37 = wp::mul(var_36, var_4);
    var_38 = wp::add(var_2, var_37);
    // boxPoint = wp.vec3(axisTip)                                                            <L 1159>
    var_39 = wp::vec_t<3, wp::float32>(var_38);
    // n_out = wp.int32(0)                                                                    <L 1161>
    var_41 = wp::int32(var_40);
    // ax_out = wp.int32(-1)                                                                  <L 1162>
    var_43 = wp::int32(var_42);
    // for j in range(3):                                                                     <L 1164>
    // if boxPoint[j] < -box_size[j]:                                                         <L 1165>
    var_45 = wp::extract(var_39, var_44);
    var_46 = wp::extract(var_box_size, var_44);
    var_47 = wp::neg(var_46);
    var_48 = (var_45 < var_47);
    if (var_48) {
        // n_out += 1                                                                         <L 1166>
        var_50 = wp::add(var_41, var_49);
        // ax_out = j                                                                         <L 1167>
        var_51 = wp::copy(var_44);
        // boxPoint[j] = -box_size[j]                                                         <L 1168>
        var_52 = wp::extract(var_box_size, var_44);
        var_53 = wp::neg(var_52);
        wp::assign_inplace(var_39, var_44, var_53);
    }
    if (!var_48) {
        // elif boxPoint[j] > box_size[j]:                                                    <L 1169>
        var_54 = wp::extract(var_39, var_44);
        var_55 = wp::extract(var_box_size, var_44);
        var_56 = (var_54 > var_55);
        if (var_56) {
            // n_out += 1                                                                     <L 1170>
            var_58 = wp::add(var_41, var_57);
            // ax_out = j                                                                     <L 1171>
            var_59 = wp::copy(var_44);
            // boxPoint[j] = box_size[j]                                                      <L 1172>
            var_60 = wp::extract(var_box_size, var_44);
            wp::assign_inplace(var_39, var_44, var_60);
        }
        var_61 = wp::where(var_56, var_58, var_41);
        var_62 = wp::where(var_56, var_59, var_43);
    }
    var_63 = wp::where(var_48, var_50, var_61);
    var_64 = wp::where(var_48, var_51, var_62);
    // if boxPoint[j] < -box_size[j]:                                                         <L 1165>
    var_66 = wp::extract(var_39, var_65);
    var_67 = wp::extract(var_box_size, var_65);
    var_68 = wp::neg(var_67);
    var_69 = (var_66 < var_68);
    if (var_69) {
        // n_out += 1                                                                         <L 1166>
        var_71 = wp::add(var_63, var_70);
        // ax_out = j                                                                         <L 1167>
        var_72 = wp::copy(var_65);
        // boxPoint[j] = -box_size[j]                                                         <L 1168>
        var_73 = wp::extract(var_box_size, var_65);
        var_74 = wp::neg(var_73);
        wp::assign_inplace(var_39, var_65, var_74);
    }
    if (!var_69) {
        // elif boxPoint[j] > box_size[j]:                                                    <L 1169>
        var_75 = wp::extract(var_39, var_65);
        var_76 = wp::extract(var_box_size, var_65);
        var_77 = (var_75 > var_76);
        if (var_77) {
            // n_out += 1                                                                     <L 1170>
            var_79 = wp::add(var_63, var_78);
            // ax_out = j                                                                     <L 1171>
            var_80 = wp::copy(var_65);
            // boxPoint[j] = box_size[j]                                                      <L 1172>
            var_81 = wp::extract(var_box_size, var_65);
            wp::assign_inplace(var_39, var_65, var_81);
        }
        var_82 = wp::where(var_77, var_79, var_63);
        var_83 = wp::where(var_77, var_80, var_64);
    }
    var_84 = wp::where(var_69, var_71, var_82);
    var_85 = wp::where(var_69, var_72, var_83);
    // if boxPoint[j] < -box_size[j]:                                                         <L 1165>
    var_87 = wp::extract(var_39, var_86);
    var_88 = wp::extract(var_box_size, var_86);
    var_89 = wp::neg(var_88);
    var_90 = (var_87 < var_89);
    if (var_90) {
        // n_out += 1                                                                         <L 1166>
        var_92 = wp::add(var_84, var_91);
        // ax_out = j                                                                         <L 1167>
        var_93 = wp::copy(var_86);
        // boxPoint[j] = -box_size[j]                                                         <L 1168>
        var_94 = wp::extract(var_box_size, var_86);
        var_95 = wp::neg(var_94);
        wp::assign_inplace(var_39, var_86, var_95);
    }
    if (!var_90) {
        // elif boxPoint[j] > box_size[j]:                                                    <L 1169>
        var_96 = wp::extract(var_39, var_86);
        var_97 = wp::extract(var_box_size, var_86);
        var_98 = (var_96 > var_97);
        if (var_98) {
            // n_out += 1                                                                     <L 1170>
            var_100 = wp::add(var_84, var_99);
            // ax_out = j                                                                     <L 1171>
            var_101 = wp::copy(var_86);
            // boxPoint[j] = box_size[j]                                                      <L 1172>
            var_102 = wp::extract(var_box_size, var_86);
            wp::assign_inplace(var_39, var_86, var_102);
        }
        var_103 = wp::where(var_98, var_100, var_84);
        var_104 = wp::where(var_98, var_101, var_85);
    }
    var_105 = wp::where(var_90, var_92, var_103);
    var_106 = wp::where(var_90, var_93, var_104);
    // if n_out > 1:                                                                          <L 1174>
    var_108 = (var_105 > var_107);
    if (var_108) {
        // continue                                                                           <L 1175>
    }
    var_110 = wp::where(var_108, var_109, var_35);
    // dist = wp.length_sq(boxPoint - axisTip)                                                <L 1177>
    if (var_110) {
        var_111 = wp::sub(var_39, var_38);
        var_112 = wp::length_sq(var_111);
        // if dist < bestdist:                                                                <L 1179>
        var_113 = (var_112 < var_27);
        if (var_113) {
            // bestdist = dist                                                                <L 1180>
            var_114 = wp::copy(var_112);
            // bestsegmentpos = wp.float32(i)                                                 <L 1181>
            var_115 = wp::float32(var_34);
            // cltype = -2 + i                                                                <L 1182>
            var_117 = wp::add(var_116, var_34);
            // clface = ax_out                                                                <L 1183>
            var_118 = wp::copy(var_106);
        }
        var_119 = wp::where(var_113, var_114, var_27);
        var_120 = wp::where(var_113, var_115, var_29);
        var_121 = wp::where(var_113, var_117, var_31);
        var_122 = wp::where(var_113, var_118, var_33);
    }
    var_123 = wp::where(var_110, var_119, var_27);
    var_124 = wp::where(var_110, var_120, var_29);
    var_125 = wp::where(var_110, var_121, var_31);
    var_126 = wp::where(var_110, var_122, var_33);
    // axisTip = pos + wp.float32(i) * halfaxis                                               <L 1158>
    var_129 = wp::float32(var_127);
    var_130 = wp::mul(var_129, var_4);
    var_131 = wp::add(var_2, var_130);
    // boxPoint = wp.vec3(axisTip)                                                            <L 1159>
    var_132 = wp::vec_t<3, wp::float32>(var_131);
    // n_out = wp.int32(0)                                                                    <L 1161>
    var_134 = wp::int32(var_133);
    // ax_out = wp.int32(-1)                                                                  <L 1162>
    var_136 = wp::int32(var_135);
    // for j in range(3):                                                                     <L 1164>
    // if boxPoint[j] < -box_size[j]:                                                         <L 1165>
    var_138 = wp::extract(var_132, var_137);
    var_139 = wp::extract(var_box_size, var_137);
    var_140 = wp::neg(var_139);
    var_141 = (var_138 < var_140);
    if (var_141) {
        // n_out += 1                                                                         <L 1166>
        var_143 = wp::add(var_134, var_142);
        // ax_out = j                                                                         <L 1167>
        var_144 = wp::copy(var_137);
        // boxPoint[j] = -box_size[j]                                                         <L 1168>
        var_145 = wp::extract(var_box_size, var_137);
        var_146 = wp::neg(var_145);
        wp::assign_inplace(var_132, var_137, var_146);
    }
    if (!var_141) {
        // elif boxPoint[j] > box_size[j]:                                                    <L 1169>
        var_147 = wp::extract(var_132, var_137);
        var_148 = wp::extract(var_box_size, var_137);
        var_149 = (var_147 > var_148);
        if (var_149) {
            // n_out += 1                                                                     <L 1170>
            var_151 = wp::add(var_134, var_150);
            // ax_out = j                                                                     <L 1171>
            var_152 = wp::copy(var_137);
            // boxPoint[j] = box_size[j]                                                      <L 1172>
            var_153 = wp::extract(var_box_size, var_137);
            wp::assign_inplace(var_132, var_137, var_153);
        }
        var_154 = wp::where(var_149, var_151, var_134);
        var_155 = wp::where(var_149, var_152, var_136);
    }
    var_156 = wp::where(var_141, var_143, var_154);
    var_157 = wp::where(var_141, var_144, var_155);
    // if boxPoint[j] < -box_size[j]:                                                         <L 1165>
    var_159 = wp::extract(var_132, var_158);
    var_160 = wp::extract(var_box_size, var_158);
    var_161 = wp::neg(var_160);
    var_162 = (var_159 < var_161);
    if (var_162) {
        // n_out += 1                                                                         <L 1166>
        var_164 = wp::add(var_156, var_163);
        // ax_out = j                                                                         <L 1167>
        var_165 = wp::copy(var_158);
        // boxPoint[j] = -box_size[j]                                                         <L 1168>
        var_166 = wp::extract(var_box_size, var_158);
        var_167 = wp::neg(var_166);
        wp::assign_inplace(var_132, var_158, var_167);
    }
    if (!var_162) {
        // elif boxPoint[j] > box_size[j]:                                                    <L 1169>
        var_168 = wp::extract(var_132, var_158);
        var_169 = wp::extract(var_box_size, var_158);
        var_170 = (var_168 > var_169);
        if (var_170) {
            // n_out += 1                                                                     <L 1170>
            var_172 = wp::add(var_156, var_171);
            // ax_out = j                                                                     <L 1171>
            var_173 = wp::copy(var_158);
            // boxPoint[j] = box_size[j]                                                      <L 1172>
            var_174 = wp::extract(var_box_size, var_158);
            wp::assign_inplace(var_132, var_158, var_174);
        }
        var_175 = wp::where(var_170, var_172, var_156);
        var_176 = wp::where(var_170, var_173, var_157);
    }
    var_177 = wp::where(var_162, var_164, var_175);
    var_178 = wp::where(var_162, var_165, var_176);
    // if boxPoint[j] < -box_size[j]:                                                         <L 1165>
    var_180 = wp::extract(var_132, var_179);
    var_181 = wp::extract(var_box_size, var_179);
    var_182 = wp::neg(var_181);
    var_183 = (var_180 < var_182);
    if (var_183) {
        // n_out += 1                                                                         <L 1166>
        var_185 = wp::add(var_177, var_184);
        // ax_out = j                                                                         <L 1167>
        var_186 = wp::copy(var_179);
        // boxPoint[j] = -box_size[j]                                                         <L 1168>
        var_187 = wp::extract(var_box_size, var_179);
        var_188 = wp::neg(var_187);
        wp::assign_inplace(var_132, var_179, var_188);
    }
    if (!var_183) {
        // elif boxPoint[j] > box_size[j]:                                                    <L 1169>
        var_189 = wp::extract(var_132, var_179);
        var_190 = wp::extract(var_box_size, var_179);
        var_191 = (var_189 > var_190);
        if (var_191) {
            // n_out += 1                                                                     <L 1170>
            var_193 = wp::add(var_177, var_192);
            // ax_out = j                                                                     <L 1171>
            var_194 = wp::copy(var_179);
            // boxPoint[j] = box_size[j]                                                      <L 1172>
            var_195 = wp::extract(var_box_size, var_179);
            wp::assign_inplace(var_132, var_179, var_195);
        }
        var_196 = wp::where(var_191, var_193, var_177);
        var_197 = wp::where(var_191, var_194, var_178);
    }
    var_198 = wp::where(var_183, var_185, var_196);
    var_199 = wp::where(var_183, var_186, var_197);
    // if n_out > 1:                                                                          <L 1174>
    var_201 = (var_198 > var_200);
    if (var_201) {
        // continue                                                                           <L 1175>
    }
    var_203 = wp::where(var_201, var_202, var_128);
    // dist = wp.length_sq(boxPoint - axisTip)                                                <L 1177>
    if (var_203) {
        var_204 = wp::sub(var_132, var_131);
        var_205 = wp::length_sq(var_204);
        // if dist < bestdist:                                                                <L 1179>
        var_206 = (var_205 < var_123);
        if (var_206) {
            // bestdist = dist                                                                <L 1180>
            var_207 = wp::copy(var_205);
            // bestsegmentpos = wp.float32(i)                                                 <L 1181>
            var_208 = wp::float32(var_127);
            // cltype = -2 + i                                                                <L 1182>
            var_210 = wp::add(var_209, var_127);
            // clface = ax_out                                                                <L 1183>
            var_211 = wp::copy(var_199);
        }
        var_212 = wp::where(var_206, var_207, var_123);
        var_213 = wp::where(var_206, var_208, var_124);
        var_214 = wp::where(var_206, var_210, var_125);
        var_215 = wp::where(var_206, var_211, var_126);
    }
    var_216 = wp::where(var_203, var_212, var_123);
    var_217 = wp::where(var_203, var_213, var_124);
    var_218 = wp::where(var_203, var_214, var_125);
    var_219 = wp::where(var_203, var_215, var_126);
    var_220 = wp::where(var_203, var_205, var_112);
    // clcorner = wp.int32(-123)  # which corner is the closest                               <L 1186>
    var_222 = wp::int32(var_221);
    // cledge = wp.int32(-123)  # which axis                                                  <L 1187>
    var_224 = wp::int32(var_223);
    // bestboxpos = wp.float32(0.0)                                                           <L 1188>
    var_226 = wp::float32(var_225);
    // for i in range(8):                                                                     <L 1190>
    // for j in range(3):                                                                     <L 1191>
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_231 = wp::lshift(var_230, var_228);
    var_232 = wp::bit_and(var_227, var_231);
    var_234 = (var_232 != var_233);
    if (var_234) {
        // continue                                                                           <L 1193>
    }
    var_236 = wp::where(var_234, var_235, var_229);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_236) {
        var_238 = wp::int32(var_237);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_240 = wp::bit_and(var_227, var_239);
        var_243 = wp::where(var_240, var_241, var_242);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_245 = wp::bit_and(var_227, var_244);
        var_248 = wp::where(var_245, var_246, var_247);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_250 = wp::bit_and(var_227, var_249);
        var_253 = wp::where(var_250, var_251, var_252);
        var_254 = wp::vec_t<3, wp::float32>(var_243, var_248, var_253);
        // box_size,                                                                          <L 1204>
        var_255 = wp::cw_mul(var_254, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_255, var_228, var_256);
        // dif = box_pt - pos                                                                 <L 1209>
        var_257 = wp::sub(var_255, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_258 = wp::extract(var_box_size, var_228);
        var_259 = wp::neg(var_258);
        var_260 = wp::extract(var_257, var_228);
        var_261 = wp::mul(var_259, var_260);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_262 = wp::dot(var_4, var_257);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_263 = wp::extract(var_box_size, var_228);
        var_264 = wp::extract(var_box_size, var_228);
        var_265 = wp::mul(var_263, var_264);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_266 = wp::extract(var_box_size, var_228);
        var_267 = wp::neg(var_266);
        var_268 = wp::extract(var_4, var_228);
        var_269 = wp::mul(var_267, var_268);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_270 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_271 = wp::mul(var_265, var_270);
        var_272 = wp::mul(var_269, var_269);
        var_273 = wp::sub(var_271, var_272);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_274 = wp::abs(var_273);
        var_276 = (var_274 < var_275);
        if (var_276) {
            // continue                                                                       <L 1218>
        }
        var_278 = wp::where(var_276, var_277, var_236);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_278) {
            var_280 = wp::div(var_279, var_273);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_281 = wp::mul(var_270, var_261);
            var_282 = wp::mul(var_269, var_262);
            var_283 = wp::sub(var_281, var_282);
            var_284 = wp::mul(var_283, var_280);
            var_285 = wp::float32(var_284);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_286 = wp::mul(var_265, var_262);
            var_287 = wp::mul(var_269, var_261);
            var_288 = wp::sub(var_286, var_287);
            var_289 = wp::mul(var_288, var_280);
            var_290 = wp::float32(var_289);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_292 = wp::int32(var_291);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_294 = wp::int32(var_293);
            // if x1 > 1:                                                                     <L 1229>
            var_296 = (var_285 > var_295);
            if (var_296) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_299 = wp::sub(var_262, var_269);
                var_300 = safe_div_0(var_299, var_270);
            }
            if (!var_296) {
                // elif x1 < -1:                                                              <L 1233>
                var_302 = (var_285 < var_301);
                if (var_302) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_305 = wp::add(var_262, var_269);
                    var_306 = safe_div_0(var_305, var_270);
                }
                var_307 = wp::where(var_302, var_303, var_285);
                var_308 = wp::where(var_302, var_306, var_290);
                var_309 = wp::where(var_302, var_304, var_292);
            }
            var_310 = wp::where(var_296, var_297, var_307);
            var_311 = wp::where(var_296, var_300, var_308);
            var_312 = wp::where(var_296, var_298, var_309);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_314 = (var_311 > var_313);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_315 = var_314;
            if (!var_315) {
                var_317 = (var_311 < var_316);
                var_315 = var_315 || var_317;
            }
            if (var_315) {
                // if x2_over:                                                                <L 1240>
                if (var_314) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_320 = wp::sub(var_261, var_269);
                    var_321 = safe_div_0(var_320, var_265);
                }
                if (!var_314) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_324 = wp::add(var_261, var_269);
                    var_325 = safe_div_0(var_324, var_265);
                }
                var_326 = wp::where(var_314, var_321, var_325);
                var_327 = wp::where(var_314, var_318, var_322);
                var_328 = wp::where(var_314, var_319, var_323);
                // if x1 > 1:                                                                 <L 1249>
                var_330 = (var_326 > var_329);
                if (var_330) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_330) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_334 = (var_326 < var_333);
                    if (var_334) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_337 = wp::where(var_334, var_335, var_326);
                    var_338 = wp::where(var_334, var_336, var_312);
                }
                var_339 = wp::where(var_330, var_331, var_337);
                var_340 = wp::where(var_330, var_332, var_338);
            }
            var_341 = wp::where(var_315, var_339, var_310);
            var_342 = wp::where(var_315, var_327, var_311);
            var_343 = wp::where(var_315, var_340, var_312);
            var_344 = wp::where(var_315, var_328, var_294);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_345 = wp::mul(var_4, var_342);
            var_346 = wp::sub(var_257, var_345);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_347 = wp::extract(var_box_size, var_228);
            var_348 = wp::mul(var_347, var_341);
            wp::add_inplace(var_346, var_228, var_348);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_350 = wp::mul(var_343, var_349);
            var_351 = wp::add(var_350, var_344);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_352 = wp::length_sq(var_346);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_353 = wp::sub(var_216, var_275);
            var_354 = (var_352 < var_353);
            if (var_354) {
                // bestdist = dif_sq                                                          <L 1264>
                var_355 = wp::copy(var_352);
                // bestsegmentpos = x2                                                        <L 1265>
                var_356 = wp::copy(var_342);
                // bestboxpos = x1                                                            <L 1266>
                var_357 = wp::copy(var_341);
                // c2 = ct // 6                                                               <L 1268>
                var_359 = wp::floordiv(var_351, var_358);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_361 = wp::lshift(var_360, var_228);
                var_362 = wp::mul(var_361, var_359);
                var_363 = wp::add(var_227, var_362);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_364 = wp::copy(var_228);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_365 = wp::copy(var_351);
            }
            var_366 = wp::where(var_354, var_355, var_216);
            var_367 = wp::where(var_354, var_356, var_217);
            var_368 = wp::where(var_354, var_365, var_218);
            var_369 = wp::where(var_354, var_363, var_222);
            var_370 = wp::where(var_354, var_364, var_224);
            var_371 = wp::where(var_354, var_357, var_226);
            var_372 = wp::where(var_354, var_359, var_238);
        }
        var_373 = wp::where(var_278, var_366, var_216);
        var_374 = wp::where(var_278, var_367, var_217);
        var_375 = wp::where(var_278, var_368, var_218);
        var_376 = wp::where(var_278, var_369, var_222);
        var_377 = wp::where(var_278, var_370, var_224);
        var_378 = wp::where(var_278, var_371, var_226);
        var_379 = wp::where(var_278, var_372, var_238);
        var_380 = wp::where(var_278, var_346, var_257);
    }
    var_381 = wp::where(var_236, var_373, var_216);
    var_382 = wp::where(var_236, var_374, var_217);
    var_383 = wp::where(var_236, var_375, var_218);
    var_384 = wp::where(var_236, var_376, var_222);
    var_385 = wp::where(var_236, var_377, var_224);
    var_386 = wp::where(var_236, var_378, var_226);
    var_387 = wp::where(var_236, var_278, var_236);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_391 = wp::lshift(var_390, var_388);
    var_392 = wp::bit_and(var_227, var_391);
    var_394 = (var_392 != var_393);
    if (var_394) {
        // continue                                                                           <L 1193>
    }
    var_396 = wp::where(var_394, var_395, var_389);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_396) {
        var_398 = wp::int32(var_397);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_400 = wp::bit_and(var_227, var_399);
        var_403 = wp::where(var_400, var_401, var_402);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_405 = wp::bit_and(var_227, var_404);
        var_408 = wp::where(var_405, var_406, var_407);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_410 = wp::bit_and(var_227, var_409);
        var_413 = wp::where(var_410, var_411, var_412);
        var_414 = wp::vec_t<3, wp::float32>(var_403, var_408, var_413);
        // box_size,                                                                          <L 1204>
        var_415 = wp::cw_mul(var_414, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_415, var_388, var_416);
        // dif = box_pt - pos                                                                 <L 1209>
        var_417 = wp::sub(var_415, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_418 = wp::extract(var_box_size, var_388);
        var_419 = wp::neg(var_418);
        var_420 = wp::extract(var_417, var_388);
        var_421 = wp::mul(var_419, var_420);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_422 = wp::dot(var_4, var_417);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_423 = wp::extract(var_box_size, var_388);
        var_424 = wp::extract(var_box_size, var_388);
        var_425 = wp::mul(var_423, var_424);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_426 = wp::extract(var_box_size, var_388);
        var_427 = wp::neg(var_426);
        var_428 = wp::extract(var_4, var_388);
        var_429 = wp::mul(var_427, var_428);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_430 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_431 = wp::mul(var_425, var_430);
        var_432 = wp::mul(var_429, var_429);
        var_433 = wp::sub(var_431, var_432);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_434 = wp::abs(var_433);
        var_435 = (var_434 < var_275);
        if (var_435) {
            // continue                                                                       <L 1218>
        }
        var_437 = wp::where(var_435, var_436, var_396);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_437) {
            var_439 = wp::div(var_438, var_433);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_440 = wp::mul(var_430, var_421);
            var_441 = wp::mul(var_429, var_422);
            var_442 = wp::sub(var_440, var_441);
            var_443 = wp::mul(var_442, var_439);
            var_444 = wp::float32(var_443);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_445 = wp::mul(var_425, var_422);
            var_446 = wp::mul(var_429, var_421);
            var_447 = wp::sub(var_445, var_446);
            var_448 = wp::mul(var_447, var_439);
            var_449 = wp::float32(var_448);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_451 = wp::int32(var_450);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_453 = wp::int32(var_452);
            // if x1 > 1:                                                                     <L 1229>
            var_455 = (var_444 > var_454);
            if (var_455) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_458 = wp::sub(var_422, var_429);
                var_459 = safe_div_0(var_458, var_430);
            }
            if (!var_455) {
                // elif x1 < -1:                                                              <L 1233>
                var_461 = (var_444 < var_460);
                if (var_461) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_464 = wp::add(var_422, var_429);
                    var_465 = safe_div_0(var_464, var_430);
                }
                var_466 = wp::where(var_461, var_462, var_444);
                var_467 = wp::where(var_461, var_465, var_449);
                var_468 = wp::where(var_461, var_463, var_451);
            }
            var_469 = wp::where(var_455, var_456, var_466);
            var_470 = wp::where(var_455, var_459, var_467);
            var_471 = wp::where(var_455, var_457, var_468);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_473 = (var_470 > var_472);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_474 = var_473;
            if (!var_474) {
                var_476 = (var_470 < var_475);
                var_474 = var_474 || var_476;
            }
            if (var_474) {
                // if x2_over:                                                                <L 1240>
                if (var_473) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_479 = wp::sub(var_421, var_429);
                    var_480 = safe_div_0(var_479, var_425);
                }
                if (!var_473) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_483 = wp::add(var_421, var_429);
                    var_484 = safe_div_0(var_483, var_425);
                }
                var_485 = wp::where(var_473, var_480, var_484);
                var_486 = wp::where(var_473, var_477, var_481);
                var_487 = wp::where(var_473, var_478, var_482);
                // if x1 > 1:                                                                 <L 1249>
                var_489 = (var_485 > var_488);
                if (var_489) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_489) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_493 = (var_485 < var_492);
                    if (var_493) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_496 = wp::where(var_493, var_494, var_485);
                    var_497 = wp::where(var_493, var_495, var_471);
                }
                var_498 = wp::where(var_489, var_490, var_496);
                var_499 = wp::where(var_489, var_491, var_497);
            }
            var_500 = wp::where(var_474, var_498, var_469);
            var_501 = wp::where(var_474, var_486, var_470);
            var_502 = wp::where(var_474, var_499, var_471);
            var_503 = wp::where(var_474, var_487, var_453);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_504 = wp::mul(var_4, var_501);
            var_505 = wp::sub(var_417, var_504);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_506 = wp::extract(var_box_size, var_388);
            var_507 = wp::mul(var_506, var_500);
            wp::add_inplace(var_505, var_388, var_507);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_509 = wp::mul(var_502, var_508);
            var_510 = wp::add(var_509, var_503);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_511 = wp::length_sq(var_505);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_512 = wp::sub(var_381, var_275);
            var_513 = (var_511 < var_512);
            if (var_513) {
                // bestdist = dif_sq                                                          <L 1264>
                var_514 = wp::copy(var_511);
                // bestsegmentpos = x2                                                        <L 1265>
                var_515 = wp::copy(var_501);
                // bestboxpos = x1                                                            <L 1266>
                var_516 = wp::copy(var_500);
                // c2 = ct // 6                                                               <L 1268>
                var_518 = wp::floordiv(var_510, var_517);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_520 = wp::lshift(var_519, var_388);
                var_521 = wp::mul(var_520, var_518);
                var_522 = wp::add(var_227, var_521);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_523 = wp::copy(var_388);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_524 = wp::copy(var_510);
            }
            var_525 = wp::where(var_513, var_514, var_381);
            var_526 = wp::where(var_513, var_515, var_382);
            var_527 = wp::where(var_513, var_524, var_383);
            var_528 = wp::where(var_513, var_522, var_384);
            var_529 = wp::where(var_513, var_523, var_385);
            var_530 = wp::where(var_513, var_516, var_386);
            var_531 = wp::where(var_513, var_518, var_398);
        }
        var_532 = wp::where(var_437, var_525, var_381);
        var_533 = wp::where(var_437, var_526, var_382);
        var_534 = wp::where(var_437, var_527, var_383);
        var_535 = wp::where(var_437, var_528, var_384);
        var_536 = wp::where(var_437, var_529, var_385);
        var_537 = wp::where(var_437, var_530, var_386);
        var_538 = wp::where(var_437, var_531, var_398);
        var_539 = wp::where(var_437, var_505, var_417);
        var_540 = wp::where(var_437, var_439, var_280);
        var_541 = wp::where(var_437, var_500, var_341);
        var_542 = wp::where(var_437, var_501, var_342);
        var_543 = wp::where(var_437, var_502, var_343);
        var_544 = wp::where(var_437, var_503, var_344);
        var_545 = wp::where(var_437, var_473, var_314);
        var_546 = wp::where(var_437, var_510, var_351);
        var_547 = wp::where(var_437, var_511, var_352);
    }
    var_548 = wp::where(var_396, var_532, var_381);
    var_549 = wp::where(var_396, var_533, var_382);
    var_550 = wp::where(var_396, var_534, var_383);
    var_551 = wp::where(var_396, var_535, var_384);
    var_552 = wp::where(var_396, var_536, var_385);
    var_553 = wp::where(var_396, var_537, var_386);
    var_554 = wp::where(var_396, var_437, var_396);
    var_555 = wp::where(var_396, var_538, var_379);
    var_556 = wp::where(var_396, var_415, var_255);
    var_557 = wp::where(var_396, var_539, var_380);
    var_558 = wp::where(var_396, var_421, var_261);
    var_559 = wp::where(var_396, var_422, var_262);
    var_560 = wp::where(var_396, var_425, var_265);
    var_561 = wp::where(var_396, var_429, var_269);
    var_562 = wp::where(var_396, var_430, var_270);
    var_563 = wp::where(var_396, var_433, var_273);
    var_564 = wp::where(var_396, var_540, var_280);
    var_565 = wp::where(var_396, var_541, var_341);
    var_566 = wp::where(var_396, var_542, var_342);
    var_567 = wp::where(var_396, var_543, var_343);
    var_568 = wp::where(var_396, var_544, var_344);
    var_569 = wp::where(var_396, var_545, var_314);
    var_570 = wp::where(var_396, var_546, var_351);
    var_571 = wp::where(var_396, var_547, var_352);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_575 = wp::lshift(var_574, var_572);
    var_576 = wp::bit_and(var_227, var_575);
    var_578 = (var_576 != var_577);
    if (var_578) {
        // continue                                                                           <L 1193>
    }
    var_580 = wp::where(var_578, var_579, var_573);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_580) {
        var_582 = wp::int32(var_581);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_584 = wp::bit_and(var_227, var_583);
        var_587 = wp::where(var_584, var_585, var_586);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_589 = wp::bit_and(var_227, var_588);
        var_592 = wp::where(var_589, var_590, var_591);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_594 = wp::bit_and(var_227, var_593);
        var_597 = wp::where(var_594, var_595, var_596);
        var_598 = wp::vec_t<3, wp::float32>(var_587, var_592, var_597);
        // box_size,                                                                          <L 1204>
        var_599 = wp::cw_mul(var_598, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_599, var_572, var_600);
        // dif = box_pt - pos                                                                 <L 1209>
        var_601 = wp::sub(var_599, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_602 = wp::extract(var_box_size, var_572);
        var_603 = wp::neg(var_602);
        var_604 = wp::extract(var_601, var_572);
        var_605 = wp::mul(var_603, var_604);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_606 = wp::dot(var_4, var_601);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_607 = wp::extract(var_box_size, var_572);
        var_608 = wp::extract(var_box_size, var_572);
        var_609 = wp::mul(var_607, var_608);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_610 = wp::extract(var_box_size, var_572);
        var_611 = wp::neg(var_610);
        var_612 = wp::extract(var_4, var_572);
        var_613 = wp::mul(var_611, var_612);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_614 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_615 = wp::mul(var_609, var_614);
        var_616 = wp::mul(var_613, var_613);
        var_617 = wp::sub(var_615, var_616);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_618 = wp::abs(var_617);
        var_619 = (var_618 < var_275);
        if (var_619) {
            // continue                                                                       <L 1218>
        }
        var_621 = wp::where(var_619, var_620, var_580);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_621) {
            var_623 = wp::div(var_622, var_617);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_624 = wp::mul(var_614, var_605);
            var_625 = wp::mul(var_613, var_606);
            var_626 = wp::sub(var_624, var_625);
            var_627 = wp::mul(var_626, var_623);
            var_628 = wp::float32(var_627);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_629 = wp::mul(var_609, var_606);
            var_630 = wp::mul(var_613, var_605);
            var_631 = wp::sub(var_629, var_630);
            var_632 = wp::mul(var_631, var_623);
            var_633 = wp::float32(var_632);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_635 = wp::int32(var_634);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_637 = wp::int32(var_636);
            // if x1 > 1:                                                                     <L 1229>
            var_639 = (var_628 > var_638);
            if (var_639) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_642 = wp::sub(var_606, var_613);
                var_643 = safe_div_0(var_642, var_614);
            }
            if (!var_639) {
                // elif x1 < -1:                                                              <L 1233>
                var_645 = (var_628 < var_644);
                if (var_645) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_648 = wp::add(var_606, var_613);
                    var_649 = safe_div_0(var_648, var_614);
                }
                var_650 = wp::where(var_645, var_646, var_628);
                var_651 = wp::where(var_645, var_649, var_633);
                var_652 = wp::where(var_645, var_647, var_635);
            }
            var_653 = wp::where(var_639, var_640, var_650);
            var_654 = wp::where(var_639, var_643, var_651);
            var_655 = wp::where(var_639, var_641, var_652);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_657 = (var_654 > var_656);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_658 = var_657;
            if (!var_658) {
                var_660 = (var_654 < var_659);
                var_658 = var_658 || var_660;
            }
            if (var_658) {
                // if x2_over:                                                                <L 1240>
                if (var_657) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_663 = wp::sub(var_605, var_613);
                    var_664 = safe_div_0(var_663, var_609);
                }
                if (!var_657) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_667 = wp::add(var_605, var_613);
                    var_668 = safe_div_0(var_667, var_609);
                }
                var_669 = wp::where(var_657, var_664, var_668);
                var_670 = wp::where(var_657, var_661, var_665);
                var_671 = wp::where(var_657, var_662, var_666);
                // if x1 > 1:                                                                 <L 1249>
                var_673 = (var_669 > var_672);
                if (var_673) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_673) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_677 = (var_669 < var_676);
                    if (var_677) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_680 = wp::where(var_677, var_678, var_669);
                    var_681 = wp::where(var_677, var_679, var_655);
                }
                var_682 = wp::where(var_673, var_674, var_680);
                var_683 = wp::where(var_673, var_675, var_681);
            }
            var_684 = wp::where(var_658, var_682, var_653);
            var_685 = wp::where(var_658, var_670, var_654);
            var_686 = wp::where(var_658, var_683, var_655);
            var_687 = wp::where(var_658, var_671, var_637);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_688 = wp::mul(var_4, var_685);
            var_689 = wp::sub(var_601, var_688);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_690 = wp::extract(var_box_size, var_572);
            var_691 = wp::mul(var_690, var_684);
            wp::add_inplace(var_689, var_572, var_691);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_693 = wp::mul(var_686, var_692);
            var_694 = wp::add(var_693, var_687);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_695 = wp::length_sq(var_689);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_696 = wp::sub(var_548, var_275);
            var_697 = (var_695 < var_696);
            if (var_697) {
                // bestdist = dif_sq                                                          <L 1264>
                var_698 = wp::copy(var_695);
                // bestsegmentpos = x2                                                        <L 1265>
                var_699 = wp::copy(var_685);
                // bestboxpos = x1                                                            <L 1266>
                var_700 = wp::copy(var_684);
                // c2 = ct // 6                                                               <L 1268>
                var_702 = wp::floordiv(var_694, var_701);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_704 = wp::lshift(var_703, var_572);
                var_705 = wp::mul(var_704, var_702);
                var_706 = wp::add(var_227, var_705);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_707 = wp::copy(var_572);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_708 = wp::copy(var_694);
            }
            var_709 = wp::where(var_697, var_698, var_548);
            var_710 = wp::where(var_697, var_699, var_549);
            var_711 = wp::where(var_697, var_708, var_550);
            var_712 = wp::where(var_697, var_706, var_551);
            var_713 = wp::where(var_697, var_707, var_552);
            var_714 = wp::where(var_697, var_700, var_553);
            var_715 = wp::where(var_697, var_702, var_582);
        }
        var_716 = wp::where(var_621, var_709, var_548);
        var_717 = wp::where(var_621, var_710, var_549);
        var_718 = wp::where(var_621, var_711, var_550);
        var_719 = wp::where(var_621, var_712, var_551);
        var_720 = wp::where(var_621, var_713, var_552);
        var_721 = wp::where(var_621, var_714, var_553);
        var_722 = wp::where(var_621, var_715, var_582);
        var_723 = wp::where(var_621, var_689, var_601);
        var_724 = wp::where(var_621, var_623, var_564);
        var_725 = wp::where(var_621, var_684, var_565);
        var_726 = wp::where(var_621, var_685, var_566);
        var_727 = wp::where(var_621, var_686, var_567);
        var_728 = wp::where(var_621, var_687, var_568);
        var_729 = wp::where(var_621, var_657, var_569);
        var_730 = wp::where(var_621, var_694, var_570);
        var_731 = wp::where(var_621, var_695, var_571);
    }
    var_732 = wp::where(var_580, var_716, var_548);
    var_733 = wp::where(var_580, var_717, var_549);
    var_734 = wp::where(var_580, var_718, var_550);
    var_735 = wp::where(var_580, var_719, var_551);
    var_736 = wp::where(var_580, var_720, var_552);
    var_737 = wp::where(var_580, var_721, var_553);
    var_738 = wp::where(var_580, var_621, var_580);
    var_739 = wp::where(var_580, var_722, var_555);
    var_740 = wp::where(var_580, var_599, var_556);
    var_741 = wp::where(var_580, var_723, var_557);
    var_742 = wp::where(var_580, var_605, var_558);
    var_743 = wp::where(var_580, var_606, var_559);
    var_744 = wp::where(var_580, var_609, var_560);
    var_745 = wp::where(var_580, var_613, var_561);
    var_746 = wp::where(var_580, var_614, var_562);
    var_747 = wp::where(var_580, var_617, var_563);
    var_748 = wp::where(var_580, var_724, var_564);
    var_749 = wp::where(var_580, var_725, var_565);
    var_750 = wp::where(var_580, var_726, var_566);
    var_751 = wp::where(var_580, var_727, var_567);
    var_752 = wp::where(var_580, var_728, var_568);
    var_753 = wp::where(var_580, var_729, var_569);
    var_754 = wp::where(var_580, var_730, var_570);
    var_755 = wp::where(var_580, var_731, var_571);
    // for j in range(3):                                                                     <L 1191>
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_760 = wp::lshift(var_759, var_757);
    var_761 = wp::bit_and(var_756, var_760);
    var_763 = (var_761 != var_762);
    if (var_763) {
        // continue                                                                           <L 1193>
    }
    var_765 = wp::where(var_763, var_764, var_758);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_765) {
        var_767 = wp::int32(var_766);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_769 = wp::bit_and(var_756, var_768);
        var_772 = wp::where(var_769, var_770, var_771);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_774 = wp::bit_and(var_756, var_773);
        var_777 = wp::where(var_774, var_775, var_776);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_779 = wp::bit_and(var_756, var_778);
        var_782 = wp::where(var_779, var_780, var_781);
        var_783 = wp::vec_t<3, wp::float32>(var_772, var_777, var_782);
        // box_size,                                                                          <L 1204>
        var_784 = wp::cw_mul(var_783, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_784, var_757, var_785);
        // dif = box_pt - pos                                                                 <L 1209>
        var_786 = wp::sub(var_784, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_787 = wp::extract(var_box_size, var_757);
        var_788 = wp::neg(var_787);
        var_789 = wp::extract(var_786, var_757);
        var_790 = wp::mul(var_788, var_789);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_791 = wp::dot(var_4, var_786);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_792 = wp::extract(var_box_size, var_757);
        var_793 = wp::extract(var_box_size, var_757);
        var_794 = wp::mul(var_792, var_793);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_795 = wp::extract(var_box_size, var_757);
        var_796 = wp::neg(var_795);
        var_797 = wp::extract(var_4, var_757);
        var_798 = wp::mul(var_796, var_797);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_799 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_800 = wp::mul(var_794, var_799);
        var_801 = wp::mul(var_798, var_798);
        var_802 = wp::sub(var_800, var_801);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_803 = wp::abs(var_802);
        var_804 = (var_803 < var_275);
        if (var_804) {
            // continue                                                                       <L 1218>
        }
        var_806 = wp::where(var_804, var_805, var_765);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_806) {
            var_808 = wp::div(var_807, var_802);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_809 = wp::mul(var_799, var_790);
            var_810 = wp::mul(var_798, var_791);
            var_811 = wp::sub(var_809, var_810);
            var_812 = wp::mul(var_811, var_808);
            var_813 = wp::float32(var_812);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_814 = wp::mul(var_794, var_791);
            var_815 = wp::mul(var_798, var_790);
            var_816 = wp::sub(var_814, var_815);
            var_817 = wp::mul(var_816, var_808);
            var_818 = wp::float32(var_817);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_820 = wp::int32(var_819);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_822 = wp::int32(var_821);
            // if x1 > 1:                                                                     <L 1229>
            var_824 = (var_813 > var_823);
            if (var_824) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_827 = wp::sub(var_791, var_798);
                var_828 = safe_div_0(var_827, var_799);
            }
            if (!var_824) {
                // elif x1 < -1:                                                              <L 1233>
                var_830 = (var_813 < var_829);
                if (var_830) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_833 = wp::add(var_791, var_798);
                    var_834 = safe_div_0(var_833, var_799);
                }
                var_835 = wp::where(var_830, var_831, var_813);
                var_836 = wp::where(var_830, var_834, var_818);
                var_837 = wp::where(var_830, var_832, var_820);
            }
            var_838 = wp::where(var_824, var_825, var_835);
            var_839 = wp::where(var_824, var_828, var_836);
            var_840 = wp::where(var_824, var_826, var_837);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_842 = (var_839 > var_841);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_843 = var_842;
            if (!var_843) {
                var_845 = (var_839 < var_844);
                var_843 = var_843 || var_845;
            }
            if (var_843) {
                // if x2_over:                                                                <L 1240>
                if (var_842) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_848 = wp::sub(var_790, var_798);
                    var_849 = safe_div_0(var_848, var_794);
                }
                if (!var_842) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_852 = wp::add(var_790, var_798);
                    var_853 = safe_div_0(var_852, var_794);
                }
                var_854 = wp::where(var_842, var_849, var_853);
                var_855 = wp::where(var_842, var_846, var_850);
                var_856 = wp::where(var_842, var_847, var_851);
                // if x1 > 1:                                                                 <L 1249>
                var_858 = (var_854 > var_857);
                if (var_858) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_858) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_862 = (var_854 < var_861);
                    if (var_862) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_865 = wp::where(var_862, var_863, var_854);
                    var_866 = wp::where(var_862, var_864, var_840);
                }
                var_867 = wp::where(var_858, var_859, var_865);
                var_868 = wp::where(var_858, var_860, var_866);
            }
            var_869 = wp::where(var_843, var_867, var_838);
            var_870 = wp::where(var_843, var_855, var_839);
            var_871 = wp::where(var_843, var_868, var_840);
            var_872 = wp::where(var_843, var_856, var_822);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_873 = wp::mul(var_4, var_870);
            var_874 = wp::sub(var_786, var_873);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_875 = wp::extract(var_box_size, var_757);
            var_876 = wp::mul(var_875, var_869);
            wp::add_inplace(var_874, var_757, var_876);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_878 = wp::mul(var_871, var_877);
            var_879 = wp::add(var_878, var_872);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_880 = wp::length_sq(var_874);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_881 = wp::sub(var_732, var_275);
            var_882 = (var_880 < var_881);
            if (var_882) {
                // bestdist = dif_sq                                                          <L 1264>
                var_883 = wp::copy(var_880);
                // bestsegmentpos = x2                                                        <L 1265>
                var_884 = wp::copy(var_870);
                // bestboxpos = x1                                                            <L 1266>
                var_885 = wp::copy(var_869);
                // c2 = ct // 6                                                               <L 1268>
                var_887 = wp::floordiv(var_879, var_886);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_889 = wp::lshift(var_888, var_757);
                var_890 = wp::mul(var_889, var_887);
                var_891 = wp::add(var_756, var_890);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_892 = wp::copy(var_757);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_893 = wp::copy(var_879);
            }
            var_894 = wp::where(var_882, var_883, var_732);
            var_895 = wp::where(var_882, var_884, var_733);
            var_896 = wp::where(var_882, var_893, var_734);
            var_897 = wp::where(var_882, var_891, var_735);
            var_898 = wp::where(var_882, var_892, var_736);
            var_899 = wp::where(var_882, var_885, var_737);
            var_900 = wp::where(var_882, var_887, var_767);
        }
        var_901 = wp::where(var_806, var_894, var_732);
        var_902 = wp::where(var_806, var_895, var_733);
        var_903 = wp::where(var_806, var_896, var_734);
        var_904 = wp::where(var_806, var_897, var_735);
        var_905 = wp::where(var_806, var_898, var_736);
        var_906 = wp::where(var_806, var_899, var_737);
        var_907 = wp::where(var_806, var_900, var_767);
        var_908 = wp::where(var_806, var_874, var_786);
        var_909 = wp::where(var_806, var_808, var_748);
        var_910 = wp::where(var_806, var_869, var_749);
        var_911 = wp::where(var_806, var_870, var_750);
        var_912 = wp::where(var_806, var_871, var_751);
        var_913 = wp::where(var_806, var_872, var_752);
        var_914 = wp::where(var_806, var_842, var_753);
        var_915 = wp::where(var_806, var_879, var_754);
        var_916 = wp::where(var_806, var_880, var_755);
    }
    var_917 = wp::where(var_765, var_901, var_732);
    var_918 = wp::where(var_765, var_902, var_733);
    var_919 = wp::where(var_765, var_903, var_734);
    var_920 = wp::where(var_765, var_904, var_735);
    var_921 = wp::where(var_765, var_905, var_736);
    var_922 = wp::where(var_765, var_906, var_737);
    var_923 = wp::where(var_765, var_907, var_739);
    var_924 = wp::where(var_765, var_784, var_740);
    var_925 = wp::where(var_765, var_908, var_741);
    var_926 = wp::where(var_765, var_790, var_742);
    var_927 = wp::where(var_765, var_791, var_743);
    var_928 = wp::where(var_765, var_794, var_744);
    var_929 = wp::where(var_765, var_798, var_745);
    var_930 = wp::where(var_765, var_799, var_746);
    var_931 = wp::where(var_765, var_802, var_747);
    var_932 = wp::where(var_765, var_909, var_748);
    var_933 = wp::where(var_765, var_910, var_749);
    var_934 = wp::where(var_765, var_911, var_750);
    var_935 = wp::where(var_765, var_912, var_751);
    var_936 = wp::where(var_765, var_913, var_752);
    var_937 = wp::where(var_765, var_914, var_753);
    var_938 = wp::where(var_765, var_915, var_754);
    var_939 = wp::where(var_765, var_916, var_755);
    var_940 = wp::where(var_765, var_806, var_765);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_944 = wp::lshift(var_943, var_941);
    var_945 = wp::bit_and(var_756, var_944);
    var_947 = (var_945 != var_946);
    if (var_947) {
        // continue                                                                           <L 1193>
    }
    var_949 = wp::where(var_947, var_948, var_942);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_949) {
        var_951 = wp::int32(var_950);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_953 = wp::bit_and(var_756, var_952);
        var_956 = wp::where(var_953, var_954, var_955);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_958 = wp::bit_and(var_756, var_957);
        var_961 = wp::where(var_958, var_959, var_960);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_963 = wp::bit_and(var_756, var_962);
        var_966 = wp::where(var_963, var_964, var_965);
        var_967 = wp::vec_t<3, wp::float32>(var_956, var_961, var_966);
        // box_size,                                                                          <L 1204>
        var_968 = wp::cw_mul(var_967, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_968, var_941, var_969);
        // dif = box_pt - pos                                                                 <L 1209>
        var_970 = wp::sub(var_968, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_971 = wp::extract(var_box_size, var_941);
        var_972 = wp::neg(var_971);
        var_973 = wp::extract(var_970, var_941);
        var_974 = wp::mul(var_972, var_973);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_975 = wp::dot(var_4, var_970);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_976 = wp::extract(var_box_size, var_941);
        var_977 = wp::extract(var_box_size, var_941);
        var_978 = wp::mul(var_976, var_977);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_979 = wp::extract(var_box_size, var_941);
        var_980 = wp::neg(var_979);
        var_981 = wp::extract(var_4, var_941);
        var_982 = wp::mul(var_980, var_981);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_983 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_984 = wp::mul(var_978, var_983);
        var_985 = wp::mul(var_982, var_982);
        var_986 = wp::sub(var_984, var_985);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_987 = wp::abs(var_986);
        var_988 = (var_987 < var_275);
        if (var_988) {
            // continue                                                                       <L 1218>
        }
        var_990 = wp::where(var_988, var_989, var_949);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_990) {
            var_992 = wp::div(var_991, var_986);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_993 = wp::mul(var_983, var_974);
            var_994 = wp::mul(var_982, var_975);
            var_995 = wp::sub(var_993, var_994);
            var_996 = wp::mul(var_995, var_992);
            var_997 = wp::float32(var_996);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_998 = wp::mul(var_978, var_975);
            var_999 = wp::mul(var_982, var_974);
            var_1000 = wp::sub(var_998, var_999);
            var_1001 = wp::mul(var_1000, var_992);
            var_1002 = wp::float32(var_1001);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_1004 = wp::int32(var_1003);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_1006 = wp::int32(var_1005);
            // if x1 > 1:                                                                     <L 1229>
            var_1008 = (var_997 > var_1007);
            if (var_1008) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_1011 = wp::sub(var_975, var_982);
                var_1012 = safe_div_0(var_1011, var_983);
            }
            if (!var_1008) {
                // elif x1 < -1:                                                              <L 1233>
                var_1014 = (var_997 < var_1013);
                if (var_1014) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_1017 = wp::add(var_975, var_982);
                    var_1018 = safe_div_0(var_1017, var_983);
                }
                var_1019 = wp::where(var_1014, var_1015, var_997);
                var_1020 = wp::where(var_1014, var_1018, var_1002);
                var_1021 = wp::where(var_1014, var_1016, var_1004);
            }
            var_1022 = wp::where(var_1008, var_1009, var_1019);
            var_1023 = wp::where(var_1008, var_1012, var_1020);
            var_1024 = wp::where(var_1008, var_1010, var_1021);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_1026 = (var_1023 > var_1025);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_1027 = var_1026;
            if (!var_1027) {
                var_1029 = (var_1023 < var_1028);
                var_1027 = var_1027 || var_1029;
            }
            if (var_1027) {
                // if x2_over:                                                                <L 1240>
                if (var_1026) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_1032 = wp::sub(var_974, var_982);
                    var_1033 = safe_div_0(var_1032, var_978);
                }
                if (!var_1026) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_1036 = wp::add(var_974, var_982);
                    var_1037 = safe_div_0(var_1036, var_978);
                }
                var_1038 = wp::where(var_1026, var_1033, var_1037);
                var_1039 = wp::where(var_1026, var_1030, var_1034);
                var_1040 = wp::where(var_1026, var_1031, var_1035);
                // if x1 > 1:                                                                 <L 1249>
                var_1042 = (var_1038 > var_1041);
                if (var_1042) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_1042) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_1046 = (var_1038 < var_1045);
                    if (var_1046) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_1049 = wp::where(var_1046, var_1047, var_1038);
                    var_1050 = wp::where(var_1046, var_1048, var_1024);
                }
                var_1051 = wp::where(var_1042, var_1043, var_1049);
                var_1052 = wp::where(var_1042, var_1044, var_1050);
            }
            var_1053 = wp::where(var_1027, var_1051, var_1022);
            var_1054 = wp::where(var_1027, var_1039, var_1023);
            var_1055 = wp::where(var_1027, var_1052, var_1024);
            var_1056 = wp::where(var_1027, var_1040, var_1006);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_1057 = wp::mul(var_4, var_1054);
            var_1058 = wp::sub(var_970, var_1057);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_1059 = wp::extract(var_box_size, var_941);
            var_1060 = wp::mul(var_1059, var_1053);
            wp::add_inplace(var_1058, var_941, var_1060);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_1062 = wp::mul(var_1055, var_1061);
            var_1063 = wp::add(var_1062, var_1056);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_1064 = wp::length_sq(var_1058);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_1065 = wp::sub(var_917, var_275);
            var_1066 = (var_1064 < var_1065);
            if (var_1066) {
                // bestdist = dif_sq                                                          <L 1264>
                var_1067 = wp::copy(var_1064);
                // bestsegmentpos = x2                                                        <L 1265>
                var_1068 = wp::copy(var_1054);
                // bestboxpos = x1                                                            <L 1266>
                var_1069 = wp::copy(var_1053);
                // c2 = ct // 6                                                               <L 1268>
                var_1071 = wp::floordiv(var_1063, var_1070);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_1073 = wp::lshift(var_1072, var_941);
                var_1074 = wp::mul(var_1073, var_1071);
                var_1075 = wp::add(var_756, var_1074);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_1076 = wp::copy(var_941);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_1077 = wp::copy(var_1063);
            }
            var_1078 = wp::where(var_1066, var_1067, var_917);
            var_1079 = wp::where(var_1066, var_1068, var_918);
            var_1080 = wp::where(var_1066, var_1077, var_919);
            var_1081 = wp::where(var_1066, var_1075, var_920);
            var_1082 = wp::where(var_1066, var_1076, var_921);
            var_1083 = wp::where(var_1066, var_1069, var_922);
            var_1084 = wp::where(var_1066, var_1071, var_951);
        }
        var_1085 = wp::where(var_990, var_1078, var_917);
        var_1086 = wp::where(var_990, var_1079, var_918);
        var_1087 = wp::where(var_990, var_1080, var_919);
        var_1088 = wp::where(var_990, var_1081, var_920);
        var_1089 = wp::where(var_990, var_1082, var_921);
        var_1090 = wp::where(var_990, var_1083, var_922);
        var_1091 = wp::where(var_990, var_1084, var_951);
        var_1092 = wp::where(var_990, var_1058, var_970);
        var_1093 = wp::where(var_990, var_992, var_932);
        var_1094 = wp::where(var_990, var_1053, var_933);
        var_1095 = wp::where(var_990, var_1054, var_934);
        var_1096 = wp::where(var_990, var_1055, var_935);
        var_1097 = wp::where(var_990, var_1056, var_936);
        var_1098 = wp::where(var_990, var_1026, var_937);
        var_1099 = wp::where(var_990, var_1063, var_938);
        var_1100 = wp::where(var_990, var_1064, var_939);
    }
    var_1101 = wp::where(var_949, var_1085, var_917);
    var_1102 = wp::where(var_949, var_1086, var_918);
    var_1103 = wp::where(var_949, var_1087, var_919);
    var_1104 = wp::where(var_949, var_1088, var_920);
    var_1105 = wp::where(var_949, var_1089, var_921);
    var_1106 = wp::where(var_949, var_1090, var_922);
    var_1107 = wp::where(var_949, var_1091, var_923);
    var_1108 = wp::where(var_949, var_968, var_924);
    var_1109 = wp::where(var_949, var_1092, var_925);
    var_1110 = wp::where(var_949, var_974, var_926);
    var_1111 = wp::where(var_949, var_975, var_927);
    var_1112 = wp::where(var_949, var_978, var_928);
    var_1113 = wp::where(var_949, var_982, var_929);
    var_1114 = wp::where(var_949, var_983, var_930);
    var_1115 = wp::where(var_949, var_986, var_931);
    var_1116 = wp::where(var_949, var_1093, var_932);
    var_1117 = wp::where(var_949, var_1094, var_933);
    var_1118 = wp::where(var_949, var_1095, var_934);
    var_1119 = wp::where(var_949, var_1096, var_935);
    var_1120 = wp::where(var_949, var_1097, var_936);
    var_1121 = wp::where(var_949, var_1098, var_937);
    var_1122 = wp::where(var_949, var_1099, var_938);
    var_1123 = wp::where(var_949, var_1100, var_939);
    var_1124 = wp::where(var_949, var_990, var_949);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_1128 = wp::lshift(var_1127, var_1125);
    var_1129 = wp::bit_and(var_756, var_1128);
    var_1131 = (var_1129 != var_1130);
    if (var_1131) {
        // continue                                                                           <L 1193>
    }
    var_1133 = wp::where(var_1131, var_1132, var_1126);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_1133) {
        var_1135 = wp::int32(var_1134);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_1137 = wp::bit_and(var_756, var_1136);
        var_1140 = wp::where(var_1137, var_1138, var_1139);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_1142 = wp::bit_and(var_756, var_1141);
        var_1145 = wp::where(var_1142, var_1143, var_1144);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_1147 = wp::bit_and(var_756, var_1146);
        var_1150 = wp::where(var_1147, var_1148, var_1149);
        var_1151 = wp::vec_t<3, wp::float32>(var_1140, var_1145, var_1150);
        // box_size,                                                                          <L 1204>
        var_1152 = wp::cw_mul(var_1151, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_1152, var_1125, var_1153);
        // dif = box_pt - pos                                                                 <L 1209>
        var_1154 = wp::sub(var_1152, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_1155 = wp::extract(var_box_size, var_1125);
        var_1156 = wp::neg(var_1155);
        var_1157 = wp::extract(var_1154, var_1125);
        var_1158 = wp::mul(var_1156, var_1157);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_1159 = wp::dot(var_4, var_1154);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_1160 = wp::extract(var_box_size, var_1125);
        var_1161 = wp::extract(var_box_size, var_1125);
        var_1162 = wp::mul(var_1160, var_1161);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_1163 = wp::extract(var_box_size, var_1125);
        var_1164 = wp::neg(var_1163);
        var_1165 = wp::extract(var_4, var_1125);
        var_1166 = wp::mul(var_1164, var_1165);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_1167 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_1168 = wp::mul(var_1162, var_1167);
        var_1169 = wp::mul(var_1166, var_1166);
        var_1170 = wp::sub(var_1168, var_1169);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_1171 = wp::abs(var_1170);
        var_1172 = (var_1171 < var_275);
        if (var_1172) {
            // continue                                                                       <L 1218>
        }
        var_1174 = wp::where(var_1172, var_1173, var_1133);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_1174) {
            var_1176 = wp::div(var_1175, var_1170);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_1177 = wp::mul(var_1167, var_1158);
            var_1178 = wp::mul(var_1166, var_1159);
            var_1179 = wp::sub(var_1177, var_1178);
            var_1180 = wp::mul(var_1179, var_1176);
            var_1181 = wp::float32(var_1180);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_1182 = wp::mul(var_1162, var_1159);
            var_1183 = wp::mul(var_1166, var_1158);
            var_1184 = wp::sub(var_1182, var_1183);
            var_1185 = wp::mul(var_1184, var_1176);
            var_1186 = wp::float32(var_1185);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_1188 = wp::int32(var_1187);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_1190 = wp::int32(var_1189);
            // if x1 > 1:                                                                     <L 1229>
            var_1192 = (var_1181 > var_1191);
            if (var_1192) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_1195 = wp::sub(var_1159, var_1166);
                var_1196 = safe_div_0(var_1195, var_1167);
            }
            if (!var_1192) {
                // elif x1 < -1:                                                              <L 1233>
                var_1198 = (var_1181 < var_1197);
                if (var_1198) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_1201 = wp::add(var_1159, var_1166);
                    var_1202 = safe_div_0(var_1201, var_1167);
                }
                var_1203 = wp::where(var_1198, var_1199, var_1181);
                var_1204 = wp::where(var_1198, var_1202, var_1186);
                var_1205 = wp::where(var_1198, var_1200, var_1188);
            }
            var_1206 = wp::where(var_1192, var_1193, var_1203);
            var_1207 = wp::where(var_1192, var_1196, var_1204);
            var_1208 = wp::where(var_1192, var_1194, var_1205);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_1210 = (var_1207 > var_1209);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_1211 = var_1210;
            if (!var_1211) {
                var_1213 = (var_1207 < var_1212);
                var_1211 = var_1211 || var_1213;
            }
            if (var_1211) {
                // if x2_over:                                                                <L 1240>
                if (var_1210) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_1216 = wp::sub(var_1158, var_1166);
                    var_1217 = safe_div_0(var_1216, var_1162);
                }
                if (!var_1210) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_1220 = wp::add(var_1158, var_1166);
                    var_1221 = safe_div_0(var_1220, var_1162);
                }
                var_1222 = wp::where(var_1210, var_1217, var_1221);
                var_1223 = wp::where(var_1210, var_1214, var_1218);
                var_1224 = wp::where(var_1210, var_1215, var_1219);
                // if x1 > 1:                                                                 <L 1249>
                var_1226 = (var_1222 > var_1225);
                if (var_1226) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_1226) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_1230 = (var_1222 < var_1229);
                    if (var_1230) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_1233 = wp::where(var_1230, var_1231, var_1222);
                    var_1234 = wp::where(var_1230, var_1232, var_1208);
                }
                var_1235 = wp::where(var_1226, var_1227, var_1233);
                var_1236 = wp::where(var_1226, var_1228, var_1234);
            }
            var_1237 = wp::where(var_1211, var_1235, var_1206);
            var_1238 = wp::where(var_1211, var_1223, var_1207);
            var_1239 = wp::where(var_1211, var_1236, var_1208);
            var_1240 = wp::where(var_1211, var_1224, var_1190);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_1241 = wp::mul(var_4, var_1238);
            var_1242 = wp::sub(var_1154, var_1241);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_1243 = wp::extract(var_box_size, var_1125);
            var_1244 = wp::mul(var_1243, var_1237);
            wp::add_inplace(var_1242, var_1125, var_1244);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_1246 = wp::mul(var_1239, var_1245);
            var_1247 = wp::add(var_1246, var_1240);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_1248 = wp::length_sq(var_1242);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_1249 = wp::sub(var_1101, var_275);
            var_1250 = (var_1248 < var_1249);
            if (var_1250) {
                // bestdist = dif_sq                                                          <L 1264>
                var_1251 = wp::copy(var_1248);
                // bestsegmentpos = x2                                                        <L 1265>
                var_1252 = wp::copy(var_1238);
                // bestboxpos = x1                                                            <L 1266>
                var_1253 = wp::copy(var_1237);
                // c2 = ct // 6                                                               <L 1268>
                var_1255 = wp::floordiv(var_1247, var_1254);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_1257 = wp::lshift(var_1256, var_1125);
                var_1258 = wp::mul(var_1257, var_1255);
                var_1259 = wp::add(var_756, var_1258);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_1260 = wp::copy(var_1125);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_1261 = wp::copy(var_1247);
            }
            var_1262 = wp::where(var_1250, var_1251, var_1101);
            var_1263 = wp::where(var_1250, var_1252, var_1102);
            var_1264 = wp::where(var_1250, var_1261, var_1103);
            var_1265 = wp::where(var_1250, var_1259, var_1104);
            var_1266 = wp::where(var_1250, var_1260, var_1105);
            var_1267 = wp::where(var_1250, var_1253, var_1106);
            var_1268 = wp::where(var_1250, var_1255, var_1135);
        }
        var_1269 = wp::where(var_1174, var_1262, var_1101);
        var_1270 = wp::where(var_1174, var_1263, var_1102);
        var_1271 = wp::where(var_1174, var_1264, var_1103);
        var_1272 = wp::where(var_1174, var_1265, var_1104);
        var_1273 = wp::where(var_1174, var_1266, var_1105);
        var_1274 = wp::where(var_1174, var_1267, var_1106);
        var_1275 = wp::where(var_1174, var_1268, var_1135);
        var_1276 = wp::where(var_1174, var_1242, var_1154);
        var_1277 = wp::where(var_1174, var_1176, var_1116);
        var_1278 = wp::where(var_1174, var_1237, var_1117);
        var_1279 = wp::where(var_1174, var_1238, var_1118);
        var_1280 = wp::where(var_1174, var_1239, var_1119);
        var_1281 = wp::where(var_1174, var_1240, var_1120);
        var_1282 = wp::where(var_1174, var_1210, var_1121);
        var_1283 = wp::where(var_1174, var_1247, var_1122);
        var_1284 = wp::where(var_1174, var_1248, var_1123);
    }
    var_1285 = wp::where(var_1133, var_1269, var_1101);
    var_1286 = wp::where(var_1133, var_1270, var_1102);
    var_1287 = wp::where(var_1133, var_1271, var_1103);
    var_1288 = wp::where(var_1133, var_1272, var_1104);
    var_1289 = wp::where(var_1133, var_1273, var_1105);
    var_1290 = wp::where(var_1133, var_1274, var_1106);
    var_1291 = wp::where(var_1133, var_1275, var_1107);
    var_1292 = wp::where(var_1133, var_1152, var_1108);
    var_1293 = wp::where(var_1133, var_1276, var_1109);
    var_1294 = wp::where(var_1133, var_1158, var_1110);
    var_1295 = wp::where(var_1133, var_1159, var_1111);
    var_1296 = wp::where(var_1133, var_1162, var_1112);
    var_1297 = wp::where(var_1133, var_1166, var_1113);
    var_1298 = wp::where(var_1133, var_1167, var_1114);
    var_1299 = wp::where(var_1133, var_1170, var_1115);
    var_1300 = wp::where(var_1133, var_1277, var_1116);
    var_1301 = wp::where(var_1133, var_1278, var_1117);
    var_1302 = wp::where(var_1133, var_1279, var_1118);
    var_1303 = wp::where(var_1133, var_1280, var_1119);
    var_1304 = wp::where(var_1133, var_1281, var_1120);
    var_1305 = wp::where(var_1133, var_1282, var_1121);
    var_1306 = wp::where(var_1133, var_1283, var_1122);
    var_1307 = wp::where(var_1133, var_1284, var_1123);
    var_1308 = wp::where(var_1133, var_1174, var_1133);
    // for j in range(3):                                                                     <L 1191>
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_1313 = wp::lshift(var_1312, var_1310);
    var_1314 = wp::bit_and(var_1309, var_1313);
    var_1316 = (var_1314 != var_1315);
    if (var_1316) {
        // continue                                                                           <L 1193>
    }
    var_1318 = wp::where(var_1316, var_1317, var_1311);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_1318) {
        var_1320 = wp::int32(var_1319);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_1322 = wp::bit_and(var_1309, var_1321);
        var_1325 = wp::where(var_1322, var_1323, var_1324);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_1327 = wp::bit_and(var_1309, var_1326);
        var_1330 = wp::where(var_1327, var_1328, var_1329);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_1332 = wp::bit_and(var_1309, var_1331);
        var_1335 = wp::where(var_1332, var_1333, var_1334);
        var_1336 = wp::vec_t<3, wp::float32>(var_1325, var_1330, var_1335);
        // box_size,                                                                          <L 1204>
        var_1337 = wp::cw_mul(var_1336, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_1337, var_1310, var_1338);
        // dif = box_pt - pos                                                                 <L 1209>
        var_1339 = wp::sub(var_1337, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_1340 = wp::extract(var_box_size, var_1310);
        var_1341 = wp::neg(var_1340);
        var_1342 = wp::extract(var_1339, var_1310);
        var_1343 = wp::mul(var_1341, var_1342);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_1344 = wp::dot(var_4, var_1339);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_1345 = wp::extract(var_box_size, var_1310);
        var_1346 = wp::extract(var_box_size, var_1310);
        var_1347 = wp::mul(var_1345, var_1346);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_1348 = wp::extract(var_box_size, var_1310);
        var_1349 = wp::neg(var_1348);
        var_1350 = wp::extract(var_4, var_1310);
        var_1351 = wp::mul(var_1349, var_1350);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_1352 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_1353 = wp::mul(var_1347, var_1352);
        var_1354 = wp::mul(var_1351, var_1351);
        var_1355 = wp::sub(var_1353, var_1354);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_1356 = wp::abs(var_1355);
        var_1357 = (var_1356 < var_275);
        if (var_1357) {
            // continue                                                                       <L 1218>
        }
        var_1359 = wp::where(var_1357, var_1358, var_1318);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_1359) {
            var_1361 = wp::div(var_1360, var_1355);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_1362 = wp::mul(var_1352, var_1343);
            var_1363 = wp::mul(var_1351, var_1344);
            var_1364 = wp::sub(var_1362, var_1363);
            var_1365 = wp::mul(var_1364, var_1361);
            var_1366 = wp::float32(var_1365);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_1367 = wp::mul(var_1347, var_1344);
            var_1368 = wp::mul(var_1351, var_1343);
            var_1369 = wp::sub(var_1367, var_1368);
            var_1370 = wp::mul(var_1369, var_1361);
            var_1371 = wp::float32(var_1370);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_1373 = wp::int32(var_1372);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_1375 = wp::int32(var_1374);
            // if x1 > 1:                                                                     <L 1229>
            var_1377 = (var_1366 > var_1376);
            if (var_1377) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_1380 = wp::sub(var_1344, var_1351);
                var_1381 = safe_div_0(var_1380, var_1352);
            }
            if (!var_1377) {
                // elif x1 < -1:                                                              <L 1233>
                var_1383 = (var_1366 < var_1382);
                if (var_1383) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_1386 = wp::add(var_1344, var_1351);
                    var_1387 = safe_div_0(var_1386, var_1352);
                }
                var_1388 = wp::where(var_1383, var_1384, var_1366);
                var_1389 = wp::where(var_1383, var_1387, var_1371);
                var_1390 = wp::where(var_1383, var_1385, var_1373);
            }
            var_1391 = wp::where(var_1377, var_1378, var_1388);
            var_1392 = wp::where(var_1377, var_1381, var_1389);
            var_1393 = wp::where(var_1377, var_1379, var_1390);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_1395 = (var_1392 > var_1394);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_1396 = var_1395;
            if (!var_1396) {
                var_1398 = (var_1392 < var_1397);
                var_1396 = var_1396 || var_1398;
            }
            if (var_1396) {
                // if x2_over:                                                                <L 1240>
                if (var_1395) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_1401 = wp::sub(var_1343, var_1351);
                    var_1402 = safe_div_0(var_1401, var_1347);
                }
                if (!var_1395) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_1405 = wp::add(var_1343, var_1351);
                    var_1406 = safe_div_0(var_1405, var_1347);
                }
                var_1407 = wp::where(var_1395, var_1402, var_1406);
                var_1408 = wp::where(var_1395, var_1399, var_1403);
                var_1409 = wp::where(var_1395, var_1400, var_1404);
                // if x1 > 1:                                                                 <L 1249>
                var_1411 = (var_1407 > var_1410);
                if (var_1411) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_1411) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_1415 = (var_1407 < var_1414);
                    if (var_1415) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_1418 = wp::where(var_1415, var_1416, var_1407);
                    var_1419 = wp::where(var_1415, var_1417, var_1393);
                }
                var_1420 = wp::where(var_1411, var_1412, var_1418);
                var_1421 = wp::where(var_1411, var_1413, var_1419);
            }
            var_1422 = wp::where(var_1396, var_1420, var_1391);
            var_1423 = wp::where(var_1396, var_1408, var_1392);
            var_1424 = wp::where(var_1396, var_1421, var_1393);
            var_1425 = wp::where(var_1396, var_1409, var_1375);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_1426 = wp::mul(var_4, var_1423);
            var_1427 = wp::sub(var_1339, var_1426);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_1428 = wp::extract(var_box_size, var_1310);
            var_1429 = wp::mul(var_1428, var_1422);
            wp::add_inplace(var_1427, var_1310, var_1429);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_1431 = wp::mul(var_1424, var_1430);
            var_1432 = wp::add(var_1431, var_1425);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_1433 = wp::length_sq(var_1427);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_1434 = wp::sub(var_1285, var_275);
            var_1435 = (var_1433 < var_1434);
            if (var_1435) {
                // bestdist = dif_sq                                                          <L 1264>
                var_1436 = wp::copy(var_1433);
                // bestsegmentpos = x2                                                        <L 1265>
                var_1437 = wp::copy(var_1423);
                // bestboxpos = x1                                                            <L 1266>
                var_1438 = wp::copy(var_1422);
                // c2 = ct // 6                                                               <L 1268>
                var_1440 = wp::floordiv(var_1432, var_1439);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_1442 = wp::lshift(var_1441, var_1310);
                var_1443 = wp::mul(var_1442, var_1440);
                var_1444 = wp::add(var_1309, var_1443);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_1445 = wp::copy(var_1310);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_1446 = wp::copy(var_1432);
            }
            var_1447 = wp::where(var_1435, var_1436, var_1285);
            var_1448 = wp::where(var_1435, var_1437, var_1286);
            var_1449 = wp::where(var_1435, var_1446, var_1287);
            var_1450 = wp::where(var_1435, var_1444, var_1288);
            var_1451 = wp::where(var_1435, var_1445, var_1289);
            var_1452 = wp::where(var_1435, var_1438, var_1290);
            var_1453 = wp::where(var_1435, var_1440, var_1320);
        }
        var_1454 = wp::where(var_1359, var_1447, var_1285);
        var_1455 = wp::where(var_1359, var_1448, var_1286);
        var_1456 = wp::where(var_1359, var_1449, var_1287);
        var_1457 = wp::where(var_1359, var_1450, var_1288);
        var_1458 = wp::where(var_1359, var_1451, var_1289);
        var_1459 = wp::where(var_1359, var_1452, var_1290);
        var_1460 = wp::where(var_1359, var_1453, var_1320);
        var_1461 = wp::where(var_1359, var_1427, var_1339);
        var_1462 = wp::where(var_1359, var_1361, var_1300);
        var_1463 = wp::where(var_1359, var_1422, var_1301);
        var_1464 = wp::where(var_1359, var_1423, var_1302);
        var_1465 = wp::where(var_1359, var_1424, var_1303);
        var_1466 = wp::where(var_1359, var_1425, var_1304);
        var_1467 = wp::where(var_1359, var_1395, var_1305);
        var_1468 = wp::where(var_1359, var_1432, var_1306);
        var_1469 = wp::where(var_1359, var_1433, var_1307);
    }
    var_1470 = wp::where(var_1318, var_1454, var_1285);
    var_1471 = wp::where(var_1318, var_1455, var_1286);
    var_1472 = wp::where(var_1318, var_1456, var_1287);
    var_1473 = wp::where(var_1318, var_1457, var_1288);
    var_1474 = wp::where(var_1318, var_1458, var_1289);
    var_1475 = wp::where(var_1318, var_1459, var_1290);
    var_1476 = wp::where(var_1318, var_1460, var_1291);
    var_1477 = wp::where(var_1318, var_1337, var_1292);
    var_1478 = wp::where(var_1318, var_1461, var_1293);
    var_1479 = wp::where(var_1318, var_1343, var_1294);
    var_1480 = wp::where(var_1318, var_1344, var_1295);
    var_1481 = wp::where(var_1318, var_1347, var_1296);
    var_1482 = wp::where(var_1318, var_1351, var_1297);
    var_1483 = wp::where(var_1318, var_1352, var_1298);
    var_1484 = wp::where(var_1318, var_1355, var_1299);
    var_1485 = wp::where(var_1318, var_1462, var_1300);
    var_1486 = wp::where(var_1318, var_1463, var_1301);
    var_1487 = wp::where(var_1318, var_1464, var_1302);
    var_1488 = wp::where(var_1318, var_1465, var_1303);
    var_1489 = wp::where(var_1318, var_1466, var_1304);
    var_1490 = wp::where(var_1318, var_1467, var_1305);
    var_1491 = wp::where(var_1318, var_1468, var_1306);
    var_1492 = wp::where(var_1318, var_1469, var_1307);
    var_1493 = wp::where(var_1318, var_1359, var_1318);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_1497 = wp::lshift(var_1496, var_1494);
    var_1498 = wp::bit_and(var_1309, var_1497);
    var_1500 = (var_1498 != var_1499);
    if (var_1500) {
        // continue                                                                           <L 1193>
    }
    var_1502 = wp::where(var_1500, var_1501, var_1495);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_1502) {
        var_1504 = wp::int32(var_1503);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_1506 = wp::bit_and(var_1309, var_1505);
        var_1509 = wp::where(var_1506, var_1507, var_1508);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_1511 = wp::bit_and(var_1309, var_1510);
        var_1514 = wp::where(var_1511, var_1512, var_1513);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_1516 = wp::bit_and(var_1309, var_1515);
        var_1519 = wp::where(var_1516, var_1517, var_1518);
        var_1520 = wp::vec_t<3, wp::float32>(var_1509, var_1514, var_1519);
        // box_size,                                                                          <L 1204>
        var_1521 = wp::cw_mul(var_1520, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_1521, var_1494, var_1522);
        // dif = box_pt - pos                                                                 <L 1209>
        var_1523 = wp::sub(var_1521, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_1524 = wp::extract(var_box_size, var_1494);
        var_1525 = wp::neg(var_1524);
        var_1526 = wp::extract(var_1523, var_1494);
        var_1527 = wp::mul(var_1525, var_1526);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_1528 = wp::dot(var_4, var_1523);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_1529 = wp::extract(var_box_size, var_1494);
        var_1530 = wp::extract(var_box_size, var_1494);
        var_1531 = wp::mul(var_1529, var_1530);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_1532 = wp::extract(var_box_size, var_1494);
        var_1533 = wp::neg(var_1532);
        var_1534 = wp::extract(var_4, var_1494);
        var_1535 = wp::mul(var_1533, var_1534);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_1536 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_1537 = wp::mul(var_1531, var_1536);
        var_1538 = wp::mul(var_1535, var_1535);
        var_1539 = wp::sub(var_1537, var_1538);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_1540 = wp::abs(var_1539);
        var_1541 = (var_1540 < var_275);
        if (var_1541) {
            // continue                                                                       <L 1218>
        }
        var_1543 = wp::where(var_1541, var_1542, var_1502);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_1543) {
            var_1545 = wp::div(var_1544, var_1539);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_1546 = wp::mul(var_1536, var_1527);
            var_1547 = wp::mul(var_1535, var_1528);
            var_1548 = wp::sub(var_1546, var_1547);
            var_1549 = wp::mul(var_1548, var_1545);
            var_1550 = wp::float32(var_1549);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_1551 = wp::mul(var_1531, var_1528);
            var_1552 = wp::mul(var_1535, var_1527);
            var_1553 = wp::sub(var_1551, var_1552);
            var_1554 = wp::mul(var_1553, var_1545);
            var_1555 = wp::float32(var_1554);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_1557 = wp::int32(var_1556);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_1559 = wp::int32(var_1558);
            // if x1 > 1:                                                                     <L 1229>
            var_1561 = (var_1550 > var_1560);
            if (var_1561) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_1564 = wp::sub(var_1528, var_1535);
                var_1565 = safe_div_0(var_1564, var_1536);
            }
            if (!var_1561) {
                // elif x1 < -1:                                                              <L 1233>
                var_1567 = (var_1550 < var_1566);
                if (var_1567) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_1570 = wp::add(var_1528, var_1535);
                    var_1571 = safe_div_0(var_1570, var_1536);
                }
                var_1572 = wp::where(var_1567, var_1568, var_1550);
                var_1573 = wp::where(var_1567, var_1571, var_1555);
                var_1574 = wp::where(var_1567, var_1569, var_1557);
            }
            var_1575 = wp::where(var_1561, var_1562, var_1572);
            var_1576 = wp::where(var_1561, var_1565, var_1573);
            var_1577 = wp::where(var_1561, var_1563, var_1574);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_1579 = (var_1576 > var_1578);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_1580 = var_1579;
            if (!var_1580) {
                var_1582 = (var_1576 < var_1581);
                var_1580 = var_1580 || var_1582;
            }
            if (var_1580) {
                // if x2_over:                                                                <L 1240>
                if (var_1579) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_1585 = wp::sub(var_1527, var_1535);
                    var_1586 = safe_div_0(var_1585, var_1531);
                }
                if (!var_1579) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_1589 = wp::add(var_1527, var_1535);
                    var_1590 = safe_div_0(var_1589, var_1531);
                }
                var_1591 = wp::where(var_1579, var_1586, var_1590);
                var_1592 = wp::where(var_1579, var_1583, var_1587);
                var_1593 = wp::where(var_1579, var_1584, var_1588);
                // if x1 > 1:                                                                 <L 1249>
                var_1595 = (var_1591 > var_1594);
                if (var_1595) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_1595) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_1599 = (var_1591 < var_1598);
                    if (var_1599) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_1602 = wp::where(var_1599, var_1600, var_1591);
                    var_1603 = wp::where(var_1599, var_1601, var_1577);
                }
                var_1604 = wp::where(var_1595, var_1596, var_1602);
                var_1605 = wp::where(var_1595, var_1597, var_1603);
            }
            var_1606 = wp::where(var_1580, var_1604, var_1575);
            var_1607 = wp::where(var_1580, var_1592, var_1576);
            var_1608 = wp::where(var_1580, var_1605, var_1577);
            var_1609 = wp::where(var_1580, var_1593, var_1559);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_1610 = wp::mul(var_4, var_1607);
            var_1611 = wp::sub(var_1523, var_1610);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_1612 = wp::extract(var_box_size, var_1494);
            var_1613 = wp::mul(var_1612, var_1606);
            wp::add_inplace(var_1611, var_1494, var_1613);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_1615 = wp::mul(var_1608, var_1614);
            var_1616 = wp::add(var_1615, var_1609);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_1617 = wp::length_sq(var_1611);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_1618 = wp::sub(var_1470, var_275);
            var_1619 = (var_1617 < var_1618);
            if (var_1619) {
                // bestdist = dif_sq                                                          <L 1264>
                var_1620 = wp::copy(var_1617);
                // bestsegmentpos = x2                                                        <L 1265>
                var_1621 = wp::copy(var_1607);
                // bestboxpos = x1                                                            <L 1266>
                var_1622 = wp::copy(var_1606);
                // c2 = ct // 6                                                               <L 1268>
                var_1624 = wp::floordiv(var_1616, var_1623);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_1626 = wp::lshift(var_1625, var_1494);
                var_1627 = wp::mul(var_1626, var_1624);
                var_1628 = wp::add(var_1309, var_1627);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_1629 = wp::copy(var_1494);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_1630 = wp::copy(var_1616);
            }
            var_1631 = wp::where(var_1619, var_1620, var_1470);
            var_1632 = wp::where(var_1619, var_1621, var_1471);
            var_1633 = wp::where(var_1619, var_1630, var_1472);
            var_1634 = wp::where(var_1619, var_1628, var_1473);
            var_1635 = wp::where(var_1619, var_1629, var_1474);
            var_1636 = wp::where(var_1619, var_1622, var_1475);
            var_1637 = wp::where(var_1619, var_1624, var_1504);
        }
        var_1638 = wp::where(var_1543, var_1631, var_1470);
        var_1639 = wp::where(var_1543, var_1632, var_1471);
        var_1640 = wp::where(var_1543, var_1633, var_1472);
        var_1641 = wp::where(var_1543, var_1634, var_1473);
        var_1642 = wp::where(var_1543, var_1635, var_1474);
        var_1643 = wp::where(var_1543, var_1636, var_1475);
        var_1644 = wp::where(var_1543, var_1637, var_1504);
        var_1645 = wp::where(var_1543, var_1611, var_1523);
        var_1646 = wp::where(var_1543, var_1545, var_1485);
        var_1647 = wp::where(var_1543, var_1606, var_1486);
        var_1648 = wp::where(var_1543, var_1607, var_1487);
        var_1649 = wp::where(var_1543, var_1608, var_1488);
        var_1650 = wp::where(var_1543, var_1609, var_1489);
        var_1651 = wp::where(var_1543, var_1579, var_1490);
        var_1652 = wp::where(var_1543, var_1616, var_1491);
        var_1653 = wp::where(var_1543, var_1617, var_1492);
    }
    var_1654 = wp::where(var_1502, var_1638, var_1470);
    var_1655 = wp::where(var_1502, var_1639, var_1471);
    var_1656 = wp::where(var_1502, var_1640, var_1472);
    var_1657 = wp::where(var_1502, var_1641, var_1473);
    var_1658 = wp::where(var_1502, var_1642, var_1474);
    var_1659 = wp::where(var_1502, var_1643, var_1475);
    var_1660 = wp::where(var_1502, var_1644, var_1476);
    var_1661 = wp::where(var_1502, var_1521, var_1477);
    var_1662 = wp::where(var_1502, var_1645, var_1478);
    var_1663 = wp::where(var_1502, var_1527, var_1479);
    var_1664 = wp::where(var_1502, var_1528, var_1480);
    var_1665 = wp::where(var_1502, var_1531, var_1481);
    var_1666 = wp::where(var_1502, var_1535, var_1482);
    var_1667 = wp::where(var_1502, var_1536, var_1483);
    var_1668 = wp::where(var_1502, var_1539, var_1484);
    var_1669 = wp::where(var_1502, var_1646, var_1485);
    var_1670 = wp::where(var_1502, var_1647, var_1486);
    var_1671 = wp::where(var_1502, var_1648, var_1487);
    var_1672 = wp::where(var_1502, var_1649, var_1488);
    var_1673 = wp::where(var_1502, var_1650, var_1489);
    var_1674 = wp::where(var_1502, var_1651, var_1490);
    var_1675 = wp::where(var_1502, var_1652, var_1491);
    var_1676 = wp::where(var_1502, var_1653, var_1492);
    var_1677 = wp::where(var_1502, var_1543, var_1502);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_1681 = wp::lshift(var_1680, var_1678);
    var_1682 = wp::bit_and(var_1309, var_1681);
    var_1684 = (var_1682 != var_1683);
    if (var_1684) {
        // continue                                                                           <L 1193>
    }
    var_1686 = wp::where(var_1684, var_1685, var_1679);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_1686) {
        var_1688 = wp::int32(var_1687);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_1690 = wp::bit_and(var_1309, var_1689);
        var_1693 = wp::where(var_1690, var_1691, var_1692);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_1695 = wp::bit_and(var_1309, var_1694);
        var_1698 = wp::where(var_1695, var_1696, var_1697);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_1700 = wp::bit_and(var_1309, var_1699);
        var_1703 = wp::where(var_1700, var_1701, var_1702);
        var_1704 = wp::vec_t<3, wp::float32>(var_1693, var_1698, var_1703);
        // box_size,                                                                          <L 1204>
        var_1705 = wp::cw_mul(var_1704, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_1705, var_1678, var_1706);
        // dif = box_pt - pos                                                                 <L 1209>
        var_1707 = wp::sub(var_1705, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_1708 = wp::extract(var_box_size, var_1678);
        var_1709 = wp::neg(var_1708);
        var_1710 = wp::extract(var_1707, var_1678);
        var_1711 = wp::mul(var_1709, var_1710);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_1712 = wp::dot(var_4, var_1707);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_1713 = wp::extract(var_box_size, var_1678);
        var_1714 = wp::extract(var_box_size, var_1678);
        var_1715 = wp::mul(var_1713, var_1714);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_1716 = wp::extract(var_box_size, var_1678);
        var_1717 = wp::neg(var_1716);
        var_1718 = wp::extract(var_4, var_1678);
        var_1719 = wp::mul(var_1717, var_1718);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_1720 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_1721 = wp::mul(var_1715, var_1720);
        var_1722 = wp::mul(var_1719, var_1719);
        var_1723 = wp::sub(var_1721, var_1722);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_1724 = wp::abs(var_1723);
        var_1725 = (var_1724 < var_275);
        if (var_1725) {
            // continue                                                                       <L 1218>
        }
        var_1727 = wp::where(var_1725, var_1726, var_1686);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_1727) {
            var_1729 = wp::div(var_1728, var_1723);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_1730 = wp::mul(var_1720, var_1711);
            var_1731 = wp::mul(var_1719, var_1712);
            var_1732 = wp::sub(var_1730, var_1731);
            var_1733 = wp::mul(var_1732, var_1729);
            var_1734 = wp::float32(var_1733);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_1735 = wp::mul(var_1715, var_1712);
            var_1736 = wp::mul(var_1719, var_1711);
            var_1737 = wp::sub(var_1735, var_1736);
            var_1738 = wp::mul(var_1737, var_1729);
            var_1739 = wp::float32(var_1738);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_1741 = wp::int32(var_1740);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_1743 = wp::int32(var_1742);
            // if x1 > 1:                                                                     <L 1229>
            var_1745 = (var_1734 > var_1744);
            if (var_1745) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_1748 = wp::sub(var_1712, var_1719);
                var_1749 = safe_div_0(var_1748, var_1720);
            }
            if (!var_1745) {
                // elif x1 < -1:                                                              <L 1233>
                var_1751 = (var_1734 < var_1750);
                if (var_1751) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_1754 = wp::add(var_1712, var_1719);
                    var_1755 = safe_div_0(var_1754, var_1720);
                }
                var_1756 = wp::where(var_1751, var_1752, var_1734);
                var_1757 = wp::where(var_1751, var_1755, var_1739);
                var_1758 = wp::where(var_1751, var_1753, var_1741);
            }
            var_1759 = wp::where(var_1745, var_1746, var_1756);
            var_1760 = wp::where(var_1745, var_1749, var_1757);
            var_1761 = wp::where(var_1745, var_1747, var_1758);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_1763 = (var_1760 > var_1762);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_1764 = var_1763;
            if (!var_1764) {
                var_1766 = (var_1760 < var_1765);
                var_1764 = var_1764 || var_1766;
            }
            if (var_1764) {
                // if x2_over:                                                                <L 1240>
                if (var_1763) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_1769 = wp::sub(var_1711, var_1719);
                    var_1770 = safe_div_0(var_1769, var_1715);
                }
                if (!var_1763) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_1773 = wp::add(var_1711, var_1719);
                    var_1774 = safe_div_0(var_1773, var_1715);
                }
                var_1775 = wp::where(var_1763, var_1770, var_1774);
                var_1776 = wp::where(var_1763, var_1767, var_1771);
                var_1777 = wp::where(var_1763, var_1768, var_1772);
                // if x1 > 1:                                                                 <L 1249>
                var_1779 = (var_1775 > var_1778);
                if (var_1779) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_1779) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_1783 = (var_1775 < var_1782);
                    if (var_1783) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_1786 = wp::where(var_1783, var_1784, var_1775);
                    var_1787 = wp::where(var_1783, var_1785, var_1761);
                }
                var_1788 = wp::where(var_1779, var_1780, var_1786);
                var_1789 = wp::where(var_1779, var_1781, var_1787);
            }
            var_1790 = wp::where(var_1764, var_1788, var_1759);
            var_1791 = wp::where(var_1764, var_1776, var_1760);
            var_1792 = wp::where(var_1764, var_1789, var_1761);
            var_1793 = wp::where(var_1764, var_1777, var_1743);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_1794 = wp::mul(var_4, var_1791);
            var_1795 = wp::sub(var_1707, var_1794);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_1796 = wp::extract(var_box_size, var_1678);
            var_1797 = wp::mul(var_1796, var_1790);
            wp::add_inplace(var_1795, var_1678, var_1797);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_1799 = wp::mul(var_1792, var_1798);
            var_1800 = wp::add(var_1799, var_1793);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_1801 = wp::length_sq(var_1795);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_1802 = wp::sub(var_1654, var_275);
            var_1803 = (var_1801 < var_1802);
            if (var_1803) {
                // bestdist = dif_sq                                                          <L 1264>
                var_1804 = wp::copy(var_1801);
                // bestsegmentpos = x2                                                        <L 1265>
                var_1805 = wp::copy(var_1791);
                // bestboxpos = x1                                                            <L 1266>
                var_1806 = wp::copy(var_1790);
                // c2 = ct // 6                                                               <L 1268>
                var_1808 = wp::floordiv(var_1800, var_1807);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_1810 = wp::lshift(var_1809, var_1678);
                var_1811 = wp::mul(var_1810, var_1808);
                var_1812 = wp::add(var_1309, var_1811);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_1813 = wp::copy(var_1678);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_1814 = wp::copy(var_1800);
            }
            var_1815 = wp::where(var_1803, var_1804, var_1654);
            var_1816 = wp::where(var_1803, var_1805, var_1655);
            var_1817 = wp::where(var_1803, var_1814, var_1656);
            var_1818 = wp::where(var_1803, var_1812, var_1657);
            var_1819 = wp::where(var_1803, var_1813, var_1658);
            var_1820 = wp::where(var_1803, var_1806, var_1659);
            var_1821 = wp::where(var_1803, var_1808, var_1688);
        }
        var_1822 = wp::where(var_1727, var_1815, var_1654);
        var_1823 = wp::where(var_1727, var_1816, var_1655);
        var_1824 = wp::where(var_1727, var_1817, var_1656);
        var_1825 = wp::where(var_1727, var_1818, var_1657);
        var_1826 = wp::where(var_1727, var_1819, var_1658);
        var_1827 = wp::where(var_1727, var_1820, var_1659);
        var_1828 = wp::where(var_1727, var_1821, var_1688);
        var_1829 = wp::where(var_1727, var_1795, var_1707);
        var_1830 = wp::where(var_1727, var_1729, var_1669);
        var_1831 = wp::where(var_1727, var_1790, var_1670);
        var_1832 = wp::where(var_1727, var_1791, var_1671);
        var_1833 = wp::where(var_1727, var_1792, var_1672);
        var_1834 = wp::where(var_1727, var_1793, var_1673);
        var_1835 = wp::where(var_1727, var_1763, var_1674);
        var_1836 = wp::where(var_1727, var_1800, var_1675);
        var_1837 = wp::where(var_1727, var_1801, var_1676);
    }
    var_1838 = wp::where(var_1686, var_1822, var_1654);
    var_1839 = wp::where(var_1686, var_1823, var_1655);
    var_1840 = wp::where(var_1686, var_1824, var_1656);
    var_1841 = wp::where(var_1686, var_1825, var_1657);
    var_1842 = wp::where(var_1686, var_1826, var_1658);
    var_1843 = wp::where(var_1686, var_1827, var_1659);
    var_1844 = wp::where(var_1686, var_1828, var_1660);
    var_1845 = wp::where(var_1686, var_1705, var_1661);
    var_1846 = wp::where(var_1686, var_1829, var_1662);
    var_1847 = wp::where(var_1686, var_1711, var_1663);
    var_1848 = wp::where(var_1686, var_1712, var_1664);
    var_1849 = wp::where(var_1686, var_1715, var_1665);
    var_1850 = wp::where(var_1686, var_1719, var_1666);
    var_1851 = wp::where(var_1686, var_1720, var_1667);
    var_1852 = wp::where(var_1686, var_1723, var_1668);
    var_1853 = wp::where(var_1686, var_1830, var_1669);
    var_1854 = wp::where(var_1686, var_1831, var_1670);
    var_1855 = wp::where(var_1686, var_1832, var_1671);
    var_1856 = wp::where(var_1686, var_1833, var_1672);
    var_1857 = wp::where(var_1686, var_1834, var_1673);
    var_1858 = wp::where(var_1686, var_1835, var_1674);
    var_1859 = wp::where(var_1686, var_1836, var_1675);
    var_1860 = wp::where(var_1686, var_1837, var_1676);
    var_1861 = wp::where(var_1686, var_1727, var_1686);
    // for j in range(3):                                                                     <L 1191>
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_1866 = wp::lshift(var_1865, var_1863);
    var_1867 = wp::bit_and(var_1862, var_1866);
    var_1869 = (var_1867 != var_1868);
    if (var_1869) {
        // continue                                                                           <L 1193>
    }
    var_1871 = wp::where(var_1869, var_1870, var_1864);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_1871) {
        var_1873 = wp::int32(var_1872);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_1875 = wp::bit_and(var_1862, var_1874);
        var_1878 = wp::where(var_1875, var_1876, var_1877);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_1880 = wp::bit_and(var_1862, var_1879);
        var_1883 = wp::where(var_1880, var_1881, var_1882);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_1885 = wp::bit_and(var_1862, var_1884);
        var_1888 = wp::where(var_1885, var_1886, var_1887);
        var_1889 = wp::vec_t<3, wp::float32>(var_1878, var_1883, var_1888);
        // box_size,                                                                          <L 1204>
        var_1890 = wp::cw_mul(var_1889, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_1890, var_1863, var_1891);
        // dif = box_pt - pos                                                                 <L 1209>
        var_1892 = wp::sub(var_1890, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_1893 = wp::extract(var_box_size, var_1863);
        var_1894 = wp::neg(var_1893);
        var_1895 = wp::extract(var_1892, var_1863);
        var_1896 = wp::mul(var_1894, var_1895);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_1897 = wp::dot(var_4, var_1892);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_1898 = wp::extract(var_box_size, var_1863);
        var_1899 = wp::extract(var_box_size, var_1863);
        var_1900 = wp::mul(var_1898, var_1899);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_1901 = wp::extract(var_box_size, var_1863);
        var_1902 = wp::neg(var_1901);
        var_1903 = wp::extract(var_4, var_1863);
        var_1904 = wp::mul(var_1902, var_1903);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_1905 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_1906 = wp::mul(var_1900, var_1905);
        var_1907 = wp::mul(var_1904, var_1904);
        var_1908 = wp::sub(var_1906, var_1907);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_1909 = wp::abs(var_1908);
        var_1910 = (var_1909 < var_275);
        if (var_1910) {
            // continue                                                                       <L 1218>
        }
        var_1912 = wp::where(var_1910, var_1911, var_1871);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_1912) {
            var_1914 = wp::div(var_1913, var_1908);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_1915 = wp::mul(var_1905, var_1896);
            var_1916 = wp::mul(var_1904, var_1897);
            var_1917 = wp::sub(var_1915, var_1916);
            var_1918 = wp::mul(var_1917, var_1914);
            var_1919 = wp::float32(var_1918);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_1920 = wp::mul(var_1900, var_1897);
            var_1921 = wp::mul(var_1904, var_1896);
            var_1922 = wp::sub(var_1920, var_1921);
            var_1923 = wp::mul(var_1922, var_1914);
            var_1924 = wp::float32(var_1923);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_1926 = wp::int32(var_1925);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_1928 = wp::int32(var_1927);
            // if x1 > 1:                                                                     <L 1229>
            var_1930 = (var_1919 > var_1929);
            if (var_1930) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_1933 = wp::sub(var_1897, var_1904);
                var_1934 = safe_div_0(var_1933, var_1905);
            }
            if (!var_1930) {
                // elif x1 < -1:                                                              <L 1233>
                var_1936 = (var_1919 < var_1935);
                if (var_1936) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_1939 = wp::add(var_1897, var_1904);
                    var_1940 = safe_div_0(var_1939, var_1905);
                }
                var_1941 = wp::where(var_1936, var_1937, var_1919);
                var_1942 = wp::where(var_1936, var_1940, var_1924);
                var_1943 = wp::where(var_1936, var_1938, var_1926);
            }
            var_1944 = wp::where(var_1930, var_1931, var_1941);
            var_1945 = wp::where(var_1930, var_1934, var_1942);
            var_1946 = wp::where(var_1930, var_1932, var_1943);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_1948 = (var_1945 > var_1947);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_1949 = var_1948;
            if (!var_1949) {
                var_1951 = (var_1945 < var_1950);
                var_1949 = var_1949 || var_1951;
            }
            if (var_1949) {
                // if x2_over:                                                                <L 1240>
                if (var_1948) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_1954 = wp::sub(var_1896, var_1904);
                    var_1955 = safe_div_0(var_1954, var_1900);
                }
                if (!var_1948) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_1958 = wp::add(var_1896, var_1904);
                    var_1959 = safe_div_0(var_1958, var_1900);
                }
                var_1960 = wp::where(var_1948, var_1955, var_1959);
                var_1961 = wp::where(var_1948, var_1952, var_1956);
                var_1962 = wp::where(var_1948, var_1953, var_1957);
                // if x1 > 1:                                                                 <L 1249>
                var_1964 = (var_1960 > var_1963);
                if (var_1964) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_1964) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_1968 = (var_1960 < var_1967);
                    if (var_1968) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_1971 = wp::where(var_1968, var_1969, var_1960);
                    var_1972 = wp::where(var_1968, var_1970, var_1946);
                }
                var_1973 = wp::where(var_1964, var_1965, var_1971);
                var_1974 = wp::where(var_1964, var_1966, var_1972);
            }
            var_1975 = wp::where(var_1949, var_1973, var_1944);
            var_1976 = wp::where(var_1949, var_1961, var_1945);
            var_1977 = wp::where(var_1949, var_1974, var_1946);
            var_1978 = wp::where(var_1949, var_1962, var_1928);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_1979 = wp::mul(var_4, var_1976);
            var_1980 = wp::sub(var_1892, var_1979);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_1981 = wp::extract(var_box_size, var_1863);
            var_1982 = wp::mul(var_1981, var_1975);
            wp::add_inplace(var_1980, var_1863, var_1982);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_1984 = wp::mul(var_1977, var_1983);
            var_1985 = wp::add(var_1984, var_1978);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_1986 = wp::length_sq(var_1980);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_1987 = wp::sub(var_1838, var_275);
            var_1988 = (var_1986 < var_1987);
            if (var_1988) {
                // bestdist = dif_sq                                                          <L 1264>
                var_1989 = wp::copy(var_1986);
                // bestsegmentpos = x2                                                        <L 1265>
                var_1990 = wp::copy(var_1976);
                // bestboxpos = x1                                                            <L 1266>
                var_1991 = wp::copy(var_1975);
                // c2 = ct // 6                                                               <L 1268>
                var_1993 = wp::floordiv(var_1985, var_1992);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_1995 = wp::lshift(var_1994, var_1863);
                var_1996 = wp::mul(var_1995, var_1993);
                var_1997 = wp::add(var_1862, var_1996);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_1998 = wp::copy(var_1863);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_1999 = wp::copy(var_1985);
            }
            var_2000 = wp::where(var_1988, var_1989, var_1838);
            var_2001 = wp::where(var_1988, var_1990, var_1839);
            var_2002 = wp::where(var_1988, var_1999, var_1840);
            var_2003 = wp::where(var_1988, var_1997, var_1841);
            var_2004 = wp::where(var_1988, var_1998, var_1842);
            var_2005 = wp::where(var_1988, var_1991, var_1843);
            var_2006 = wp::where(var_1988, var_1993, var_1873);
        }
        var_2007 = wp::where(var_1912, var_2000, var_1838);
        var_2008 = wp::where(var_1912, var_2001, var_1839);
        var_2009 = wp::where(var_1912, var_2002, var_1840);
        var_2010 = wp::where(var_1912, var_2003, var_1841);
        var_2011 = wp::where(var_1912, var_2004, var_1842);
        var_2012 = wp::where(var_1912, var_2005, var_1843);
        var_2013 = wp::where(var_1912, var_2006, var_1873);
        var_2014 = wp::where(var_1912, var_1980, var_1892);
        var_2015 = wp::where(var_1912, var_1914, var_1853);
        var_2016 = wp::where(var_1912, var_1975, var_1854);
        var_2017 = wp::where(var_1912, var_1976, var_1855);
        var_2018 = wp::where(var_1912, var_1977, var_1856);
        var_2019 = wp::where(var_1912, var_1978, var_1857);
        var_2020 = wp::where(var_1912, var_1948, var_1858);
        var_2021 = wp::where(var_1912, var_1985, var_1859);
        var_2022 = wp::where(var_1912, var_1986, var_1860);
    }
    var_2023 = wp::where(var_1871, var_2007, var_1838);
    var_2024 = wp::where(var_1871, var_2008, var_1839);
    var_2025 = wp::where(var_1871, var_2009, var_1840);
    var_2026 = wp::where(var_1871, var_2010, var_1841);
    var_2027 = wp::where(var_1871, var_2011, var_1842);
    var_2028 = wp::where(var_1871, var_2012, var_1843);
    var_2029 = wp::where(var_1871, var_2013, var_1844);
    var_2030 = wp::where(var_1871, var_1890, var_1845);
    var_2031 = wp::where(var_1871, var_2014, var_1846);
    var_2032 = wp::where(var_1871, var_1896, var_1847);
    var_2033 = wp::where(var_1871, var_1897, var_1848);
    var_2034 = wp::where(var_1871, var_1900, var_1849);
    var_2035 = wp::where(var_1871, var_1904, var_1850);
    var_2036 = wp::where(var_1871, var_1905, var_1851);
    var_2037 = wp::where(var_1871, var_1908, var_1852);
    var_2038 = wp::where(var_1871, var_2015, var_1853);
    var_2039 = wp::where(var_1871, var_2016, var_1854);
    var_2040 = wp::where(var_1871, var_2017, var_1855);
    var_2041 = wp::where(var_1871, var_2018, var_1856);
    var_2042 = wp::where(var_1871, var_2019, var_1857);
    var_2043 = wp::where(var_1871, var_2020, var_1858);
    var_2044 = wp::where(var_1871, var_2021, var_1859);
    var_2045 = wp::where(var_1871, var_2022, var_1860);
    var_2046 = wp::where(var_1871, var_1912, var_1871);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_2050 = wp::lshift(var_2049, var_2047);
    var_2051 = wp::bit_and(var_1862, var_2050);
    var_2053 = (var_2051 != var_2052);
    if (var_2053) {
        // continue                                                                           <L 1193>
    }
    var_2055 = wp::where(var_2053, var_2054, var_2048);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_2055) {
        var_2057 = wp::int32(var_2056);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_2059 = wp::bit_and(var_1862, var_2058);
        var_2062 = wp::where(var_2059, var_2060, var_2061);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_2064 = wp::bit_and(var_1862, var_2063);
        var_2067 = wp::where(var_2064, var_2065, var_2066);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_2069 = wp::bit_and(var_1862, var_2068);
        var_2072 = wp::where(var_2069, var_2070, var_2071);
        var_2073 = wp::vec_t<3, wp::float32>(var_2062, var_2067, var_2072);
        // box_size,                                                                          <L 1204>
        var_2074 = wp::cw_mul(var_2073, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_2074, var_2047, var_2075);
        // dif = box_pt - pos                                                                 <L 1209>
        var_2076 = wp::sub(var_2074, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_2077 = wp::extract(var_box_size, var_2047);
        var_2078 = wp::neg(var_2077);
        var_2079 = wp::extract(var_2076, var_2047);
        var_2080 = wp::mul(var_2078, var_2079);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_2081 = wp::dot(var_4, var_2076);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_2082 = wp::extract(var_box_size, var_2047);
        var_2083 = wp::extract(var_box_size, var_2047);
        var_2084 = wp::mul(var_2082, var_2083);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_2085 = wp::extract(var_box_size, var_2047);
        var_2086 = wp::neg(var_2085);
        var_2087 = wp::extract(var_4, var_2047);
        var_2088 = wp::mul(var_2086, var_2087);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_2089 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_2090 = wp::mul(var_2084, var_2089);
        var_2091 = wp::mul(var_2088, var_2088);
        var_2092 = wp::sub(var_2090, var_2091);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_2093 = wp::abs(var_2092);
        var_2094 = (var_2093 < var_275);
        if (var_2094) {
            // continue                                                                       <L 1218>
        }
        var_2096 = wp::where(var_2094, var_2095, var_2055);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_2096) {
            var_2098 = wp::div(var_2097, var_2092);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_2099 = wp::mul(var_2089, var_2080);
            var_2100 = wp::mul(var_2088, var_2081);
            var_2101 = wp::sub(var_2099, var_2100);
            var_2102 = wp::mul(var_2101, var_2098);
            var_2103 = wp::float32(var_2102);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_2104 = wp::mul(var_2084, var_2081);
            var_2105 = wp::mul(var_2088, var_2080);
            var_2106 = wp::sub(var_2104, var_2105);
            var_2107 = wp::mul(var_2106, var_2098);
            var_2108 = wp::float32(var_2107);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_2110 = wp::int32(var_2109);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_2112 = wp::int32(var_2111);
            // if x1 > 1:                                                                     <L 1229>
            var_2114 = (var_2103 > var_2113);
            if (var_2114) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_2117 = wp::sub(var_2081, var_2088);
                var_2118 = safe_div_0(var_2117, var_2089);
            }
            if (!var_2114) {
                // elif x1 < -1:                                                              <L 1233>
                var_2120 = (var_2103 < var_2119);
                if (var_2120) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_2123 = wp::add(var_2081, var_2088);
                    var_2124 = safe_div_0(var_2123, var_2089);
                }
                var_2125 = wp::where(var_2120, var_2121, var_2103);
                var_2126 = wp::where(var_2120, var_2124, var_2108);
                var_2127 = wp::where(var_2120, var_2122, var_2110);
            }
            var_2128 = wp::where(var_2114, var_2115, var_2125);
            var_2129 = wp::where(var_2114, var_2118, var_2126);
            var_2130 = wp::where(var_2114, var_2116, var_2127);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_2132 = (var_2129 > var_2131);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_2133 = var_2132;
            if (!var_2133) {
                var_2135 = (var_2129 < var_2134);
                var_2133 = var_2133 || var_2135;
            }
            if (var_2133) {
                // if x2_over:                                                                <L 1240>
                if (var_2132) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_2138 = wp::sub(var_2080, var_2088);
                    var_2139 = safe_div_0(var_2138, var_2084);
                }
                if (!var_2132) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_2142 = wp::add(var_2080, var_2088);
                    var_2143 = safe_div_0(var_2142, var_2084);
                }
                var_2144 = wp::where(var_2132, var_2139, var_2143);
                var_2145 = wp::where(var_2132, var_2136, var_2140);
                var_2146 = wp::where(var_2132, var_2137, var_2141);
                // if x1 > 1:                                                                 <L 1249>
                var_2148 = (var_2144 > var_2147);
                if (var_2148) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_2148) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_2152 = (var_2144 < var_2151);
                    if (var_2152) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_2155 = wp::where(var_2152, var_2153, var_2144);
                    var_2156 = wp::where(var_2152, var_2154, var_2130);
                }
                var_2157 = wp::where(var_2148, var_2149, var_2155);
                var_2158 = wp::where(var_2148, var_2150, var_2156);
            }
            var_2159 = wp::where(var_2133, var_2157, var_2128);
            var_2160 = wp::where(var_2133, var_2145, var_2129);
            var_2161 = wp::where(var_2133, var_2158, var_2130);
            var_2162 = wp::where(var_2133, var_2146, var_2112);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_2163 = wp::mul(var_4, var_2160);
            var_2164 = wp::sub(var_2076, var_2163);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_2165 = wp::extract(var_box_size, var_2047);
            var_2166 = wp::mul(var_2165, var_2159);
            wp::add_inplace(var_2164, var_2047, var_2166);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_2168 = wp::mul(var_2161, var_2167);
            var_2169 = wp::add(var_2168, var_2162);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_2170 = wp::length_sq(var_2164);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_2171 = wp::sub(var_2023, var_275);
            var_2172 = (var_2170 < var_2171);
            if (var_2172) {
                // bestdist = dif_sq                                                          <L 1264>
                var_2173 = wp::copy(var_2170);
                // bestsegmentpos = x2                                                        <L 1265>
                var_2174 = wp::copy(var_2160);
                // bestboxpos = x1                                                            <L 1266>
                var_2175 = wp::copy(var_2159);
                // c2 = ct // 6                                                               <L 1268>
                var_2177 = wp::floordiv(var_2169, var_2176);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_2179 = wp::lshift(var_2178, var_2047);
                var_2180 = wp::mul(var_2179, var_2177);
                var_2181 = wp::add(var_1862, var_2180);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_2182 = wp::copy(var_2047);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_2183 = wp::copy(var_2169);
            }
            var_2184 = wp::where(var_2172, var_2173, var_2023);
            var_2185 = wp::where(var_2172, var_2174, var_2024);
            var_2186 = wp::where(var_2172, var_2183, var_2025);
            var_2187 = wp::where(var_2172, var_2181, var_2026);
            var_2188 = wp::where(var_2172, var_2182, var_2027);
            var_2189 = wp::where(var_2172, var_2175, var_2028);
            var_2190 = wp::where(var_2172, var_2177, var_2057);
        }
        var_2191 = wp::where(var_2096, var_2184, var_2023);
        var_2192 = wp::where(var_2096, var_2185, var_2024);
        var_2193 = wp::where(var_2096, var_2186, var_2025);
        var_2194 = wp::where(var_2096, var_2187, var_2026);
        var_2195 = wp::where(var_2096, var_2188, var_2027);
        var_2196 = wp::where(var_2096, var_2189, var_2028);
        var_2197 = wp::where(var_2096, var_2190, var_2057);
        var_2198 = wp::where(var_2096, var_2164, var_2076);
        var_2199 = wp::where(var_2096, var_2098, var_2038);
        var_2200 = wp::where(var_2096, var_2159, var_2039);
        var_2201 = wp::where(var_2096, var_2160, var_2040);
        var_2202 = wp::where(var_2096, var_2161, var_2041);
        var_2203 = wp::where(var_2096, var_2162, var_2042);
        var_2204 = wp::where(var_2096, var_2132, var_2043);
        var_2205 = wp::where(var_2096, var_2169, var_2044);
        var_2206 = wp::where(var_2096, var_2170, var_2045);
    }
    var_2207 = wp::where(var_2055, var_2191, var_2023);
    var_2208 = wp::where(var_2055, var_2192, var_2024);
    var_2209 = wp::where(var_2055, var_2193, var_2025);
    var_2210 = wp::where(var_2055, var_2194, var_2026);
    var_2211 = wp::where(var_2055, var_2195, var_2027);
    var_2212 = wp::where(var_2055, var_2196, var_2028);
    var_2213 = wp::where(var_2055, var_2197, var_2029);
    var_2214 = wp::where(var_2055, var_2074, var_2030);
    var_2215 = wp::where(var_2055, var_2198, var_2031);
    var_2216 = wp::where(var_2055, var_2080, var_2032);
    var_2217 = wp::where(var_2055, var_2081, var_2033);
    var_2218 = wp::where(var_2055, var_2084, var_2034);
    var_2219 = wp::where(var_2055, var_2088, var_2035);
    var_2220 = wp::where(var_2055, var_2089, var_2036);
    var_2221 = wp::where(var_2055, var_2092, var_2037);
    var_2222 = wp::where(var_2055, var_2199, var_2038);
    var_2223 = wp::where(var_2055, var_2200, var_2039);
    var_2224 = wp::where(var_2055, var_2201, var_2040);
    var_2225 = wp::where(var_2055, var_2202, var_2041);
    var_2226 = wp::where(var_2055, var_2203, var_2042);
    var_2227 = wp::where(var_2055, var_2204, var_2043);
    var_2228 = wp::where(var_2055, var_2205, var_2044);
    var_2229 = wp::where(var_2055, var_2206, var_2045);
    var_2230 = wp::where(var_2055, var_2096, var_2055);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_2234 = wp::lshift(var_2233, var_2231);
    var_2235 = wp::bit_and(var_1862, var_2234);
    var_2237 = (var_2235 != var_2236);
    if (var_2237) {
        // continue                                                                           <L 1193>
    }
    var_2239 = wp::where(var_2237, var_2238, var_2232);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_2239) {
        var_2241 = wp::int32(var_2240);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_2243 = wp::bit_and(var_1862, var_2242);
        var_2246 = wp::where(var_2243, var_2244, var_2245);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_2248 = wp::bit_and(var_1862, var_2247);
        var_2251 = wp::where(var_2248, var_2249, var_2250);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_2253 = wp::bit_and(var_1862, var_2252);
        var_2256 = wp::where(var_2253, var_2254, var_2255);
        var_2257 = wp::vec_t<3, wp::float32>(var_2246, var_2251, var_2256);
        // box_size,                                                                          <L 1204>
        var_2258 = wp::cw_mul(var_2257, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_2258, var_2231, var_2259);
        // dif = box_pt - pos                                                                 <L 1209>
        var_2260 = wp::sub(var_2258, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_2261 = wp::extract(var_box_size, var_2231);
        var_2262 = wp::neg(var_2261);
        var_2263 = wp::extract(var_2260, var_2231);
        var_2264 = wp::mul(var_2262, var_2263);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_2265 = wp::dot(var_4, var_2260);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_2266 = wp::extract(var_box_size, var_2231);
        var_2267 = wp::extract(var_box_size, var_2231);
        var_2268 = wp::mul(var_2266, var_2267);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_2269 = wp::extract(var_box_size, var_2231);
        var_2270 = wp::neg(var_2269);
        var_2271 = wp::extract(var_4, var_2231);
        var_2272 = wp::mul(var_2270, var_2271);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_2273 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_2274 = wp::mul(var_2268, var_2273);
        var_2275 = wp::mul(var_2272, var_2272);
        var_2276 = wp::sub(var_2274, var_2275);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_2277 = wp::abs(var_2276);
        var_2278 = (var_2277 < var_275);
        if (var_2278) {
            // continue                                                                       <L 1218>
        }
        var_2280 = wp::where(var_2278, var_2279, var_2239);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_2280) {
            var_2282 = wp::div(var_2281, var_2276);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_2283 = wp::mul(var_2273, var_2264);
            var_2284 = wp::mul(var_2272, var_2265);
            var_2285 = wp::sub(var_2283, var_2284);
            var_2286 = wp::mul(var_2285, var_2282);
            var_2287 = wp::float32(var_2286);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_2288 = wp::mul(var_2268, var_2265);
            var_2289 = wp::mul(var_2272, var_2264);
            var_2290 = wp::sub(var_2288, var_2289);
            var_2291 = wp::mul(var_2290, var_2282);
            var_2292 = wp::float32(var_2291);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_2294 = wp::int32(var_2293);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_2296 = wp::int32(var_2295);
            // if x1 > 1:                                                                     <L 1229>
            var_2298 = (var_2287 > var_2297);
            if (var_2298) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_2301 = wp::sub(var_2265, var_2272);
                var_2302 = safe_div_0(var_2301, var_2273);
            }
            if (!var_2298) {
                // elif x1 < -1:                                                              <L 1233>
                var_2304 = (var_2287 < var_2303);
                if (var_2304) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_2307 = wp::add(var_2265, var_2272);
                    var_2308 = safe_div_0(var_2307, var_2273);
                }
                var_2309 = wp::where(var_2304, var_2305, var_2287);
                var_2310 = wp::where(var_2304, var_2308, var_2292);
                var_2311 = wp::where(var_2304, var_2306, var_2294);
            }
            var_2312 = wp::where(var_2298, var_2299, var_2309);
            var_2313 = wp::where(var_2298, var_2302, var_2310);
            var_2314 = wp::where(var_2298, var_2300, var_2311);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_2316 = (var_2313 > var_2315);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_2317 = var_2316;
            if (!var_2317) {
                var_2319 = (var_2313 < var_2318);
                var_2317 = var_2317 || var_2319;
            }
            if (var_2317) {
                // if x2_over:                                                                <L 1240>
                if (var_2316) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_2322 = wp::sub(var_2264, var_2272);
                    var_2323 = safe_div_0(var_2322, var_2268);
                }
                if (!var_2316) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_2326 = wp::add(var_2264, var_2272);
                    var_2327 = safe_div_0(var_2326, var_2268);
                }
                var_2328 = wp::where(var_2316, var_2323, var_2327);
                var_2329 = wp::where(var_2316, var_2320, var_2324);
                var_2330 = wp::where(var_2316, var_2321, var_2325);
                // if x1 > 1:                                                                 <L 1249>
                var_2332 = (var_2328 > var_2331);
                if (var_2332) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_2332) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_2336 = (var_2328 < var_2335);
                    if (var_2336) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_2339 = wp::where(var_2336, var_2337, var_2328);
                    var_2340 = wp::where(var_2336, var_2338, var_2314);
                }
                var_2341 = wp::where(var_2332, var_2333, var_2339);
                var_2342 = wp::where(var_2332, var_2334, var_2340);
            }
            var_2343 = wp::where(var_2317, var_2341, var_2312);
            var_2344 = wp::where(var_2317, var_2329, var_2313);
            var_2345 = wp::where(var_2317, var_2342, var_2314);
            var_2346 = wp::where(var_2317, var_2330, var_2296);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_2347 = wp::mul(var_4, var_2344);
            var_2348 = wp::sub(var_2260, var_2347);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_2349 = wp::extract(var_box_size, var_2231);
            var_2350 = wp::mul(var_2349, var_2343);
            wp::add_inplace(var_2348, var_2231, var_2350);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_2352 = wp::mul(var_2345, var_2351);
            var_2353 = wp::add(var_2352, var_2346);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_2354 = wp::length_sq(var_2348);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_2355 = wp::sub(var_2207, var_275);
            var_2356 = (var_2354 < var_2355);
            if (var_2356) {
                // bestdist = dif_sq                                                          <L 1264>
                var_2357 = wp::copy(var_2354);
                // bestsegmentpos = x2                                                        <L 1265>
                var_2358 = wp::copy(var_2344);
                // bestboxpos = x1                                                            <L 1266>
                var_2359 = wp::copy(var_2343);
                // c2 = ct // 6                                                               <L 1268>
                var_2361 = wp::floordiv(var_2353, var_2360);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_2363 = wp::lshift(var_2362, var_2231);
                var_2364 = wp::mul(var_2363, var_2361);
                var_2365 = wp::add(var_1862, var_2364);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_2366 = wp::copy(var_2231);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_2367 = wp::copy(var_2353);
            }
            var_2368 = wp::where(var_2356, var_2357, var_2207);
            var_2369 = wp::where(var_2356, var_2358, var_2208);
            var_2370 = wp::where(var_2356, var_2367, var_2209);
            var_2371 = wp::where(var_2356, var_2365, var_2210);
            var_2372 = wp::where(var_2356, var_2366, var_2211);
            var_2373 = wp::where(var_2356, var_2359, var_2212);
            var_2374 = wp::where(var_2356, var_2361, var_2241);
        }
        var_2375 = wp::where(var_2280, var_2368, var_2207);
        var_2376 = wp::where(var_2280, var_2369, var_2208);
        var_2377 = wp::where(var_2280, var_2370, var_2209);
        var_2378 = wp::where(var_2280, var_2371, var_2210);
        var_2379 = wp::where(var_2280, var_2372, var_2211);
        var_2380 = wp::where(var_2280, var_2373, var_2212);
        var_2381 = wp::where(var_2280, var_2374, var_2241);
        var_2382 = wp::where(var_2280, var_2348, var_2260);
        var_2383 = wp::where(var_2280, var_2282, var_2222);
        var_2384 = wp::where(var_2280, var_2343, var_2223);
        var_2385 = wp::where(var_2280, var_2344, var_2224);
        var_2386 = wp::where(var_2280, var_2345, var_2225);
        var_2387 = wp::where(var_2280, var_2346, var_2226);
        var_2388 = wp::where(var_2280, var_2316, var_2227);
        var_2389 = wp::where(var_2280, var_2353, var_2228);
        var_2390 = wp::where(var_2280, var_2354, var_2229);
    }
    var_2391 = wp::where(var_2239, var_2375, var_2207);
    var_2392 = wp::where(var_2239, var_2376, var_2208);
    var_2393 = wp::where(var_2239, var_2377, var_2209);
    var_2394 = wp::where(var_2239, var_2378, var_2210);
    var_2395 = wp::where(var_2239, var_2379, var_2211);
    var_2396 = wp::where(var_2239, var_2380, var_2212);
    var_2397 = wp::where(var_2239, var_2381, var_2213);
    var_2398 = wp::where(var_2239, var_2258, var_2214);
    var_2399 = wp::where(var_2239, var_2382, var_2215);
    var_2400 = wp::where(var_2239, var_2264, var_2216);
    var_2401 = wp::where(var_2239, var_2265, var_2217);
    var_2402 = wp::where(var_2239, var_2268, var_2218);
    var_2403 = wp::where(var_2239, var_2272, var_2219);
    var_2404 = wp::where(var_2239, var_2273, var_2220);
    var_2405 = wp::where(var_2239, var_2276, var_2221);
    var_2406 = wp::where(var_2239, var_2383, var_2222);
    var_2407 = wp::where(var_2239, var_2384, var_2223);
    var_2408 = wp::where(var_2239, var_2385, var_2224);
    var_2409 = wp::where(var_2239, var_2386, var_2225);
    var_2410 = wp::where(var_2239, var_2387, var_2226);
    var_2411 = wp::where(var_2239, var_2388, var_2227);
    var_2412 = wp::where(var_2239, var_2389, var_2228);
    var_2413 = wp::where(var_2239, var_2390, var_2229);
    var_2414 = wp::where(var_2239, var_2280, var_2239);
    // for j in range(3):                                                                     <L 1191>
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_2419 = wp::lshift(var_2418, var_2416);
    var_2420 = wp::bit_and(var_2415, var_2419);
    var_2422 = (var_2420 != var_2421);
    if (var_2422) {
        // continue                                                                           <L 1193>
    }
    var_2424 = wp::where(var_2422, var_2423, var_2417);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_2424) {
        var_2426 = wp::int32(var_2425);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_2428 = wp::bit_and(var_2415, var_2427);
        var_2431 = wp::where(var_2428, var_2429, var_2430);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_2433 = wp::bit_and(var_2415, var_2432);
        var_2436 = wp::where(var_2433, var_2434, var_2435);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_2438 = wp::bit_and(var_2415, var_2437);
        var_2441 = wp::where(var_2438, var_2439, var_2440);
        var_2442 = wp::vec_t<3, wp::float32>(var_2431, var_2436, var_2441);
        // box_size,                                                                          <L 1204>
        var_2443 = wp::cw_mul(var_2442, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_2443, var_2416, var_2444);
        // dif = box_pt - pos                                                                 <L 1209>
        var_2445 = wp::sub(var_2443, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_2446 = wp::extract(var_box_size, var_2416);
        var_2447 = wp::neg(var_2446);
        var_2448 = wp::extract(var_2445, var_2416);
        var_2449 = wp::mul(var_2447, var_2448);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_2450 = wp::dot(var_4, var_2445);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_2451 = wp::extract(var_box_size, var_2416);
        var_2452 = wp::extract(var_box_size, var_2416);
        var_2453 = wp::mul(var_2451, var_2452);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_2454 = wp::extract(var_box_size, var_2416);
        var_2455 = wp::neg(var_2454);
        var_2456 = wp::extract(var_4, var_2416);
        var_2457 = wp::mul(var_2455, var_2456);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_2458 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_2459 = wp::mul(var_2453, var_2458);
        var_2460 = wp::mul(var_2457, var_2457);
        var_2461 = wp::sub(var_2459, var_2460);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_2462 = wp::abs(var_2461);
        var_2463 = (var_2462 < var_275);
        if (var_2463) {
            // continue                                                                       <L 1218>
        }
        var_2465 = wp::where(var_2463, var_2464, var_2424);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_2465) {
            var_2467 = wp::div(var_2466, var_2461);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_2468 = wp::mul(var_2458, var_2449);
            var_2469 = wp::mul(var_2457, var_2450);
            var_2470 = wp::sub(var_2468, var_2469);
            var_2471 = wp::mul(var_2470, var_2467);
            var_2472 = wp::float32(var_2471);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_2473 = wp::mul(var_2453, var_2450);
            var_2474 = wp::mul(var_2457, var_2449);
            var_2475 = wp::sub(var_2473, var_2474);
            var_2476 = wp::mul(var_2475, var_2467);
            var_2477 = wp::float32(var_2476);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_2479 = wp::int32(var_2478);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_2481 = wp::int32(var_2480);
            // if x1 > 1:                                                                     <L 1229>
            var_2483 = (var_2472 > var_2482);
            if (var_2483) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_2486 = wp::sub(var_2450, var_2457);
                var_2487 = safe_div_0(var_2486, var_2458);
            }
            if (!var_2483) {
                // elif x1 < -1:                                                              <L 1233>
                var_2489 = (var_2472 < var_2488);
                if (var_2489) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_2492 = wp::add(var_2450, var_2457);
                    var_2493 = safe_div_0(var_2492, var_2458);
                }
                var_2494 = wp::where(var_2489, var_2490, var_2472);
                var_2495 = wp::where(var_2489, var_2493, var_2477);
                var_2496 = wp::where(var_2489, var_2491, var_2479);
            }
            var_2497 = wp::where(var_2483, var_2484, var_2494);
            var_2498 = wp::where(var_2483, var_2487, var_2495);
            var_2499 = wp::where(var_2483, var_2485, var_2496);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_2501 = (var_2498 > var_2500);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_2502 = var_2501;
            if (!var_2502) {
                var_2504 = (var_2498 < var_2503);
                var_2502 = var_2502 || var_2504;
            }
            if (var_2502) {
                // if x2_over:                                                                <L 1240>
                if (var_2501) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_2507 = wp::sub(var_2449, var_2457);
                    var_2508 = safe_div_0(var_2507, var_2453);
                }
                if (!var_2501) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_2511 = wp::add(var_2449, var_2457);
                    var_2512 = safe_div_0(var_2511, var_2453);
                }
                var_2513 = wp::where(var_2501, var_2508, var_2512);
                var_2514 = wp::where(var_2501, var_2505, var_2509);
                var_2515 = wp::where(var_2501, var_2506, var_2510);
                // if x1 > 1:                                                                 <L 1249>
                var_2517 = (var_2513 > var_2516);
                if (var_2517) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_2517) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_2521 = (var_2513 < var_2520);
                    if (var_2521) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_2524 = wp::where(var_2521, var_2522, var_2513);
                    var_2525 = wp::where(var_2521, var_2523, var_2499);
                }
                var_2526 = wp::where(var_2517, var_2518, var_2524);
                var_2527 = wp::where(var_2517, var_2519, var_2525);
            }
            var_2528 = wp::where(var_2502, var_2526, var_2497);
            var_2529 = wp::where(var_2502, var_2514, var_2498);
            var_2530 = wp::where(var_2502, var_2527, var_2499);
            var_2531 = wp::where(var_2502, var_2515, var_2481);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_2532 = wp::mul(var_4, var_2529);
            var_2533 = wp::sub(var_2445, var_2532);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_2534 = wp::extract(var_box_size, var_2416);
            var_2535 = wp::mul(var_2534, var_2528);
            wp::add_inplace(var_2533, var_2416, var_2535);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_2537 = wp::mul(var_2530, var_2536);
            var_2538 = wp::add(var_2537, var_2531);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_2539 = wp::length_sq(var_2533);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_2540 = wp::sub(var_2391, var_275);
            var_2541 = (var_2539 < var_2540);
            if (var_2541) {
                // bestdist = dif_sq                                                          <L 1264>
                var_2542 = wp::copy(var_2539);
                // bestsegmentpos = x2                                                        <L 1265>
                var_2543 = wp::copy(var_2529);
                // bestboxpos = x1                                                            <L 1266>
                var_2544 = wp::copy(var_2528);
                // c2 = ct // 6                                                               <L 1268>
                var_2546 = wp::floordiv(var_2538, var_2545);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_2548 = wp::lshift(var_2547, var_2416);
                var_2549 = wp::mul(var_2548, var_2546);
                var_2550 = wp::add(var_2415, var_2549);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_2551 = wp::copy(var_2416);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_2552 = wp::copy(var_2538);
            }
            var_2553 = wp::where(var_2541, var_2542, var_2391);
            var_2554 = wp::where(var_2541, var_2543, var_2392);
            var_2555 = wp::where(var_2541, var_2552, var_2393);
            var_2556 = wp::where(var_2541, var_2550, var_2394);
            var_2557 = wp::where(var_2541, var_2551, var_2395);
            var_2558 = wp::where(var_2541, var_2544, var_2396);
            var_2559 = wp::where(var_2541, var_2546, var_2426);
        }
        var_2560 = wp::where(var_2465, var_2553, var_2391);
        var_2561 = wp::where(var_2465, var_2554, var_2392);
        var_2562 = wp::where(var_2465, var_2555, var_2393);
        var_2563 = wp::where(var_2465, var_2556, var_2394);
        var_2564 = wp::where(var_2465, var_2557, var_2395);
        var_2565 = wp::where(var_2465, var_2558, var_2396);
        var_2566 = wp::where(var_2465, var_2559, var_2426);
        var_2567 = wp::where(var_2465, var_2533, var_2445);
        var_2568 = wp::where(var_2465, var_2467, var_2406);
        var_2569 = wp::where(var_2465, var_2528, var_2407);
        var_2570 = wp::where(var_2465, var_2529, var_2408);
        var_2571 = wp::where(var_2465, var_2530, var_2409);
        var_2572 = wp::where(var_2465, var_2531, var_2410);
        var_2573 = wp::where(var_2465, var_2501, var_2411);
        var_2574 = wp::where(var_2465, var_2538, var_2412);
        var_2575 = wp::where(var_2465, var_2539, var_2413);
    }
    var_2576 = wp::where(var_2424, var_2560, var_2391);
    var_2577 = wp::where(var_2424, var_2561, var_2392);
    var_2578 = wp::where(var_2424, var_2562, var_2393);
    var_2579 = wp::where(var_2424, var_2563, var_2394);
    var_2580 = wp::where(var_2424, var_2564, var_2395);
    var_2581 = wp::where(var_2424, var_2565, var_2396);
    var_2582 = wp::where(var_2424, var_2566, var_2397);
    var_2583 = wp::where(var_2424, var_2443, var_2398);
    var_2584 = wp::where(var_2424, var_2567, var_2399);
    var_2585 = wp::where(var_2424, var_2449, var_2400);
    var_2586 = wp::where(var_2424, var_2450, var_2401);
    var_2587 = wp::where(var_2424, var_2453, var_2402);
    var_2588 = wp::where(var_2424, var_2457, var_2403);
    var_2589 = wp::where(var_2424, var_2458, var_2404);
    var_2590 = wp::where(var_2424, var_2461, var_2405);
    var_2591 = wp::where(var_2424, var_2568, var_2406);
    var_2592 = wp::where(var_2424, var_2569, var_2407);
    var_2593 = wp::where(var_2424, var_2570, var_2408);
    var_2594 = wp::where(var_2424, var_2571, var_2409);
    var_2595 = wp::where(var_2424, var_2572, var_2410);
    var_2596 = wp::where(var_2424, var_2573, var_2411);
    var_2597 = wp::where(var_2424, var_2574, var_2412);
    var_2598 = wp::where(var_2424, var_2575, var_2413);
    var_2599 = wp::where(var_2424, var_2465, var_2424);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_2603 = wp::lshift(var_2602, var_2600);
    var_2604 = wp::bit_and(var_2415, var_2603);
    var_2606 = (var_2604 != var_2605);
    if (var_2606) {
        // continue                                                                           <L 1193>
    }
    var_2608 = wp::where(var_2606, var_2607, var_2601);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_2608) {
        var_2610 = wp::int32(var_2609);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_2612 = wp::bit_and(var_2415, var_2611);
        var_2615 = wp::where(var_2612, var_2613, var_2614);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_2617 = wp::bit_and(var_2415, var_2616);
        var_2620 = wp::where(var_2617, var_2618, var_2619);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_2622 = wp::bit_and(var_2415, var_2621);
        var_2625 = wp::where(var_2622, var_2623, var_2624);
        var_2626 = wp::vec_t<3, wp::float32>(var_2615, var_2620, var_2625);
        // box_size,                                                                          <L 1204>
        var_2627 = wp::cw_mul(var_2626, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_2627, var_2600, var_2628);
        // dif = box_pt - pos                                                                 <L 1209>
        var_2629 = wp::sub(var_2627, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_2630 = wp::extract(var_box_size, var_2600);
        var_2631 = wp::neg(var_2630);
        var_2632 = wp::extract(var_2629, var_2600);
        var_2633 = wp::mul(var_2631, var_2632);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_2634 = wp::dot(var_4, var_2629);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_2635 = wp::extract(var_box_size, var_2600);
        var_2636 = wp::extract(var_box_size, var_2600);
        var_2637 = wp::mul(var_2635, var_2636);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_2638 = wp::extract(var_box_size, var_2600);
        var_2639 = wp::neg(var_2638);
        var_2640 = wp::extract(var_4, var_2600);
        var_2641 = wp::mul(var_2639, var_2640);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_2642 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_2643 = wp::mul(var_2637, var_2642);
        var_2644 = wp::mul(var_2641, var_2641);
        var_2645 = wp::sub(var_2643, var_2644);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_2646 = wp::abs(var_2645);
        var_2647 = (var_2646 < var_275);
        if (var_2647) {
            // continue                                                                       <L 1218>
        }
        var_2649 = wp::where(var_2647, var_2648, var_2608);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_2649) {
            var_2651 = wp::div(var_2650, var_2645);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_2652 = wp::mul(var_2642, var_2633);
            var_2653 = wp::mul(var_2641, var_2634);
            var_2654 = wp::sub(var_2652, var_2653);
            var_2655 = wp::mul(var_2654, var_2651);
            var_2656 = wp::float32(var_2655);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_2657 = wp::mul(var_2637, var_2634);
            var_2658 = wp::mul(var_2641, var_2633);
            var_2659 = wp::sub(var_2657, var_2658);
            var_2660 = wp::mul(var_2659, var_2651);
            var_2661 = wp::float32(var_2660);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_2663 = wp::int32(var_2662);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_2665 = wp::int32(var_2664);
            // if x1 > 1:                                                                     <L 1229>
            var_2667 = (var_2656 > var_2666);
            if (var_2667) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_2670 = wp::sub(var_2634, var_2641);
                var_2671 = safe_div_0(var_2670, var_2642);
            }
            if (!var_2667) {
                // elif x1 < -1:                                                              <L 1233>
                var_2673 = (var_2656 < var_2672);
                if (var_2673) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_2676 = wp::add(var_2634, var_2641);
                    var_2677 = safe_div_0(var_2676, var_2642);
                }
                var_2678 = wp::where(var_2673, var_2674, var_2656);
                var_2679 = wp::where(var_2673, var_2677, var_2661);
                var_2680 = wp::where(var_2673, var_2675, var_2663);
            }
            var_2681 = wp::where(var_2667, var_2668, var_2678);
            var_2682 = wp::where(var_2667, var_2671, var_2679);
            var_2683 = wp::where(var_2667, var_2669, var_2680);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_2685 = (var_2682 > var_2684);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_2686 = var_2685;
            if (!var_2686) {
                var_2688 = (var_2682 < var_2687);
                var_2686 = var_2686 || var_2688;
            }
            if (var_2686) {
                // if x2_over:                                                                <L 1240>
                if (var_2685) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_2691 = wp::sub(var_2633, var_2641);
                    var_2692 = safe_div_0(var_2691, var_2637);
                }
                if (!var_2685) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_2695 = wp::add(var_2633, var_2641);
                    var_2696 = safe_div_0(var_2695, var_2637);
                }
                var_2697 = wp::where(var_2685, var_2692, var_2696);
                var_2698 = wp::where(var_2685, var_2689, var_2693);
                var_2699 = wp::where(var_2685, var_2690, var_2694);
                // if x1 > 1:                                                                 <L 1249>
                var_2701 = (var_2697 > var_2700);
                if (var_2701) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_2701) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_2705 = (var_2697 < var_2704);
                    if (var_2705) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_2708 = wp::where(var_2705, var_2706, var_2697);
                    var_2709 = wp::where(var_2705, var_2707, var_2683);
                }
                var_2710 = wp::where(var_2701, var_2702, var_2708);
                var_2711 = wp::where(var_2701, var_2703, var_2709);
            }
            var_2712 = wp::where(var_2686, var_2710, var_2681);
            var_2713 = wp::where(var_2686, var_2698, var_2682);
            var_2714 = wp::where(var_2686, var_2711, var_2683);
            var_2715 = wp::where(var_2686, var_2699, var_2665);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_2716 = wp::mul(var_4, var_2713);
            var_2717 = wp::sub(var_2629, var_2716);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_2718 = wp::extract(var_box_size, var_2600);
            var_2719 = wp::mul(var_2718, var_2712);
            wp::add_inplace(var_2717, var_2600, var_2719);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_2721 = wp::mul(var_2714, var_2720);
            var_2722 = wp::add(var_2721, var_2715);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_2723 = wp::length_sq(var_2717);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_2724 = wp::sub(var_2576, var_275);
            var_2725 = (var_2723 < var_2724);
            if (var_2725) {
                // bestdist = dif_sq                                                          <L 1264>
                var_2726 = wp::copy(var_2723);
                // bestsegmentpos = x2                                                        <L 1265>
                var_2727 = wp::copy(var_2713);
                // bestboxpos = x1                                                            <L 1266>
                var_2728 = wp::copy(var_2712);
                // c2 = ct // 6                                                               <L 1268>
                var_2730 = wp::floordiv(var_2722, var_2729);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_2732 = wp::lshift(var_2731, var_2600);
                var_2733 = wp::mul(var_2732, var_2730);
                var_2734 = wp::add(var_2415, var_2733);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_2735 = wp::copy(var_2600);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_2736 = wp::copy(var_2722);
            }
            var_2737 = wp::where(var_2725, var_2726, var_2576);
            var_2738 = wp::where(var_2725, var_2727, var_2577);
            var_2739 = wp::where(var_2725, var_2736, var_2578);
            var_2740 = wp::where(var_2725, var_2734, var_2579);
            var_2741 = wp::where(var_2725, var_2735, var_2580);
            var_2742 = wp::where(var_2725, var_2728, var_2581);
            var_2743 = wp::where(var_2725, var_2730, var_2610);
        }
        var_2744 = wp::where(var_2649, var_2737, var_2576);
        var_2745 = wp::where(var_2649, var_2738, var_2577);
        var_2746 = wp::where(var_2649, var_2739, var_2578);
        var_2747 = wp::where(var_2649, var_2740, var_2579);
        var_2748 = wp::where(var_2649, var_2741, var_2580);
        var_2749 = wp::where(var_2649, var_2742, var_2581);
        var_2750 = wp::where(var_2649, var_2743, var_2610);
        var_2751 = wp::where(var_2649, var_2717, var_2629);
        var_2752 = wp::where(var_2649, var_2651, var_2591);
        var_2753 = wp::where(var_2649, var_2712, var_2592);
        var_2754 = wp::where(var_2649, var_2713, var_2593);
        var_2755 = wp::where(var_2649, var_2714, var_2594);
        var_2756 = wp::where(var_2649, var_2715, var_2595);
        var_2757 = wp::where(var_2649, var_2685, var_2596);
        var_2758 = wp::where(var_2649, var_2722, var_2597);
        var_2759 = wp::where(var_2649, var_2723, var_2598);
    }
    var_2760 = wp::where(var_2608, var_2744, var_2576);
    var_2761 = wp::where(var_2608, var_2745, var_2577);
    var_2762 = wp::where(var_2608, var_2746, var_2578);
    var_2763 = wp::where(var_2608, var_2747, var_2579);
    var_2764 = wp::where(var_2608, var_2748, var_2580);
    var_2765 = wp::where(var_2608, var_2749, var_2581);
    var_2766 = wp::where(var_2608, var_2750, var_2582);
    var_2767 = wp::where(var_2608, var_2627, var_2583);
    var_2768 = wp::where(var_2608, var_2751, var_2584);
    var_2769 = wp::where(var_2608, var_2633, var_2585);
    var_2770 = wp::where(var_2608, var_2634, var_2586);
    var_2771 = wp::where(var_2608, var_2637, var_2587);
    var_2772 = wp::where(var_2608, var_2641, var_2588);
    var_2773 = wp::where(var_2608, var_2642, var_2589);
    var_2774 = wp::where(var_2608, var_2645, var_2590);
    var_2775 = wp::where(var_2608, var_2752, var_2591);
    var_2776 = wp::where(var_2608, var_2753, var_2592);
    var_2777 = wp::where(var_2608, var_2754, var_2593);
    var_2778 = wp::where(var_2608, var_2755, var_2594);
    var_2779 = wp::where(var_2608, var_2756, var_2595);
    var_2780 = wp::where(var_2608, var_2757, var_2596);
    var_2781 = wp::where(var_2608, var_2758, var_2597);
    var_2782 = wp::where(var_2608, var_2759, var_2598);
    var_2783 = wp::where(var_2608, var_2649, var_2608);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_2787 = wp::lshift(var_2786, var_2784);
    var_2788 = wp::bit_and(var_2415, var_2787);
    var_2790 = (var_2788 != var_2789);
    if (var_2790) {
        // continue                                                                           <L 1193>
    }
    var_2792 = wp::where(var_2790, var_2791, var_2785);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_2792) {
        var_2794 = wp::int32(var_2793);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_2796 = wp::bit_and(var_2415, var_2795);
        var_2799 = wp::where(var_2796, var_2797, var_2798);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_2801 = wp::bit_and(var_2415, var_2800);
        var_2804 = wp::where(var_2801, var_2802, var_2803);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_2806 = wp::bit_and(var_2415, var_2805);
        var_2809 = wp::where(var_2806, var_2807, var_2808);
        var_2810 = wp::vec_t<3, wp::float32>(var_2799, var_2804, var_2809);
        // box_size,                                                                          <L 1204>
        var_2811 = wp::cw_mul(var_2810, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_2811, var_2784, var_2812);
        // dif = box_pt - pos                                                                 <L 1209>
        var_2813 = wp::sub(var_2811, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_2814 = wp::extract(var_box_size, var_2784);
        var_2815 = wp::neg(var_2814);
        var_2816 = wp::extract(var_2813, var_2784);
        var_2817 = wp::mul(var_2815, var_2816);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_2818 = wp::dot(var_4, var_2813);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_2819 = wp::extract(var_box_size, var_2784);
        var_2820 = wp::extract(var_box_size, var_2784);
        var_2821 = wp::mul(var_2819, var_2820);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_2822 = wp::extract(var_box_size, var_2784);
        var_2823 = wp::neg(var_2822);
        var_2824 = wp::extract(var_4, var_2784);
        var_2825 = wp::mul(var_2823, var_2824);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_2826 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_2827 = wp::mul(var_2821, var_2826);
        var_2828 = wp::mul(var_2825, var_2825);
        var_2829 = wp::sub(var_2827, var_2828);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_2830 = wp::abs(var_2829);
        var_2831 = (var_2830 < var_275);
        if (var_2831) {
            // continue                                                                       <L 1218>
        }
        var_2833 = wp::where(var_2831, var_2832, var_2792);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_2833) {
            var_2835 = wp::div(var_2834, var_2829);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_2836 = wp::mul(var_2826, var_2817);
            var_2837 = wp::mul(var_2825, var_2818);
            var_2838 = wp::sub(var_2836, var_2837);
            var_2839 = wp::mul(var_2838, var_2835);
            var_2840 = wp::float32(var_2839);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_2841 = wp::mul(var_2821, var_2818);
            var_2842 = wp::mul(var_2825, var_2817);
            var_2843 = wp::sub(var_2841, var_2842);
            var_2844 = wp::mul(var_2843, var_2835);
            var_2845 = wp::float32(var_2844);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_2847 = wp::int32(var_2846);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_2849 = wp::int32(var_2848);
            // if x1 > 1:                                                                     <L 1229>
            var_2851 = (var_2840 > var_2850);
            if (var_2851) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_2854 = wp::sub(var_2818, var_2825);
                var_2855 = safe_div_0(var_2854, var_2826);
            }
            if (!var_2851) {
                // elif x1 < -1:                                                              <L 1233>
                var_2857 = (var_2840 < var_2856);
                if (var_2857) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_2860 = wp::add(var_2818, var_2825);
                    var_2861 = safe_div_0(var_2860, var_2826);
                }
                var_2862 = wp::where(var_2857, var_2858, var_2840);
                var_2863 = wp::where(var_2857, var_2861, var_2845);
                var_2864 = wp::where(var_2857, var_2859, var_2847);
            }
            var_2865 = wp::where(var_2851, var_2852, var_2862);
            var_2866 = wp::where(var_2851, var_2855, var_2863);
            var_2867 = wp::where(var_2851, var_2853, var_2864);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_2869 = (var_2866 > var_2868);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_2870 = var_2869;
            if (!var_2870) {
                var_2872 = (var_2866 < var_2871);
                var_2870 = var_2870 || var_2872;
            }
            if (var_2870) {
                // if x2_over:                                                                <L 1240>
                if (var_2869) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_2875 = wp::sub(var_2817, var_2825);
                    var_2876 = safe_div_0(var_2875, var_2821);
                }
                if (!var_2869) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_2879 = wp::add(var_2817, var_2825);
                    var_2880 = safe_div_0(var_2879, var_2821);
                }
                var_2881 = wp::where(var_2869, var_2876, var_2880);
                var_2882 = wp::where(var_2869, var_2873, var_2877);
                var_2883 = wp::where(var_2869, var_2874, var_2878);
                // if x1 > 1:                                                                 <L 1249>
                var_2885 = (var_2881 > var_2884);
                if (var_2885) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_2885) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_2889 = (var_2881 < var_2888);
                    if (var_2889) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_2892 = wp::where(var_2889, var_2890, var_2881);
                    var_2893 = wp::where(var_2889, var_2891, var_2867);
                }
                var_2894 = wp::where(var_2885, var_2886, var_2892);
                var_2895 = wp::where(var_2885, var_2887, var_2893);
            }
            var_2896 = wp::where(var_2870, var_2894, var_2865);
            var_2897 = wp::where(var_2870, var_2882, var_2866);
            var_2898 = wp::where(var_2870, var_2895, var_2867);
            var_2899 = wp::where(var_2870, var_2883, var_2849);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_2900 = wp::mul(var_4, var_2897);
            var_2901 = wp::sub(var_2813, var_2900);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_2902 = wp::extract(var_box_size, var_2784);
            var_2903 = wp::mul(var_2902, var_2896);
            wp::add_inplace(var_2901, var_2784, var_2903);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_2905 = wp::mul(var_2898, var_2904);
            var_2906 = wp::add(var_2905, var_2899);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_2907 = wp::length_sq(var_2901);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_2908 = wp::sub(var_2760, var_275);
            var_2909 = (var_2907 < var_2908);
            if (var_2909) {
                // bestdist = dif_sq                                                          <L 1264>
                var_2910 = wp::copy(var_2907);
                // bestsegmentpos = x2                                                        <L 1265>
                var_2911 = wp::copy(var_2897);
                // bestboxpos = x1                                                            <L 1266>
                var_2912 = wp::copy(var_2896);
                // c2 = ct // 6                                                               <L 1268>
                var_2914 = wp::floordiv(var_2906, var_2913);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_2916 = wp::lshift(var_2915, var_2784);
                var_2917 = wp::mul(var_2916, var_2914);
                var_2918 = wp::add(var_2415, var_2917);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_2919 = wp::copy(var_2784);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_2920 = wp::copy(var_2906);
            }
            var_2921 = wp::where(var_2909, var_2910, var_2760);
            var_2922 = wp::where(var_2909, var_2911, var_2761);
            var_2923 = wp::where(var_2909, var_2920, var_2762);
            var_2924 = wp::where(var_2909, var_2918, var_2763);
            var_2925 = wp::where(var_2909, var_2919, var_2764);
            var_2926 = wp::where(var_2909, var_2912, var_2765);
            var_2927 = wp::where(var_2909, var_2914, var_2794);
        }
        var_2928 = wp::where(var_2833, var_2921, var_2760);
        var_2929 = wp::where(var_2833, var_2922, var_2761);
        var_2930 = wp::where(var_2833, var_2923, var_2762);
        var_2931 = wp::where(var_2833, var_2924, var_2763);
        var_2932 = wp::where(var_2833, var_2925, var_2764);
        var_2933 = wp::where(var_2833, var_2926, var_2765);
        var_2934 = wp::where(var_2833, var_2927, var_2794);
        var_2935 = wp::where(var_2833, var_2901, var_2813);
        var_2936 = wp::where(var_2833, var_2835, var_2775);
        var_2937 = wp::where(var_2833, var_2896, var_2776);
        var_2938 = wp::where(var_2833, var_2897, var_2777);
        var_2939 = wp::where(var_2833, var_2898, var_2778);
        var_2940 = wp::where(var_2833, var_2899, var_2779);
        var_2941 = wp::where(var_2833, var_2869, var_2780);
        var_2942 = wp::where(var_2833, var_2906, var_2781);
        var_2943 = wp::where(var_2833, var_2907, var_2782);
    }
    var_2944 = wp::where(var_2792, var_2928, var_2760);
    var_2945 = wp::where(var_2792, var_2929, var_2761);
    var_2946 = wp::where(var_2792, var_2930, var_2762);
    var_2947 = wp::where(var_2792, var_2931, var_2763);
    var_2948 = wp::where(var_2792, var_2932, var_2764);
    var_2949 = wp::where(var_2792, var_2933, var_2765);
    var_2950 = wp::where(var_2792, var_2934, var_2766);
    var_2951 = wp::where(var_2792, var_2811, var_2767);
    var_2952 = wp::where(var_2792, var_2935, var_2768);
    var_2953 = wp::where(var_2792, var_2817, var_2769);
    var_2954 = wp::where(var_2792, var_2818, var_2770);
    var_2955 = wp::where(var_2792, var_2821, var_2771);
    var_2956 = wp::where(var_2792, var_2825, var_2772);
    var_2957 = wp::where(var_2792, var_2826, var_2773);
    var_2958 = wp::where(var_2792, var_2829, var_2774);
    var_2959 = wp::where(var_2792, var_2936, var_2775);
    var_2960 = wp::where(var_2792, var_2937, var_2776);
    var_2961 = wp::where(var_2792, var_2938, var_2777);
    var_2962 = wp::where(var_2792, var_2939, var_2778);
    var_2963 = wp::where(var_2792, var_2940, var_2779);
    var_2964 = wp::where(var_2792, var_2941, var_2780);
    var_2965 = wp::where(var_2792, var_2942, var_2781);
    var_2966 = wp::where(var_2792, var_2943, var_2782);
    var_2967 = wp::where(var_2792, var_2833, var_2792);
    // for j in range(3):                                                                     <L 1191>
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_2972 = wp::lshift(var_2971, var_2969);
    var_2973 = wp::bit_and(var_2968, var_2972);
    var_2975 = (var_2973 != var_2974);
    if (var_2975) {
        // continue                                                                           <L 1193>
    }
    var_2977 = wp::where(var_2975, var_2976, var_2970);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_2977) {
        var_2979 = wp::int32(var_2978);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_2981 = wp::bit_and(var_2968, var_2980);
        var_2984 = wp::where(var_2981, var_2982, var_2983);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_2986 = wp::bit_and(var_2968, var_2985);
        var_2989 = wp::where(var_2986, var_2987, var_2988);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_2991 = wp::bit_and(var_2968, var_2990);
        var_2994 = wp::where(var_2991, var_2992, var_2993);
        var_2995 = wp::vec_t<3, wp::float32>(var_2984, var_2989, var_2994);
        // box_size,                                                                          <L 1204>
        var_2996 = wp::cw_mul(var_2995, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_2996, var_2969, var_2997);
        // dif = box_pt - pos                                                                 <L 1209>
        var_2998 = wp::sub(var_2996, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_2999 = wp::extract(var_box_size, var_2969);
        var_3000 = wp::neg(var_2999);
        var_3001 = wp::extract(var_2998, var_2969);
        var_3002 = wp::mul(var_3000, var_3001);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_3003 = wp::dot(var_4, var_2998);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_3004 = wp::extract(var_box_size, var_2969);
        var_3005 = wp::extract(var_box_size, var_2969);
        var_3006 = wp::mul(var_3004, var_3005);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_3007 = wp::extract(var_box_size, var_2969);
        var_3008 = wp::neg(var_3007);
        var_3009 = wp::extract(var_4, var_2969);
        var_3010 = wp::mul(var_3008, var_3009);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_3011 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_3012 = wp::mul(var_3006, var_3011);
        var_3013 = wp::mul(var_3010, var_3010);
        var_3014 = wp::sub(var_3012, var_3013);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_3015 = wp::abs(var_3014);
        var_3016 = (var_3015 < var_275);
        if (var_3016) {
            // continue                                                                       <L 1218>
        }
        var_3018 = wp::where(var_3016, var_3017, var_2977);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_3018) {
            var_3020 = wp::div(var_3019, var_3014);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_3021 = wp::mul(var_3011, var_3002);
            var_3022 = wp::mul(var_3010, var_3003);
            var_3023 = wp::sub(var_3021, var_3022);
            var_3024 = wp::mul(var_3023, var_3020);
            var_3025 = wp::float32(var_3024);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_3026 = wp::mul(var_3006, var_3003);
            var_3027 = wp::mul(var_3010, var_3002);
            var_3028 = wp::sub(var_3026, var_3027);
            var_3029 = wp::mul(var_3028, var_3020);
            var_3030 = wp::float32(var_3029);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_3032 = wp::int32(var_3031);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_3034 = wp::int32(var_3033);
            // if x1 > 1:                                                                     <L 1229>
            var_3036 = (var_3025 > var_3035);
            if (var_3036) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_3039 = wp::sub(var_3003, var_3010);
                var_3040 = safe_div_0(var_3039, var_3011);
            }
            if (!var_3036) {
                // elif x1 < -1:                                                              <L 1233>
                var_3042 = (var_3025 < var_3041);
                if (var_3042) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_3045 = wp::add(var_3003, var_3010);
                    var_3046 = safe_div_0(var_3045, var_3011);
                }
                var_3047 = wp::where(var_3042, var_3043, var_3025);
                var_3048 = wp::where(var_3042, var_3046, var_3030);
                var_3049 = wp::where(var_3042, var_3044, var_3032);
            }
            var_3050 = wp::where(var_3036, var_3037, var_3047);
            var_3051 = wp::where(var_3036, var_3040, var_3048);
            var_3052 = wp::where(var_3036, var_3038, var_3049);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_3054 = (var_3051 > var_3053);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_3055 = var_3054;
            if (!var_3055) {
                var_3057 = (var_3051 < var_3056);
                var_3055 = var_3055 || var_3057;
            }
            if (var_3055) {
                // if x2_over:                                                                <L 1240>
                if (var_3054) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_3060 = wp::sub(var_3002, var_3010);
                    var_3061 = safe_div_0(var_3060, var_3006);
                }
                if (!var_3054) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_3064 = wp::add(var_3002, var_3010);
                    var_3065 = safe_div_0(var_3064, var_3006);
                }
                var_3066 = wp::where(var_3054, var_3061, var_3065);
                var_3067 = wp::where(var_3054, var_3058, var_3062);
                var_3068 = wp::where(var_3054, var_3059, var_3063);
                // if x1 > 1:                                                                 <L 1249>
                var_3070 = (var_3066 > var_3069);
                if (var_3070) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_3070) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_3074 = (var_3066 < var_3073);
                    if (var_3074) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_3077 = wp::where(var_3074, var_3075, var_3066);
                    var_3078 = wp::where(var_3074, var_3076, var_3052);
                }
                var_3079 = wp::where(var_3070, var_3071, var_3077);
                var_3080 = wp::where(var_3070, var_3072, var_3078);
            }
            var_3081 = wp::where(var_3055, var_3079, var_3050);
            var_3082 = wp::where(var_3055, var_3067, var_3051);
            var_3083 = wp::where(var_3055, var_3080, var_3052);
            var_3084 = wp::where(var_3055, var_3068, var_3034);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_3085 = wp::mul(var_4, var_3082);
            var_3086 = wp::sub(var_2998, var_3085);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_3087 = wp::extract(var_box_size, var_2969);
            var_3088 = wp::mul(var_3087, var_3081);
            wp::add_inplace(var_3086, var_2969, var_3088);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_3090 = wp::mul(var_3083, var_3089);
            var_3091 = wp::add(var_3090, var_3084);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_3092 = wp::length_sq(var_3086);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_3093 = wp::sub(var_2944, var_275);
            var_3094 = (var_3092 < var_3093);
            if (var_3094) {
                // bestdist = dif_sq                                                          <L 1264>
                var_3095 = wp::copy(var_3092);
                // bestsegmentpos = x2                                                        <L 1265>
                var_3096 = wp::copy(var_3082);
                // bestboxpos = x1                                                            <L 1266>
                var_3097 = wp::copy(var_3081);
                // c2 = ct // 6                                                               <L 1268>
                var_3099 = wp::floordiv(var_3091, var_3098);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_3101 = wp::lshift(var_3100, var_2969);
                var_3102 = wp::mul(var_3101, var_3099);
                var_3103 = wp::add(var_2968, var_3102);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_3104 = wp::copy(var_2969);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_3105 = wp::copy(var_3091);
            }
            var_3106 = wp::where(var_3094, var_3095, var_2944);
            var_3107 = wp::where(var_3094, var_3096, var_2945);
            var_3108 = wp::where(var_3094, var_3105, var_2946);
            var_3109 = wp::where(var_3094, var_3103, var_2947);
            var_3110 = wp::where(var_3094, var_3104, var_2948);
            var_3111 = wp::where(var_3094, var_3097, var_2949);
            var_3112 = wp::where(var_3094, var_3099, var_2979);
        }
        var_3113 = wp::where(var_3018, var_3106, var_2944);
        var_3114 = wp::where(var_3018, var_3107, var_2945);
        var_3115 = wp::where(var_3018, var_3108, var_2946);
        var_3116 = wp::where(var_3018, var_3109, var_2947);
        var_3117 = wp::where(var_3018, var_3110, var_2948);
        var_3118 = wp::where(var_3018, var_3111, var_2949);
        var_3119 = wp::where(var_3018, var_3112, var_2979);
        var_3120 = wp::where(var_3018, var_3086, var_2998);
        var_3121 = wp::where(var_3018, var_3020, var_2959);
        var_3122 = wp::where(var_3018, var_3081, var_2960);
        var_3123 = wp::where(var_3018, var_3082, var_2961);
        var_3124 = wp::where(var_3018, var_3083, var_2962);
        var_3125 = wp::where(var_3018, var_3084, var_2963);
        var_3126 = wp::where(var_3018, var_3054, var_2964);
        var_3127 = wp::where(var_3018, var_3091, var_2965);
        var_3128 = wp::where(var_3018, var_3092, var_2966);
    }
    var_3129 = wp::where(var_2977, var_3113, var_2944);
    var_3130 = wp::where(var_2977, var_3114, var_2945);
    var_3131 = wp::where(var_2977, var_3115, var_2946);
    var_3132 = wp::where(var_2977, var_3116, var_2947);
    var_3133 = wp::where(var_2977, var_3117, var_2948);
    var_3134 = wp::where(var_2977, var_3118, var_2949);
    var_3135 = wp::where(var_2977, var_3119, var_2950);
    var_3136 = wp::where(var_2977, var_2996, var_2951);
    var_3137 = wp::where(var_2977, var_3120, var_2952);
    var_3138 = wp::where(var_2977, var_3002, var_2953);
    var_3139 = wp::where(var_2977, var_3003, var_2954);
    var_3140 = wp::where(var_2977, var_3006, var_2955);
    var_3141 = wp::where(var_2977, var_3010, var_2956);
    var_3142 = wp::where(var_2977, var_3011, var_2957);
    var_3143 = wp::where(var_2977, var_3014, var_2958);
    var_3144 = wp::where(var_2977, var_3121, var_2959);
    var_3145 = wp::where(var_2977, var_3122, var_2960);
    var_3146 = wp::where(var_2977, var_3123, var_2961);
    var_3147 = wp::where(var_2977, var_3124, var_2962);
    var_3148 = wp::where(var_2977, var_3125, var_2963);
    var_3149 = wp::where(var_2977, var_3126, var_2964);
    var_3150 = wp::where(var_2977, var_3127, var_2965);
    var_3151 = wp::where(var_2977, var_3128, var_2966);
    var_3152 = wp::where(var_2977, var_3018, var_2977);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_3156 = wp::lshift(var_3155, var_3153);
    var_3157 = wp::bit_and(var_2968, var_3156);
    var_3159 = (var_3157 != var_3158);
    if (var_3159) {
        // continue                                                                           <L 1193>
    }
    var_3161 = wp::where(var_3159, var_3160, var_3154);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_3161) {
        var_3163 = wp::int32(var_3162);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_3165 = wp::bit_and(var_2968, var_3164);
        var_3168 = wp::where(var_3165, var_3166, var_3167);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_3170 = wp::bit_and(var_2968, var_3169);
        var_3173 = wp::where(var_3170, var_3171, var_3172);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_3175 = wp::bit_and(var_2968, var_3174);
        var_3178 = wp::where(var_3175, var_3176, var_3177);
        var_3179 = wp::vec_t<3, wp::float32>(var_3168, var_3173, var_3178);
        // box_size,                                                                          <L 1204>
        var_3180 = wp::cw_mul(var_3179, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_3180, var_3153, var_3181);
        // dif = box_pt - pos                                                                 <L 1209>
        var_3182 = wp::sub(var_3180, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_3183 = wp::extract(var_box_size, var_3153);
        var_3184 = wp::neg(var_3183);
        var_3185 = wp::extract(var_3182, var_3153);
        var_3186 = wp::mul(var_3184, var_3185);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_3187 = wp::dot(var_4, var_3182);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_3188 = wp::extract(var_box_size, var_3153);
        var_3189 = wp::extract(var_box_size, var_3153);
        var_3190 = wp::mul(var_3188, var_3189);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_3191 = wp::extract(var_box_size, var_3153);
        var_3192 = wp::neg(var_3191);
        var_3193 = wp::extract(var_4, var_3153);
        var_3194 = wp::mul(var_3192, var_3193);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_3195 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_3196 = wp::mul(var_3190, var_3195);
        var_3197 = wp::mul(var_3194, var_3194);
        var_3198 = wp::sub(var_3196, var_3197);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_3199 = wp::abs(var_3198);
        var_3200 = (var_3199 < var_275);
        if (var_3200) {
            // continue                                                                       <L 1218>
        }
        var_3202 = wp::where(var_3200, var_3201, var_3161);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_3202) {
            var_3204 = wp::div(var_3203, var_3198);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_3205 = wp::mul(var_3195, var_3186);
            var_3206 = wp::mul(var_3194, var_3187);
            var_3207 = wp::sub(var_3205, var_3206);
            var_3208 = wp::mul(var_3207, var_3204);
            var_3209 = wp::float32(var_3208);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_3210 = wp::mul(var_3190, var_3187);
            var_3211 = wp::mul(var_3194, var_3186);
            var_3212 = wp::sub(var_3210, var_3211);
            var_3213 = wp::mul(var_3212, var_3204);
            var_3214 = wp::float32(var_3213);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_3216 = wp::int32(var_3215);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_3218 = wp::int32(var_3217);
            // if x1 > 1:                                                                     <L 1229>
            var_3220 = (var_3209 > var_3219);
            if (var_3220) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_3223 = wp::sub(var_3187, var_3194);
                var_3224 = safe_div_0(var_3223, var_3195);
            }
            if (!var_3220) {
                // elif x1 < -1:                                                              <L 1233>
                var_3226 = (var_3209 < var_3225);
                if (var_3226) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_3229 = wp::add(var_3187, var_3194);
                    var_3230 = safe_div_0(var_3229, var_3195);
                }
                var_3231 = wp::where(var_3226, var_3227, var_3209);
                var_3232 = wp::where(var_3226, var_3230, var_3214);
                var_3233 = wp::where(var_3226, var_3228, var_3216);
            }
            var_3234 = wp::where(var_3220, var_3221, var_3231);
            var_3235 = wp::where(var_3220, var_3224, var_3232);
            var_3236 = wp::where(var_3220, var_3222, var_3233);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_3238 = (var_3235 > var_3237);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_3239 = var_3238;
            if (!var_3239) {
                var_3241 = (var_3235 < var_3240);
                var_3239 = var_3239 || var_3241;
            }
            if (var_3239) {
                // if x2_over:                                                                <L 1240>
                if (var_3238) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_3244 = wp::sub(var_3186, var_3194);
                    var_3245 = safe_div_0(var_3244, var_3190);
                }
                if (!var_3238) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_3248 = wp::add(var_3186, var_3194);
                    var_3249 = safe_div_0(var_3248, var_3190);
                }
                var_3250 = wp::where(var_3238, var_3245, var_3249);
                var_3251 = wp::where(var_3238, var_3242, var_3246);
                var_3252 = wp::where(var_3238, var_3243, var_3247);
                // if x1 > 1:                                                                 <L 1249>
                var_3254 = (var_3250 > var_3253);
                if (var_3254) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_3254) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_3258 = (var_3250 < var_3257);
                    if (var_3258) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_3261 = wp::where(var_3258, var_3259, var_3250);
                    var_3262 = wp::where(var_3258, var_3260, var_3236);
                }
                var_3263 = wp::where(var_3254, var_3255, var_3261);
                var_3264 = wp::where(var_3254, var_3256, var_3262);
            }
            var_3265 = wp::where(var_3239, var_3263, var_3234);
            var_3266 = wp::where(var_3239, var_3251, var_3235);
            var_3267 = wp::where(var_3239, var_3264, var_3236);
            var_3268 = wp::where(var_3239, var_3252, var_3218);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_3269 = wp::mul(var_4, var_3266);
            var_3270 = wp::sub(var_3182, var_3269);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_3271 = wp::extract(var_box_size, var_3153);
            var_3272 = wp::mul(var_3271, var_3265);
            wp::add_inplace(var_3270, var_3153, var_3272);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_3274 = wp::mul(var_3267, var_3273);
            var_3275 = wp::add(var_3274, var_3268);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_3276 = wp::length_sq(var_3270);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_3277 = wp::sub(var_3129, var_275);
            var_3278 = (var_3276 < var_3277);
            if (var_3278) {
                // bestdist = dif_sq                                                          <L 1264>
                var_3279 = wp::copy(var_3276);
                // bestsegmentpos = x2                                                        <L 1265>
                var_3280 = wp::copy(var_3266);
                // bestboxpos = x1                                                            <L 1266>
                var_3281 = wp::copy(var_3265);
                // c2 = ct // 6                                                               <L 1268>
                var_3283 = wp::floordiv(var_3275, var_3282);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_3285 = wp::lshift(var_3284, var_3153);
                var_3286 = wp::mul(var_3285, var_3283);
                var_3287 = wp::add(var_2968, var_3286);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_3288 = wp::copy(var_3153);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_3289 = wp::copy(var_3275);
            }
            var_3290 = wp::where(var_3278, var_3279, var_3129);
            var_3291 = wp::where(var_3278, var_3280, var_3130);
            var_3292 = wp::where(var_3278, var_3289, var_3131);
            var_3293 = wp::where(var_3278, var_3287, var_3132);
            var_3294 = wp::where(var_3278, var_3288, var_3133);
            var_3295 = wp::where(var_3278, var_3281, var_3134);
            var_3296 = wp::where(var_3278, var_3283, var_3163);
        }
        var_3297 = wp::where(var_3202, var_3290, var_3129);
        var_3298 = wp::where(var_3202, var_3291, var_3130);
        var_3299 = wp::where(var_3202, var_3292, var_3131);
        var_3300 = wp::where(var_3202, var_3293, var_3132);
        var_3301 = wp::where(var_3202, var_3294, var_3133);
        var_3302 = wp::where(var_3202, var_3295, var_3134);
        var_3303 = wp::where(var_3202, var_3296, var_3163);
        var_3304 = wp::where(var_3202, var_3270, var_3182);
        var_3305 = wp::where(var_3202, var_3204, var_3144);
        var_3306 = wp::where(var_3202, var_3265, var_3145);
        var_3307 = wp::where(var_3202, var_3266, var_3146);
        var_3308 = wp::where(var_3202, var_3267, var_3147);
        var_3309 = wp::where(var_3202, var_3268, var_3148);
        var_3310 = wp::where(var_3202, var_3238, var_3149);
        var_3311 = wp::where(var_3202, var_3275, var_3150);
        var_3312 = wp::where(var_3202, var_3276, var_3151);
    }
    var_3313 = wp::where(var_3161, var_3297, var_3129);
    var_3314 = wp::where(var_3161, var_3298, var_3130);
    var_3315 = wp::where(var_3161, var_3299, var_3131);
    var_3316 = wp::where(var_3161, var_3300, var_3132);
    var_3317 = wp::where(var_3161, var_3301, var_3133);
    var_3318 = wp::where(var_3161, var_3302, var_3134);
    var_3319 = wp::where(var_3161, var_3303, var_3135);
    var_3320 = wp::where(var_3161, var_3180, var_3136);
    var_3321 = wp::where(var_3161, var_3304, var_3137);
    var_3322 = wp::where(var_3161, var_3186, var_3138);
    var_3323 = wp::where(var_3161, var_3187, var_3139);
    var_3324 = wp::where(var_3161, var_3190, var_3140);
    var_3325 = wp::where(var_3161, var_3194, var_3141);
    var_3326 = wp::where(var_3161, var_3195, var_3142);
    var_3327 = wp::where(var_3161, var_3198, var_3143);
    var_3328 = wp::where(var_3161, var_3305, var_3144);
    var_3329 = wp::where(var_3161, var_3306, var_3145);
    var_3330 = wp::where(var_3161, var_3307, var_3146);
    var_3331 = wp::where(var_3161, var_3308, var_3147);
    var_3332 = wp::where(var_3161, var_3309, var_3148);
    var_3333 = wp::where(var_3161, var_3310, var_3149);
    var_3334 = wp::where(var_3161, var_3311, var_3150);
    var_3335 = wp::where(var_3161, var_3312, var_3151);
    var_3336 = wp::where(var_3161, var_3202, var_3161);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_3340 = wp::lshift(var_3339, var_3337);
    var_3341 = wp::bit_and(var_2968, var_3340);
    var_3343 = (var_3341 != var_3342);
    if (var_3343) {
        // continue                                                                           <L 1193>
    }
    var_3345 = wp::where(var_3343, var_3344, var_3338);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_3345) {
        var_3347 = wp::int32(var_3346);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_3349 = wp::bit_and(var_2968, var_3348);
        var_3352 = wp::where(var_3349, var_3350, var_3351);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_3354 = wp::bit_and(var_2968, var_3353);
        var_3357 = wp::where(var_3354, var_3355, var_3356);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_3359 = wp::bit_and(var_2968, var_3358);
        var_3362 = wp::where(var_3359, var_3360, var_3361);
        var_3363 = wp::vec_t<3, wp::float32>(var_3352, var_3357, var_3362);
        // box_size,                                                                          <L 1204>
        var_3364 = wp::cw_mul(var_3363, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_3364, var_3337, var_3365);
        // dif = box_pt - pos                                                                 <L 1209>
        var_3366 = wp::sub(var_3364, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_3367 = wp::extract(var_box_size, var_3337);
        var_3368 = wp::neg(var_3367);
        var_3369 = wp::extract(var_3366, var_3337);
        var_3370 = wp::mul(var_3368, var_3369);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_3371 = wp::dot(var_4, var_3366);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_3372 = wp::extract(var_box_size, var_3337);
        var_3373 = wp::extract(var_box_size, var_3337);
        var_3374 = wp::mul(var_3372, var_3373);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_3375 = wp::extract(var_box_size, var_3337);
        var_3376 = wp::neg(var_3375);
        var_3377 = wp::extract(var_4, var_3337);
        var_3378 = wp::mul(var_3376, var_3377);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_3379 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_3380 = wp::mul(var_3374, var_3379);
        var_3381 = wp::mul(var_3378, var_3378);
        var_3382 = wp::sub(var_3380, var_3381);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_3383 = wp::abs(var_3382);
        var_3384 = (var_3383 < var_275);
        if (var_3384) {
            // continue                                                                       <L 1218>
        }
        var_3386 = wp::where(var_3384, var_3385, var_3345);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_3386) {
            var_3388 = wp::div(var_3387, var_3382);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_3389 = wp::mul(var_3379, var_3370);
            var_3390 = wp::mul(var_3378, var_3371);
            var_3391 = wp::sub(var_3389, var_3390);
            var_3392 = wp::mul(var_3391, var_3388);
            var_3393 = wp::float32(var_3392);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_3394 = wp::mul(var_3374, var_3371);
            var_3395 = wp::mul(var_3378, var_3370);
            var_3396 = wp::sub(var_3394, var_3395);
            var_3397 = wp::mul(var_3396, var_3388);
            var_3398 = wp::float32(var_3397);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_3400 = wp::int32(var_3399);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_3402 = wp::int32(var_3401);
            // if x1 > 1:                                                                     <L 1229>
            var_3404 = (var_3393 > var_3403);
            if (var_3404) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_3407 = wp::sub(var_3371, var_3378);
                var_3408 = safe_div_0(var_3407, var_3379);
            }
            if (!var_3404) {
                // elif x1 < -1:                                                              <L 1233>
                var_3410 = (var_3393 < var_3409);
                if (var_3410) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_3413 = wp::add(var_3371, var_3378);
                    var_3414 = safe_div_0(var_3413, var_3379);
                }
                var_3415 = wp::where(var_3410, var_3411, var_3393);
                var_3416 = wp::where(var_3410, var_3414, var_3398);
                var_3417 = wp::where(var_3410, var_3412, var_3400);
            }
            var_3418 = wp::where(var_3404, var_3405, var_3415);
            var_3419 = wp::where(var_3404, var_3408, var_3416);
            var_3420 = wp::where(var_3404, var_3406, var_3417);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_3422 = (var_3419 > var_3421);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_3423 = var_3422;
            if (!var_3423) {
                var_3425 = (var_3419 < var_3424);
                var_3423 = var_3423 || var_3425;
            }
            if (var_3423) {
                // if x2_over:                                                                <L 1240>
                if (var_3422) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_3428 = wp::sub(var_3370, var_3378);
                    var_3429 = safe_div_0(var_3428, var_3374);
                }
                if (!var_3422) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_3432 = wp::add(var_3370, var_3378);
                    var_3433 = safe_div_0(var_3432, var_3374);
                }
                var_3434 = wp::where(var_3422, var_3429, var_3433);
                var_3435 = wp::where(var_3422, var_3426, var_3430);
                var_3436 = wp::where(var_3422, var_3427, var_3431);
                // if x1 > 1:                                                                 <L 1249>
                var_3438 = (var_3434 > var_3437);
                if (var_3438) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_3438) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_3442 = (var_3434 < var_3441);
                    if (var_3442) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_3445 = wp::where(var_3442, var_3443, var_3434);
                    var_3446 = wp::where(var_3442, var_3444, var_3420);
                }
                var_3447 = wp::where(var_3438, var_3439, var_3445);
                var_3448 = wp::where(var_3438, var_3440, var_3446);
            }
            var_3449 = wp::where(var_3423, var_3447, var_3418);
            var_3450 = wp::where(var_3423, var_3435, var_3419);
            var_3451 = wp::where(var_3423, var_3448, var_3420);
            var_3452 = wp::where(var_3423, var_3436, var_3402);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_3453 = wp::mul(var_4, var_3450);
            var_3454 = wp::sub(var_3366, var_3453);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_3455 = wp::extract(var_box_size, var_3337);
            var_3456 = wp::mul(var_3455, var_3449);
            wp::add_inplace(var_3454, var_3337, var_3456);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_3458 = wp::mul(var_3451, var_3457);
            var_3459 = wp::add(var_3458, var_3452);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_3460 = wp::length_sq(var_3454);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_3461 = wp::sub(var_3313, var_275);
            var_3462 = (var_3460 < var_3461);
            if (var_3462) {
                // bestdist = dif_sq                                                          <L 1264>
                var_3463 = wp::copy(var_3460);
                // bestsegmentpos = x2                                                        <L 1265>
                var_3464 = wp::copy(var_3450);
                // bestboxpos = x1                                                            <L 1266>
                var_3465 = wp::copy(var_3449);
                // c2 = ct // 6                                                               <L 1268>
                var_3467 = wp::floordiv(var_3459, var_3466);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_3469 = wp::lshift(var_3468, var_3337);
                var_3470 = wp::mul(var_3469, var_3467);
                var_3471 = wp::add(var_2968, var_3470);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_3472 = wp::copy(var_3337);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_3473 = wp::copy(var_3459);
            }
            var_3474 = wp::where(var_3462, var_3463, var_3313);
            var_3475 = wp::where(var_3462, var_3464, var_3314);
            var_3476 = wp::where(var_3462, var_3473, var_3315);
            var_3477 = wp::where(var_3462, var_3471, var_3316);
            var_3478 = wp::where(var_3462, var_3472, var_3317);
            var_3479 = wp::where(var_3462, var_3465, var_3318);
            var_3480 = wp::where(var_3462, var_3467, var_3347);
        }
        var_3481 = wp::where(var_3386, var_3474, var_3313);
        var_3482 = wp::where(var_3386, var_3475, var_3314);
        var_3483 = wp::where(var_3386, var_3476, var_3315);
        var_3484 = wp::where(var_3386, var_3477, var_3316);
        var_3485 = wp::where(var_3386, var_3478, var_3317);
        var_3486 = wp::where(var_3386, var_3479, var_3318);
        var_3487 = wp::where(var_3386, var_3480, var_3347);
        var_3488 = wp::where(var_3386, var_3454, var_3366);
        var_3489 = wp::where(var_3386, var_3388, var_3328);
        var_3490 = wp::where(var_3386, var_3449, var_3329);
        var_3491 = wp::where(var_3386, var_3450, var_3330);
        var_3492 = wp::where(var_3386, var_3451, var_3331);
        var_3493 = wp::where(var_3386, var_3452, var_3332);
        var_3494 = wp::where(var_3386, var_3422, var_3333);
        var_3495 = wp::where(var_3386, var_3459, var_3334);
        var_3496 = wp::where(var_3386, var_3460, var_3335);
    }
    var_3497 = wp::where(var_3345, var_3481, var_3313);
    var_3498 = wp::where(var_3345, var_3482, var_3314);
    var_3499 = wp::where(var_3345, var_3483, var_3315);
    var_3500 = wp::where(var_3345, var_3484, var_3316);
    var_3501 = wp::where(var_3345, var_3485, var_3317);
    var_3502 = wp::where(var_3345, var_3486, var_3318);
    var_3503 = wp::where(var_3345, var_3487, var_3319);
    var_3504 = wp::where(var_3345, var_3364, var_3320);
    var_3505 = wp::where(var_3345, var_3488, var_3321);
    var_3506 = wp::where(var_3345, var_3370, var_3322);
    var_3507 = wp::where(var_3345, var_3371, var_3323);
    var_3508 = wp::where(var_3345, var_3374, var_3324);
    var_3509 = wp::where(var_3345, var_3378, var_3325);
    var_3510 = wp::where(var_3345, var_3379, var_3326);
    var_3511 = wp::where(var_3345, var_3382, var_3327);
    var_3512 = wp::where(var_3345, var_3489, var_3328);
    var_3513 = wp::where(var_3345, var_3490, var_3329);
    var_3514 = wp::where(var_3345, var_3491, var_3330);
    var_3515 = wp::where(var_3345, var_3492, var_3331);
    var_3516 = wp::where(var_3345, var_3493, var_3332);
    var_3517 = wp::where(var_3345, var_3494, var_3333);
    var_3518 = wp::where(var_3345, var_3495, var_3334);
    var_3519 = wp::where(var_3345, var_3496, var_3335);
    var_3520 = wp::where(var_3345, var_3386, var_3345);
    // for j in range(3):                                                                     <L 1191>
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_3525 = wp::lshift(var_3524, var_3522);
    var_3526 = wp::bit_and(var_3521, var_3525);
    var_3528 = (var_3526 != var_3527);
    if (var_3528) {
        // continue                                                                           <L 1193>
    }
    var_3530 = wp::where(var_3528, var_3529, var_3523);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_3530) {
        var_3532 = wp::int32(var_3531);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_3534 = wp::bit_and(var_3521, var_3533);
        var_3537 = wp::where(var_3534, var_3535, var_3536);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_3539 = wp::bit_and(var_3521, var_3538);
        var_3542 = wp::where(var_3539, var_3540, var_3541);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_3544 = wp::bit_and(var_3521, var_3543);
        var_3547 = wp::where(var_3544, var_3545, var_3546);
        var_3548 = wp::vec_t<3, wp::float32>(var_3537, var_3542, var_3547);
        // box_size,                                                                          <L 1204>
        var_3549 = wp::cw_mul(var_3548, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_3549, var_3522, var_3550);
        // dif = box_pt - pos                                                                 <L 1209>
        var_3551 = wp::sub(var_3549, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_3552 = wp::extract(var_box_size, var_3522);
        var_3553 = wp::neg(var_3552);
        var_3554 = wp::extract(var_3551, var_3522);
        var_3555 = wp::mul(var_3553, var_3554);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_3556 = wp::dot(var_4, var_3551);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_3557 = wp::extract(var_box_size, var_3522);
        var_3558 = wp::extract(var_box_size, var_3522);
        var_3559 = wp::mul(var_3557, var_3558);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_3560 = wp::extract(var_box_size, var_3522);
        var_3561 = wp::neg(var_3560);
        var_3562 = wp::extract(var_4, var_3522);
        var_3563 = wp::mul(var_3561, var_3562);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_3564 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_3565 = wp::mul(var_3559, var_3564);
        var_3566 = wp::mul(var_3563, var_3563);
        var_3567 = wp::sub(var_3565, var_3566);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_3568 = wp::abs(var_3567);
        var_3569 = (var_3568 < var_275);
        if (var_3569) {
            // continue                                                                       <L 1218>
        }
        var_3571 = wp::where(var_3569, var_3570, var_3530);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_3571) {
            var_3573 = wp::div(var_3572, var_3567);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_3574 = wp::mul(var_3564, var_3555);
            var_3575 = wp::mul(var_3563, var_3556);
            var_3576 = wp::sub(var_3574, var_3575);
            var_3577 = wp::mul(var_3576, var_3573);
            var_3578 = wp::float32(var_3577);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_3579 = wp::mul(var_3559, var_3556);
            var_3580 = wp::mul(var_3563, var_3555);
            var_3581 = wp::sub(var_3579, var_3580);
            var_3582 = wp::mul(var_3581, var_3573);
            var_3583 = wp::float32(var_3582);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_3585 = wp::int32(var_3584);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_3587 = wp::int32(var_3586);
            // if x1 > 1:                                                                     <L 1229>
            var_3589 = (var_3578 > var_3588);
            if (var_3589) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_3592 = wp::sub(var_3556, var_3563);
                var_3593 = safe_div_0(var_3592, var_3564);
            }
            if (!var_3589) {
                // elif x1 < -1:                                                              <L 1233>
                var_3595 = (var_3578 < var_3594);
                if (var_3595) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_3598 = wp::add(var_3556, var_3563);
                    var_3599 = safe_div_0(var_3598, var_3564);
                }
                var_3600 = wp::where(var_3595, var_3596, var_3578);
                var_3601 = wp::where(var_3595, var_3599, var_3583);
                var_3602 = wp::where(var_3595, var_3597, var_3585);
            }
            var_3603 = wp::where(var_3589, var_3590, var_3600);
            var_3604 = wp::where(var_3589, var_3593, var_3601);
            var_3605 = wp::where(var_3589, var_3591, var_3602);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_3607 = (var_3604 > var_3606);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_3608 = var_3607;
            if (!var_3608) {
                var_3610 = (var_3604 < var_3609);
                var_3608 = var_3608 || var_3610;
            }
            if (var_3608) {
                // if x2_over:                                                                <L 1240>
                if (var_3607) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_3613 = wp::sub(var_3555, var_3563);
                    var_3614 = safe_div_0(var_3613, var_3559);
                }
                if (!var_3607) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_3617 = wp::add(var_3555, var_3563);
                    var_3618 = safe_div_0(var_3617, var_3559);
                }
                var_3619 = wp::where(var_3607, var_3614, var_3618);
                var_3620 = wp::where(var_3607, var_3611, var_3615);
                var_3621 = wp::where(var_3607, var_3612, var_3616);
                // if x1 > 1:                                                                 <L 1249>
                var_3623 = (var_3619 > var_3622);
                if (var_3623) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_3623) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_3627 = (var_3619 < var_3626);
                    if (var_3627) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_3630 = wp::where(var_3627, var_3628, var_3619);
                    var_3631 = wp::where(var_3627, var_3629, var_3605);
                }
                var_3632 = wp::where(var_3623, var_3624, var_3630);
                var_3633 = wp::where(var_3623, var_3625, var_3631);
            }
            var_3634 = wp::where(var_3608, var_3632, var_3603);
            var_3635 = wp::where(var_3608, var_3620, var_3604);
            var_3636 = wp::where(var_3608, var_3633, var_3605);
            var_3637 = wp::where(var_3608, var_3621, var_3587);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_3638 = wp::mul(var_4, var_3635);
            var_3639 = wp::sub(var_3551, var_3638);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_3640 = wp::extract(var_box_size, var_3522);
            var_3641 = wp::mul(var_3640, var_3634);
            wp::add_inplace(var_3639, var_3522, var_3641);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_3643 = wp::mul(var_3636, var_3642);
            var_3644 = wp::add(var_3643, var_3637);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_3645 = wp::length_sq(var_3639);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_3646 = wp::sub(var_3497, var_275);
            var_3647 = (var_3645 < var_3646);
            if (var_3647) {
                // bestdist = dif_sq                                                          <L 1264>
                var_3648 = wp::copy(var_3645);
                // bestsegmentpos = x2                                                        <L 1265>
                var_3649 = wp::copy(var_3635);
                // bestboxpos = x1                                                            <L 1266>
                var_3650 = wp::copy(var_3634);
                // c2 = ct // 6                                                               <L 1268>
                var_3652 = wp::floordiv(var_3644, var_3651);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_3654 = wp::lshift(var_3653, var_3522);
                var_3655 = wp::mul(var_3654, var_3652);
                var_3656 = wp::add(var_3521, var_3655);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_3657 = wp::copy(var_3522);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_3658 = wp::copy(var_3644);
            }
            var_3659 = wp::where(var_3647, var_3648, var_3497);
            var_3660 = wp::where(var_3647, var_3649, var_3498);
            var_3661 = wp::where(var_3647, var_3658, var_3499);
            var_3662 = wp::where(var_3647, var_3656, var_3500);
            var_3663 = wp::where(var_3647, var_3657, var_3501);
            var_3664 = wp::where(var_3647, var_3650, var_3502);
            var_3665 = wp::where(var_3647, var_3652, var_3532);
        }
        var_3666 = wp::where(var_3571, var_3659, var_3497);
        var_3667 = wp::where(var_3571, var_3660, var_3498);
        var_3668 = wp::where(var_3571, var_3661, var_3499);
        var_3669 = wp::where(var_3571, var_3662, var_3500);
        var_3670 = wp::where(var_3571, var_3663, var_3501);
        var_3671 = wp::where(var_3571, var_3664, var_3502);
        var_3672 = wp::where(var_3571, var_3665, var_3532);
        var_3673 = wp::where(var_3571, var_3639, var_3551);
        var_3674 = wp::where(var_3571, var_3573, var_3512);
        var_3675 = wp::where(var_3571, var_3634, var_3513);
        var_3676 = wp::where(var_3571, var_3635, var_3514);
        var_3677 = wp::where(var_3571, var_3636, var_3515);
        var_3678 = wp::where(var_3571, var_3637, var_3516);
        var_3679 = wp::where(var_3571, var_3607, var_3517);
        var_3680 = wp::where(var_3571, var_3644, var_3518);
        var_3681 = wp::where(var_3571, var_3645, var_3519);
    }
    var_3682 = wp::where(var_3530, var_3666, var_3497);
    var_3683 = wp::where(var_3530, var_3667, var_3498);
    var_3684 = wp::where(var_3530, var_3668, var_3499);
    var_3685 = wp::where(var_3530, var_3669, var_3500);
    var_3686 = wp::where(var_3530, var_3670, var_3501);
    var_3687 = wp::where(var_3530, var_3671, var_3502);
    var_3688 = wp::where(var_3530, var_3672, var_3503);
    var_3689 = wp::where(var_3530, var_3549, var_3504);
    var_3690 = wp::where(var_3530, var_3673, var_3505);
    var_3691 = wp::where(var_3530, var_3555, var_3506);
    var_3692 = wp::where(var_3530, var_3556, var_3507);
    var_3693 = wp::where(var_3530, var_3559, var_3508);
    var_3694 = wp::where(var_3530, var_3563, var_3509);
    var_3695 = wp::where(var_3530, var_3564, var_3510);
    var_3696 = wp::where(var_3530, var_3567, var_3511);
    var_3697 = wp::where(var_3530, var_3674, var_3512);
    var_3698 = wp::where(var_3530, var_3675, var_3513);
    var_3699 = wp::where(var_3530, var_3676, var_3514);
    var_3700 = wp::where(var_3530, var_3677, var_3515);
    var_3701 = wp::where(var_3530, var_3678, var_3516);
    var_3702 = wp::where(var_3530, var_3679, var_3517);
    var_3703 = wp::where(var_3530, var_3680, var_3518);
    var_3704 = wp::where(var_3530, var_3681, var_3519);
    var_3705 = wp::where(var_3530, var_3571, var_3530);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_3709 = wp::lshift(var_3708, var_3706);
    var_3710 = wp::bit_and(var_3521, var_3709);
    var_3712 = (var_3710 != var_3711);
    if (var_3712) {
        // continue                                                                           <L 1193>
    }
    var_3714 = wp::where(var_3712, var_3713, var_3707);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_3714) {
        var_3716 = wp::int32(var_3715);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_3718 = wp::bit_and(var_3521, var_3717);
        var_3721 = wp::where(var_3718, var_3719, var_3720);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_3723 = wp::bit_and(var_3521, var_3722);
        var_3726 = wp::where(var_3723, var_3724, var_3725);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_3728 = wp::bit_and(var_3521, var_3727);
        var_3731 = wp::where(var_3728, var_3729, var_3730);
        var_3732 = wp::vec_t<3, wp::float32>(var_3721, var_3726, var_3731);
        // box_size,                                                                          <L 1204>
        var_3733 = wp::cw_mul(var_3732, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_3733, var_3706, var_3734);
        // dif = box_pt - pos                                                                 <L 1209>
        var_3735 = wp::sub(var_3733, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_3736 = wp::extract(var_box_size, var_3706);
        var_3737 = wp::neg(var_3736);
        var_3738 = wp::extract(var_3735, var_3706);
        var_3739 = wp::mul(var_3737, var_3738);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_3740 = wp::dot(var_4, var_3735);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_3741 = wp::extract(var_box_size, var_3706);
        var_3742 = wp::extract(var_box_size, var_3706);
        var_3743 = wp::mul(var_3741, var_3742);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_3744 = wp::extract(var_box_size, var_3706);
        var_3745 = wp::neg(var_3744);
        var_3746 = wp::extract(var_4, var_3706);
        var_3747 = wp::mul(var_3745, var_3746);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_3748 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_3749 = wp::mul(var_3743, var_3748);
        var_3750 = wp::mul(var_3747, var_3747);
        var_3751 = wp::sub(var_3749, var_3750);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_3752 = wp::abs(var_3751);
        var_3753 = (var_3752 < var_275);
        if (var_3753) {
            // continue                                                                       <L 1218>
        }
        var_3755 = wp::where(var_3753, var_3754, var_3714);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_3755) {
            var_3757 = wp::div(var_3756, var_3751);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_3758 = wp::mul(var_3748, var_3739);
            var_3759 = wp::mul(var_3747, var_3740);
            var_3760 = wp::sub(var_3758, var_3759);
            var_3761 = wp::mul(var_3760, var_3757);
            var_3762 = wp::float32(var_3761);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_3763 = wp::mul(var_3743, var_3740);
            var_3764 = wp::mul(var_3747, var_3739);
            var_3765 = wp::sub(var_3763, var_3764);
            var_3766 = wp::mul(var_3765, var_3757);
            var_3767 = wp::float32(var_3766);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_3769 = wp::int32(var_3768);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_3771 = wp::int32(var_3770);
            // if x1 > 1:                                                                     <L 1229>
            var_3773 = (var_3762 > var_3772);
            if (var_3773) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_3776 = wp::sub(var_3740, var_3747);
                var_3777 = safe_div_0(var_3776, var_3748);
            }
            if (!var_3773) {
                // elif x1 < -1:                                                              <L 1233>
                var_3779 = (var_3762 < var_3778);
                if (var_3779) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_3782 = wp::add(var_3740, var_3747);
                    var_3783 = safe_div_0(var_3782, var_3748);
                }
                var_3784 = wp::where(var_3779, var_3780, var_3762);
                var_3785 = wp::where(var_3779, var_3783, var_3767);
                var_3786 = wp::where(var_3779, var_3781, var_3769);
            }
            var_3787 = wp::where(var_3773, var_3774, var_3784);
            var_3788 = wp::where(var_3773, var_3777, var_3785);
            var_3789 = wp::where(var_3773, var_3775, var_3786);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_3791 = (var_3788 > var_3790);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_3792 = var_3791;
            if (!var_3792) {
                var_3794 = (var_3788 < var_3793);
                var_3792 = var_3792 || var_3794;
            }
            if (var_3792) {
                // if x2_over:                                                                <L 1240>
                if (var_3791) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_3797 = wp::sub(var_3739, var_3747);
                    var_3798 = safe_div_0(var_3797, var_3743);
                }
                if (!var_3791) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_3801 = wp::add(var_3739, var_3747);
                    var_3802 = safe_div_0(var_3801, var_3743);
                }
                var_3803 = wp::where(var_3791, var_3798, var_3802);
                var_3804 = wp::where(var_3791, var_3795, var_3799);
                var_3805 = wp::where(var_3791, var_3796, var_3800);
                // if x1 > 1:                                                                 <L 1249>
                var_3807 = (var_3803 > var_3806);
                if (var_3807) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_3807) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_3811 = (var_3803 < var_3810);
                    if (var_3811) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_3814 = wp::where(var_3811, var_3812, var_3803);
                    var_3815 = wp::where(var_3811, var_3813, var_3789);
                }
                var_3816 = wp::where(var_3807, var_3808, var_3814);
                var_3817 = wp::where(var_3807, var_3809, var_3815);
            }
            var_3818 = wp::where(var_3792, var_3816, var_3787);
            var_3819 = wp::where(var_3792, var_3804, var_3788);
            var_3820 = wp::where(var_3792, var_3817, var_3789);
            var_3821 = wp::where(var_3792, var_3805, var_3771);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_3822 = wp::mul(var_4, var_3819);
            var_3823 = wp::sub(var_3735, var_3822);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_3824 = wp::extract(var_box_size, var_3706);
            var_3825 = wp::mul(var_3824, var_3818);
            wp::add_inplace(var_3823, var_3706, var_3825);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_3827 = wp::mul(var_3820, var_3826);
            var_3828 = wp::add(var_3827, var_3821);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_3829 = wp::length_sq(var_3823);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_3830 = wp::sub(var_3682, var_275);
            var_3831 = (var_3829 < var_3830);
            if (var_3831) {
                // bestdist = dif_sq                                                          <L 1264>
                var_3832 = wp::copy(var_3829);
                // bestsegmentpos = x2                                                        <L 1265>
                var_3833 = wp::copy(var_3819);
                // bestboxpos = x1                                                            <L 1266>
                var_3834 = wp::copy(var_3818);
                // c2 = ct // 6                                                               <L 1268>
                var_3836 = wp::floordiv(var_3828, var_3835);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_3838 = wp::lshift(var_3837, var_3706);
                var_3839 = wp::mul(var_3838, var_3836);
                var_3840 = wp::add(var_3521, var_3839);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_3841 = wp::copy(var_3706);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_3842 = wp::copy(var_3828);
            }
            var_3843 = wp::where(var_3831, var_3832, var_3682);
            var_3844 = wp::where(var_3831, var_3833, var_3683);
            var_3845 = wp::where(var_3831, var_3842, var_3684);
            var_3846 = wp::where(var_3831, var_3840, var_3685);
            var_3847 = wp::where(var_3831, var_3841, var_3686);
            var_3848 = wp::where(var_3831, var_3834, var_3687);
            var_3849 = wp::where(var_3831, var_3836, var_3716);
        }
        var_3850 = wp::where(var_3755, var_3843, var_3682);
        var_3851 = wp::where(var_3755, var_3844, var_3683);
        var_3852 = wp::where(var_3755, var_3845, var_3684);
        var_3853 = wp::where(var_3755, var_3846, var_3685);
        var_3854 = wp::where(var_3755, var_3847, var_3686);
        var_3855 = wp::where(var_3755, var_3848, var_3687);
        var_3856 = wp::where(var_3755, var_3849, var_3716);
        var_3857 = wp::where(var_3755, var_3823, var_3735);
        var_3858 = wp::where(var_3755, var_3757, var_3697);
        var_3859 = wp::where(var_3755, var_3818, var_3698);
        var_3860 = wp::where(var_3755, var_3819, var_3699);
        var_3861 = wp::where(var_3755, var_3820, var_3700);
        var_3862 = wp::where(var_3755, var_3821, var_3701);
        var_3863 = wp::where(var_3755, var_3791, var_3702);
        var_3864 = wp::where(var_3755, var_3828, var_3703);
        var_3865 = wp::where(var_3755, var_3829, var_3704);
    }
    var_3866 = wp::where(var_3714, var_3850, var_3682);
    var_3867 = wp::where(var_3714, var_3851, var_3683);
    var_3868 = wp::where(var_3714, var_3852, var_3684);
    var_3869 = wp::where(var_3714, var_3853, var_3685);
    var_3870 = wp::where(var_3714, var_3854, var_3686);
    var_3871 = wp::where(var_3714, var_3855, var_3687);
    var_3872 = wp::where(var_3714, var_3856, var_3688);
    var_3873 = wp::where(var_3714, var_3733, var_3689);
    var_3874 = wp::where(var_3714, var_3857, var_3690);
    var_3875 = wp::where(var_3714, var_3739, var_3691);
    var_3876 = wp::where(var_3714, var_3740, var_3692);
    var_3877 = wp::where(var_3714, var_3743, var_3693);
    var_3878 = wp::where(var_3714, var_3747, var_3694);
    var_3879 = wp::where(var_3714, var_3748, var_3695);
    var_3880 = wp::where(var_3714, var_3751, var_3696);
    var_3881 = wp::where(var_3714, var_3858, var_3697);
    var_3882 = wp::where(var_3714, var_3859, var_3698);
    var_3883 = wp::where(var_3714, var_3860, var_3699);
    var_3884 = wp::where(var_3714, var_3861, var_3700);
    var_3885 = wp::where(var_3714, var_3862, var_3701);
    var_3886 = wp::where(var_3714, var_3863, var_3702);
    var_3887 = wp::where(var_3714, var_3864, var_3703);
    var_3888 = wp::where(var_3714, var_3865, var_3704);
    var_3889 = wp::where(var_3714, var_3755, var_3714);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_3893 = wp::lshift(var_3892, var_3890);
    var_3894 = wp::bit_and(var_3521, var_3893);
    var_3896 = (var_3894 != var_3895);
    if (var_3896) {
        // continue                                                                           <L 1193>
    }
    var_3898 = wp::where(var_3896, var_3897, var_3891);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_3898) {
        var_3900 = wp::int32(var_3899);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_3902 = wp::bit_and(var_3521, var_3901);
        var_3905 = wp::where(var_3902, var_3903, var_3904);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_3907 = wp::bit_and(var_3521, var_3906);
        var_3910 = wp::where(var_3907, var_3908, var_3909);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_3912 = wp::bit_and(var_3521, var_3911);
        var_3915 = wp::where(var_3912, var_3913, var_3914);
        var_3916 = wp::vec_t<3, wp::float32>(var_3905, var_3910, var_3915);
        // box_size,                                                                          <L 1204>
        var_3917 = wp::cw_mul(var_3916, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_3917, var_3890, var_3918);
        // dif = box_pt - pos                                                                 <L 1209>
        var_3919 = wp::sub(var_3917, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_3920 = wp::extract(var_box_size, var_3890);
        var_3921 = wp::neg(var_3920);
        var_3922 = wp::extract(var_3919, var_3890);
        var_3923 = wp::mul(var_3921, var_3922);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_3924 = wp::dot(var_4, var_3919);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_3925 = wp::extract(var_box_size, var_3890);
        var_3926 = wp::extract(var_box_size, var_3890);
        var_3927 = wp::mul(var_3925, var_3926);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_3928 = wp::extract(var_box_size, var_3890);
        var_3929 = wp::neg(var_3928);
        var_3930 = wp::extract(var_4, var_3890);
        var_3931 = wp::mul(var_3929, var_3930);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_3932 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_3933 = wp::mul(var_3927, var_3932);
        var_3934 = wp::mul(var_3931, var_3931);
        var_3935 = wp::sub(var_3933, var_3934);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_3936 = wp::abs(var_3935);
        var_3937 = (var_3936 < var_275);
        if (var_3937) {
            // continue                                                                       <L 1218>
        }
        var_3939 = wp::where(var_3937, var_3938, var_3898);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_3939) {
            var_3941 = wp::div(var_3940, var_3935);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_3942 = wp::mul(var_3932, var_3923);
            var_3943 = wp::mul(var_3931, var_3924);
            var_3944 = wp::sub(var_3942, var_3943);
            var_3945 = wp::mul(var_3944, var_3941);
            var_3946 = wp::float32(var_3945);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_3947 = wp::mul(var_3927, var_3924);
            var_3948 = wp::mul(var_3931, var_3923);
            var_3949 = wp::sub(var_3947, var_3948);
            var_3950 = wp::mul(var_3949, var_3941);
            var_3951 = wp::float32(var_3950);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_3953 = wp::int32(var_3952);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_3955 = wp::int32(var_3954);
            // if x1 > 1:                                                                     <L 1229>
            var_3957 = (var_3946 > var_3956);
            if (var_3957) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_3960 = wp::sub(var_3924, var_3931);
                var_3961 = safe_div_0(var_3960, var_3932);
            }
            if (!var_3957) {
                // elif x1 < -1:                                                              <L 1233>
                var_3963 = (var_3946 < var_3962);
                if (var_3963) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_3966 = wp::add(var_3924, var_3931);
                    var_3967 = safe_div_0(var_3966, var_3932);
                }
                var_3968 = wp::where(var_3963, var_3964, var_3946);
                var_3969 = wp::where(var_3963, var_3967, var_3951);
                var_3970 = wp::where(var_3963, var_3965, var_3953);
            }
            var_3971 = wp::where(var_3957, var_3958, var_3968);
            var_3972 = wp::where(var_3957, var_3961, var_3969);
            var_3973 = wp::where(var_3957, var_3959, var_3970);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_3975 = (var_3972 > var_3974);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_3976 = var_3975;
            if (!var_3976) {
                var_3978 = (var_3972 < var_3977);
                var_3976 = var_3976 || var_3978;
            }
            if (var_3976) {
                // if x2_over:                                                                <L 1240>
                if (var_3975) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_3981 = wp::sub(var_3923, var_3931);
                    var_3982 = safe_div_0(var_3981, var_3927);
                }
                if (!var_3975) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_3985 = wp::add(var_3923, var_3931);
                    var_3986 = safe_div_0(var_3985, var_3927);
                }
                var_3987 = wp::where(var_3975, var_3982, var_3986);
                var_3988 = wp::where(var_3975, var_3979, var_3983);
                var_3989 = wp::where(var_3975, var_3980, var_3984);
                // if x1 > 1:                                                                 <L 1249>
                var_3991 = (var_3987 > var_3990);
                if (var_3991) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_3991) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_3995 = (var_3987 < var_3994);
                    if (var_3995) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_3998 = wp::where(var_3995, var_3996, var_3987);
                    var_3999 = wp::where(var_3995, var_3997, var_3973);
                }
                var_4000 = wp::where(var_3991, var_3992, var_3998);
                var_4001 = wp::where(var_3991, var_3993, var_3999);
            }
            var_4002 = wp::where(var_3976, var_4000, var_3971);
            var_4003 = wp::where(var_3976, var_3988, var_3972);
            var_4004 = wp::where(var_3976, var_4001, var_3973);
            var_4005 = wp::where(var_3976, var_3989, var_3955);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_4006 = wp::mul(var_4, var_4003);
            var_4007 = wp::sub(var_3919, var_4006);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_4008 = wp::extract(var_box_size, var_3890);
            var_4009 = wp::mul(var_4008, var_4002);
            wp::add_inplace(var_4007, var_3890, var_4009);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_4011 = wp::mul(var_4004, var_4010);
            var_4012 = wp::add(var_4011, var_4005);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_4013 = wp::length_sq(var_4007);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_4014 = wp::sub(var_3866, var_275);
            var_4015 = (var_4013 < var_4014);
            if (var_4015) {
                // bestdist = dif_sq                                                          <L 1264>
                var_4016 = wp::copy(var_4013);
                // bestsegmentpos = x2                                                        <L 1265>
                var_4017 = wp::copy(var_4003);
                // bestboxpos = x1                                                            <L 1266>
                var_4018 = wp::copy(var_4002);
                // c2 = ct // 6                                                               <L 1268>
                var_4020 = wp::floordiv(var_4012, var_4019);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_4022 = wp::lshift(var_4021, var_3890);
                var_4023 = wp::mul(var_4022, var_4020);
                var_4024 = wp::add(var_3521, var_4023);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_4025 = wp::copy(var_3890);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_4026 = wp::copy(var_4012);
            }
            var_4027 = wp::where(var_4015, var_4016, var_3866);
            var_4028 = wp::where(var_4015, var_4017, var_3867);
            var_4029 = wp::where(var_4015, var_4026, var_3868);
            var_4030 = wp::where(var_4015, var_4024, var_3869);
            var_4031 = wp::where(var_4015, var_4025, var_3870);
            var_4032 = wp::where(var_4015, var_4018, var_3871);
            var_4033 = wp::where(var_4015, var_4020, var_3900);
        }
        var_4034 = wp::where(var_3939, var_4027, var_3866);
        var_4035 = wp::where(var_3939, var_4028, var_3867);
        var_4036 = wp::where(var_3939, var_4029, var_3868);
        var_4037 = wp::where(var_3939, var_4030, var_3869);
        var_4038 = wp::where(var_3939, var_4031, var_3870);
        var_4039 = wp::where(var_3939, var_4032, var_3871);
        var_4040 = wp::where(var_3939, var_4033, var_3900);
        var_4041 = wp::where(var_3939, var_4007, var_3919);
        var_4042 = wp::where(var_3939, var_3941, var_3881);
        var_4043 = wp::where(var_3939, var_4002, var_3882);
        var_4044 = wp::where(var_3939, var_4003, var_3883);
        var_4045 = wp::where(var_3939, var_4004, var_3884);
        var_4046 = wp::where(var_3939, var_4005, var_3885);
        var_4047 = wp::where(var_3939, var_3975, var_3886);
        var_4048 = wp::where(var_3939, var_4012, var_3887);
        var_4049 = wp::where(var_3939, var_4013, var_3888);
    }
    var_4050 = wp::where(var_3898, var_4034, var_3866);
    var_4051 = wp::where(var_3898, var_4035, var_3867);
    var_4052 = wp::where(var_3898, var_4036, var_3868);
    var_4053 = wp::where(var_3898, var_4037, var_3869);
    var_4054 = wp::where(var_3898, var_4038, var_3870);
    var_4055 = wp::where(var_3898, var_4039, var_3871);
    var_4056 = wp::where(var_3898, var_4040, var_3872);
    var_4057 = wp::where(var_3898, var_3917, var_3873);
    var_4058 = wp::where(var_3898, var_4041, var_3874);
    var_4059 = wp::where(var_3898, var_3923, var_3875);
    var_4060 = wp::where(var_3898, var_3924, var_3876);
    var_4061 = wp::where(var_3898, var_3927, var_3877);
    var_4062 = wp::where(var_3898, var_3931, var_3878);
    var_4063 = wp::where(var_3898, var_3932, var_3879);
    var_4064 = wp::where(var_3898, var_3935, var_3880);
    var_4065 = wp::where(var_3898, var_4042, var_3881);
    var_4066 = wp::where(var_3898, var_4043, var_3882);
    var_4067 = wp::where(var_3898, var_4044, var_3883);
    var_4068 = wp::where(var_3898, var_4045, var_3884);
    var_4069 = wp::where(var_3898, var_4046, var_3885);
    var_4070 = wp::where(var_3898, var_4047, var_3886);
    var_4071 = wp::where(var_3898, var_4048, var_3887);
    var_4072 = wp::where(var_3898, var_4049, var_3888);
    var_4073 = wp::where(var_3898, var_3939, var_3898);
    // for j in range(3):                                                                     <L 1191>
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_4078 = wp::lshift(var_4077, var_4075);
    var_4079 = wp::bit_and(var_4074, var_4078);
    var_4081 = (var_4079 != var_4080);
    if (var_4081) {
        // continue                                                                           <L 1193>
    }
    var_4083 = wp::where(var_4081, var_4082, var_4076);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_4083) {
        var_4085 = wp::int32(var_4084);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_4087 = wp::bit_and(var_4074, var_4086);
        var_4090 = wp::where(var_4087, var_4088, var_4089);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_4092 = wp::bit_and(var_4074, var_4091);
        var_4095 = wp::where(var_4092, var_4093, var_4094);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_4097 = wp::bit_and(var_4074, var_4096);
        var_4100 = wp::where(var_4097, var_4098, var_4099);
        var_4101 = wp::vec_t<3, wp::float32>(var_4090, var_4095, var_4100);
        // box_size,                                                                          <L 1204>
        var_4102 = wp::cw_mul(var_4101, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_4102, var_4075, var_4103);
        // dif = box_pt - pos                                                                 <L 1209>
        var_4104 = wp::sub(var_4102, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_4105 = wp::extract(var_box_size, var_4075);
        var_4106 = wp::neg(var_4105);
        var_4107 = wp::extract(var_4104, var_4075);
        var_4108 = wp::mul(var_4106, var_4107);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_4109 = wp::dot(var_4, var_4104);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_4110 = wp::extract(var_box_size, var_4075);
        var_4111 = wp::extract(var_box_size, var_4075);
        var_4112 = wp::mul(var_4110, var_4111);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_4113 = wp::extract(var_box_size, var_4075);
        var_4114 = wp::neg(var_4113);
        var_4115 = wp::extract(var_4, var_4075);
        var_4116 = wp::mul(var_4114, var_4115);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_4117 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_4118 = wp::mul(var_4112, var_4117);
        var_4119 = wp::mul(var_4116, var_4116);
        var_4120 = wp::sub(var_4118, var_4119);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_4121 = wp::abs(var_4120);
        var_4122 = (var_4121 < var_275);
        if (var_4122) {
            // continue                                                                       <L 1218>
        }
        var_4124 = wp::where(var_4122, var_4123, var_4083);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_4124) {
            var_4126 = wp::div(var_4125, var_4120);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_4127 = wp::mul(var_4117, var_4108);
            var_4128 = wp::mul(var_4116, var_4109);
            var_4129 = wp::sub(var_4127, var_4128);
            var_4130 = wp::mul(var_4129, var_4126);
            var_4131 = wp::float32(var_4130);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_4132 = wp::mul(var_4112, var_4109);
            var_4133 = wp::mul(var_4116, var_4108);
            var_4134 = wp::sub(var_4132, var_4133);
            var_4135 = wp::mul(var_4134, var_4126);
            var_4136 = wp::float32(var_4135);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_4138 = wp::int32(var_4137);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_4140 = wp::int32(var_4139);
            // if x1 > 1:                                                                     <L 1229>
            var_4142 = (var_4131 > var_4141);
            if (var_4142) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_4145 = wp::sub(var_4109, var_4116);
                var_4146 = safe_div_0(var_4145, var_4117);
            }
            if (!var_4142) {
                // elif x1 < -1:                                                              <L 1233>
                var_4148 = (var_4131 < var_4147);
                if (var_4148) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_4151 = wp::add(var_4109, var_4116);
                    var_4152 = safe_div_0(var_4151, var_4117);
                }
                var_4153 = wp::where(var_4148, var_4149, var_4131);
                var_4154 = wp::where(var_4148, var_4152, var_4136);
                var_4155 = wp::where(var_4148, var_4150, var_4138);
            }
            var_4156 = wp::where(var_4142, var_4143, var_4153);
            var_4157 = wp::where(var_4142, var_4146, var_4154);
            var_4158 = wp::where(var_4142, var_4144, var_4155);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_4160 = (var_4157 > var_4159);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_4161 = var_4160;
            if (!var_4161) {
                var_4163 = (var_4157 < var_4162);
                var_4161 = var_4161 || var_4163;
            }
            if (var_4161) {
                // if x2_over:                                                                <L 1240>
                if (var_4160) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_4166 = wp::sub(var_4108, var_4116);
                    var_4167 = safe_div_0(var_4166, var_4112);
                }
                if (!var_4160) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_4170 = wp::add(var_4108, var_4116);
                    var_4171 = safe_div_0(var_4170, var_4112);
                }
                var_4172 = wp::where(var_4160, var_4167, var_4171);
                var_4173 = wp::where(var_4160, var_4164, var_4168);
                var_4174 = wp::where(var_4160, var_4165, var_4169);
                // if x1 > 1:                                                                 <L 1249>
                var_4176 = (var_4172 > var_4175);
                if (var_4176) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_4176) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_4180 = (var_4172 < var_4179);
                    if (var_4180) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_4183 = wp::where(var_4180, var_4181, var_4172);
                    var_4184 = wp::where(var_4180, var_4182, var_4158);
                }
                var_4185 = wp::where(var_4176, var_4177, var_4183);
                var_4186 = wp::where(var_4176, var_4178, var_4184);
            }
            var_4187 = wp::where(var_4161, var_4185, var_4156);
            var_4188 = wp::where(var_4161, var_4173, var_4157);
            var_4189 = wp::where(var_4161, var_4186, var_4158);
            var_4190 = wp::where(var_4161, var_4174, var_4140);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_4191 = wp::mul(var_4, var_4188);
            var_4192 = wp::sub(var_4104, var_4191);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_4193 = wp::extract(var_box_size, var_4075);
            var_4194 = wp::mul(var_4193, var_4187);
            wp::add_inplace(var_4192, var_4075, var_4194);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_4196 = wp::mul(var_4189, var_4195);
            var_4197 = wp::add(var_4196, var_4190);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_4198 = wp::length_sq(var_4192);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_4199 = wp::sub(var_4050, var_275);
            var_4200 = (var_4198 < var_4199);
            if (var_4200) {
                // bestdist = dif_sq                                                          <L 1264>
                var_4201 = wp::copy(var_4198);
                // bestsegmentpos = x2                                                        <L 1265>
                var_4202 = wp::copy(var_4188);
                // bestboxpos = x1                                                            <L 1266>
                var_4203 = wp::copy(var_4187);
                // c2 = ct // 6                                                               <L 1268>
                var_4205 = wp::floordiv(var_4197, var_4204);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_4207 = wp::lshift(var_4206, var_4075);
                var_4208 = wp::mul(var_4207, var_4205);
                var_4209 = wp::add(var_4074, var_4208);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_4210 = wp::copy(var_4075);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_4211 = wp::copy(var_4197);
            }
            var_4212 = wp::where(var_4200, var_4201, var_4050);
            var_4213 = wp::where(var_4200, var_4202, var_4051);
            var_4214 = wp::where(var_4200, var_4211, var_4052);
            var_4215 = wp::where(var_4200, var_4209, var_4053);
            var_4216 = wp::where(var_4200, var_4210, var_4054);
            var_4217 = wp::where(var_4200, var_4203, var_4055);
            var_4218 = wp::where(var_4200, var_4205, var_4085);
        }
        var_4219 = wp::where(var_4124, var_4212, var_4050);
        var_4220 = wp::where(var_4124, var_4213, var_4051);
        var_4221 = wp::where(var_4124, var_4214, var_4052);
        var_4222 = wp::where(var_4124, var_4215, var_4053);
        var_4223 = wp::where(var_4124, var_4216, var_4054);
        var_4224 = wp::where(var_4124, var_4217, var_4055);
        var_4225 = wp::where(var_4124, var_4218, var_4085);
        var_4226 = wp::where(var_4124, var_4192, var_4104);
        var_4227 = wp::where(var_4124, var_4126, var_4065);
        var_4228 = wp::where(var_4124, var_4187, var_4066);
        var_4229 = wp::where(var_4124, var_4188, var_4067);
        var_4230 = wp::where(var_4124, var_4189, var_4068);
        var_4231 = wp::where(var_4124, var_4190, var_4069);
        var_4232 = wp::where(var_4124, var_4160, var_4070);
        var_4233 = wp::where(var_4124, var_4197, var_4071);
        var_4234 = wp::where(var_4124, var_4198, var_4072);
    }
    var_4235 = wp::where(var_4083, var_4219, var_4050);
    var_4236 = wp::where(var_4083, var_4220, var_4051);
    var_4237 = wp::where(var_4083, var_4221, var_4052);
    var_4238 = wp::where(var_4083, var_4222, var_4053);
    var_4239 = wp::where(var_4083, var_4223, var_4054);
    var_4240 = wp::where(var_4083, var_4224, var_4055);
    var_4241 = wp::where(var_4083, var_4225, var_4056);
    var_4242 = wp::where(var_4083, var_4102, var_4057);
    var_4243 = wp::where(var_4083, var_4226, var_4058);
    var_4244 = wp::where(var_4083, var_4108, var_4059);
    var_4245 = wp::where(var_4083, var_4109, var_4060);
    var_4246 = wp::where(var_4083, var_4112, var_4061);
    var_4247 = wp::where(var_4083, var_4116, var_4062);
    var_4248 = wp::where(var_4083, var_4117, var_4063);
    var_4249 = wp::where(var_4083, var_4120, var_4064);
    var_4250 = wp::where(var_4083, var_4227, var_4065);
    var_4251 = wp::where(var_4083, var_4228, var_4066);
    var_4252 = wp::where(var_4083, var_4229, var_4067);
    var_4253 = wp::where(var_4083, var_4230, var_4068);
    var_4254 = wp::where(var_4083, var_4231, var_4069);
    var_4255 = wp::where(var_4083, var_4232, var_4070);
    var_4256 = wp::where(var_4083, var_4233, var_4071);
    var_4257 = wp::where(var_4083, var_4234, var_4072);
    var_4258 = wp::where(var_4083, var_4124, var_4083);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_4262 = wp::lshift(var_4261, var_4259);
    var_4263 = wp::bit_and(var_4074, var_4262);
    var_4265 = (var_4263 != var_4264);
    if (var_4265) {
        // continue                                                                           <L 1193>
    }
    var_4267 = wp::where(var_4265, var_4266, var_4260);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_4267) {
        var_4269 = wp::int32(var_4268);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_4271 = wp::bit_and(var_4074, var_4270);
        var_4274 = wp::where(var_4271, var_4272, var_4273);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_4276 = wp::bit_and(var_4074, var_4275);
        var_4279 = wp::where(var_4276, var_4277, var_4278);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_4281 = wp::bit_and(var_4074, var_4280);
        var_4284 = wp::where(var_4281, var_4282, var_4283);
        var_4285 = wp::vec_t<3, wp::float32>(var_4274, var_4279, var_4284);
        // box_size,                                                                          <L 1204>
        var_4286 = wp::cw_mul(var_4285, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_4286, var_4259, var_4287);
        // dif = box_pt - pos                                                                 <L 1209>
        var_4288 = wp::sub(var_4286, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_4289 = wp::extract(var_box_size, var_4259);
        var_4290 = wp::neg(var_4289);
        var_4291 = wp::extract(var_4288, var_4259);
        var_4292 = wp::mul(var_4290, var_4291);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_4293 = wp::dot(var_4, var_4288);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_4294 = wp::extract(var_box_size, var_4259);
        var_4295 = wp::extract(var_box_size, var_4259);
        var_4296 = wp::mul(var_4294, var_4295);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_4297 = wp::extract(var_box_size, var_4259);
        var_4298 = wp::neg(var_4297);
        var_4299 = wp::extract(var_4, var_4259);
        var_4300 = wp::mul(var_4298, var_4299);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_4301 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_4302 = wp::mul(var_4296, var_4301);
        var_4303 = wp::mul(var_4300, var_4300);
        var_4304 = wp::sub(var_4302, var_4303);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_4305 = wp::abs(var_4304);
        var_4306 = (var_4305 < var_275);
        if (var_4306) {
            // continue                                                                       <L 1218>
        }
        var_4308 = wp::where(var_4306, var_4307, var_4267);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_4308) {
            var_4310 = wp::div(var_4309, var_4304);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_4311 = wp::mul(var_4301, var_4292);
            var_4312 = wp::mul(var_4300, var_4293);
            var_4313 = wp::sub(var_4311, var_4312);
            var_4314 = wp::mul(var_4313, var_4310);
            var_4315 = wp::float32(var_4314);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_4316 = wp::mul(var_4296, var_4293);
            var_4317 = wp::mul(var_4300, var_4292);
            var_4318 = wp::sub(var_4316, var_4317);
            var_4319 = wp::mul(var_4318, var_4310);
            var_4320 = wp::float32(var_4319);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_4322 = wp::int32(var_4321);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_4324 = wp::int32(var_4323);
            // if x1 > 1:                                                                     <L 1229>
            var_4326 = (var_4315 > var_4325);
            if (var_4326) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_4329 = wp::sub(var_4293, var_4300);
                var_4330 = safe_div_0(var_4329, var_4301);
            }
            if (!var_4326) {
                // elif x1 < -1:                                                              <L 1233>
                var_4332 = (var_4315 < var_4331);
                if (var_4332) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_4335 = wp::add(var_4293, var_4300);
                    var_4336 = safe_div_0(var_4335, var_4301);
                }
                var_4337 = wp::where(var_4332, var_4333, var_4315);
                var_4338 = wp::where(var_4332, var_4336, var_4320);
                var_4339 = wp::where(var_4332, var_4334, var_4322);
            }
            var_4340 = wp::where(var_4326, var_4327, var_4337);
            var_4341 = wp::where(var_4326, var_4330, var_4338);
            var_4342 = wp::where(var_4326, var_4328, var_4339);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_4344 = (var_4341 > var_4343);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_4345 = var_4344;
            if (!var_4345) {
                var_4347 = (var_4341 < var_4346);
                var_4345 = var_4345 || var_4347;
            }
            if (var_4345) {
                // if x2_over:                                                                <L 1240>
                if (var_4344) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_4350 = wp::sub(var_4292, var_4300);
                    var_4351 = safe_div_0(var_4350, var_4296);
                }
                if (!var_4344) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_4354 = wp::add(var_4292, var_4300);
                    var_4355 = safe_div_0(var_4354, var_4296);
                }
                var_4356 = wp::where(var_4344, var_4351, var_4355);
                var_4357 = wp::where(var_4344, var_4348, var_4352);
                var_4358 = wp::where(var_4344, var_4349, var_4353);
                // if x1 > 1:                                                                 <L 1249>
                var_4360 = (var_4356 > var_4359);
                if (var_4360) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_4360) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_4364 = (var_4356 < var_4363);
                    if (var_4364) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_4367 = wp::where(var_4364, var_4365, var_4356);
                    var_4368 = wp::where(var_4364, var_4366, var_4342);
                }
                var_4369 = wp::where(var_4360, var_4361, var_4367);
                var_4370 = wp::where(var_4360, var_4362, var_4368);
            }
            var_4371 = wp::where(var_4345, var_4369, var_4340);
            var_4372 = wp::where(var_4345, var_4357, var_4341);
            var_4373 = wp::where(var_4345, var_4370, var_4342);
            var_4374 = wp::where(var_4345, var_4358, var_4324);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_4375 = wp::mul(var_4, var_4372);
            var_4376 = wp::sub(var_4288, var_4375);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_4377 = wp::extract(var_box_size, var_4259);
            var_4378 = wp::mul(var_4377, var_4371);
            wp::add_inplace(var_4376, var_4259, var_4378);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_4380 = wp::mul(var_4373, var_4379);
            var_4381 = wp::add(var_4380, var_4374);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_4382 = wp::length_sq(var_4376);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_4383 = wp::sub(var_4235, var_275);
            var_4384 = (var_4382 < var_4383);
            if (var_4384) {
                // bestdist = dif_sq                                                          <L 1264>
                var_4385 = wp::copy(var_4382);
                // bestsegmentpos = x2                                                        <L 1265>
                var_4386 = wp::copy(var_4372);
                // bestboxpos = x1                                                            <L 1266>
                var_4387 = wp::copy(var_4371);
                // c2 = ct // 6                                                               <L 1268>
                var_4389 = wp::floordiv(var_4381, var_4388);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_4391 = wp::lshift(var_4390, var_4259);
                var_4392 = wp::mul(var_4391, var_4389);
                var_4393 = wp::add(var_4074, var_4392);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_4394 = wp::copy(var_4259);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_4395 = wp::copy(var_4381);
            }
            var_4396 = wp::where(var_4384, var_4385, var_4235);
            var_4397 = wp::where(var_4384, var_4386, var_4236);
            var_4398 = wp::where(var_4384, var_4395, var_4237);
            var_4399 = wp::where(var_4384, var_4393, var_4238);
            var_4400 = wp::where(var_4384, var_4394, var_4239);
            var_4401 = wp::where(var_4384, var_4387, var_4240);
            var_4402 = wp::where(var_4384, var_4389, var_4269);
        }
        var_4403 = wp::where(var_4308, var_4396, var_4235);
        var_4404 = wp::where(var_4308, var_4397, var_4236);
        var_4405 = wp::where(var_4308, var_4398, var_4237);
        var_4406 = wp::where(var_4308, var_4399, var_4238);
        var_4407 = wp::where(var_4308, var_4400, var_4239);
        var_4408 = wp::where(var_4308, var_4401, var_4240);
        var_4409 = wp::where(var_4308, var_4402, var_4269);
        var_4410 = wp::where(var_4308, var_4376, var_4288);
        var_4411 = wp::where(var_4308, var_4310, var_4250);
        var_4412 = wp::where(var_4308, var_4371, var_4251);
        var_4413 = wp::where(var_4308, var_4372, var_4252);
        var_4414 = wp::where(var_4308, var_4373, var_4253);
        var_4415 = wp::where(var_4308, var_4374, var_4254);
        var_4416 = wp::where(var_4308, var_4344, var_4255);
        var_4417 = wp::where(var_4308, var_4381, var_4256);
        var_4418 = wp::where(var_4308, var_4382, var_4257);
    }
    var_4419 = wp::where(var_4267, var_4403, var_4235);
    var_4420 = wp::where(var_4267, var_4404, var_4236);
    var_4421 = wp::where(var_4267, var_4405, var_4237);
    var_4422 = wp::where(var_4267, var_4406, var_4238);
    var_4423 = wp::where(var_4267, var_4407, var_4239);
    var_4424 = wp::where(var_4267, var_4408, var_4240);
    var_4425 = wp::where(var_4267, var_4409, var_4241);
    var_4426 = wp::where(var_4267, var_4286, var_4242);
    var_4427 = wp::where(var_4267, var_4410, var_4243);
    var_4428 = wp::where(var_4267, var_4292, var_4244);
    var_4429 = wp::where(var_4267, var_4293, var_4245);
    var_4430 = wp::where(var_4267, var_4296, var_4246);
    var_4431 = wp::where(var_4267, var_4300, var_4247);
    var_4432 = wp::where(var_4267, var_4301, var_4248);
    var_4433 = wp::where(var_4267, var_4304, var_4249);
    var_4434 = wp::where(var_4267, var_4411, var_4250);
    var_4435 = wp::where(var_4267, var_4412, var_4251);
    var_4436 = wp::where(var_4267, var_4413, var_4252);
    var_4437 = wp::where(var_4267, var_4414, var_4253);
    var_4438 = wp::where(var_4267, var_4415, var_4254);
    var_4439 = wp::where(var_4267, var_4416, var_4255);
    var_4440 = wp::where(var_4267, var_4417, var_4256);
    var_4441 = wp::where(var_4267, var_4418, var_4257);
    var_4442 = wp::where(var_4267, var_4308, var_4267);
    // if (i & (1 << j)) != 0:                                                                <L 1192>
    var_4446 = wp::lshift(var_4445, var_4443);
    var_4447 = wp::bit_and(var_4074, var_4446);
    var_4449 = (var_4447 != var_4448);
    if (var_4449) {
        // continue                                                                           <L 1193>
    }
    var_4451 = wp::where(var_4449, var_4450, var_4444);
    // c2 = wp.int32(-123)                                                                    <L 1195>
    if (var_4451) {
        var_4453 = wp::int32(var_4452);
        // box_pt = wp.cw_mul(                                                                <L 1198>
        // wp.vec3(                                                                           <L 1199>
        // wp.where(i & 1, 1.0, -1.0),                                                        <L 1200>
        var_4455 = wp::bit_and(var_4074, var_4454);
        var_4458 = wp::where(var_4455, var_4456, var_4457);
        // wp.where(i & 2, 1.0, -1.0),                                                        <L 1201>
        var_4460 = wp::bit_and(var_4074, var_4459);
        var_4463 = wp::where(var_4460, var_4461, var_4462);
        // wp.where(i & 4, 1.0, -1.0),                                                        <L 1202>
        var_4465 = wp::bit_and(var_4074, var_4464);
        var_4468 = wp::where(var_4465, var_4466, var_4467);
        var_4469 = wp::vec_t<3, wp::float32>(var_4458, var_4463, var_4468);
        // box_size,                                                                          <L 1204>
        var_4470 = wp::cw_mul(var_4469, var_box_size);
        // box_pt[j] = 0.0                                                                    <L 1206>
        wp::assign_inplace(var_4470, var_4443, var_4471);
        // dif = box_pt - pos                                                                 <L 1209>
        var_4472 = wp::sub(var_4470, var_2);
        // u = -box_size[j] * dif[j]                                                          <L 1211>
        var_4473 = wp::extract(var_box_size, var_4443);
        var_4474 = wp::neg(var_4473);
        var_4475 = wp::extract(var_4472, var_4443);
        var_4476 = wp::mul(var_4474, var_4475);
        // v = wp.dot(halfaxis, dif)                                                          <L 1212>
        var_4477 = wp::dot(var_4, var_4472);
        // ma = box_size[j] * box_size[j]                                                     <L 1213>
        var_4478 = wp::extract(var_box_size, var_4443);
        var_4479 = wp::extract(var_box_size, var_4443);
        var_4480 = wp::mul(var_4478, var_4479);
        // mb = -box_size[j] * halfaxis[j]                                                    <L 1214>
        var_4481 = wp::extract(var_box_size, var_4443);
        var_4482 = wp::neg(var_4481);
        var_4483 = wp::extract(var_4, var_4443);
        var_4484 = wp::mul(var_4482, var_4483);
        // mc = capsule_half_length * capsule_half_length                                     <L 1215>
        var_4485 = wp::mul(var_capsule_half_length, var_capsule_half_length);
        // det = ma * mc - mb * mb                                                            <L 1216>
        var_4486 = wp::mul(var_4480, var_4485);
        var_4487 = wp::mul(var_4484, var_4484);
        var_4488 = wp::sub(var_4486, var_4487);
        // if wp.abs(det) < MJ_MINVAL:                                                        <L 1217>
        var_4489 = wp::abs(var_4488);
        var_4490 = (var_4489 < var_275);
        if (var_4490) {
            // continue                                                                       <L 1218>
        }
        var_4492 = wp::where(var_4490, var_4491, var_4451);
        // idet = 1.0 / det                                                                   <L 1220>
        if (var_4492) {
            var_4494 = wp::div(var_4493, var_4488);
            // x1 = wp.float32((mc * u - mb * v) * idet)                                      <L 1223>
            var_4495 = wp::mul(var_4485, var_4476);
            var_4496 = wp::mul(var_4484, var_4477);
            var_4497 = wp::sub(var_4495, var_4496);
            var_4498 = wp::mul(var_4497, var_4494);
            var_4499 = wp::float32(var_4498);
            // x2 = wp.float32((ma * v - mb * u) * idet)                                      <L 1224>
            var_4500 = wp::mul(var_4480, var_4477);
            var_4501 = wp::mul(var_4484, var_4476);
            var_4502 = wp::sub(var_4500, var_4501);
            var_4503 = wp::mul(var_4502, var_4494);
            var_4504 = wp::float32(var_4503);
            // s1 = wp.int32(1)                                                               <L 1226>
            var_4506 = wp::int32(var_4505);
            // s2 = wp.int32(1)                                                               <L 1227>
            var_4508 = wp::int32(var_4507);
            // if x1 > 1:                                                                     <L 1229>
            var_4510 = (var_4499 > var_4509);
            if (var_4510) {
                // x1 = 1.0                                                                   <L 1230>
                // s1 = 2                                                                     <L 1231>
                // x2 = safe_div(v - mb, mc)                                                  <L 1232>
                var_4513 = wp::sub(var_4477, var_4484);
                var_4514 = safe_div_0(var_4513, var_4485);
            }
            if (!var_4510) {
                // elif x1 < -1:                                                              <L 1233>
                var_4516 = (var_4499 < var_4515);
                if (var_4516) {
                    // x1 = -1.0                                                              <L 1234>
                    // s1 = 0                                                                 <L 1235>
                    // x2 = safe_div(v + mb, mc)                                              <L 1236>
                    var_4519 = wp::add(var_4477, var_4484);
                    var_4520 = safe_div_0(var_4519, var_4485);
                }
                var_4521 = wp::where(var_4516, var_4517, var_4499);
                var_4522 = wp::where(var_4516, var_4520, var_4504);
                var_4523 = wp::where(var_4516, var_4518, var_4506);
            }
            var_4524 = wp::where(var_4510, var_4511, var_4521);
            var_4525 = wp::where(var_4510, var_4514, var_4522);
            var_4526 = wp::where(var_4510, var_4512, var_4523);
            // x2_over = x2 > 1.0                                                             <L 1238>
            var_4528 = (var_4525 > var_4527);
            // if x2_over or x2 < -1.0:                                                       <L 1239>
            var_4529 = var_4528;
            if (!var_4529) {
                var_4531 = (var_4525 < var_4530);
                var_4529 = var_4529 || var_4531;
            }
            if (var_4529) {
                // if x2_over:                                                                <L 1240>
                if (var_4528) {
                    // x2 = 1.0                                                               <L 1241>
                    // s2 = 2                                                                 <L 1242>
                    // x1 = safe_div(u - mb, ma)                                              <L 1243>
                    var_4534 = wp::sub(var_4476, var_4484);
                    var_4535 = safe_div_0(var_4534, var_4480);
                }
                if (!var_4528) {
                    // x2 = -1.0                                                              <L 1245>
                    // s2 = 0                                                                 <L 1246>
                    // x1 = safe_div(u + mb, ma)                                              <L 1247>
                    var_4538 = wp::add(var_4476, var_4484);
                    var_4539 = safe_div_0(var_4538, var_4480);
                }
                var_4540 = wp::where(var_4528, var_4535, var_4539);
                var_4541 = wp::where(var_4528, var_4532, var_4536);
                var_4542 = wp::where(var_4528, var_4533, var_4537);
                // if x1 > 1:                                                                 <L 1249>
                var_4544 = (var_4540 > var_4543);
                if (var_4544) {
                    // x1 = 1.0                                                               <L 1250>
                    // s1 = 2                                                                 <L 1251>
                }
                if (!var_4544) {
                    // elif x1 < -1:                                                          <L 1252>
                    var_4548 = (var_4540 < var_4547);
                    if (var_4548) {
                        // x1 = -1.0                                                          <L 1253>
                        // s1 = 0                                                             <L 1254>
                    }
                    var_4551 = wp::where(var_4548, var_4549, var_4540);
                    var_4552 = wp::where(var_4548, var_4550, var_4526);
                }
                var_4553 = wp::where(var_4544, var_4545, var_4551);
                var_4554 = wp::where(var_4544, var_4546, var_4552);
            }
            var_4555 = wp::where(var_4529, var_4553, var_4524);
            var_4556 = wp::where(var_4529, var_4541, var_4525);
            var_4557 = wp::where(var_4529, var_4554, var_4526);
            var_4558 = wp::where(var_4529, var_4542, var_4508);
            // dif -= halfaxis * x2                                                           <L 1256>
            var_4559 = wp::mul(var_4, var_4556);
            var_4560 = wp::sub(var_4472, var_4559);
            // dif[j] += box_size[j] * x1                                                     <L 1257>
            var_4561 = wp::extract(var_box_size, var_4443);
            var_4562 = wp::mul(var_4561, var_4555);
            wp::add_inplace(var_4560, var_4443, var_4562);
            // ct = s1 * 3 + s2                                                               <L 1260>
            var_4564 = wp::mul(var_4557, var_4563);
            var_4565 = wp::add(var_4564, var_4558);
            // dif_sq = wp.length_sq(dif)                                                     <L 1262>
            var_4566 = wp::length_sq(var_4560);
            // if dif_sq < bestdist - MJ_MINVAL:                                              <L 1263>
            var_4567 = wp::sub(var_4419, var_275);
            var_4568 = (var_4566 < var_4567);
            if (var_4568) {
                // bestdist = dif_sq                                                          <L 1264>
                var_4569 = wp::copy(var_4566);
                // bestsegmentpos = x2                                                        <L 1265>
                var_4570 = wp::copy(var_4556);
                // bestboxpos = x1                                                            <L 1266>
                var_4571 = wp::copy(var_4555);
                // c2 = ct // 6                                                               <L 1268>
                var_4573 = wp::floordiv(var_4565, var_4572);
                // clcorner = i + (1 << j) * c2  # index of closest box corner                <L 1270>
                var_4575 = wp::lshift(var_4574, var_4443);
                var_4576 = wp::mul(var_4575, var_4573);
                var_4577 = wp::add(var_4074, var_4576);
                // cledge = j  # axis index of closest box edge                               <L 1271>
                var_4578 = wp::copy(var_4443);
                // cltype = ct  # encoded collision configuration                             <L 1272>
                var_4579 = wp::copy(var_4565);
            }
            var_4580 = wp::where(var_4568, var_4569, var_4419);
            var_4581 = wp::where(var_4568, var_4570, var_4420);
            var_4582 = wp::where(var_4568, var_4579, var_4421);
            var_4583 = wp::where(var_4568, var_4577, var_4422);
            var_4584 = wp::where(var_4568, var_4578, var_4423);
            var_4585 = wp::where(var_4568, var_4571, var_4424);
            var_4586 = wp::where(var_4568, var_4573, var_4453);
        }
        var_4587 = wp::where(var_4492, var_4580, var_4419);
        var_4588 = wp::where(var_4492, var_4581, var_4420);
        var_4589 = wp::where(var_4492, var_4582, var_4421);
        var_4590 = wp::where(var_4492, var_4583, var_4422);
        var_4591 = wp::where(var_4492, var_4584, var_4423);
        var_4592 = wp::where(var_4492, var_4585, var_4424);
        var_4593 = wp::where(var_4492, var_4586, var_4453);
        var_4594 = wp::where(var_4492, var_4560, var_4472);
        var_4595 = wp::where(var_4492, var_4494, var_4434);
        var_4596 = wp::where(var_4492, var_4555, var_4435);
        var_4597 = wp::where(var_4492, var_4556, var_4436);
        var_4598 = wp::where(var_4492, var_4557, var_4437);
        var_4599 = wp::where(var_4492, var_4558, var_4438);
        var_4600 = wp::where(var_4492, var_4528, var_4439);
        var_4601 = wp::where(var_4492, var_4565, var_4440);
        var_4602 = wp::where(var_4492, var_4566, var_4441);
    }
    var_4603 = wp::where(var_4451, var_4587, var_4419);
    var_4604 = wp::where(var_4451, var_4588, var_4420);
    var_4605 = wp::where(var_4451, var_4589, var_4421);
    var_4606 = wp::where(var_4451, var_4590, var_4422);
    var_4607 = wp::where(var_4451, var_4591, var_4423);
    var_4608 = wp::where(var_4451, var_4592, var_4424);
    var_4609 = wp::where(var_4451, var_4593, var_4425);
    var_4610 = wp::where(var_4451, var_4470, var_4426);
    var_4611 = wp::where(var_4451, var_4594, var_4427);
    var_4612 = wp::where(var_4451, var_4476, var_4428);
    var_4613 = wp::where(var_4451, var_4477, var_4429);
    var_4614 = wp::where(var_4451, var_4480, var_4430);
    var_4615 = wp::where(var_4451, var_4484, var_4431);
    var_4616 = wp::where(var_4451, var_4485, var_4432);
    var_4617 = wp::where(var_4451, var_4488, var_4433);
    var_4618 = wp::where(var_4451, var_4595, var_4434);
    var_4619 = wp::where(var_4451, var_4596, var_4435);
    var_4620 = wp::where(var_4451, var_4597, var_4436);
    var_4621 = wp::where(var_4451, var_4598, var_4437);
    var_4622 = wp::where(var_4451, var_4599, var_4438);
    var_4623 = wp::where(var_4451, var_4600, var_4439);
    var_4624 = wp::where(var_4451, var_4601, var_4440);
    var_4625 = wp::where(var_4451, var_4602, var_4441);
    var_4626 = wp::where(var_4451, var_4492, var_4451);
    // best = wp.float32(0.0)                                                                 <L 1274>
    var_4628 = wp::float32(var_4627);
    // p = wp.vec2(pos.x, pos.y)                                                              <L 1276>
    var_4630 = wp::extract(var_2, var_4629);
    var_4632 = wp::extract(var_2, var_4631);
    var_4633 = wp::vec_t<2, wp::float32>(var_4630, var_4632);
    // dd = wp.vec2(halfaxis.x, halfaxis.y)                                                   <L 1277>
    var_4635 = wp::extract(var_4, var_4634);
    var_4637 = wp::extract(var_4, var_4636);
    var_4638 = wp::vec_t<2, wp::float32>(var_4635, var_4637);
    // s = wp.vec2(box_size[0], box_size[1])                                                  <L 1278>
    var_4640 = wp::extract(var_box_size, var_4639);
    var_4642 = wp::extract(var_box_size, var_4641);
    var_4643 = wp::vec_t<2, wp::float32>(var_4640, var_4642);
    // secondpos = wp.float32(-4.0)                                                           <L 1279>
    var_4645 = wp::float32(var_4644);
    // uu = dd.x * s.y                                                                        <L 1281>
    var_4647 = wp::extract(var_4638, var_4646);
    var_4649 = wp::extract(var_4643, var_4648);
    var_4650 = wp::mul(var_4647, var_4649);
    // vv = dd.y * s.x                                                                        <L 1282>
    var_4652 = wp::extract(var_4638, var_4651);
    var_4654 = wp::extract(var_4643, var_4653);
    var_4655 = wp::mul(var_4652, var_4654);
    // w_neg = dd.x * p.y - dd.y * p.x < 0                                                    <L 1283>
    var_4657 = wp::extract(var_4638, var_4656);
    var_4659 = wp::extract(var_4633, var_4658);
    var_4660 = wp::mul(var_4657, var_4659);
    var_4662 = wp::extract(var_4638, var_4661);
    var_4664 = wp::extract(var_4633, var_4663);
    var_4665 = wp::mul(var_4662, var_4664);
    var_4666 = wp::sub(var_4660, var_4665);
    var_4668 = (var_4666 < var_4667);
    // best = wp.float32(-1.0)                                                                <L 1285>
    var_4670 = wp::float32(var_4669);
    // ee1 = uu - vv                                                                          <L 1287>
    var_4671 = wp::sub(var_4650, var_4655);
    // ee2 = uu + vv                                                                          <L 1288>
    var_4672 = wp::add(var_4650, var_4655);
    // if wp.abs(ee1) > best:                                                                 <L 1290>
    var_4673 = wp::abs(var_4671);
    var_4674 = (var_4673 > var_4670);
    if (var_4674) {
        // best = wp.abs(ee1)                                                                 <L 1291>
        var_4675 = wp::abs(var_4671);
        // c1 = wp.where((ee1 < 0) == w_neg, 0, 3)                                            <L 1292>
        var_4677 = (var_4671 < var_4676);
        var_4678 = (var_4677 == var_4668);
        var_4681 = wp::where(var_4678, var_4679, var_4680);
    }
    var_4682 = wp::where(var_4674, var_4675, var_4670);
    // if wp.abs(ee2) > best:                                                                 <L 1294>
    var_4683 = wp::abs(var_4672);
    var_4684 = (var_4683 > var_4682);
    if (var_4684) {
        // best = wp.abs(ee2)                                                                 <L 1295>
        var_4685 = wp::abs(var_4672);
        // c1 = wp.where((ee2 > 0) == w_neg, 1, 2)                                            <L 1296>
        var_4687 = (var_4672 > var_4686);
        var_4688 = (var_4687 == var_4668);
        var_4691 = wp::where(var_4688, var_4689, var_4690);
    }
    var_4692 = wp::where(var_4684, var_4685, var_4682);
    var_4693 = wp::where(var_4684, var_4691, var_4681);
    // if cltype == -4:  # invalid type                                                       <L 1298>
    var_4695 = (var_4605 == var_4694);
    if (var_4695) {
        // return wp.vec2(MJ_MAXVAL), mat23f(), mat23f()                                      <L 1299>
        var_4697 = wp::vec_t<2, wp::float32>(var_4696);
        var_4698 = wp::mat_t<2, 3, wp::float32>();
        var_4699 = wp::mat_t<2, 3, wp::float32>();
        ret_0 = var_4697;
        ret_1 = var_4698;
        ret_2 = var_4699;
        return;
    }
    // if cltype >= 0 and cltype // 3 != 1:  # closest to a corner of the box                 <L 1301>
    var_4702 = (var_4605 >= var_4701);
    var_4700 = var_4702;
    if (var_4700) {
        var_4704 = wp::floordiv(var_4605, var_4703);
        var_4706 = (var_4704 != var_4705);
        var_4700 = var_4700 && var_4706;
    }
    if (var_4700) {
        // c1 = axisdir ^ clcorner                                                            <L 1302>
        var_4707 = wp::bit_xor(var_25, var_4606);
        // if c1 != 0 and c1 != 7:  # create second contact point                             <L 1307>
        var_4710 = (var_4707 != var_4709);
        var_4708 = var_4710;
        if (var_4708) {
            var_4712 = (var_4707 != var_4711);
            var_4708 = var_4708 && var_4712;
        }
        if (var_4708) {
            // if c1 == 1 or c1 == 2 or c1 == 4:                                              <L 1308>
            var_4715 = (var_4707 == var_4714);
            var_4713 = var_4715;
            if (!var_4713) {
                var_4717 = (var_4707 == var_4716);
                var_4713 = var_4713 || var_4717;
            }
            if (!var_4713) {
                var_4719 = (var_4707 == var_4718);
                var_4713 = var_4713 || var_4719;
            }
            if (var_4713) {
                // mul = 1                                                                    <L 1309>
            }
            if (!var_4713) {
                // mul = -1                                                                   <L 1311>
                // c1 = 7 - c1                                                                <L 1312>
                var_4723 = wp::sub(var_4722, var_4707);
            }
            var_4724 = wp::where(var_4713, var_4707, var_4723);
            var_4725 = wp::where(var_4713, var_4720, var_4721);
            // if c1 == 1:                                                                    <L 1317>
            var_4727 = (var_4724 == var_4726);
            if (var_4727) {
                // ax = 0                                                                     <L 1318>
                // ax1 = 1                                                                    <L 1319>
                // ax2 = 2                                                                    <L 1320>
            }
            if (!var_4727) {
                // elif c1 == 2:                                                              <L 1321>
                var_4732 = (var_4724 == var_4731);
                if (var_4732) {
                    // ax = 1                                                                 <L 1322>
                    // ax1 = 2                                                                <L 1323>
                    // ax2 = 0                                                                <L 1324>
                }
                if (!var_4732) {
                    // elif c1 == 4:                                                          <L 1325>
                    var_4737 = (var_4724 == var_4736);
                    if (var_4737) {
                        // ax = 2                                                             <L 1326>
                        // ax1 = 0                                                            <L 1327>
                        // ax2 = 1                                                            <L 1328>
                    }
                }
                var_4741 = wp::where(var_4732, var_4733, var_4738);
                var_4742 = wp::where(var_4732, var_4734, var_4739);
                var_4743 = wp::where(var_4732, var_4735, var_4740);
            }
            var_4744 = wp::where(var_4727, var_4728, var_4741);
            var_4745 = wp::where(var_4727, var_4729, var_4742);
            var_4746 = wp::where(var_4727, var_4730, var_4743);
            // if axis[ax] * axis[ax] > 0.5:  # second point along the edge of the box        <L 1330>
            var_4747 = wp::extract(var_3, var_4744);
            var_4748 = wp::extract(var_3, var_4744);
            var_4749 = wp::mul(var_4747, var_4748);
            var_4751 = (var_4749 > var_4750);
            if (var_4751) {
                // m = 2.0 * safe_div(box_size[ax], wp.abs(halfaxis[ax]))                     <L 1331>
                var_4753 = wp::extract(var_box_size, var_4744);
                var_4754 = wp::extract(var_4, var_4744);
                var_4755 = wp::abs(var_4754);
                var_4756 = safe_div_0(var_4753, var_4755);
                var_4757 = wp::mul(var_4752, var_4756);
                // secondpos = min(1.0 - wp.float32(mul) * bestsegmentpos, m)                 <L 1332>
                var_4759 = wp::float32(var_4725);
                var_4760 = wp::mul(var_4759, var_4604);
                var_4761 = wp::sub(var_4758, var_4760);
                var_4762 = wp::min(var_4761, var_4757);
            }
            if (!var_4751) {
                // m = 2.0 * min(                                                             <L 1335>
                // safe_div(box_size[ax1], wp.abs(halfaxis[ax1])),                            <L 1336>
                var_4764 = wp::extract(var_box_size, var_4745);
                var_4765 = wp::extract(var_4, var_4745);
                var_4766 = wp::abs(var_4765);
                var_4767 = safe_div_0(var_4764, var_4766);
                // safe_div(box_size[ax2], wp.abs(halfaxis[ax2])),                            <L 1337>
                var_4768 = wp::extract(var_box_size, var_4746);
                var_4769 = wp::extract(var_4, var_4746);
                var_4770 = wp::abs(var_4769);
                var_4771 = safe_div_0(var_4768, var_4770);
                var_4772 = wp::min(var_4767, var_4771);
                var_4773 = wp::mul(var_4763, var_4772);
                // secondpos = -min(1.0 + wp.float32(mul) * bestsegmentpos, m)                <L 1339>
                var_4775 = wp::float32(var_4725);
                var_4776 = wp::mul(var_4775, var_4604);
                var_4777 = wp::add(var_4774, var_4776);
                var_4778 = wp::min(var_4777, var_4773);
                var_4779 = wp::neg(var_4778);
            }
            var_4780 = wp::where(var_4751, var_4762, var_4779);
            var_4781 = wp::where(var_4751, var_4757, var_4773);
            // secondpos *= wp.float32(mul)                                                   <L 1340>
            var_4782 = wp::float32(var_4725);
            var_4783 = wp::mul(var_4780, var_4782);
        }
        var_4784 = wp::where(var_4708, var_4783, var_4645);
        var_4785 = wp::where(var_4708, var_4724, var_4707);
    }
    if (!var_4700) {
        // elif cltype >= 0 and cltype // 3 == 1:  # we are on box's edge                     <L 1342>
        var_4788 = (var_4605 >= var_4787);
        var_4786 = var_4788;
        if (var_4786) {
            var_4790 = wp::floordiv(var_4605, var_4789);
            var_4792 = (var_4790 == var_4791);
            var_4786 = var_4786 && var_4792;
        }
        if (var_4786) {
            // c1 = axisdir ^ clcorner                                                        <L 1347>
            var_4793 = wp::bit_xor(var_25, var_4606);
            // c1 &= 7 - (1 << cledge)  # mask out edge axis to determine configuration       <L 1348>
            var_4796 = wp::lshift(var_4795, var_4607);
            var_4797 = wp::sub(var_4794, var_4796);
            var_4798 = wp::bit_and(var_4793, var_4797);
            // if c1 == 1 or c1 == 2 or c1 == 4:  # create second contact point               <L 1350>
            var_4801 = (var_4798 == var_4800);
            var_4799 = var_4801;
            if (!var_4799) {
                var_4803 = (var_4798 == var_4802);
                var_4799 = var_4799 || var_4803;
            }
            if (!var_4799) {
                var_4805 = (var_4798 == var_4804);
                var_4799 = var_4799 || var_4805;
            }
            if (var_4799) {
                // if cledge == 0:                                                            <L 1351>
                var_4807 = (var_4607 == var_4806);
                if (var_4807) {
                    // ax1 = 1                                                                <L 1352>
                    // ax2 = 2                                                                <L 1353>
                }
                // if cledge == 1:                                                            <L 1354>
                var_4811 = (var_4607 == var_4810);
                if (var_4811) {
                    // ax1 = 2                                                                <L 1355>
                    // ax2 = 0                                                                <L 1356>
                }
                var_4814 = wp::where(var_4811, var_4812, var_4808);
                var_4815 = wp::where(var_4811, var_4813, var_4809);
                // if cledge == 2:                                                            <L 1357>
                var_4817 = (var_4607 == var_4816);
                if (var_4817) {
                    // ax1 = 0                                                                <L 1358>
                    // ax2 = 1                                                                <L 1359>
                }
                var_4820 = wp::where(var_4817, var_4818, var_4814);
                var_4821 = wp::where(var_4817, var_4819, var_4815);
                // ax = cledge                                                                <L 1360>
                var_4822 = wp::copy(var_4607);
                // if wp.abs(axis[ax1]) > wp.abs(axis[ax2]):                                  <L 1363>
                var_4823 = wp::extract(var_3, var_4820);
                var_4824 = wp::abs(var_4823);
                var_4825 = wp::extract(var_3, var_4821);
                var_4826 = wp::abs(var_4825);
                var_4827 = (var_4824 > var_4826);
                if (var_4827) {
                    // ax1 = ax2                                                              <L 1364>
                    var_4828 = wp::copy(var_4821);
                }
                var_4829 = wp::where(var_4827, var_4828, var_4820);
                // ax2 = 3 - ax - ax1                                                         <L 1365>
                var_4831 = wp::sub(var_4830, var_4822);
                var_4832 = wp::sub(var_4831, var_4829);
                // if c1 & (1 << ax2):                                                        <L 1368>
                var_4834 = wp::lshift(var_4833, var_4832);
                var_4835 = wp::bit_and(var_4798, var_4834);
                if (var_4835) {
                    // mul = 1                                                                <L 1369>
                    // secondpos = 1.0 - bestsegmentpos                                       <L 1370>
                    var_4838 = wp::sub(var_4837, var_4604);
                }
                if (!var_4835) {
                    // mul = -1                                                               <L 1372>
                    // secondpos = 1.0 + bestsegmentpos                                       <L 1373>
                    var_4841 = wp::add(var_4840, var_4604);
                }
                var_4842 = wp::where(var_4835, var_4838, var_4841);
                var_4843 = wp::where(var_4835, var_4836, var_4839);
                // e1 = 2.0 * safe_div(box_size[ax2], wp.abs(halfaxis[ax2]))                  <L 1378>
                var_4845 = wp::extract(var_box_size, var_4832);
                var_4846 = wp::extract(var_4, var_4832);
                var_4847 = wp::abs(var_4846);
                var_4848 = safe_div_0(var_4845, var_4847);
                var_4849 = wp::mul(var_4844, var_4848);
                // secondpos = min(e1, secondpos)                                             <L 1379>
                var_4850 = wp::min(var_4849, var_4842);
                // if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):               <L 1381>
                var_4852 = wp::lshift(var_4851, var_4822);
                var_4853 = wp::bit_and(var_25, var_4852);
                var_4855 = (var_4853 != var_4854);
                var_4857 = wp::lshift(var_4856, var_4832);
                var_4858 = wp::bit_and(var_4798, var_4857);
                var_4860 = (var_4858 != var_4859);
                var_4861 = (var_4855 == var_4860);
                if (var_4861) {
                    // e2 = 1.0 - bestboxpos                                                  <L 1382>
                    var_4863 = wp::sub(var_4862, var_4608);
                }
                if (!var_4861) {
                    // e2 = 1.0 + bestboxpos                                                  <L 1384>
                    var_4865 = wp::add(var_4864, var_4608);
                }
                var_4866 = wp::where(var_4861, var_4863, var_4865);
                // e1 = box_size[ax] * safe_div(e2, wp.abs(halfaxis[ax]))                     <L 1386>
                var_4867 = wp::extract(var_box_size, var_4822);
                var_4868 = wp::extract(var_4, var_4822);
                var_4869 = wp::abs(var_4868);
                var_4870 = safe_div_0(var_4866, var_4869);
                var_4871 = wp::mul(var_4867, var_4870);
                // secondpos = min(e1, secondpos)                                             <L 1388>
                var_4872 = wp::min(var_4871, var_4850);
                // secondpos *= wp.float32(mul)                                               <L 1389>
                var_4873 = wp::float32(var_4843);
                var_4874 = wp::mul(var_4872, var_4873);
            }
            var_4875 = wp::where(var_4799, var_4874, var_4645);
        }
        if (!var_4786) {
            // elif cltype < 0:                                                               <L 1391>
            var_4877 = (var_4605 < var_4876);
            if (var_4877) {
                // if clface != -1:  # create second contact point                            <L 1397>
                var_4879 = (var_219 != var_4878);
                if (var_4879) {
                    // mul = wp.where(cltype == -3, 1, -1)                                    <L 1398>
                    var_4881 = (var_4605 == var_4880);
                    var_4884 = wp::where(var_4881, var_4882, var_4883);
                    // secondpos = 2.0                                                        <L 1399>
                    // tmp1 = pos - halfaxis * wp.float32(mul)                                <L 1401>
                    var_4886 = wp::float32(var_4884);
                    var_4887 = wp::mul(var_4, var_4886);
                    var_4888 = wp::sub(var_2, var_4887);
                    // for i in range(3):                                                     <L 1403>
                    // if i != clface:                                                        <L 1404>
                    var_4890 = (var_4889 != var_219);
                    if (var_4890) {
                        // ha_r = safe_div(wp.float32(mul), halfaxis[i])                      <L 1405>
                        var_4891 = wp::float32(var_4884);
                        var_4892 = wp::extract(var_4, var_4889);
                        var_4893 = safe_div_0(var_4891, var_4892);
                        // e1 = (box_size[i] - tmp1[i]) * ha_r                                <L 1406>
                        var_4894 = wp::extract(var_box_size, var_4889);
                        var_4895 = wp::extract(var_4888, var_4889);
                        var_4896 = wp::sub(var_4894, var_4895);
                        var_4897 = wp::mul(var_4896, var_4893);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1407>
                        var_4900 = (var_4899 < var_4897);
                        var_4898 = var_4900;
                        if (var_4898) {
                            var_4901 = (var_4897 < var_4885);
                            var_4898 = var_4898 && var_4901;
                        }
                        if (var_4898) {
                            // secondpos = e1                                                 <L 1408>
                            var_4902 = wp::copy(var_4897);
                        }
                        var_4903 = wp::where(var_4898, var_4902, var_4885);
                        // e1 = (-box_size[i] - tmp1[i]) * ha_r                               <L 1410>
                        var_4904 = wp::extract(var_box_size, var_4889);
                        var_4905 = wp::neg(var_4904);
                        var_4906 = wp::extract(var_4888, var_4889);
                        var_4907 = wp::sub(var_4905, var_4906);
                        var_4908 = wp::mul(var_4907, var_4893);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1411>
                        var_4911 = (var_4910 < var_4908);
                        var_4909 = var_4911;
                        if (var_4909) {
                            var_4912 = (var_4908 < var_4903);
                            var_4909 = var_4909 && var_4912;
                        }
                        if (var_4909) {
                            // secondpos = e1                                                 <L 1412>
                            var_4913 = wp::copy(var_4908);
                        }
                        var_4914 = wp::where(var_4909, var_4913, var_4903);
                    }
                    var_4915 = wp::where(var_4890, var_4914, var_4885);
                    // if i != clface:                                                        <L 1404>
                    var_4917 = (var_4916 != var_219);
                    if (var_4917) {
                        // ha_r = safe_div(wp.float32(mul), halfaxis[i])                      <L 1405>
                        var_4918 = wp::float32(var_4884);
                        var_4919 = wp::extract(var_4, var_4916);
                        var_4920 = safe_div_0(var_4918, var_4919);
                        // e1 = (box_size[i] - tmp1[i]) * ha_r                                <L 1406>
                        var_4921 = wp::extract(var_box_size, var_4916);
                        var_4922 = wp::extract(var_4888, var_4916);
                        var_4923 = wp::sub(var_4921, var_4922);
                        var_4924 = wp::mul(var_4923, var_4920);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1407>
                        var_4927 = (var_4926 < var_4924);
                        var_4925 = var_4927;
                        if (var_4925) {
                            var_4928 = (var_4924 < var_4915);
                            var_4925 = var_4925 && var_4928;
                        }
                        if (var_4925) {
                            // secondpos = e1                                                 <L 1408>
                            var_4929 = wp::copy(var_4924);
                        }
                        var_4930 = wp::where(var_4925, var_4929, var_4915);
                        // e1 = (-box_size[i] - tmp1[i]) * ha_r                               <L 1410>
                        var_4931 = wp::extract(var_box_size, var_4916);
                        var_4932 = wp::neg(var_4931);
                        var_4933 = wp::extract(var_4888, var_4916);
                        var_4934 = wp::sub(var_4932, var_4933);
                        var_4935 = wp::mul(var_4934, var_4920);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1411>
                        var_4938 = (var_4937 < var_4935);
                        var_4936 = var_4938;
                        if (var_4936) {
                            var_4939 = (var_4935 < var_4930);
                            var_4936 = var_4936 && var_4939;
                        }
                        if (var_4936) {
                            // secondpos = e1                                                 <L 1412>
                            var_4940 = wp::copy(var_4935);
                        }
                        var_4941 = wp::where(var_4936, var_4940, var_4930);
                    }
                    var_4942 = wp::where(var_4917, var_4941, var_4915);
                    var_4943 = wp::where(var_4917, var_4920, var_4893);
                    var_4944 = wp::where(var_4917, var_4935, var_4908);
                    // if i != clface:                                                        <L 1404>
                    var_4946 = (var_4945 != var_219);
                    if (var_4946) {
                        // ha_r = safe_div(wp.float32(mul), halfaxis[i])                      <L 1405>
                        var_4947 = wp::float32(var_4884);
                        var_4948 = wp::extract(var_4, var_4945);
                        var_4949 = safe_div_0(var_4947, var_4948);
                        // e1 = (box_size[i] - tmp1[i]) * ha_r                                <L 1406>
                        var_4950 = wp::extract(var_box_size, var_4945);
                        var_4951 = wp::extract(var_4888, var_4945);
                        var_4952 = wp::sub(var_4950, var_4951);
                        var_4953 = wp::mul(var_4952, var_4949);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1407>
                        var_4956 = (var_4955 < var_4953);
                        var_4954 = var_4956;
                        if (var_4954) {
                            var_4957 = (var_4953 < var_4942);
                            var_4954 = var_4954 && var_4957;
                        }
                        if (var_4954) {
                            // secondpos = e1                                                 <L 1408>
                            var_4958 = wp::copy(var_4953);
                        }
                        var_4959 = wp::where(var_4954, var_4958, var_4942);
                        // e1 = (-box_size[i] - tmp1[i]) * ha_r                               <L 1410>
                        var_4960 = wp::extract(var_box_size, var_4945);
                        var_4961 = wp::neg(var_4960);
                        var_4962 = wp::extract(var_4888, var_4945);
                        var_4963 = wp::sub(var_4961, var_4962);
                        var_4964 = wp::mul(var_4963, var_4949);
                        // if 0 < e1 and e1 < secondpos:                                      <L 1411>
                        var_4967 = (var_4966 < var_4964);
                        var_4965 = var_4967;
                        if (var_4965) {
                            var_4968 = (var_4964 < var_4959);
                            var_4965 = var_4965 && var_4968;
                        }
                        if (var_4965) {
                            // secondpos = e1                                                 <L 1412>
                            var_4969 = wp::copy(var_4964);
                        }
                        var_4970 = wp::where(var_4965, var_4969, var_4959);
                    }
                    var_4971 = wp::where(var_4946, var_4970, var_4942);
                    var_4972 = wp::where(var_4946, var_4949, var_4943);
                    var_4973 = wp::where(var_4946, var_4964, var_4944);
                    // secondpos *= wp.float32(mul)                                           <L 1414>
                    var_4974 = wp::float32(var_4884);
                    var_4975 = wp::mul(var_4971, var_4974);
                }
                var_4976 = wp::where(var_4879, var_4945, var_4074);
                var_4977 = wp::where(var_4879, var_4975, var_4645);
            }
            var_4978 = wp::where(var_4877, var_4976, var_4074);
            var_4979 = wp::where(var_4877, var_4977, var_4645);
        }
        var_4980 = wp::where(var_4786, var_4074, var_4978);
        var_4981 = wp::where(var_4786, var_4875, var_4979);
        var_4982 = wp::where(var_4786, var_4798, var_4693);
        var_4983 = wp::where(var_4786, var_4843, var_4884);
        var_4984 = wp::where(var_4786, var_4871, var_4973);
    }
    var_4985 = wp::where(var_4700, var_4074, var_4980);
    var_4986 = wp::where(var_4700, var_4784, var_4981);
    var_4987 = wp::where(var_4700, var_4785, var_4982);
    var_4988 = wp::where(var_4700, var_4725, var_4983);
    var_4989 = wp::where(var_4700, var_4744, var_4822);
    var_4990 = wp::where(var_4700, var_4745, var_4829);
    var_4991 = wp::where(var_4700, var_4746, var_4832);
    // s1_pos_l = pos + halfaxis * bestsegmentpos                                             <L 1417>
    var_4992 = wp::mul(var_4, var_4604);
    var_4993 = wp::add(var_2, var_4992);
    // s1_pos_g = box_rot @ s1_pos_l + box_pos                                                <L 1418>
    var_4994 = wp::mul(var_box_rot, var_4993);
    var_4995 = wp::add(var_4994, var_box_pos);
    // dist1, pos1, normal1 = sphere_box(s1_pos_g, capsule_radius, box_pos, box_rot, box_size)       <L 1421>
    sphere_box_0(var_4995, var_capsule_radius, var_box_pos, var_box_rot, var_box_size, var_4996, var_4997, var_4998);
    // if secondpos > -3:  # secondpos was modified                                           <L 1423>
    var_5000 = (var_4986 > var_4999);
    if (var_5000) {
        // s2_pos_l = pos + halfaxis * (secondpos + bestsegmentpos)                           <L 1424>
        var_5001 = wp::add(var_4986, var_4604);
        var_5002 = wp::mul(var_4, var_5001);
        var_5003 = wp::add(var_2, var_5002);
        // s2_pos_g = box_rot @ s2_pos_l + box_pos                                            <L 1425>
        var_5004 = wp::mul(var_box_rot, var_5003);
        var_5005 = wp::add(var_5004, var_box_pos);
        // dist2, pos2, normal2 = sphere_box(s2_pos_g, capsule_radius, box_pos, box_rot, box_size)       <L 1428>
        sphere_box_0(var_5005, var_capsule_radius, var_box_pos, var_box_rot, var_box_size, var_5006, var_5007, var_5008);
    }
    if (!var_5000) {
        // dist2 = MJ_MAXVAL                                                                  <L 1430>
        var_5009 = wp::copy(var_4696);
        // pos2 = wp.vec3()                                                                   <L 1431>
        var_5010 = wp::vec_t<3, wp::float32>();
        // normal2 = wp.vec3()                                                                <L 1432>
        var_5011 = wp::vec_t<3, wp::float32>();
    }
    var_5012 = wp::where(var_5000, var_5006, var_5009);
    var_5013 = wp::where(var_5000, var_5007, var_5010);
    var_5014 = wp::where(var_5000, var_5008, var_5011);
    // return (                                                                               <L 1434>
    // wp.vec2(dist1, dist2),                                                                 <L 1435>
    var_5015 = wp::vec_t<2, wp::float32>(var_4996, var_5012);
    // mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]),                          <L 1436>
    var_5017 = wp::extract(var_4997, var_5016);
    var_5019 = wp::extract(var_4997, var_5018);
    var_5021 = wp::extract(var_4997, var_5020);
    var_5023 = wp::extract(var_5013, var_5022);
    var_5025 = wp::extract(var_5013, var_5024);
    var_5027 = wp::extract(var_5013, var_5026);
    var_5028 = wp::mat_t<2, 3, wp::float32>({var_5017, var_5019, var_5021, var_5023, var_5025, var_5027});
    // mat23f(normal1[0], normal1[1], normal1[2], normal2[0], normal2[1], normal2[2]),        <L 1437>
    var_5030 = wp::extract(var_4998, var_5029);
    var_5032 = wp::extract(var_4998, var_5031);
    var_5034 = wp::extract(var_4998, var_5033);
    var_5036 = wp::extract(var_5014, var_5035);
    var_5038 = wp::extract(var_5014, var_5037);
    var_5040 = wp::extract(var_5014, var_5039);
    var_5041 = wp::mat_t<2, 3, wp::float32>({var_5030, var_5032, var_5034, var_5036, var_5038, var_5040});
    ret_0 = var_5015;
    ret_1 = var_5028;
    ret_2 = var_5041;
    return;
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:1092
static CUDA_CALLABLE void capsule_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_cap,
    Geom_4022bfa7 var_box,
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
    const wp::int32 var_22 = 1;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32>* var_25;
    wp::mat_t<3, 3, wp::float32>* var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::vec_t<2, wp::float32> var_28;
    wp::mat_t<2, 3, wp::float32> var_29;
    wp::mat_t<2, 3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::mat_t<3, 3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::mat_t<3, 3, wp::float32> var_39;
    wp::int32 var_40;
    const wp::int32 var_41 = 1;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::mat_t<3, 3, wp::float32> var_45;
    wp::int32 var_46;
    //---------
    // forward
    // def capsule_box_wrapper(                                                               <L 1093>
    // axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])                            <L 1130>
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
    // dist, pos, normal = capsule_box(                                                       <L 1133>
    // cap.pos,                                                                               <L 1134>
    var_16 = &((var_cap).pos);
    // axis,                                                                                  <L 1135>
    // cap.size[0],  # capsule radius                                                         <L 1136>
    var_17 = &((var_cap).size);
    var_20 = wp::load(var_17);
    var_19 = wp::extract(var_20, var_18);
    // cap.size[1],  # capsule half length                                                    <L 1137>
    var_21 = &((var_cap).size);
    var_24 = wp::load(var_21);
    var_23 = wp::extract(var_24, var_22);
    // box.pos,                                                                               <L 1138>
    var_25 = &((var_box).pos);
    // box.rot,                                                                               <L 1139>
    var_26 = &((var_box).rot);
    // box.size,                                                                              <L 1140>
    var_27 = &((var_box).size);
    var_31 = wp::load(var_16);
    var_32 = wp::load(var_25);
    var_33 = wp::load(var_26);
    var_34 = wp::load(var_27);
    capsule_box_0(var_31, var_15, var_19, var_23, var_32, var_33, var_34, var_28, var_29, var_30);
    // for i in range(2):                                                                     <L 1144>
    // write_contact(                                                                         <L 1145>
    // naconmax_in,                                                                           <L 1146>
    // i,                                                                                     <L 1147>
    // dist[i],                                                                               <L 1148>
    var_36 = wp::extract(var_28, var_35);
    // pos[i],                                                                                <L 1149>
    var_37 = wp::extract(var_29, var_35);
    // make_frame(normal[i]),                                                                 <L 1150>
    var_38 = wp::extract(var_30, var_35);
    var_39 = make_frame_0(var_38);
    // margin,                                                                                <L 1151>
    // gap,                                                                                   <L 1152>
    // condim,                                                                                <L 1153>
    // friction,                                                                              <L 1154>
    // solref,                                                                                <L 1155>
    // solreffriction,                                                                        <L 1156>
    // solimp,                                                                                <L 1157>
    // adhesion,                                                                              <L 1158>
    // geoms,                                                                                 <L 1159>
    // pairid,                                                                                <L 1160>
    // worldid,                                                                               <L 1161>
    // contact_dist_out,                                                                      <L 1162>
    // contact_pos_out,                                                                       <L 1163>
    // contact_frame_out,                                                                     <L 1164>
    // contact_includemargin_out,                                                             <L 1165>
    // contact_friction_out,                                                                  <L 1166>
    // contact_solref_out,                                                                    <L 1167>
    // contact_solreffriction_out,                                                            <L 1168>
    // contact_solimp_out,                                                                    <L 1169>
    // contact_dim_out,                                                                       <L 1170>
    // contact_geom_out,                                                                      <L 1171>
    // contact_efc_address_out,                                                               <L 1172>
    // contact_worldid_out,                                                                   <L 1173>
    // contact_type_out,                                                                      <L 1174>
    // contact_geomcollisionid_out,                                                           <L 1175>
    // contact_adhesion_out,                                                                  <L 1176>
    // nacon_out,                                                                             <L 1177>
    var_40 = write_contact_0(var_naconmax_in, var_35, var_36, var_37, var_39, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
    // write_contact(                                                                         <L 1145>
    // naconmax_in,                                                                           <L 1146>
    // i,                                                                                     <L 1147>
    // dist[i],                                                                               <L 1148>
    var_42 = wp::extract(var_28, var_41);
    // pos[i],                                                                                <L 1149>
    var_43 = wp::extract(var_29, var_41);
    // make_frame(normal[i]),                                                                 <L 1150>
    var_44 = wp::extract(var_30, var_41);
    var_45 = make_frame_0(var_44);
    // margin,                                                                                <L 1151>
    // gap,                                                                                   <L 1152>
    // condim,                                                                                <L 1153>
    // friction,                                                                              <L 1154>
    // solref,                                                                                <L 1155>
    // solreffriction,                                                                        <L 1156>
    // solimp,                                                                                <L 1157>
    // adhesion,                                                                              <L 1158>
    // geoms,                                                                                 <L 1159>
    // pairid,                                                                                <L 1160>
    // worldid,                                                                               <L 1161>
    // contact_dist_out,                                                                      <L 1162>
    // contact_pos_out,                                                                       <L 1163>
    // contact_frame_out,                                                                     <L 1164>
    // contact_includemargin_out,                                                             <L 1165>
    // contact_friction_out,                                                                  <L 1166>
    // contact_solref_out,                                                                    <L 1167>
    // contact_solreffriction_out,                                                            <L 1168>
    // contact_solimp_out,                                                                    <L 1169>
    // contact_dim_out,                                                                       <L 1170>
    // contact_geom_out,                                                                      <L 1171>
    // contact_efc_address_out,                                                               <L 1172>
    // contact_worldid_out,                                                                   <L 1173>
    // contact_type_out,                                                                      <L 1174>
    // contact_geomcollisionid_out,                                                           <L 1175>
    // contact_adhesion_out,                                                                  <L 1176>
    // nacon_out,                                                                             <L 1177>
    var_46 = write_contact_0(var_naconmax_in, var_41, var_42, var_43, var_45, var_margin, var_gap, var_condim, var_friction, var_solref, var_solreffriction, var_solimp, var_adhesion, var_geoms, var_pairid, var_worldid, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:465
static CUDA_CALLABLE void adj_plane_cylinder_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_cylinder_center,
    wp::vec_t<3, wp::float32> var_cylinder_axis,
    wp::float32 var_cylinder_radius,
    wp::float32 var_cylinder_half_height,
    wp::vec_t<4, wp::float32> & ret_0,
    wp::mat_t<4, 3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_cylinder_center,
    wp::vec_t<3, wp::float32> & adj_cylinder_axis,
    wp::float32 & adj_cylinder_radius,
    wp::float32 & adj_cylinder_half_height,
    wp::vec_t<4, wp::float32> & adj_ret_0,
    wp::mat_t<4, 3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:931
static CUDA_CALLABLE void adj_plane_cylinder_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_plane,
    Geom_4022bfa7 var_cylinder,
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
    Geom_4022bfa7 & adj_cylinder,
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:342
static CUDA_CALLABLE void adj_plane_box_0(
    wp::vec_t<3, wp::float32> var_plane_normal,
    wp::vec_t<3, wp::float32> var_plane_pos,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::vec_t<8, wp::float32> & ret_0,
    wp::mat_t<8, 3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_plane_normal,
    wp::vec_t<3, wp::float32> & adj_plane_pos,
    wp::vec_t<3, wp::float32> & adj_box_pos,
    wp::mat_t<3, 3, wp::float32> & adj_box_rot,
    wp::vec_t<3, wp::float32> & adj_box_size,
    wp::vec_t<8, wp::float32> & adj_ret_0,
    wp::mat_t<8, 3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:692
static CUDA_CALLABLE void adj_plane_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_plane,
    Geom_4022bfa7 var_box,
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
    Geom_4022bfa7 & adj_box,
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:393
static CUDA_CALLABLE void adj_sphere_cylinder_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_cylinder_pos,
    wp::vec_t<3, wp::float32> var_cylinder_axis,
    wp::float32 var_cylinder_radius,
    wp::float32 var_cylinder_half_height,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_cylinder_pos,
    wp::vec_t<3, wp::float32> & adj_cylinder_axis,
    wp::float32 & adj_cylinder_radius,
    wp::float32 & adj_cylinder_half_height,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:846
static CUDA_CALLABLE void adj_sphere_cylinder_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_sphere,
    Geom_4022bfa7 var_cylinder,
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
    Geom_4022bfa7 & adj_cylinder,
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:1049
static CUDA_CALLABLE void adj_sphere_box_0(
    wp::vec_t<3, wp::float32> var_sphere_pos,
    wp::float32 var_sphere_radius,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::float32 & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::vec_t<3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_sphere_pos,
    wp::float32 & adj_sphere_radius,
    wp::vec_t<3, wp::float32> & adj_box_pos,
    wp::mat_t<3, 3, wp::float32> & adj_box_rot,
    wp::vec_t<3, wp::float32> & adj_box_size,
    wp::float32 & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::vec_t<3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:1018
static CUDA_CALLABLE void adj_sphere_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_sphere,
    Geom_4022bfa7 var_box,
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
    Geom_4022bfa7 & adj_box,
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


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive_core.py:1104
static CUDA_CALLABLE void adj_capsule_box_0(
    wp::vec_t<3, wp::float32> var_capsule_pos,
    wp::vec_t<3, wp::float32> var_capsule_axis,
    wp::float32 var_capsule_radius,
    wp::float32 var_capsule_half_length,
    wp::vec_t<3, wp::float32> var_box_pos,
    wp::mat_t<3, 3, wp::float32> var_box_rot,
    wp::vec_t<3, wp::float32> var_box_size,
    wp::vec_t<2, wp::float32> & ret_0,
    wp::mat_t<2, 3, wp::float32> & ret_1,
    wp::mat_t<2, 3, wp::float32> & ret_2,
    wp::vec_t<3, wp::float32> & adj_capsule_pos,
    wp::vec_t<3, wp::float32> & adj_capsule_axis,
    wp::float32 & adj_capsule_radius,
    wp::float32 & adj_capsule_half_length,
    wp::vec_t<3, wp::float32> & adj_box_pos,
    wp::mat_t<3, 3, wp::float32> & adj_box_rot,
    wp::vec_t<3, wp::float32> & adj_box_size,
    wp::vec_t<2, wp::float32> & adj_ret_0,
    wp::mat_t<2, 3, wp::float32> & adj_ret_1,
    wp::mat_t<2, 3, wp::float32> & adj_ret_2)
{
	// reverse mode disabled (no backward-enabled kernel depends on this function)
}


// C:\Users\ASUS\Documents\human-imitation-learning\.venv\Lib\site-packages\mujoco_warp\_src\collision_primitive.py:1092
static CUDA_CALLABLE void adj_capsule_box_wrapper_0(
    wp::int32 var_naconmax_in,
    Geom_4022bfa7 var_cap,
    Geom_4022bfa7 var_box,
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
    Geom_4022bfa7 & adj_cap,
    Geom_4022bfa7 & adj_box,
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



extern "C" __global__ void _primitive_narrowphase__locals__primitive_narrowphase_1daf64ca_cuda_kernel_forward(
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
        const wp::int32 var_59 = 0;
        const wp::int32 var_60 = 5;
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
        const wp::int32 var_77 = 0;
        const wp::int32 var_78 = 6;
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
        const wp::int32 var_95 = 2;
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
        const wp::int32 var_112 = 5;
        const wp::int32 var_113 = 2;
        const wp::int32 var_114 = 5;
        const wp::int32 var_115 = 0;
        wp::int32 var_116;
        wp::int32* var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        const wp::int32 var_120 = 1;
        wp::int32 var_121;
        wp::int32* var_122;
        wp::int32 var_123;
        wp::int32 var_124;
        bool var_125;
        bool var_126;
        bool var_127;
        wp::vec_t<2, wp::int32>* var_128;
        wp::vec_t<2, wp::int32> var_129;
        const wp::int32 var_130 = 6;
        const wp::int32 var_131 = 2;
        const wp::int32 var_132 = 6;
        const wp::int32 var_133 = 0;
        wp::int32 var_134;
        wp::int32* var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        const wp::int32 var_138 = 1;
        wp::int32 var_139;
        wp::int32* var_140;
        wp::int32 var_141;
        wp::int32 var_142;
        bool var_143;
        bool var_144;
        bool var_145;
        wp::vec_t<2, wp::int32>* var_146;
        wp::vec_t<2, wp::int32> var_147;
        const wp::int32 var_148 = 7;
        const wp::int32 var_149 = 3;
        const wp::int32 var_150 = 3;
        const wp::int32 var_151 = 0;
        wp::int32 var_152;
        wp::int32* var_153;
        wp::int32 var_154;
        wp::int32 var_155;
        const wp::int32 var_156 = 1;
        wp::int32 var_157;
        wp::int32* var_158;
        wp::int32 var_159;
        wp::int32 var_160;
        bool var_161;
        bool var_162;
        bool var_163;
        wp::vec_t<2, wp::int32>* var_164;
        wp::vec_t<2, wp::int32> var_165;
        const wp::int32 var_166 = 8;
        const wp::int32 var_167 = 3;
        const wp::int32 var_168 = 6;
        const wp::int32 var_169 = 0;
        wp::int32 var_170;
        wp::int32* var_171;
        wp::int32 var_172;
        wp::int32 var_173;
        const wp::int32 var_174 = 1;
        wp::int32 var_175;
        wp::int32* var_176;
        wp::int32 var_177;
        wp::int32 var_178;
        bool var_179;
        bool var_180;
        bool var_181;
        wp::vec_t<2, wp::int32>* var_182;
        wp::vec_t<2, wp::int32> var_183;
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
            plane_cylinder_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_75, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
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
            plane_box_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_93, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
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
            sphere_capsule_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_111, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1407>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1408>
        // type1 = geom_type[geoms[0]]                                                            <L 1409>
        var_116 = wp::extract(var_6, var_115);
        var_117 = wp::address(var_geom_type, var_116);
        var_119 = wp::load(var_117);
        var_118 = wp::copy(var_119);
        // type2 = geom_type[geoms[1]]                                                            <L 1410>
        var_121 = wp::extract(var_6, var_120);
        var_122 = wp::address(var_geom_type, var_121);
        var_124 = wp::load(var_122);
        var_123 = wp::copy(var_124);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1411>
        var_126 = (var_113 == var_118);
        var_125 = var_126;
        if (var_125) {
            var_127 = (var_114 == var_123);
            var_125 = var_125 && var_127;
        }
        if (var_125) {
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
            var_128 = wp::address(var_collision_pairid_in, var_0);
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
            var_129 = wp::load(var_128);
            sphere_cylinder_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_129, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1407>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1408>
        // type1 = geom_type[geoms[0]]                                                            <L 1409>
        var_134 = wp::extract(var_6, var_133);
        var_135 = wp::address(var_geom_type, var_134);
        var_137 = wp::load(var_135);
        var_136 = wp::copy(var_137);
        // type2 = geom_type[geoms[1]]                                                            <L 1410>
        var_139 = wp::extract(var_6, var_138);
        var_140 = wp::address(var_geom_type, var_139);
        var_142 = wp::load(var_140);
        var_141 = wp::copy(var_142);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1411>
        var_144 = (var_131 == var_136);
        var_143 = var_144;
        if (var_143) {
            var_145 = (var_132 == var_141);
            var_143 = var_143 && var_145;
        }
        if (var_143) {
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
            var_146 = wp::address(var_collision_pairid_in, var_0);
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
            var_147 = wp::load(var_146);
            sphere_box_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_147, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1407>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1408>
        // type1 = geom_type[geoms[0]]                                                            <L 1409>
        var_152 = wp::extract(var_6, var_151);
        var_153 = wp::address(var_geom_type, var_152);
        var_155 = wp::load(var_153);
        var_154 = wp::copy(var_155);
        // type2 = geom_type[geoms[1]]                                                            <L 1410>
        var_157 = wp::extract(var_6, var_156);
        var_158 = wp::address(var_geom_type, var_157);
        var_160 = wp::load(var_158);
        var_159 = wp::copy(var_160);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1411>
        var_162 = (var_149 == var_154);
        var_161 = var_162;
        if (var_161) {
            var_163 = (var_150 == var_159);
            var_161 = var_161 && var_163;
        }
        if (var_161) {
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
            var_164 = wp::address(var_collision_pairid_in, var_0);
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
            var_165 = wp::load(var_164);
            capsule_capsule_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_165, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
        }
        // collision_type1 = wp.static(primitive_collisions_types[i][0])                          <L 1407>
        // collision_type2 = wp.static(primitive_collisions_types[i][1])                          <L 1408>
        // type1 = geom_type[geoms[0]]                                                            <L 1409>
        var_170 = wp::extract(var_6, var_169);
        var_171 = wp::address(var_geom_type, var_170);
        var_173 = wp::load(var_171);
        var_172 = wp::copy(var_173);
        // type2 = geom_type[geoms[1]]                                                            <L 1410>
        var_175 = wp::extract(var_6, var_174);
        var_176 = wp::address(var_geom_type, var_175);
        var_178 = wp::load(var_176);
        var_177 = wp::copy(var_178);
        // if collision_type1 == type1 and collision_type2 == type2:                              <L 1411>
        var_180 = (var_167 == var_172);
        var_179 = var_180;
        if (var_179) {
            var_181 = (var_168 == var_177);
            var_179 = var_179 && var_181;
        }
        if (var_179) {
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
            var_182 = wp::address(var_collision_pairid_in, var_0);
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
            var_183 = wp::load(var_182);
            capsule_box_wrapper_0(var_naconmax_in, var_20, var_21, var_9, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_6, var_183, var_contact_dist_out, var_contact_pos_out, var_contact_frame_out, var_contact_includemargin_out, var_contact_friction_out, var_contact_solref_out, var_contact_solreffriction_out, var_contact_solimp_out, var_contact_dim_out, var_contact_geom_out, var_contact_efc_address_out, var_contact_worldid_out, var_contact_type_out, var_contact_geomcollisionid_out, var_contact_adhesion_out, var_nacon_out);
        }
    }
}

