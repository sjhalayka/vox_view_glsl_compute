/*
    opengametools vox file reader/writer/merger - v0.997 - MIT license - Justin Paver, Oct 2019

    This is a single-header-file library that provides easy-to-use
    support for reading MagicaVoxel .vox files into structures that
    are easy to dereference and extract information from. It also
    supports writing back out to .vox file from those structures.

    Please see the MIT license information at the end of this file.

    Also, please consider sharing any improvements you make.

    For more information and more tools, visit:
      https://github.com/jpaver/opengametools

    HOW TO COMPILE THIS LIBRARY

    1.  To compile this library, do this in *one* C or C++ file:
        #define OGT_VOX_IMPLEMENTATION
        #include "ogt_vox.h"

    2. From any other module, it is sufficient to just #include this as usual:
        #include "ogt_vox.h"

    HOW TO READ A VOX SCENE (See demo_vox.cpp)

    1. load a .vox file off disk into a memory buffer.

    2. construct a scene from the memory buffer:
       ogt_vox_scene* scene = ogt_vox_read_scene(buffer, buffer_size);

    3. use the scene members to extract the information you need. eg.
       printf("# of layers: %u\n", scene->num_layers );

    4. destroy the scene:
       ogt_vox_destroy_scene(scene);

    HOW TO MERGE MULTIPLE VOX SCENES (See merge_vox.cpp)

    1. construct multiple scenes from files you want to merge.

        // read buffer1/buffer_size1 from "test1.vox"
        // read buffer2/buffer_size2 from "test2.vox"
        // read buffer3/buffer_size3 from "test3.vox"
        ogt_vox_scene* scene1 = ogt_vox_read_scene(buffer1, buffer_size1);
        ogt_vox_scene* scene2 = ogt_vox_read_scene(buffer2, buffer_size2);
        ogt_vox_scene* scene3 = ogt_vox_read_scene(buffer3, buffer_size3);

    2. construct a merged scene

        const ogt_vox_scene* scenes[] = {scene1, scene2, scene3};
        ogt_vox_scene* merged_scene = ogt_vox_merge_scenes(scenes, 3, NULL, 0);

    3. save out the merged scene

        uint8_t* out_buffer = ogt_vox_write_scene(merged_scene, &out_buffer_size);
        // save out_buffer to disk as a .vox file (it has length out_buffer_size)

    4. destroy the merged scene:

        ogt_vox_destroy_scene(merged_scene);

    EXPLANATION OF SCENE ELEMENTS:

    A ogt_vox_scene comprises primarily a set of instances, models, layers and a palette.

    A ogt_vox_palette contains a set of 256 colors that is used for the scene.
    Each color is represented by a 4-tuple called an ogt_vox_rgba which contains red,
    green, blue and alpha values for the color.

    A ogt_vox_model is a 3-dimensional grid of voxels, where each of those voxels
    is represented by an 8-bit color index. Voxels are arranged in order of increasing
    X then increasing Y then increasing Z.

    Given the x,y,z values for a voxel within the model dimensions, the voxels index
    in the grid can be obtained as follows:

        voxel_index = x + (y * model->size_x) + (z * model->size_x * model->size_y)

    The index is only valid if the coordinate x,y,z satisfy the following conditions:
            0 <= x < model->size_x -AND-
            0 <= y < model->size_y -AND-
            0 <= z < model->size_z

    A voxels color index can be obtained as follows:

        uint8_t color_index = model->voxel_data[voxel_index];

    If color_index == 0, the voxel is not solid and can be skipped,
    If color_index != 0, the voxel is solid and can be used to lookup the color in the palette:

        ogt_vox_rgba color = scene->palette.color[ color_index]

    A ogt_vox_instance is an individual placement of a voxel model within the scene. Each
    instance has a transform that determines its position and orientation within the scene,
    but it also has an index that specifies which model the instance uses for its shape. It
    is expected that there is a many-to-one mapping of instances to models.

    An ogt_vox_layer is used to conceptually group instances. Each instance indexes the
    layer that it belongs to, but the layer itself has its own name and hidden/shown state.

    EXPLANATION OF MERGED SCENES:

    A merged scene contains all the models and all the scene instances from
    each of the scenes that were passed into it.

    The merged scene will have a combined palette of all the source scene
    palettes by trying to match existing colors exactly, and falling back
    to an RGB-distance matched color when all 256 colors in the merged
    scene palette has been allocated.

    You can explicitly control up to 255 merge palette colors by providing
    those colors to ogt_vox_merge_scenes in the required_colors parameters eg.

        const ogt_vox_palette palette;  // load this via .vox or procedurally or whatever
        const ogt_vox_scene* scenes[] = {scene1, scene2, scene3};
        // palette.color[0] is always the empty color which is why we pass 255 colors starting from index 1 only:
        ogt_vox_scene* merged_scene = ogt_vox_merge_scenes(scenes, 3, &palette.color[1], 255);

    EXPLANATION OF MODEL PIVOTS

    If a voxel model grid has dimension size.xyz in terms of number of voxels, the centre pivot
    for that model is located at floor( size.xyz / 2).

    eg. for a 3x4x1 voxel model, the pivot would be at (1,2,0), or the X in the below ascii art.

           4 +-----+-----+-----+
             |  .  |  .  |  .  |
           3 +-----+-----+-----+
             |  .  |  .  |  .  |
           2 +-----X-----+-----+
             |  .  |  .  |  .  |
           1 +-----+-----+-----+
             |  .  |  .  |  .  |
           0 +-----+-----+-----+
             0     1     2     3

     An example model in this grid form factor might look like this:

           4 +-----+-----+-----+
             |  .  |  .  |  .  |
           3 +-----+-----+-----+
                   |  .  |
           2       X-----+
                   |  .  |
           1       +-----+
                   |  .  |
           0       +-----+
             0     1     2     3

     If you were to generate a mesh from this, clearly each vertex and each face would be on an integer
     coordinate eg. 1, 2, 3 etc. while the centre of each grid location (ie. the . in the above diagram)
     will be on a coordinate that is halfway between integer coordinates. eg. 1.5, 2.5, 3.5 etc.

     To ensure your mesh is properly centered such that instance transforms are correctly applied, you
     want the pivot to be treated as if it were (0,0,0) in model space. To achieve this, simply
     subtract the pivot from any geometry that is generated (eg. vertices in a mesh).

     For the 3x4x1 voxel model above, doing this would look like this:

           2 +-----+-----+-----+
             |  .  |  .  |  .  |
           1 +-----+-----+-----+
                   |  .  |
           0       X-----+
                   |  .  |
          -1       +-----+
                   |  .  |
          -2       +-----+
            -1     0     1     2

    To replace asserts within this library with your own implementation, simply #define ogt_assert before defining your implementation
    eg.
        #include "my_assert.h"
        #define ogt_assert(condition, message_str)    my_assert(condition, message_str)

        #define OGT_VOX_IMPLEMENTATION
        #include "path/to/ogt_vox.h"

*/
#ifndef OGT_VOX_H__
#define OGT_VOX_H__

#if _MSC_VER == 1400
    // VS2005 doesn't have inttypes or stdint so we just define what we need here.
    typedef unsigned char uint8_t;
    typedef signed int    int32_t;
    typedef unsigned int  uint32_t;
    #ifndef UINT32_MAX
        #define UINT32_MAX	((uint32_t)0xFFFFFFFF)
    #endif
    #ifndef INT32_MAX
        #define INT32_MAX	((int32_t)0x7FFFFFFF)
    #endif
    #ifndef UINT8_MAX
        #define UINT8_MAX	((uint8_t)0xFF)
    #endif
#elif defined(_MSC_VER)
    // general VS*
    #include <inttypes.h>
#elif __APPLE__
    // general Apple compiler
#elif defined(__GNUC__)
    // any GCC*
    #include <inttypes.h>
    #include <stdlib.h> // for size_t
#else
    #error some fixup needed for this platform?
#endif


static void* _vox_realloc(void* old_ptr, size_t old_size, size_t new_size);
static void _vox_free(void* old_ptr);

    // denotes an invalid group index. Usually this is only applicable to the scene's root group's parent.
    static const uint32_t k_invalid_group_index = UINT32_MAX;

    // color
    typedef struct ogt_vox_rgba
    {
        uint8_t r,g,b,a;            // red, green, blue and alpha components of a color.
    } ogt_vox_rgba;

    // column-major 4x4 matrix
    typedef struct ogt_vox_transform
    {
        float m00, m01, m02, m03;   // column 0 of 4x4 matrix, 1st three elements = x axis vector, last element always 0.0
        float m10, m11, m12, m13;   // column 1 of 4x4 matrix, 1st three elements = y axis vector, last element always 0.0
        float m20, m21, m22, m23;   // column 2 of 4x4 matrix, 1st three elements = z axis vector, last element always 0.0
        float m30, m31, m32, m33;   // column 3 of 4x4 matrix. 1st three elements = translation vector, last element always 1.0
    } ogt_vox_transform;

    // a palette of colors
    typedef struct ogt_vox_palette
    {
        ogt_vox_rgba color[256];      // palette of colors. use the voxel indices to lookup color from the palette.
    } ogt_vox_palette;

    // Extended Material Chunk MATL types
    enum ogt_matl_type
    {
        ogt_matl_type_diffuse = 0, // diffuse is default
        ogt_matl_type_metal   = 1,
        ogt_matl_type_glass   = 2,
        ogt_matl_type_emit    = 3,
        ogt_matl_type_blend   = 4,
        ogt_matl_type_media   = 5,
    };

    enum ogt_cam_mode
    {
        ogt_cam_mode_perspective  = 0,
        ogt_cam_mode_free         = 1,
        ogt_cam_mode_pano         = 2,
        ogt_cam_mode_orthographic = 3,
        ogt_cam_mode_isometric    = 4,
        ogt_cam_mode_unknown      = 5
    };

    // Content Flags for ogt_vox_matl values for a given material
    static const uint32_t k_ogt_vox_matl_have_metal  = 1 << 0;
    static const uint32_t k_ogt_vox_matl_have_rough  = 1 << 1;
    static const uint32_t k_ogt_vox_matl_have_spec   = 1 << 2;
    static const uint32_t k_ogt_vox_matl_have_ior    = 1 << 3;
    static const uint32_t k_ogt_vox_matl_have_att    = 1 << 4;
    static const uint32_t k_ogt_vox_matl_have_flux   = 1 << 5;
    static const uint32_t k_ogt_vox_matl_have_emit   = 1 << 6;
    static const uint32_t k_ogt_vox_matl_have_ldr    = 1 << 7;
    static const uint32_t k_ogt_vox_matl_have_trans  = 1 << 8;
    static const uint32_t k_ogt_vox_matl_have_alpha  = 1 << 9;
    static const uint32_t k_ogt_vox_matl_have_d      = 1 << 10;
    static const uint32_t k_ogt_vox_matl_have_sp     = 1 << 11;
    static const uint32_t k_ogt_vox_matl_have_g      = 1 << 12;
    static const uint32_t k_ogt_vox_matl_have_media  = 1 << 13;

    // Extended Material Chunk MATL information
    typedef struct ogt_vox_matl
    {
        uint32_t      content_flags; // set of k_ogt_vox_matl_* OR together to denote contents available
        ogt_matl_type type;
        float         metal;
        float         rough;
        float         spec;
        float         ior;
        float         att;
        float         flux;
        float         emit;
        float         ldr;
        float         trans;
        float         alpha;
        float         d;
        float         sp;
        float         g;
        float         media;
    } ogt_vox_matl;

    // Extended Material Chunk MATL array of materials
    typedef struct ogt_vox_matl_array
    {
        ogt_vox_matl matl[256];      // extended material information from Material Chunk MATL
    } ogt_vox_matl_array;

    typedef struct ogt_vox_cam
    {
        uint32_t     camera_id;
        ogt_cam_mode mode;
        float        focus[3];    // the target position
        float        angle[3];    // rotation in degree
        int          radius;
        float        frustum;
        int          fov;         // angle in degree
    } ogt_vox_cam;

    // a 3-dimensional model of voxels
    typedef struct ogt_vox_model
    {
        uint32_t       size_x;        // number of voxels in the local x dimension
        uint32_t       size_y;        // number of voxels in the local y dimension
        uint32_t       size_z;        // number of voxels in the local z dimension
        uint32_t       voxel_hash;    // hash of the content of the grid.
        const uint8_t* voxel_data;    // grid of voxel data comprising color indices in x -> y -> z order. a color index of 0 means empty, all other indices mean solid and can be used to index the scene's palette to obtain the color for the voxel.
    } ogt_vox_model;

    // a keyframe for animation of a transform
    typedef struct ogt_vox_keyframe_transform {
        uint32_t          frame_index;
        ogt_vox_transform transform;
    } ogt_vox_keyframe_transform;

    // a keyframe for animation of a model
    typedef struct ogt_vox_keyframe_model {
        uint32_t frame_index;
        uint32_t model_index;
    } ogt_vox_keyframe_model;

    // an animated transform
    typedef struct ogt_vox_anim_transform {
        const ogt_vox_keyframe_transform* keyframes;
        uint32_t                          num_keyframes;
        bool                              loop;
    } ogt_vox_anim_transform;

    // an animated model
    typedef struct ogt_vox_anim_model {
        const ogt_vox_keyframe_model* keyframes;
        uint32_t                      num_keyframes;
        bool                          loop;
    } ogt_vox_anim_model;

    // an instance of a model within the scene
    typedef struct ogt_vox_instance
    {
        const char*            name;                   // name of the instance if there is one, will be NULL otherwise.
        ogt_vox_transform      transform;              // orientation and position of this instance on first frame of the scene. This is relative to its group local transform if group_index is not 0
        uint32_t               model_index;            // index of the model used by this instance on the first frame of the scene. used to lookup the model in the scene's models[] array.
        uint32_t               layer_index;            // index of the layer used by this instance. used to lookup the layer in the scene's layers[] array.
        uint32_t               group_index;            // this will be the index of the group in the scene's groups[] array. If group is zero it will be the scene root group and the instance transform will be a world-space transform, otherwise the transform is relative to the group.
        bool                   hidden;                 // whether this instance is individually hidden or not. Note: the instance can also be hidden when its layer is hidden, or if it belongs to a group that is hidden.
        ogt_vox_anim_transform transform_anim;         // animation for the transform
        ogt_vox_anim_model     model_anim;             // animation for the model_index
    } ogt_vox_instance;

    // describes a layer within the scene
    typedef struct ogt_vox_layer
    {
        const char*  name;               // name of this layer if there is one, will be NULL otherwise.
        ogt_vox_rgba color;              // color of the layer.
        bool         hidden;             // whether this layer is hidden or not.
    } ogt_vox_layer;

    // describes a group within the scene
    typedef struct ogt_vox_group
    {
        const char*            name;                    // name of the group if there is one, will be NULL otherwise
        ogt_vox_transform      transform;               // transform of this group relative to its parent group (if any), otherwise this will be relative to world-space.
        uint32_t               parent_group_index;      // if this group is parented to another group, this will be the index of its parent in the scene's groups[] array, otherwise this group will be the scene root group and this value will be k_invalid_group_index
        uint32_t               layer_index;             // which layer this group belongs to. used to lookup the layer in the scene's layers[] array.
        bool                   hidden;                  // whether this group is hidden or not.
        ogt_vox_anim_transform transform_anim;          // animated transform data
    } ogt_vox_group;

    // the scene parsed from a .vox file.
    typedef struct ogt_vox_scene
    {
        uint32_t                num_models;     // number of models within the scene.
        uint32_t                num_instances;  // number of instances in the scene (on anim frame 0)
        uint32_t                num_layers;     // number of layers in the scene
        uint32_t                num_groups;     // number of groups in the scene
        const ogt_vox_model**   models;         // array of models. size is num_models
        const ogt_vox_instance* instances;      // array of instances. size is num_instances
        const ogt_vox_layer*    layers;         // array of layers. size is num_layers
        const ogt_vox_group*    groups;         // array of groups. size is num_groups
        ogt_vox_palette         palette;        // the palette for this scene
        ogt_vox_matl_array      materials;      // the extended materials for this scene
        uint32_t                num_cameras;    // number of cameras for this scene
        const ogt_vox_cam*      cameras;        // the cameras for this scene
    } ogt_vox_scene;

    // allocate memory function interface. pass in size, and get a pointer to memory with at least that size available.
    typedef void* (*ogt_vox_alloc_func)(size_t size);

    // free memory function interface. pass in a pointer previously allocated and it will be released back to the system managing memory.
    typedef void  (*ogt_vox_free_func)(void* ptr);

    // override the default scene memory allocator if you need to control memory precisely.
    void  ogt_vox_set_memory_allocator(ogt_vox_alloc_func alloc_func, ogt_vox_free_func free_func);
    void* ogt_vox_malloc(size_t size);
    void  ogt_vox_free(void* mem);

    // flags for ogt_vox_read_scene_with_flags
    static const uint32_t k_read_scene_flags_groups                      = 1 << 0; // if not specified, all instance transforms will be flattened into world space. If specified, will read group information and keep all transforms as local transform relative to the group they are in.
    static const uint32_t k_read_scene_flags_keyframes                   = 1 << 1; // if specified, all instances and groups will contain keyframe data.
    static const uint32_t k_read_scene_flags_keep_empty_models_instances = 1 << 2; // if specified, all empty models and instances referencing those will be kept rather than culled.

    // creates a scene from a vox file within a memory buffer of a given size.
    // you can destroy the input buffer once you have the scene as this function will allocate separate memory for the scene objecvt.
    const ogt_vox_scene* ogt_vox_read_scene(const uint8_t* buffer, uint32_t buffer_size);

    // just like ogt_vox_read_scene, but you can additionally pass a union of k_read_scene_flags
    const ogt_vox_scene* ogt_vox_read_scene_with_flags(const uint8_t* buffer, uint32_t buffer_size, uint32_t read_flags);

    // destroys a scene object to release its memory.
    void ogt_vox_destroy_scene(const ogt_vox_scene* scene);

    // writes the scene to a new buffer and returns the buffer size. free the buffer with ogt_vox_free
    uint8_t* ogt_vox_write_scene(const ogt_vox_scene* scene, uint32_t* buffer_size);

    // merges the specified scenes together to create a bigger scene. Merged scene can be destroyed using ogt_vox_destroy_scene
    // If you require specific colors in the merged scene palette, provide up to and including 255 of them via required_colors/required_color_count.
    ogt_vox_scene* ogt_vox_merge_scenes(const ogt_vox_scene** scenes, uint32_t scene_count, const ogt_vox_rgba* required_colors, const uint32_t required_color_count);

    // samples which model_index the given animation produces at the given frame
    uint32_t          ogt_vox_sample_anim_model(const ogt_vox_anim_model* anim, uint32_t frame_index);

    // // sample which transform the given animation produces at the given frame
    ogt_vox_transform ogt_vox_sample_anim_transform(const ogt_vox_anim_transform* anim, uint32_t frame_index);

    // sample the model for a given instance at the given frame
    uint32_t          ogt_vox_sample_instance_model(const ogt_vox_instance* instance, uint32_t frame_index);

    // sample the flattened transform for a given instance at the given frame (takes into account group hierarchy and group animations)
    ogt_vox_transform ogt_vox_sample_instance_transform(const ogt_vox_instance* instance, uint32_t frame_index, const ogt_vox_scene* scene);

#endif // OGT_VOX_H__

//-----------------------------------------------------------------------------------------------------------------
//
// If you're only interested in using this library, everything you need is above this point.
// If you're interested in how this library works, everything you need is below this point.
//
//-----------------------------------------------------------------------------------------------------------------
#ifdef OGT_VOX_IMPLEMENTATION
    // callers can override asserts in ogt_vox by defining their own macro before the implementation
#ifndef ogt_assert
    #include <assert.h>
    #define ogt_assert(x, msg_str)      do { assert((x) && (msg_str)); } while(0)
#endif
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>

    // MAKE_VOX_CHUNK_ID: used to construct a literal to describe a chunk in a .vox file.
    #define MAKE_VOX_CHUNK_ID(c0,c1,c2,c3)     ( (c0<<0) | (c1<<8) | (c2<<16) | (c3<<24) )

    static const uint32_t CHUNK_ID_VOX_ = MAKE_VOX_CHUNK_ID('V','O','X',' ');
    static const uint32_t CHUNK_ID_MAIN = MAKE_VOX_CHUNK_ID('M','A','I','N');
    static const uint32_t CHUNK_ID_SIZE = MAKE_VOX_CHUNK_ID('S','I','Z','E');
    static const uint32_t CHUNK_ID_XYZI = MAKE_VOX_CHUNK_ID('X','Y','Z','I');
    static const uint32_t CHUNK_ID_RGBA = MAKE_VOX_CHUNK_ID('R','G','B','A');
    static const uint32_t CHUNK_ID_nTRN = MAKE_VOX_CHUNK_ID('n','T','R','N');
    static const uint32_t CHUNK_ID_nGRP = MAKE_VOX_CHUNK_ID('n','G','R','P');
    static const uint32_t CHUNK_ID_nSHP = MAKE_VOX_CHUNK_ID('n','S','H','P');
    static const uint32_t CHUNK_ID_IMAP = MAKE_VOX_CHUNK_ID('I','M','A','P');
    static const uint32_t CHUNK_ID_LAYR = MAKE_VOX_CHUNK_ID('L','A','Y','R');
    static const uint32_t CHUNK_ID_MATL = MAKE_VOX_CHUNK_ID('M','A','T','L');
    static const uint32_t CHUNK_ID_MATT = MAKE_VOX_CHUNK_ID('M','A','T','T');
    static const uint32_t CHUNK_ID_rOBJ = MAKE_VOX_CHUNK_ID('r','O','B','J');
    static const uint32_t CHUNK_ID_rCAM = MAKE_VOX_CHUNK_ID('r','C','A','M');

    // Some older .vox files will not store a palette, in which case the following palette will be used!
    static const uint8_t k_default_vox_palette[256 * 4] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xcc, 0xff, 0xff, 0xff, 0x99, 0xff, 0xff, 0xff, 0x66, 0xff, 0xff, 0xff, 0x33, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xcc, 0xff, 0xff, 0xff, 0xcc, 0xcc, 0xff,
        0xff, 0xcc, 0x99, 0xff, 0xff, 0xcc, 0x66, 0xff, 0xff, 0xcc, 0x33, 0xff, 0xff, 0xcc, 0x00, 0xff, 0xff, 0x99, 0xff, 0xff, 0xff, 0x99, 0xcc, 0xff, 0xff, 0x99, 0x99, 0xff, 0xff, 0x99, 0x66, 0xff,
        0xff, 0x99, 0x33, 0xff, 0xff, 0x99, 0x00, 0xff, 0xff, 0x66, 0xff, 0xff, 0xff, 0x66, 0xcc, 0xff, 0xff, 0x66, 0x99, 0xff, 0xff, 0x66, 0x66, 0xff, 0xff, 0x66, 0x33, 0xff, 0xff, 0x66, 0x00, 0xff,
        0xff, 0x33, 0xff, 0xff, 0xff, 0x33, 0xcc, 0xff, 0xff, 0x33, 0x99, 0xff, 0xff, 0x33, 0x66, 0xff, 0xff, 0x33, 0x33, 0xff, 0xff, 0x33, 0x00, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0x00, 0xcc, 0xff,
        0xff, 0x00, 0x99, 0xff, 0xff, 0x00, 0x66, 0xff, 0xff, 0x00, 0x33, 0xff, 0xff, 0x00, 0x00, 0xff, 0xcc, 0xff, 0xff, 0xff, 0xcc, 0xff, 0xcc, 0xff, 0xcc, 0xff, 0x99, 0xff, 0xcc, 0xff, 0x66, 0xff,
        0xcc, 0xff, 0x33, 0xff, 0xcc, 0xff, 0x00, 0xff, 0xcc, 0xcc, 0xff, 0xff, 0xcc, 0xcc, 0xcc, 0xff, 0xcc, 0xcc, 0x99, 0xff, 0xcc, 0xcc, 0x66, 0xff, 0xcc, 0xcc, 0x33, 0xff, 0xcc, 0xcc, 0x00, 0xff,
        0xcc, 0x99, 0xff, 0xff, 0xcc, 0x99, 0xcc, 0xff, 0xcc, 0x99, 0x99, 0xff, 0xcc, 0x99, 0x66, 0xff, 0xcc, 0x99, 0x33, 0xff, 0xcc, 0x99, 0x00, 0xff, 0xcc, 0x66, 0xff, 0xff, 0xcc, 0x66, 0xcc, 0xff,
        0xcc, 0x66, 0x99, 0xff, 0xcc, 0x66, 0x66, 0xff, 0xcc, 0x66, 0x33, 0xff, 0xcc, 0x66, 0x00, 0xff, 0xcc, 0x33, 0xff, 0xff, 0xcc, 0x33, 0xcc, 0xff, 0xcc, 0x33, 0x99, 0xff, 0xcc, 0x33, 0x66, 0xff,
        0xcc, 0x33, 0x33, 0xff, 0xcc, 0x33, 0x00, 0xff, 0xcc, 0x00, 0xff, 0xff, 0xcc, 0x00, 0xcc, 0xff, 0xcc, 0x00, 0x99, 0xff, 0xcc, 0x00, 0x66, 0xff, 0xcc, 0x00, 0x33, 0xff, 0xcc, 0x00, 0x00, 0xff,
        0x99, 0xff, 0xff, 0xff, 0x99, 0xff, 0xcc, 0xff, 0x99, 0xff, 0x99, 0xff, 0x99, 0xff, 0x66, 0xff, 0x99, 0xff, 0x33, 0xff, 0x99, 0xff, 0x00, 0xff, 0x99, 0xcc, 0xff, 0xff, 0x99, 0xcc, 0xcc, 0xff,
        0x99, 0xcc, 0x99, 0xff, 0x99, 0xcc, 0x66, 0xff, 0x99, 0xcc, 0x33, 0xff, 0x99, 0xcc, 0x00, 0xff, 0x99, 0x99, 0xff, 0xff, 0x99, 0x99, 0xcc, 0xff, 0x99, 0x99, 0x99, 0xff, 0x99, 0x99, 0x66, 0xff,
        0x99, 0x99, 0x33, 0xff, 0x99, 0x99, 0x00, 0xff, 0x99, 0x66, 0xff, 0xff, 0x99, 0x66, 0xcc, 0xff, 0x99, 0x66, 0x99, 0xff, 0x99, 0x66, 0x66, 0xff, 0x99, 0x66, 0x33, 0xff, 0x99, 0x66, 0x00, 0xff,
        0x99, 0x33, 0xff, 0xff, 0x99, 0x33, 0xcc, 0xff, 0x99, 0x33, 0x99, 0xff, 0x99, 0x33, 0x66, 0xff, 0x99, 0x33, 0x33, 0xff, 0x99, 0x33, 0x00, 0xff, 0x99, 0x00, 0xff, 0xff, 0x99, 0x00, 0xcc, 0xff,
        0x99, 0x00, 0x99, 0xff, 0x99, 0x00, 0x66, 0xff, 0x99, 0x00, 0x33, 0xff, 0x99, 0x00, 0x00, 0xff, 0x66, 0xff, 0xff, 0xff, 0x66, 0xff, 0xcc, 0xff, 0x66, 0xff, 0x99, 0xff, 0x66, 0xff, 0x66, 0xff,
        0x66, 0xff, 0x33, 0xff, 0x66, 0xff, 0x00, 0xff, 0x66, 0xcc, 0xff, 0xff, 0x66, 0xcc, 0xcc, 0xff, 0x66, 0xcc, 0x99, 0xff, 0x66, 0xcc, 0x66, 0xff, 0x66, 0xcc, 0x33, 0xff, 0x66, 0xcc, 0x00, 0xff,
        0x66, 0x99, 0xff, 0xff, 0x66, 0x99, 0xcc, 0xff, 0x66, 0x99, 0x99, 0xff, 0x66, 0x99, 0x66, 0xff, 0x66, 0x99, 0x33, 0xff, 0x66, 0x99, 0x00, 0xff, 0x66, 0x66, 0xff, 0xff, 0x66, 0x66, 0xcc, 0xff,
        0x66, 0x66, 0x99, 0xff, 0x66, 0x66, 0x66, 0xff, 0x66, 0x66, 0x33, 0xff, 0x66, 0x66, 0x00, 0xff, 0x66, 0x33, 0xff, 0xff, 0x66, 0x33, 0xcc, 0xff, 0x66, 0x33, 0x99, 0xff, 0x66, 0x33, 0x66, 0xff,
        0x66, 0x33, 0x33, 0xff, 0x66, 0x33, 0x00, 0xff, 0x66, 0x00, 0xff, 0xff, 0x66, 0x00, 0xcc, 0xff, 0x66, 0x00, 0x99, 0xff, 0x66, 0x00, 0x66, 0xff, 0x66, 0x00, 0x33, 0xff, 0x66, 0x00, 0x00, 0xff,
        0x33, 0xff, 0xff, 0xff, 0x33, 0xff, 0xcc, 0xff, 0x33, 0xff, 0x99, 0xff, 0x33, 0xff, 0x66, 0xff, 0x33, 0xff, 0x33, 0xff, 0x33, 0xff, 0x00, 0xff, 0x33, 0xcc, 0xff, 0xff, 0x33, 0xcc, 0xcc, 0xff,
        0x33, 0xcc, 0x99, 0xff, 0x33, 0xcc, 0x66, 0xff, 0x33, 0xcc, 0x33, 0xff, 0x33, 0xcc, 0x00, 0xff, 0x33, 0x99, 0xff, 0xff, 0x33, 0x99, 0xcc, 0xff, 0x33, 0x99, 0x99, 0xff, 0x33, 0x99, 0x66, 0xff,
        0x33, 0x99, 0x33, 0xff, 0x33, 0x99, 0x00, 0xff, 0x33, 0x66, 0xff, 0xff, 0x33, 0x66, 0xcc, 0xff, 0x33, 0x66, 0x99, 0xff, 0x33, 0x66, 0x66, 0xff, 0x33, 0x66, 0x33, 0xff, 0x33, 0x66, 0x00, 0xff,
        0x33, 0x33, 0xff, 0xff, 0x33, 0x33, 0xcc, 0xff, 0x33, 0x33, 0x99, 0xff, 0x33, 0x33, 0x66, 0xff, 0x33, 0x33, 0x33, 0xff, 0x33, 0x33, 0x00, 0xff, 0x33, 0x00, 0xff, 0xff, 0x33, 0x00, 0xcc, 0xff,
        0x33, 0x00, 0x99, 0xff, 0x33, 0x00, 0x66, 0xff, 0x33, 0x00, 0x33, 0xff, 0x33, 0x00, 0x00, 0xff, 0x00, 0xff, 0xff, 0xff, 0x00, 0xff, 0xcc, 0xff, 0x00, 0xff, 0x99, 0xff, 0x00, 0xff, 0x66, 0xff,
        0x00, 0xff, 0x33, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xcc, 0xff, 0xff, 0x00, 0xcc, 0xcc, 0xff, 0x00, 0xcc, 0x99, 0xff, 0x00, 0xcc, 0x66, 0xff, 0x00, 0xcc, 0x33, 0xff, 0x00, 0xcc, 0x00, 0xff,
        0x00, 0x99, 0xff, 0xff, 0x00, 0x99, 0xcc, 0xff, 0x00, 0x99, 0x99, 0xff, 0x00, 0x99, 0x66, 0xff, 0x00, 0x99, 0x33, 0xff, 0x00, 0x99, 0x00, 0xff, 0x00, 0x66, 0xff, 0xff, 0x00, 0x66, 0xcc, 0xff,
        0x00, 0x66, 0x99, 0xff, 0x00, 0x66, 0x66, 0xff, 0x00, 0x66, 0x33, 0xff, 0x00, 0x66, 0x00, 0xff, 0x00, 0x33, 0xff, 0xff, 0x00, 0x33, 0xcc, 0xff, 0x00, 0x33, 0x99, 0xff, 0x00, 0x33, 0x66, 0xff,
        0x00, 0x33, 0x33, 0xff, 0x00, 0x33, 0x00, 0xff, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0xcc, 0xff, 0x00, 0x00, 0x99, 0xff, 0x00, 0x00, 0x66, 0xff, 0x00, 0x00, 0x33, 0xff, 0xee, 0x00, 0x00, 0xff,
        0xdd, 0x00, 0x00, 0xff, 0xbb, 0x00, 0x00, 0xff, 0xaa, 0x00, 0x00, 0xff, 0x88, 0x00, 0x00, 0xff, 0x77, 0x00, 0x00, 0xff, 0x55, 0x00, 0x00, 0xff, 0x44, 0x00, 0x00, 0xff, 0x22, 0x00, 0x00, 0xff,
        0x11, 0x00, 0x00, 0xff, 0x00, 0xee, 0x00, 0xff, 0x00, 0xdd, 0x00, 0xff, 0x00, 0xbb, 0x00, 0xff, 0x00, 0xaa, 0x00, 0xff, 0x00, 0x88, 0x00, 0xff, 0x00, 0x77, 0x00, 0xff, 0x00, 0x55, 0x00, 0xff,
        0x00, 0x44, 0x00, 0xff, 0x00, 0x22, 0x00, 0xff, 0x00, 0x11, 0x00, 0xff, 0x00, 0x00, 0xee, 0xff, 0x00, 0x00, 0xdd, 0xff, 0x00, 0x00, 0xbb, 0xff, 0x00, 0x00, 0xaa, 0xff, 0x00, 0x00, 0x88, 0xff,
        0x00, 0x00, 0x77, 0xff, 0x00, 0x00, 0x55, 0xff, 0x00, 0x00, 0x44, 0xff, 0x00, 0x00, 0x22, 0xff, 0x00, 0x00, 0x11, 0xff, 0xee, 0xee, 0xee, 0xff, 0xdd, 0xdd, 0xdd, 0xff, 0xbb, 0xbb, 0xbb, 0xff,
        0xaa, 0xaa, 0xaa, 0xff, 0x88, 0x88, 0x88, 0xff, 0x77, 0x77, 0x77, 0xff, 0x55, 0x55, 0x55, 0xff, 0x44, 0x44, 0x44, 0xff, 0x22, 0x22, 0x22, 0xff, 0x11, 0x11, 0x11, 0xff, 0x00, 0x00, 0x00, 0xff,
    };



    // string utilities
    #ifdef _MSC_VER
        #define _vox_str_scanf(str,...)      sscanf_s(str,__VA_ARGS__)
        #define _vox_strcpy_static(dst,src)  strcpy_s(dst,src)
        #define _vox_strcasecmp(a,b)         _stricmp(a,b)
        #define _vox_strcmp(a,b)             strcmp(a,b)
        #define _vox_strlen(a)               strlen(a)
        #define _vox_sprintf(str,str_max,fmt,...)    sprintf_s(str, str_max, fmt, __VA_ARGS__)
    #else
        #define _vox_str_scanf(str,...)      sscanf(str,__VA_ARGS__)
        #define _vox_strcpy_static(dst,src)  strcpy(dst,src)
        #define _vox_strcasecmp(a,b)         strcasecmp(a,b)
        #define _vox_strcmp(a,b)             strcmp(a,b)
        #define _vox_strlen(a)               strlen(a)
        #define _vox_sprintf(str,str_max,fmt,...)    snprintf(str, str_max, fmt, __VA_ARGS__)
    #endif

    // 3d vector utilities
    struct vector_3 {
        float x, y, z;
    };
    static inline vector_3 vector_3_make(float x, float y, float z) { vector_3 v; v.x = x; v.y = y; v.z = z; return v; }
    static inline vector_3 vector_3_negate(const vector_3& v) { vector_3 r; r.x = -v.x;  r.y = -v.y; r.z = -v.z; return r; }

    // API for emulating file transactions on an in-memory buffer of data.
    struct _vox_file {
        const  uint8_t* buffer;       // source buffer data
        const uint32_t  buffer_size;  // size of the data in the buffer
        uint32_t        offset;       // current offset in the buffer data.
    };


    static ogt_vox_transform _vox_transform_multiply(const ogt_vox_transform& a, const ogt_vox_transform& b) {
        ogt_vox_transform r;
        r.m00 = (a.m00 * b.m00) + (a.m01 * b.m10) + (a.m02 * b.m20) + (a.m03 * b.m30);
        r.m01 = (a.m00 * b.m01) + (a.m01 * b.m11) + (a.m02 * b.m21) + (a.m03 * b.m31);
        r.m02 = (a.m00 * b.m02) + (a.m01 * b.m12) + (a.m02 * b.m22) + (a.m03 * b.m32);
        r.m03 = (a.m00 * b.m03) + (a.m01 * b.m13) + (a.m02 * b.m23) + (a.m03 * b.m33);
        r.m10 = (a.m10 * b.m00) + (a.m11 * b.m10) + (a.m12 * b.m20) + (a.m13 * b.m30);
        r.m11 = (a.m10 * b.m01) + (a.m11 * b.m11) + (a.m12 * b.m21) + (a.m13 * b.m31);
        r.m12 = (a.m10 * b.m02) + (a.m11 * b.m12) + (a.m12 * b.m22) + (a.m13 * b.m32);
        r.m13 = (a.m10 * b.m03) + (a.m11 * b.m13) + (a.m12 * b.m23) + (a.m13 * b.m33);
        r.m20 = (a.m20 * b.m00) + (a.m21 * b.m10) + (a.m22 * b.m20) + (a.m23 * b.m30);
        r.m21 = (a.m20 * b.m01) + (a.m21 * b.m11) + (a.m22 * b.m21) + (a.m23 * b.m31);
        r.m22 = (a.m20 * b.m02) + (a.m21 * b.m12) + (a.m22 * b.m22) + (a.m23 * b.m32);
        r.m23 = (a.m20 * b.m03) + (a.m21 * b.m13) + (a.m22 * b.m23) + (a.m23 * b.m33);
        r.m30 = (a.m30 * b.m00) + (a.m31 * b.m10) + (a.m32 * b.m20) + (a.m33 * b.m30);
        r.m31 = (a.m30 * b.m01) + (a.m31 * b.m11) + (a.m32 * b.m21) + (a.m33 * b.m31);
        r.m32 = (a.m30 * b.m02) + (a.m31 * b.m12) + (a.m32 * b.m22) + (a.m33 * b.m32);
        r.m33 = (a.m30 * b.m03) + (a.m31 * b.m13) + (a.m32 * b.m23) + (a.m33 * b.m33);
        return r;
    }

    // dictionary utilities
    static const uint32_t k_vox_max_dict_buffer_size = 4096;
    static const uint32_t k_vox_max_dict_key_value_pairs = 256;
    struct _vox_dictionary {
        const char* keys[k_vox_max_dict_key_value_pairs];
        const char* values[k_vox_max_dict_key_value_pairs];
        uint32_t    num_key_value_pairs;
        char        buffer[k_vox_max_dict_buffer_size + 4];    // max 4096, +4 for safety
        uint32_t    buffer_mem_used;
    };

 

    // lookup table for _vox_make_transform_from_dict_strings
    static const vector_3 k_vectors[4] = {
    vector_3_make(1.0f, 0.0f, 0.0f),
    vector_3_make(0.0f, 1.0f, 0.0f),
    vector_3_make(0.0f, 0.0f, 1.0f),
    vector_3_make(0.0f, 0.0f, 0.0f)    // invalid!
    };

    // lookup table for _vox_make_transform_from_dict_strings
    static const uint32_t k_row2_index[] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, 2, UINT32_MAX, 1, 0, UINT32_MAX };


  

    enum _vox_scene_node_type
    {
        k_nodetype_invalid   = 0,    // has not been parsed yet.
        k_nodetype_group     = 1,
        k_nodetype_transform = 2,
        k_nodetype_shape     = 3,
    };

    struct _vox_keyframe_transform {
        uint32_t          frame_index;
        ogt_vox_transform frame_transform;
    };

    struct _vox_keyframe_shape {
        uint32_t frame_index;
        uint32_t model_index;
    };

    struct _vox_scene_node_ {
        _vox_scene_node_type node_type;    // only gets assigned when this has been parsed, otherwise will be k_nodetype_invalid
        union {
            // used only when node_type == k_nodetype_transform
            struct {
                char              name[65];                 // max name size is 64 plus 1 for null terminator
                ogt_vox_transform transform;                // root transform (always the first anim frame transform in the case of animated transform)
                uint32_t          child_node_id;
                uint32_t          layer_id;
                bool              hidden;
                uint32_t          num_keyframes;           // number of key frames in this transform
                size_t            keyframe_offset;         // offset in misc_data array where the _vox_keyframe_transform data is stored.
                bool              loop;                    // keyframes are marked as looping
            } transform;
            // used only when node_type == k_nodetype_group
            struct {
                uint32_t first_child_node_id_index; // the index of the first child node ID within the ChildNodeID array
                uint32_t num_child_nodes;           // number of child node IDs starting at the first index
            } group;
            // used only when node_type == k_nodetype_shape
            struct {
                uint32_t model_id;                 // always the first model_id in the case of an animated shape
                uint32_t num_keyframes;            // number of key frames in this transform
                size_t   keyframe_offset;          // offset in misc_data array where the _vox_keyframe_shape data is stored
                bool     loop;                     // keyframes are marked as looping
            } shape;
        } u;
    };


    // std::vector-style allocator, which use client-provided allocation functions.
    template <class T> struct _vox_array {
        _vox_array() : data(NULL), capacity(0), count(0) { }
        ~_vox_array() {
            _vox_free(data);
            data = NULL;
            count = 0;
            capacity = 0;
        }
        void reserve(size_t new_capacity) {
            data = (T*)_vox_realloc(data, capacity * sizeof(T), new_capacity * sizeof(T));
            capacity = new_capacity;
        }
        void grow_to_fit_index(size_t index) {
            if (index >= count)
                resize(index + 1);
        }
        void resize(size_t new_count) {
            if (new_count > capacity)
            {
                size_t new_capacity = capacity ? (capacity * 3) >> 1 : 2;   // grow by 50% each time, otherwise start at 2 elements.
                new_capacity = new_count > new_capacity ? new_count : new_capacity; // ensure fits new_count
                reserve(new_capacity);
                ogt_assert(capacity >= new_count, "failed to resize array");
            }
            count = new_count;
        }
        void push_back(const T& new_element) {
            if (count == capacity) {
                size_t new_capacity = capacity ? (capacity * 3) >> 1 : 2;   // grow by 50% each time, otherwise start at 2 elements.
                reserve(new_capacity);
                ogt_assert(capacity > count, "failed to push_back in array");
            }
            data[count++] = new_element;
        }
        void pop_back() {
            ogt_assert(count > 0, "cannot pop_back on empty array");
            count--;
        }
        const T& peek_back(size_t back_offset = 0) const {
            ogt_assert(back_offset < count, "can't peek back further than the number of elements in an array");
            size_t index = count - 1 - back_offset;
            return data[index];
        }

        void push_back_many(const T* new_elements, size_t num_elements) {
            if (count + num_elements > capacity) {
                size_t new_capacity = capacity + num_elements;
                new_capacity = new_capacity ? (new_capacity * 3) >> 1 : 2;   // grow by 50% each time, otherwise start at 2 elements.
                reserve(new_capacity);
                ogt_assert(capacity >= (count + num_elements), "failed to push_back_many in array");
            }
            for (size_t i = 0; i < num_elements; i++)
                data[count + i] = new_elements[i];
            count += num_elements;
        }
        T* alloc_many(size_t num_elements) {
            if (count + num_elements > capacity) {
                size_t new_capacity = capacity + num_elements;
                new_capacity = new_capacity ? (new_capacity * 3) >> 1 : 2;   // grow by 50% each time, otherwise start at 2 elements.
                reserve(new_capacity);
                ogt_assert(capacity >= (count + num_elements), "failed to push_back_many in array");
            }
            T* ret = &data[count];
            count += num_elements;
            return ret;
        }

        // returns the index that it was inserted at.
        uint32_t insert_unique_sorted(const T& value, uint32_t start_index) {
            for (uint32_t i = start_index; i < (uint32_t)count; i++) {
                if (data[i] == value)
                    return i;
                if (data[i] >= value) {
                    resize(count + 1);
                    for (size_t j = count - 1; j > i; j--)
                        data[j] = data[j - 1];
                    data[i] = value;
                    return i;
                }
            }
            push_back(value);
            return (uint32_t)(count - 1);
        }

        size_t size() const {
            return count;
        }
        T& operator[](size_t index) {
            ogt_assert(index < count, "index out of bounds");
            return data[index];
        }
        const T& operator[](size_t index) const {
            ogt_assert(index < count, "index out of bounds");
            return data[index];
        }
        T* data;      // data for the array
        size_t capacity;  // capacity of the array
        size_t count;      // size of the array
    };



    //void ogt_vox_test()
    //{
    //    // frame_index looping tests
    //    {
    //        const char* test_message = "failed compute_looped_frame_index test";
    //        (void)test_message;
    //        // [0,0] = 1 keyframe animation starting at frame 0
    //        ogt_assert(compute_looped_frame_index( 0,  0, 0 ) == 0, test_message);
    //        ogt_assert(compute_looped_frame_index( 0,  0, 1 ) == 0, test_message);
    //        ogt_assert(compute_looped_frame_index( 0,  0, 15) == 0, test_message);
    //        // [1,1] = 1 keyframe animation starting at frame 1
    //        ogt_assert(compute_looped_frame_index( 1,  1,  0) == 1, test_message);
    //        ogt_assert(compute_looped_frame_index( 1,  1,  1) == 1, test_message);
    //        ogt_assert(compute_looped_frame_index( 1,  1, 15) == 1, test_message);
    //        // [0,9] = 10 keyframe animation starting at frame 0
    //        ogt_assert(compute_looped_frame_index( 0,  9,  0) == 0, test_message);
    //        ogt_assert(compute_looped_frame_index( 0,  9,  4) == 4, test_message);
    //        ogt_assert(compute_looped_frame_index( 0,  9,  9) == 9, test_message);
    //        ogt_assert(compute_looped_frame_index( 0,  9, 10) == 0, test_message);
    //        ogt_assert(compute_looped_frame_index( 0,  9, 11) == 1, test_message);
    //        ogt_assert(compute_looped_frame_index( 0,  9, 14) == 4, test_message);
    //        ogt_assert(compute_looped_frame_index( 0,  9, 19) == 9, test_message);
    //        ogt_assert(compute_looped_frame_index( 0,  9, 21) == 1, test_message);
    //        // [4,13] = 10 keyframe animation starting at frame 4
    //        ogt_assert(compute_looped_frame_index(4, 13, 0 ) == 10, test_message);
    //        ogt_assert(compute_looped_frame_index(4, 13, 3 ) == 13, test_message);
    //        ogt_assert(compute_looped_frame_index(4, 13, 4 ) == 4,  test_message);
    //        ogt_assert(compute_looped_frame_index(4, 13, 5 ) == 5,  test_message);
    //        ogt_assert(compute_looped_frame_index(4, 13, 12) == 12, test_message);
    //        ogt_assert(compute_looped_frame_index(4, 13, 13) == 13, test_message);
    //        ogt_assert(compute_looped_frame_index(4, 13, 14) == 4,  test_message);
    //        ogt_assert(compute_looped_frame_index(4, 13, 21) == 11, test_message);
    //    }

    //}

 #endif // #ifdef OGT_VOX_IMPLEMENTATION

/* -------------------------------------------------------------------------------------------------------------------------------------------------

    MIT License

    Copyright (c) 2019 Justin Paver

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

------------------------------------------------------------------------------------------------------------------------------------------------- */
