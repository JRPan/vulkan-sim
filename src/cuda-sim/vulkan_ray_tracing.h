// Copyright (c) 2022, Mohammadreza Saed, Yuan Hsi Chou, Lufei Liu, Tor M. Aamodt,
// The University of British Columbia
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef VULKAN_RAY_TRACING_H
#define VULKAN_RAY_TRACING_H

#include "vulkan/vulkan.h"
#if defined(MESA_USE_INTEL_DRIVER)
#include "vulkan/vulkan_intel.h"
#include "vulkan/anv_acceleration_structure.h"

#elif defined(MESA_USE_LVPIPE_DRIVER)
#include "lvp_acceleration_structure.h"
#include "gallium/drivers/llvmpipe/lp_texture.h"
#endif

#include "intersection_table.h"
#include "compiler/spirv/spirv.h"

// #include "ptx_ir.h"
#include "ptx_ir.h"
#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "compiler/shader_enums.h"
#include <fstream>
#include <cmath>
#include <unordered_map>
#include <mutex>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN_MAX(a,b,c) MAX(MIN((a), (b)), (c))
#define MAX_MIN(a,b,c) MIN(MAX((a), (b)), (c))

#define MAX_DESCRIPTOR_SETS 1
#define MAX_DESCRIPTOR_SET_BINDINGS 32

// enum class TransactionType {
//     BVH_STRUCTURE,
//     BVH_INTERNAL_NODE,
//     BVH_INSTANCE_LEAF,
//     BVH_PRIMITIVE_LEAF_DESCRIPTOR,
//     BVH_QUAD_LEAF,
//     BVH_PROCEDURAL_LEAF,
//     Intersection_Table_Load,
// };

// typedef struct MemoryTransactionRecord {
//     MemoryTransactionRecord(void* address, uint32_t size, TransactionType type)
//     : address(address), size(size), type(type) {}
//     void* address;
//     uint32_t size;
//     TransactionType type;
// } MemoryTransactionRecord;
// typedef struct float4 {
//     float x, y, z, w;
// } float4;

// enum class StoreTransactionType {
//     Intersection_Table_Store,
//     Traversal_Results,
// };

// typedef struct MemoryStoreTransactionRecord {
//     MemoryStoreTransactionRecord(void* address, uint32_t size, StoreTransactionType type)
//     : address(address), size(size), type(type) {}
//     void* address;
//     uint32_t size;
//     StoreTransactionType type;
// } MemoryStoreTransactionRecord;



extern bool use_external_launcher;

typedef struct float4x4 {
  float m[4][4];

  float4 operator*(const float4& _vec) const
  {
    float vec[] = {_vec.x, _vec.y, _vec.z, _vec.w};
    float res[] = {0, 0, 0, 0};
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            res[i] += this->m[j][i] * vec[j];
    return {res[0], res[1], res[2], res[3]};
  }
} float4x4;

typedef struct RayDebugGPUData
{
    bool valid;
    int launchIDx;
    int launchIDy;
    int instanceCustomIndex;
    int primitiveID;
    float3 v0pos;
    float3 v1pos;
    float3 v2pos;
    float3 attribs;
    float3 P_object;
    float3 P; //world intersection point
    float3 N_object;
    float3 N;
    float NdotL;
    float3 hitValue;
} RayDebugGPUData;

// float4 operator*(const float4& _vec, const float4x4& matrix)
// {
//     float vec[] = {_vec.x, _vec.y, _vec.z, _vec.w};
//     float res[] = {0, 0, 0, 0};
//     for(int i = 0; i < 4; i++)
//         for(int j = 0; j < 4; j++)
//             res[i] += matrix.m[j][i] * vec[j];
//     return {res[0], res[1], res[2], res[3]};
// }


typedef struct Descriptor
{
    uint32_t setID;
    uint32_t descID;
    void *address;
    uint32_t size;
    VkDescriptorType type;
} Descriptor;

typedef struct shader_stage_info {
    uint32_t ID;
    gl_shader_stage type;
    char* function_name;
} shader_stage_info;


typedef struct Pixel{
    Pixel(float c0, float c1, float c2, float c3)
    : c0(c0), c1(c1), c2(c2), c3(c3) {}
    Pixel() {}

    union
    {
        float r;
        float c0;
    };
    union
    {
        float g;
        float c1;
    };
    union
    {
        float b;
        float c2;
    };
    union
    {
        float a;
        float c3;
    };
} Pixel;

// For launcher
typedef struct storage_image_metadata
{
    void *address;
    void *deviceAddress;
    uint32_t setID;
    uint32_t descID;
    uint32_t width;
    uint32_t height;
    VkFormat format;
    uint32_t VkDescriptorTypeNum;
    uint32_t n_planes;
    uint32_t n_samples;
    VkImageTiling tiling;
    uint32_t isl_tiling_mode; 
    uint32_t row_pitch_B;
} storage_image_metadata;

typedef struct texture_metadata
{
    void *address;
    void *deviceAddress;
    uint32_t setID;
    uint32_t descID;
    uint64_t size;
    uint32_t width;
    uint32_t height;
    VkFormat format;
    uint32_t VkDescriptorTypeNum;
    uint32_t n_planes;
    uint32_t n_samples;
    VkImageTiling tiling;
    uint32_t isl_tiling_mode;
    uint32_t row_pitch_B;
    VkFilter filter;
} texture_metadata;


#define MAX_VERTEX 28
typedef struct VertexAttrib {
    std::vector<unsigned> location;
    std::vector<unsigned> binding;
    std::vector<unsigned> offset;
    std::vector<unsigned> rate;
} VertexAttrib;

typedef struct desc_ptr {
    void *addr;
    uint32_t size;
    bool is_texture;

    desc_ptr() {
        addr = NULL;
        size = 0;
        is_texture = false;
    }
} desc_ptr;

typedef struct issue_info {
    unsigned shader_id;
    unsigned batch_index;
    std::vector<unsigned> warp;
    issue_info(unsigned bid, unsigned sid, std::vector<unsigned> w) {
        shader_id = sid;
        batch_index = bid;
        warp = w;
    }
} issue_info;

typedef struct vertex_metadata
{
    struct pipe_context *pipe;
    // assuming all data are 4-Byte
    // *device* vertex buffer
    std::vector<unsigned> vb;
    std::vector<std::unordered_map<unsigned, unsigned>> batched_idx_to_tid;
    std::vector<std::vector<unsigned>> batched_prim;
    std::unordered_set<unsigned> vb_deactive;
    // std::unordered_map<unsigned, unsigned> index_to_batched;
    float *vertex_buffers[MAX_VERTEX] = {NULL};
    void *ubo_addr[PIPE_SHADER_MESH_TYPES][16] = {NULL};
    void *ubo_addr_dev[PIPE_SHADER_MESH_TYPES][16] = {NULL};
    unsigned ubo_offset[PIPE_SHADER_MESH_TYPES][16] = {0};
    unsigned ubo_size[PIPE_SHADER_MESH_TYPES][16] = {0};
    uint32_t* vertex_addr[MAX_VERTEX] = {NULL};
    // vertex buffer size
    uint32_t vertex_size[MAX_VERTEX] = {0};
    uint32_t vertex_count[MAX_VERTEX] = {0};
    uint32_t vertex_stride[MAX_VERTEX] = {0};

    void* index_buffer = NULL;
    unsigned index_size = -1;
    unsigned index_buf_size = 0;
    uint32_t *constants_dev_addr = 0;

    std::map<std::string, uint32_t*> vertex_out_devptr;
    std::unordered_map<std::string, unsigned> vertex_out_stride;
    std::unordered_map<std::string, unsigned> vertex_out_count;
    std::unordered_map<std::string, unsigned> vertex_out_size;
    std::unordered_map<std::string, float*> vertex_out;
    std::vector<std::vector<unsigned>> primitives;
    // vertex-post processing
    // tranform & clipping
    std::vector<std::vector<float>> vertex_ndc;
    std::vector<std::vector<float>> vertex_screen;
    std::vector<std::vector<float>> vertex_raw;

    // FS
    std::unordered_map<std::string, std::vector<std::vector<float>>> attribs;
    std::vector<std::vector<unsigned>> target_sm;
    std::vector<unsigned> thread_info_pixel;
    std::vector<issue_info> issue_order;
    std::unordered_map<unsigned, unsigned> pixel_map;

    unsigned VertexCountPerInstance;
    unsigned StartVertexLocation;
    unsigned InstanceCount;
    unsigned StartInstanceLocation;
    unsigned BaseVertexLocation;
    VkCompareOp DepthcmpOp;
    struct VertexAttrib *VertexAttrib = NULL;
    VkViewport viewports;
    struct anv_descriptor_set *descriptor_set[8] = {NULL};

    ~vertex_metadata() {
      for (auto attrib : vertex_out) {
        delete[] attrib.second;
      }
      delete VertexAttrib;
    }

}vertex_metadata;

typedef struct FBO {
  float *fbo = NULL;
  float *fbo_dev = NULL;
  float *depthout = NULL;
  unsigned fbo_size = 0;
  unsigned fbo_count = 0;
  unsigned fbo_stride = 0;
  unsigned width = -1;
  unsigned height = -1;
  unsigned x = -1;
  unsigned y = -1;
  std::vector<float> thread_info_lod;

//   ~FBO() {
//     delete[] fbo;
//     delete[] depthout;
//   }
} FBO;


#if defined(MESA_USE_INTEL_DRIVER)
#define DESCRIPTOR_SET_STRUCT anv_descriptor_set
#define DESCRIPTOR_STRUCT anv_descriptor
#define DESCRIPTOR_LAYOUT_STRUCT anv_descriptor_set_binding_layout

#define VSIM_DEBUG_PRINT 0
struct anv_descriptor_set;
struct anv_descriptor;

#elif defined(MESA_USE_LVPIPE_DRIVER)
#define DESCRIPTOR_SET_STRUCT lvp_descriptor_set
#define DESCRIPTOR_STRUCT lvp_descriptor
#define DESCRIPTOR_LAYOUT_STRUCT lvp_descriptor_set_binding_layout

struct lvp_descriptor_set;
struct lvp_descriptor;

#endif

#define VSIM_DPRINTF(...) \
   if(VSIM_DEBUG_PRINT) { \
      printf(__VA_ARGS__); \
      fflush(stdout); \
   }

class VulkanRayTracing
{
private:
    static VkRayTracingPipelineCreateInfoKHR* pCreateInfos;
    static VkAccelerationStructureGeometryKHR* pGeometries;
    static uint32_t geometryCount;
    static VkAccelerationStructureKHR topLevelAS;
    static std::vector<std::vector<Descriptor> > descriptors;
    static std::ofstream imageFile;
    static bool firstTime;
    static struct DESCRIPTOR_SET_STRUCT *descriptorSet;

    // For Launcher
    static void* launcher_descriptorSets[MAX_DESCRIPTOR_SETS][MAX_DESCRIPTOR_SET_BINDINGS];
    static void* launcher_deviceDescriptorSets[MAX_DESCRIPTOR_SETS][MAX_DESCRIPTOR_SET_BINDINGS];
    static std::vector<void*> child_addrs_from_driver;
    static std::map<void*, void*> VulkanRayTracing::blas_addr_map;
    static void* tlas_addr;
    static bool dumped;
    static bool _init_;
public:
    // static RayDebugGPUData rayDebugGPUData[2000][2000];
    static warp_intersection_table*** intersection_table;
    static warp_intersection_table*** anyhit_table;
    static IntersectionTableType intersectionTableType;

private:
    static bool mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit);
    static float3 Barycentric(float3 p, float3 a, float3 b, float3 c);
    static std::vector<shader_stage_info> shaders;

    static void init(uint32_t launch_width, uint32_t launch_height);


public:
    static void traceRay( // called by raygen shader
                       VkAccelerationStructureKHR _topLevelAS,
    				   uint rayFlags,
                       uint cullMask,
                       uint sbtRecordOffset,
                       uint sbtRecordStride,
                       uint missIndex,
                       float3 origin,
                       float Tmin,
                       float3 direction,
                       float Tmax,
                       int payload,
                       const ptx_instruction *pI,
                       ptx_thread_info *thread);
    static void endTraceRay(const ptx_instruction *pI, ptx_thread_info *thread);
    
    static void load_descriptor(const ptx_instruction *pI, ptx_thread_info *thread);

    static void setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos);
    static void setGeometries(VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount);
    static void setAccelerationStructure(VkAccelerationStructureKHR accelerationStructure);
    static void setDescriptorSet(struct DESCRIPTOR_SET_STRUCT *set);
    static void invoke_gpgpusim();
    static uint32_t registerShaders(char * shaderPath, gl_shader_stage shaderType);
    static void vkCmdTraceRaysKHR( // called by vulkan application
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr);
    static void callShader(const ptx_instruction *pI, ptx_thread_info *thread, function_info *target_func);
    static void callMissShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void callClosestHitShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void callIntersectionShader(const ptx_instruction *pI, ptx_thread_info *thread, uint32_t shader_counter);
    static void callAnyHitShader(const ptx_instruction *pI, ptx_thread_info *thread, uint32_t shader_counter);
    static void setDescriptor(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);
    static void* getDescriptorAddress(uint32_t setID, uint32_t binding);

    static void image_store(struct DESCRIPTOR_STRUCT* desc, uint32_t gl_LaunchIDEXT_X, uint32_t gl_LaunchIDEXT_Y, uint32_t gl_LaunchIDEXT_Z, uint32_t gl_LaunchsIDEXT_W, 
              float hitValue_X, float hitValue_Y, float hitValue_Z, float hitValue_W, const ptx_instruction *pI, ptx_thread_info *thread);
    static void getTexture(struct DESCRIPTOR_STRUCT *desc, float x, float y, float lod, float &c0, float &c1, float &c2, float &c3, std::vector<ImageMemoryTransactionRecord>& transactions, uint64_t launcher_offset = 0);
    static void image_load(struct DESCRIPTOR_STRUCT *desc, uint32_t x, uint32_t y, float &c0, float &c1, float &c2, float &c3);

    static void dump_descriptor_set(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);
    static void dump_descriptor_set_for_AS(uint32_t setID, uint32_t descID, void *address, uint32_t desc_size, VkDescriptorType type, uint32_t backwards_range, uint32_t forward_range, bool split_files, VkAccelerationStructureKHR _topLevelAS);
    static void dump_descriptor_sets(struct DESCRIPTOR_SET_STRUCT *set);
    static void dump_AS(struct DESCRIPTOR_SET_STRUCT *set, VkAccelerationStructureKHR _topLevelAS);
    static void dump_callparams_and_sbt(void *raygen_sbt, void *miss_sbt, void *hit_sbt, void *callable_sbt, bool is_indirect, uint32_t launch_width, uint32_t launch_height, uint32_t launch_depth, uint32_t launch_size_addr);
    static void dumpTextures(struct DESCRIPTOR_STRUCT *desc, uint32_t setID, uint32_t binding, VkDescriptorType type);
    static void dumpStorageImage(struct DESCRIPTOR_STRUCT *desc, uint32_t setID, uint32_t binding, VkDescriptorType type);
    static void setDescriptorSetFromLauncher(void *address, void *deviceAddress, uint32_t setID, uint32_t descID);
    static void setStorageImageFromLauncher(void *address, 
                                            void *deviceAddress,
                                            uint32_t setID, 
                                            uint32_t descID, 
                                            uint32_t width,
                                            uint32_t height,
                                            VkFormat format,
                                            uint32_t VkDescriptorTypeNum,
                                            uint32_t n_planes,
                                            uint32_t n_samples,
                                            VkImageTiling tiling,
                                            uint32_t isl_tiling_mode, 
                                            uint32_t row_pitch_B);
    static void setTextureFromLauncher(void *address,
                                       void *deviceAddress, 
                                       uint32_t setID, 
                                       uint32_t descID, 
                                       uint64_t size,
                                       uint32_t width,
                                       uint32_t height,
                                       VkFormat format,
                                       uint32_t VkDescriptorTypeNum,
                                       uint32_t n_planes,
                                       uint32_t n_samples,
                                       VkImageTiling tiling,
                                       uint32_t isl_tiling_mode,
                                       uint32_t row_pitch_B,
                                       uint32_t filter);
    static void pass_child_addr(void *address);
    static void allocBLAS(void* rootAddr, uint64_t bufferSize, void* gpgpusimAddr);
    static void allocTLAS(void* rootAddr, uint64_t bufferSize, void* gpgpusimAddr);
    static void* allocBuffer(void* bufferAddr, uint64_t bufferSize);
    static void findOffsetBounds(int64_t &max_backwards, int64_t &min_backwards, int64_t &min_forwards, int64_t &max_forwards, VkAccelerationStructureKHR _topLevelAS);
    static void* gpgpusim_alloc(uint32_t size);

    // CRISP
    static bool use_CRISP;
    static std::mutex mtx;
    static struct FBO *FBO;
    static struct vertex_metadata *VertexMeta;
    static unsigned draw;
    static bool is_FS;
    static unsigned thread_count;
    static unsigned batch_index;
    static unsigned batch_size;
    static std::set<unsigned> active_vs_threads;
    static std::deque<struct vertex_metadata* > draw_meta;
    static std::unordered_map<void *, unsigned> pipeline_shader_map;
    static std::unordered_map<void *, struct VertexAttrib *> pipeline_vertex_map;
    static void VulkanRayTracing::run_shader(unsigned shader_id,
                                             unsigned thread_count);
    static void VulkanRayTracing::vkCmdDraw();
    static void VulkanRayTracing::read_binary_file(std::string path, void *ptr,
                                                   unsigned size);
    static void VulkanRayTracing::saveIndexBuffer(struct anv_buffer *ptr,
                                                  VkIndexType type);
    static uint64_t getVertexAddr(uint32_t buffer_index, uint32_t tid);
    static uint64_t getVertexOutAddr(std::string index, uint32_t tid);
    static uint64_t VulkanRayTracing::getFBOAddr(uint32_t tid);
    static void VulkanRayTracing::getFragCoord(uint32_t thread_id, uint32_t &x,
                                               uint32_t &y);
    static uint64_t VulkanRayTracing::getConst();
    static float VulkanRayTracing::getTexLOD(unsigned thread_id);
    static void bindVertex(unsigned index, float *addr, unsigned size, unsigned stride);
    static void saveIndexBuffer(void *ptr, unsigned index_size, unsigned buf_size);
    static void saveVertexInfo(unsigned location, unsigned binding, unsigned offset, unsigned rate);
    static void saveInstance(unsigned instanceCount, unsigned startInstance);
    static void saveUBO(pipe_shader_type stage, unsigned index, unsigned offset, unsigned size, void *addr);
    static void savePipeCtx(void *pipe);
    static addr_t getUBOAddr( unsigned index, unsigned offset);
    static void post_vertex();
    static void generate_frag(unsigned batch_index);
    static void genLoD();
    static void cleanup();
    static void dumpFBO();
    static void pre_frag();
    static float linearRGB_to_SRGB(float s) {
      // assert(0 <= s && s <= 1);
      if (s < 0.0031308)
        return s * 12.92;
      else
        return 1.055 * std::pow(s, 1 / 2.4) - 0.055;
    }

    static float SRGB_to_linearRGB(float s) {
      assert(0 <= s && s <= 1);
      if (s <= 0.04045)
        return s / 12.92;
      else
        return pow(((s + 0.055) / 1.055), 2.4);
    }
};

#endif /* VULKAN_RAY_TRACING_H */
