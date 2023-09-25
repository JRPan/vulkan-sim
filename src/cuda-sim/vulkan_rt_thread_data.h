#ifndef VULKAN_RT_THREAD_DATA_H
#define VULKAN_RT_THREAD_DATA_H

#include "vulkan/vulkan.h"
#include "vulkan/vulkan_intel.h"

#include "vulkan_ray_tracing.h"

// #include "ptx_ir.h"
#include <cmath>
#include <fstream>
#include <stack>
#include "../../libcuda/gpgpu_context.h"
#include "compiler/shader_enums.h"
#include "ptx_ir.h"

typedef struct variable_decleration_entry {
  uint64_t type;
  std::string name;
  uint64_t address;
  uint32_t size;
} variable_decleration_entry;

typedef struct Hit_data {
  VkGeometryTypeKHR geometryType;
  float world_min_thit;
  uint32_t geometry_index;
  uint32_t primitive_index;
  float3 intersection_point;
  float3 barycentric_coordinates;
  int32_t
      hitGroupIndex;  // Shader ID of the closest hit for procedural geometries

  uint32_t instance_index;
  float4x4 worldToObjectMatrix;
  float4x4 objectToWorldMatrix;
} Hit_data;

typedef struct Traversal_data {
  bool hit_geometry;
  Hit_data closest_hit;
  float3 ray_world_direction;
  float3 ray_world_origin;
  float Tmin;
  float Tmax;
  int32_t current_shader_counter;  // set to shader_counter in call_intersection
                                   // and -1 in call_miss and call_closest_hit

  uint32_t rayFlags;
  uint32_t cullMask;
  uint32_t sbtRecordOffset;
  uint32_t sbtRecordStride;
  uint32_t missIndex;
} Traversal_data;

typedef struct Vulkan_RT_thread_data {
  std::vector<variable_decleration_entry> variable_decleration_table;

  std::vector<Traversal_data*> traversal_data;

  variable_decleration_entry* get_variable_decleration_entry(uint64_t type,
                                                             std::string name,
                                                             uint32_t size) {
    if (type == 8192) return get_hitAttribute();

    for (int i = 0; i < variable_decleration_table.size(); i++) {
      if (variable_decleration_table[i].name == name) {
        assert(variable_decleration_table[i].address != NULL);
        return &(variable_decleration_table[i]);
      }
    }
    return NULL;
  }

  uint64_t add_variable_decleration_entry(uint64_t type, std::string name,
                                          std::string identifier, uint32_t size,
                                          uint32_t offset) {
    variable_decleration_entry entry;
    entry.type = type;
    entry.name = name;

    gpgpu_context* ctx = GPGPU_Context();
    CUctx_st* context = GPGPUSim_Context(ctx);

    std::vector<std::string> v;
    std::stringstream ss(identifier);

    while (ss.good()) {
      std::string substr;
      getline(ss, substr, '_');
      v.push_back(substr);
    }

    unsigned attrib_index = -1;
    if (!VulkanRayTracing::is_FS) {
      if (name.find("\%draw_call") != std::string::npos) {
        entry.address = VulkanRayTracing::getConst();

        context->get_device()->get_gpgpu()->identifier_addr["push_const"] =
            "push_const";

      } else
      if (identifier.find("VERT_ATTRIB_GENERIC") != std::string::npos) {
        // vertex shader input attributes
        assert(v.size() == 4);
        attrib_index = std::atoi(&v[2].back());
        entry.address = VulkanRayTracing::getVertexAddr(attrib_index, offset);

        std::string attrib_name =
            "VERT_ATTRIB_GENERIC" + std::to_string(attrib_index);
        context->get_device()->get_gpgpu()->identifier_addr[attrib_name] =
            identifier;

      } else if (identifier.find("VARYING_SLOT_POS") != std::string::npos) {
        assert(identifier.find("xyzw") != std::string::npos);
        if (VulkanRayTracing::VertexMeta->vertex_out_devptr.find(name) ==
            VulkanRayTracing::VertexMeta->vertex_out_devptr.end()) {
          uint32_t* dev_ptr = context->get_device()->get_gpgpu()->gpu_malloc(
              VulkanRayTracing::thread_count * size);
          VulkanRayTracing::VertexMeta->vertex_out_devptr.insert(
              std::make_pair(name, dev_ptr));
          VulkanRayTracing::VertexMeta->vertex_id_map.insert(
              std::make_pair(identifier, name));
          VulkanRayTracing::VertexMeta->vertex_out_stride.insert(
              std::make_pair(name, size));
          context->get_device()
              ->get_gpgpu()
              ->identifier_addr["VARYING_SLOT_POS"] = identifier;
        }
        entry.address = VulkanRayTracing::getVertexOutAddr(name, offset);

      } else if (identifier.find("VARYING_SLOT_VAR") != std::string::npos) {
        if (VulkanRayTracing::VertexMeta->vertex_out_devptr.find(name) ==
            VulkanRayTracing::VertexMeta->vertex_out_devptr.end()) {
          uint32_t* dev_ptr = context->get_device()->get_gpgpu()->gpu_malloc(
              VulkanRayTracing::thread_count * size);
          VulkanRayTracing::VertexMeta->vertex_out_devptr.insert(
              std::make_pair(name, dev_ptr));
          VulkanRayTracing::VertexMeta->vertex_id_map.insert(
              std::make_pair(identifier, name));
          VulkanRayTracing::VertexMeta->vertex_out_stride.insert(
              std::make_pair(name, size));
          attrib_index = std::atoi(&v[2].back());
          std::string attrib_name =
              "VARYING_SLOT_VAR" + std::to_string(attrib_index);
          context->get_device()->get_gpgpu()->identifier_addr[attrib_name] =
              identifier;
        }

        entry.address = VulkanRayTracing::getVertexOutAddr(name, offset);
      } else {
        assert(identifier.find("UNDEFINED") != std::string::npos);
        if (VulkanRayTracing::VertexMeta->vertex_out_devptr.find(name) ==
            VulkanRayTracing::VertexMeta->vertex_out_devptr.end()) {
          uint32_t* dev_ptr = context->get_device()->get_gpgpu()->gpu_malloc(
              VulkanRayTracing::thread_count * size);
          VulkanRayTracing::VertexMeta->vertex_out_devptr.insert(
              std::make_pair(name, dev_ptr));
          VulkanRayTracing::VertexMeta->vertex_out_stride.insert(
              std::make_pair(name, size));
        }

        entry.address = VulkanRayTracing::getVertexOutAddr(name, offset);
      }
    } else {
      /*if (identifier.find("VARYING_SLOT_VAR0_xyzw") != std::string::npos){
        entry.address = VulkanRayTracing::getVertexOutAddr("\%field0", offset);

      } else */if (identifier.find("VARYING_SLOT_VAR") != std::string::npos) {
        assert(VulkanRayTracing::VertexMeta->vertex_id_map.find(identifier) !=
               VulkanRayTracing::VertexMeta->vertex_id_map.end());
        
        std::string attrib_name =
            "VARYING_SLOT_VAR" + std::to_string(attrib_index);
        context->get_device()->get_gpgpu()->identifier_addr[attrib_name] =
            identifier;
        name = VulkanRayTracing::VertexMeta->vertex_id_map.at(identifier);
        attrib_index = std::atoi(&v[2].back());

        entry.address = VulkanRayTracing::getVertexOutAddr(name, offset);
      } else if (identifier.find("FRAG_RESULT_DATA0_xyzw") != std::string::npos) {
        entry.address = VulkanRayTracing::getFBOAddr(offset);
        context->get_device()
            ->get_gpgpu()
            ->identifier_addr["FRAG_RESULT_DATA0"] = identifier;
      } else {
        assert(0);
      }
    }

    // unsigned buffer_index = -1;
    // // TODO: update for each app
    // if (VulkanRayTracing::app_id == INSTANCING && VulkanRayTracing::draw == 1) {
    //   // instancing draw #1
    //   if (!VulkanRayTracing::is_FS) {
    //     // VS in
    //     if (name == "\%inPos") {
    //       buffer_index = 0;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%inNormal") {
    //       buffer_index = 1;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%inUV") {
    //       buffer_index = 2;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%instancePos") {
    //       buffer_index = 3;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%instanceRot") {
    //       buffer_index = 4;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%instanceScale") {
    //       buffer_index = 5;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%instanceTexIndex") {
    //       buffer_index = 6;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     }
    //     // VS out
    //     else if (name == "\%outNormal8") {
    //       buffer_index = 0;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%") {
    //       assert(0);
    //       buffer_index = 1;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%outUV9") {
    //       buffer_index = 2;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%outViewVec10") {
    //       buffer_index = 3;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%outLightVec11") {
    //       buffer_index = 4;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%field0") {
    //       buffer_index = 5;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else {
    //       entry.address = (uint64_t)VulkanRayTracing::gpgpusim_alloc(size);
    //     }
    //   } else {
    //     // FS in
    //     if (name == "\%inNormal") {
    //       buffer_index = 0;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inColor") {
    //       buffer_index = 1;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inUV") {
    //       buffer_index = 2;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inViewVec") {
    //       buffer_index = 3;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inLightVec") {
    //       buffer_index = 4;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     }
    //     // FBO
    //     else if (name == "\%outFragColor") {
    //       entry.address = VulkanRayTracing::getFBOAddr(offset);
    //     } else {
    //       entry.address = (uint64_t)VulkanRayTracing::gpgpusim_alloc(size);
    //     }
    //   }
    // } else if (VulkanRayTracing::app_id == INSTANCING &&
    //            VulkanRayTracing::draw == 0) {
    //   // instancing draw #0
    //   if (!VulkanRayTracing::is_FS) {
    //     // VS in
    //     if (name == "\%inPos") {
    //       buffer_index = 0;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%inNormal") {
    //       buffer_index = 1;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%inUV") {
    //       buffer_index = 2;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     }
    //     // VS out
    //     else if (name == "\%outNormal7") {
    //       buffer_index = 0;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%") {
    //       assert(0);
    //       buffer_index = 1;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%outUV8") {
    //       buffer_index = 2;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%outViewVec9") {
    //       buffer_index = 3;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%outLightVec10") {
    //       buffer_index = 4;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%field0") {
    //       buffer_index = 5;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else {
    //       entry.address = (uint64_t)VulkanRayTracing::gpgpusim_alloc(size);
    //     }
    //   } else {
    //     // FS in
    //     if (name == "\%inNormal") {
    //       buffer_index = 0;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inColor") {
    //       buffer_index = 1;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inUV") {
    //       buffer_index = 2;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inViewVec") {
    //       buffer_index = 3;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inLightVec") {
    //       buffer_index = 4;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     }
    //     // FBO
    //     else if (name == "\%outFragColor") {
    //       entry.address = VulkanRayTracing::getFBOAddr(offset);
    //     } else {
    //       entry.address = (uint64_t)VulkanRayTracing::gpgpusim_alloc(size);
    //     }
    //   }
    // } else if (VulkanRayTracing::app_id == RENDER_PASSES) {
    //   if (!VulkanRayTracing::is_FS) {
    //     // VS in
    //     if (name == "\%position") {
    //       buffer_index = 0;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%texcoord_0") {
    //       buffer_index = 1;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%normal") {
    //       buffer_index = 2;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     }
    //     // VS out
    //     else if (name == "\%field0") {
    //       buffer_index = 0;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%o_uv3") {
    //       buffer_index = 1;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%o_normal4") {
    //       buffer_index = 2;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else {
    //       entry.address = (uint64_t)VulkanRayTracing::gpgpusim_alloc(size);
    //     }
    //   } else {
    //     // FS in
    //     if (name == "\%in_pos") {
    //       buffer_index = 0;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%in_uv") {
    //       buffer_index = 1;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%in_normal") {
    //       buffer_index = 2;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     }
    //     // FBO
    //     else if (name == "\%o_color") {
    //       entry.address = VulkanRayTracing::getFBOAddr(offset);
    //     } else {
    //       entry.address = (uint64_t)VulkanRayTracing::gpgpusim_alloc(size);
    //     }
    //   }
    // } else if (VulkanRayTracing::app_id == PBRBASIC) {
    //   if (!VulkanRayTracing::is_FS) {
    //     // VS in
    //     if (name == "\%inPos") {
    //       buffer_index = 0;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%inNormal") {
    //       buffer_index = 1;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%pushConsts") {
    //       entry.address = VulkanRayTracing::getConst();
    //       // VS out
    //     } else if (name == "\%outWorldPos4") {
    //       buffer_index = 0;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%outNormal5") {
    //       buffer_index = 1;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%field0") {
    //       buffer_index = 2;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else {
    //       entry.address = (uint64_t)VulkanRayTracing::gpgpusim_alloc(size);
    //     }
    //   } else {
    //     // FS in
    //     if (name == "\%inWorldPos") {
    //       buffer_index = 0;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inNormal") {
    //       buffer_index = 1;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%material") {
    //       entry.address = VulkanRayTracing::getConst();
    //     }
    //     // FBO
    //     else if (name == "\%outColor") {
    //       entry.address = VulkanRayTracing::getFBOAddr(offset);
    //     } else {
    //       entry.address = (uint64_t)VulkanRayTracing::gpgpusim_alloc(size);
    //     }
    //   }
    // } else if (VulkanRayTracing::app_id == PBRTEXTURE) {
    //   if (!VulkanRayTracing::is_FS) {
    //     // VS in
    //     if (name == "\%inPos") {
    //       buffer_index = 0;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%inNormal") {
    //       buffer_index = 1;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%inUV") {
    //       buffer_index = 2;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //     } else if (name == "\%inTangent") {
    //       buffer_index = 3;
    //       entry.address = VulkanRayTracing::getVertexAddr(buffer_index, offset);
    //       // VS out
    //     } else if (name == "\%outWorldPos7") {
    //       buffer_index = 0;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%outNormal8") {
    //       buffer_index = 1;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%outUV9") {
    //       buffer_index = 2;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%outTangent10") {
    //       buffer_index = 3;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%field0") {
    //       buffer_index = 4;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else {
    //       entry.address = (uint64_t)VulkanRayTracing::gpgpusim_alloc(size);
    //     }
    //   } else {
    //     // FS in
    //     if (name == "\%inWorldPos") {
    //       buffer_index = 0;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inNormal") {
    //       buffer_index = 1;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inUV") {
    //       buffer_index = 2;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     } else if (name == "\%inTangent") {
    //       buffer_index = 3;
    //       entry.address =
    //           VulkanRayTracing::getVertexOutAddr(buffer_index, offset);
    //     }
    //     // FBO
    //     else if (name == "\%outColor") {
    //       entry.address = VulkanRayTracing::getFBOAddr(offset);
    //     } else {
    //       entry.address = (uint64_t)VulkanRayTracing::gpgpusim_alloc(size);
    //     }
    //   }
    // } else {
    //   assert(0);
    // }

    entry.size = size;
    variable_decleration_table.push_back(entry);

    return entry.address;
  }

  variable_decleration_entry* get_hitAttribute() {
    variable_decleration_entry* hitAttribute = NULL;
    for (int i = 0; i < variable_decleration_table.size(); i++) {
      if (variable_decleration_table[i].type == 8192) {
        assert(variable_decleration_table[i].address != NULL);
        assert(hitAttribute == NULL);  // There should be only 1 hitAttribute
        hitAttribute = &(variable_decleration_table[i]);
      }
    }
    return hitAttribute;
  }

  void set_hitAttribute(float3 barycentric) {
    variable_decleration_entry* hitAttribute = get_hitAttribute();
    float* address;
    if (hitAttribute == NULL) {
      address = (float*)add_variable_decleration_entry(8192, "attribs", "UNDEFINED", 12, 0);
    } else {
      assert(hitAttribute->type == 8192);
      assert(hitAttribute->address != NULL);
      // hitAttribute->name = name;
      address = (float*)(hitAttribute->address);
    }
    // address[0] = barycentric.x;
    // address[1] = barycentric.y;
    // address[2] = barycentric.z;
    gpgpu_context* ctx = GPGPU_Context();
    CUctx_st* context = GPGPUSim_Context(ctx);
    context->get_device()->get_gpgpu()->memcpy_to_gpu(address, &barycentric,
                                                      sizeof(float3));
  }
} Vulkan_RT_thread_data;

#endif /* VULKAN_RT_THREAD_DATA_H */