/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

#include "CudaRtHandler.h"

// extern const textureReference *getTexture(const textureReference *handler);

CUDA_ROUTINE_HANDLER(BindTexture) {
  try {
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    size_t *offset = out->Delegate<size_t>();
    *offset = *(input_buffer->Assign<size_t>());
    char *texrefHandler = input_buffer->AssignString();
    cudaTextureObject_t guestTexref = *(input_buffer->Assign<cudaTextureObject_t>());
    cudaTextureObject_t* texref = pThis->GetTexture(texrefHandler);
    *texref = guestTexref; 
    
    void *devPtr = input_buffer->GetFromMarshal<void *>();
    cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();
    size_t size = input_buffer->Get<size_t>();

    // Allocate and copy to CUDA array
    cudaArray_t devArray;
    if (cudaMallocArray(&devArray, desc, size, 1) != cudaSuccess ||
        cudaMemcpyToArray(devArray, 0, 0, devPtr, size, cudaMemcpyDeviceToDevice) != cudaSuccess) {
      return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }

    // Create texture object with resource and texture descriptors
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj;
    cudaError_t exit_code = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    return std::make_shared<Result>(exit_code, out);
  } catch (const std::string& e) {
    std::cerr << e << std::endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030

CUDA_ROUTINE_HANDLER(BindTexture2D) {
  try {
    auto out = std::make_shared<Buffer>();
    size_t *offset = out->Delegate<size_t>();
    size_t *offsetIn = input_buffer->Assign<size_t>();
    *offset = offsetIn ? *offsetIn : 0;

    char *texrefHandler = input_buffer->AssignString();
    cudaTextureObject_t *guestTexref = input_buffer->Assign<cudaTextureObject_t>();
    cudaTextureObject_t *texref = pThis->GetTexture(texrefHandler);
    *texref = *guestTexref;

    void *devPtr = reinterpret_cast<void *>(input_buffer->Get<pointer_t>());
    auto desc = input_buffer->Assign<cudaChannelFormatDesc>();
    size_t width = input_buffer->Get<size_t>();
    size_t height = input_buffer->Get<size_t>();
    size_t pitch = input_buffer->Get<size_t>();

    cudaArray_t devArray;
    cudaMallocArray(&devArray, desc, width, height, cudaArrayDefault);
    cudaMemcpy2DToArray(devArray, 0, 0, devPtr, pitch, width * (desc->x + desc->y + desc->z + desc->w) / 8, height, cudaMemcpyDeviceToDevice);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = devArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj;
    cudaError_t exit_code = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    *texref = texObj;

    return std::make_shared<Result>(exit_code, out);
  } catch (const std::string &e) {
    std::cerr << e << std::endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}
#endif

CUDA_ROUTINE_HANDLER(BindTextureToArray) {
  try {
    auto texrefHandler = input_buffer->AssignString();
    cudaTextureObject_t guestTexref = *input_buffer->Assign<cudaTextureObject_t>();
    cudaTextureObject_t *texref = pThis->GetTexture(texrefHandler);
    *texref = guestTexref;

    auto array = reinterpret_cast<cudaArray_t>(input_buffer->Get<pointer_t>());
    auto desc = input_buffer->Assign<cudaChannelFormatDesc>();

    // Prepare resource and texture descriptors
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    // Create the texture object
    cudaTextureObject_t texObj;
    cudaError_t exit_code = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    *texref = texObj;

    return std::make_shared<Result>(exit_code);
  } catch (const std::string& e) {
    std::cerr << e << std::endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}


CUDA_ROUTINE_HANDLER(GetChannelDesc) {
  try {
    cudaChannelFormatDesc *guestDesc =
        input_buffer->Assign<cudaChannelFormatDesc>();
    cudaArray *array = (cudaArray *)input_buffer->GetFromMarshal<cudaArray *>();
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    cudaChannelFormatDesc *desc = out->Delegate<cudaChannelFormatDesc>();
    memmove(desc, guestDesc, sizeof(cudaChannelFormatDesc));
    cudaError_t exit_code = cudaGetChannelDesc(desc, array);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(GetTextureAlignmentOffset) {
  try {
    auto out = std::make_shared<Buffer>();
    size_t *offset = out->Delegate<size_t>();

    *offset = *(input_buffer->Assign<size_t>());

    char *texrefHandler = input_buffer->AssignString();
    cudaTextureObject_t* guestTexref = input_buffer->Assign<cudaTextureObject_t>();
    cudaTextureObject_t *texref = pThis->GetTexture(texrefHandler);

    *texref = *guestTexref;

    cudaError_t exit_code = cudaSuccess;      // Alignment offset retrieval is deprecated with cudaTextureObject_t
    return std::make_shared<Result>(exit_code, out);
  } catch (const std::string &e) {
    std::cerr << e << std::endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}



CUDA_ROUTINE_HANDLER(GetTextureReference) {
  try {
    auto symbol_handler = input_buffer->AssignString();
    auto symbol = input_buffer->AssignString();
    auto our_symbol = const_cast<char *>(pThis->GetVar(symbol_handler));
    if (our_symbol != nullptr) symbol = our_symbol;

    // Texture references are deprecated, so just return a placeholder or use your own registry
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    cudaTextureObject_t* texref = pThis->GetTexture(symbol);

    if (texref && *texref != 0)
      out->AddString(pThis->GetTextureHandler(texref));
    else
      out->AddString("0x0");

    return std::make_shared<Result>(cudaSuccess, out);
  } catch (const std::string& e) {
    std::cerr << e << std::endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}


CUDA_ROUTINE_HANDLER(UnbindTexture) {
  try {
    char *texrefHandler = input_buffer->AssignString();
    cudaTextureObject_t *texref = pThis->GetTexture(texrefHandler);
    cudaError_t exit_code = cudaDestroyTextureObject(*texref);
        return std::make_shared<Result>(exit_code);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(CreateTextureObject) {
  cudaTextureObject_t tex = 0;
  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
  try {
    cudaResourceDesc *pResDesc = input_buffer->Assign<cudaResourceDesc>();
    cudaTextureDesc *pTexDesc =
        CudaUtil::UnmarshalTextureDesc(input_buffer.get());
    cudaResourceViewDesc *pResViewDesc =
        input_buffer->Assign<cudaResourceViewDesc>();

    cudaError_t exit_code =
        cudaCreateTextureObject(&tex, pResDesc, pTexDesc, pResViewDesc);

    out->Add<cudaTextureObject_t>(tex);
    return std::make_shared<Result>(exit_code, out);

  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorUnknown);
  }
}
