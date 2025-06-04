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
#ifndef CUDNNFRONTEND_H
#define	CUDNNFRONTEND_H

#include <map>
#include <set>
#include <stack>
#include <list>
#include <iostream>

#include <cudnn.h>

#include <gvirtus/frontend/Frontend.h>

using gvirtus::communicators::Buffer;
using gvirtus::frontend::Frontend;

typedef struct __configureFunction {
      gvirtus::common::funcs __f;
      gvirtus::communicators::Buffer* buffer;
} configureFunction;

class CudnnFrontend {
public:
    static inline void Execute(const char * routine, const Buffer * input_buffer = NULL) {
        Frontend::GetFrontend()->Execute(routine, input_buffer);
    }

    /**
     * Prepares the Frontend for the execution. This method _must_ be called
     * before any requests of execution or any method for adding parameters for
     * the next execution.
     */
    static inline void Prepare() {
        Frontend::GetFrontend()->Prepare();
    }

    static inline Buffer *GetLaunchBuffer() {
        return Frontend::GetFrontend()->GetInputBuffer();
    }

    /**
     * Adds a scalar variabile as an input parameter for the next execution
     * request.
     *
     * @param var the variable to add as a parameter.

    template <class T> static inline void AddVariableForArguments(T var) {
        Frontend::GetFrontend()->GetInputBuffer()->Add(var);
    }
*/
//
template <typename T>
static inline typename std::enable_if<
    !std::is_void<T>::value && !std::is_function<T>::value>::type
AddVariableForArguments(T val) {
    Frontend::GetFrontend()->GetInputBuffer()->Add(&val, 1);
}

//
static inline void AddVariableForArguments(void* ptr) {
    long long int val = reinterpret_cast<long long int>(ptr);
    Frontend::GetFrontend()->GetInputBuffer()->Add(&val, 1);
}

//
static inline void AddVariableForArguments(const void* ptr) {
    long long int val = reinterpret_cast<long long int>(ptr);
    Frontend::GetFrontend()->GetInputBuffer()->Add(&val, 1);
}

    /**
     * Adds a string (array of char(s)) as an input parameter for the next
     * execution request.
     *
     * @param s the string to add as a parameter.
     */
    static inline void AddStringForArguments(const char *s) {
        Frontend::GetFrontend()->GetInputBuffer()->AddString(s);
    }

    /**
     * Adds, marshalling it, an host pointer as an input parameter for the next
     * execution request.
     * The optional parameter n is usefull when adding an array: with n is
     * possible to specify the length of the array in terms of elements.
     *
     * @param ptr the pointer to add as a parameter.
     * @param n the length of the array, if ptr is an array.

    template <class T>static inline void AddHostPointerForArguments(T *ptr, size_t n = 1) {
        Frontend::GetFrontend()->GetInputBuffer()->Add(ptr, n);
    }
    */
// ✅ 通用模板：禁止 void
template <typename T>
static inline typename std::enable_if<!std::is_void<T>::value>::type
AddHostPointerForArguments(T* ptr, size_t n = 1) {
    Frontend::GetFrontend()->GetInputBuffer()->Add(ptr, n);
}

// ✅ 明确支持 void*（默认按 sizeof(void*) 传输）
static inline void AddHostPointerForArguments(void* ptr) {
    auto byte_ptr = reinterpret_cast<const uint8_t*>(&ptr);
    Frontend::GetFrontend()->GetInputBuffer()->Add(byte_ptr, sizeof(void*));
}

// ✅ 明确支持 const void*（同上）
static inline void AddHostPointerForArguments(const void* ptr) {
    auto byte_ptr = reinterpret_cast<const uint8_t*>(&ptr);
    Frontend::GetFrontend()->GetInputBuffer()->Add(byte_ptr, sizeof(void*));
}

// ✅ 可选：手动指定大小（用于原始 buffer 拷贝）
static inline void AddHostPointerForArguments(const void* ptr, size_t bytes) {
    auto byte_ptr = reinterpret_cast<const uint8_t*>(ptr);
    Frontend::GetFrontend()->GetInputBuffer()->Add(byte_ptr, bytes);
}


    /**
     * Adds a device pointer as an input parameter for the next execution
     * request.
     *
     * @param ptr the pointer to add as a parameter.
     */
    static inline void AddDevicePointerForArguments(const void *ptr) {
        Frontend::GetFrontend()->GetInputBuffer()->Add((uint64_t) ptr);
    }

    /**
     * Adds a symbol, a named variable, as an input parameter for the next
     * execution request.
     *
     * @param symbol the symbol to add as a parameter.
     */
    static inline void AddSymbolForArguments(const char *symbol) {
        /* TODO: implement AddSymbolForArguments
         * AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
         * AddStringForArguments(symbol);
         * */
    }

    static inline cudnnStatus_t GetExitCode() {
        return (cudnnStatus_t) Frontend::GetFrontend()->GetExitCode();
    }

    static inline bool Success() {
        return Frontend::GetFrontend()->Success(cudaSuccess);
    }

    template <class T> static inline T GetOutputVariable() {
        return Frontend::GetFrontend()->GetOutputBuffer()->Get<T> ();
    }

    /**
     * Retrives an host pointer from the output parameters of the last execution
     * request.
     * The optional parameter n is usefull when retriving an array: with n is
     * possible to specify the length of the array in terms of elements.
     *
     * @param n the length of the array.
     *
     * @return the pointer from the output parameters.
     */
    template <class T>static inline T * GetOutputHostPointer(size_t n = 1) {
        return Frontend::GetFrontend()->GetOutputBuffer()->Assign<T> (n);
    }

    /**
     * Retrives a device pointer from the output parameters of the last
     * execution request.
     *
     * @return the pointer to the device memory.
     */
    static inline void * GetOutputDevicePointer() {
        return (void *) Frontend::GetFrontend()->GetOutputBuffer()->Get<uint64_t>();
    }

    /**
     * Retrives a string, array of chars, from the output parameters of the last
     * execution request.
     *
     * @return the string from the output parameters.
     */
    static inline char * GetOutputString() {
        return Frontend::GetFrontend()->GetOutputBuffer()->AssignString();
    }
    CudnnFrontend();
    static void * handler;
};
#endif	/* CUDNNFRONTEND_H */
