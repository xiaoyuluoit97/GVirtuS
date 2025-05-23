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

#include <iostream>
#include <cstdio>
#include <string>

#include "CudnnFrontend.h"
#include "HandleManager.h"   // 你放 HandleManager 类的头文件路径
#include <chrono>
#include <unistd.h>
int g_session_id = -1;

int GenerateSessionId() {
    using namespace std::chrono;
    auto now = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    return static_cast<int>((now ^ getpid()) & 0x7FFFFFFF);
}

void EnsureSessionInitialized() {
    if (g_session_id < 0)
        g_session_id = GenerateSessionId();
}


/**
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreate        (cudnnHandle_t *handle) {
    CudnnFrontend::Prepare();
    //CudnnFrontend::AddHostPointerForArguments<cudnnHandle_t>(handle);
    CudnnFrontend::Execute("cudnnCreate");
    if(CudnnFrontend::Success())
        *handle = *(CudnnFrontend::GetOutputHostPointer<cudnnHandle_t>());
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroy       (cudnnHandle_t handle) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::Execute("cudnnDestroy");
    return CudnnFrontend::GetExitCode();
}
**/

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle) {
    EnsureSessionInitialized();

    int handle_id = HandleManager::Instance().GenerateHandleId();
    *handle = reinterpret_cast<cudnnHandle_t>((uintptr_t)handle_id);

    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<int>(g_session_id);
    CudnnFrontend::AddVariableForArguments<int>(handle_id);
    CudnnFrontend::Execute("cudnnCreate");

    if (CudnnFrontend::Success()) {
        HandleManager::Instance().Register(handle_id, *handle);
    }

    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
    EnsureSessionInitialized();

    int handle_id = static_cast<int>(reinterpret_cast<uintptr_t>(handle));

    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<int>(g_session_id);
    CudnnFrontend::AddVariableForArguments<int>(handle_id);
    CudnnFrontend::Execute("cudnnDestroy");

    if (CudnnFrontend::Success()) {
        HandleManager::Instance().Unregister(handle_id);
    }

    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetStream     (cudnnHandle_t handle, cudaStream_t streamId) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::AddDevicePointerForArguments(streamId);
    CudnnFrontend::Execute("cudnnSetStream");
    return CudnnFrontend::GetExitCode();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetStream     (cudnnHandle_t handle, cudaStream_t *streamId) {
    CudnnFrontend::Prepare();
    CudnnFrontend::AddDevicePointerForArguments(handle);
    CudnnFrontend::Execute("cudnnGetStream");
    return CudnnFrontend::GetExitCode();    
}
