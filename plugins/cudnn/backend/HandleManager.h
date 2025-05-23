#pragma once
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <cudnn.h>
class HandleManager {
public:
    static HandleManager& Instance() {
        static HandleManager instance;
        return instance;
    }

    // 生成唯一 handle ID
    int GenerateHandleId() {
        std::lock_guard<std::mutex> lock(mutex_);
        return next_id_++;
    }

    // 注册 handle（前端 Create 后调用）
    void Register(int handle_id, cudnnHandle_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_[handle_id] = handle;
    }

    // 获取 handle
    cudnnHandle_t Get(int handle_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pool_.find(handle_id);
        return (it != pool_.end()) ? it->second : nullptr;
    }

    // 注销 handle（前端 Destroy 后调用）
    void Unregister(int handle_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.erase(handle_id);
    }

    // 清空所有 handle（可选）
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear();
    }

private:
    HandleManager() : next_id_(100000) {}
    std::unordered_map<int, cudnnHandle_t> pool_;
    int next_id_;
    std::mutex mutex_;
};
