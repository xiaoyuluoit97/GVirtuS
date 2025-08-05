/**
 * @file   HybirdCommunicator.h
 * @author Xiaoyu Luo <xilu@es.aau.dk>
 * @date   05/08/2025
 */
#pragma once

#include "gvirtus/communicators/Communicator.h"
#include <memory>
#include <string>

constexpr uint8_t FLAG_TCP  = 0x01;
constexpr uint8_t FLAG_RDMA = 0x02;


namespace gvirtus::communicators {

/**
 * HybridCommunicator combines two communicators (e.g., TCP and RDMA) and
 * selects one at runtime based on data size or other heuristics.
 */
class HybridCommunicator : public Communicator {
 private:
  std::shared_ptr<Communicator> _tcp;
  std::shared_ptr<Communicator> _rdma;
  size_t _threshold = 784000; // Default threshold (bytes) to switch protocol

 public:
  HybridCommunicator() = default;

  HybridCommunicator(std::shared_ptr<Communicator> tcp,
                     std::shared_ptr<Communicator> rdma,
                     size_t threshold = 784,000)
      : _tcp(std::move(tcp)), _rdma(std::move(rdma)), _threshold(threshold) {}

  ~HybridCommunicator() override = default;

  void Serve() override;
  const Communicator *const Accept() const override;
  void Connect() override;
  size_t Read(char *buffer, size_t size) override;
  size_t Write(const char *buffer, size_t size) override;
  void Sync() override;
  void Close() override;

  std::string to_string() override { return "hybridcommunicator"; }
};

} // namespace gvirtus::communicators