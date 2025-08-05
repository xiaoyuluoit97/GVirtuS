#include <iostream>
#include <memory>
#include <string>
#include <cstring>
#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
#include <netdb.h>
#else
#include <WinSock2.h>
static bool initialized = false;
#endif
#include "gvirtus/communicators/Communicator.h"
#include "gvirtus/communicators/TcpCommunicator.h"
#include "gvirtus/communicators/RdmaCommunicator.h"
#include "gvirtus/communicators/Endpoint.h"
#include "gvirtus/communicators/Endpoint_Tcp.h"
#include "gvirtus/communicators/Endpoint_Rdma.h"
#include "gvirtus/communicators/Endpoint_Hybrid.h"  // new endpoint for hybrid communicator

#include "gvirtus/communicators/HybridCommunicator.h"

using gvirtus::communicators::HybridCommunicator;
using gvirtus::communicators::TcpCommunicator;
using gvirtus::communicators::RdmaCommunicator;

HybridCommunicator::HybridCommunicator(const std::string &hostname,
                                       const std::string &tcpPort,
                                       const std::string &rdmaPort,
                                       bool isRoce,
                                       size_t threshold)
    : _threshold(threshold) {
#ifdef DEBUG
    std::cout << "HybridCommunicator initializing with:" << std::endl;
    std::cout << "  Hostname: " << hostname << std::endl;
    std::cout << "  TCP Port: " << tcpPort << std::endl;
    std::cout << "  RDMA Port: " << rdmaPort << std::endl;
    std::cout << "  Threshold: " << _threshold << std::endl;
#endif

    if (tcpPort.empty() || rdmaPort.empty()) {
        throw std::runtime_error("HybridCommunicator: TCP or RDMA port not specified");
    }

    struct hostent *ent = gethostbyname(hostname.c_str());
    if (ent == nullptr) {
        std::ostringstream oss;
        oss << "HybridCommunicator: Cannot resolve hostname '" << hostname << "'";
        throw std::runtime_error(oss.str());
    }

    // Construct TCP communicator using hostname and port
    _tcp = std::make_shared<TcpCommunicator>(hostname.c_str(), static_cast<short>(std::stoi(tcpPort)));

    // Construct RDMA communicator using hostname and port
    _rdma = std::make_shared<RdmaCommunicator>(hostname, rdmaPort, isRoce);

#ifdef DEBUG
    std::cout << "HybridCommunicator successfully constructed." << std::endl;
#endif
}

// Constructor used on the server side when accepted connection objects are ready
HybridCommunicator::HybridCommunicator(std::shared_ptr<TcpCommunicator> tcp,
                                       std::shared_ptr<RdmaCommunicator> rdma,
                                       size_t threshold)
    : _tcp(std::move(tcp)), _rdma(std::move(rdma)), _threshold(threshold) {
#ifdef DEBUG
    std::cout << "HybridCommunicator accepted with existing TCP and RDMA communicators." << std::endl;
#endif
}

HybridCommunicator::~HybridCommunicator() {
#ifdef DEBUG
    std::cout << "Called ~HybridCommunicator()" << std::endl;
#endif
    if (_tcp) _tcp->Close();
    if (_rdma) _rdma->Close();
}

void HybridCommunicator::Serve() {
#ifdef DEBUG
    std::cout << "HybridCommunicator::Serve() called" << std::endl;
#endif
    if (_tcp) _tcp->Serve();
    if (_rdma) _rdma->Serve();
#ifdef DEBUG
    std::cout << "HybridCommunicator::Serve() completed" << std::endl;
#endif
}

const Communicator *const HybridCommunicator::Accept() const {
#ifdef DEBUG
    std::cout << "HybridCommunicator::Accept() called" << std::endl;
#endif
    const Communicator *tcpComm = _tcp ? _tcp->Accept() : nullptr;
    const Communicator *rdmaComm = _rdma ? _rdma->Accept() : nullptr;

    if (!tcpComm || !rdmaComm) {
#ifdef DEBUG
        std::cerr << "HybridCommunicator::Accept() failed: missing one of the connections." << std::endl;
#endif
        return nullptr;
    }

    auto tcpPtr = std::shared_ptr<TcpCommunicator>(const_cast<TcpCommunicator *>(dynamic_cast<const TcpCommunicator *>(tcpComm)));
    auto rdmaPtr = std::shared_ptr<RdmaCommunicator>(const_cast<RdmaCommunicator *>(dynamic_cast<const RdmaCommunicator *>(rdmaComm)));

    return new HybridCommunicator(tcpPtr, rdmaPtr, _threshold);
}
void HybridCommunicator::Connect() {
#ifdef DEBUG
    std::cout << "HybridCommunicator::Connect() called" << std::endl;
#endif
    if (_tcp) _tcp->Connect();
    if (_rdma) _rdma->Connect();
#ifdef DEBUG
    std::cout << "HybridCommunicator::Connect() completed" << std::endl;
#endif
}

void HybridCommunicator::Close() {
#ifdef DEBUG
    std::cout << "HybridCommunicator::Close() called" << std::endl;
#endif
    if (_tcp) _tcp->Close();
    if (_rdma) _rdma->Close();
}

size_t HybridCommunicator::Write(const char *buffer, size_t size) {
#ifdef DEBUG
    std::cout << "HybridCommunicator::Write() called with size: " << size << std::endl;
#endif

    if (size < _threshold) {
#ifdef DEBUG
        std::cout << "HybridCommunicator::Write() using TCP" << std::endl;
#endif
        // 1 byte flag + payload
        std::vector<char> packet(size + 1);
        packet[0] = FLAG_TCP;  // Protocol flag
        std::memcpy(&packet[1], buffer, size);
        _tcp->Write(packet.data(), size + 1);
        return size;
    } else {
#ifdef DEBUG
        std::cout << "HybridCommunicator::Write() using RDMA" << std::endl;
#endif
        uint8_t flag = FLAG_RDMA;
        _tcp->Write(reinterpret_cast<const char*>(&flag), 1);  // Only protocol flag
        _rdma->Write(buffer, size);
        return size;
    }
}

size_t HybridCommunicator::Read(char *buffer, size_t size) {
#ifdef DEBUG
    std::cout << "HybridCommunicator::Read() called, expecting " << size << " bytes" << std::endl;
#endif

    // Step 1: read protocol flag from TCP
    uint8_t flag;
    size_t flag_read = _tcp->Read(reinterpret_cast<char*>(&flag), 1);

    if (flag_read != 1) {
        throw std::runtime_error("HybridCommunicator::Read(): Failed to read protocol flag from TCP");
    }

    // Step 2: judge protocol based on flag
    if (flag == FLAG_TCP) {
#ifdef DEBUG
        std::cout << "HybridCommunicator::Read() using TCP" << std::endl;
#endif
        return _tcp->Read(buffer, size);
    } else if (flag == FLAG_RDMA) {
#ifdef DEBUG
        std::cout << "HybridCommunicator::Read() using RDMA" << std::endl;
#endif
        return _rdma->Read(buffer, size);
    } else {
        throw std::runtime_error("HybridCommunicator::Read(): Unknown protocol flag received");
    }
}

void HybridCommunicator::Sync() {
#ifdef DEBUG
    std::cout << "HybridCommunicator::Sync() called" << std::endl;
#endif
    if (_tcp) _tcp->Sync();
    if (_rdma) _rdma->Sync();
}

extern "C" std::shared_ptr<gvirtus::communicators::HybridCommunicator> create_communicator(
    std::shared_ptr<gvirtus::communicators::Endpoint> end) {
    
    using namespace gvirtus::communicators;

    auto hybrid = std::dynamic_pointer_cast<Endpoint_Hybrid>(end);
    if (!hybrid) {
        throw std::runtime_error("create_communicator: invalid endpoint type for hybrid");
    }

    const std::string &host = hybrid->address();

    // TCP communicator
    short tcpPort = hybrid->tcp_port();
    std::shared_ptr<TcpCommunicator> tcp =
        std::make_shared<TcpCommunicator>(host.c_str(), tcpPort);

    // RDMA communicator
    std::string rdmaPortStr = std::to_string(hybrid->rdma_port());
    bool isRoce = hybrid->rdma_suite() == "roce-rdma";
    std::shared_ptr<RdmaCommunicator> rdma =
        std::make_shared<RdmaCommunicator>(host, rdmaPortStr, isRoce);

    return std::make_shared<HybridCommunicator>(tcp, rdma, /* threshold = */ 1024 * 5);
}