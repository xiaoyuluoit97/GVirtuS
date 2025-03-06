#include "RdmaRoceCommunicator.h"
#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include "gvirtus/communicators/Endpoint_Rdma_Roce.h"


namespace gvirtus::communicators {

RdmaRoceCommunicator::RdmaRoceCommunicator(char *hostname, char *port) {
    if (port == nullptr || std::string(port).empty()) {
        throw std::runtime_error("RdmaRoceCommunicator: Port not specified...");
    }

    // Resolve the hostname (RoCE uses Ethernet, so ensure the host resolves correctly)
    hostent *ent = gethostbyname(hostname);
    if (ent == NULL) {
        throw std::runtime_error("RdmaRoceCommunicator: Can't resolve hostname \"" + std::string(hostname) + "\"...");
    }

    // Copy the hostname and port to the member variables
    strncpy(this->hostname, hostname, sizeof(this->hostname) - 1);
    strncpy(this->port, port, sizeof(this->port) - 1);

    // Initialize RDMA context structures for RoCE
    memset(&roceCmId, 0, sizeof(roceCmId));
    memset(&roceCmListenId, 0, sizeof(roceCmListenId));
}


// Constructor for RoCE with rdma_cm_id
RdmaRoceCommunicator::RdmaRoceCommunicator(rdma_cm_id *roceCmId) {
    // Store the RDMA communication manager ID
    this->roceCmId = roceCmId;

    // Optionally, register the memory region
    preregisteredMr = ktm_rdma_reg_msgs(roceCmId, preregisteredBuffer, 1024 * 5);
}

// Destructor for RoCE - Cleanup
RdmaRoceCommunicator::~RdmaRoceCommunicator() {
    // Cleanup by calling Close method
    Close();
}


void RdmaRoceCommunicator::Serve() {
    // std::cout << "Starting RoCE server on " << hostname << ":" << port << std::endl;

    // Step 1: Setup the rdma_addrinfo struct for RoCE (RDMA over Converged Ethernet)
    rdma_addrinfo hints;
    memset(&hints, 0, sizeof(hints));

    // Set the address family to AF_INET (IPv4) because RoCE typically uses Ethernet with IPv4
    hints.ai_family = AF_INET;
    // Set the port space to RDMA_PS_TCP for RoCE (not RDMA_PS_IB)
    hints.ai_port_space = RDMA_PS_TCP;
    // RAI_PASSIVE indicates we are the passive side of the connection
    hints.ai_flags = RAI_PASSIVE;

    rdma_addrinfo *rdmaAddrinfo = nullptr;

    // Step 2: Resolve the destination host and port using the rdma_getaddrinfo function
    int ret = rdma_getaddrinfo(this->hostname, this->port, &hints, &rdmaAddrinfo);
    if (ret != 0) {
        std::cerr << "rdma_getaddrinfo failed: " << gai_strerror(ret) << std::endl;
        throw std::runtime_error("rdma_getaddrinfo() failed while resolving hostname and port.");
    }

    // Step 3: Initialize Queue Pair (QP) attributes
    ibv_qp_init_attr qpInitAttr;
    memset(&qpInitAttr, 0, sizeof(qpInitAttr));
    qpInitAttr.cap.max_send_wr = 10;  // Maximum send work requests
    qpInitAttr.cap.max_recv_wr = 10;  // Maximum receive work requests
    qpInitAttr.cap.max_send_sge = 10; // Maximum send scatter-gather elements
    qpInitAttr.cap.max_recv_sge = 10; // Maximum receive scatter-gather elements
    qpInitAttr.sq_sig_all = 1;        // Sign all send completions
    qpInitAttr.qp_type = IBV_QPT_RC;  // Reliable Connection (RC) Queue Pair type

    // Step 4: Create the endpoint for RDMA communication
    ktm_rdma_create_ep(&roceCmListenId, rdmaAddrinfo, nullptr, &qpInitAttr);

    // Step 5: Free the address info structure as it's no longer needed
    rdma_freeaddrinfo(rdmaAddrinfo);

    // Step 6: Start listening for incoming connections
    ktm_rdma_listen(roceCmListenId, BACKLOG);

}




const Communicator *const RdmaRoceCommunicator::Accept() const {
    // 1. Get the incoming connection request
    rdma_cm_id *clientRdmaCmId;
    ktm_rdma_get_request(roceCmListenId, &clientRdmaCmId);  // Receives connection request

    // 2. Accept the incoming request
    ktm_rdma_accept(clientRdmaCmId, nullptr);  // Accepts the connection request

    // 3. Modify Queue Pair attributes to set minimum RNR timer (to avoid connection timeouts)
    auto *ibvQpAttr = static_cast<ibv_qp_attr *>(malloc(sizeof(ibv_qp_attr)));
    ibvQpAttr->min_rnr_timer = 1;  // Set minimal RNR timer for faster connection setup
    if (ibv_modify_qp(clientRdmaCmId->qp, ibvQpAttr, IBV_QP_MIN_RNR_TIMER)) {
        std::cerr << "Failed to modify Queue Pair attributes: " << strerror(errno) << std::endl;
    }

    // 4. Return a new communicator instance to handle the established connection
    return new RdmaRoceCommunicator(clientRdmaCmId);  // Returns a new communicator with the accepted connection
}


void RdmaRoceCommunicator::Connect() {
    // std::cout << "Connecting to RoCE endpoint at " << hostname << ":" << port << std::endl;

    // Step 1: Setup the rdma_addrinfo struct for RoCE (RDMA over Converged Ethernet)
    rdma_addrinfo hints;
    memset(&hints, 0, sizeof(hints));

    // Set the address family to AF_INET (IPv4) because RoCE typically uses Ethernet with IPv4
    hints.ai_family = AF_INET;  // Use IPv4 address family (Ethernet typically uses IPv4)
    // Set the port space to RDMA_PS_TCP for RoCE (not RDMA_PS_IB)
    hints.ai_port_space = RDMA_PS_TCP;  
    // No RAI_PASSIVE flag for the client-side
    hints.ai_flags = 0;  // Active side of the connection (no passive flag)

    rdma_addrinfo *rdmaAddrinfo = nullptr;

    // Step 2: Resolve the destination host and port using the rdma_getaddrinfo function
    int ret = rdma_getaddrinfo(this->hostname, this->port, &hints, &rdmaAddrinfo);
    if (ret != 0) {
        std::cerr << "rdma_getaddrinfo failed: " << gai_strerror(ret) << std::endl;
        throw std::runtime_error("rdma_getaddrinfo() failed while resolving hostname and port.");
    }

    // Step 3: Initialize Queue Pair (QP) attributes for client
    ibv_qp_init_attr qpInitAttr;
    memset(&qpInitAttr, 0, sizeof(qpInitAttr));
    qpInitAttr.cap.max_send_wr = 10;  // Maximum send work requests
    qpInitAttr.cap.max_recv_wr = 10;  // Maximum receive work requests
    qpInitAttr.cap.max_send_sge = 10; // Maximum send scatter-gather elements
    qpInitAttr.cap.max_recv_sge = 10; // Maximum receive scatter-gather elements
    qpInitAttr.sq_sig_all = 1;        // Sign all send completions
    qpInitAttr.qp_type = IBV_QPT_RC;  // Reliable Connection (RC) Queue Pair type

    // Step 4: Create the RDMA endpoint for the client (this initiates the connection)
    ktm_rdma_create_ep(&roceCmId, rdmaAddrinfo, nullptr, &qpInitAttr);

    // Step 5: Free the address info structure as it's no longer needed
    rdma_freeaddrinfo(rdmaAddrinfo);

    // Step 6: Perform the connection to the remote server (client-side)
    ktm_rdma_connect(roceCmId, nullptr);

    // Step 7: Modify the Queue Pair (QP) attributes (set minimum retry timer)
    auto *ibvQpAttr = static_cast<ibv_qp_attr *>(malloc(sizeof(ibv_qp_attr)));
    ibvQpAttr->min_rnr_timer = 1;  // Set to the lowest retry timer value
    if (ibv_modify_qp(roceCmId->qp, ibvQpAttr, IBV_QP_MIN_RNR_TIMER)) {
        std::cerr << "ibv_modify_qp() failed: " << strerror(errno) << std::endl;
        throw std::runtime_error("Failed to modify Queue Pair (QP) attributes.");
    }

    // Step 8: Perform memory region registration (buffer preregistration)
    preregisteredMr = ktm_rdma_reg_msgs(roceCmId, preregisteredBuffer, 1024 * 5);
}



size_t RdmaRoceCommunicator::Read(char *buffer, size_t size) {

    // Check if the size of the data to be read is smaller or equal to the preregistered buffer
    if (size <= 1024 * 5) {
        // If the output buffer is smaller than or equal to the preregistered buffer size
        // Just use the preregistered buffer for reading the data
        ktm_rdma_post_recv(roceCmId, nullptr, preregisteredBuffer, size, preregisteredMr);
    }
    else {
        // If the output buffer is larger than the preregistered buffer
        // Register a new memory region for the new buffer
        memoryRegion = ktm_rdma_reg_msgs(roceCmId, buffer, size);
        // Post the receive operation into the newly registered memory region
        ktm_rdma_post_recv(roceCmId, nullptr, buffer, size, memoryRegion);
    }

    // Poll the completion queue for the completion of the receive operation
    int num_comp;
    do num_comp = ibv_poll_cq(roceCmId->recv_cq, 1, &workCompletion); while (num_comp == 0);
    if (num_comp < 0) {
        throw std::runtime_error("ibv_poll_cq() failed during read operation");
    }
    if (workCompletion.status != IBV_WC_SUCCESS) {
        throw std::runtime_error("Failed status in ibv_poll_cq: " + std::string(ibv_wc_status_str(workCompletion.status)));
    }

    // If the buffer is smaller than or equal to the preregistered buffer, copy the data from the preregistered buffer
    if (size <= 1024 * 5) {
        memcpy(buffer, preregisteredBuffer, size);
    }

    return size;
}

size_t RdmaRoceCommunicator::Write(const char *buffer, size_t size) {
    // std::cout << "Writing " << size << " bytes over RoCE" << std::endl;

    char *actualBuffer = nullptr;

    // If the data size is smaller than the pre-registered buffer, use the pre-registered buffer
    if (size < 1024 * 5) {
        memcpy(preregisteredBuffer, buffer, size);
        // Post the send request to the RoCE endpoint using the pre-registered buffer
        ktm_rdma_post_send(roceCmId, nullptr, preregisteredBuffer, size, preregisteredMr, IBV_SEND_SIGNALED);
    } else {
        // Allocate memory for the large data and register it for RoCE communication
        actualBuffer = (char *)malloc(size);
        memcpy(actualBuffer, buffer, size);
        memoryRegion = ktm_rdma_reg_msgs(roceCmId, actualBuffer, size);
        
        // Post the send request to the RoCE endpoint using the allocated memory
        ktm_rdma_post_send(roceCmId, nullptr, actualBuffer, size, memoryRegion, IBV_SEND_SIGNALED);
    }

    int num_comp;
    // Poll the send completion queue for work completion
    do {
        num_comp = ibv_poll_cq(roceCmId->send_cq, 1, &workCompletion);
    } while (num_comp == 0);  // Continue polling if no completion yet

    if (num_comp < 0) {
        throw std::runtime_error("ibv_poll_cq() failed during write operation");
    }

    if (workCompletion.status != IBV_WC_SUCCESS) {
        throw std::runtime_error("Failed to write data: " + std::string(ibv_wc_status_str(workCompletion.status)));
    }

    // If the size is larger than the pre-registered buffer, free the allocated memory
    if (size > 1024 * 5) {
        free(actualBuffer);
    }

    return size; // Return the number of bytes written
}


void RdmaRoceCommunicator::Sync() { }


void RdmaRoceCommunicator::Close() {
    // std::cout << "Closing RoCE connection..." << std::endl;

    // Ensure that any active RDMA connections are properly disconnected before cleanup
    if (roceCmId) {
        rdma_disconnect(roceCmId);  // Disconnect the RDMA connection
        rdma_destroy_id(roceCmId);  // Destroy the communication manager ID
        // std::cout << "RoCE connection closed successfully." << std::endl;
    } else {
        std::cout << "No active RoCE connection found to close." << std::endl;
    }

    // Deregister any memory regions associated with the RDMA communication
    if (preregisteredMr) {
        ibv_dereg_mr(preregisteredMr);  // Deregister the memory region
        preregisteredMr = nullptr;      // Nullify the pointer to avoid further access
    }

    if (memoryRegion) {
        ibv_dereg_mr(memoryRegion);  // Deregister the memory region
        memoryRegion = nullptr;      // Nullify the pointer to avoid further access
    }

}


extern "C" std::shared_ptr<gvirtus::communicators::Communicator> create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> endpoint) {
    // Extract hostname and port from the endpoint
    std::string hostname = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma_Roce>(endpoint)->address();
    std::string port = std::to_string(std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma_Roce>(endpoint)->port());

    // Create and return a new RdmaRoceCommunicator object using the extracted values
    return std::make_shared<gvirtus::communicators::RdmaRoceCommunicator>(hostname.data(), port.data());
}


} // namespace gvirtus::communicators

