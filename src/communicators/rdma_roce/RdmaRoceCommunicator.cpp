#include "RdmaRoceCommunicator.h"
#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include "gvirtus/communicators/Endpoint_Rdma_Roce.h"


namespace gvirtus::communicators {

RdmaRoceCommunicator::RdmaRoceCommunicator(char *hostname, char *port) {
#ifdef DEBUG
    std::cout << "Called " << "RdmaRoceCommunicator(char *hostname, char *port)" << std::endl;
#endif

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

#ifdef DEBUG
    std::cout << "RdmaRoceCommunicator(" << this->hostname << ", " << this->port << ")" << std::endl;
#endif

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
#ifdef DEBUG
    std::cout << "Called " << "Serve()" << std::endl;
#endif

    // Setup address info for RDMA over Converged Ethernet (RoCE)
    rdma_addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_port_space = RDMA_PS_TCP;  // Use TCP port space for RoCE (not RDMA_PS_IB)
    hints.ai_flags = RAI_PASSIVE;  // Passive mode to accept incoming connections

    rdma_addrinfo *rdmaAddrinfo;

    // Get address info based on hostname and port for RoCE communication
    ktm_rdma_getaddrinfo(this->hostname, this->port, &hints, &rdmaAddrinfo);

    // Create communication manager ID with queue pair attributes
    ibv_qp_init_attr qpInitAttr;
    memset(&qpInitAttr, 0, sizeof(qpInitAttr));

    // Set up the queue pair attributes for RoCE
    qpInitAttr.cap.max_send_wr = 10;
    qpInitAttr.cap.max_recv_wr = 10;
    qpInitAttr.cap.max_send_sge = 10;
    qpInitAttr.cap.max_recv_sge = 10;
    qpInitAttr.sq_sig_all = 1;  // Sign all send completions
    qpInitAttr.qp_type = IBV_QPT_RC;  // Reliable Connection (RC) queue pair type

    // Create endpoint for RDMA RoCE communication
    ktm_rdma_create_ep(&roceCmListenId, rdmaAddrinfo, NULL, &qpInitAttr);

    // Free the address info once it's no longer needed
    rdma_freeaddrinfo(rdmaAddrinfo);

    // Listen for incoming connections on the RoCE endpoint
    ktm_rdma_listen(roceCmListenId, BACKLOG);

    std::cout << "RdmaRoceCommunicator: Listening for incoming RoCE connections on port " << this->port << std::endl;
}



const gvirtus::communicators::Communicator *const RdmaRoceCommunicator::Accept() const {
    std::cout << "Accepting incoming RoCE connection..." << std::endl;

    rdma_cm_id *clientRdmaCmId;

    // Get the request for the incoming RoCE connection
    ktm_rdma_get_request(roceCmListenId, &clientRdmaCmId);

    // Accept the RoCE connection
    ktm_rdma_accept(clientRdmaCmId, nullptr);

    // Set the queue pair attributes for RoCE (using IBV_QP_MIN_RNR_TIMER here)
    auto *ibvQpAttr = static_cast<ibv_qp_attr *>(malloc(sizeof(ibv_qp_attr)));
    ibvQpAttr->min_rnr_timer = 1;  // Set RoCE-specific parameters

    // Modify the queue pair with the new attributes
    if (ibv_modify_qp(clientRdmaCmId->qp, ibvQpAttr, IBV_QP_MIN_RNR_TIMER)) {
        std::cerr << "ibv_modify_qp() failed: " << strerror(errno) << std::endl;
        return nullptr;
    }

    // Return a new RdmaRoceCommunicator with the accepted client connection ID
    return new RdmaRoceCommunicator(clientRdmaCmId);
}


void RdmaRoceCommunicator::Connect() {
    std::cout << "Connecting to RoCE endpoint at " << hostname << ":" << port << std::endl;

    // Setup address info for RDMA over Converged Ethernet (RoCE)
    rdma_addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;              // Use IPv4 address family (RoCE uses Ethernet typically)
    hints.ai_port_space = RDMA_PS_TCP;      // RoCE uses TCP port space (not IB port space)
    hints.ai_flags = 0;                     // No flags set for client-side connection

    rdma_addrinfo *rdmaAddrinfo;

    // Resolve the address using hostname and port for RoCE communication
    ktm_rdma_getaddrinfo(this->hostname, this->port, &hints, &rdmaAddrinfo);

    // Create communication manager ID with queue pair attributes for RoCE
    ibv_qp_init_attr qpInitAttr;
    memset(&qpInitAttr, 0, sizeof(qpInitAttr));

    // Set up the queue pair attributes for RoCE
    qpInitAttr.cap.max_send_wr = 10;
    qpInitAttr.cap.max_recv_wr = 10;
    qpInitAttr.cap.max_send_sge = 10;
    qpInitAttr.cap.max_recv_sge = 10;
    qpInitAttr.sq_sig_all = 1;              // Sign all send completions
    qpInitAttr.qp_type = IBV_QPT_RC;        // Use reliable connection queue pair type

    // Create RoCE endpoint
    ktm_rdma_create_ep(&roceCmId, rdmaAddrinfo, nullptr, &qpInitAttr);
    rdma_freeaddrinfo(rdmaAddrinfo);        // Free the address info once it's no longer needed

    // Connect to the RoCE endpoint
    ktm_rdma_connect(roceCmId, nullptr);

    // Modify the queue pair attributes after connection (RoCE-specific)
    auto *ibvQpAttr = static_cast<ibv_qp_attr *>(malloc(sizeof(ibv_qp_attr)));
    ibvQpAttr->min_rnr_timer = 1;  // Set RoCE-specific parameter (Minimum Retry Not Ready Timer)

    if (ibv_modify_qp(roceCmId->qp, ibvQpAttr, IBV_QP_MIN_RNR_TIMER)) {
        std::cerr << "ibv_modify_qp() failed: " << strerror(errno) << std::endl;
    }

    // Register the memory for RoCE communication (if needed)
    preregisteredMr = ktm_rdma_reg_msgs(roceCmId, preregisteredBuffer, 1024 * 5);
}


size_t RdmaRoceCommunicator::Read(char *buffer, size_t size) {
    std::cout << "Reading " << size << " bytes over RoCE" << std::endl;

    // If the size is smaller than the pre-registered buffer, use the pre-registered buffer
    if (size < 1024 * 5) {
        ktm_rdma_post_recv(roceCmId, nullptr, preregisteredBuffer, size, preregisteredMr);
    } else {
        // Register memory for receiving data if the size is larger than the pre-registered buffer
        memoryRegion = ktm_rdma_reg_msgs(roceCmId, buffer, size);
        // Post the receive request to the RoCE endpoint
        ktm_rdma_post_recv(roceCmId, nullptr, buffer, size, memoryRegion);
    }

    // Poll the completion queue for the work completion event (indicating read operation completion)
    int num_comp;
    do {
        num_comp = ibv_poll_cq(roceCmId->recv_cq, 1, &workCompletion);
    } while (num_comp == 0); // Continue polling until completion

    if (num_comp < 0) {
        throw std::runtime_error("ibv_poll_cq() failed during read operation");
    }

    if (workCompletion.status != IBV_WC_SUCCESS) {
        throw std::runtime_error("Failed to read data: " + std::string(ibv_wc_status_str(workCompletion.status)));
    }

    // If the size is smaller than the pre-registered buffer, copy the data into the user-provided buffer
    if (size < 1024 * 5) {
        memcpy(buffer, preregisteredBuffer, size);
    }

    return size; // Return the number of bytes read
}


size_t RdmaRoceCommunicator::Write(const char *buffer, size_t size) {
    std::cout << "Writing " << size << " bytes over RoCE" << std::endl;

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


void RdmaRoceCommunicator::Sync() {
    std::cout << "Synchronizing RoCE connection..." << std::endl;

    // Ensure that all RDMA operations are completed by polling the completion queue
    int num_comp;
    do {
        // Poll the send completion queue for any work completions
        num_comp = ibv_poll_cq(roceCmId->send_cq, 1, &workCompletion);
    } while (num_comp == 0);  // Continue polling if no completions yet

    // Handle errors during polling for send operations
    if (num_comp < 0) {
        throw std::runtime_error("ibv_poll_cq() failed during synchronization");
    }

    if (workCompletion.status != IBV_WC_SUCCESS) {
        throw std::runtime_error("Failed to synchronize RoCE connection: " + std::string(ibv_wc_status_str(workCompletion.status)));
    }

    // Similarly, you may want to poll the receive completion queue to ensure receive operations are completed
    do {
        // Poll the receive completion queue for any work completions
        num_comp = ibv_poll_cq(roceCmId->recv_cq, 1, &workCompletion);
    } while (num_comp == 0);  // Continue polling if no completions yet

    // Handle errors during polling for receive operations
    if (num_comp < 0) {
        throw std::runtime_error("ibv_poll_cq() failed during receive synchronization");
    }

    if (workCompletion.status != IBV_WC_SUCCESS) {
        throw std::runtime_error("Failed to synchronize receive: " + std::string(ibv_wc_status_str(workCompletion.status)));
    }

    std::cout << "RoCE synchronization complete." << std::endl;
}


void RdmaRoceCommunicator::Close() {
    std::cout << "Closing RoCE connection..." << std::endl;

    // Ensure that any active RDMA connections are properly disconnected before cleanup
    if (roceCmId) {
        rdma_disconnect(roceCmId);  // Disconnect the RDMA connection
        rdma_destroy_id(roceCmId);  // Destroy the communication manager ID
        std::cout << "RoCE connection closed successfully." << std::endl;
    } else {
        std::cout << "No active RoCE connection found to close." << std::endl;
    }

    // Deregister any memory regions associated with the RDMA communication
    if (preregisteredMr) {
        ibv_dereg_mr(preregisteredMr);  // Deregister the memory region
        preregisteredMr = nullptr;      // Nullify the pointer to avoid further access
        std::cout << "Deregistered preregistered memory region." << std::endl;
    }

    if (memoryRegion) {
        ibv_dereg_mr(memoryRegion);  // Deregister the memory region
        memoryRegion = nullptr;      // Nullify the pointer to avoid further access
        std::cout << "Deregistered additional memory region." << std::endl;
    }

    // Perform any other necessary resource cleanup (if applicable)
    // Placeholder for additional cleanup logic
}


extern "C" std::shared_ptr<gvirtus::communicators::Communicator> create_communicator(std::shared_ptr<gvirtus::communicators::Endpoint> endpoint) {
    // Extract hostname and port from the endpoint
    std::string hostname = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma_Roce>(endpoint)->address();
    std::string port = std::to_string(std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma_Roce>(endpoint)->port());

    // Create and return a new RdmaRoceCommunicator object using the extracted values
    return std::make_shared<gvirtus::communicators::RdmaRoceCommunicator>(hostname.data(), port.data());
}


} // namespace gvirtus::communicators
