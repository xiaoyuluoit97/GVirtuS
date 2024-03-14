//
// Created by Mariano Aponte on 07/12/23.
//

#include <iostream>
#include <sstream>
#include <arpa/inet.h>
#include "RdmaCommunicator.h"

#include <gvirtus/communicators/Endpoint.h>
#include <gvirtus/communicators/Endpoint_Tcp.h>
#include <gvirtus/communicators/Endpoint_Rdma.h>


using gvirtus::communicators::RdmaCommunicator;

RdmaCommunicator::RdmaCommunicator(char * hostname, char * port) {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::RdmaCommunicator(char * hostname, char * port): called." << std::endl;
#endif

    if (port == nullptr or std::string(port).empty()) {
        throw "RdmaCommunicator: Port not specified...";
    }

    hostent *ent = gethostbyname(hostname);
    if (ent == NULL) {
        throw "RdmaCommunicator: Can't resolve hostname \"" + std::string(hostname) + "\"...";
    }

    auto addrLen = ent->h_length;
    this->hostname = new char[addrLen];
    memcpy(this->hostname, *ent->h_addr_list, addrLen);
    this->port = port;

    std::cout << "Hostname: " << this->hostname << ", port: " << this->port << std::endl;

    memset(&rdmaCmId, 0, sizeof(rdmaCmId));
    memset(&rdmaCmListenId, 0, sizeof(rdmaCmListenId));
}

RdmaCommunicator::RdmaCommunicator(rdma_cm_id *rdmaCmId, bool isServing) {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::RdmaCommunicator(rdma_cm_id *rdmaCmId, bool isServing): called." << std::endl;
#endif
    this->rdmaCmId = rdmaCmId;
    this->isServing = isServing;
}

RdmaCommunicator::~RdmaCommunicator() {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::~RdmaCommunicator(): called." << std::endl;
#endif
    rdma_disconnect(rdmaCmId);
    rdma_destroy_id(rdmaCmId);
}

void RdmaCommunicator::Serve() {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::Serve(): called." << std::endl;
#endif
    // Setup address info
    rdma_addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_port_space = RDMA_PS_TCP;
    hints.ai_flags = RAI_PASSIVE;

    rdma_addrinfo * rdmaAddrinfo;

    char testhost[50] = "192.168.4.99";
    char testport[50] = "9999";
    ktm_rdma_getaddrinfo(testhost, testport, &hints, &rdmaAddrinfo);

    // Create communication manager id with queue pair attributes
    ibv_qp_init_attr qpInitAttr;
    memset(&qpInitAttr, 0, sizeof(qpInitAttr));

    qpInitAttr.cap.max_send_wr = 20;
    qpInitAttr.cap.max_recv_wr = 20;
    qpInitAttr.cap.max_send_sge = 20;
    qpInitAttr.cap.max_recv_sge = 20;
    //qpInitAttr.qp_context = rdmaCmId;
    qpInitAttr.sq_sig_all = 1;

    ktm_rdma_create_ep(&rdmaCmListenId, rdmaAddrinfo, NULL, &qpInitAttr);
    rdma_freeaddrinfo(rdmaAddrinfo);

    // Listen for connections
    ktm_rdma_listen(rdmaCmListenId, BACKLOG);

    isServing = true;
}

const gvirtus::communicators::Communicator *const RdmaCommunicator::Accept() const {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::Accept(): called." << std::endl;
#endif
    rdma_cm_id * clientRdmaCmId;

#ifdef DEBUG
    std::cout << "RdmaCommunicator::Accept(): waiting for connection..." << std::endl;
#endif
    ktm_rdma_get_request(rdmaCmListenId, &clientRdmaCmId);
    ktm_rdma_accept(clientRdmaCmId, nullptr);

#ifdef DEBUG
    std::cout << "RdmaCommunicator::Accept(): connected..." << std::endl;
#endif
    return new RdmaCommunicator(clientRdmaCmId);
}

void RdmaCommunicator::Connect() {
    // Setup address info
    rdma_addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_port_space = RDMA_PS_TCP;

    rdma_addrinfo * rdmaAddrinfo;

    char testhost[50] = "192.168.4.99";
    char testport[50] = "9999";
    ktm_rdma_getaddrinfo(testhost, testport, &hints, &rdmaAddrinfo);

    // Create communication manager id with queue pair attributes
    ibv_qp_init_attr qpInitAttr;
    memset(&qpInitAttr, 0, sizeof(qpInitAttr));

    qpInitAttr.cap.max_send_wr = 20;
    qpInitAttr.cap.max_recv_wr = 20;
    qpInitAttr.cap.max_send_sge = 20;
    qpInitAttr.cap.max_recv_sge = 20;
    qpInitAttr.sq_sig_all = 1;

    ktm_rdma_create_ep(&rdmaCmId, rdmaAddrinfo, nullptr, &qpInitAttr);
    rdma_freeaddrinfo(rdmaAddrinfo);

    ktm_rdma_connect(rdmaCmId, nullptr);
}

size_t RdmaCommunicator::Read(char *buffer, size_t size) {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::Read(): called."
    << "Size: " << size << std::endl;
#endif
    // buffer needed for metadata exchange
    char * rdmaExcBuffer = static_cast<char *>(malloc(BUF_SIZE));
    ibv_mr * rdmaExcMr = ktm_rdma_reg_msgs(rdmaCmId, rdmaExcBuffer, BUF_SIZE);

    int num_wc;
    ibv_wc workCompletion;
    memset(&workCompletion, 0, sizeof(workCompletion));

    // GETTING WRITE SIZE
    ktm_rdma_post_recv(rdmaCmId, nullptr, rdmaExcBuffer, sizeof(size), rdmaExcMr);
    num_wc = ktm_rdma_get_recv_comp(rdmaCmId, &workCompletion);
    memcpy(&size, rdmaExcBuffer, sizeof(size));

#ifdef DEBUG
    std::cout << "ACTUAL SIZE: " << size << std::endl;
#endif

    // checking the buffer size and pointer
    if (buffer == NULL or size < 1) {
        char * newBuffer = (char *) realloc(buffer, size);
        if (newBuffer == NULL) {
            throw "RdmaCommunicator::Read(): realloc returned null...";
        }
        *buffer = *newBuffer;
    }

    // registering the buffer to write state
    ibv_mr * memoryRegion = ktm_rdma_reg_write(rdmaCmId, buffer, size);


    // SENDING WRITE MEMORY INFO TO THE PEER
    uintptr_t self_address = (uintptr_t) memoryRegion->addr;
    uint32_t self_rkey = memoryRegion->rkey;

#ifdef DEBUG
    std::cout << "using self address: " << self_address << " and self rkey: " << self_rkey << std::endl;
#endif

    // sending address info
    memcpy(rdmaExcBuffer, &self_address, sizeof(self_address));
    ktm_rdma_post_send(rdmaCmId, nullptr, rdmaExcBuffer, sizeof(self_address), rdmaExcMr, IBV_SEND_INLINE);
    num_wc = ktm_rdma_get_send_comp(rdmaCmId, &workCompletion);

    // sending rkey info
    memcpy(rdmaExcBuffer, &self_rkey, sizeof(self_rkey));
    ktm_rdma_post_send(rdmaCmId, nullptr, rdmaExcBuffer, sizeof(self_rkey), rdmaExcMr, IBV_SEND_INLINE);
    num_wc = ktm_rdma_get_send_comp(rdmaCmId, &workCompletion);

    // some cleanup
    rdma_dereg_mr(rdmaExcMr);
    free(rdmaExcBuffer);

    // message for signaling
    char * completionMsg = static_cast<char *>(calloc(1, sizeof(char)));
    ibv_mr * completionMr = ktm_rdma_reg_msgs(rdmaCmId, completionMsg, 1);

    // signal ready to be written
    ktm_rdma_post_send(rdmaCmId, nullptr, completionMsg, 1, completionMr, IBV_SEND_INLINE);
    num_wc = ktm_rdma_get_send_comp(rdmaCmId, &workCompletion);

    // (writing happens now)

    // wait peer to end writing
    ktm_rdma_post_recv(rdmaCmId, nullptr, completionMsg, 1, completionMr);
    num_wc = ktm_rdma_get_recv_comp(rdmaCmId, &workCompletion);

#ifdef DEBUG
    //for (unsigned int i = 0; i < size; i++) printf("%d LETTO %02X\n", i, buffer[i]);
    printf("letto: ");
    for (unsigned int i = 0; i < size; i++) printf("%c", buffer[i]);
    printf("\n");
#endif

    // some cleanup
    rdma_dereg_mr(memoryRegion);
    rdma_dereg_mr(completionMr);
    free(completionMsg);

    return size;
}

size_t RdmaCommunicator::Write(const char *buffer, size_t size) {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::Write(): called."
    << "Size: " << size << std::endl;
#endif
    // buffer needed for metadata exchange
    char * rdmaExcBuffer = static_cast<char *>(malloc(BUF_SIZE));
    ibv_mr * rdmaExcMr = ktm_rdma_reg_msgs(rdmaCmId, rdmaExcBuffer, BUF_SIZE);

    int num_wc;
    ibv_wc workCompletion;
    memset(&workCompletion, 0, sizeof(workCompletion));

    // SENDING WRITE SIZE
    memcpy(rdmaExcBuffer, &size, sizeof(size));
    ktm_rdma_post_send(rdmaCmId, nullptr, rdmaExcBuffer, sizeof(size), rdmaExcMr, IBV_SEND_INLINE);
    num_wc = ktm_rdma_get_send_comp(rdmaCmId, &workCompletion);

    // registering the buffer to be sent
    char * actualBuffer = (char *) malloc(size);
    memcpy(actualBuffer, buffer, size);
    ibv_mr * memoryRegion = ktm_rdma_reg_msgs(rdmaCmId, actualBuffer, size);

    // GETTING WRITE MEMORY INFO FROM THE PEER
    uintptr_t peer_address = -1;
    uint32_t peer_rkey = -1;

    // getting address
    ktm_rdma_post_recv(rdmaCmId, nullptr, rdmaExcBuffer, sizeof(peer_address), rdmaExcMr);
    num_wc = ktm_rdma_get_recv_comp(rdmaCmId, &workCompletion);
    memcpy(&peer_address, rdmaExcBuffer, sizeof(peer_address));

    // getting rkey
    ktm_rdma_post_recv(rdmaCmId, nullptr, rdmaExcBuffer, sizeof(peer_rkey), rdmaExcMr);
    num_wc = ktm_rdma_get_recv_comp(rdmaCmId, &workCompletion);
    memcpy(&peer_rkey, rdmaExcBuffer, sizeof(peer_rkey));

#ifdef DEBUG
    std::cout << "using peer address: " << peer_address << " and peer rkey: " << peer_rkey << std::endl;
#endif

    // some cleanup
    rdma_dereg_mr(rdmaExcMr);
    free(rdmaExcBuffer);

    // message for signaling
    char * completionMsg = static_cast<char *>(calloc(1, sizeof(char)));
    ibv_mr * completionMr = ktm_rdma_reg_msgs(rdmaCmId, completionMsg, 1);

    // wait signal to write
    ktm_rdma_post_recv(rdmaCmId, nullptr, completionMsg, 1, completionMr);
    num_wc = ktm_rdma_get_recv_comp(rdmaCmId, &workCompletion);

    // write
    ktm_rdma_post_write(rdmaCmId, nullptr, actualBuffer, size, memoryRegion, IBV_WR_RDMA_WRITE, peer_address, peer_rkey);
    num_wc = ktm_rdma_get_send_comp(rdmaCmId, &workCompletion);

    // signal write ended
    ktm_rdma_post_send(rdmaCmId, nullptr, completionMsg, 1, completionMr, IBV_SEND_INLINE);
    num_wc = ktm_rdma_get_send_comp(rdmaCmId, &workCompletion);

#ifdef DEBUG
    //for (unsigned int i = 0; i < size; i++) printf("%d SCRITTO %02X \n", i, actualBuffer[i]);
    printf("scritto: ");
    for (unsigned int i = 0; i < size; i++) printf("%c", buffer[i]);
    printf("\n");
#endif

    // some cleanup
    rdma_dereg_mr(memoryRegion);
    rdma_dereg_mr(completionMr);
    free(actualBuffer);
    free(completionMsg);

    return size;
}

void RdmaCommunicator::Sync() {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::Sync(): called." << std::endl;
#endif
    // TODO: Implement.
    std::cerr << "RdmaCommunicator::Sync() is not yet implemented!!!" << std::endl;
}

void RdmaCommunicator::Close() {
#ifdef DEBUG
    std::cout << "RdmaCommunicator::Close(): called." << std::endl;
#endif
    rdma_disconnect(rdmaCmId);
    rdma_destroy_id(rdmaCmId);
}


extern "C" std::shared_ptr <RdmaCommunicator> create_communicator(std::shared_ptr <gvirtus::communicators::Endpoint> end) {
    /*
    std::string arg =
            "rdma://" +
            std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma>(end)->address() +
            ":" +
            std::to_string(std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma>(end)->port());
*/
    std::string hostname;
    std::string port;

    hostname = std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma>(end)->address();
    port = std::to_string(std::dynamic_pointer_cast<gvirtus::communicators::Endpoint_Rdma>(end)->port());

    return std::make_shared<RdmaCommunicator>(hostname.data(), port.data());
}


