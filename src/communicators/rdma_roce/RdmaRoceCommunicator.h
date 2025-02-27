#ifndef RDMA_ROCE_COMMUNICATOR_H
#define RDMA_ROCE_COMMUNICATOR_H

#include "gvirtus/communicators/Communicator.h"
#include "ktmrdma.h"
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <string>
#include <netdb.h>

#define BACKLOG 8
#define BUF_SIZE 1024
#define DEBUG

namespace gvirtus::communicators {
    class RdmaRoceCommunicator : public Communicator {
    private:
        rdma_cm_id * roceCmId;
        rdma_cm_id * roceCmListenId;

        char hostname[256];
        char port[6];

        ibv_wc workCompletion;

        ibv_mr * memoryRegion;

        char preregisteredBuffer[1024 * 5];
        ibv_mr * preregisteredMr;

    public:
        RdmaRoceCommunicator() = default;
        RdmaRoceCommunicator(char * hostname, char * port);
        RdmaRoceCommunicator(rdma_cm_id * roceCmId);

        ~RdmaRoceCommunicator();

        void Serve();
        const Communicator *const Accept() const;

        void Connect();

        size_t Read(char * buffer, size_t size);
        size_t Write(const char * buffer, size_t size);

        void Sync();

        void Close();

        std::string to_string() {return "rdma_roce_communicator";};
    };
}

#endif // RDMA_ROCE_COMMUNICATOR_H
