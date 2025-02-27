#include "gvirtus/communicators/Endpoint_Rdma_Roce.h"
#include "gvirtus/communicators/EndpointFactory.h"
#include <regex>
#include <iostream>

namespace gvirtus::communicators {

Endpoint_Rdma_Roce::Endpoint_Rdma_Roce(const std::string &endp_suite,
                                       const std::string &endp_protocol,
                                       const std::string &endp_address,
                                       const std::string &endp_port) {
    suite(endp_suite);
    protocol(endp_protocol);
    address(endp_address);
    port(endp_port);
}

Endpoint &Endpoint_Rdma_Roce::suite(const std::string &suite) {
    std::regex pattern{R"([[:alpha:]]*-[[:alpha:]]*)"};
    std::smatch matches;
    if (std::regex_search(suite, matches, pattern) && suite == matches[0]) {
        _suite = suite;
    }
    return *this;
}

Endpoint &Endpoint_Rdma_Roce::protocol(const std::string &protocol) {
    if (protocol == "roce") {
        _protocol = protocol;
    }
    return *this;
}

Endpoint_Rdma_Roce &Endpoint_Rdma_Roce::address(const std::string &address) {
    std::regex pattern{
            R"(^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$)"};
    if (std::regex_match(address, pattern)) {
        _address = address;
    }
    return *this;
}

Endpoint_Rdma_Roce &Endpoint_Rdma_Roce::port(const std::string &port) {
    std::regex pattern{R"((6553[0-5]|655[0-2][0-9]\d|65[0-4](\d){2}|6[0-4](\d){3}|[1-5](\d){4}|[1-9](\d){0,3}))"};
    if (std::regex_match(port, pattern)) {
        _port = static_cast<std::uint16_t>(std::stoi(port));
    }
    return *this;
}

void from_json(const nlohmann::json &j, Endpoint_Rdma_Roce &end) {
    auto el = j["communicator"][EndpointFactory::index()]["endpoint"];
    end.suite(el.at("suite"));
    end.protocol(el.at("protocol"));
    end.address(el.at("server_address"));
    end.port(el.at("port"));
}

} // namespace gvirtus::communicators
