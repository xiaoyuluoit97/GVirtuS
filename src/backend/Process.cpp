#include "gvirtus/backend/Process.h"

#include <gvirtus/common/JSON.h>
#include <gvirtus/common/SignalException.h>
#include <gvirtus/common/SignalState.h>

#include <gvirtus/backend/Process.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <functional>
#include <thread>
#include <iostream>
#include <cstring>

#define DEBUG

using gvirtus::backend::Process;
using gvirtus::common::LD_Lib;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::Endpoint;

using std::chrono::steady_clock;

using namespace std;

Process::Process(std::shared_ptr<LD_Lib<Communicator, std::shared_ptr<Endpoint>>> communicator, vector <string> &plugins) : Observable() {
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Process"));

    // Set the logging level
    log4cplus::LogLevel logLevel = log4cplus::INFO_LOG_LEVEL;
    char *val = getenv("GVIRTUS_LOGLEVEL");

    std::string logLevelString = (val == NULL ? std::string("") : std::string(val));

    if (!logLevelString.empty()) {
        logLevel = std::stoi(logLevelString);
    }
    logger.setLogLevel(logLevel);

    signal(SIGCHLD, SIG_IGN);
    _communicator = communicator;
    mPlugins = plugins;
}

bool getstring(Communicator *c, std::string &s) {
    // #ifdef DEBUG
    //     printf("getstring called.\n");
    // #endif

        std::string communicator_type = c->to_string();  // Get the communicator type
    
        // TODO: FIX LISKOV SUBSTITUTION AND DEPENDENCE INVERSION!!!!!
        if (communicator_type == "tcpcommunicator") {
            // TCP communicator (original logic)
            s = "";
            char ch = 0;
            while (c->Read(&ch, 1) == 1) {
                // If reading is ended, return true
                if (ch == 0) {
                    return true;
                }
                s += ch;
            }
            return false;
        }


        else if (communicator_type == "rdmacommunicator" || communicator_type == "rdma_roce_communicator") {
        // RDMA communicator (modified logic for RDMA)
        try {
            s = "";

            // Initial buffer size set to a reasonable default (e.g., 30 bytes)
            size_t size = 30;
            char *buf = (char *)malloc(size);  // Allocate memory for reading

            // Read data from the communicator
            size_t read_size = c->Read(buf, size);  

            // Dynamically adjust the buffer size if the read data exceeds the initial buffer size
            if (read_size > size) {
                free(buf);  // Free the previous buffer if it’s too small
                buf = (char *)malloc(read_size);  // Allocate new buffer with the correct size
                read_size = c->Read(buf, read_size);  // Re-read the data into the new buffer
            }

            // Trim null characters (ASCII: 0) from the end of the data (Remove the last bytes if that equal to 0)
            size_t trimmed_size = read_size;

            // Trace the entire string and remove characters after the first null character
            for (size_t i = 0; i < trimmed_size; ++i) {
                if (buf[i] == 0) {
                    trimmed_size = i;  // Cut off the string at the first null character
                break;
                }
            }

            // Check if the read was successful and append the trimmed data to the string
            if (trimmed_size > 0) {
                s += std::string(buf, trimmed_size);  // Append the trimmed read data to the string
                free(buf);  // Free the allocated buffer after use

                return true;
            }
            }
            catch (std::string &exc) {
                std::cerr << "Exception: " << exc << std::endl;
            }
            catch (const char *exc) {
                std::cerr << "Exception: " << exc << std::endl;
            }
            return false;
        }
        
        // If an unrecognized communicator type is encountered
        std::cerr << "Unknown communicator type: " << communicator_type << std::endl;
        throw "Communicator getstring read error... Unknown communicator type...";
    }
    
    

extern std::string getEnvVar(std::string const &key);

std::string getGVirtuSHome() {
    std::string gvirtus_home = getEnvVar("GVIRTUS_HOME");
    return gvirtus_home;
}

void Process::Start() {
    LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] Process::Start() called.");

    // Log loaded plugins
    for (const auto &plug : mPlugins) {
        LOG4CPLUS_DEBUG(logger, "✓ - Loaded plugin: " << plug);
    }

    for_each(mPlugins.begin(), mPlugins.end(), [this](const std::string &plug) {
                 std::string gvirtus_home = getGVirtuSHome();

                 std::string to_append = "libgvirtus-plugin-" + plug + ".so";
                 LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "] appending " << to_append << ".");

                 auto ld_path = fs::path(gvirtus_home + "/lib").append(to_append);

                 try {
                     auto dl = std::make_shared<LD_Lib<Handler>>(ld_path, "create_t");
                     dl->build_obj();
                     _handlers.push_back(dl);
                 }
                 catch (const std::string &e) {
                     LOG4CPLUS_ERROR(logger, e);
                 }
             }
    );

    // Log the handlers to check if the routine is registered
    LOG4CPLUS_DEBUG(logger, "✓ - Handlers loaded:");
    for (auto &handler : _handlers) {
        LOG4CPLUS_DEBUG(logger, "✓ - Handler for: " << typeid(*handler->obj_ptr()).name());
    }

    // inserisci i sym dei plugin in h
    std::function<void(Communicator *)> execute = [=](Communicator *client_comm) {
        LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "]" << "Process::Start()'s \"execute\" lambda called");

        string routine;    //Empty string
        std::shared_ptr<Buffer> input_buffer = std::make_shared<Buffer>();

        while (getstring(client_comm, routine)) {           //Both CLient communication and rountine out is True from getstring function
            LOG4CPLUS_DEBUG(logger, "✓ - Received routine: " << routine);
            std::cout << "✓ - Received routine: " << routine << std::endl;  //Unitll this point "cudaRegisterFatBinary" present

            input_buffer->Reset(client_comm);

            std::shared_ptr<Handler> h = nullptr;
            for (auto &ptr_el : _handlers) {
                // Log handler type and routine check
                LOG4CPLUS_DEBUG(logger, "✓ - Checking if handler can execute routine: " << routine);
                if (ptr_el->obj_ptr()->CanExecute(routine)) {   //passing the routine(cudaRegisterFatBinary Library) Checks the possibility of ececution?
                    h = ptr_el->obj_ptr();
                    LOG4CPLUS_DEBUG(logger, "✓ - Found handler for routine: " << routine);
                    break;
                } else {
                    LOG4CPLUS_DEBUG(logger, "✖ - Handler cannot execute routine: " << routine);
                }
            }

            std::shared_ptr<communicators::Result> result;
            if (h == nullptr) {
                LOG4CPLUS_ERROR(logger, "✖ - [Process " << getpid() << "]: Requested unknown routine___ " << routine << ".");
                result = std::make_shared<communicators::Result>(-1, std::make_shared<Buffer>());
            } else {
                // Execute the routine and save the result
                auto start = steady_clock::now();
                result = h->Execute(routine, input_buffer);
                result->TimeTaken(std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - start).count() / 1000.0);
            }

            // Write the result to the communicator
            result->Dump(client_comm);
            if (result->GetExitCode() != 0 && routine.compare("cudaLaunch")) {
                LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "]: Requested '" << routine << "' routine.");
                LOG4CPLUS_DEBUG(logger, "✓ - - [Process " << getpid() << "]: Exit Code '" << result->GetExitCode() << "'.");
            }
        }

        Notify("process-ended");
    };

    try {
        _communicator->obj_ptr()->Serve();

        int pid = 0;
        while (true) {
            Communicator *client = const_cast<Communicator *>(_communicator->obj_ptr()->Accept());

            if (client != nullptr) {
                std::thread(execute, client).detach();
            } else {
                _communicator->obj_ptr()->run();
            }

            // Check for SIGINT
            if (common::SignalState::get_signal_state(SIGINT)) {
                LOG4CPLUS_DEBUG(logger, "✓ - SIGINT received, killing server on [Process " << getpid() << "]...");
                break;
            }
        }
    }
    catch (std::string &exc) {
        LOG4CPLUS_ERROR(logger, "✖ - [Process " << getpid() << "]: " << exc);
    }

    LOG4CPLUS_DEBUG(logger, "✓ - Process::Start() returned [Process " << getpid() << "].");
}


Process::~Process() {
    _communicator.reset();
    _handlers.clear();
    mPlugins.clear();
}


