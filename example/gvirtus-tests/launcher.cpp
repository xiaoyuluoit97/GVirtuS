#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdio>
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <ctime>

void sigint_handler(int signum) {
    std::cout << "Received SIGINT (" << signum << "), terminating launcher." << std::endl;
    exit(0);
}

std::vector<std::string> get_test_cases(const std::string& test_binary) {
    std::vector<std::string> test_cases;
    FILE* fp = popen((test_binary + " --gtest_list_tests").c_str(), "r");
    if (!fp) {
        perror("popen failed");
        return test_cases;
    }

    char buffer[1024];
    std::string current_suite;

    while (fgets(buffer, sizeof(buffer), fp)) {
        std::string line(buffer);
        if (line.empty() || line == "\n") continue;

        if (line[0] != ' ') {
            current_suite = line.substr(0, line.find_first_of('\n'));
        } else {
            std::string testcase = line.substr(line.find_first_not_of(' '));
            testcase = testcase.substr(0, testcase.find_first_of('\n'));
            test_cases.push_back("--gtest_filter=" + current_suite + testcase);
        }
    }

    pclose(fp);
    return test_cases;
}

int main() {
    struct sigaction sa;
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);

    const char* test_binary = "./test";
    auto test_cases = get_test_cases(test_binary);

    for (const auto& test_case : test_cases) {
        pid_t pid = fork();

        if (pid < 0) {
            std::cerr << "Failed to fork" << std::endl;
            return 1;
        }

        if (pid == 0) {
            std::cout << "Launching: " << test_case << std::endl;
            execlp(test_binary, test_binary, test_case.c_str(), (char*)nullptr);
            perror("exec failed");
            exit(1);
        }

        // 超时版 waitpid (5秒超时)
        int status;
        time_t start_time = time(nullptr);
        while (true) {
            pid_t result = waitpid(pid, &status, WNOHANG);
            if (result == pid) break;
            if (result < 0) {
                perror("waitpid failed");
                break;
            }
            if (time(nullptr) - start_time > 5) {
                std::cout << "Timeout! Killing child process " << pid << std::endl;
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                break;
            }
            usleep(100000); // 0.1s
        }

        if (WIFEXITED(status)) {
            std::cout << "Exited with status " << WEXITSTATUS(status) << std::endl;
        } else if (WIFSIGNALED(status)) {
            std::cout << "Killed by signal " << WTERMSIG(status) << std::endl;
        }
    }

    std::cout << "✓ - All tests finished. Press CTRL+C to quit the launcher." << std::endl;
    pause();
    return 0;
}
