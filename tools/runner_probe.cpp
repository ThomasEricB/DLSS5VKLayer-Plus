#include "../common/runner_discovery.h"

#include <iostream>
#include <string>

static void printJson(const std::vector<dlssnr::RunnerInfo>& runners) {
    std::cout << "[";
    for (size_t i = 0; i < runners.size(); ++i) {
        const auto& r = runners[i];
        if (i) std::cout << ",";
        std::cout << "{\"name\":\"" << r.name << "\",\"path\":\"" << r.path
                  << "\",\"source\":\"" << r.source << "\",\"score\":" << r.score << "}";
    }
    std::cout << "]\n";
}

int main(int argc, char** argv) {
    bool paths = false;
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--paths") paths = true;
        else if (a == "--json") json = true;
        else if (a == "--help" || a == "-h") {
            std::cout << "usage: runner_probe [--paths|--json]\n";
            return 0;
        }
    }

    auto runners = dlssnr::discoverCustomRunners();

    if (json) {
        printJson(runners);
    } else if (paths) {
        for (const auto& r : runners) std::cout << r.path << "\n";
    } else {
        for (const auto& r : runners) std::cout << r.name << "\t" << r.path << "\n";
    }

    return runners.empty() ? 1 : 0;
}