#pragma once
#include <string>
#include <vector>

namespace dlssnr {

struct RunnerInfo {
    std::string id;
    std::string name;
    std::string path;
    std::string source;
    int score = 0;
};

std::vector<RunnerInfo> discoverCustomRunners();
std::string runnerDisplayName(const RunnerInfo& runner);

}