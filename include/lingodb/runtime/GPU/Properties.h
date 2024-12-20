#ifndef LINGODB_RUNTIME_GPU_PROPERTIES_H
#define LINGODB_RUNTIME_GPU_PROPERTIES_H

#include <string>
namespace lingodb::runtime::gpu{
std::string getChipStr(const uint32_t deviceId=0);
} // end namespace lingodb::runtime::gpu

#endif //LINGODB_RUNTIME_GPU_PROPERTIES_H
