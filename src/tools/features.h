#ifndef LINGODB_TOOLS_FEATURES_H
#define LINGODB_TOOLS_FEATURES_H

#include <iostream>

void printFeatures() {
#if BASELINE_ENABLED == 1
   std::cerr << "baseline-backend\n";
#endif
}
#endif //LINGODB_TOOLS_FEATURES_H
