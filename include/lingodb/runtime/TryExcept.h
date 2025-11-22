//
// Created by michael on 11/22/25.
//

#ifndef LINGODB_RUNTIME_TRYEXCEPT_H
#define LINGODB_RUNTIME_TRYEXCEPT_H

namespace lingodb::runtime {
class TryExcept {
        public:
        static void run(void (*tryBlock)(void*, void*), void (*exceptBlock)(void*, void*), void* tryArg, void* exceptArg, void* res);
};
} // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_TRYEXCEPT_H
