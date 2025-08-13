#pragma once

#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/execution/baseline/utils.hpp"

#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include <mlir/Support/LLVM.h>

#include <memory>


namespace lingodb::execution::baseline {
    using namespace compiler;

    class SpdLogSpoof {
        // storage for the log messages
        std::ostringstream oss;
        std::shared_ptr<spdlog::sinks::ostream_sink_mt> ostream_sink;
        std::shared_ptr<spdlog::logger> logger;
        std::shared_ptr<spdlog::logger> old_logger;
        unsigned current_listeners{0};
        std::mutex mutex;

    public:
        SpdLogSpoof() {
        }

        ~SpdLogSpoof() {
            if (current_listeners > 0) {
                // If we are destructing while still in a spoofed state, we need to restore the old logger
                // to avoid leaving spdlog in an inconsistent state.
                assert(old_logger && "SpdLogSpoof destructor called without a valid old logger");
                spdlog::set_default_logger(old_logger);
            }
        }

        void enter() {
            std::lock_guard lock(mutex);
            if (current_listeners == 0) {
                ostream_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(oss);
                logger = std::make_shared<spdlog::logger>("string_logger", ostream_sink);
                old_logger = spdlog::default_logger();
                spdlog::set_default_logger(logger);
            }
            current_listeners++;
        }

        void exit() {
           std::lock_guard<std::mutex> lock(mutex);
           assert(current_listeners > 0 && "SpdLogSpoof::exit called without matching enter");
           current_listeners--;
           if (current_listeners == 0) {
              spdlog::set_default_logger(old_logger);
              ostream_sink.reset();
              old_logger.reset();
              logger.reset();
              oss.clear();
           }
        }

        std::string logs() {
            return oss.str();
        }
    };

    static std::optional<size_t> get_size(mlir::Type type) noexcept {
        return mlir::TypeSwitch<mlir::Type, size_t>(type)
                .Case<mlir::IntegerType, mlir::FloatType>([](auto intType) {
                    return intType.getIntOrFloatBitWidth() / 8;
                })
                .Case<dialect::util::VarLen32Type, dialect::util::BufferType>([](auto) { return 16; })
                .Case<mlir::TupleType>([](auto t) { return TupleHelper{t}.sizeAndPadding().first; })
                .Case<mlir::IndexType, dialect::util::RefType>([](auto) { return 8; })
                .Default([](mlir::Type t) {
                    t.dump();
                    assert(0 && "Unsupported type for size calculation");
                    return 0;
                });
    }

    // TODO: remove this as soon as either llvm iters have a richer api or tpde drops some requirements
    template<typename Iter>
    struct IRIterWrapper {
        Iter it;
        using value_type = typename std::iterator_traits<Iter>::value_type;

        IRIterWrapper() = default;

        explicit IRIterWrapper(Iter it) : it(it) {
        }

        IRIterWrapper &operator++() {
            ++it;
            return *this;
        }

        value_type operator*() const { return *it; }
        bool operator!=(const IRIterWrapper &other) const { return it != other.it; }
        bool operator==(const IRIterWrapper &other) const { return it == other.it; }
    };

    template<typename Iter>
    struct IRRangeWrapper {
        IRIterWrapper<Iter> b, e;
        IRIterWrapper<Iter> begin() const { return b; }
        IRIterWrapper<Iter> end() const { return e; }
    };

    template<class Iter>
    auto llvm_iter_range_to_tpde_iter(llvm::iterator_range<Iter> iter) {
        return IRRangeWrapper<Iter>{IRIterWrapper<Iter>(iter.begin()), IRIterWrapper<Iter>(iter.end())};
    }
} // lingodb::execution::baseline
