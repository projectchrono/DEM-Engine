//	Copyright (c) 2021, SBEL GPU Development Team
//	Copyright (c) 2021, University of Wisconsin - Madison
//
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef DEME_LOGGER_HPP
#define DEME_LOGGER_HPP

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <memory>

#include <cuda_runtime_api.h>
#include "BaseClasses.hpp"
// #include "../../DEM/Defines.h"

namespace deme {

// Verbosity
typedef int verbosity_t;
const verbosity_t VERBOSITY_QUIET = 0;
const verbosity_t VERBOSITY_ERROR = 1;
const verbosity_t VERBOSITY_WARNING = 2;
const verbosity_t VERBOSITY_INFO = 3;
const verbosity_t VERBOSITY_METRIC = 4;
const verbosity_t VERBOSITY_DEBUG = 5;

// -----------------------------
// Logging types and structure
// -----------------------------

enum class MessageType { Info, Warning, Error, Status };

struct LogMessage {
    MessageType type;
    std::string source;
    std::string message;
    std::string file;
    int line;
    std::string identifier;
};

// -----------------------------
// Logger and exception class (thread-safe, singleton)
// -----------------------------

class SolverException : public std::runtime_error {
  public:
    SolverException(const std::string& msg) : std::runtime_error(msg) {}
};

class Logger : private NonCopyable, public Singleton<Logger> {
  public:
    void SetVerbosity(verbosity_t level) {
        std::lock_guard<std::mutex> lock(mutex_);
        verbosity = level;
    }

    void Log(MessageType type, const std::string& source, const std::string& msg, const std::string& file, int line) {
        std::lock_guard<std::mutex> lock(mutex_);
        LogMessage Log{type, source, msg, file, line};
        logs.push_back(Log);

        if (should_print_immediately(type)) {
            std::cout << format(Log) << std::endl;
        }
    }

    // snprintf version of logging
    template <typename... Args>
    std::string Logf(MessageType type, const char* func, const char* file, int line, const char* fmt, Args&&... args) {
        constexpr size_t BUF_SIZE = 2048;
        char buffer[BUF_SIZE];
        std::snprintf(buffer, BUF_SIZE, fmt, std::forward<Args>(args)...);
        std::string message(buffer);
        Log(type, func, message, file, line);
        return message;
    }

    // std::string version of logging
    std::string Logf(MessageType type, const char* func, const char* file, int line, const std::string& message) {
        Log(type, func, message, file, line);
        return message;
    }

    void LogStatus(const std::string& identifier,
                   const std::string& func,
                   const std::string& msg,
                   const std::string& file,
                   int line) {
        std::lock_guard<std::mutex> lock(mutex_);

        LogMessage Log{MessageType::Status, func, msg, file, line, identifier};
        status_messages[identifier] = Log;

        if (should_print_immediately(MessageType::Status)) {
            std::cout << format(Log) << std::endl;
        }
    }

    void PrintAll(std::ostream& os = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& Log : logs) {
            os << format(Log) << std::endl;
        }
    }

    void PrintWarningsAndErrors(std::ostream& os = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& Log : logs) {
            if (Log.type == MessageType::Warning || Log.type == MessageType::Error) {
                os << format(Log) << std::endl;
            }
        }
    }

    void PrintStatusMessages(std::ostream& os = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& pair : status_messages) {
            os << format(pair.second) << std::endl;
        }
    }

    bool HasErrors() const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& Log : logs) {
            if (Log.type == MessageType::Error)
                return true;
        }
        return false;
    }

    bool HasWarnings() const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& Log : logs) {
            if (Log.type == MessageType::Warning)
                return true;
        }
        return false;
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        logs.clear();
        status_messages.clear();
    }

    verbosity_t GetVerbosity() const { return verbosity; }

  private:
    friend class Singleton<Logger>;  // Allows Singleton base to construct it

    Logger() : verbosity(VERBOSITY_WARNING) {}
    ~Logger() {}

    bool should_print_immediately(MessageType type) const {
        switch (type) {
            case MessageType::Error:
                return verbosity >= VERBOSITY_ERROR;
            case MessageType::Warning:
                return verbosity >= VERBOSITY_WARNING;
            case MessageType::Info:
                return verbosity >= VERBOSITY_INFO;
            case MessageType::Status:
                return verbosity >= VERBOSITY_METRIC;
            default:
                return false;
        }
    }

    std::string format(const LogMessage& Log) const {
        std::ostringstream oss;
        switch (Log.type) {
            case MessageType::Error:
                oss << "[ERROR]   ";
                break;
            case MessageType::Warning:
                oss << "[WARNING] ";
                break;
            case MessageType::Info:
                oss << "[INFO]    ";
                break;
            case MessageType::Status:
                oss << "[STATUS]  ";
                break;
        }
        oss << Log.source << ": " << Log.message;
        if (Log.type != MessageType::Info)
            oss << " (" << Log.file << ":" << Log.line << ")";
        if (!Log.identifier.empty() && Log.type == MessageType::Status)
            oss << " [id: " << Log.identifier << "]";
        return oss.str();
    }

    mutable std::mutex mutex_;
    std::vector<LogMessage> logs;
    std::unordered_map<std::string, LogMessage> status_messages;
    verbosity_t verbosity;
};

// -----------------------------
// Logging utils for easy usage
// -----------------------------

#define DEME_GET_VERBOSITY() Logger::GetInstance().GetVerbosity()

#define DEME_ERROR(...) \
    throw SolverException(Logger::GetInstance().Logf(MessageType::Error, __func__, __FILE__, __LINE__, __VA_ARGS__))

#define DEME_ERROR_NOTHROW(...) \
    Logger::GetInstance().Logf(MessageType::Error, __func__, __FILE__, __LINE__, __VA_ARGS__)

#define DEME_WARNING(...) Logger::GetInstance().Logf(MessageType::Warning, __func__, __FILE__, __LINE__, __VA_ARGS__)

#define DEME_INFO(...) Logger::GetInstance().Logf(MessageType::Info, __func__, __FILE__, __LINE__, __VA_ARGS__)

#define DEME_STATUS(identifier, ...)                                                       \
    do {                                                                                   \
        constexpr size_t BUF_SIZE = 2048;                                                  \
        char buffer[BUF_SIZE];                                                             \
        std::snprintf(buffer, BUF_SIZE, __VA_ARGS__);                                      \
        Logger::GetInstance().LogStatus(identifier, __func__, buffer, __FILE__, __LINE__); \
    } while (0)

#define DEME_GPU_CALL(code)                                                                                           \
    {                                                                                                                 \
        cudaError_t res = (code);                                                                                     \
        if (res != cudaSuccess) {                                                                                     \
            DEME_ERROR(                                                                                               \
                "GPU Error: %s\nYou can check out the troubleshoot section of DEME to see if it helps, or post this " \
                "error on Chrono's forum https://groups.google.com/g/projectchrono \n",                               \
                cudaGetErrorString(res));                                                                             \
        }                                                                                                             \
    }

#define DEME_GPU_CALL_NOTHROW(code)                                       \
    {                                                                     \
        cudaError_t res = (code);                                         \
        if (res != cudaSuccess) {                                         \
            DEME_ERROR_NOTHROW("GPU Error: %s", cudaGetErrorString(res)); \
        }                                                                 \
    }

}  // namespace deme

#endif