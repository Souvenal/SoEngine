#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/common.h>
#include <memory>
#include <filesystem>

// Logging initialization options
struct InitOptions {
    std::string engineLoggerName = "SoEngine";
    std::string clientLoggerName = "App";
    std::filesystem::path logDirectory  = "logs";
    std::string fileName         = "engine.log";
    bool        enableFile       = true;
    bool        patternColored   = true;
    bool        recordTimeInConsole = false;
    spdlog::level::level_enum level = spdlog::level::info;
    bool        flushEveryTrace  = false;
};

namespace logging {

// Initialize logging (must be called first)
void init(const InitOptions& opts = {});

// Shutdown/cleanup logging (optional)
void shutdown();

// Dynamically change log level
void setLevel(spdlog::level::level_enum level);

// Flush all sinks immediately
void flush();

// Get logger (do not cache raw pointer, check for nullptr before use)
std::shared_ptr<spdlog::logger> core();   // Engine internal logger
std::shared_ptr<spdlog::logger> client(); // Application/client logger

// Check if logging is initialized
bool ready();

// Vulkan Debug Utils adapter (can be called in debugCallback)
// Requires including <vulkan/vulkan.h> or vulkan.hpp, implement in .cpp
void logVulkanMessage(
    int32_t severity,  // VkDebugUtilsMessageSeverityFlagBitsEXT
    uint32_t types,    // VkDebugUtilsMessageTypeFlagsEXT
    const char* messageIdName,
    int32_t messageIdNumber,
    const char* message
);

// Optionally log a message only once (deduplicated by key)
bool once(const std::string& key);

} // namespace logging

// ---------- Logging macros (ensure logging::init is called before use) ----------
#define LOG_CORE_TRACE(...)   do { if (auto _lg = ::logging::core())   _lg->trace(__VA_ARGS__); } while(0)
#define LOG_CORE_DEBUG(...)   do { if (auto _lg = ::logging::core())   _lg->debug(__VA_ARGS__); } while(0)
#define LOG_CORE_INFO(...)    do { if (auto _lg = ::logging::core())   _lg->info(__VA_ARGS__); } while(0)
#define LOG_CORE_WARN(...)    do { if (auto _lg = ::logging::core())   _lg->warn(__VA_ARGS__); } while(0)
#define LOG_CORE_ERROR(...)   do { if (auto _lg = ::logging::core())   _lg->error(__VA_ARGS__); } while(0)
#define LOG_CORE_CRIT(...)    do { if (auto _lg = ::logging::core())   _lg->critical(__VA_ARGS__); } while(0)

#define LOG_TRACE(...)        do { if (auto _lg = ::logging::client()) _lg->trace(__VA_ARGS__); } while(0)
#define LOG_DEBUG(...)        do { if (auto _lg = ::logging::client()) _lg->debug(__VA_ARGS__); } while(0)
#define LOG_INFO(...)         do { if (auto _lg = ::logging::client()) _lg->info(__VA_ARGS__); } while(0)
#define LOG_WARN(...)         do { if (auto _lg = ::logging::client()) _lg->warn(__VA_ARGS__); } while(0)
#define LOG_ERROR(...)        do { if (auto _lg = ::logging::client()) _lg->error(__VA_ARGS__); } while(0)
#define LOG_CRIT(...)         do { if (auto _lg = ::logging::client()) _lg->critical(__VA_ARGS__); } while(0)

#define LOG_INFO_ONCE(key, ...)  do { if(::logging::once(key)) LOG_INFO(__VA_ARGS__); } while(0)

#ifndef NDEBUG
    // Assertion macro with logging and abort in debug mode
    #define ASSERT(cond, ...) \
        do { if(!(cond)) { LOG_CORE_ERROR("Assertion Failed: " __VA_ARGS__); std::abort(); } } while(0)
#else
    // No-op in release mode
    #define ASSERT(cond, ...) do { (void)sizeof(cond); } while(0)
#endif