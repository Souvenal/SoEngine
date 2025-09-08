#include "logging.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/async.h>
#include <unordered_set>
#include <filesystem>
#include <mutex>

namespace {
    std::shared_ptr<spdlog::logger> g_core;
    std::shared_ptr<spdlog::logger> g_client;
    std::optional<InitOptions> g_opts {};

    std::unordered_set<std::string> g_onceKeys;
    std::mutex g_mutex;
}

namespace logging {

void init(const InitOptions& opts) {
    std::scoped_lock lock(g_mutex);
    if (g_opts) return;
    g_opts = opts;

    try {
        std::vector<spdlog::sink_ptr> sinks;
        auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        std::string timePattern = opts.recordTimeInConsole ? "[%Y-%m-%d %H:%M:%S.%e] " : "";
        if (opts.patternColored) {
            consoleSink->set_pattern(timePattern + "[%^%l%$] [%n] %v");
        } else {
            consoleSink->set_pattern(timePattern + "[%l] [%n] %v");
        }
        sinks.push_back(consoleSink);

        if (opts.enableFile) {
            std::error_code ec;
            std::filesystem::create_directories(opts.logDirectory, ec);
            auto filePath = std::filesystem::path(opts.logDirectory) / opts.fileName;
            auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filePath.string(), true);
            fileSink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
            sinks.push_back(fileSink);
        }

        g_core = std::make_shared<spdlog::logger>(opts.engineLoggerName, sinks.begin(), sinks.end());
        g_client = std::make_shared<spdlog::logger>(opts.clientLoggerName, sinks.begin(), sinks.end());

        spdlog::register_logger(g_core);
        spdlog::register_logger(g_client);

        g_core->set_level(opts.level);
        g_client->set_level(opts.level);

        if (opts.flushEveryTrace) {
            g_core->flush_on(spdlog::level::trace);
            g_client->flush_on(spdlog::level::trace);
        } else {
            g_core->flush_on(spdlog::level::err);
            g_client->flush_on(spdlog::level::err);
        }

    } catch (const std::exception& e) {
        // Fallback: 标记未初始化
        g_core.reset();
        g_client.reset();
        g_opts.reset();
        throw;
    }
}

void shutdown() {
    std::scoped_lock lock(g_mutex);
    if (!g_opts) return;
    spdlog::drop_all();
    g_core.reset();
    g_client.reset();
    g_onceKeys.clear();
    g_opts.reset();
}

void setLevel(spdlog::level::level_enum level) {
    std::scoped_lock lock(g_mutex);
    if (g_core)   g_core->set_level(level);
    if (g_client) g_client->set_level(level);
}

void flush() {
    std::scoped_lock lock(g_mutex);
    if (g_core)   g_core->flush();
    if (g_client) g_client->flush();
    spdlog::apply_all([](const std::shared_ptr<spdlog::logger>& lg) {
        if (lg) lg->flush();
    });
}

std::shared_ptr<spdlog::logger> core() {
    return g_core;
}

std::shared_ptr<spdlog::logger> client() {
    return g_client;
}

bool ready() { return g_opts.has_value(); }

static spdlog::level::level_enum mapSeverity(int32_t severity) {
#ifdef VK_VERSION_1_0
    // 若包含 vulkan_core.h
    switch (severity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: return spdlog::level::trace;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:    return spdlog::level::info;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: return spdlog::level::warn;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:   return spdlog::level::err;
        default: break;
    }
#endif
    // 兜底
    if (severity <= 0) return spdlog::level::trace;
    return spdlog::level::info;
}

void logVulkanMessage(
    int32_t severity,
    uint32_t types,
    const char* messageIdName,
    int32_t messageIdNumber,
    const char* message
) {
    auto lvl = mapSeverity(severity);
    auto lg = core();
    if (!lg) return;

    // 简单类型标记
    std::string typeStr;
    {
#ifdef VK_VERSION_1_0
        if (types & 0x00000001) typeStr += "GEN|"; // GENERAL
        if (types & 0x00000002) typeStr += "VALID|";
        if (types & 0x00000004) typeStr += "PERF|";
        if (types & 0x00000008) typeStr += "DEV|";
#endif
        if (typeStr.empty()) typeStr = "UNK|";
        if (!typeStr.empty() && typeStr.back()=='|') typeStr.pop_back();
    }

    lg->log(lvl, "[Vulkan:{} id:{}({})] {}", typeStr, messageIdName ? messageIdName : "None", messageIdNumber, message ? message : "");
}

bool once(const std::string& key) {
    std::scoped_lock lock(g_mutex);
    auto [it, inserted] = g_onceKeys.emplace(key);
    return inserted;
}

}