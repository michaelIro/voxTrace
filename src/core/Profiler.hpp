#pragma once
/**
 * @file Profiler.hpp
 * @brief Coarse wall-clock sections, throughput counters and a resource report.
 *
 * Answers "where does the time go and what does the machine do" for a run
 * without external tools: wrap each phase in a VT_PROFILE scope, add counters
 * with vtprof::count(), and call vtprof::report() at the end:
 *
 *     phase                   calls      seconds        %
 *     beam-trace                  1        12.61     57.9
 *     scan (measurement)          1         4.31     19.8
 *     ...
 *     counters: primaries=10000000  events=36030 ...
 *     resources: wall 21.8 s | cpu 158.7 s (7.3 cores busy) | peak rss 412 MB
 *
 * Sections are meant for PHASES (a handful per run), not inner loops — they
 * take a mutex. Counters are cheap if accumulated locally and added once per
 * phase. CPU utilisation = (user+system time)/wall from getrusage, so an
 * OpenMP/Kokkos build shows how many cores were actually busy; peak RSS covers
 * memory. For kernel-level GPU/device profiling in Kokkos builds, point
 * KOKKOS_TOOLS_LIBS at a kokkos-tools library (kp_kernel_timer,
 * kp_memory_events, …) — no rebuild needed.
 */

#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>
#include <sys/resource.h>

namespace vtprof {

using Clock = std::chrono::steady_clock;

struct Entry { std::string name; double sec = 0; long calls = 0; long count = 0; bool isCounter = false; };

inline std::vector<Entry>& entries() { static std::vector<Entry> e; return e; }
inline std::mutex&         mtx()     { static std::mutex m; return m; }
inline const Clock::time_point programStart = Clock::now();   // static-init = program start

inline Entry& find(const char* name, bool counter) {
    for (Entry& e : entries())
        if (e.isCounter == counter && e.name == name) return e;
    entries().push_back({name, 0, 0, 0, counter});
    return entries().back();
}

/// Add @p n to counter @p name (accumulate locally in hot loops, add once).
inline void count(const char* name, long n = 1) {
    std::lock_guard<std::mutex> lock(mtx());
    find(name, true).count += n;
}

inline void add(const char* name, double dt) {
    std::lock_guard<std::mutex> lock(mtx());
    Entry& e = find(name, false);
    e.sec += dt;
    ++e.calls;
}

/// RAII wall-clock section; nest freely, use one per program phase.
class Scope {
    const char* name_;
    Clock::time_point start_;
public:
    explicit Scope(const char* name) : name_(name), start_(Clock::now()) {}
    ~Scope() { add(name_, std::chrono::duration<double>(Clock::now() - start_).count()); }
};

/// Sequential phases inside one block: next() closes the current section and
/// opens the following one (for code where blocks can't nest cleanly).
class Phase {
    const char* name_;
    Clock::time_point start_;
public:
    explicit Phase(const char* name) : name_(name), start_(Clock::now()) {}
    void next(const char* name) { close(); name_ = name; start_ = Clock::now(); }
    void close() {
        if (!name_) return;
        add(name_, std::chrono::duration<double>(Clock::now() - start_).count());
        name_ = nullptr;
    }
    ~Phase() { close(); }
};

/// Print the section table, counters and OS resource usage to @p out.
inline void report(std::FILE* out = stdout) {
    double wall = std::chrono::duration<double>(Clock::now() - programStart).count();
    std::lock_guard<std::mutex> lock(mtx());
    std::fprintf(out, "\nprofile — %-22s %6s %10s %7s\n", "phase", "calls", "seconds", "%");
    for (const Entry& e : entries())
        if (!e.isCounter)
            std::fprintf(out, "          %-22s %6ld %10.2f %6.1f\n",
                         e.name.c_str(), e.calls, e.sec, 100.0 * e.sec / wall);
    std::fprintf(out, "  counters:");
    for (const Entry& e : entries())
        if (e.isCounter) std::fprintf(out, "  %s=%ld", e.name.c_str(), e.count);
    std::fprintf(out, "\n");

    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    double cpu = ru.ru_utime.tv_sec + 1e-6*ru.ru_utime.tv_usec
               + ru.ru_stime.tv_sec + 1e-6*ru.ru_stime.tv_usec;
#ifdef __APPLE__
    double rssMB = ru.ru_maxrss / (1024.0 * 1024.0);   // bytes on macOS
#else
    double rssMB = ru.ru_maxrss / 1024.0;              // kilobytes on Linux
#endif
    std::fprintf(out, "  resources: wall %.1f s | cpu %.1f s (%.1f cores busy) | peak rss %.0f MB\n",
                 wall, cpu, cpu / (wall > 0 ? wall : 1), rssMB);
}

}  // namespace vtprof

#define VT_PROF_CAT2(a, b) a##b
#define VT_PROF_CAT(a, b) VT_PROF_CAT2(a, b)
#define VT_PROFILE(name) vtprof::Scope VT_PROF_CAT(vt_prof_scope_, __LINE__)(name)
