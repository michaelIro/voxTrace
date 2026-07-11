#pragma once
/**
 * @file DeviceBuffer.hpp
 * @brief Host-fillable, device-readable array — the one place host↔device data moves.
 *
 * The dispatch layer fills plain arrays on the host (voxels, materials,
 * elements, beam rays, event slots) and kernels read/write them by raw
 * pointer. DeviceBuffer hides where that memory lives:
 *
 *   - Kokkos build:    a `Kokkos::View` in the default memory space plus its
 *     host mirror; `toDevice()`/`toHost()` are `deep_copy`s (no-ops when the
 *     backend is host-side, real transfers under CUDA/HIP).
 *   - host-only build: a `std::vector`; host and device pointers alias and
 *     the copies are no-ops.
 *
 * Usage: fill `host()`, call `toDevice()`, hand `device()` to the kernel
 * functor (a raw pointer is what device code wants); after the kernel, call
 * `toHost()` before reading results. T must be trivially copyable — which is
 * exactly the voxTrace device-class pattern.
 */

#include "Platform.hpp"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

template <class T>
class DeviceBuffer {
#ifndef VOXTRACE_HOST_ONLY
    // deduced mirror type — the HostMirror alias moved between Kokkos versions
    using DevView  = Kokkos::View<T*>;
    using HostView = decltype(Kokkos::create_mirror_view(std::declval<const DevView&>()));
    DevView  dev_;
    HostView host_;

public:
    DeviceBuffer(const char* name, size_t n)
        : dev_(Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), n),
          host_(Kokkos::create_mirror_view(dev_)) {}

    T*       host()         { return host_.data(); }
    const T* host()   const { return host_.data(); }
    T*       device() const { return dev_.data(); }
    size_t   size()   const { return dev_.extent(0); }
    void toDevice() { Kokkos::deep_copy(dev_, host_); }
    void toHost()   { Kokkos::deep_copy(host_, dev_); }
#else
    std::vector<T> data_;

public:
    DeviceBuffer(const char*, size_t n) : data_(n) {}

    T*       host()         { return data_.data(); }
    const T* host()   const { return data_.data(); }
    T*       device() const { return const_cast<T*>(data_.data()); }
    size_t   size()   const { return data_.size(); }
    void toDevice() {}
    void toHost()   {}
#endif

    /// Convenience: allocate from an existing host array and push it down.
    DeviceBuffer(const char* name, const std::vector<T>& src) : DeviceBuffer(name, src.size()) {
        std::memcpy(host(), src.data(), src.size() * sizeof(T));
        toDevice();
    }
};
