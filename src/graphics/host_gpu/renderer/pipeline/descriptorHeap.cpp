#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <array>

namespace Libs::Graphics {
namespace {

constexpr uint32_t   DescriptorHeapCount = 1024;
constexpr std::array<uint32_t, 4> BasePoolSizes = {65536u, 65536u, 65536u, 65536u};

constexpr std::array<vk::DescriptorType, 4> PoolTypes = {
    vk::DescriptorType::eStorageBuffer,
    vk::DescriptorType::eSampledImage,
    vk::DescriptorType::eStorageImage,
    vk::DescriptorType::eSampler,
};

size_t DescriptorTypeIndex(vk::DescriptorType type) {
    for (size_t i = 0; i < PoolTypes.size(); ++i) {
        if (PoolTypes[i] == type) {
            return i;
        }
    }
    return PoolTypes.size();
}

} // namespace

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
    for (size_t i = 0; i < m_required_sizes.size(); ++i) {
        m_required_sizes[i] = std::max(m_required_sizes[i], BasePoolSizes[i]);
    }
    CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
    m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
    for (const auto& entry: m_pending_pools) {
        m_master_semaphore.Wait(entry.tick);
        m_graphics.device.destroyDescriptorPool(entry.pool, nullptr);
    }
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout,
                                        std::span<const vk::WriteDescriptorSet> writes) {
    KYTY_PROFILER_FUNCTION();
    EXIT_IF(layout == nullptr);

    // Grow required pool sizes based on this layout's descriptor counts.
    if (!writes.empty()) {
        std::array<uint32_t, 4> counts {};
        for (const auto& write: writes) {
            const size_t index = DescriptorTypeIndex(write.descriptorType);
            if (index < counts.size()) {
                counts[index] += write.descriptorCount;
            }
        }
        for (size_t i = 0; i < m_required_sizes.size(); ++i) {
            m_required_sizes[i] = std::max(m_required_sizes[i], counts[i]);
        }
    }

    auto& batch = m_sets[layout];
    if (batch.size != 0) {
        return batch.sets[--batch.size];
    }
    if (Allocate(layout, batch)) {
        return batch.sets[--batch.size];
    }

    m_pending_pools.push_back({m_current_pool, m_master_semaphore.CurrentTick(),
                               m_current_sizes, m_current_max_sets});
    m_current_pool = nullptr;

    // Try to reuse a free pool that is large enough.
    bool reused = false;
    // Search for a free, sufficiently large pending pool.
    for (auto it = m_pending_pools.begin(); it != m_pending_pools.end(); ++it) {
        if (it->pool == m_pending_pools.back().pool) {
            continue; // skip the one we just retired
        }
        if (!m_master_semaphore.IsFree(it->tick)) {
            continue;
        }
        bool fits = true;
        for (size_t i = 0; i < m_required_sizes.size(); ++i) {
            if (it->sizes[i] < m_required_sizes[i]) {
                fits = false;
                break;
            }
        }
        if (!fits) {
            continue;
        }
        m_current_pool = it->pool;
        m_current_sizes = it->sizes;
        m_current_max_sets = it->max_sets;
        m_pending_pools.erase(it);
        reused = true;
        break;
    }

    if (reused) {
        EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
    } else {
        // Remove the just-retired entry? Keep it pending for later destruction.
        CreateDescriptorPool();
    }

    m_sets.clear();
    auto& fresh_batch = m_sets[layout];
    if (!Allocate(layout, fresh_batch)) {
        // Recompute required sizes from writes and grow aggressively.
        if (!writes.empty()) {
            std::array<uint32_t, 4> counts {};
            for (const auto& write: writes) {
                const size_t index = DescriptorTypeIndex(write.descriptorType);
                if (index < counts.size()) {
                    counts[index] += write.descriptorCount;
                }
            }
            for (size_t i = 0; i < m_required_sizes.size(); ++i) {
                m_required_sizes[i] = std::max(m_required_sizes[i], counts[i]);
                if (m_required_sizes[i] == m_current_sizes[i]) {
                    m_required_sizes[i] = m_current_sizes[i] * 2u + 1u;
                }
            }
        } else {
            for (size_t i = 0; i < m_required_sizes.size(); ++i) {
                m_required_sizes[i] = m_current_sizes[i] * 2u + 1u;
            }
        }
        for (size_t attempt = 0; attempt < 4; ++attempt) {
            m_pending_pools.push_back({m_current_pool, m_master_semaphore.CurrentTick(),
                                       m_current_sizes, m_current_max_sets});
            m_current_pool = nullptr;
            CreateDescriptorPool();
            m_sets.clear();
            auto& retry_batch = m_sets[layout];
            if (Allocate(layout, retry_batch)) {
                return retry_batch.sets[--retry_batch.size];
            }
            for (size_t i = 0; i < m_required_sizes.size(); ++i) {
                m_required_sizes[i] = m_required_sizes[i] * 2u + 1u;
            }
        }
        auto& final_batch = m_sets[layout];
        EXIT_IF(!Allocate(layout, final_batch));
        return final_batch.sets[--final_batch.size];
    }
    return fresh_batch.sets[--fresh_batch.size];
}

bool DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, Batch& batch) {
    std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
    layouts.fill(layout);

    vk::DescriptorSetAllocateInfo allocate {};
    allocate.descriptorPool = m_current_pool;
    allocate.pSetLayouts    = layouts.data();

    for (;;) {
        allocate.descriptorSetCount = batch.allocation;
        const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
        if (result == vk::Result::eSuccess) {
            batch.size = batch.allocation;
            return true;
        }
        EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
                result != vk::Result::eErrorFragmentedPool);
        if (batch.allocation == 1) {
            return false;
        }
        batch.allocation /= 2;
    }
}

void DescriptorHeap::CreateDescriptorPool() {
    vk::DescriptorPoolCreateInfo create {};
    create.maxSets       = m_required_max_sets;
    std::array<vk::DescriptorPoolSize, 4> pool_sizes {};
    for (size_t i = 0; i < pool_sizes.size(); ++i) {
        pool_sizes[i].type            = PoolTypes[i];
        pool_sizes[i].descriptorCount  = m_required_sizes[i];
    }
    create.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    create.pPoolSizes    = pool_sizes.data();
    EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
            vk::Result::eSuccess);
    m_current_sizes = m_required_sizes;
    m_current_max_sets = m_required_max_sets;
}

} // namespace Libs::Graphics