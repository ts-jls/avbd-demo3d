/*
 * Minimal persistent worker pool for per-frame data parallelism.
 *
 * Spawning std::threads per phase costs tens of microseconds per thread on
 * Windows; with several parallel phases per simulation step that overhead
 * exceeds the work being parallelized. This pool starts its workers once and
 * hands out [begin, end) chunks from an atomic cursor; the calling thread
 * participates and returns as soon as all chunks have completed — a worker
 * that is slow to wake never blocks completion (it finds the cursor
 * exhausted, or the job already cleared, and goes back to sleep).
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class WorkerPool
{
public:
    static WorkerPool &instance()
    {
        static WorkerPool pool;
        return pool;
    }

    // Runs fn(begin, end) over [0, count) in chunks. Serial when the range is
    // small or the pool has no workers.
    void parallelFor(size_t count, size_t chunkSize, const std::function<void(size_t, size_t)> &fn)
    {
        if (count == 0)
            return;
        if (workers.empty() || count <= chunkSize)
        {
            fn(0, count);
            return;
        }

        size_t totalChunks = (count + chunkSize - 1) / chunkSize;
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobFn = &fn;
            jobCount = count;
            jobChunk = chunkSize;
            jobCursor.store(0, std::memory_order_relaxed);
            completedChunks.store(0, std::memory_order_relaxed);
            generation++;
        }
        wakeCondition.notify_all();

        runChunks(fn, count, chunkSize);

        // All chunks done and no worker still inside the job loop. Chunks are
        // short, so a yield spin outperforms a condition variable here.
        while (completedChunks.load(std::memory_order_acquire) < totalChunks ||
               activeWorkers.load(std::memory_order_acquire) != 0)
            std::this_thread::yield();

        std::lock_guard<std::mutex> lock(mutex);
        jobFn = nullptr;
    }

    ~WorkerPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            generation++;
        }
        wakeCondition.notify_all();
        for (std::thread &t : workers)
            t.join();
    }

private:
    WorkerPool()
    {
        unsigned int n = std::thread::hardware_concurrency();
        unsigned int workerCount = n > 8 ? 7 : (n > 1 ? n - 1 : 0);
        for (unsigned int t = 0; t < workerCount; ++t)
            workers.emplace_back([this] { workerLoop(); });
    }

    void runChunks(const std::function<void(size_t, size_t)> &fn, size_t count, size_t chunkSize)
    {
        for (;;)
        {
            size_t begin = jobCursor.fetch_add(chunkSize, std::memory_order_relaxed);
            if (begin >= count)
                return;
            size_t end = begin + chunkSize < count ? begin + chunkSize : count;
            fn(begin, end);
            completedChunks.fetch_add(1, std::memory_order_release);
        }
    }

    void workerLoop()
    {
        uint64_t seenGeneration = 0;
        for (;;)
        {
            const std::function<void(size_t, size_t)> *fn = nullptr;
            size_t count = 0, chunkSize = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wakeCondition.wait(lock, [&] { return stopping || generation != seenGeneration; });
                seenGeneration = generation;
                if (stopping)
                    return;
                // jobFn is cleared once the job completes, so a late-waking
                // worker simply goes back to sleep.
                fn = jobFn;
                count = jobCount;
                chunkSize = jobChunk;
                if (fn)
                    activeWorkers.fetch_add(1, std::memory_order_acq_rel);
            }
            if (fn)
            {
                runChunks(*fn, count, chunkSize);
                activeWorkers.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }

    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable wakeCondition;
    const std::function<void(size_t, size_t)> *jobFn = nullptr;
    size_t jobCount = 0;
    size_t jobChunk = 0;
    std::atomic<size_t> jobCursor{0};
    std::atomic<size_t> completedChunks{0};
    std::atomic<int> activeWorkers{0};
    uint64_t generation = 0;
    bool stopping = false;
};
