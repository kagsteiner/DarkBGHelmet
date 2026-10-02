// Minimal thread pool with a blocking parallel_for.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace bg {

class ThreadPool {
 public:
  explicit ThreadPool(int threads = 0) { resize(threads); }
  ~ThreadPool() { stop(); }

  int size() const { return int(workers_.size()) + 1; }  // the caller also works

  void resize(int threads) {
    stop();
    if (threads <= 0) threads = int(std::thread::hardware_concurrency());
    quit_ = false;
    for (int i = 0; i < threads - 1; ++i) workers_.emplace_back([this, i] { loop(i + 1); });
  }

  // Runs fn(task_index, worker_index) for task_index in [0, n). Blocks until done.
  void parallel_for(int n, const std::function<void(int, int)>& fn) {
    if (n <= 0) return;
    if (workers_.empty() || n == 1) {
      for (int i = 0; i < n; ++i) fn(i, 0);
      return;
    }
    std::unique_lock<std::mutex> job_lock(job_mutex_);  // one job at a time
    {
      std::lock_guard<std::mutex> g(m_);
      fn_ = &fn;
      n_ = n;
      next_.store(0);
      pending_ = int(workers_.size());
      ++generation_;
    }
    cv_.notify_all();
    run(0);
    std::unique_lock<std::mutex> l(m_);
    done_cv_.wait(l, [&] { return pending_ == 0; });
    fn_ = nullptr;
  }

 private:
  void run(int worker) {
    for (;;) {
      int i = next_.fetch_add(1);
      if (i >= n_) break;
      (*fn_)(i, worker);
    }
  }
  void loop(int worker) {
    unsigned seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return quit_ || generation_ != seen; });
        if (quit_) return;
        seen = generation_;
      }
      run(worker);
      std::lock_guard<std::mutex> g(m_);
      if (--pending_ == 0) done_cv_.notify_all();
    }
  }
  void stop() {
    {
      std::lock_guard<std::mutex> g(m_);
      quit_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
    workers_.clear();
  }

  std::vector<std::thread> workers_;
  std::mutex m_, job_mutex_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int, int)>* fn_ = nullptr;
  int n_ = 0;
  std::atomic<int> next_{0};
  int pending_ = 0;
  unsigned generation_ = 0;
  bool quit_ = false;
};

}  // namespace bg
