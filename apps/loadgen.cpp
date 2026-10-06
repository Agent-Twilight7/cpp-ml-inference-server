// HTTP load generator for the inference server.
//
// Two modes:
//   closed  N connections; each sends its next request as soon as the previous
//           response arrives. Measures capacity at a given concurrency.
//   open    Requests are *scheduled* at a target rate (Poisson arrivals) and
//           spread over N connections. Latency is measured from each request's
//           scheduled time, not from when a connection got around to sending
//           it, so a stalled server is charged for the requests that queued
//           up behind the stall. This avoids "coordinated omission", which
//           makes closed-loop tools under-report tail latency.
//
// Usage:
//   loadgen [--host 127.0.0.1] [--port 8080] [--mode closed|open]
//           [--connections 64] [--rate 1000] [--duration 10] [--warmup 2]
//           [--input models/ref_input.bin] [--bytes 602112] [--path /v1/infer]
//           [--json]

#include "infer/metrics/histogram.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string host = "127.0.0.1";
    std::uint16_t port = 8080;
    std::string mode = "closed";
    int connections = 64;
    double rate = 1000.0;  // open mode, requests/s
    double duration = 10.0;
    double warmup = 2.0;
    std::string input;  // file whose first `bytes` bytes are the request body
    std::size_t bytes = 3 * 224 * 224 * sizeof(float);
    std::string path = "/v1/infer";
    bool json = false;
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--host H] [--port N] [--mode closed|open] [--connections N] [--rate R]\n"
                 "          [--duration S] [--warmup S] [--input FILE] [--bytes N] [--path P] [--json]\n",
                 argv0);
    std::exit(2);
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string f = argv[i];
        if (f == "--json") {
            o.json = true;
            continue;
        }
        if (f == "-h" || f == "--help" || i + 1 >= argc) usage(argv[0]);
        const std::string v = argv[++i];
        if (f == "--host") o.host = v;
        else if (f == "--port") o.port = static_cast<std::uint16_t>(std::stoi(v));
        else if (f == "--mode") o.mode = v;
        else if (f == "--connections") o.connections = std::stoi(v);
        else if (f == "--rate") o.rate = std::stod(v);
        else if (f == "--duration") o.duration = std::stod(v);
        else if (f == "--warmup") o.warmup = std::stod(v);
        else if (f == "--input") o.input = v;
        else if (f == "--bytes") o.bytes = std::stoul(v);
        else if (f == "--path") o.path = v;
        else usage(argv[0]);
    }
    if (o.mode != "closed" && o.mode != "open") usage(argv[0]);
    return o;
}

// One keep-alive HTTP/1.1 connection with blocking I/O.
class Connection {
public:
    Connection(const std::string& host, std::uint16_t port) : host_(host), port_(port) {}
    ~Connection() { close_fd(); }

    struct Result {
        int status = 0;  // 0 = transport error
        int batch_size = 0;
    };

    // Sends `request` and reads one response. Reconnects once on a broken connection.
    Result roundtrip(const std::string& request) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (fd_ < 0 && !connect_fd()) return {};
            if (send_all(request)) {
                Result r = read_response();
                if (r.status != 0) return r;
            }
            close_fd();  // stale keep-alive connection: retry on a fresh one
        }
        return {};
    }

private:
    bool connect_fd() {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) return false;
        const int one = 1;
        ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        timeval tv{.tv_sec = 30, .tv_usec = 0};
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        ::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);
        if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            close_fd();
            return false;
        }
        buf_.clear();
        return true;
    }

    void close_fd() {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
    }

    bool send_all(const std::string& data) {
        std::size_t off = 0;
        while (off < data.size()) {
            const ssize_t n = ::send(fd_, data.data() + off, data.size() - off, MSG_NOSIGNAL);
            if (n <= 0) return false;
            off += static_cast<std::size_t>(n);
        }
        return true;
    }

    bool fill() {
        char tmp[16384];
        const ssize_t n = ::recv(fd_, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf_.append(tmp, static_cast<std::size_t>(n));
        return true;
    }

    Result read_response() {
        std::size_t head_end;
        while ((head_end = buf_.find("\r\n\r\n")) == std::string::npos) {
            if (!fill()) return {};
        }
        Result r;
        if (buf_.size() < 12) return {};
        r.status = std::atoi(buf_.c_str() + 9);
        std::size_t len = 0;
        if (auto pos = buf_.find("Content-Length: "); pos != std::string::npos && pos < head_end) {
            len = std::strtoul(buf_.c_str() + pos + 16, nullptr, 10);
        }
        const std::size_t total = head_end + 4 + len;
        while (buf_.size() < total) {
            if (!fill()) return {};
        }
        if (auto pos = buf_.find("\"batch_size\":", head_end); pos != std::string::npos && pos < total) {
            r.batch_size = std::atoi(buf_.c_str() + pos + 13);
        }
        buf_.erase(0, total);
        return r;
    }

    std::string host_;
    std::uint16_t port_;
    int fd_ = -1;
    std::string buf_;
};

struct Stats {
    infer::metrics::LatencyHistogram latency_us;
    std::atomic<std::uint64_t> batch_size_sum{0};
    std::atomic<std::uint64_t> batch_size_n{0};
    std::mutex mu;
    std::map<int, std::uint64_t> status_counts;  // 0 = transport error

    void record(const Connection::Result& r, std::uint64_t latency_us_value) {
        if (r.status == 200) {
            latency_us.record(latency_us_value);
            if (r.batch_size > 0) {
                batch_size_sum.fetch_add(static_cast<std::uint64_t>(r.batch_size), std::memory_order_relaxed);
                batch_size_n.fetch_add(1, std::memory_order_relaxed);
            }
        }
        std::lock_guard lock(mu);
        ++status_counts[r.status];
    }
};

std::uint64_t micros(Clock::duration d) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);

    // ---- Request body ----
    std::string body(opt.bytes, '\0');
    if (!opt.input.empty()) {
        std::ifstream f(opt.input, std::ios::binary);
        if (!f.read(body.data(), static_cast<std::streamsize>(body.size()))) {
            std::fprintf(stderr, "cannot read %zu bytes from %s\n", opt.bytes, opt.input.c_str());
            return 1;
        }
    } else {
        std::mt19937 rng(1);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (std::size_t i = 0; i + sizeof(float) <= body.size(); i += sizeof(float)) {
            const float v = nd(rng);
            std::memcpy(body.data() + i, &v, sizeof(float));
        }
    }
    const std::string request = "POST " + opt.path + " HTTP/1.1\r\nHost: " + opt.host +
                                "\r\nContent-Type: application/octet-stream\r\nContent-Length: " +
                                std::to_string(body.size()) + "\r\n\r\n" + body;

    Stats stats;
    const auto start = Clock::now() + std::chrono::milliseconds(200);  // let threads connect first
    const auto measure_from = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(opt.warmup));
    const auto end = measure_from + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(opt.duration));

    // ---- Open loop: precomputed Poisson schedule (offsets from `start`) ----
    std::vector<Clock::duration> schedule;
    if (opt.mode == "open") {
        std::mt19937_64 rng(12345);
        std::exponential_distribution<double> gap(opt.rate);
        double t = 0.0;
        const double total = opt.warmup + opt.duration;
        while (t < total) {
            schedule.push_back(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(t)));
            t += gap(rng);
        }
    }
    std::atomic<std::size_t> next_index{0};
    std::atomic<std::uint64_t> late_starts{0};  // open loop: no free connection at the scheduled time

    std::vector<std::thread> threads;
    for (int c = 0; c < opt.connections; ++c) {
        threads.emplace_back([&] {
            Connection conn(opt.host, opt.port);
            if (opt.mode == "closed") {
                std::this_thread::sleep_until(start);
                for (;;) {
                    const auto t0 = Clock::now();
                    if (t0 >= end) break;
                    const auto r = conn.roundtrip(request);
                    const auto t1 = Clock::now();
                    if (t0 >= measure_from) stats.record(r, micros(t1 - t0));
                }
            } else {
                for (;;) {
                    const std::size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
                    if (i >= schedule.size()) break;
                    const auto intended = start + schedule[i];
                    const auto now = Clock::now();
                    if (now < intended) {
                        std::this_thread::sleep_until(intended);
                    } else if (now - intended > std::chrono::milliseconds(1)) {
                        late_starts.fetch_add(1, std::memory_order_relaxed);
                    }
                    const auto r = conn.roundtrip(request);
                    const auto t1 = Clock::now();
                    // Latency from the *intended* send time (coordinated-omission safe).
                    if (intended >= measure_from) stats.record(r, micros(t1 - intended));
                }
            }
        });
    }
    for (auto& t : threads) t.join();

    // ---- Report ----
    const double measured_s =
        opt.mode == "closed"
            ? opt.duration
            : std::max(std::chrono::duration<double>(Clock::now() - measure_from).count(), opt.duration);
    const auto snap = stats.latency_us.snapshot();
    std::uint64_t errors = 0, total = 0;
    for (const auto& [status, n] : stats.status_counts) {
        total += n;
        if (status != 200) errors += n;
    }
    const double ok_rps = static_cast<double>(snap.count) / measured_s;
    const double avg_batch = stats.batch_size_n ? static_cast<double>(stats.batch_size_sum) /
                                                      static_cast<double>(stats.batch_size_n)
                                                : 0.0;
    auto ms = [&](double q) { return snap.percentile(q) / 1000.0; };

    if (opt.json) {
        std::printf("{\"mode\":\"%s\",\"connections\":%d,\"target_rps\":%.1f,\"duration_s\":%.2f,"
                    "\"requests\":%llu,\"ok\":%llu,\"errors\":%llu,\"throughput_rps\":%.1f,"
                    "\"mean_ms\":%.3f,\"p50_ms\":%.3f,\"p90_ms\":%.3f,\"p99_ms\":%.3f,\"p999_ms\":%.3f,"
                    "\"max_ms\":%.3f,\"avg_batch\":%.2f,\"late_starts\":%llu,\"status\":{",
                    opt.mode.c_str(), opt.connections, opt.mode == "open" ? opt.rate : 0.0, measured_s,
                    static_cast<unsigned long long>(total), static_cast<unsigned long long>(snap.count),
                    static_cast<unsigned long long>(errors), ok_rps, snap.mean() / 1000.0, ms(0.5), ms(0.9), ms(0.99),
                    ms(0.999), snap.max() / 1000.0, avg_batch, static_cast<unsigned long long>(late_starts.load()));
        bool first = true;
        for (const auto& [status, n] : stats.status_counts) {
            std::printf("%s\"%d\":%llu", first ? "" : ",", status, static_cast<unsigned long long>(n));
            first = false;
        }
        std::printf("}}\n");
    } else {
        std::printf("mode=%s connections=%d%s duration=%.1fs\n", opt.mode.c_str(), opt.connections,
                    opt.mode == "open" ? (" rate=" + std::to_string(static_cast<int>(opt.rate)) + "/s").c_str() : "",
                    measured_s);
        std::printf("requests=%llu ok=%llu errors=%llu  throughput=%.1f req/s  avg_batch=%.2f\n",
                    static_cast<unsigned long long>(total), static_cast<unsigned long long>(snap.count),
                    static_cast<unsigned long long>(errors), ok_rps, avg_batch);
        std::printf("latency ms: mean=%.2f p50=%.2f p90=%.2f p99=%.2f p99.9=%.2f max=%.2f\n", snap.mean() / 1000.0,
                    ms(0.5), ms(0.9), ms(0.99), ms(0.999), snap.max() / 1000.0);
        if (opt.mode == "open") {
            std::printf("late starts (no free connection on schedule): %llu\n",
                        static_cast<unsigned long long>(late_starts.load()));
        }
        for (const auto& [status, n] : stats.status_counts) {
            if (status != 200) std::printf("  status %d: %llu\n", status, static_cast<unsigned long long>(n));
        }
    }
    return errors == 0 ? 0 : 3;
}
