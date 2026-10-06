#include "infer/server/web_demo.hpp"

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace infer {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Directory of the running executable (loadgen is built next to it).
std::string exe_dir() {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return ".";
    std::string path(buf, static_cast<std::size_t>(n));
    return path.substr(0, path.rfind('/'));
}

int query_int(std::string_view target, std::string_view key, int fallback) {
    const std::size_t q = target.find('?');
    if (q == std::string_view::npos) return fallback;
    std::string_view qs = target.substr(q + 1);
    while (!qs.empty()) {
        const std::size_t amp = qs.find('&');
        std::string_view kv = qs.substr(0, amp);
        if (kv.size() > key.size() && kv.substr(0, key.size()) == key && kv[key.size()] == '=') {
            return std::atoi(std::string(kv.substr(key.size() + 1)).c_str());
        }
        if (amp == std::string_view::npos) break;
        qs.remove_prefix(amp + 1);
    }
    return fallback;
}

}  // namespace

WebDemo::WebDemo(WebDemoOptions opts) : opts_(std::move(opts)) {
    index_html_ = read_file(opts_.web_dir + "/index.html");
    labels_json_ = read_file(opts_.web_dir + "/labels.json");
    loadgen_path_ = exe_dir() + "/loadgen";
}

WebDemo::~WebDemo() {
    std::lock_guard lock(mu_);
    if (child_ > 0) {
        ::kill(child_, SIGTERM);
        ::waitpid(child_, nullptr, 0);
        child_ = -1;
    }
}

void WebDemo::install(net::HttpServer& server) {
    server.route("GET", "/", [this](net::HttpRequest&&, net::Responder r) {
        r.send(net::HttpResponse{200, "text/html; charset=utf-8", index_html_, {}});
    });
    server.route("GET", "/labels.json", [this](net::HttpRequest&&, net::Responder r) {
        net::HttpResponse resp = net::HttpResponse::json(200, labels_json_);
        resp.extra_headers.emplace_back("Cache-Control", "max-age=3600");
        r.send(std::move(resp));
    });
    server.route("GET", "/info", [this](net::HttpRequest&&, net::Responder r) {
        std::lock_guard lock(mu_);
        std::string body = opts_.info_json;
        body.pop_back();  // drop the closing brace to append the load status
        body += "," + load_status_locked() + "}";
        r.send(net::HttpResponse::json(200, body));
    });
    server.route("POST", "/demo/load", [this](net::HttpRequest&& req, net::Responder r) {
        const int connections = query_int(req.target, "connections", 64);
        const int seconds = query_int(req.target, "seconds", 15);
        if (connections < 1 || connections > 256 || seconds < 1 || seconds > 60) {
            r.send(net::HttpResponse::text(400, "connections must be 1-256, seconds 1-60\n"));
            return;
        }
        r.send(net::HttpResponse::json(200, start_load(connections, seconds)));
    });
}

void WebDemo::reap_locked() {
    if (child_ > 0 && ::waitpid(child_, nullptr, WNOHANG) == child_) child_ = -1;
}

std::string WebDemo::load_status_locked() {
    reap_locked();
    double remaining = 0;
    if (child_ > 0) {
        remaining = std::max(0.0, std::chrono::duration<double>(load_until_ - std::chrono::steady_clock::now()).count());
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf), "\"load_running\":%s,\"load_connections\":%d,\"load_remaining_s\":%.1f",
                  child_ > 0 ? "true" : "false", child_ > 0 ? load_connections_ : 0, remaining);
    return buf;
}

std::string WebDemo::start_load(int connections, int seconds) {
    std::lock_guard lock(mu_);
    reap_locked();
    if (child_ > 0) return "{\"started\":false,\"reason\":\"a load test is already running\"," + load_status_locked() + "}";

    std::vector<std::string> args = {loadgen_path_, "--port", std::to_string(opts_.port), "--mode", "closed",
                                     "--connections", std::to_string(connections), "--duration",
                                     std::to_string(seconds), "--warmup", "0"};
    if (!opts_.loadgen_input.empty()) {
        args.push_back("--input");
        args.push_back(opts_.loadgen_input);
    }
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);

    // The server blocks SIGINT/SIGTERM in all threads; give the child a clean
    // signal mask so it can still be stopped, and silence its output.
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    sigset_t empty;
    sigemptyset(&empty);
    posix_spawnattr_setsigmask(&attr, &empty);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSIGMASK);
    posix_spawn_file_actions_t files;
    posix_spawn_file_actions_init(&files);
    posix_spawn_file_actions_addopen(&files, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);

    pid_t pid = -1;
    const int rc = ::posix_spawn(&pid, loadgen_path_.c_str(), &files, &attr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&files);
    posix_spawnattr_destroy(&attr);
    if (rc != 0) {
        return "{\"started\":false,\"reason\":\"could not start " + loadgen_path_ + "\"}";
    }
    child_ = pid;
    load_connections_ = connections;
    load_until_ = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    return "{\"started\":true," + load_status_locked() + "}";
}

}  // namespace infer
