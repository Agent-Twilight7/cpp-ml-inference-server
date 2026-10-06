#pragma once

#include "infer/net/http_server.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <sys/types.h>

namespace infer {

struct WebDemoOptions {
    std::string web_dir;       // contains index.html and labels.json
    std::string info_json;     // served at GET /info (model, device, batching config)
    std::uint16_t port = 8080; // where the spawned load generator should connect
    std::string loadgen_input; // request body file for the load generator ("" = random)
};

// Optional browser front end for demos (enabled with --web DIR):
//   GET  /             the single-page UI (web/index.html)
//   GET  /labels.json  ImageNet class names
//   GET  /info         server configuration as JSON
//   POST /demo/load?connections=N&seconds=S
//                      starts the bundled `loadgen` against this server.
//                      Browsers open only ~6 connections per site, too few to
//                      show batching, so the load comes from a separate
//                      process instead. One run at a time.
class WebDemo {
public:
    explicit WebDemo(WebDemoOptions opts);
    ~WebDemo();  // stops a running load generator

    WebDemo(const WebDemo&) = delete;
    WebDemo& operator=(const WebDemo&) = delete;

    void install(net::HttpServer& server);

private:
    std::string start_load(int connections, int seconds);  // returns JSON
    std::string load_status_locked();                      // JSON fragment; mu_ held
    void reap_locked();

    WebDemoOptions opts_;
    std::string index_html_;
    std::string labels_json_;
    std::string loadgen_path_;

    std::mutex mu_;
    pid_t child_ = -1;
    std::chrono::steady_clock::time_point load_until_{};
    int load_connections_ = 0;
};

}  // namespace infer
