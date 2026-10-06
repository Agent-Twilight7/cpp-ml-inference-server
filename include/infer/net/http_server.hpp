#pragma once

#include "infer/net/http.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infer::net {

class Reactor;

// Handle for answering one request. Cheap to copy. send() may be called from
// any thread and must be called exactly once per request: the connection reads
// no further requests until it is. If the client has disconnected in the
// meantime the response is silently dropped.
//
// A Responder must not outlive the HttpServer that issued it.
class Responder {
public:
    void send(HttpResponse resp) const;

private:
    friend class Reactor;
    Responder(Reactor* reactor, std::uint64_t conn_id) : reactor_(reactor), conn_id_(conn_id) {}

    Reactor* reactor_;
    std::uint64_t conn_id_;
};

// Called on a reactor thread. Long-running work should be handed off to another
// thread (keeping the Responder) so the reactor can keep serving other sockets.
using Handler = std::function<void(HttpRequest&& req, Responder responder)>;

struct HttpServerOptions {
    std::string host = "0.0.0.0";
    std::uint16_t port = 8080;  // 0 picks a free port; see HttpServer::port()
    int num_reactors = 4;
    HttpParser::Limits limits{};
};

// Multi-reactor HTTP/1.1 server. Each reactor thread owns an epoll instance
// and its own listening socket on the same port (SO_REUSEPORT), so the kernel
// spreads new connections across reactors with no shared accept lock.
class HttpServer {
public:
    explicit HttpServer(HttpServerOptions opts);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Registers a handler for an exact (method, path) pair. Call before start().
    void route(std::string method, std::string path, Handler handler);

    // Binds the sockets and launches the reactor threads. Throws on failure.
    void start();
    // Stops all reactors and joins their threads. Idempotent.
    void stop();

    // The bound port (resolves port 0 after start()).
    std::uint16_t port() const { return bound_port_; }

    // Thread-safe; for metrics.
    std::size_t active_connections() const;
    std::uint64_t accepted_connections() const;

private:
    friend class Reactor;
    void dispatch(HttpRequest&& req, Responder responder) const;

    HttpServerOptions opts_;
    std::uint16_t bound_port_ = 0;
    // path -> [(method, handler)]
    std::unordered_map<std::string, std::vector<std::pair<std::string, Handler>>> routes_;
    std::vector<std::unique_ptr<Reactor>> reactors_;
    std::vector<std::thread> threads_;
};

}  // namespace infer::net
