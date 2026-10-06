#include "infer/net/http_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <system_error>

namespace infer::net {
namespace {

// epoll_event.data.u64 tags. Connections use ids starting at kFirstConnId so a
// stale event for an already-closed connection simply fails the id lookup.
constexpr std::uint64_t kListenerTag = 0;
constexpr std::uint64_t kWakeTag = 1;
constexpr std::uint64_t kFirstConnId = 2;

constexpr int kMaxEvents = 256;
constexpr std::size_t kReadChunk = 64 * 1024;

[[noreturn]] void throw_errno(const char* what) { throw std::system_error(errno, std::generic_category(), what); }

int make_listener(const std::string& host, std::uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) throw_errno("socket");
    const int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one)) < 0) {
        ::close(fd);
        throw_errno("setsockopt(SO_REUSEPORT)");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        ::close(fd);
        throw std::invalid_argument("invalid IPv4 host: " + host);
    }
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 || ::listen(fd, SOMAXCONN) < 0) {
        const int err = errno;
        ::close(fd);
        throw std::system_error(err, std::generic_category(), "bind/listen on port " + std::to_string(port));
    }
    return fd;
}

std::uint16_t local_port(int fd) {
    sockaddr_in addr{};
    socklen_t len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) < 0) throw_errno("getsockname");
    return ntohs(addr.sin_port);
}

}  // namespace

// ---------------------------------------------------------------------------
// Reactor: one epoll loop, its listening socket, and the connections it owns.
// Everything except complete() and stop() runs on the reactor's own thread.
// ---------------------------------------------------------------------------

struct Connection {
    int fd = -1;
    std::uint64_t id = 0;
    std::string in;            // received, not yet parsed
    std::string out;           // serialized, not yet sent
    std::size_t out_off = 0;   // bytes of `out` already sent
    HttpParser parser;
    bool busy = false;         // a request is with a handler; don't parse the next one yet
    bool keep_alive = true;    // of the request being handled
    bool close_after_flush = false;
    bool read_paused = false;  // stopped reading because `in` hit its cap while busy

    explicit Connection(HttpParser::Limits limits) : parser(limits) {}
};

class Reactor {
public:
    Reactor(const HttpServer& server, int listen_fd, HttpParser::Limits limits)
        : server_(server), listen_fd_(listen_fd), limits_(limits) {
        epoll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd_ < 0) throw_errno("epoll_create1");
        wake_fd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (wake_fd_ < 0) throw_errno("eventfd");
        // Listener and wake fd are level-triggered: simple and never lose a wakeup.
        add_fd(listen_fd_, EPOLLIN, kListenerTag);
        add_fd(wake_fd_, EPOLLIN, kWakeTag);
    }

    ~Reactor() {
        for (auto& [id, conn] : conns_) ::close(conn->fd);
        ::close(listen_fd_);
        ::close(wake_fd_);
        ::close(epoll_fd_);
    }

    void run() {
        loop_thread_.store(std::this_thread::get_id());
        epoll_event events[kMaxEvents];
        while (!stopping_.load(std::memory_order_acquire)) {
            const int n = ::epoll_wait(epoll_fd_, events, kMaxEvents, -1);
            if (n < 0) {
                if (errno == EINTR) continue;
                std::perror("epoll_wait");
                break;
            }
            for (int i = 0; i < n; ++i) {
                const std::uint64_t tag = events[i].data.u64;
                if (tag == kListenerTag) {
                    on_accept();
                } else if (tag == kWakeTag) {
                    on_wake();
                } else {
                    on_conn_event(tag, events[i].events);
                }
            }
            drain_local();
        }
    }

    void stop() {
        stopping_.store(true, std::memory_order_release);
        wake();
    }

    // Thread-safe. Queues a response for connection `conn_id`.
    void complete(std::uint64_t conn_id, HttpResponse resp) {
        if (std::this_thread::get_id() == loop_thread_.load()) {
            // Called from inside a handler on this thread: defer to the end of
            // the event batch instead of re-entering connection code.
            local_pending_.emplace_back(conn_id, std::move(resp));
            return;
        }
        bool was_empty;
        {
            std::lock_guard lock(mu_);
            was_empty = pending_.empty();
            pending_.emplace_back(conn_id, std::move(resp));
        }
        // Only the first completion after a drain needs a syscall to wake the loop.
        if (was_empty) wake();
    }

private:
    void add_fd(int fd, std::uint32_t events, std::uint64_t tag) {
        epoll_event ev{};
        ev.events = events;
        ev.data.u64 = tag;
        if (::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) throw_errno("epoll_ctl(ADD)");
    }

    void wake() {
        const std::uint64_t one = 1;
        [[maybe_unused]] ssize_t r = ::write(wake_fd_, &one, sizeof(one));
    }

    void on_accept() {
        for (;;) {
            const int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
            if (fd < 0) {
                if (errno == EINTR) continue;
                if (errno != EAGAIN && errno != EWOULDBLOCK) std::perror("accept4");
                return;
            }
            const int one = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

            auto conn = std::make_unique<Connection>(limits_);
            conn->fd = fd;
            conn->id = next_id_++;
            // Edge-triggered: we always drain reads/writes until EAGAIN, and
            // EPOLLOUT only fires when a full socket buffer frees up.
            epoll_event ev{};
            ev.events = EPOLLIN | EPOLLOUT | EPOLLRDHUP | EPOLLET;
            ev.data.u64 = conn->id;
            if (::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
                std::perror("epoll_ctl(ADD conn)");
                ::close(fd);
                continue;
            }
            conns_.emplace(conn->id, std::move(conn));
        }
    }

    void on_wake() {
        std::uint64_t count;
        [[maybe_unused]] ssize_t r = ::read(wake_fd_, &count, sizeof(count));
        std::vector<std::pair<std::uint64_t, HttpResponse>> batch;
        {
            std::lock_guard lock(mu_);
            batch.swap(pending_);
        }
        for (auto& [id, resp] : batch) deliver(id, resp);
    }

    void drain_local() {
        // Delivering a response can parse and dispatch the next pipelined
        // request, whose handler may complete inline and append here again.
        while (!local_pending_.empty()) {
            std::vector<std::pair<std::uint64_t, HttpResponse>> batch;
            batch.swap(local_pending_);
            for (auto& [id, resp] : batch) deliver(id, resp);
        }
    }

    void on_conn_event(std::uint64_t id, std::uint32_t events) {
        Connection* c = find(id);
        if (!c) return;
        if (events & EPOLLERR) {
            close_conn(*c);
            return;
        }
        if ((events & EPOLLOUT) && !flush(*c)) return;
        if (events & (EPOLLIN | EPOLLRDHUP | EPOLLHUP)) {
            if (!read_input(*c)) return;
            process_input(*c);
        }
    }

    Connection* find(std::uint64_t id) {
        auto it = conns_.find(id);
        return it == conns_.end() ? nullptr : it->second.get();
    }

    // Reads until EAGAIN. Returns false if the connection was closed.
    bool read_input(Connection& c) {
        // While a request is in flight, buffer at most one more max-size request.
        const std::size_t cap = limits_.max_header_bytes + limits_.max_body_bytes;
        char buf[kReadChunk];
        for (;;) {
            if (c.busy && c.in.size() >= cap) {
                c.read_paused = true;  // resumed after the response is written
                return true;
            }
            const ssize_t n = ::recv(c.fd, buf, sizeof(buf), 0);
            if (n > 0) {
                c.in.append(buf, static_cast<std::size_t>(n));
            } else if (n == 0) {
                close_conn(c);  // peer closed; any in-flight response is dropped
                return false;
            } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                c.read_paused = false;
                return true;
            } else if (errno != EINTR) {
                close_conn(c);
                return false;
            }
        }
    }

    // Parses and dispatches buffered requests, one at a time.
    void process_input(Connection& c) {
        while (!c.busy && !c.in.empty()) {
            switch (c.parser.parse(c.in)) {
                case HttpParser::Status::kNeedMore:
                    if (c.parser.take_continue_flag()) {
                        c.out += "HTTP/1.1 100 Continue\r\n\r\n";
                        flush(c);
                    }
                    return;
                case HttpParser::Status::kError:
                    c.busy = true;  // stop parsing; this connection is done
                    write_response(c, HttpResponse::text(c.parser.error_status(), reason_phrase(c.parser.error_status())),
                                   /*keep_alive=*/false);
                    return;
                case HttpParser::Status::kComplete: {
                    const std::size_t used = c.parser.consumed();
                    HttpRequest req = c.parser.take_request();
                    c.busy = true;
                    c.keep_alive = req.keep_alive;
                    // req.body views c.in, so erase only after the handler
                    // returns. (A handler never closes the connection
                    // synchronously: responses are delivered later.)
                    server_.dispatch(std::move(req), Responder(this, c.id));
                    c.in.erase(0, used);
                    break;  // loop exits: busy until the handler responds
                }
            }
        }
    }

    void deliver(std::uint64_t id, const HttpResponse& resp) {
        Connection* c = find(id);
        if (!c || !c->busy) return;  // client went away (or a duplicate send)
        if (!write_response(*c, resp, c->keep_alive)) return;
        c->busy = false;
        // Resume: drain the socket if we paused it, then handle pipelined requests.
        if (c->read_paused && !read_input(*c)) return;
        process_input(*c);
    }

    // Returns false if the connection was closed.
    bool write_response(Connection& c, const HttpResponse& resp, bool keep_alive) {
        serialize(resp, keep_alive, c.out);
        if (!keep_alive) c.close_after_flush = true;
        return flush(c);
    }

    // Sends until done or EAGAIN. Returns false if the connection was closed.
    bool flush(Connection& c) {
        while (c.out_off < c.out.size()) {
            const ssize_t n = ::send(c.fd, c.out.data() + c.out_off, c.out.size() - c.out_off, MSG_NOSIGNAL);
            if (n > 0) {
                c.out_off += static_cast<std::size_t>(n);
            } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                return true;  // EPOLLOUT will call us again
            } else if (n < 0 && errno == EINTR) {
                continue;
            } else {
                close_conn(c);
                return false;
            }
        }
        c.out.clear();
        c.out_off = 0;
        if (c.close_after_flush) {
            close_conn(c);
            return false;
        }
        return true;
    }

    void close_conn(Connection& c) {
        ::close(c.fd);  // also removes it from the epoll set
        conns_.erase(c.id);
    }

    const HttpServer& server_;
    int listen_fd_;
    int epoll_fd_ = -1;
    int wake_fd_ = -1;
    HttpParser::Limits limits_;
    std::atomic<bool> stopping_{false};
    std::atomic<std::thread::id> loop_thread_{};

    std::unordered_map<std::uint64_t, std::unique_ptr<Connection>> conns_;
    std::uint64_t next_id_ = kFirstConnId;

    std::mutex mu_;
    std::vector<std::pair<std::uint64_t, HttpResponse>> pending_;        // from other threads; guarded by mu_
    std::vector<std::pair<std::uint64_t, HttpResponse>> local_pending_;  // from this thread
};

// ---------------------------------------------------------------------------

void Responder::send(HttpResponse resp) const { reactor_->complete(conn_id_, std::move(resp)); }

HttpServer::HttpServer(HttpServerOptions opts) : opts_(std::move(opts)) {
    if (opts_.num_reactors < 1) throw std::invalid_argument("num_reactors must be >= 1");
}

HttpServer::~HttpServer() { stop(); }

void HttpServer::route(std::string method, std::string path, Handler handler) {
    routes_[std::move(path)].emplace_back(std::move(method), std::move(handler));
}

void HttpServer::start() {
    if (!reactors_.empty()) throw std::logic_error("HttpServer already started");

    // The first bind resolves port 0; the rest join that port via SO_REUSEPORT.
    std::uint16_t port = opts_.port;
    for (int i = 0; i < opts_.num_reactors; ++i) {
        const int fd = make_listener(opts_.host, port);
        if (i == 0) port = local_port(fd);
        try {
            reactors_.push_back(std::make_unique<Reactor>(*this, fd, opts_.limits));
        } catch (...) {
            ::close(fd);
            reactors_.clear();
            throw;
        }
    }
    bound_port_ = port;

    for (auto& r : reactors_) threads_.emplace_back([reactor = r.get()] { reactor->run(); });
}

void HttpServer::stop() {
    for (auto& r : reactors_) r->stop();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
    reactors_.clear();
}

void HttpServer::dispatch(HttpRequest&& req, Responder responder) const {
    auto it = routes_.find(std::string(req.path()));
    if (it == routes_.end()) {
        responder.send(HttpResponse::text(404, "not found\n"));
        return;
    }
    for (const auto& [method, handler] : it->second) {
        if (method != req.method) continue;
        try {
            handler(std::move(req), responder);
        } catch (const std::exception& e) {
            // Only reached if the handler threw before handing the request off.
            responder.send(HttpResponse::text(500, std::string("internal error: ") + e.what() + "\n"));
        }
        return;
    }
    std::string allow;
    for (const auto& [method, handler] : it->second) allow += (allow.empty() ? "" : ", ") + method;
    HttpResponse resp = HttpResponse::text(405, "method not allowed\n");
    resp.extra_headers.emplace_back("Allow", allow);
    responder.send(std::move(resp));
}

}  // namespace infer::net
