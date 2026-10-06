#include "infer/net/http_server.hpp"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace infer::net;

namespace {

// Minimal blocking HTTP client over a raw socket, with a receive timeout so a
// broken server fails the test instead of hanging it.
class Client {
public:
    explicit Client(std::uint16_t port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        timeval tv{.tv_sec = 5, .tv_usec = 0};
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        connected_ = ::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    }
    ~Client() { ::close(fd_); }

    bool connected() const { return connected_; }

    void send_raw(const std::string& data) {
        std::size_t off = 0;
        while (off < data.size()) {
            const ssize_t n = ::send(fd_, data.data() + off, data.size() - off, MSG_NOSIGNAL);
            if (n <= 0) return;
            off += static_cast<std::size_t>(n);
        }
    }

    struct Response {
        int status = 0;
        std::string head;
        std::string body;
    };

    // Reads exactly one response (uses Content-Length). status 0 = closed/timeout.
    Response read_response() {
        Response r;
        std::size_t head_end;
        while ((head_end = buf_.find("\r\n\r\n")) == std::string::npos) {
            if (!fill()) return r;
        }
        r.head = buf_.substr(0, head_end);
        r.status = std::stoi(r.head.substr(9, 3));
        std::size_t len = 0;
        if (auto pos = r.head.find("Content-Length: "); pos != std::string::npos) len = std::stoul(r.head.substr(pos + 16));
        while (buf_.size() < head_end + 4 + len) {
            if (!fill()) return Response{};
        }
        r.body = buf_.substr(head_end + 4, len);
        buf_.erase(0, head_end + 4 + len);
        return r;
    }

    // True if the server closed the connection (recv returns 0).
    bool closed_by_peer() {
        char c;
        return buf_.empty() && ::recv(fd_, &c, 1, 0) == 0;
    }

private:
    bool fill() {
        char tmp[4096];
        const ssize_t n = ::recv(fd_, tmp, sizeof(tmp), 0);
        if (n <= 0) return false;
        buf_.append(tmp, static_cast<std::size_t>(n));
        return true;
    }

    int fd_;
    bool connected_ = false;
    std::string buf_;
};

class HttpServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        server_ = std::make_unique<HttpServer>(HttpServerOptions{.host = "127.0.0.1", .port = 0, .num_reactors = 2});
        server_->route("GET", "/hello", [](HttpRequest&&, Responder r) { r.send(HttpResponse::text(200, "hi")); });
        server_->route("POST", "/echo", [](HttpRequest&& req, Responder r) {
            r.send(HttpResponse::text(200, std::string(req.body)));  // copy: body dies with the handler
        });
        // Completes from a different thread, like the batcher will.
        server_->route("GET", "/async", [this](HttpRequest&&, Responder r) {
            std::lock_guard lock(workers_mu_);
            workers_.emplace_back([r] {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                r.send(HttpResponse::text(200, "later"));
            });
        });
        server_->start();
    }

    void TearDown() override {
        {
            std::lock_guard lock(workers_mu_);
            for (auto& t : workers_) t.join();
        }
        server_->stop();
    }

    std::unique_ptr<HttpServer> server_;
    std::mutex workers_mu_;
    std::vector<std::thread> workers_;
};

}  // namespace

TEST_F(HttpServerTest, ServesGet) {
    Client c(server_->port());
    ASSERT_TRUE(c.connected());
    c.send_raw("GET /hello HTTP/1.1\r\nHost: x\r\n\r\n");
    auto r = c.read_response();
    EXPECT_EQ(r.status, 200);
    EXPECT_EQ(r.body, "hi");
}

TEST_F(HttpServerTest, KeepAliveServesSequentialRequests) {
    Client c(server_->port());
    for (int i = 0; i < 3; ++i) {
        c.send_raw("POST /echo HTTP/1.1\r\nContent-Length: 1\r\n\r\n" + std::to_string(i));
        auto r = c.read_response();
        ASSERT_EQ(r.status, 200);
        EXPECT_EQ(r.body, std::to_string(i));
    }
}

TEST_F(HttpServerTest, PipelinedRequestsAnsweredInOrder) {
    Client c(server_->port());
    c.send_raw(
        "GET /async HTTP/1.1\r\n\r\n"
        "POST /echo HTTP/1.1\r\nContent-Length: 3\r\n\r\nabc"
        "GET /hello HTTP/1.1\r\n\r\n");
    EXPECT_EQ(c.read_response().body, "later");
    EXPECT_EQ(c.read_response().body, "abc");
    EXPECT_EQ(c.read_response().body, "hi");
}

TEST_F(HttpServerTest, LargeBodyArrivingInPieces) {
    Client c(server_->port());
    const std::string body(1 << 20, 'z');  // 1 MiB
    c.send_raw("POST /echo HTTP/1.1\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n");
    for (std::size_t off = 0; off < body.size(); off += 100000) c.send_raw(body.substr(off, 100000));
    auto r = c.read_response();
    ASSERT_EQ(r.status, 200);
    EXPECT_EQ(r.body.size(), body.size());
    EXPECT_EQ(r.body, body);
}

TEST_F(HttpServerTest, SendsContinueForExpectHeader) {
    Client c(server_->port());
    c.send_raw("POST /echo HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\n");
    auto interim = c.read_response();
    EXPECT_EQ(interim.status, 100);
    c.send_raw("ok");
    auto r = c.read_response();
    EXPECT_EQ(r.status, 200);
    EXPECT_EQ(r.body, "ok");
}

TEST_F(HttpServerTest, NotFoundAndMethodNotAllowed) {
    Client c(server_->port());
    c.send_raw("GET /nope HTTP/1.1\r\n\r\n");
    EXPECT_EQ(c.read_response().status, 404);
    c.send_raw("DELETE /hello HTTP/1.1\r\n\r\n");
    auto r = c.read_response();
    EXPECT_EQ(r.status, 405);
    EXPECT_NE(r.head.find("Allow: GET"), std::string::npos);
}

TEST_F(HttpServerTest, MalformedRequestGets400AndClose) {
    Client c(server_->port());
    c.send_raw("NONSENSE\r\n\r\n");
    auto r = c.read_response();
    EXPECT_EQ(r.status, 400);
    EXPECT_NE(r.head.find("Connection: close"), std::string::npos);
    EXPECT_TRUE(c.closed_by_peer());
}

TEST_F(HttpServerTest, ConnectionCloseHonored) {
    Client c(server_->port());
    c.send_raw("GET /hello HTTP/1.1\r\nConnection: close\r\n\r\n");
    EXPECT_EQ(c.read_response().status, 200);
    EXPECT_TRUE(c.closed_by_peer());
}

TEST_F(HttpServerTest, ClientDisconnectWhileInFlightIsHarmless) {
    {
        Client c(server_->port());
        c.send_raw("GET /async HTTP/1.1\r\n\r\n");
    }  // closes before the async response is ready
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Client c2(server_->port());
    c2.send_raw("GET /hello HTTP/1.1\r\n\r\n");
    EXPECT_EQ(c2.read_response().status, 200);
}

TEST_F(HttpServerTest, ManyConcurrentClients) {
    constexpr int kClients = 32;
    constexpr int kRequestsEach = 20;
    std::atomic<int> ok{0};
    std::vector<std::thread> clients;
    for (int i = 0; i < kClients; ++i) {
        clients.emplace_back([&, i] {
            Client c(server_->port());
            for (int j = 0; j < kRequestsEach; ++j) {
                const std::string body = std::to_string(i) + ":" + std::to_string(j);
                c.send_raw("POST /echo HTTP/1.1\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body);
                if (c.read_response().body == body) ok.fetch_add(1);
            }
        });
    }
    for (auto& t : clients) t.join();
    EXPECT_EQ(ok.load(), kClients * kRequestsEach);
}
