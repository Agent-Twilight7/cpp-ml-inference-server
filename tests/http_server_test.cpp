#include "infer/net/http_server.hpp"

#include <gtest/gtest.h>

#include "test_client.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace infer::net;

namespace {

using testing_support::Client;

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
        // Slower async route, for drain tests.
        server_->route("GET", "/slow", [this](HttpRequest&&, Responder r) {
            std::lock_guard lock(workers_mu_);
            workers_.emplace_back([r] {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                r.send(HttpResponse::text(200, "slow"));
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

TEST_F(HttpServerTest, GracefulDrainFinishesInFlightAndRefusesNew) {
    Client idle(server_->port());
    ASSERT_EQ(idle.request("GET", "/hello").status, 200);  // now an idle keep-alive connection

    Client busy(server_->port());
    busy.send_raw("GET /slow HTTP/1.1\r\n\r\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));  // request is with the handler

    server_->begin_drain();

    auto r = busy.read_response();  // the in-flight request still completes...
    EXPECT_EQ(r.status, 200);
    EXPECT_EQ(r.body, "slow");
    EXPECT_NE(r.head.find("Connection: close"), std::string::npos);  // ...as the connection's last
    EXPECT_TRUE(busy.closed_by_peer());
    EXPECT_TRUE(idle.closed_by_peer());  // idle connections are closed right away

    Client fresh(server_->port());
    EXPECT_FALSE(fresh.connected());  // listeners are closed: new connections refused
    EXPECT_TRUE(server_->wait_drained(std::chrono::seconds(2)));
}
