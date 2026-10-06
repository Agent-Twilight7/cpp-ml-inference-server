// End-to-end tests of the HTTP API (infer, stats, metrics) over real sockets,
// with a fake model so they run without ONNX Runtime or a GPU.

#include "infer/server/inference_service.hpp"

#include <gtest/gtest.h>

#include "fake_backend.hpp"
#include "test_client.hpp"

#include <chrono>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using testing_support::Client;
using testing_support::FakeBackend;

namespace {

std::string sample_body(float v) {
    std::vector<float> x(FakeBackend::kIn, v);
    return std::string(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(float));
}

class InferenceServiceTest : public ::testing::Test {
protected:
    void start(std::size_t pool_blocks = 64, std::chrono::microseconds compute_delay = 0us) {
        shared_.compute_delay = compute_delay;
        batcher_.emplace(
            infer::BatcherOptions{.max_batch_size = 8, .max_queue_delay = 1ms, .num_workers = 1, .metrics = &bm_},
            [this] { return std::make_unique<FakeBackend>(shared_); });
        batcher_->start();
        pool_.emplace(FakeBackend::kIn, pool_blocks);
        service_.emplace(*batcher_, *pool_, &bm_,
                         infer::InferenceServiceOptions{.top_k = 2, .tick = std::chrono::milliseconds(50)});
        server_.emplace(infer::net::HttpServerOptions{.host = "127.0.0.1", .port = 0, .num_reactors = 1});
        service_->install(*server_);
        service_->start();
        server_->start();
    }

    void TearDown() override {
        if (batcher_) batcher_->stop();
        if (server_) server_->stop();
        if (service_) service_->stop();
    }

    // Declared in dependency order so destruction runs in reverse.
    FakeBackend::Shared shared_;
    infer::BatcherMetrics bm_{8};
    std::optional<infer::BlockPool> pool_;
    std::optional<infer::DynamicBatcher> batcher_;
    std::optional<infer::InferenceService> service_;
    std::optional<infer::net::HttpServer> server_;
};

}  // namespace

TEST_F(InferenceServiceTest, InferReturnsTopKWithBatchInfo) {
    start();
    Client c(server_->port());
    auto r = c.request("POST", "/v1/infer", sample_body(1.0f));
    ASSERT_EQ(r.status, 200) << r.body;
    // Fake model outputs [sum=4, 2*x0=2] -> class 0 first.
    EXPECT_NE(r.body.find("\"top_k\":[{\"class\":0"), std::string::npos) << r.body;
    EXPECT_NE(r.body.find("\"batch_size\":1"), std::string::npos) << r.body;
    EXPECT_NE(r.body.find("\"queue_us\":"), std::string::npos);
}

TEST_F(InferenceServiceTest, WrongBodySizeIs400) {
    start();
    Client c(server_->port());
    EXPECT_EQ(c.request("POST", "/v1/infer", "abc").status, 400);
    EXPECT_EQ(service_->metrics().bad_request.value(), 1u);
}

TEST_F(InferenceServiceTest, PoolExhaustionIs503) {
    start(/*pool_blocks=*/1, /*compute_delay=*/300ms);
    Client a(server_->port()), b(server_->port()), c(server_->port());
    // a's request is batched and its block returned to the pool as soon as it
    // is gathered, while the (slow) model runs.
    a.send_raw("POST /v1/infer HTTP/1.1\r\nContent-Length: 16\r\n\r\n" + sample_body(1));
    std::this_thread::sleep_for(50ms);
    // b takes the freed block and waits in the queue behind a's batch.
    b.send_raw("POST /v1/infer HTTP/1.1\r\nContent-Length: 16\r\n\r\n" + sample_body(2));
    std::this_thread::sleep_for(50ms);
    // c finds the only block held by b.
    auto rc = c.request("POST", "/v1/infer", sample_body(3));
    auto ra = a.read_response();
    auto rb = b.read_response();
    EXPECT_EQ(ra.status, 200);
    EXPECT_EQ(rb.status, 200);
    EXPECT_EQ(rc.status, 503);
    EXPECT_GE(service_->metrics().rejected.value(), 1u);
}

TEST_F(InferenceServiceTest, StatsReportWindowedCountsAndBatching) {
    start();
    Client c(server_->port());
    for (int i = 0; i < 20; ++i) ASSERT_EQ(c.request("POST", "/v1/infer", sample_body(1)).status, 200);
    c.request("POST", "/v1/infer", "bad");

    auto r = c.request("GET", "/stats?window=5");
    ASSERT_EQ(r.status, 200);
    EXPECT_NE(r.head.find("application/json"), std::string::npos);
    for (const char* key : {"\"throughput_rps\":", "\"latency_ms\":{", "\"p99\":", "\"queue_wait_ms\":",
                            "\"batch_compute_ms\":", "\"avg_batch_size\":", "\"queue_depth\":", "\"request_pool\":"}) {
        EXPECT_NE(r.body.find(key), std::string::npos) << key << " missing in " << r.body;
    }
    EXPECT_NE(r.body.find("\"ok\":20"), std::string::npos) << r.body;
    EXPECT_NE(r.body.find("\"bad_request\":1"), std::string::npos) << r.body;
    EXPECT_NE(r.body.find("\"connections\":1"), std::string::npos) << r.body;
}

TEST_F(InferenceServiceTest, StatsWindowExcludesOldTraffic) {
    start();
    Client c(server_->port());
    for (int i = 0; i < 5; ++i) ASSERT_EQ(c.request("POST", "/v1/infer", sample_body(1)).status, 200);
    std::this_thread::sleep_for(1300ms);  // ticks every 50 ms: this traffic is now >1 s old
    ASSERT_EQ(c.request("POST", "/v1/infer", sample_body(1)).status, 200);
    auto r = c.request("GET", "/stats?window=1");
    EXPECT_NE(r.body.find("\"ok\":1,"), std::string::npos) << r.body;
}

TEST_F(InferenceServiceTest, PrometheusExposition) {
    start();
    Client c(server_->port());
    for (int i = 0; i < 3; ++i) ASSERT_EQ(c.request("POST", "/v1/infer", sample_body(1)).status, 200);
    auto r = c.request("GET", "/metrics");
    ASSERT_EQ(r.status, 200);
    EXPECT_NE(r.head.find("text/plain; version=0.0.4"), std::string::npos);
    for (const char* line : {"infer_requests_total{code=\"200\"} 3\n", "# TYPE infer_request_latency_seconds summary\n",
                             "infer_request_latency_seconds{quantile=\"0.99\"} ", "infer_request_latency_seconds_count 3\n",
                             "# TYPE infer_batch_size histogram\n", "infer_batch_size_bucket{le=\"+Inf\"} ",
                             "infer_queue_depth 0\n", "infer_connections_active 1\n"}) {
        EXPECT_NE(r.body.find(line), std::string::npos) << "missing: " << line << "\n" << r.body;
    }
    // Every non-comment line is "name[{labels}] value".
    std::size_t pos = 0;
    while (pos < r.body.size()) {
        const std::size_t eol = r.body.find('\n', pos);
        const std::string line = r.body.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.empty() || line[0] == '#') continue;
        const std::size_t sp = line.rfind(' ');
        ASSERT_NE(sp, std::string::npos) << line;
        EXPECT_NO_THROW((void)std::stod(line.substr(sp + 1))) << line;
    }
}
