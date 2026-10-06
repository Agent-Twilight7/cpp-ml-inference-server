#include "infer/net/http.hpp"

#include <gtest/gtest.h>

#include <string>

using infer::net::HttpParser;
using infer::net::HttpRequest;
using Status = HttpParser::Status;

TEST(HttpParser, ParsesSimpleGet) {
    HttpParser p;
    const std::string raw = "GET /health?x=1 HTTP/1.1\r\nHost: a\r\nX-Thing:  spaced  \r\n\r\n";
    ASSERT_EQ(p.parse(raw), Status::kComplete);
    EXPECT_EQ(p.consumed(), raw.size());
    HttpRequest req = p.take_request();
    EXPECT_EQ(req.method, "GET");
    EXPECT_EQ(req.target, "/health?x=1");
    EXPECT_EQ(req.path(), "/health");
    ASSERT_NE(req.header("host"), nullptr);
    EXPECT_EQ(*req.header("x-thing"), "spaced");
    EXPECT_TRUE(req.keep_alive);
    EXPECT_TRUE(req.body.empty());
}

TEST(HttpParser, ParsesBodyAndLeavesPipelinedBytes) {
    HttpParser p;
    const std::string first = "POST /v1/infer HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello";
    const std::string raw = first + "GET / HTTP/1.1\r\n\r\n";
    ASSERT_EQ(p.parse(raw), Status::kComplete);
    EXPECT_EQ(p.consumed(), first.size());
    EXPECT_EQ(p.take_request().body, "hello");

    ASSERT_EQ(p.parse(std::string_view(raw).substr(first.size())), Status::kComplete);
    EXPECT_EQ(p.take_request().method, "GET");
}

TEST(HttpParser, HandlesByteAtATimeDelivery) {
    HttpParser p;
    const std::string raw = "POST /a HTTP/1.1\r\nContent-Length: 3\r\n\r\nxyz";
    std::string buf;
    for (std::size_t i = 0; i + 1 < raw.size(); ++i) {
        buf += raw[i];
        ASSERT_EQ(p.parse(buf), Status::kNeedMore) << "at byte " << i;
    }
    buf += raw.back();
    ASSERT_EQ(p.parse(buf), Status::kComplete);
    EXPECT_EQ(p.take_request().body, "xyz");
}

TEST(HttpParser, KeepAliveRules) {
    HttpParser p;
    ASSERT_EQ(p.parse("GET / HTTP/1.1\r\nConnection: close\r\n\r\n"), Status::kComplete);
    EXPECT_FALSE(p.take_request().keep_alive);
    ASSERT_EQ(p.parse("GET / HTTP/1.0\r\n\r\n"), Status::kComplete);
    EXPECT_FALSE(p.take_request().keep_alive);
    ASSERT_EQ(p.parse("GET / HTTP/1.0\r\nConnection: Keep-Alive\r\n\r\n"), Status::kComplete);
    EXPECT_TRUE(p.take_request().keep_alive);
}

TEST(HttpParser, ExpectContinueFlagFiresOnceWhileBodyPending) {
    HttpParser p;
    const std::string head = "POST /a HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\n";
    ASSERT_EQ(p.parse(head), Status::kNeedMore);
    EXPECT_TRUE(p.take_continue_flag());
    EXPECT_FALSE(p.take_continue_flag());
    ASSERT_EQ(p.parse(head + "ok"), Status::kComplete);
}

struct ErrorCase {
    const char* raw;
    int status;
};

class HttpParserErrors : public ::testing::TestWithParam<ErrorCase> {};

TEST_P(HttpParserErrors, RejectsWithStatus) {
    HttpParser p({.max_header_bytes = 256, .max_body_bytes = 100});
    ASSERT_EQ(p.parse(GetParam().raw), Status::kError);
    EXPECT_EQ(p.error_status(), GetParam().status);
}

INSTANTIATE_TEST_SUITE_P(
    Cases, HttpParserErrors,
    ::testing::Values(ErrorCase{"GARBAGE\r\n\r\n", 400},
                      ErrorCase{"GET noslash HTTP/1.1\r\n\r\n", 400},
                      ErrorCase{"GET / HTTP/2.0\r\n\r\n", 505},
                      ErrorCase{"GET / HTTP/1.1\r\nNoColon\r\n\r\n", 400},
                      ErrorCase{"GET / HTTP/1.1\r\nBad Name: x\r\n\r\n", 400},
                      ErrorCase{"POST / HTTP/1.1\r\nContent-Length: abc\r\n\r\n", 400},
                      ErrorCase{"POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n", 400},
                      ErrorCase{"POST / HTTP/1.1\r\nContent-Length: 101\r\n\r\n", 413},
                      ErrorCase{"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", 501}));

TEST(HttpParser, RejectsOversizedHeadersBeforeTerminator) {
    HttpParser p({.max_header_bytes = 64, .max_body_bytes = 100});
    const std::string raw = "GET / HTTP/1.1\r\nX: " + std::string(100, 'a');
    ASSERT_EQ(p.parse(raw), Status::kError);
    EXPECT_EQ(p.error_status(), 431);
}

TEST(HttpSerialize, WritesStatusHeadersAndBody) {
    auto resp = infer::net::HttpResponse::json(200, "{}");
    resp.extra_headers.emplace_back("X-Test", "1");
    std::string out;
    infer::net::serialize(resp, /*keep_alive=*/false, out);
    EXPECT_EQ(out,
              "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n"
              "Connection: close\r\nX-Test: 1\r\n\r\n{}");
}
