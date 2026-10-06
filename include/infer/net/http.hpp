#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infer::net {

using HeaderList = std::vector<std::pair<std::string, std::string>>;

struct HttpRequest {
    std::string method;
    std::string target;  // path plus optional "?query"
    int version_minor = 1;  // HTTP/1.<minor>
    HeaderList headers;     // names lower-cased, values trimmed
    // Points into the connection's receive buffer: no copy, no allocation, but
    // valid only until the handler returns. Copy it to keep it longer.
    std::string_view body;
    bool keep_alive = true;

    std::string_view path() const;
    // Looks up a header by its lower-case name.
    const std::string* header(std::string_view lower_name) const;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "text/plain";
    std::string body;
    HeaderList extra_headers;

    static HttpResponse text(int status, std::string body);
    static HttpResponse json(int status, std::string body);
};

const char* reason_phrase(int status);

// Appends the wire form of `resp` (status line, headers, body) to `out`.
void serialize(const HttpResponse& resp, bool keep_alive, std::string& out);

// Incremental HTTP/1.x request parser. The caller owns the receive buffer and
// calls parse() each time more bytes arrive; on kComplete it takes the request
// (whose body views the buffer), uses it, then removes `consumed()` bytes from
// the front of the buffer.
//
// Only Content-Length bodies are supported (chunked uploads get 501).
class HttpParser {
public:
    struct Limits {
        std::size_t max_header_bytes = 8 * 1024;
        std::size_t max_body_bytes = 16 * 1024 * 1024;
    };

    enum class Status { kNeedMore, kComplete, kError };

    HttpParser() : HttpParser(Limits{}) {}
    explicit HttpParser(Limits limits) : limits_(limits) {}

    Status parse(std::string_view buf);

    // Valid after kComplete: bytes of `buf` that made up the request.
    std::size_t consumed() const { return consumed_; }
    // Valid after kComplete: moves the request out and readies the parser for the next one.
    HttpRequest take_request();
    // Valid after kError: the HTTP status to reply with (400, 413, 431, 501, 505).
    int error_status() const { return error_status_; }

    // True exactly once per request, when its headers asked for
    // "Expect: 100-continue" and the body hasn't fully arrived yet.
    bool take_continue_flag();

private:
    Status fail(int status);    // records the error status, returns kError
    bool reject(int status);    // records the error status, returns false
    bool parse_head(std::string_view head);

    Limits limits_;
    std::size_t scan_from_ = 0;      // where to resume searching for "\r\n\r\n"
    std::optional<std::size_t> head_len_;  // set once headers are parsed
    std::size_t body_len_ = 0;
    std::size_t consumed_ = 0;
    bool expect_continue_ = false;
    int error_status_ = 0;
    HttpRequest req_;
};

}  // namespace infer::net
