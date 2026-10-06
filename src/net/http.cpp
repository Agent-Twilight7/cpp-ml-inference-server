#include "infer/net/http.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>

namespace infer::net {
namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
    return s;
}

std::string to_lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// Case-insensitive "does this comma-separated header value contain `token`".
bool has_token(std::string_view value, std::string_view token) {
    while (!value.empty()) {
        const std::size_t comma = value.find(',');
        std::string_view item = trim(value.substr(0, comma));
        if (item.size() == token.size() &&
            std::equal(item.begin(), item.end(), token.begin(), [](char a, char b) {
                return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
            })) {
            return true;
        }
        if (comma == std::string_view::npos) break;
        value.remove_prefix(comma + 1);
    }
    return false;
}

bool is_token_char(char c) {
    // RFC 9110 tchar
    return std::isalnum(static_cast<unsigned char>(c)) || std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
}

}  // namespace

std::string_view HttpRequest::path() const {
    std::string_view t = target;
    return t.substr(0, t.find('?'));
}

const std::string* HttpRequest::header(std::string_view lower_name) const {
    for (const auto& [name, value] : headers) {
        if (name == lower_name) return &value;
    }
    return nullptr;
}

HttpResponse HttpResponse::text(int status, std::string body) {
    return HttpResponse{status, "text/plain; charset=utf-8", std::move(body), {}};
}

HttpResponse HttpResponse::json(int status, std::string body) {
    return HttpResponse{status, "application/json", std::move(body), {}};
}

const char* reason_phrase(int status) {
    switch (status) {
        case 100: return "Continue";
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 413: return "Content Too Large";
        case 415: return "Unsupported Media Type";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        case 505: return "HTTP Version Not Supported";
        default: return "Unknown";
    }
}

void serialize(const HttpResponse& resp, bool keep_alive, std::string& out) {
    out += "HTTP/1.1 ";
    out += std::to_string(resp.status);
    out += ' ';
    out += reason_phrase(resp.status);
    out += "\r\nContent-Type: ";
    out += resp.content_type;
    out += "\r\nContent-Length: ";
    out += std::to_string(resp.body.size());
    out += keep_alive ? "\r\nConnection: keep-alive" : "\r\nConnection: close";
    for (const auto& [name, value] : resp.extra_headers) {
        out += "\r\n";
        out += name;
        out += ": ";
        out += value;
    }
    out += "\r\n\r\n";
    out += resp.body;
}

HttpParser::Status HttpParser::fail(int status) {
    error_status_ = status;
    return Status::kError;
}

bool HttpParser::reject(int status) {
    error_status_ = status;
    return false;
}

HttpParser::Status HttpParser::parse(std::string_view buf) {
    if (!head_len_) {
        // Resume a few bytes back in case "\r\n\r\n" straddles two reads.
        const std::size_t from = scan_from_ >= 3 ? scan_from_ - 3 : 0;
        const std::size_t end = buf.find("\r\n\r\n", from);
        if (end == std::string_view::npos) {
            if (buf.size() > limits_.max_header_bytes) return fail(431);
            scan_from_ = buf.size();
            return Status::kNeedMore;
        }
        const std::size_t head_len = end + 4;
        if (head_len > limits_.max_header_bytes) return fail(431);
        if (!parse_head(buf.substr(0, end))) return Status::kError;
        head_len_ = head_len;
    }

    const std::size_t total = *head_len_ + body_len_;
    if (buf.size() < total) return Status::kNeedMore;

    req_.body.assign(buf.substr(*head_len_, body_len_));
    consumed_ = total;
    return Status::kComplete;
}

bool HttpParser::parse_head(std::string_view head) {
    req_ = HttpRequest{};
    body_len_ = 0;
    expect_continue_ = false;

    // ---- Request line: METHOD SP target SP HTTP/1.x ----
    const std::size_t line_end = head.find("\r\n");
    std::string_view line = head.substr(0, line_end);
    const std::size_t sp1 = line.find(' ');
    const std::size_t sp2 = sp1 == std::string_view::npos ? sp1 : line.find(' ', sp1 + 1);
    if (sp1 == 0 || sp2 == std::string_view::npos || sp2 == sp1 + 1) return reject(400);

    std::string_view method = line.substr(0, sp1);
    std::string_view target = line.substr(sp1 + 1, sp2 - sp1 - 1);
    std::string_view version = line.substr(sp2 + 1);
    if (!std::all_of(method.begin(), method.end(), is_token_char)) return reject(400);
    if (target.empty() || target.front() != '/') return reject(400);
    if (version == "HTTP/1.1") {
        req_.version_minor = 1;
    } else if (version == "HTTP/1.0") {
        req_.version_minor = 0;
    } else if (version.substr(0, 5) == "HTTP/") {
        return reject(505);
    } else {
        return reject(400);
    }
    req_.method.assign(method);
    req_.target.assign(target);

    // ---- Header fields ----
    bool have_length = false;
    std::string_view rest = line_end == std::string_view::npos ? std::string_view{} : head.substr(line_end + 2);
    while (!rest.empty()) {
        const std::size_t eol = rest.find("\r\n");
        std::string_view field = rest.substr(0, eol);
        rest = eol == std::string_view::npos ? std::string_view{} : rest.substr(eol + 2);

        const std::size_t colon = field.find(':');
        if (colon == 0 || colon == std::string_view::npos) return reject(400);
        std::string_view name = field.substr(0, colon);
        if (!std::all_of(name.begin(), name.end(), is_token_char)) return reject(400);  // also rejects "name :"
        std::string lname = to_lower(name);
        std::string_view value = trim(field.substr(colon + 1));

        if (lname == "content-length") {
            std::size_t n = 0;
            const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), n);
            if (value.empty() || ec != std::errc{} || ptr != value.data() + value.size()) return reject(400);
            if (have_length && n != body_len_) return reject(400);  // conflicting duplicates
            if (n > limits_.max_body_bytes) return reject(413);
            body_len_ = n;
            have_length = true;
        } else if (lname == "transfer-encoding") {
            return reject(501);
        } else if (lname == "expect" && has_token(value, "100-continue")) {
            expect_continue_ = true;
        }
        req_.headers.emplace_back(std::move(lname), std::string(value));
    }

    const std::string* conn = req_.header("connection");
    if (req_.version_minor == 1) {
        req_.keep_alive = !(conn && has_token(*conn, "close"));
    } else {
        req_.keep_alive = conn && has_token(*conn, "keep-alive");
    }
    return true;
}

HttpRequest HttpParser::take_request() {
    HttpRequest out = std::move(req_);
    req_ = HttpRequest{};
    scan_from_ = 0;
    head_len_.reset();
    body_len_ = 0;
    consumed_ = 0;
    expect_continue_ = false;
    return out;
}

bool HttpParser::take_continue_flag() {
    const bool flag = expect_continue_;
    expect_continue_ = false;
    return flag;
}

}  // namespace infer::net
