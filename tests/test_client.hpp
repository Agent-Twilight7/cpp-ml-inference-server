#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdint>
#include <string>

namespace testing_support {

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
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

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

    // Sends a request and reads its response.
    Response request(const std::string& method, const std::string& path, const std::string& body = "") {
        std::string req = method + " " + path + " HTTP/1.1\r\nHost: test\r\n";
        if (!body.empty() || method == "POST") req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        req += "\r\n" + body;
        send_raw(req);
        return read_response();
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

}  // namespace testing_support
