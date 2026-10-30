# C++ ML Inference Server: Learning Roadmap

This document is the persistent source of truth for the learning journey.
Update it as lessons are completed: progress checklist, concepts learned,
misunderstandings corrected, and open questions.

---

## 1. Overall goal

Understand this repository well enough to explain and defend it in an SDE (AI)
interview. Specifically:

- **Understand the whole repository:** important files, classes, functions,
  data structures, and execution flows.
- **For each component, know:** what it does, where it is implemented, why it
  exists, and how it connects to the other components.
- **Understand the engineering:** design decisions, alternatives considered,
  bottlenecks, failure scenarios, and how it would scale.
- **Be able to defend it:** answer "why did you do it this way?" and "what
  happens when X fails?" with answers grounded in the actual code.

The goal is the whole repository, not isolated concepts. Every fundamentals
lesson must end by connecting back to a specific place in this codebase.

---

## 2. Complete learning roadmap

Status legend: ⬜ pending · 🟨 in progress · ✅ understood (student confirmed)

### Phase 1: Repository and architecture ⬜
*An overview was delivered once in dense form. It will be revisited in
tutor style after the networking fundamentals.*

- Directory structure: `include/infer/` + `src/` (library `infer_core`),
  `apps/` (executables), `tests/`, `bench/`, `deploy/`, `models/`, `web/`
- Tech stack: C++20, CMake + Ninja, ONNX Runtime (CPU/CUDA), raw Linux
  `epoll`, GoogleTest, sanitizers (ASan/UBSan/TSan), GitHub Actions, Docker,
  Prometheus/Grafana
- Entry points: `apps/server_main.cpp` (`main`), `apps/loadgen.cpp`,
  `apps/ort_smoke.cpp`
- Components and the high-level architecture diagram
- Startup order and shutdown order in `main`

### Phase 2: Networking fundamentals 🟨
- [ ] Client and server, request and response *(Lesson 1, in progress)*
- [ ] IP addresses and ports
- [ ] HTTP: request line, headers, body, status codes
- [ ] TCP connections: what a "connection" is, why bytes arrive in pieces
- [ ] Sockets and file descriptors
- [ ] Listening, accepting, reading, writing
- [ ] Keep-alive and pipelining
- Code links: `apps/server_main.cpp` (`Config::port`, `parse_args`),
  `src/net/http_server.cpp` (`make_listener`, `ensure_port_free`,
  `on_accept`), `include/infer/net/http.hpp` (`HttpRequest`, `HttpResponse`)

### Phase 3: Detailed codebase walkthrough ⬜
- [ ] File-by-file: `net/`, `batching/`, `runtime/`, `memory/`, `metrics/`, `server/`, `apps/`
- [ ] Function-by-function for the important classes:
  `HttpParser`, `HttpServer`, `Reactor`, `Connection`, `Responder`,
  `InferenceService`, `DynamicBatcher`, `ModelBackend`, `OnnxBackend`,
  `BlockPool`, `LatencyHistogram`
- [ ] Trace one `POST /v1/infer` from the client to the final response, one
  transition at a time (which thread, what data moves, why the next step happens)
- [ ] Routes: `/health`, `/v1/infer`, `/stats`, `/metrics`, web demo routes

### Phase 4: Reactor and asynchronous I/O ⬜
- [ ] The problem: many connections, one thread can't block on each one
- [ ] Blocking vs non-blocking sockets
- [ ] `epoll`: what it does and why it exists (vs checking every socket / `select` / `poll`)
- [ ] Edge-triggered vs level-triggered (`EPOLLET`)
- [ ] The reactor event loop: `Reactor::run`
- [ ] Multiple reactors and `SO_REUSEPORT`
- [ ] Connection state: `Connection` fields (`in`, `out`, `busy`, `read_paused`, …)
- [ ] Incremental HTTP parsing: `HttpParser::parse`, `parse_head`, limits (400/413/431/501/505)
- [ ] Routing: `HttpServer::route`, `HttpServer::dispatch`
- [ ] Waking a reactor from another thread: `eventfd`, `Reactor::complete`, `on_wake`
- Code: `src/net/http_server.cpp`, `src/net/http.cpp`

### Phase 5: C++ concurrency ⬜
- [ ] Threads: reactor threads, worker threads, ticker thread, main thread
- [ ] Shared data and race conditions
- [ ] Mutexes and `std::lock_guard` / `std::unique_lock`
- [ ] Condition variables: waiting without polling (`cv_.wait`, `wait_until`)
- [ ] Producer-consumer queues
- [ ] Atomics and memory ordering (relaxed / acquire / release)
- [ ] Lock-free stack and the ABA problem: `BlockPool::try_acquire` / `release`
- [ ] Memory ownership: RAII, `std::unique_ptr`, move-only `BlockPool::Block`,
  `std::string_view` lifetimes, `std::span`
- [ ] Deadlock and starvation: why `leader_mu_` and `mu_` are ordered the way they are
- [ ] Signal handling across threads: `pthread_sigmask` + `sigwait`
- Code: `src/batching/dynamic_batcher.cpp`, `src/memory/block_pool.cpp`,
  `src/net/http_server.cpp`, `src/metrics/histogram.cpp`, `apps/server_main.cpp`

### Phase 6: Dynamic batching ⬜
- [ ] Why batch: GPU per-image cost falls with batch size; CPU stays flat (README "Why batching?" table)
- [ ] The bounded queue: `DynamicBatcher::submit`, `max_queue_size`
- [ ] Batch formation: `form_batch`, full batch OR oldest request waited `max_queue_delay`
- [ ] Leader/follower workers: `leader_mu_`, `notify_one` only on two transitions
- [ ] Running a batch: `run_batch` (timeout filter, gather, padding, callbacks)
- [ ] Batch-size buckets and padding: `padded_batch_size`
- [ ] Result association: callback per request, row `i` of `batch_out`
- [ ] Latency vs throughput trade-off (single-client penalty, max-delay sweep)

### Phase 7: ML inference ⬜
- [ ] What a model, a tensor, and inference are
- [ ] Model loading: `OnnxBackend` constructor, `Ort::Env`, `Ort::Session`
- [ ] Shapes and the dynamic batch dimension
- [ ] Running inference: `OnnxBackend::run`, wrapping caller memory as tensors
- [ ] CPU vs CUDA execution provider; pinned memory (`allocate_host_buffer`)
- [ ] The ~30 ms shape-change penalty and per-bucket sessions
- [ ] Warm-up in `DynamicBatcher::start`
- [ ] Post-processing: softmax top-k in `result_json`
- [ ] Inference errors and how they become HTTP 500

### Phase 8: Failure handling and debugging ⬜
- [ ] Malformed requests (400/413/431/501/505) and wrong body size (400)
- [ ] Overload: full queue or empty pool → 503
- [ ] Queue timeout → 504
- [ ] Backend exception → 500 for the whole batch; throwing callbacks caught
- [ ] Client disconnects: connection ids, dropped responses
- [ ] Graceful shutdown: drain sequence in `main`
- [ ] What isn't handled: hung backend, worker crash, process restart
- [ ] Tests and sanitizers as debugging tools

### Phase 9: Performance and bottlenecks ⬜
- [ ] Throughput, latency, p50/p95/p99, queue depth
- [ ] Closed-loop vs open-loop load; coordinated omission (`apps/loadgen.cpp`)
- [ ] Metrics implementation: wait-free histograms, windowed snapshots (`InferenceService::ticker_loop`)
- [ ] Reading the benchmark results (`bench/results/`, README tables)
- [ ] Where the time goes: copies, reactor-thread work, worker-thread post-processing, GPU
- [ ] The top bottlenecks and proposed fixes

### Phase 10: System design and scaling ⬜
- [ ] What breaks first at 10×, 100×, 1000×
- [ ] Scaling across cores, GPUs, machines; load balancing
- [ ] Multiple models and versions
- [ ] gRPC vs REST; Triton / TorchServe / ONNX Runtime server vs custom
- [ ] Monitoring, autoscaling, fault recovery
- [ ] Always separate "exists today" from "proposed"

### Phase 11: Design-decision interrogation ⬜
- [ ] 10–15 key decisions/weaknesses, one question at a time, graded /10

### Phase 12: Mock interview ⬜
- [ ] Progressively harder questions across all areas, graded /10

### Final deliverables ⬜
- [ ] 60-second project explanation
- [ ] Codebase reference sheet
- [ ] Architecture and request-lifecycle summary
- [ ] Design decisions, alternatives, trade-offs
- [ ] Top 5 bottlenecks/weaknesses and fixes
- [ ] Concurrency and C++ revision sheet
- [ ] 25 most likely interview questions with answers
- [ ] One-page final-10-minutes revision sheet

---

## 3. Current progress

**Current phase:** Phase 2, Networking fundamentals
**Current lesson:** Lesson 1, "What actually happens when I send a request from my computer to a server?"

### Concepts introduced so far
| Concept | Plain-English meaning | Where it appears in the code |
|---|---|---|
| Client | The program that asks (browser, `curl`, `loadgen`) | `apps/loadgen.cpp`, `scripts/infer_client.py` |
| Server | The program that waits and answers | `apps/server_main.cpp` |
| Request | The message the client sends | `HttpRequest` in `include/infer/net/http.hpp` |
| Response | The message the server sends back | `HttpResponse` in `include/infer/net/http.hpp` |
| IP address | Which machine ("which building") | `HttpServerOptions::host` (default `0.0.0.0`) |
| Port | Which program on that machine ("which counter") | `Config::port = 8080`, `--port` in `parse_args` |
| HTTP | The text format for requests and responses | `GET /health HTTP/1.1` … `HTTP/1.1 200 OK` |
| `/health` endpoint | A simple "are you alive?" route | `InferenceService::install` in `src/server/inference_service.cpp` |

### Open question (awaiting answer)
> The server is running with `--port 9000`. Someone runs
> `curl http://127.0.0.1:8080/health` on the same laptop. Will the server
> answer? Why or why not?

### Misunderstandings recorded
*(none yet)*

### Notes to verify before the interview
- README claims 58 tests; a grep for `TEST` macros found 51. Confirm the real count.

---

## 4. Teaching rules

1. Start from first principles; assume no prior knowledge of unfamiliar concepts.
2. One concept at a time: plain English → analogy → tiny example → why it's
   needed → connection to the actual code.
3. Don't introduce several unfamiliar terms in one sentence.
4. Connect every concept to the real source: file path, function, small snippet,
   who calls it, what happens next.
5. Explain what, why, how, alternatives, and trade-offs.
6. Ask one understanding-check question at a time, then wait for the answer.
7. If the student is confused, explain differently, not with more jargon.
8. Never invent implementation details or performance claims; inspect the code first.
9. No dense numbered execution traces or huge tables in lessons.
10. Don't skip repository sections or abandon the roadmap.
11. Don't advance to another phase until the student understands the current
    material and explicitly agrees.
12. Keep this document updated after each lesson: checklist, concepts,
    misunderstandings, open questions.
