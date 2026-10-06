cmake_minimum_required(VERSION 3.22)
project(cpp_ml_inference_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(INFER_ENABLE_ASAN "Build with AddressSanitizer + UBSan" OFF)
option(INFER_ENABLE_TSAN "Build with ThreadSanitizer" OFF)

if(INFER_ENABLE_ASAN AND INFER_ENABLE_TSAN)
    message(FATAL_ERROR "ASan and TSan cannot be enabled together")
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(INFER_ENABLE_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()
if(INFER_ENABLE_TSAN)
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
endif()

# ---- ONNX Runtime (prebuilt, fetched by scripts/fetch_onnxruntime.sh) ----
# INFER_ENABLE_CUDA picks the GPU build of ORT, which also runs on CPU; the
# CUDA provider itself is loaded at runtime only when --device cuda is used.
option(INFER_ENABLE_CUDA "Link the CUDA 12 build of ONNX Runtime" OFF)
if(INFER_ENABLE_CUDA)
    set(_ort_default "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-gpu")
    set(_ort_flavor gpu)
else()
    set(_ort_default "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
    set(_ort_flavor cpu)
endif()
set(ONNXRUNTIME_ROOT "${_ort_default}" CACHE PATH "ONNX Runtime install dir")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime not found in ${ONNXRUNTIME_ROOT}. "
                        "Run: ORT_FLAVOR=${_ort_flavor} scripts/fetch_onnxruntime.sh")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")
# (Imported targets' include dirs are treated as system headers, so our warning
# flags don't apply to ORT's headers.)
set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib")

find_package(Threads REQUIRED)

# ---- Core library ----
add_library(infer_core STATIC
    src/batching/dynamic_batcher.cpp
    src/memory/block_pool.cpp
    src/net/http.cpp
    src/net/http_server.cpp
    src/runtime/onnx_backend.cpp
)
target_include_directories(infer_core PUBLIC include)
target_link_libraries(infer_core PUBLIC onnxruntime Threads::Threads)
if(INFER_ENABLE_CUDA)
    target_compile_definitions(infer_core PUBLIC INFER_HAS_CUDA=1)
endif()

# ---- Executables ----
add_executable(ort_smoke apps/ort_smoke.cpp)
target_link_libraries(ort_smoke PRIVATE infer_core)

add_executable(inference_server apps/server_main.cpp)
target_link_libraries(inference_server PRIVATE infer_core)

# ---- Tests ----
option(INFER_BUILD_TESTS "Build unit tests (fetches GoogleTest)" ON)
if(INFER_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(unit_tests
        tests/block_pool_test.cpp
        tests/dynamic_batcher_test.cpp
        tests/http_parser_test.cpp
        tests/http_server_test.cpp
    )
    target_link_libraries(unit_tests PRIVATE infer_core GTest::gtest_main)
    include(GoogleTest)
    # PRE_TEST: list tests when ctest runs, not at build time (sanitized
    # binaries may need a special launcher, e.g. `setarch -R` for TSan).
    gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)
endif()
