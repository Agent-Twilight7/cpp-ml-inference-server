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
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime install dir")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime not found in ${ONNXRUNTIME_ROOT}. Run scripts/fetch_onnxruntime.sh")
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
    src/runtime/onnx_backend.cpp
)
target_include_directories(infer_core PUBLIC include)
target_link_libraries(infer_core PUBLIC onnxruntime Threads::Threads)

# ---- Executables ----
add_executable(ort_smoke apps/ort_smoke.cpp)
target_link_libraries(ort_smoke PRIVATE infer_core)
