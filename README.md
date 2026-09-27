# neo-libdct

A static F32 DCT library for video blocks, with a C API and C++17 implementations using scalar code and Highway SIMD.

Supports two-dimensional forward DCTs of multiple sizes, 8×8 inverse DCTs and coefficient filtering, batch processing, and in-place operation.

## Build

Requires CMake 3.24 or later and a C++17 compiler. CMake fetches the Highway dependency automatically.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 2
ctest --test-dir build -C Release --output-on-failure
```

## Usage

Include the project with CMake's `add_subdirectory`, link against `neo::dct`, and include the public header `neo_dct.h`.

## License

[GNU GPL v2 or later](LICENSE)
