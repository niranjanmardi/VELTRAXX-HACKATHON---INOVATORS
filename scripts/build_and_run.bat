@echo off
echo Building VELTRAXX Sparse Engine...
mkdir build 2>nul
cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..
echo.
echo Running sparse engine...
build\sparse_engine.exe --samples 1000
