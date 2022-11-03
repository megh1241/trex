cmake_minimum_required(VERSION 3.8)
project (exe2)
set(CMAKE_CXX_STANDARD 17)


   #add_subdirectory(${morton_source}  ${CMAKE_CURRENT_BINARY_DIR}/morton-nd )
#target_link_libraries(exe2 PRIVATE morton-nd::MortonND)

add_executable(exe2 src/mortonND_BMI2.h src/mortonND_LUT.h src/static_sort.h src/hilbert.h src/hilbert.c src/utils.h src/utils.cpp src/model.h src/model.cpp src/polytope.h src/polytope.cpp src/polydb.h src/polydb.cpp src/config.h src/config.cpp src/json_reader.cpp src/json_reader.h src/main.cpp)



target_include_directories(exe2 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -std=c++17")

#set(CMAKE_C_COMPILER "gcc-9")
#set(CMAKE_CXX_COMPILER "g++-9")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
#set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Wall -O3 -mavx512f -mavx512cd")
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(exe2 PUBLIC OpenMP::OpenMP_CXX)
endif()

