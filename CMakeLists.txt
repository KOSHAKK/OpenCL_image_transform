cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(PROJECT_NAME OpenCL_image_transform)
project(${PROJECT_NAME})

add_executable(${PROJECT_NAME} 
	src/main.cpp
	src/stb_image.hpp
	src/stb_image_write.h
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)



target_link_directories(${PROJECT_NAME} PRIVATE libs/OpenCL/common/lib/x64)


include_directories(libs/OpenCL)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL.lib)




