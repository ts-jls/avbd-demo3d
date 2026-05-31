cmake_minimum_required(VERSION 3.13)

# Project Name
project(avbd_demo3d)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AVBD_ENABLE_WEBGPU "Enable Dawn/WebGPU architecture scaffolding" OFF)
option(AVBD_ENABLE_VIEWER_BRIDGE "Enable localhost WebSocket snapshots for the browser viewer" OFF)
set(AVBD_DAWN_DIR "" CACHE PATH "Path to a Dawn/WebGPU native install or checkout")

# Automatically collect all source files from the "source" directory
file(GLOB_RECURSE SOURCES "source/*.cpp" "source/*.h")

# Add Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include SDL2 first (ImGui needs it)
add_subdirectory(external/SDL)

# Add ImGui as a separate project
set(IMGUI_PROJECT_NAME "imgui")

file(GLOB IMGUI_SRC
    external/imgui/*.cpp
    external/imgui/backends/imgui_impl_sdl2.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

add_library(${IMGUI_PROJECT_NAME} STATIC ${IMGUI_SRC})

# Set include directories for ImGui
target_include_directories(${IMGUI_PROJECT_NAME} PUBLIC
    external/imgui
    external/imgui/backends
    external/SDL/include
)

# Link ImGui with SDL2
target_link_libraries(${IMGUI_PROJECT_NAME} PUBLIC SDL2::SDL2 SDL2::SDL2main)

# Link the main project with SDL2 and ImGui
target_include_directories(${PROJECT_NAME} PRIVATE external/SDL/include)
target_link_libraries(${PROJECT_NAME} PRIVATE SDL2::SDL2 SDL2::SDL2main ${IMGUI_PROJECT_NAME})

if(AVBD_ENABLE_WEBGPU)
    target_compile_definitions(${PROJECT_NAME} PRIVATE AVBD_ENABLE_WEBGPU=1)
    set(AVBD_DAWN_TARGET "")

    if(AVBD_DAWN_DIR)
        find_package(Threads QUIET)
        find_package(Dawn CONFIG QUIET
            PATHS "${AVBD_DAWN_DIR}"
            NO_DEFAULT_PATH
        )

        if(TARGET dawn::webgpu_dawn)
            set(AVBD_DAWN_TARGET dawn::webgpu_dawn)
        else()
            find_path(AVBD_DAWN_INCLUDE_DIR
                NAMES webgpu/webgpu.h dawn/webgpu.h
                PATHS
                    "${AVBD_DAWN_DIR}/include"
                    "${AVBD_DAWN_DIR}"
                NO_DEFAULT_PATH
            )
            find_library(AVBD_DAWN_LIBRARY
                NAMES webgpu_dawn dawn_native dawn
                PATHS
                    "${AVBD_DAWN_DIR}/lib"
                    "${AVBD_DAWN_DIR}/out/Release"
                    "${AVBD_DAWN_DIR}/out/Debug"
                    "${AVBD_DAWN_DIR}"
                NO_DEFAULT_PATH
            )
        endif()
    endif()

    if(AVBD_DAWN_TARGET)
        target_compile_definitions(${PROJECT_NAME} PRIVATE AVBD_HAS_DAWN=1)
        target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${AVBD_DAWN_TARGET})
        set(AVBD_HAS_DAWN_RUNTIME ON)
        message(STATUS "Dawn/WebGPU found: ${AVBD_DAWN_TARGET}")
    elseif(AVBD_DAWN_INCLUDE_DIR AND AVBD_DAWN_LIBRARY)
        target_compile_definitions(${PROJECT_NAME} PRIVATE AVBD_HAS_DAWN=1)
        target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
        target_include_directories(${PROJECT_NAME} PRIVATE "${AVBD_DAWN_INCLUDE_DIR}")
        target_link_libraries(${PROJECT_NAME} PRIVATE "${AVBD_DAWN_LIBRARY}")
        set(AVBD_HAS_DAWN_RUNTIME ON)
        message(STATUS "Dawn/WebGPU found: ${AVBD_DAWN_INCLUDE_DIR}; ${AVBD_DAWN_LIBRARY}")
    else()
        message(STATUS "AVBD_ENABLE_WEBGPU is enabled, but Dawn was not found. Set AVBD_DAWN_DIR to enable real device creation.")
    endif()
endif()

if(AVBD_ENABLE_VIEWER_BRIDGE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE AVBD_ENABLE_VIEWER_BRIDGE=1)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -s USE_SDL=2 -s USE_WEBGL2=1 -s ALLOW_MEMORY_GROWTH=1 -s SINGLE_FILE=1 -s LEGACY_GL_EMULATION=1 --shell-file ../source/shell.html")
    set(CMAKE_EXECUTABLE_SUFFIX ".html")
else()
    # Find and Link OpenGL
    find_package(OpenGL REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)
endif()

# Copy SDL2.dll to the output folder on Windows
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE Psapi)
    if(AVBD_ENABLE_VIEWER_BRIDGE)
        target_link_libraries(${PROJECT_NAME} PRIVATE Ws2_32)
    endif()

    add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL2::SDL2>    # Path to the built SDL2.dll
            $<TARGET_FILE_DIR:${PROJECT_NAME}> # Output directory of the executable
    )

    if(AVBD_HAS_DAWN_RUNTIME)
        set(AVBD_D3D_RUNTIME_SEARCH_DIRS
            "C:/Program Files (x86)/Windows Kits/10/Redist/D3D/x64"
            "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
            "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64"
            "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22000.0/x64"
            "C:/Program Files (x86)/Windows Kits/10/bin/10.0.19041.0/x64"
        )

        foreach(AVBD_D3D_RUNTIME_DLL d3dcompiler_47.dll dxcompiler.dll dxil.dll)
            find_file(AVBD_${AVBD_D3D_RUNTIME_DLL}_PATH
                NAMES ${AVBD_D3D_RUNTIME_DLL}
                PATHS ${AVBD_D3D_RUNTIME_SEARCH_DIRS}
                NO_DEFAULT_PATH
            )

            if(AVBD_${AVBD_D3D_RUNTIME_DLL}_PATH)
                add_custom_command(
                    TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${AVBD_${AVBD_D3D_RUNTIME_DLL}_PATH}"
                        $<TARGET_FILE_DIR:${PROJECT_NAME}>
                )
                message(STATUS "Dawn D3D runtime found: ${AVBD_${AVBD_D3D_RUNTIME_DLL}_PATH}")
            else()
                message(WARNING "Dawn D3D runtime not found: ${AVBD_D3D_RUNTIME_DLL}")
            endif()
        endforeach()
    endif()
endif()
