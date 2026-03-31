set(QBRUNTIME_PATH "" CACHE PATH "Path to qbruntime package root (contains include/, lib/, and on Windows bin/)")
set(VENDOR "" CACHE STRING "Vendor name")
set(PRODUCT "" CACHE STRING "Product name")
set(DRIVER_TYPE "" CACHE STRING "Driver type")

# 1. Use FetchContent
# 2. Use Windows DLL
# 3. Use Linux .so file or installed library via apt
if(VENDOR AND PRODUCT AND DRIVER_TYPE)
    include(FetchContent)
    FetchContent_Declare(
      build_tools
      GIT_REPOSITORY https://git.mobilint.com/sdk/common/build_tools.git
      GIT_TAG master)
    FetchContent_MakeAvailable(build_tools)
    include(MobilintExtLib)
    mobilint_extlib_declare(qbruntime GIT_TAG master)
    mobilint_extlib_make_available(qbruntime)
elseif(WIN32)
    add_library(qbruntime SHARED IMPORTED GLOBAL)
    if(NOT QBRUNTIME_PATH)
        message(FATAL_ERROR "On Windows, -DQBRUNTIME_PATH=<path> is required.")
    endif()

    get_filename_component(QBRUNTIME_ABS_PATH
        "${QBRUNTIME_PATH}"
        ABSOLUTE
        BASE_DIR "${CMAKE_BINARY_DIR}"
    )

    set_target_properties(qbruntime PROPERTIES
        IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
        IMPORTED_IMPLIB_DEBUG "${QBRUNTIME_ABS_PATH}/lib/qbruntimed.lib"
        IMPORTED_LOCATION_DEBUG "${QBRUNTIME_ABS_PATH}/bin/qbruntimed.dll"
        IMPORTED_IMPLIB_RELEASE "${QBRUNTIME_ABS_PATH}/lib/qbruntime.lib"
        IMPORTED_LOCATION_RELEASE "${QBRUNTIME_ABS_PATH}/bin/qbruntime.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${QBRUNTIME_ABS_PATH}/include"
    )
else()
    add_library(qbruntime SHARED IMPORTED GLOBAL)
    if(QBRUNTIME_PATH)
        get_filename_component(QBRUNTIME_ABS_PATH
            "${QBRUNTIME_PATH}"
            ABSOLUTE
            BASE_DIR "${CMAKE_BINARY_DIR}"
        )
        set(QBRUNTIME_INCLUDE_DIR "${QBRUNTIME_ABS_PATH}/include")
        set(QBRUNTIME_LIBRARY "${QBRUNTIME_ABS_PATH}/lib/libqbruntime.so")
    else()
        find_path(QBRUNTIME_INCLUDE_DIR
            NAMES qbruntime/qbruntime.h
        )
        find_library(QBRUNTIME_LIBRARY
            NAMES qbruntime
        )
    endif()

    if(NOT QBRUNTIME_INCLUDE_DIR OR NOT QBRUNTIME_LIBRARY)
        message(FATAL_ERROR
            "qbruntime not found. Install mobilint-qb-runtime (e.g. sudo apt install mobilint-qb-runtime) "
            "or set -DQBRUNTIME_PATH=<path>."
        )
    endif()

    set_target_properties(qbruntime PROPERTIES
        IMPORTED_LOCATION "${QBRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${QBRUNTIME_INCLUDE_DIR}"
    )
endif()

find_package(yaml-cpp CONFIG QUIET)
if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp)
else()
    include(FetchContent)
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(yaml-cpp)
    if(TARGET yaml-cpp::yaml-cpp)
        set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
    else()
        set(YAML_CPP_TARGET yaml-cpp)
    endif()
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

set(DEMO_SOURCES
    demo.cc
    feeder.cc

    model.cc
    model_inference.cc

    post_yolo_base.cc
    post_yolo_anchor.cc
    post_yolo_anchor_face.cc
    
    post_ssd.cc
    
    post_yolo_anchorless.cc
    post_yolo_anchorless_face.cc
    post_yolo_nmsfree.cc
    post_yolo_anchorless_pose.cc
    post_yolo_anchorless_seg.cc

    post_yolo_dflfree.cc
    post_yolo_dflfree_pose.cc
    post_yolo_dflfree_seg.cc

    yaml_util.cc
)
set(DEMO_EXTERNAL_LIBRARIES
    qbruntime
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    ${YAML_CPP_TARGET}
)

add_executable(demo ${DEMO_SOURCES})
target_link_libraries(demo PRIVATE ${DEMO_EXTERNAL_LIBRARIES})
target_include_directories(demo PRIVATE "${PROJECT_SOURCE_DIR}/src")
# NOTE: Post-process codes from algorithm team use c++17 features.
target_compile_features(demo PRIVATE cxx_std_17)

if(MSVC)
    set_target_properties(demo PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    # OpenCV's FFmpeg backend DLL is not always captured by TARGET_RUNTIME_DLLS.
    # Try common OpenCV Windows layouts and copy it explicitly when found.
    find_file(OPENCV_FFMPEG_DLL
        NAMES opencv_videoio_ffmpeg*.dll
        HINTS
            "${OpenCV_DIR}"
            "${OpenCV_DIR}/../.."
            "${OpenCV_DIR}/.."
        PATH_SUFFIXES
            x64/vc17/bin
            x64/vc16/bin
            x64/vc15/bin
            bin
    )

    add_custom_command(TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:demo>
        $<TARGET_FILE_DIR:demo>
        COMMAND_EXPAND_LISTS
    )

    if(OPENCV_FFMPEG_DLL)
        add_custom_command(TARGET demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_FFMPEG_DLL}"
            "$<TARGET_FILE_DIR:demo>"
        )
    endif()
endif()

if(DEMO_DEV_MODE)
    target_compile_definitions(demo PRIVATE USE_SLEEP_DRIVER)
endif()
