cmake_minimum_required(VERSION 3.20)
project(Source VERSION 0.1.0 LANGUAGES C CXX)

## ============================================================================
##  Global CMake Variables.
## ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## ============================================================================
##  Global compiler options.
## ============================================================================
## Turn on diagnostics colours.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
endif()

## Use mold as the default linker, if it exists.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    find_program(MOLD_LINKER "mold")
    if (MOLD_LINKER)
        add_compile_options(-fuse-ld=mold)
        add_link_options(-fuse-ld=mold)
    endif()
endif()

## ============================================================================
##  Compiler options.
## ============================================================================
add_library(options INTERFACE)

## Flags for Clang and GCC.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(options INTERFACE
        -fno-exceptions
        -fno-rtti

        ## Warnings.
        -Wall -Wextra     # Enable ‘all’ warnings.
        -Wundef           # Invalid #undef or undefined macro in #if.
        -Wcast-align      # Casting that changes alignment.
        -Wconversion      # Implicit conversions.
        -Wsign-conversion # Implicit sign conversions.
        -Wformat=2        # Stricter format checking.

        ## Disabled warnings.
        -Wno-unused-function
        -Wno-unused-local-typedefs
        -Wno-unused-parameter

        ## NULL Errors.
        -Werror=nonnull # Passing NULL to nonnull parameter.

        ## Memory Errors.
        -Werror=address              # Suspicious use of addresses.
        -Werror=init-self            # Initialization of a variable with itself.
        -Werror=uninitialized

        ## Return type.
        -Werror=return-type
        -Wmissing-noreturn

        ## C/C++.
        -Werror=implicit-fallthrough
        -Werror=missing-include-dirs # User-specified include dir does not exist.
        -Werror=pointer-arith        # Disallow void* and function pointer arithmetic.
        -Werror=string-compare       # Nonsensical string comparisons.
        -Werror=switch               # Missing switch cases.
        # -Werror=switch-enum          # Switch on enum (even if there is a default case).
        -Werror=write-strings        # Strings in C should be const char*.

        ## C++.
        -Werror=missing-field-initializers
        -Werror=non-virtual-dtor
        -Werror=pessimizing-move
    )
endif()

## Additional flags for GCC.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(options INTERFACE 
        -Wlogical-op      # Duplicate or unintended logical operators.
        -Werror=invalid-memory-model # For atomics.
        -Werror=maybe-uninitialized
        -Werror=missing-requires
        -Werror=return-local-addr
    )
endif()

## Additional flags for Clang.
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(options INTERFACE 
        -Werror=dangling 
        -Werror=return-stack-address
    )
endif()

## Flags for MSVC.
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(options INTERFACE
        /W4 # Enable ‘all’ warnings.

        # Allow unnamed structs/unions.
        /wd4201
    )
endif()

## On Windows, don’t suggest the _s nonsense functions.
if (WIN32)
    target_compile_definitions(options INTERFACE
        _CRT_SECURE_NO_WARNINGS
        _CRT_SECURE_NO_WARNINGS_GLOBALS
        _CRT_NONSTDC_NO_WARNINGS
    )
endif()

## Debug/Release flags.
if (NOT MSVC)
    target_compile_options(options INTERFACE
        $<$<CONFIG:DEBUG>:-O0 -g3 -ggdb3 -rdynamic>
        $<$<CONFIG:RELEASE>:-O3 -march=native>
    )
    target_link_options(options INTERFACE
        $<$<CONFIG:DEBUG>:-O0 -g3 -ggdb3 -rdynamic>
        $<$<CONFIG:RELEASE>:-O3 -march=native>
    )
else()
    target_compile_options(options INTERFACE
        $<$<CONFIG:DEBUG>:/Od>
        $<$<CONFIG:RELEASE>:/O2>
    )
endif()

## ============================================================================
##  Submodules and include dirs.
## ============================================================================
## Find libfmt.
find_package(fmt CONFIG)
if (NOT DEFINED fmt_FOUND)
    include(FetchContent)
	message(STATUS "Fmtlib not found, downloading ...")
	FetchContent_Declare(fmtlib
		GIT_REPOSITORY https://github.com/fmtlib/fmt.git
		GIT_TAG 10.0.0
		SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/libs/fmt"
    )
	FetchContent_MakeAvailable(fmtlib)
endif()

## Link against libfmt.
target_link_libraries(options INTERFACE fmt)

## Include dir.
include_directories("${PROJECT_SOURCE_DIR}/include")

## Add everything in ‘libs’ too.
if (EXISTS ${PROJECT_SOURCE_DIR}/libs)
    file(GLOB libs ${PROJECT_SOURCE_DIR}/libs/*)

    ## Add the include directory to the include path, or the root
    ## directory if there is no include directory.
    foreach(lib ${libs})
        if (IS_DIRECTORY ${lib}/include)
            target_include_directories(options INTERFACE ${lib}/include)
        else()
            target_include_directories(options INTERFACE ${lib})
        endif()
    endforeach()

    ## Also add all of them as subdirectories if they have a CMakeLists.txt.
    foreach (lib ${libs})
        if (EXISTS ${lib}/CMakeLists.txt)
            add_subdirectory(${lib})
        endif()
    endforeach()
endif()

## ============================================================================
##  LLVM.
## ============================================================================
set(LLVM_DIR "${PROJECT_SOURCE_DIR}/llvm/out/lib/cmake/llvm")
set(MLIR_DIR "${PROJECT_SOURCE_DIR}/llvm/out/lib/cmake/mlir")
set(Clang_DIR "${PROJECT_SOURCE_DIR}/llvm/out/lib/cmake/clang")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

## Add the LLVM include directories.
include_directories(SYSTEM
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
)

## Add the LLVM definitions.
add_definitions(
    ${LLVM_DEFINITIONS}
    ${MLIR_DEFINITIONS}
    ${CLANG_DEFINITIONS}
    "-D__SRCC_CLANG_EXE=\"${LLVM_TOOLS_BINARY_DIR}/clang\""
)

## Use llvm-config to get all LLVM libraries.
execute_process(
    COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-config" --libs
    OUTPUT_VARIABLE llvm_libs
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

## Add MLIR libraries.
link_directories(${LLVM_BUILD_LIBRARY_DIR})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

## Link against all of them.
separate_arguments(llvm_libs)
target_link_libraries(options INTERFACE
    ${llvm_libs}
    ${dialect_libs}
    ${conversion_libs}
    clang clangTooling clangASTMatchers
    MLIRSourceHLIR
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIROptLib
)

## ============================================================================
##  Tablegen.
## ============================================================================
file(GLOB_RECURSE hlir_lib_sources "${PROJECT_SOURCE_DIR}/lib/hlir/*.cc")
add_subdirectory(include/hlir)
add_mlir_dialect_library(MLIRSourceHLIR
    ${hlir_lib_sources}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/hlir

    DEPENDS
    MLIRHLIROpsIncGen

    LINK_LIBS PUBLIC MLIRIR
)

include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")



#[[## Get the libraries via LLVM config.
execute_process(
    COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-config" --libs
    OUTPUT_VARIABLE llvm_libs
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)

## Link against them
target_link_directories(options INTERFACE ${LLVM_LIBRARY_DIRS})
target_link_libraries(options INTERFACE
    ${llvm_libs}
    MLIRAnalysis MLIRIR MLIRParser MLIRSideEffects MLIRTransforms
)]]

## ============================================================================
##  Executables and libraries.
## ============================================================================
## Everything in ‘src’ is part of the project.
file(GLOB_RECURSE sources lib/*.cc src/*.cc)

## As well as everything in ‘include’, if it exists.
if (EXISTS ${PROJECT_SOURCE_DIR}/include)
    file(GLOB_RECURSE includes include/*.hh)
    list(APPEND sources ${includes})
endif()

## Add the executable.
add_llvm_executable(srcc ${sources})
target_link_libraries(srcc PRIVATE options)
#llvm_update_compile_flags(srcc)
#mlir_check_all_link_libraries(srcc)
