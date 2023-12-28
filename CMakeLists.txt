cmake_minimum_required(VERSION 3.20)
project(Source VERSION 0.1.0 LANGUAGES C CXX)

## ============================================================================
##  Global CMake Variables.
## ============================================================================
set(CMAKE_CXX_STANDARD 23)
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
        add_link_options(-fuse-ld=mold -Wl,--color-diagnostics)
    endif()
endif()

## ============================================================================
##  Compiler options.
## ============================================================================
add_library(options INTERFACE)

add_compile_options(-fdeclspec -fms-extensions -Wno-ctad-maybe-unsupported)

## Flags for Clang and GCC.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(options INTERFACE
        -fno-exceptions
        -fno-rtti
        -fdeclspec

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

        ## NULL Errors.
        -Werror=nonnull # Passing NULL to nonnull parameter.

        ## Memory Errors.
        -Werror=address              # Suspicious use of addresses.
        -Werror=init-self            # Initialization of a variable with itself.
        -Werror=uninitialized

        ## Return type.
        -Werror=return-type
        #-Wmissing-noreturn

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

        ## This is typically nonsense if you’re not an idiot.
        -Wno-ctad-maybe-unsupported
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

## Enable ASAN if requested.
if (ENABLE_ASAN)
    target_compile_options(options INTERFACE -fsanitize=address)
    target_link_options(options INTERFACE -fsanitize=address)
endif()

## Debug/Release flags.
if (NOT MSVC)
#[[
    target_compile_definitions(options INTERFACE
        $<$<CONFIG:DEBUG>:_GLIBCXX_DEBUG>
    )
]]
    target_compile_options(options INTERFACE
        $<$<CONFIG:DEBUG>:-O0 -g3 -glldb -funwind-tables>
        $<$<CONFIG:RELEASE>:-O3 -march=native>
    )
    target_link_options(options INTERFACE
        $<$<CONFIG:DEBUG>:-O0 -g3 -glldb -Wl,-export-dynamic -funwind-tables>
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

## Find clopts.
include(CheckIncludeFileCXX)
check_include_file_cxx(clopts.hh HAVE_CLOPTS_HH)
if (NOT HAVE_CLOPTS_HH)
    include(FetchContent)
	message(STATUS "Clopts not found, downloading ...")
	FetchContent_Declare(clopts
		GIT_REPOSITORY https://github.com/Sirraide/clopts.git
		GIT_TAG v2.0.2
		SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/libs/clopts"
    )
	FetchContent_MakeAvailable(clopts)
    target_include_directories(options INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/libs/clopts/include")
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
set(SOURCE_LLVM_BUILD_DIR "${PROJECT_SOURCE_DIR}/llvm/out")
set(LLVM_DIR "${SOURCE_LLVM_BUILD_DIR}/lib/cmake/llvm")
set(MLIR_DIR "${SOURCE_LLVM_BUILD_DIR}/lib/cmake/mlir")
set(Clang_DIR "${SOURCE_LLVM_BUILD_DIR}/lib/cmake/clang")

## Set these or be prepared for horrible linker errors.
##
## As of now (LLVM version 17.0.0), AddLLVM.cmake adds `-Wl,-rpath-link,${LLVM_LIBRARY_OUTPUT_INTDIR}`
## to the linker command line. The problem with this is that, if LLVM_LIBRARY_OUTPUT_INTDIR is not
## defined, which, by default, it isn’t, you end up with `-Wl,-rpath-link,`, as a result of which the
## compiler driver just throws an `-rpath-link` with no argument in the middle of the linker command
## line, which causes whatever file or option after it to be be swallowed.
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX}")

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
file(GLOB_RECURSE hlir_lib_sources "${PROJECT_SOURCE_DIR}/lib/HLIR/*.cc")
add_subdirectory(include/source/HLIR)
add_mlir_dialect_library(MLIRSourceHLIR
    ${hlir_lib_sources}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/source/HLIR

    DEPENDS
    MLIRHLIROpsIncGen

    LINK_LIBS PUBLIC MLIRIR
)

include_directories(SYSTEM "${CMAKE_CURRENT_BINARY_DIR}/include")

## Our tablegen executable.
#[[add_llvm_executable(srcc-tblgen "${PROJECT_SOURCE_DIR}/src/Tablegen.cc")
target_link_libraries(srcc-tblgen PRIVATE options)]]

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
file(GLOB_RECURSE sources lib/*.cc)

## As well as everything in ‘include’, if it exists.
if (EXISTS "${PROJECT_SOURCE_DIR}/include")
    file(GLOB_RECURSE includes include/*.hh)
    list(APPEND sources ${includes})
endif()

## Add the executable.
add_llvm_library(source ${sources})
target_link_libraries(source PRIVATE options cpptrace-lib)
target_precompile_headers(source PRIVATE include/source/Support/Utils.hh)

add_llvm_executable(srcc src/Main.cc)
set_target_properties(srcc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")
target_link_libraries(srcc PRIVATE options cpptrace-lib source)
target_precompile_headers(srcc PRIVATE include/source/Support/Utils.hh)

set(SOURCE_RUNTIME_NAME "__src_runtime")
set(SOURCE_RUNTIME_PATH "${PROJECT_SOURCE_DIR}/modules/${SOURCE_RUNTIME_NAME}")
set(SOURCE_BUILTIN_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules")
add_compile_definitions(
    "__SRCC_DIR=\"${PROJECT_SOURCE_DIR}\""
    "__SRCC_RT_PATH=\"${SOURCE_RUNTIME_PATH}.o\""
    "__SRCC_BUILTIN_MODULE_PATH=\"${SOURCE_BUILTIN_MODULE_PATH}\""
    "__SRCC_RUNTIME_NAME=\"${SOURCE_RUNTIME_NAME}\""
    "__SRCC_CLANG_EXE=\"${LLVM_TOOLS_BINARY_DIR}/clang\""
    "__SRCC_PROJECT_DIR_NAME=\"Source-MLIR\""
    "__SRCC_OBJ_FILE_EXT=\".o\""
    "__SRCC_ASM_FILE_EXT=\".s\""
    "__LLVM_DEBUG_METADATA_VERSION=3"
)

#llvm_update_compile_flags(srcc)
#mlir_check_all_link_libraries(srcc)

## Tests
include(CTest)
include(fchk/cmake/FCHKDiscoverTests.cmake)
set(FCHK_EXE_PATH "${PROJECT_SOURCE_DIR}/fchk/out/fchk")
FCHKAddAllTestsInDir(
    IN tests
    PATTERN "*.src"
    PREFIX  "///"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    ARGS -P re -P nocap -P captype -l . -D "srcc=$<TARGET_FILE:srcc>"
    RECURSIVE
)

## ============================================================================
##  Stdlib.
## ============================================================================
set(source-compiler "$<TARGET_FILE:srcc>")

add_custom_target(source-std ALL)

## Build runtime.
add_custom_target(source-runtime
    COMMAND "${source-compiler}"
        "${PROJECT_SOURCE_DIR}/std/__src_runtime.src"
        --colour always
        --dir "${SOURCE_BUILTIN_MODULE_PATH}"
        -O3
        --nostdrt --nostdinc
    DEPENDS srcc
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    BYPRODUCTS
        "${SOURCE_RUNTIME_PATH}.o"
    SOURCES "${PROJECT_SOURCE_DIR}/std/__src_runtime.src"
    COMMENT "Compiling Source runtime"
)

add_dependencies(source-std source-runtime)

## Build a stdlib module.
function(source_stdlib_module name)
    list(TRANSFORM ARGN PREPEND "source-module-")
    string(REPLACE "." "/" path "${name}")
    add_custom_target("source-module-${name}"
        COMMAND "${source-compiler}"
            "${PROJECT_SOURCE_DIR}/std/${path}.src"
            --colour always
            --dir "${SOURCE_BUILTIN_MODULE_PATH}"
            -O3
        DEPENDS srcc source-runtime ${ARGN}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        BYPRODUCTS
            "${SOURCE_BUILTIN_MODULE_PATH}/${name}.o"
        SOURCES "${PROJECT_SOURCE_DIR}/std/${path}.src"
        COMMENT "Compiling standard library module '${name}'"
    )
    add_dependencies(source-std "source-module-${name}")
endfunction()

#source_stdlib_module(libc)
