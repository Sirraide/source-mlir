#ifndef SOURCE_DRIVER_DRIVER_HH
#define SOURCE_DRIVER_DRIVER_HH

#include <source/Core.hh>
#include <source/Support/Utils.hh>

namespace src {
struct CompileOptions {
    enum struct Action : u8 {
        /// Compile everything to libraries/executables.
        Compile,

        /// Execute the code. Any modules will still be emitted to disk.
        Execute,

        /// Print the assembly output for every module/executable.
        EmitASM,

        /// Print the AST of every module/executable.
        PrintAST,

        /// Print all exported declarations of every module.
        PrintExports,

        /// Print the HLIR of every module/executable.
        PrintHLIR,

        /// Print the LLVM IR of every module/executable.
        PrintLLVM,

        /// Print the MLIR of every module during the LLVM lowering process.
        PrintLLVMLowering,

        /// Print the lowered HLIR of every module/executable.
        PrintLoweredHLIR,

        /// Run sema only. This is useful for regression tests.
        Sema,
    };

    /// Module output directory.
    fs::path module_output_dir = fs::current_path();

    /// Output name of the executable.
    fs::path executable_output_name = "a.out";

    /// Features to enable/disable for the target. These are applied
    /// after any host features if the opt level is set to -O4.
    StringMap<bool> target_features;

    /// The action to perform.
    Action action = Action::Compile;

    /// Optimisation level.
    u8 opt_level = 0;

    /// Number of threads to use for compilation. If this is 0, the
    /// number of threads will be determined automatically.
    u16 threads = 0;

    /// Debug C++ header imports.
    bool debug_cxx : 1 = false;

    /// Automatically import the runtime module in every module
    /// and executable.
    bool include_runtime : 1 = true;

    /// Stop after parsing (and before sema).
    bool syntax_only : 1 = false;

    /// Automatically add the standard import paths to the module
    /// search path.
    bool use_default_import_paths : 1 = true;

    /// Use colours in diagnostic messages and other output.
    bool use_colours : 1 = true;

    /// Print HLIR using the generic assembly format.
    bool use_generic_assembly_format : 1 = false;

    /// Verify HLIR. This should be enabled by default. Disabling this
    /// allows printing invalid HLIR for debugging purposes, but may lead
    /// to miscompilations if any actual code is generated from it!
    bool verify_hlir : 1 = true;
};

/// Compiler driver. This is the main entry point for the compiler.
class Driver {
protected:
    Driver() = default;
    auto Impl();

public:
    Driver(const Driver&) = delete;
    Driver(Driver&&) = delete;
    Driver& operator=(const Driver&) = delete;
    Driver& operator=(Driver&&) = delete;
    ~Driver() = default;

    /// This needs to delete the derived class.
    void operator delete(Driver*, std::destroying_delete_t);

    /// Create a new driver.
    static auto Create(CompileOptions options) -> std::unique_ptr<Driver>;

    /// Add an import path to use for module lookup.
    void add_import_path(fs::path path);

    /// Compile a set of files.
    ///
    /// This supports building multiple modules and at most one executable
    /// concurrently. Any module dependencies between the files will be
    /// resolved automatically. Any files that are not part of a logical
    /// module are combined and compiled to an executable as though they
    /// were one file.
    ///
    /// \param files The files to compile.
    /// \return -1 if there was an error. Otherwise, 0, unless the Execute
    /// action was specified in the options, in which case the return value
    /// will be the bottom 8 bits of the exit code of the program.
    int compile(std::vector<File*> files);

    /// Compile a set of files.
    ///
    /// \see compile(std::vector<File*>)
    int compile(std::vector<fs::path> file_paths);

    /// Look up a module and print its contents to stdout.
    ///
    /// This also works for builtin modules such as the runtime
    /// as well as for C++ header imports.
    ///
    /// \param name The name of the module to look up. This must
    ///        be in angle brackets if it is a C++ header.
    ///
    /// \return `false` if there was an error.
    bool describe_module(StringRef name);
};
} // namespace src

#endif // SOURCE_DRIVER_DRIVER_HH
