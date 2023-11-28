#include <llvm/Support/CommandLine.h>
#include <llvm/TableGen/Main.h>
#include <llvm/TableGen/Record.h>
#include <source/Support/Utils.hh>

using namespace llvm;

cl::OptionCategory tblgen_category{"Tblgen options"};
cl::opt<bool> debug{"print-records", cl::desc("Enable debug output"), cl::cat{tblgen_category}};

namespace src::tblgen {
static constexpr llvm::StringLiteral ExprClassName = "Expr";
static constexpr llvm::StringLiteral ExprFieldsName = "fields";
static constexpr llvm::StringLiteral ExprExtraFieldsName = "extra_fields";
static constexpr llvm::StringLiteral TrivialCtorFieldName = "trivial_constructor";

namespace {
struct Generator {
    raw_ostream& OS;
    RecordKeeper& RK;

    struct Class {
        std::string name{};
        Record* rec{};
        SmallVector<Class*> children{};
        readonly_const(bool, is_class, return not children.empty());
    };

    /// Inheritance tree.
    std::deque<Class> classes{};

    template <typename... Args>
    void Write(fmt::format_string<Args...> fmt, Args&&... args) {
        OS << fmt::format(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void Write(usz indent, fmt::format_string<Args...> fmt, Args&&... args) {
        OS << std::string(indent, ' ');
        OS << fmt::format(fmt, std::forward<Args>(args)...);
    }

    auto DagValues(DagInit* dag) {
        struct Field {
            StringInit* type;
            StringInit* name;
        };

        SmallVector<Field, 10> values;
        for (auto [arg, name] : llvm::zip_equal(dag->getArgs(), dag->getArgNames()))
            values.push_back({cast<StringInit>(arg), cast<StringInit>(name)});
        return values;
    }

    void operator()() {
        Write("// This file is generated from {}. Do not modify. //\n\n", RK.getInputFilename());

        /// Build the inheritance tree. This is needed for RTTI.
        for (auto r : RK.getAllDerivedDefinitions(ExprClassName)) {
            /// Get or add a new class.
            auto GetClass = [&](std::string name) -> Class* {
                auto it = rgs::find_if(classes, [&](auto& c) { return c.name == name; });
                if (it == classes.end()) return &classes.emplace_back(name);
                return &*it;
            };

            auto parent_init = r->getValue("parent")->getValue();
            auto parent = parent_init->isComplete() ? parent_init->getAsUnquotedString() : "Expr";
            auto parent_class = GetClass(parent);
            auto name = r->getName();
            auto this_class = GetClass(std::string{name});
            this_class->rec = r;
            parent_class->children.push_back(this_class);
        }

        /// Forward-declare all classes.
        Write("namespace src {{\n");
        for (auto& c : classes) Write("class {};\n", c.name);
        Write("\n");

        /// Emit Kind enum.
        Write("enum struct ExprKind {{\n");
        for (auto& c : classes[0].children) EmitKind(*c);
        Write("}};\n}} // namespace src\n\n");

        /// Emit all classes.
        Write("#ifdef SOURCE_AST_CLASSES\n");
        Write("namespace src {{\n");
        for (auto& c : classes[0].children) EmitClass(*c);
        Write("}} // namespace src\n");
        Write("#endif // SOURCE_AST_CLASSES\n");
    }

    void EmitKind(Class& c) {
        if (c.children.empty()) Write(4, "{},\n", c.name);
        for (auto& ch : c.children) EmitKind(*ch);
    }

    void EmitClass(Class& c) {
        auto rec = c.rec;
        auto fields = DagValues(cast<DagInit>(rec->getValue(ExprFieldsName)->getValue()));
        auto extra_fields = DagValues(cast<DagInit>(rec->getValue(ExprExtraFieldsName)->getValue()));
        auto has_fields = not fields.empty() or not extra_fields.empty();

        /// Get parent class.
        auto parent = rec->getValue("parent")->getValue();
        auto parent_class_name = parent->isComplete() ? parent->getAsUnquotedString() : "Expr";
        Write("class {} : public {} {{\n", rec->getName(), parent_class_name);

        /// Emit fields.
        if (has_fields or not c.is_class) Write("public:\n");
        for (auto [arg, name] : fields)
            Write(4, "{} {};\n", arg->getValue(), name->getValue());
        for (auto [arg, name] : extra_fields)
            Write(4, "{} {}{{}};\n", arg->getValue(), name->getValue());

        /// Emit constructor. The constructor of a base class should be protected.
        if (has_fields) Write("\n");
        if (c.is_class) Write("protected:\n");
        Write(4, "{}(\n", rec->getName());
        if (c.is_class) Write(8, "Kind kind,\n");
        for (auto [arg, name] : fields) Write(8, "{} {},\n", arg->getValue(), name->getValue());
        Write(8, "Location loc\n");
        if (rec->getValueAsBit(TrivialCtorFieldName)) {
            if (c.is_class) Write(4, ") : {}(kind, loc)", parent_class_name);
            else Write(4, ") : {}(Kind::{}, loc)", parent_class_name, rec->getName());
            for (auto [_, name] : fields) {
                Write(",\n");
                Write(8, "{}(std::move({}))", name->getValue(), name->getValue());
            }

            /// Extra constructor body.
            if (auto s = rec->getValueAsString("constructor_body"); not s.empty()) {
                Write(" {{\n");
                Write(8, "{}\n", s);
                Write(4, "}}\n\n");
            } else {
                Write(" {{}}\n\n");
            }
        } else {
            Write(4, ");\n\n");
        }

        /// Emit classof.
        if (c.is_class) {
            StringRef first = [&] -> StringRef {
                auto ch = c.children.front();
                while (ch->is_class) ch = ch->children.front();
                return ch->name;
            }();

            StringRef last = [&] -> StringRef {
                auto ch = c.children.back();
                while (ch->is_class) ch = ch->children.back();
                return ch->name;
            }();

            Write("public:\n");
            if (first == last) {
                Write(4, "static bool classof(const Expr* e) {{ return e->kind == Kind::{}; }}\n", first);
            } else {
                Write(4, "static bool classof(const Expr* e) {{\n");
                Write(8, " return e->kind >= Kind::{} and e->kind <= Kind::{};\n", first, last);
                Write(4, "}}\n");
            }
        } else {
            Write(4, "static bool classof(const Expr* e) {{ return e->kind == Kind::{}; }}\n", rec->getName());
        }
        Write("}}\n\n");

        /// Emit children.
        for (auto ch : c.children) EmitClass(*ch);
    }
};

bool Generate(raw_ostream& OS, RecordKeeper& RK) {
    if (debug.getValue()) OS << RK;
    else Generator{OS, RK}();
    return false;
}
} // namespace
} // namespace src::tblgen

int main(int argc, char** argv) {
    cl::ParseCommandLineOptions(argc, argv);
    return llvm::TableGenMain(argv[0], &src::tblgen::Generate);
}
