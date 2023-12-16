#ifndef SOURCE_SUPPORT_TOPOLOGICAL_SORT_HH
#define SOURCE_SUPPORT_TOPOLOGICAL_SORT_HH

#include <source/Support/Utils.hh>

namespace src::utils {
/// FIXME: Replace w/ std::mdspan once we can use that.
struct TSortInputGraph {
    std::span<bool> data;
    usz nodes;

    auto operator[](usz from, usz to) const -> bool& {
        Assert(from < nodes);
        Assert(to < nodes);
        return data[from * nodes + to];
    }
};

/// \brief Perform a topological sort on a graph.
///
/// This partitions a graph into groups of nodes that are mutually
/// independent. The graph is represented as an adjacency matrix such
/// that graph[from, to] is `true` iff there is an edge from `from` to
/// `to`.
///
/// Note that cyclic graphs do not admit of topological sorting, so
/// this function will return false if the graph is cyclic. Note that
/// the \p out_groups parameter may still be modified in either case.
///
/// \param[inout] graph The graph to sort. May be modified by this function.
/// \param out_groups A list of groups to which the node sets will be added.
/// \return `true` if a topological sort could be performed, `false` otherwise.
template <typename VectorType = SmallVector<usz>, typename Groups>
[[nodiscard]] bool TopologicalSort(TSortInputGraph graph, Groups& out_groups)
requires requires { out_groups.emplace_back(VectorType{}); }
{
    Buffer<bool> removed(graph.nodes, false);
    for (;;) {
        VectorType nodes;

        /// Find all nodes that have no incoming edges.
        for (usz i = 0; i < graph.nodes; i++) {
            if (removed[i]) continue;
            bool has_incoming_edges = false;
            for (usz j = 0; j < graph.nodes; j++) {
                if (graph[j, i]) {
                    has_incoming_edges = true;
                    break;
                }
            }

            if (not has_incoming_edges) {
                removed[i] = true;
                nodes.push_back(i);
            }
        }

        /// No progress was made.
        if (nodes.empty()) break;

        /// Remove all edges from the graph that originate from the nodes
        /// we just added to the group.
        for (auto n : nodes)
            for (usz i = 0; i < graph.nodes; i++)
                if (graph[n, i])
                    graph[n, i] = false;

        /// Add all nodes to a new group.
        out_groups.emplace_back(std::move(nodes));
    }

    /// If there are any edges left, there is a cycle.
    return rgs::none_of(graph.data, std::identity{});
}
}

#endif // SOURCE_TOPOLOGICALSORT_HH
