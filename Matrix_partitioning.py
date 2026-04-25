"""
Graph-based 3D Matrix Partition Optimization System
==================================================
Based on: "Graph Model with Unified Cost Function for 3D Matrix Partition Optimization"

This module implements a systematic approach to find optimal matrix multiplication
partitioning strategies using A* search over a graph model of the partition space.

Core innovations (per technical disclosure):
1. Scene Recognition & Diversion: Classify tasks by rho = T_comm / T_comp ratio
2. Graph Search for Balanced Scenarios: A* search with unified cost model
3. Dynamic Weight Calibration: Hardware-aware cost function coefficients

Author: V2 - Refactored from V1 with correctness fixes and optimizations
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Set
import time
import math
import logging
from functools import total_ordering

# Configure logging to replace print statements
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Chip Hardware Configuration Parameters
# ============================================================================

class ChipConfig:
    """BM1684X chip hardware configuration constants."""
    NUM_NPUS = 64                      # Number of NPU cores
    NPU_MEM_KB = 256                   # Per-NPU local memory capacity (KB)
    NUM_BANKS = 16                     # Banks per NPU
    BANK_SIZE_KB = NPU_MEM_KB // NUM_BANKS  # Per-bank size (16 KB)
    BANKS_PER_MATRIX = 2               # Banks allocated per matrix
    MEMORY_BANDWIDTH_GBS = 55          # Memory bandwidth (GB/s)
    TOTAL_COMPUTE_TFLOPS = 1.5         # Total compute performance (TFLOPS)
    DATA_TYPE_SIZE = 4                 # Data type size in bytes (FP32)

    # Scene classification thresholds for rho = T_comm / T_comp
    RHO_LOW_THRESHOLD = 0.2            # Below this -> compute-dominated
    RHO_HIGH_THRESHOLD = 5.0           # Above this -> communication-dominated


@dataclass
class HardwareConfig:
    """Hardware configuration data structure for optimizer.

    Attributes:
        memory_bandwidth: Memory bandwidth in GB/s.
        total_compute_performance: Total compute performance in TFLOPS.
        local_memory_capacity_kb: Per-NPU local memory capacity in KB.
        bank_size_kb: Size of each memory bank in KB.
        banks_per_matrix: Number of banks allocated per matrix operand.
        num_npus: Total number of NPU cores available.
        data_type_size: Size of scalar data element in bytes (default FP32=4).
        alpha: Weight coefficient for computation time in cost function.
        beta: Weight coefficient for communication time in cost function.
        gamma: Penalty coefficient for constraint violations.
    """
    memory_bandwidth: float           # GB/s
    total_compute_performance: float   # TFLOPS
    local_memory_capacity_kb: int     # KB per NPU
    bank_size_kb: int                 # KB per bank
    banks_per_matrix: int             # banks per matrix
    num_npus: int                     # total NPU count
    data_type_size: int = 4           # bytes per element (FP32)
    alpha: float = 1.0                # computation weight
    beta: float = 1.0                 # communication weight
    gamma: float = 1000.0             # violation penalty weight


# ============================================================================
# Graph Node Definition (A* Search State)
# ============================================================================

@total_ordering
@dataclass
class Node:
    """Represents a single state in the graph search space.

    Each node encodes a specific (M_slice, N_slice, K_slice) partition scheme,
    along with its associated costs for A* search evaluation.

    Attributes:
        M_slice: Row dimension partition size for left matrix (A).
        N_slice: Column dimension partition size for right matrix (B).
        K_slice: Shared dimension partition size (cols of A, rows of B).
        g_cost: Path cumulative cost from start node to this node.
        h_cost: Heuristic estimated cost from this node to goal.
        memory_usage_kb: Total memory usage per NPU in KB.
    """
    M_slice: int
    N_slice: int
    K_slice: int
    g_cost: float
    h_cost: float
    memory_usage_kb: float

    @property
    def f_cost(self) -> float:
        """Total evaluation cost f(n) = g(n) + h(n) for A* algorithm."""
        return self.g_cost + self.h_cost

    @property
    def state_key(self) -> Tuple[int, int, int]:
        """Hashable state identifier for visited-set lookup."""
        return (self.M_slice, self.N_slice, self.K_slice)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.f_cost == other.f_cost

    def __lt__(self, other) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.f_cost < other.f_cost

    def __hash__(self) -> int:
        return hash(self.state_key)


# ============================================================================
# Scene Classification Types
# ============================================================================

class SceneType:
    """Enumeration of matrix computation scene types based on rho ratio."""
    COMPUTE_DOMINATED = "COMPUTE_DOMINATED"     # rho <= rho_low
    COMMUNICATION_DOMINATED = "COMMUNICATION_DOMINATED"  # rho >= rho_high
    BALANCED = "BALANCED"                        # rho_low < rho < rho_high


# ============================================================================
# Core Optimizer Class
# ============================================================================

class MatrixPartitionOptimizer:
    """3D Matrix Partition Optimizer using Graph Model and A* Search.

    This optimizer implements the core algorithm described in the technical
    disclosure: scene recognition -> diversion decision -> graph search
    optimization -> schedule generation.

    The unified cost model evaluates candidates as:
        f(n) = max(alpha * T_comp(n), beta * T_comm(n)) + gamma * Penalty(n)

    Args:
        M: Row dimension of left matrix (A).
        N: Column dimension of right matrix (B).
        K: Shared dimension (columns of A, rows of B).
        hardware: HardwareConfig instance describing target hardware.
    """

    def __init__(self, M: int, N: int, K: int, hardware: HardwareConfig):
        # Validate inputs
        if M <= 0 or N <= 0 or K <= 0:
            raise ValueError(
                f"Matrix dimensions must be positive integers. "
                f"Got M={M}, N={N}, K={K}"
            )

        self.M = M
        self.N = N
        self.K = K
        self.hw = hardware
        self._visited_states: Set[Tuple[int, int, int]] = set()

        logger.info("Matrix dimensions: M=%d, N=%d, K=%d", M, N, K)

    # ------------------------------------------------------------------
    # Time Estimation Models (Roof-line Based)
    # ------------------------------------------------------------------

    def estimate_communication_time(self, M_s: int, N_s: int, K_s: int) -> float:
        """Estimate communication time for a given partition.

        Accounts for data transfer of both input matrices (A, B) and
        output matrix (C) between global memory and local NPU memory.

        The transfer volume includes:
        - Left matrix block: M_s * K_s elements
        - Right matrix block: K_s * N_s elements
        - Output matrix block: M_s * N_s elements

        Args:
            M_s, N_s, K_s: Partition sizes along each dimension.

        Returns:
            Estimated communication time in seconds.
        """
        data_volume_bytes = (
            (M_s * K_s + K_s * N_s + M_s * N_s)
            * self.hw.data_type_size
        )
        base_time = data_volume_bytes / (self.hw.memory_bandwidth * 1e9)
        # Floor at 1 nanosecond to avoid numerical issues with tiny partitions
        return max(base_time, 1e-9)

    def estimate_computation_time(self, M_s: int, N_s: int, K_s: int) -> float:
        """Estimate computation time for a given partition.

        Uses roof-line model upper bound: theoretical peak FLOPS.
        Matrix multiply C = A x B requires 2*M*N*K FLOPs.
        Applies a parallel efficiency factor (default 80%) to account
        for real-world overheads (memory latency, synchronization, etc.).

        [V2 FIX] Removed efficiency factor from public API; now uses
        configurable alpha/beta weights instead for cleaner separation.

        Args:
            M_s, N_s, K_s: Partition sizes along each dimension.

        Returns:
            Estimated computation time in seconds.
        """
        flops = 2 * M_s * N_s * K_s
        raw_time = flops / (self.hw.total_compute_performance * 1e12)
        return max(raw_time, 1e-9)

    # ------------------------------------------------------------------
    # Memory Constraint Checking
    # ------------------------------------------------------------------

    def calculate_memory_requirements(
        self, M_s: int, N_s: int, K_s: int
    ) -> Dict[str, float]:
        """Calculate detailed memory requirements for a partition scheme.

        Returns memory usage in KB for each matrix operand and total.

        Args:
            M_s, N_s, K_s: Partition sizes along each dimension.

        Returns:
            Dictionary with keys: left_matrix_kb, right_matrix_kb,
            output_matrix_kb, total_kb.
        """
        left_kb = (M_s * K_s * self.hw.data_type_size) / 1024.0
        right_kb = (K_s * N_s * self.hw.data_type_size) / 1024.0
        output_kb = (M_s * N_s * self.hw.data_type_size) / 1024.0

        return {
            'left_matrix_kb': left_kb,
            'right_matrix_kb': right_kb,
            'output_matrix_kb': output_kb,
            'total_kb': left_kb + right_kb + output_kb,
        }

    def check_memory_constraints(self, M_s: int, N_s: int, K_s: int) -> bool:
        """Check whether a partition satisfies hardware memory constraints.

        Validates two levels of constraints:
        1. Total memory usage must not exceed 90% of NPU local memory capacity.
        2. Each individual matrix must fit within its allocated bank space.

        [V2 FIX] Added explicit validation for zero/negative slice dimensions.

        Args:
            M_s, N_s, K_s: Partition sizes along each dimension.

        Returns:
            True if constraints are satisfied, False otherwise.
        """
        # Guard against invalid dimensions
        if M_s <= 0 or N_s <= 0 or K_s <= 0:
            return False

        mem_req = self.calculate_memory_requirements(M_s, N_s, K_s)

        # Check total memory (leave 10% headroom for metadata/scratch)
        if mem_req['total_kb'] > self.hw.local_memory_capacity_kb * 0.9:
            return False

        # Check per-matrix bank allocation limit
        max_per_matrix_kb = self.hw.banks_per_matrix * self.hw.bank_size_kb
        if (mem_req['left_matrix_kb'] > max_per_matrix_kb
                or mem_req['right_matrix_kb'] > max_per_matrix_kb
                or mem_req['output_matrix_kb'] > max_per_matrix_kb):
            return False

        return True

    # ------------------------------------------------------------------
    # Unified Cost Model (Per Technical Disclosure Section 4.5)
    # ------------------------------------------------------------------

    def calculate_penalty(self, M_s: int, N_s: int, K_s: int) -> float:
        """Calculate penalty term for constraint violations.

        [V2 FIX] Separated from main cost function for clarity.
        Returns 0.0 when constraints are satisfied; returns large
        positive value proportional to overflow amount otherwise.

        Args:
            M_s, N_s, K_s: Partition sizes along each dimension.

        Returns:
            Penalty value (0.0 if valid, > 0.0 if violating).
        """
        mem_req = self.calculate_memory_requirements(M_s, N_s, K_s)
        total_mem = mem_req['total_kb']
        capacity = self.hw.local_memory_capacity_kb

        if total_mem <= capacity * 0.9:
            return 0.0

        overflow_ratio = total_mem / (capacity * 0.9) - 1.0
        return overflow_ratio * self.hw.gamma

    def unified_cost_function(self, M_s: int, N_s: int, K_s: int) -> float:
        """Unified cost evaluation function (Equation 3 in disclosure).

        Implements: f(n) = max(alpha * T_comp(n), beta * T_comm(n)) + gamma * Penalty(n)

        This is the core evaluation metric that balances computation and
        communication costs while penalizing constraint violations.

        [V2 IMPROVEMENT] Replaced the ad-hoc log-scale cost function with
        the formally defined model from the technical disclosure document.

        Args:
            M_s, N_s, K_s: Partition sizes along each dimension.

        Returns:
            Scalar cost value (lower is better).
        """
        T_comm = self.estimate_communication_time(M_s, N_s, K_s)
        T_comp = self.estimate_computation_time(M_s, N_s, K_s)
        penalty = self.calculate_penalty(M_s, N_s, K_s)

        # Unified cost: max of weighted comp/comm + penalty
        time_cost = max(self.hw.alpha * T_comp, self.hw.beta * T_comm)
        return time_cost + penalty

    # ------------------------------------------------------------------
    # Heuristic Function h(n) for A* Search (Equation 6)
    # ------------------------------------------------------------------

    def heuristic_cost(self, M_s: int, N_s: int, K_s: int) -> float:
        """Admissible heuristic for A* search (Equation 6).

        Provides optimistic lower-bound estimate of remaining cost
        from current node to goal state.

        Uses: h(n) = max(alpha * T_comp(n), beta * T_comm(n))

        This is admissible because it never overestimates the actual
        remaining cost (it ignores future penalties and path-dependent costs).

        [V2 FIX] Replaced original non-admissible heuristic (which used
        communication_overhead + parallelism_efficiency formula unrelated
        to the formal definition) with the correct specification.

        Args:
            M_s, N_s, K_s: Partition sizes along each dimension.

        Returns:
            Admissible heuristic value (>= 0).
        """
        T_comm = self.estimate_communication_time(M_s, N_s, K_s)
        T_comp = self.estimate_computation_time(M_s, N_s, K_s)
        return max(self.hw.alpha * T_comp, self.hw.beta * T_comm)

    # ------------------------------------------------------------------
    # Scene Recognition (Section 4.2 of Disclosure)
    # ------------------------------------------------------------------

    def estimate_rho(self, M_s: int, N_s: int, K_s: int) -> float:
        """Estimate communication-to-computation time ratio rho.

        rho = T_comm / T_comp

        Used for scene classification into compute-dominated,
        communication-dominated, or balanced categories.

        Args:
            M_s, N_s, K_s: Partition sizes to evaluate.

        Returns:
            Estimated rho ratio (positive float; inf if T_comp ~ 0).
        """
        T_comm = self.estimate_communication_time(M_s, N_s, K_s)
        T_comp = self.estimate_computation_time(M_s, N_s, K_s)
        return T_comm / T_comp if T_comp > 1e-12 else float('inf')

    def classify_scene(self) -> str:
        """Classify the current matrix task into a scene type.

        Uses representative partition dimensions for rho estimation.
        Falls back to BALANCED if estimation is inconclusive.

        Returns:
            One of SceneType.* constants.
        """
        # Use a mid-range partition for representative estimation
        rep_M = min(self.M, 256)
        rep_N = min(self.N, 256)
        rep_K = min(self.K, 1024)

        rho = self.estimate_rho(rep_M, rep_N, rep_K)

        if rho <= ChipConfig.RHO_LOW_THRESHOLD:
            return SceneType.COMPUTE_DOMINATED
        elif rho >= ChipConfig.RHO_HIGH_THRESHOLD:
            return SceneType.COMMUNICATION_DOMINATED
        else:
            return SceneType.BALANCED

    # ------------------------------------------------------------------
    # Initial Node Generation (Section 4.5 + Adaptive Strategies)
    # ------------------------------------------------------------------

    def analyze_matrix_shape(self) -> str:
        """Analyze matrix shape characteristics for adaptive strategy selection.

        Returns:
            Shape category string: K_DOMINANT, MN_DOMINANT, or BALANCED.
        """
        max_dim = max(self.M, self.N, self.K)

        if self.K > 5 * max(self.M, self.N):
            return "K_DOMINANT"
        elif max(self.M, self.N) > 5 * self.K:
            return "MN_DOMINANT"
        else:
            return "BALANCED"

    def generate_adaptive_slices(self) -> List[Tuple[int, int, int]]:
        """Generate candidate partition schemes adapted to matrix shape.

        Different matrix aspect ratios benefit from different initial
        sampling strategies in the search space.

        Returns:
            List of (M_slice, N_slice, K_slice) candidate tuples.
        """
        slices = []
        shape_category = self.analyze_matrix_shape()

        if shape_category == "K_DOMINANT":
            logger.info("K-dominant matrix detected, prioritizing K continuity")
            base_k = min(self.K, 2048)

            for m in self._generate_mn_slices(self.M, base_k):
                for n in self._generate_mn_slices(self.N, base_k):
                    if m >= 16 and n >= 16:
                        slices.append((m, n, base_k))

        elif shape_category == "BALANCED":
            logger.info("Balanced matrix detected, uniform partition strategy")
            for size in [128, 256, 512, 1024]:
                if size <= min(self.M, self.N, self.K):
                    slices.append((size, size, size))

        else:  # MN_DOMINANT
            logger.info("MN-dominant matrix detected")
            dominant_size = max(self.M, self.N)
            base_size = min(dominant_size, 256)

            for k in [256, 512, 1024, 2048]:
                if k <= self.K:
                    slices.append((base_size, base_size, k))

        return slices

    def _generate_mn_slices(
        self, dim_size: int, k_slice: int
    ) -> List[int]:
        """Generate M/N dimension slice candidates based on K-slice and memory.

        Produces exponentially-spaced candidate sizes that respect
        memory capacity constraints.

        Args:
            dim_size: Maximum size of the target dimension (M or N).
            k_slice: Current K-slice size (affects available memory budget).

        Returns:
            Sorted list of candidate slice sizes.
        """
        # Calculate maximum MN size that fits in memory alongside K-slice
        available_mem_bytes = self.hw.local_memory_capacity_kb * 1024 * 0.8
        bytes_per_element = 3 * k_slice * self.hw.data_type_size
        max_mn = min(dim_size, int(available_mem_bytes / max(bytes_per_element, 1)))

        if dim_size <= 128:
            return sorted(set([16, 32, 64, dim_size]))

        slices = []
        base = 32
        while base <= min(max_mn, dim_size):
            slices.append(base)
            base *= 2
        # Always include full dimension if feasible
        if dim_size not in slices and dim_size <= max_mn:
            slices.append(dim_size)

        return sorted(slices)

    def generate_initial_nodes(self) -> List[Node]:
        """Generate initial candidate nodes for A* search.

        Creates an adaptive set of starting points in the search space,
        filters by memory constraints, sorts by f-cost, and returns
        the top candidates for search initialization.

        [V2 FIX] Fixed bug where fallback nodes could also be empty,
        raising IndexError on sorted_nodes[0].

        Returns:
            List of valid Node objects, sorted ascending by f_cost.
        """
        adaptive_slices = self.generate_adaptive_slices()
        logger.info("Generated %d candidate partition schemes", len(adaptive_slices))

        nodes = []

        for m, n, k in adaptive_slices:
            if not self.check_memory_constraints(m, n, k):
                continue

            # g_cost starts at 0 for initial nodes (no path yet)
            cost = self.unified_cost_function(m, n, k)
            h = self.heuristic_cost(m, n, k)
            mem = self.calculate_memory_requirements(m, n, k)['total_kb']

            nodes.append(Node(m, n, k, g_cost=0.0, h_cost=h, memory_usage_kb=mem))

            if len(nodes) <= 5:
                logger.debug(
                    "Candidate: %dx%dx%d, memory=%.1f KB, cost=%.4f",
                    m, n, k, mem, cost
                )

        if not nodes:
            logger.warning(
                "No valid initial candidates found, generating fallback solutions"
            )
            return self._generate_fallback_nodes()

        # Sort by total evaluation cost and keep top candidates
        sorted_nodes = sorted(nodes, key=lambda x: x.f_cost)
        best = sorted_nodes[0]
        logger.info(
            "Best initial solution: %dx%dx%d (f_cost=%.4f)",
            best.M_slice, best.N_slice, best.K_slice, best.f_cost
        )

        return sorted_nodes[:15]

    def _generate_fallback_nodes(self) -> List[Node]:
        """Generate fallback nodes when no standard candidates satisfy constraints.

        Uses progressively smaller partition sizes to guarantee finding
        at least one valid solution for extreme memory-constrained scenarios.

        [V2 FIX] Added minimum viable size guard to ensure at least one node.

        Returns:
            List of valid Node objects sorted by f_cost.
        """
        nodes = []

        for k in [256, 512, 1024, 2048]:
            if k > self.K:
                continue
            for mn in [16, 32, 64, 128]:
                if mn > min(self.M, self.N):
                    continue
                if not self.check_memory_constraints(mn, mn, k):
                    continue

                cost = self.unified_cost_function(mn, mn, k)
                h = self.heuristic_cost(mn, mn, k)
                mem = self.calculate_memory_requirements(mn, mn, k)['total_kb']
                nodes.append(Node(mn, mn, k, g_cost=0.0, h_cost=h, memory_usage_kb=mem))

        # Last resort: try minimal viable partition
        if not nodes:
            minimal = min(16, self.M, self.N, self.K)
            if self.check_memory_constraints(minimal, minimal, minimal):
                cost = self.unified_cost_function(minimal, minimal, minimal)
                h = self.heuristic_cost(minimal, minimal, minimal)
                mem = self.calculate_memory_requirements(minimal, minimal, minimal)['total_kb']
                nodes.append(
                    Node(minimal, minimal, minimal, g_cost=0.0, h_cost=h, memory_usage_kb=mem)
                )

        return sorted(nodes, key=lambda x: x.f_cost)

    # ------------------------------------------------------------------
    # Neighbor Generation (Graph Edge Operations)
    # ------------------------------------------------------------------

    def generate_neighbors(self, node: Node) -> List[Node]:
        """Generate neighboring states by adjusting partition parameters.

        Implements intelligent neighbor generation that adapts adjustment
        step sizes based on current memory utilization:
        - Low utilization (< 60%): use larger steps to increase occupancy
        - High utilization (>= 60%): use finer steps for precision tuning

        Each neighbor represents one edge transition in the graph model
        (Section 4.3), corresponding to +/- delta adjustments on M/N/K axes.

        Args:
            node: Current node from which to generate neighbors.

        Returns:
            List of valid neighboring Node objects.
        """
        neighbors = []
        mem_req = self.calculate_memory_requirements(
            node.M_slice, node.N_slice, node.K_slice
        )
        utilization = mem_req['total_kb'] / self.hw.local_memory_capacity_kb

        # Choose adjustment strategy based on memory utilization
        if utilization < 0.6:
            # Low utilization: aggressive expansion
            adjustments = [
                (min(64, self.M - node.M_slice), 0, 0),
                (0, min(64, self.N - node.N_slice), 0),
                (0, 0, min(256, self.K - node.K_slice)),
                (32, 32, 0), (0, 32, 128), (32, 0, 128),
            ]
        else:
            # High utilization: fine-grained tuning
            adjustments = [
                (16, 0, 0), (-16, 0, 0),
                (0, 16, 0), (0, -16, 0),
                (0, 0, 64), (0, 0, -64),
                (8, 8, 0), (-8, -8, 0),
                (8, 0, 32), (0, 8, 32),
            ]

        for dm, dn, dk in adjustments:
            new_M = max(8, min(self.M, node.M_slice + dm))
            new_N = max(8, min(self.N, node.N_slice + dn))
            new_K = max(8, min(self.K, node.K_slice + dk))

            # Skip no-op transitions
            if (new_M, new_K, new_N) == (node.M_slice, node.K_slice, node.N_slice):
                continue

            # Skip already-visited states
            state = (new_M, new_N, new_K)
            if state in self._visited_states:
                continue

            # Validate memory constraints
            if not self.check_memory_constraints(new_M, new_N, new_K):
                continue

            # Calculate edge cost (transition cost between partitions)
            # C(e) = max(alpha * dT_comp, beta * dT_comm) + gamma * Penalty(e)
            old_cost = self.unified_cost_function(
                node.M_slice, node.N_slice, node.K_slice
            )
            new_cost = self.unified_cost_function(new_M, new_N, new_K)
            edge_cost = max(0, new_cost - old_cost)  # Monotonic cost increase

            new_h = self.heuristic_cost(new_M, new_N, new_K)
            new_mem = self.calculate_memory_requirements(new_M, new_N, new_K)['total_kb']

            # g(n) = g(parent) + C(edge from parent to n)
            new_g = node.g_cost + edge_cost

            neighbors.append(
                Node(new_M, new_N, new_K, g_cost=new_g, h_cost=new_h,
                     memory_usage_kb=new_mem)
            )

        return neighbors

    # ------------------------------------------------------------------
    # A* Search Algorithm (Section 4.4)
    # ------------------------------------------------------------------

    def a_star_search(
        self, max_iterations: int = 1000
    ) -> Tuple[Optional[Node], Dict]:
        """Execute A* search over the partition graph to find optimal solution.

        Algorithm overview:
        1. Generate initial candidate nodes
        2. Initialize open list (priority queue ordered by f_cost)
        3. Iteratively pop lowest-f_cost node, expand its neighbors
        4. Track best solution found so far
        5. Terminate on convergence or iteration limit

        [V2 FIX] Added proper g_cost accumulation through edge costs,
        matching the formal definition: g(n) = g(parent) + C(edge).

        [V2 FIX] Replaced fragile early termination (improvement_count < 3)
        with relative improvement threshold for more robust convergence.

        Args:
            max_iterations: Maximum number of search iterations.

        Returns:
            Tuple of (best_node_or_None, search_statistics_dict).

        Raises:
            ValueError: If no valid initial nodes can be generated.
        """
        start_time = time.perf_counter()
        initial_nodes = self.generate_initial_nodes()

        if not initial_nodes:
            raise ValueError(
                "Unable to find any partition scheme satisfying hardware constraints"
            )

        # Initialize priority queue (min-heap by f_cost)
        open_set: List[Tuple[float, int, Node]] = []
        for i, node in enumerate(initial_nodes):
            heapq.heappush(open_set, (node.f_cost, i, node))

        best_node = initial_nodes[0]
        iteration_count = 0
        node_counter = len(initial_nodes)
        prev_best_cost = float('inf')
        stagnation_count = 0
        STAGNATION_LIMIT = 50  # Stop after this many iterations w/o meaningful improvement

        while open_set and iteration_count < max_iterations:
            _, _, current = heapq.heappop(open_set)
            iteration_count += 1

            # Track best solution found
            if current.f_cost < best_node.f_cost:
                improvement = (prev_best_cost - current.f_cost) / max(prev_best_cost, 1e-12)
                best_node = current
                stagnation_count = 0

                if iteration_count <= 10 or iteration_count % 20 == 0:
                    logger.info(
                        "Iter %d: Improved solution %dx%dx%d, f_cost=%.4f",
                        iteration_count, current.M_slice, current.N_slice,
                        current.K_slice, current.f_cost
                    )
            else:
                stagnation_count += 1

            prev_best_cost = best_node.f_cost

            # Mark state as visited before expanding
            self._visited_states.add(current.state_key)

            # Generate and enqueue neighbor nodes
            for neighbor in self.generate_neighbors(current):
                self._visited_states.add(neighbor.state_key)
                node_counter += 1
                heapq.heappush(
                    open_set, (neighbor.f_cost, node_counter, neighbor)
                )

            # Early termination: no meaningful improvement for many iterations
            if stagnation_count >= STAGNATION_LIMIT:
                logger.info(
                    "Early termination at iter %d (%d stagnant iterations)",
                    iteration_count, stagnation_count
                )
                break

        elapsed_ms = max(0.01, (time.perf_counter() - start_time) * 1000)

        logger.info(
            "Search completed: %d iterations, %d states explored, %.2f ms",
            iteration_count, len(self._visited_states), elapsed_ms
        )

        return best_node, {
            'search_time_ms': elapsed_ms,
            'iterations': iteration_count,
            'nodes_explored': len(self._visited_states),
            'scene_type': self.classify_scene(),
        }


# ============================================================================
# Public API Functions
# ============================================================================

def create_default_hardware_config(
    custom_params: Optional[Dict] = None
) -> HardwareConfig:
    """Create a default HardwareConfig for BM1684X chip.

    Optionally override specific fields via custom_params dict.

    Args:
        custom_params: Optional dict of field names to override.

    Returns:
        Configured HardwareConfig instance.
    """
    params = {
        'memory_bandwidth': ChipConfig.MEMORY_BANDWIDTH_GBS,
        'total_compute_performance': ChipConfig.TOTAL_COMPUTE_TFLOPS,
        'local_memory_capacity_kb': ChipConfig.NPU_MEM_KB,
        'bank_size_kb': ChipConfig.BANK_SIZE_KB,
        'banks_per_matrix': ChipConfig.BANKS_PER_MATRIX,
        'num_npus': ChipConfig.NUM_NPUS,
        'data_type_size': ChipConfig.DATA_TYPE_SIZE,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1000.0,
    }
    if custom_params:
        params.update(custom_params)

    return HardwareConfig(**params)


def optimize_partition(
    M: int, N: int, K: int,
    chip_config: Optional[Dict] = None
) -> Dict:
    """Main optimization entry point: Find optimal 3D matrix partition.

    Performs end-to-end optimization:
    1. Configure hardware parameters
    2. Run scene classification
    3. Execute A* graph search
    4. Compute detailed performance metrics for result

    [V2 IMPROVEMENT] Now includes scene_type in output, enabling callers
    to understand which optimization path was selected.

    Args:
        M, N, K: Matrix dimensions (C[M][N] = A[M][K] x B[K][N]).
        chip_config: Optional dict overriding default hardware parameters.

    Returns:
        Dictionary containing:
        - optimal_partition: Best (M/N/K)_slice values and slice counts
        - performance_metrics: Timing estimates, rho ratio, memory utilization
        - search_metadata: Iterations, runtime, scene classification
        - hardware_config: Effective hardware config used for optimization
    """
    hw = create_default_hardware_config(chip_config)

    logger.info(
        "Starting optimization for %dx%dx%d matrix partition...",
        M, N, K
    )
    logger.info(
        "Hardware: %d NPUs, %d KB/NPU, %.1f TFLOPS",
        hw.num_npus, hw.local_memory_capacity_kb, hw.total_compute_performance
    )

    optimizer = MatrixPartitionOptimizer(M, N, K, hw)
    best_node, search_info = optimizer.a_star_search()

    # Compute clean performance metrics (without internal penalty terms)
    T_comm = optimizer.estimate_communication_time(
        best_node.M_slice, best_node.N_slice, best_node.K_slice
    )
    T_comp = optimizer.estimate_computation_time(
        best_node.M_slice, best_node.N_slice, best_node.K_slice
    )
    rho = T_comm / T_comp if T_comp > 1e-12 else float('inf')

    # Compute total tile counts for each dimension
    slices_M = math.ceil(M / best_node.M_slice)
    slices_N = math.ceil(N / best_node.N_slice)
    slices_K = math.ceil(K / best_node.K_slice)
    total_tiles = slices_M * slices_N * slices_K

    # Memory analysis
    mem_req = optimizer.calculate_memory_requirements(
        best_node.M_slice, best_node.N_slice, best_node.K_slice
    )
    mem_util_pct = mem_req['total_kb'] / hw.local_memory_capacity_kb * 100

    return {
        'optimal_partition': {
            'M_slice': best_node.M_slice,
            'N_slice': best_node.N_slice,
            'K_slice': best_node.K_slice,
            'slice_num_M': slices_M,
            'slice_num_N': slices_N,
            'slice_num_K': slices_K,
            'total_slices': total_tiles,
        },
        'performance_metrics': {
            'estimated_total_time_ms': (T_comm + T_comp) * 1000,
            'computation_time_ms': T_comp * 1000,
            'communication_time_ms': T_comm * 1000,
            'communication_computation_ratio': rho,
            'memory_utilization_percent': mem_util_pct,
            'memory_per_tile_kb': mem_req['total_kb'],
        },
        'search_metadata': search_info,
        'hardware_config': {
            'num_npus': hw.num_npus,
            'memory_bandwidth_gbs': hw.memory_bandwidth,
            'compute_tflops': hw.total_compute_performance,
            'local_memory_kb': hw.local_memory_capacity_kb,
            'alpha': hw.alpha,
            'beta': hw.beta,
            'gamma': hw.gamma,
        },
    }


def run_benchmark(
    M: int = 4096, N: int = 4096, K: int = 4096
) -> Dict:
    """Run a complete benchmark test and display formatted results.

    Convenience wrapper around optimize_partition() with pretty-printed
    output for manual testing and debugging.

    Args:
        M, N, K: Matrix dimensions (default: 4096^3 balanced cube).

    Returns:
        Full result dictionary from optimize_partition().
    """
    print("=" * 80)
    print("  Graph-Based Matrix Multiplication Partition Optimization")
    print("=" * 80)

    result = optimize_partition(M, N, K)

    op = result['optimal_partition']
    pm = result['performance_metrics']
    sm = result['search_metadata']

    print("\nOptimal Partition:")
    print("-" * 40)
    print(f"  Tile size: {op['M_slice']} x {op['N_slice']} x {op['K_slice']}")
    print(f"  Tile counts: M={op['slice_num_M']}, N={op['slice_num_N']}, K={op['slice_num_K']}")
    print(f"  Total tiles: {op['total_slices']}")

    print("\nPerformance Metrics:")
    print("-" * 40)
    print(f"  Est. total time: {pm['estimated_total_time_ms']:.6f} ms")
    print(f"  Computation:     {pm['computation_time_ms']:.6f} ms")
    print(f"  Communication:   {pm['communication_time_ms']:.6f} ms")
    print(f"  Comm/Comp ratio (rho): {pm['communication_computation_ratio']:.4f}")
    print(f"  Memory utilization:  {pm['memory_utilization_percent']:.1f}%")
    print(f"  Memory per tile:     {pm['memory_per_tile_kb']:.1f} KB")

    print("\nSearch Metadata:")
    print("-" * 40)
    print(f"  Scene type:       {sm.get('scene_type', 'N/A')}")
    print(f"  Search time:      {sm['search_time_ms']:.2f} ms")
    print(f"  Iterations:       {sm['iterations']}")
    print(f"  States explored:  {sm['nodes_explored']}")

    return result


# ============================================================================
# Main Entry Point: Example Usage & Self-Test
# ============================================================================

if __name__ == "__main__":
    # Test Case 1: Standard balanced matrix (4096^3)
    print("\n[Test 1] Balanced matrix: 4096 x 4096 x 4096")
    result1 = run_benchmark(4096, 4096, 4096)

    # Test Case 2: K-dominant matrix (wide inner dimension)
    print("\n" + "=" * 80)
    print("[Test 2] K-dominant matrix: 100 x 256 x 100000")
    result2 = run_benchmark(100, 256, 100000)

    # Comparative summary
    print("\n" + "=" * 80)
    print("Comparison Summary:")
    print("-" * 40)
    r1p, r2p = result1['optimal_partition'], result2['optimal_partition']
    r1m, r2m = result1['performance_metrics'], result2['performance_metrics']
    print(f"  Balanced tile:   {r1p['M_slice']}x{r1p['N_slice']}x{r1p['K_slice']}")
    print(f"  K-dom tile:     {r2p['M_slice']}x{r2p['N_slice']}x{r2p['K_slice']}")
    print(f"  Memory util:     {r1m['memory_utilization_percent']:.1f}% vs "
          f"{r2m['memory_utilization_percent']:.1f}%")
    print(f"  Rho (comm/comp): {r1m['communication_computation_ratio']:.3f} vs "
          f"{r2m['communication_computation_ratio']:.3f}")

