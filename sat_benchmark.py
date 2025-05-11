import time
import tracemalloc
import random
from typing import List, Tuple, Optional
from pysat.solvers import Solver
import csv
import gc
import sys

# === CNF Generator ===
def generate_cnf_instance(num_vars: int, num_clauses: int, min_clause_len=1, max_clause_len=3) -> List[List[int]]:
    cnf = []
    for _ in range(num_clauses):
        clause = random.sample(range(1, num_vars + 1), k=random.randint(min_clause_len, max_clause_len))
        clause = [lit if random.random() < 0.5 else -lit for lit in clause]
        cnf.append(clause)
    return cnf

def generate_dataset(level: str, count: int) -> List[Tuple[int, int, List[List[int]]]]:
    dataset = []
    ranges = {
        "low": (5, 10, 10, 20),
        "medium": (20, 40, 40, 80),
        "high": (80, 150, 200, 300)
    }
    min_vars, max_vars, min_clauses, max_clauses = ranges[level]

    for _ in range(count):
        num_vars = random.randint(min_vars, max_vars)
        num_clauses = random.randint(min_clauses, max_clauses)
        cnf = generate_cnf_instance(num_vars, num_clauses)
        dataset.append((num_vars, num_clauses, cnf))
    return dataset

# === Resolution Solver ===
def resolution_solver(cnf: List[List[int]]) -> Optional[bool]:
    from collections import deque
    import time

    MAX_RESOLUTION_TIME = 15.0  # seconds timeout
    MAX_CLAUSE_SIZE = 5        # optional clause size cap

    start_time = time.perf_counter()

    clauses = [frozenset(clause) for clause in cnf]
    seen = set(clauses)
    agenda = deque(clauses)

    while agenda:
        if time.perf_counter() - start_time > MAX_RESOLUTION_TIME:
            print("Resolution timeout.")
            return None

        ci = agenda.popleft()
        for cj in list(seen):  # make a copy to avoid modification during iteration
            if ci == cj:
                continue
            for lit in ci:
                if -lit in cj:
                    resolvent = (ci | cj) - {lit, -lit}

                    # Skip tautologies
                    if any(-l in resolvent for l in resolvent):
                        continue

                    if len(resolvent) > MAX_CLAUSE_SIZE:
                        continue

                    new_clause = frozenset(resolvent)

                    if not new_clause:
                        return False  # contradiction

                    if new_clause not in seen:
                        seen.add(new_clause)
                        agenda.append(new_clause)

    return True  # satisfiable

# === Iterative DP Solver (Robust) ===
def dp_solver(cnf: List[List[int]]) -> Optional[bool]:
    from copy import deepcopy
    def simplify(clauses, assignment):
        changed = True
        while changed:
            changed = False
            # Unit propagation
            unit_clauses = [c for c in clauses if len(c) == 1]
            for unit in unit_clauses:
                lit = next(iter(unit))
                assignment.add(lit)
                new_clauses = []
                for c in clauses:
                    if lit in c:
                        continue
                    new_c = set(c)
                    new_c.discard(-lit)
                    if not new_c and lit not in c:
                        return None
                    new_clauses.append(new_c)
                clauses = new_clauses
                changed = True
        return clauses

    def backtrack(clauses, assignment):
        clauses = simplify(clauses, assignment)
        if clauses is None:
            return False
        if not clauses:
            return True
        # Choose a variable to split on
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                break
            break
        for choice in [var, -var]:
            new_assignment = assignment.copy()
            new_assignment.add(choice)
            new_clauses = [set(c) for c in clauses] + [{choice}]
            if backtrack(new_clauses, new_assignment):
                return True
        return False

    try:
        return backtrack([set(c) for c in cnf], set())
    except RecursionError:
        return None

# === DPLL Solver (PySAT) ===
def dpll_solver(cnf: List[List[int]]) -> Optional[bool]:
    with Solver(name='g4', bootstrap_with=cnf) as solver:
        try:
            return solver.solve()
        except:
            return None

# === Benchmark ===
def benchmark_solver(solver_name: str, solver_fn, dataset: List[Tuple[int, int, List[List[int]]]]):
    results = []
    for idx, (num_vars, num_clauses, cnf) in enumerate(dataset):
        print(f"[{solver_name}] Instance {idx+1}/{len(dataset)}")
        gc.collect()
        tracemalloc.start()
        start_time = time.perf_counter()

        result = None
        try:
            result = solver_fn(cnf)
        except Exception as e:
            print(f"Error in {solver_name} on instance {idx}: {str(e)}", file=sys.stderr)

        elapsed = time.perf_counter() - start_time
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append((solver_name, idx, num_vars, num_clauses, elapsed, peak_mem, str(result)))
    return results

# === Run Benchmark ===
def run_benchmark():
    complexities = ['low', 'medium', 'high']
    all_results = []
    solvers = [
        ("Resolution", resolution_solver),
        ("DP", dp_solver),
        ("DPLL", dpll_solver)
    ]
    dataset_sizes = {
        'low': 100,
        'medium': 100,
        'high': 100
    }

    for level in complexities:
        print(f"\n=== {level.upper()} COMPLEXITY ===")
        dataset = generate_dataset(level, dataset_sizes[level])

        level_results = {}
        for solver_name, solver_fn in solvers:
            res = benchmark_solver(solver_name, solver_fn, dataset)
            level_results[solver_name] = res
            all_results.extend(res)

        # Validate using DPLL as reference
        if "DPLL" in level_results:
            dpll_results = level_results["DPLL"]
            for solver_name in level_results:
                if solver_name == "DPLL":
                    continue
                for i, (_, _, _, _, _, _, res) in enumerate(level_results[solver_name]):
                    dpll_res = dpll_results[i][6]
                    if res != "" and dpll_res != "" and res != dpll_res:
                        print(f"Warning: Result mismatch for {solver_name} instance {i} (got {res}, expected {dpll_res})")
    return all_results

# === CSV Writer ===
def save_results_csv(results, filename="sat_benchmark_results.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Solver", "Instance ID", "Num Vars", "Num Clauses",
            "Time (s)", "Memory (bytes)", "Result"
        ])
        writer.writerows(results)

# === Main ===
if __name__ == "__main__":
    print("Starting SAT solver benchmarking...")
    results = run_benchmark()
    save_results_csv(results)
    print("Benchmarking complete. Results saved to 'sat_benchmark_results.csv'")
