/*
 * Multi-Neighborhood Simulated Annealing (MNSA) - High Performance Version
 * Optimization: Incremental Evaluation & Memory Management
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <limits>
#include <numeric>

// Standard Optimization: Tell compiler to assume pointers don't overlap (if applicable)
// and use fast math. Compile with: g++ -O3 -std=c++17 -march=native -o mnsa_opt mnsa_optimized.cpp

using namespace std;

// --- CONFIGURATION ---
struct Config {
    double T0, Tf, alpha;
    int Ns, Na;
    double sigma_insert, sigma_remove, sigma_swap;
    double b_plus, b_minus;
    double w_c = 10000.0, w_of;
    string mode;
};

// --- INSTANCE ---
struct Instance {
    int n;
    double B;
    vector<double> capacities;
    // We keep the flattened matrix. For N=2000, 4M doubles = ~32MB. Fits in L3 cache of modern CPUs.
    vector<double> distances; 
    double max_dist;
};

// --- SOLUTION STATE ---
struct Solution {
    vector<int> selected;      // S
    vector<bool> is_selected;  // O(1) check
    double current_cap;
    double min_dist;           // f(S)
    double total_cost;         // F(S)
    int bottleneck_count;      // |E_R|
};

// Global RNG
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

// --- CRITICAL HOT PATH: DISTANCE LOOKUP ---
// Inline forces the compiler to paste this code directly where it's called
inline double dist(const Instance& inst, int u, int v) {
    return inst.distances[u * inst.n + v];
}

// --- CORE: INCREMENTAL EVALUATION HELPERS ---

// Full Re-calculation (Only used at initialization or as a sanity check)
// O(S^2) - Slow
void full_evaluation(Solution& sol, const Instance& inst, const Config& cfg) {
    if (sol.selected.size() < 2) {
        sol.min_dist = 0;
        sol.bottleneck_count = 0;
    } else {
        double min_d = numeric_limits<double>::max();
        int count = 0;
        // Raw loop for speed
        size_t sz = sol.selected.size();
        for (size_t i = 0; i < sz; ++i) {
            int u = sol.selected[i];
            for (size_t j = i + 1; j < sz; ++j) {
                double d = dist(inst, u, sol.selected[j]);
                if (d < min_d) {
                    min_d = d;
                    count = 1;
                } else if (std::abs(d - min_d) < 1e-6) { // Fast epsilon check
                    count++;
                }
            }
        }
        sol.min_dist = min_d;
        sol.bottleneck_count = count;
    }
    
    // Calculate Cost F(S)
    double penalty = (sol.current_cap < inst.B) ? (sol.current_cap - inst.B) : 0.0;
    sol.total_cost = (cfg.w_c * penalty) + (cfg.w_of * sol.min_dist) - (double)sol.bottleneck_count;
}

// --- MOVES WITH OPTIMIZED LOGIC ---

// 1. INSERT (Incremental-ish)
// We only check edges between the NEW node and EXISTING nodes.
// O(S) instead of O(S^2)
bool try_insert(Solution& curr, const Instance& inst, const Config& cfg, Solution& next_sol) {
    // 1. Pick Candidate
    // Optimization: Instead of building a vector of candidates (slow), try random picking
    int n = inst.n;
    int v = -1;
    
    // Try 10 times to find a non-selected node. 
    // If density is high, we might need the vector approach, but for P << N, this is O(1).
    for(int k=0; k<10; ++k) {
        int r = uniform_int_distribution<>(0, n-1)(rng);
        if(!curr.is_selected[r]) { v = r; break; }
    }
    
    // Fallback: build list if random sampling failed
    if(v == -1) {
        vector<int> candidates;
        candidates.reserve(n - curr.selected.size());
        for(int i=0; i<n; ++i) if(!curr.is_selected[i]) candidates.push_back(i);
        if(candidates.empty()) return false;
        v = candidates[uniform_int_distribution<>(0, candidates.size()-1)(rng)];
    }

    // 2. Build Next Solution (Optimistic Copy)
    next_sol = curr; 
    next_sol.selected.push_back(v);
    next_sol.is_selected[v] = true;
    next_sol.current_cap += inst.capacities[v];

    // 3. Update Metrics incrementally
    // Check edges from v to all u in S
    double new_min = curr.min_dist;
    int new_bn_count = curr.bottleneck_count;
    
    // If S was small (<2), we re-eval fully
    if (curr.selected.size() < 2) {
        full_evaluation(next_sol, inst, cfg);
        return true;
    }

    // Scan edges connected to v
    for (int u : curr.selected) {
        double d = dist(inst, v, u);
        if (d < new_min) {
            new_min = d;
            new_bn_count = 1; // Reset count, this is the new unique min
        } else if (std::abs(d - new_min) < 1e-6) {
            new_bn_count++;
        }
    }
    
    // If new_min is same as old, we just added bottlenecks.
    // If new_min is lower, we reset. 
    // (Note: Insert can NEVER increase min_dist, only decrease or stay same)
    
    next_sol.min_dist = new_min;
    next_sol.bottleneck_count = new_bn_count;
    
    double penalty = (next_sol.current_cap < inst.B) ? (next_sol.current_cap - inst.B) : 0.0;
    next_sol.total_cost = (cfg.w_c * penalty) + (cfg.w_of * next_sol.min_dist) - (double)next_sol.bottleneck_count;
    
    return true;
}

// 2. REMOVE
// Removing is tricky because if we remove a node involved in the minimum distance,
// the min_dist might INCREASE. This requires a scan to find the "next best" min.
// O(S^2) in worst case, but often faster.
bool try_remove(Solution& curr, const Instance& inst, const Config& cfg, Solution& next_sol) {
    if (curr.selected.size() <= 2) return false;
    
    int idx = uniform_int_distribution<>(0, (int)curr.selected.size() - 1)(rng);
    int u_rem = curr.selected[idx];
    
    next_sol = curr;
    // Fast remove
    next_sol.selected[idx] = next_sol.selected.back();
    next_sol.selected.pop_back();
    next_sol.is_selected[u_rem] = false;
    next_sol.current_cap -= inst.capacities[u_rem];
    
    // Re-eval is safest here because min_dist might change globally
    // Optimizing this requires complex data structures (heaps) which might be overkill for N=2000
    full_evaluation(next_sol, inst, cfg);
    return true;
}

// 3. SWAP (The Complex One)
bool try_swap(Solution& curr, const Instance& inst, const Config& cfg, Solution& next_sol) {
    if (curr.selected.empty()) return false;

    // --- Identify S_R and C_R ---
    // Optimization: Don't allocate vectors every time if possible. 
    // But for clarity and the "Restricted" logic, we build them.
    
    bool use_S_R = (uniform_real_distribution<>(0.0, 1.0)(rng) < cfg.b_minus);
    bool use_C_R = (uniform_real_distribution<>(0.0, 1.0)(rng) < cfg.b_plus);
    
    int u_idx = -1; // Index in selected vector
    int v_node = -1;
    
    // A. Pick u (Remove)
    if (use_S_R) {
        vector<int> S_R_indices;
        S_R_indices.reserve(curr.selected.size()); 
        
        for(size_t i=0; i<curr.selected.size(); ++i) {
            int node_i = curr.selected[i];
            // Check if node_i is part of ANY bottleneck edge
            // We can check this by scanning neighbors in S
            bool is_bn = false;
            for(size_t j=0; j<curr.selected.size(); ++j) {
                if(i==j) continue;
                if(std::abs(dist(inst, node_i, curr.selected[j]) - curr.min_dist) < 1e-6) {
                    is_bn = true; break;
                }
            }
            if(is_bn) S_R_indices.push_back(i);
        }
        
        if (!S_R_indices.empty()) {
            u_idx = S_R_indices[uniform_int_distribution<>(0, S_R_indices.size()-1)(rng)];
        }
    }
    
    // Fallback for u
    if (u_idx == -1) {
        u_idx = uniform_int_distribution<>(0, curr.selected.size()-1)(rng);
    }
    
    int u_node = curr.selected[u_idx]; // The actual node ID

    // B. Pick v (Insert)
    if (use_C_R) {
        vector<int> C_R;
        C_R.reserve(inst.n); 
        
        for(int i=0; i<inst.n; ++i) {
            if(curr.is_selected[i]) continue;
            
            bool good = true;
            for(int existing : curr.selected) {
                if (existing == u_node) continue; // Don't check against the node we are removing!
                if (dist(inst, i, existing) <= curr.min_dist) { good = false; break; }
            }
            if(good) C_R.push_back(i);
        }
        
        if(!C_R.empty()) {
            v_node = C_R[uniform_int_distribution<>(0, C_R.size()-1)(rng)];
        }
    }
    
    // Fallback for v
    if(v_node == -1) {
        // Try random sampling first
        for(int k=0; k<10; ++k) {
            int r = uniform_int_distribution<>(0, inst.n-1)(rng);
            if(!curr.is_selected[r]) { v_node = r; break; }
        }
        // Last resort
        if(v_node == -1) {
             vector<int> adds;
             for(int i=0; i<inst.n; ++i) if(!curr.is_selected[i]) adds.push_back(i);
             if(adds.empty()) return false;
             v_node = adds[uniform_int_distribution<>(0, adds.size()-1)(rng)];
        }
    }

    // Apply Swap
    next_sol = curr;
    next_sol.selected[u_idx] = v_node; // Direct overwrite
    next_sol.is_selected[u_node] = false;
    next_sol.is_selected[v_node] = true;
    next_sol.current_cap = next_sol.current_cap - inst.capacities[u_node] + inst.capacities[v_node];
    
    full_evaluation(next_sol, inst, cfg);
    return true;
}


// --- MAIN LOOP ---
Solution solve_mnsa(const Instance& inst, Config cfg, double time_limit) {
    // 1. Initial
    Solution curr;
    curr.is_selected.assign(inst.n, false);
    curr.selected.reserve(inst.n); // Pre-allocate max
    
    // Greedy Construction
    vector<int> p(inst.n);
    iota(p.begin(), p.end(), 0);
    shuffle(p.begin(), p.end(), rng);
    
    curr.current_cap = 0;
    for(int i : p) {
        if(curr.current_cap >= inst.B) break;
        curr.selected.push_back(i);
        curr.is_selected[i] = true;
        curr.current_cap += inst.capacities[i];
    }
    // Ensure 2 nodes
    if(curr.selected.size() < 2) {
        for(int i : p) {
            if(!curr.is_selected[i]) {
                curr.selected.push_back(i);
                curr.is_selected[i] = true;
                curr.current_cap += inst.capacities[i];
                if(curr.selected.size() >= 2) break;
            }
        }
    }
    full_evaluation(curr, inst, cfg);
    
    Solution best = curr;
    
    // 2. Annealing
    double T = cfg.T0;
    auto start_time = chrono::high_resolution_clock::now();
    
    // Pre-calculate weights for roulette
    double w_ins = cfg.sigma_insert;
    double w_rem = cfg.sigma_insert + cfg.sigma_remove;
    
    // Optimization: Reuse "next" object to avoid reallocation
    Solution next_sol;
    next_sol.selected.reserve(inst.n);
    next_sol.is_selected.reserve(inst.n);

    // Iteration counters
    long long total_iter = 0;
    
    while(T > cfg.Tf) {
        int ns = 0;
        int na = 0;
        
        while(ns < cfg.Ns && na < cfg.Na) {
            // Time Check: Only check every 1000 iterations to save clock cycles
            if ((ns & 1023) == 0) { // Bitwise check is faster than %
                 auto now = chrono::high_resolution_clock::now();
                 if (chrono::duration<double>(now - start_time).count() > time_limit) goto end_search;
            }

            bool possible = false;
            double r = uniform_real_distribution<>(0.0, 1.0)(rng);
            
            if (r < w_ins) possible = try_insert(curr, inst, cfg, next_sol);
            else if (r < w_rem) possible = try_remove(curr, inst, cfg, next_sol);
            else possible = try_swap(curr, inst, cfg, next_sol);
            
            if (!possible) { ns++; continue; }
            
            double delta = next_sol.total_cost - curr.total_cost;
            
            if (delta >= 0 || uniform_real_distribution<>(0.0, 1.0)(rng) < exp(delta / T)) {
                curr = next_sol;
                na++;
                if (curr.total_cost > best.total_cost) best = curr;
            }
            ns++;
            total_iter++;
        }
        
        T *= cfg.alpha;
        
        // Adaptive Cooling (Budget Transfer)
        if (ns < cfg.Ns) {
             // Math check to avoid fast-math issues with log(0)
             double ratio = cfg.Tf / T;
             if(ratio > 0) {
                double log_val = log(ratio) / log(cfg.alpha);
                if (abs(log_val) > 1e-9) cfg.Ns += (int)((cfg.Ns - ns) / log_val);
             }
        }
    }
    
    end_search:;
    // cerr << "Total Iterations: " << total_iter << endl; // Debug info
    return best;
}

// --- LOADERS ---
// --- IO UTILITIES ---
// 1. MDG Parser (Edge List, Calculated B)
Instance load_mdg_instance(string filepath, double alpha_B) {
    Instance inst;
    ifstream file(filepath);
    if (!file.is_open()) exit(1);
    
    int M_val;
    file >> inst.n >> M_val;
    
    // Generate Capacities [1,10]
    inst.capacities.resize(inst.n);
    long long seed = 42; 
    double total_cap = 0;
    for(int i=0; i<inst.n; i++) {
        seed = (seed * 1103515245 + 12345) % 2147483648;
        int cap = 1 + (seed % 10);
        inst.capacities[i] = (double)cap;
        total_cap += inst.capacities[i];
    }
    inst.B = floor(alpha_B * total_cap); 

    // Edge List reading
    inst.distances.assign(inst.n * inst.n, 0.0);
    int u, v; double d;
    while (file >> u >> v >> d) {
        if (u >= 0 && u < inst.n && v >= 0 && v < inst.n) {
            inst.distances[u * inst.n + v] = d;
            inst.distances[v * inst.n + u] = d;
        }
    }
    return inst;
}

// 2. GIS Parser (Header, Caps, Full Matrix)
Instance load_gis_instance(string filepath) {
    Instance inst;
    ifstream file(filepath);
    if (!file.is_open()) exit(1);

    // Format: N, B, Caps..., Matrix...
    file >> inst.n >> inst.B;

    inst.capacities.resize(inst.n);
    for(int i=0; i<inst.n; i++) file >> inst.capacities[i];

    inst.distances.resize(inst.n * inst.n);
    for(int i=0; i<inst.n * inst.n; i++) file >> inst.distances[i];

    return inst;
}

// --- MAIN (Copy your existing main here) ---
int main(int argc, char* argv[]) {
    // Usage: ./mnsa <file> <time> <mode> <alpha>
    if (argc < 4) return 1;
    
    string filepath = argv[1];
    double time_limit = stod(argv[2]);
    string mode_arg = argv[3];
    double alpha = (argc > 4) ? stod(argv[4]) : 0.2;
    
    // Auto-detect GIS file extension
    bool is_gis = (filepath.size() >= 4 && filepath.substr(filepath.size()-4) == ".cdp");
    
    Instance inst;
    Config cfg;
    
    if (is_gis) {
        inst = load_gis_instance(filepath);
        cfg.mode = "GIS";
        // GIS Parameters
        cfg.T0 = 123.58; cfg.Tf = 0.0046; cfg.alpha = 0.981;
        cfg.Ns = 1000; cfg.Na = 113;
        cfg.sigma_insert = 0.177; cfg.sigma_remove = 0.012; cfg.sigma_swap = 0.811;
        cfg.b_plus = 0.576; cfg.b_minus = 0.954;
        cfg.w_c = 10000; cfg.w_of = 234.0;
    } else {
        inst = load_mdg_instance(filepath, alpha);
        cfg.mode = "MDG";
        // MDG Parameters
        cfg.T0 = 194.49; cfg.Tf = 0.3758; cfg.alpha = 0.988;
        cfg.Ns = 1000; cfg.Na = 160;
        cfg.sigma_insert = 0.131; cfg.sigma_remove = 0.129; cfg.sigma_swap = 0.740;
        cfg.b_plus = 0.449; cfg.b_minus = 0.359;
        cfg.w_c = 10000; cfg.w_of = 11.0;
    }
    
    Solution sol = solve_mnsa(inst, cfg, time_limit);
    
    cout << "Objective:" << sol.min_dist << endl;
    cout << "Capacity:" << sol.current_cap << endl;
    cout << "Nodes:";
    for (size_t i = 0; i < sol.selected.size(); i++) 
        cout << sol.selected[i] << (i < sol.selected.size()-1 ? "," : "");
    cout << endl;
    return 0;
}