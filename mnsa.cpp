/*
 * Multi-Neighborhood Simulated Annealing (MNSA) for CDP
 * Based on Rosati & Schaerf (2024)
 * Implemented for High-Performance Benchmarking
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
#include <set>

using namespace std;

// --- CONFIGURATION STRUCT ---
struct Config {
    // Annealing Parameters (Table 2 in paper)
    double T0;
    double Tf;
    double alpha;
    int Ns;     // Iterations per temp
    int Na;     // Accepted limit per temp
    
    // Neighborhood Weights (Sigma)
    double sigma_insert;
    double sigma_remove;
    double sigma_swap;
    
    // Biases for Restricted Sets (Section 4.4)
    double b_plus;  // Bias for adding from Candidate List C_R
    double b_minus; // Bias for removing from Restricted Set S_R
    
    // Cost Function Weights (Section 4.1)
    double w_c = 10000.0; // Weight for Capacity Penalty
    double w_of;          // Weight for Objective Function (Dist)
    
    // Mode Selection
    string mode; // "MDG" or "GIS"
};

// --- INSTANCE DATA ---
struct Instance {
    int n;
    double B; // Minimum required capacity
    vector<double> capacities;
    // Flattened distance matrix for cache locality (1D array)
    // Access: distances[i * n + j]
    vector<double> distances; 
    double max_dist;
};

// --- SOLUTION STATE ---
struct Solution {
    vector<int> selected;      // List of selected node indices (S)
    vector<bool> is_selected;  // Boolean mask for O(1) lookups
    double current_cap;        // Sum of capacities
    double min_dist;           // f(S): Minimum distance in S
    double total_cost;         // F(S): The hierarchical cost function
    
    // Cached bottleneck count |E_R| (Section 4.1)
    // Number of edges with length equal to min_dist
    int bottleneck_count; 
};

// --- GLOBAL RANDOM ENGINE ---
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

// --- HELPER: FAST CALCULATIONS ---

// Re-calculate objective and bottlenecks from scratch O(S^2)
// Used after moves to ensure numerical stability
void full_evaluation(Solution& sol, const Instance& inst) {
    if (sol.selected.size() < 2) {
        sol.min_dist = 0;
        sol.bottleneck_count = 0;
        return;
    }

    double min_d = numeric_limits<double>::max();
    int count = 0;

    // First pass: Find Min Dist
    for (size_t i = 0; i < sol.selected.size(); ++i) {
        for (size_t j = i + 1; j < sol.selected.size(); ++j) {
            int u = sol.selected[i];
            int v = sol.selected[j];
            double d = inst.distances[u * inst.n + v];
            if (d < min_d) {
                min_d = d;
                count = 1;
            } else if (abs(d - min_d) < 1e-6) {
                count++;
            }
        }
    }
    sol.min_dist = min_d;
    sol.bottleneck_count = count;
}

// Section 4.1: Cost Function F(S)
// F(S) = w_c * min(0, cap(S) - B) + w_of * f(S) - |E_R|
double get_cost(const Solution& sol, const Instance& inst, const Config& cfg) {
    double penalty = 0.0;
    if (sol.current_cap < inst.B) {
        penalty = sol.current_cap - inst.B; // Negative value
    }
    return (cfg.w_c * penalty) + (cfg.w_of * sol.min_dist) - (double)sol.bottleneck_count;
}

// --- NEIGHBORHOODS (SECTIONS 4.2 - 4.4) ---

// 1. INSERT MOVE
bool move_insert(Solution& cand, const Instance& inst) {
    vector<int> candidates;
    for (int i = 0; i < inst.n; ++i) {
        if (!cand.is_selected[i]) candidates.push_back(i);
    }
    
    if (candidates.empty()) return false;
    
    int v = candidates[uniform_int_distribution<>(0, candidates.size() - 1)(rng)];
    
    cand.selected.push_back(v);
    cand.is_selected[v] = true;
    cand.current_cap += inst.capacities[v];
    return true;
}

// 2. REMOVE MOVE
bool move_remove(Solution& cand, const Instance& inst) {
    if (cand.selected.size() <= 2) return false;
    
    int idx = uniform_int_distribution<>(0, cand.selected.size() - 1)(rng);
    int u = cand.selected[idx];
    
    // Fast remove from vector (swap with back)
    cand.selected[idx] = cand.selected.back();
    cand.selected.pop_back();
    cand.is_selected[u] = false;
    cand.current_cap -= inst.capacities[u];
    return true;
}

// 3. COMPLEX SWAP (Section 4.4)
bool move_swap(Solution& cand, const Instance& inst, const Config& cfg) {
    if (cand.selected.empty()) return false;
    
    // Identify Candidate Pools based on Bias
    bool use_S_R = (uniform_real_distribution<>(0.0, 1.0)(rng) < cfg.b_minus);
    bool use_C_R = (uniform_real_distribution<>(0.0, 1.0)(rng) < cfg.b_plus);
    
    // --- Identify S_R (Restricted Set: Nodes involved in bottleneck edges) ---
    vector<int> S_R;
    vector<int> removable; // Fallback if not using S_R
    
    if (use_S_R) {
        // Find nodes involved in min_dist edges
        vector<bool> in_SR_mask(inst.n, false);
        for (size_t i = 0; i < cand.selected.size(); ++i) {
            for (size_t j = i + 1; j < cand.selected.size(); ++j) {
                int u = cand.selected[i];
                int v = cand.selected[j];
                double d = inst.distances[u * inst.n + v];
                if (abs(d - cand.min_dist) < 1e-6) {
                    if (!in_SR_mask[u]) { S_R.push_back(u); in_SR_mask[u] = true; }
                    if (!in_SR_mask[v]) { S_R.push_back(v); in_SR_mask[v] = true; }
                }
            }
        }
    }
    
    // --- Identify C_R (Candidate List: Nodes that don't worsen min_dist) ---
    vector<int> C_R;
    vector<int> addable; // Fallback
    
    // Populate addable (all non-selected) and C_R if needed
    for (int i = 0; i < inst.n; ++i) {
        if (!cand.is_selected[i]) {
            addable.push_back(i);
            if (use_C_R) {
                // Check if adding 'i' creates an edge smaller than current min_dist
                bool good_candidate = true;
                for (int u : cand.selected) {
                    if (inst.distances[i * inst.n + u] <= cand.min_dist) {
                        good_candidate = false; 
                        break; 
                    }
                }
                if (good_candidate) C_R.push_back(i);
            }
        }
    }

    // --- SELECTION LOGIC ---
    int u = -1; // Node to remove
    int v = -1; // Node to add
    
    // Pick u (Remove)
    if (use_S_R && !S_R.empty()) {
        u = S_R[uniform_int_distribution<>(0, S_R.size() - 1)(rng)];
    } else {
        u = cand.selected[uniform_int_distribution<>(0, cand.selected.size() - 1)(rng)];
    }
    
    // Pick v (Insert)
    if (use_C_R && !C_R.empty()) {
        v = C_R[uniform_int_distribution<>(0, C_R.size() - 1)(rng)];
    } else if (!addable.empty()) {
        v = addable[uniform_int_distribution<>(0, addable.size() - 1)(rng)];
    } else {
        return false; // No nodes to add
    }
    
    // Perform Swap
    for (size_t i = 0; i < cand.selected.size(); ++i) {
        if (cand.selected[i] == u) {
            cand.selected[i] = v;
            break;
        }
    }
    cand.is_selected[u] = false;
    cand.is_selected[v] = true;
    cand.current_cap = cand.current_cap - inst.capacities[u] + inst.capacities[v];
    return true;
}

// --- INITIAL SOLUTION (Random Greedy) ---
Solution generate_initial(const Instance& inst) {
    Solution sol;
    sol.is_selected.assign(inst.n, false);
    sol.current_cap = 0;
    
    vector<int> p(inst.n);
    iota(p.begin(), p.end(), 0);
    shuffle(p.begin(), p.end(), rng);
    
    for (int i : p) {
        if (sol.current_cap >= inst.B) break;
        sol.selected.push_back(i);
        sol.is_selected[i] = true;
        sol.current_cap += inst.capacities[i];
    }
    
    // Ensure minimal size
    int idx = 0;
    while(sol.selected.size() < 2) {
        while(sol.is_selected[p[idx]]) idx++;
        sol.selected.push_back(p[idx]);
        sol.is_selected[p[idx]] = true;
        sol.current_cap += inst.capacities[p[idx]];
    }
    
    full_evaluation(sol, inst);
    return sol;
}

// --- MAIN SOLVER LOOP ---
Solution solve_mnsa(const Instance& inst, Config cfg, double time_limit) {
    // 1. Initialization
    Solution curr = generate_initial(inst);
    curr.total_cost = get_cost(curr, inst, cfg);
    
    Solution best = curr;
    
    double T = cfg.T0;
    auto start_time = chrono::high_resolution_clock::now();
    
    // Main SA Loop
    while (T > cfg.Tf) {
        int ns = 0; // Steps
        int na = 0; // Accepted
        
        while (ns < cfg.Ns && na < cfg.Na) {
            // Check time limit inside inner loop for responsiveness
            // if (ns % 100 == 0) {
            //      auto now = chrono::high_resolution_clock::now();
            //      if (chrono::duration<double>(now - start_time).count() > time_limit) goto end_search;
            // }

            // Copy State
            Solution cand = curr; 
            
            // Select Neighborhood (Roulette Wheel / Weights)
            double r = uniform_real_distribution<>(0.0, 1.0)(rng);
            double w_ins = cfg.sigma_insert;
            double w_rem = cfg.sigma_insert + cfg.sigma_remove;
            // Normalize implicitly by checking bounds (assuming sums to ~1)
            
            bool possible = false;
            
            if (r < w_ins) {
                possible = move_insert(cand, inst);
            } else if (r < w_rem) {
                possible = move_remove(cand, inst);
            } else {
                possible = move_swap(cand, inst, cfg);
            }
            
            if (!possible) {
                ns++;
                continue;
            }
            
            // Evaluate Candidate
            full_evaluation(cand, inst);
            cand.total_cost = get_cost(cand, inst, cfg);
            
            // Delta Calculation
            double delta = cand.total_cost - curr.total_cost;
            
            // Acceptance
            bool accept = false;
            if (delta >= 0) {
                accept = true;
            } else {
                if (uniform_real_distribution<>(0.0, 1.0)(rng) < exp(delta / T)) {
                    accept = true;
                }
            }
            
            if (accept) {
                curr = cand;
                na++;
                
                if (curr.total_cost > best.total_cost) {
                    best = curr;
                }
            }
            ns++;
        }
        
        // Cooling
        T *= cfg.alpha;
        
        // Cutoff Logic (Budget Transfer)
        // If we finished the temp early (na reached limit), add unused steps to next Ns
        if (ns < cfg.Ns) {
            double log_val = log(cfg.Tf / T) / log(cfg.alpha);
            if (abs(log_val) > 1e-9) {
                cfg.Ns += (int)((cfg.Ns - ns) / log_val);
            }
        }
    }

    end_search:;
    return best;
}

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
        // GIS Parameters (Rosati Table 2)
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