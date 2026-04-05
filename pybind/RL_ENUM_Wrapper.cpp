#include "RL_ENUM_Wrapper.h"
#include "enum_state.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iostream>

RL_ENUM_Wrapper::RL_ENUM_Wrapper(std::shared_ptr<Lattice<int>> lattice)
    : m_lattice(lattice), m_num_rows(lattice->numRows()),
      m_current_R(0.0), m_has_solution(false),
      m_last_nonzero(0), m_temp(0.0),
      m_total_steps(0), prev_k(-1),
      backtrack_count(0), solution_count(0),
      m_episode_best_norm(std::numeric_limits<double>::max()) {
    
    if (!m_lattice) {
        throw std::invalid_argument("Lattice pointer cannot be null");
    }
    //std::cout << m_lattice;
    // Initialize array sizes
    m_num_rows = m_lattice->numRows();
    m_r = std::make_unique<long[]>(m_num_rows + 1);
    m_weight.resize(m_num_rows, 0);
    m_coeff_vector.resize(m_num_rows, 0);
    m_temp_vec.resize(m_num_rows, 0);
    m_center.resize(m_num_rows, 0.0);
    m_sigma.resize(m_num_rows + 1, std::vector<double>(m_num_rows, 0.0));
    m_rho.resize(m_num_rows + 1, 0.0);
    
    // Initialize current state
    m_current_state.num_rows = m_num_rows;
    m_current_state.best_norm = std::numeric_limits<double>::max();
}
void RL_ENUM_Wrapper::print_current_vectors() const {
    std::cout << "=== Current vector info ===" << std::endl;
    
    // Print coefficient vector
    std::cout << "Coefficient vector temp_vec: [";
    for (long i = 0; i < std::min(10L, m_num_rows); ++i) {
        std::cout << m_temp_vec[i];
        if (i < m_num_rows - 1 && i < 9) std::cout << ", ";
    }
    if (m_num_rows > 10) std::cout << ", ...";
    std::cout << "]" << std::endl;
    
    // Compute and print lattice vector
    if (m_lattice) {
        auto lattice_vector = m_lattice->mulVecBasis(m_temp_vec);
        std::cout << "Lattice vector: [";
        for (size_t i = 0; i < std::min((size_t)10, lattice_vector.size()); ++i) {
            std::cout << lattice_vector[i];
            if (i < lattice_vector.size() - 1 && i < 9) std::cout << ", ";
        }
        if (lattice_vector.size() > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
        
        // Compute norm
        double norm_sq = 0.0;
        for (const auto& x : lattice_vector) {
            norm_sq += static_cast<double>(x) * static_cast<double>(x);
        }
        std::cout << "Vector norm: " << std::sqrt(norm_sq) << std::endl;
    }
}
void RL_ENUM_Wrapper::reset(double R) {
    m_current_R = R;
    m_has_solution = false;
    m_last_nonzero = 0;
    m_temp = 0.0;
    m_total_steps = 0;
    //std::cout << m_lattice;
    // Reset arrays
    std::fill(m_weight.begin(), m_weight.end(), 0);
    std::fill(m_coeff_vector.begin(), m_coeff_vector.end(), 0);
    std::fill(m_temp_vec.begin(), m_temp_vec.end(), 0);
    std::fill(m_center.begin(), m_center.end(), 0.0);
    
    for (auto& row : m_sigma) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    
    std::fill(m_rho.begin(), m_rho.end(), 0.0);
    
    // Initialize r array
    for (long i = 0; i < m_num_rows; ++i) {
        m_r[i] = i;
    }
    
    // Key fix: ENUM starts from the bottom, so k should be initialized to m_num_rows-1
    // instead of starting from 0
    m_current_state.current_k = m_num_rows - 1;
    
    // Key fix: initialize temp_vec[m_num_rows-1] = 1 (instead of temp_vec[0] = 1)
    // because the ENUM algorithm starts searching from the bottom
    if (m_num_rows > 0) {
        m_temp_vec[m_num_rows - 1] = 1;
    }
    
    // Reset current state
    m_current_state = EnumState();
    m_current_state.num_rows = m_num_rows;
    m_current_state.radius = R;
    m_current_state.best_norm = std::numeric_limits<double>::max();
    m_current_state.current_k = m_num_rows - 1;  // Important: start from bottom
    m_current_state.current_rho = 0.0;
    m_current_state.current_center = 0.0;
    m_current_state.has_solution = false;

    // Reset tracking variables
    prev_k = m_num_rows - 1;
    backtrack_count = 0;
    solution_count = 0;
    m_episode_best_norm = std::numeric_limits<double>::max();

    m_tried_coeffs_history.clear();
    
    /*std::cout << "RL_ENUM_Wrapper reset complete: R=" << R 
              << ", k=" << m_current_state.current_k 
              << ", num_rows=" << m_num_rows << std::endl;*/
}

// New: decode_action implementation
long RL_ENUM_Wrapper::decode_action(long action, double center) const {
    // Action decoding: map discrete action to coefficient value
    // action range is typically [-5, 5], corresponding to 11 discrete actions
    long base_coeff = static_cast<long>(std::round(center));
    
    // Ensure action is within a reasonable range
    // We assume action is an already-offset value (-5 to +5)
    // If you need to map [0, 10] to [-5, 5], it can be done as:
    // long offset = action - 5;
    // long chosen_coeff = base_coeff + offset;
    
    // Use action directly as offset
    long chosen_coeff = base_coeff + action;
    
    return chosen_coeff;
}
double RL_ENUM_Wrapper::calculate_immediate_reward(double prev_rho) {
    double reward = 0.0;
    long curr_k = m_current_state.current_k;

    // 1. Reward for finding a NEW best (shorter) vector — delta-based
    if (m_current_state.found_solution &&
        m_current_state.best_norm < m_episode_best_norm - 1e-6) {
        double improvement_ratio = (m_episode_best_norm - m_current_state.best_norm)
                                   / m_episode_best_norm;
        reward += 500.0 * improvement_ratio;
        m_episode_best_norm = m_current_state.best_norm;
    }

    // 2. Small reward for descending to a deeper level (efficient tree traversal)
    if (curr_k < prev_k) {
        reward += 0.5;
    }

    // 3. Per-step cost (encourage efficiency — fewer nodes visited is better)
    reward -= 0.02;

    return reward;
}
bool RL_ENUM_Wrapper::check_termination() const {
    // 1. Solution found and it is very short (GH bound approximation)
    if (m_has_solution && m_current_state.best_norm < m_current_R * 0.01) {
        return true;
    }

    // 2. Search space exhausted (k has reached n)
    if (m_current_state.current_k >= m_num_rows) {
        return true;
    }

    // NOTE: Do NOT add a condition on current_rho here.
    // After backtracking, current_rho at the parent level is always <= R (it was
    // a valid node). A rho-based heuristic termination was previously written as
    // "m_current_R * 00.0" which equals 0.0 and terminates on ANY nonzero rho —
    // this was a bug that prevented any backtracking. The ENUM algorithm's natural
    // termination (k >= n_rows) is sufficient.

    return false;
}

std::tuple<double, bool, std::string> RL_ENUM_Wrapper::step(long action) {
    if (m_current_state.terminated) {
        return std::make_tuple(0.0, true, "Already terminated");
    }
    
    m_total_steps++;

    // Record previous k and rho BEFORE the step
    prev_k = m_current_state.current_k;
    double prev_rho = m_current_state.current_rho;
    
    // Execute one step of core ENUM logic
    bool should_terminate = execute_enum_step(action);
    
    // Update current state
    update_state_record();
    
    // Compute reward (using actual immediate reward calculation)
    double reward = calculate_immediate_reward(prev_rho);
    
    // Check whether to terminate
    bool done = should_terminate || check_termination();
    m_current_state.terminated = done;
    
    // Generate info string
    std::stringstream info;
    info << "Step: " << m_total_steps 
         << ", k: " << m_current_state.current_k
         << ", rho: " << m_current_state.current_rho
         << "/" << m_current_state.radius
         << ", center: " << m_current_state.current_center
         << ", found_solution: " << m_current_state.found_solution
         << ", best_norm: " << m_current_state.best_norm;
    
    return std::make_tuple(reward, done, info.str());
}

bool RL_ENUM_Wrapper::execute_enum_step(long action) {
    long k = m_current_state.current_k;
    
    /*std::cout << "=== ENUM Step Debug ===" << std::endl;
    std::cout << "k = " << k << std::endl;
    std::cout << "temp_vec[" << k << "] = " << m_temp_vec[k] << std::endl;
    std::cout << "center[" << k << "] = " << m_center[k] << std::endl;*/
    
    // Compute difference
    double diff = static_cast<double>(m_temp_vec[k]) - m_center[k];
    /*std::cout << "diff = " << diff << std::endl;
    std::cout << "diff^2 = " << diff * diff << std::endl;*/
    
    // Check m_B[k]
    double B_k = m_lattice->m_B[k];
    //std::cout << "m_B[" << k << "] = " << B_k << std::endl;
    
    // Compute current rho
    m_temp = diff * diff;
    m_rho[k] = m_rho[k + 1] + m_temp * B_k;
    //std::cout << "rho[" << k+1 << "] = " << m_rho[k+1] << std::endl;
    //std::cout << "rho[" << k << "] = " << m_rho[k] << std::endl;
    
    // Check sigma and center computation
    /*if (k > 0) {
        std::cout << "sigma[" << k+1 << "][" << k << "] = " << m_sigma[k+1][k] << std::endl;
    }*/
    
    // Check if radius condition is satisfied
    if (m_rho[k] <= m_current_R) {
        if (k == 0) {
            // Solution found
            m_has_solution = true;
            m_current_state.found_solution = true;
            
            // Update best coefficients
            for (long i = 0; i < m_num_rows; ++i) {
                m_coeff_vector[i] = m_temp_vec[i];
            }
            
            // Update best norm
            auto v = m_lattice->mulVecBasis(m_coeff_vector);
            double norm_sq = 0.0;
            for (const auto& x : v) {
                norm_sq += static_cast<double>(x) * static_cast<double>(x);
            }
            m_current_state.best_norm = std::sqrt(norm_sq);
            
            // Shrink radius (keep slightly smaller than current best solution)
            if (m_rho[0] > 0) {
                m_current_R = std::fmin(0.99 * m_rho[0], m_current_R);
            }
            
            return false; // Continue searching (there may be a better solution)
        } else {
            // Move down one level
            k--;
            m_current_state.current_k = k;
            
            if (m_r[k + 1] >= m_r[k]) {
                m_r[k] = m_r[k + 1];
            }
            
            // Update sigma
            update_sigma(k);
            
            // Update center value
            update_center(k);
            
            // Use RL action to select coefficient
            double center_val = m_center[k];
            long chosen_coeff = decode_action(action, center_val);
            
            m_temp_vec[k] = chosen_coeff;
            m_weight[k] = 1;
            
            // Record attempted coefficient
            m_tried_coeffs_history.push_back(chosen_coeff);
            if (m_tried_coeffs_history.size() > 100) {
                m_tried_coeffs_history.erase(m_tried_coeffs_history.begin());
            }
            
            return false;
        }
    } else {
        // Backtrack
        k++;
        m_current_state.current_k = k;
        
        if (k == m_num_rows) {
            // Search complete
            if (!m_has_solution) {
                std::fill(m_coeff_vector.begin(), m_coeff_vector.end(), 0);
            }
            return true; // Terminate
        } else {
            m_r[k] = k;
            
            if (k >= m_last_nonzero) {
                m_last_nonzero = k;
                m_temp_vec[k]++;
            } else {
                // Use original backtracking strategy
                if (m_temp_vec[k] > m_center[k]) {
                    m_temp_vec[k] -= m_weight[k];
                } else {
                    m_temp_vec[k] += m_weight[k];
                }
                m_weight[k]++;
            }
            
            // Record attempted coefficient
            m_tried_coeffs_history.push_back(m_temp_vec[k]);
            if (m_tried_coeffs_history.size() > 100) {
                m_tried_coeffs_history.erase(m_tried_coeffs_history.begin());
            }
            
            return false;
        }
    }
}

void RL_ENUM_Wrapper::update_sigma(long k) {
    /*std::cout << "update_sigma: k=" << k << ", r[" << k << "]=" << m_r[k] << std::endl;*/
    
    for (long i = m_r[k]; i > k; --i) {
        double old_sigma = m_sigma[i + 1][k];
        double mu = m_lattice->m_mu[i][k];
        double temp_vec_i = m_temp_vec[i];
        double new_sigma = old_sigma + mu * temp_vec_i;
        
        /*std::cout << "  i=" << i << ": sigma=" << old_sigma 
                  << " + mu=" << mu << " * temp_vec=" << temp_vec_i
                  << " = " << new_sigma << std::endl;*/
                  
        m_sigma[i][k] = new_sigma;
    }
}

void RL_ENUM_Wrapper::update_center(long k) {
    m_center[k] = -m_sigma[k + 1][k];
    /*std::cout << "update_center: k=" << k 
              << ", sigma[" << k+1 << "][" << k << "]=" << m_sigma[k+1][k]
              << ", center=" << m_center[k] << std::endl;*/
}

// New: compute immediate reward (using previous rho value for comparison)


// Fix: this function is no longer needed, since we already have calculate_immediate_reward(double prev_rho)
// double RL_ENUM_Wrapper::calculate_reward(bool found_solution, bool backtrack) const {
//     // This function is deprecated, use calculate_immediate_reward instead
//     return 0.0;
// }

void RL_ENUM_Wrapper::update_state_record() {
    long k = m_current_state.current_k;
    
    // Update current state
    m_current_state.current_k = k;
    m_current_state.current_rho = m_rho[k];
    m_current_state.current_center = m_center[k];
    m_current_state.radius = m_current_R;
    m_current_state.has_solution = m_has_solution;
    
    // Update GSO information
    m_current_state.gs_norms.clear();
    // Note: assumes m_lattice's m_B is public or accessible via getter
    // Actual implementation needs to be adjusted based on Lattice class design
    for (long i = 0; i < m_num_rows; ++i) {
        m_current_state.gs_norms.push_back(m_lattice->m_B[i]);
    }
    
    // Update mu value for current level
    m_current_state.mu_values.clear();
    for (long i = 0; i < m_num_rows; ++i) {
        m_current_state.mu_values.push_back(m_lattice->m_mu[i][k]);
    }
    
    // Update attempted coefficient history
    m_current_state.tried_coeffs = m_tried_coeffs_history;
    
    // Update current coefficient
    m_current_state.current_coeffs = m_temp_vec;
    
    // Update best coefficients
    m_current_state.best_coeffs = m_coeff_vector;
    
    // Update array state
    m_current_state.r.assign(m_r.get(), m_r.get() + m_num_rows);
    m_current_state.weight = m_weight;
    m_current_state.center_array = m_center;
    m_current_state.rho_array = m_rho;
    m_current_state.last_nonzero = m_last_nonzero;
}

EnumState RL_ENUM_Wrapper::get_state() const {
    EnumState state = m_current_state;
    
    // Ensure state is up to date
    state.current_k = m_current_state.current_k;
    state.current_rho = m_rho[state.current_k];
    state.current_center = m_center[state.current_k];
    state.radius = m_current_R;
    state.has_solution = m_has_solution;
    state.best_norm = m_current_state.best_norm;
    
    // Fill in GSO information
    state.gs_norms.clear();
    for (long i = 0; i < m_num_rows; ++i) {
        state.gs_norms.push_back(m_lattice->m_B[i]);
    }
    
    // Fill in mu values for current level
    state.mu_values.clear();
    long k = state.current_k;
    if (k >= 0 && k < m_num_rows) {
        for (long i = 0; i < m_num_rows; ++i) {
            state.mu_values.push_back(m_lattice->m_mu[i][k]);
        }
    }
    
    // Fill in attempted coefficients
    state.tried_coeffs = m_tried_coeffs_history;
    
    // Fill in current coefficient
    state.current_coeffs = m_temp_vec;
    
    return state;
}

std::vector<long> RL_ENUM_Wrapper::get_best_coeffs() const {
    return m_coeff_vector;
}

// New: get_best_vector implementation
std::vector<int> RL_ENUM_Wrapper::get_best_vector() const {
    // Call lattice's mulVecBasis method
    return m_lattice->mulVecBasis(m_coeff_vector);
}

bool RL_ENUM_Wrapper::is_terminated() const {
    return m_current_state.terminated;
}

// New: get_statistics implementation
RL_ENUM_Wrapper::Statistics RL_ENUM_Wrapper::get_statistics() const {
    Statistics stats;
    stats.total_steps = m_total_steps;
    stats.best_norm = m_current_state.best_norm;
    
    // Compute backtrack count (simple estimate: number of times k increases)
    // Requires history recording; simplified here
    stats.backtracks = 0;  // Actual implementation requires tracking
    
    stats.solutions_found = m_has_solution ? 1 : 0;
    
    // rho history (record recent values)
    // Actual implementation needs to maintain a rho history array
    
    return stats;
}

// New: compute_rho implementation (if needed)
double RL_ENUM_Wrapper::compute_rho(long k) const {
    if (k < 0 || k >= m_num_rows) return 0.0;
    
    double temp_val = static_cast<double>(m_temp_vec[k]) - m_center[k];
    temp_val *= temp_val;
    return m_rho[k + 1] + temp_val * m_lattice->m_B[k];
}