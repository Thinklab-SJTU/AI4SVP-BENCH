#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "PyLatticeEnv.h"
#include "../include/lattice.h"
#include "RL_ENUM_Wrapper.h"
#include "enum_state.h"
namespace py = pybind11;

// Bind the Lattice<int> class (matching the original C++ code)
void bind_lattice_int(py::module &m) {
    // Using Lattice<int> from the original C++ code
    py::class_<Lattice<int>, std::shared_ptr<Lattice<int>>>(m, "LatticeInt")
        .def(py::init<long, long>(), 
             py::arg("n"), py::arg("m"),
             "Create a lattice with n rows and m columns (int version)")
        
        // Expose required methods
        .def("numRows", &Lattice<int>::numRows,
             "Get number of rows (dimension)")
        .def("numCols", &Lattice<int>::numCols,
             "Get number of columns")
        
        .def("computeGSO", &Lattice<int>::computeGSO,
             "Compute Gram-Schmidt orthogonalization")
        
        .def("setSVPChallenge", &Lattice<int>::setSVPChallenge,
             py::arg("dim"), py::arg("seed") = 0,
             "Load SVP challenge basis")
        
        .def("setRandom", &Lattice<int>::setRandom,
             py::arg("n"), py::arg("m"), py::arg("min"), py::arg("max"),
             "Set random basis")
        
        .def("setGoldesteinMayerLattice", &Lattice<int>::setGoldesteinMayerLattice,
             py::arg("p"), py::arg("q"),
             "Set Goldstein-Mayer lattice")
        
        // Algorithms
        .def("ENUM", &Lattice<int>::ENUM,
             py::arg("R"),
             "Run enumeration algorithm with radius R")
        
        // Vector operations
        .def("mulVecBasis", &Lattice<int>::mulVecBasis,
             py::arg("coeff_vector"),
             "Multiply coefficient vector with basis")
        
        // Index calculations
        .def("b1Norm", &Lattice<int>::b1Norm,
             "Get norm of first basis vector")
        
        .def("rhf", &Lattice<int>::rhf,
             "Compute root Hermite factor")
        
        .def("sl", &Lattice<int>::sl,
             "Compute sequence length indicator")
        
        .def("volume", &Lattice<int>::volume,
             py::arg("compute_gso") = true,
             "Compute lattice volume")
        
        .def("LLL", &Lattice<int>::LLL,
             py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             py::arg("start_") = 0, py::arg("end_") = -1, py::arg("h") = 0,
             "Perform LLL reduction")
        
        .def("BKZ", &Lattice<int>::BKZ,
             py::arg("beta"), py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             "Perform BKZ reduction")
        
        .def("HKZ", &Lattice<int>::HKZ,
             py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             "Perform HKZ reduction")
        
        .def("L2", &Lattice<int>::L2,
             py::arg("delta") = 0.75, py::arg("eta") = 0.51,
             "Perform L2 reduction")
        
        .def("deepLLL", &Lattice<int>::deepLLL,
             py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             py::arg("start_") = 0, py::arg("end_") = -1, py::arg("h") = 0,
             "Perform DeepLLL reduction")
        
        .def("potLLL", &Lattice<int>::potLLL,
             py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             "Perform PotLLL reduction")
        
        .def("dualLLL", &Lattice<int>::dualLLL,
             py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             "Perform dual LLL reduction")
        
        .def("dualDeepLLL", &Lattice<int>::dualDeepLLL,
             py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             "Perform dual DeepLLL reduction")
        
        .def("dualPotLLL", &Lattice<int>::dualPotLLL,
             py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             "Perform dual PotLLL reduction")
        
        .def("dualBKZ", &Lattice<int>::dualBKZ,
             py::arg("beta"), py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             "Perform dual BKZ reduction")
        
        .def("deepBKZ", &Lattice<int>::deepBKZ,
             py::arg("beta"), py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             "Perform DeepBKZ reduction")
        
        .def("dualDeepBKZ", &Lattice<int>::dualDeepBKZ,
             py::arg("beta"), py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             "Perform dual DeepBKZ reduction")
        
        .def("potBKZ", &Lattice<int>::potBKZ,
             py::arg("beta"), py::arg("delta") = 0.75, py::arg("compute_gso") = true,
             "Perform PotBKZ reduction")
        
        .def("babaiNearPlane", &Lattice<int>::babaiNearPlane,
             py::arg("target"),
             "Babai's nearest plane algorithm")
        
        // Set maximum loop count
        .def("setMaxLoop", &Lattice<int>::setMaxLoop,
             py::arg("max_loop"),
             "Set maximum loop count for BKZ")
        
        .def("__repr__", [](Lattice<int> &lat) {
            std::ostringstream oss;
            oss << "LatticeInt " << lat.numRows() << "x" << lat.numCols();
            return oss.str();
        })
        /*.def("create_rl_wrapper", [](std::shared_ptr<Lattice<int>> lattice) {
            return std::make_shared<RL_ENUM_Wrapper>(lattice.get());
        }, "Create RL wrapper for this lattice")*/
        .def("__str__", [](Lattice<int> &lat) {
            std::ostringstream oss;
            oss << lat;  // Call the original operator<<
            return oss.str();
        });
}


// Bind the Lattice<double> class (if needed)
void bind_lattice_double(py::module &m) {
    py::class_<Lattice<double>>(m, "LatticeDouble")
        .def(py::init<long, long>())
        .def("numRows", &Lattice<double>::numRows)
        .def("numCols", &Lattice<double>::numCols)
        .def("computeGSO", &Lattice<double>::computeGSO)
        .def("setSVPChallenge", &Lattice<double>::setSVPChallenge)
        .def("setRandom", &Lattice<double>::setRandom)
        .def("ENUM", &Lattice<double>::ENUM)
        .def("mulVecBasis", &Lattice<double>::mulVecBasis)
        .def("b1Norm", &Lattice<double>::b1Norm)
        .def("rhf", &Lattice<double>::rhf)
        .def("__repr__", [](const Lattice<double> &lat) {
            return "<LatticeDouble object>";
        });
}
// Add RL_ENUM_Wrapper binding to the existing binding code

// Bind EnumState

// Helper function for creating Lattice<int> objects
std::shared_ptr<Lattice<int>> create_lattice_int(int n, int m) {
    return std::make_shared<Lattice<int>>(n, m);
}

// Helper function for creating Lattice<double> objects
std::shared_ptr<Lattice<double>> create_lattice_double(int n, int m) {
    return std::make_shared<Lattice<double>>(n, m);
}

PYBIND11_MODULE(lattice_env, m) {
    m.doc() = "Lattice algorithms for SVP with Python bindings";
    
    // Bind Lattice<int> - use this primarily, matches original C++ code
    bind_lattice_int(m);
    
    // Optional: Bind Lattice<double>
    bind_lattice_double(m);
    
    py::class_<EnumState>(m, "EnumState")
        .def(py::init<>())
        .def_readwrite("current_k", &EnumState::current_k)
        .def_readwrite("current_rho", &EnumState::current_rho)
        .def_readwrite("current_center", &EnumState::current_center)
        .def_readwrite("radius", &EnumState::radius)
        .def_readwrite("num_rows", &EnumState::num_rows)
        .def_readwrite("has_solution", &EnumState::has_solution)
        .def_readwrite("gs_norms", &EnumState::gs_norms)
        .def_readwrite("mu_values", &EnumState::mu_values)
        .def_readwrite("tried_coeffs", &EnumState::tried_coeffs)
        .def_readwrite("current_coeffs", &EnumState::current_coeffs)
        .def_readwrite("best_coeffs", &EnumState::best_coeffs)
        .def_readwrite("best_norm", &EnumState::best_norm)
        .def_readwrite("terminated", &EnumState::terminated)
        .def_readwrite("found_solution", &EnumState::found_solution);

// Bind RL_ENUM_Wrapper
    py::class_<RL_ENUM_Wrapper>(m, "RL_ENUM_Wrapper")
        .def(py::init<std::shared_ptr<Lattice<int>>>(),
             py::arg("lattice"),
             "Create environment from LatticeInt object")
        .def("reset", &RL_ENUM_Wrapper::reset, py::arg("R"),
             "Reset ENUM algorithm with radius R")
        .def("step", &RL_ENUM_Wrapper::step, py::arg("action"),
             "Execute one step with given action (coefficient offset)")
        .def("get_state", &RL_ENUM_Wrapper::get_state,
             "Get current ENUM state")
        .def("get_best_coeffs", &RL_ENUM_Wrapper::get_best_coeffs,
             "Get best coefficient vector found so far")
        .def("get_best_vector", &RL_ENUM_Wrapper::get_best_vector,
             "Get best lattice vector found so far")
        .def("is_terminated", &RL_ENUM_Wrapper::is_terminated,
             "Check if ENUM has terminated")
        .def("calculate_immediate_reward", &RL_ENUM_Wrapper::calculate_immediate_reward,
             "Calculate immediate reward for RL")
        .def("__repr__", [](const RL_ENUM_Wrapper& wrapper) {
            return "<RL_ENUM_Wrapper object>";
        })
        .def("__str__", [](RL_ENUM_Wrapper& wrapper) {
            std::ostringstream oss;
            
            // Get vector
            auto coeffs = wrapper.get_best_coeffs();
            
            if (coeffs.empty()) {
                oss << "[]";
                return oss.str();
            }
            
            // Mimic Lattice output format
            oss << "[" << std::endl;
            
            // If multi-dimensional vector, output row by row
            if (coeffs.size() > 1) {
                oss << "[";
                for (size_t i = 0; i < coeffs.size(); ++i) {
                    oss << coeffs[i];
                    if (i < coeffs.size() - 1) oss << ", ";
                }
                oss << "]" << std::endl;
            } else {
                // Single line output
                oss << "[";
                for (size_t i = 0; i < coeffs.size(); ++i) {
                    oss << coeffs[i];
                    if (i < coeffs.size() - 1) oss << ", ";
                }
                oss << "]" << std::endl;
            }
            
            oss << "]";
            return oss.str();
        });
    // Bind Config
    /*py::class_<PyLatticeEnv::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("max_dimension", &PyLatticeEnv::Config::max_dimension)
        .def_readwrite("action_range", &PyLatticeEnv::Config::action_range)
        .def_readwrite("use_pruning", &PyLatticeEnv::Config::use_pruning)
        .def_readwrite("max_steps", &PyLatticeEnv::Config::max_steps);
    
    // Bind PyLatticeEnv - modified to use Lattice<int>
    py::class_<PyLatticeEnv>(m, "LatticeEnv")
        .def(py::init<std::shared_ptr<Lattice<int>>>(),
             py::arg("lattice"),
             "Create environment from LatticeInt object")
        
        // Environment control
        .def("reset", &PyLatticeEnv::reset,
             py::arg("R") = 100.0,
             "Reset the environment with given radius R")
        
        .def("step", &PyLatticeEnv::step,
             py::arg("action"),
             "Take one step with given action (coefficient offset)")
        
        .def("get_state", &PyLatticeEnv::get_state,
             "Get current state features")
        
        .def("print_debug", &PyLatticeEnv::print_debug_info,
             "Print debug information")
        
        // Property access
        .def_property_readonly("dimension", &PyLatticeEnv::get_dimension)
        .def_property_readonly("best_norm", &PyLatticeEnv::get_best_norm)
        .def_property_readonly("solved", &PyLatticeEnv::is_solved)
        .def_property_readonly("current_k", &PyLatticeEnv::get_current_k)
        .def_property_readonly("current_rho", &PyLatticeEnv::get_current_rho)
        
        // Configuration
        .def("set_config", &PyLatticeEnv::set_config,
             py::arg("config"),
             "Set environment configuration");*/
    
    // Helper functions
    m.def("create_lattice_int", &create_lattice_int,
          py::arg("n"), py::arg("m"),
          "Create a new Lattice<int> object (int version)");
    
    m.def("create_lattice_double", &create_lattice_double,
          py::arg("n"), py::arg("m"),
          "Create a new Lattice<double> object (double version)");
    
    // For backward compatibility, keep old function name (but creates int version)
    m.def("create_lattice", [](int n, int m) -> std::shared_ptr<Lattice<int>> {
        auto lattice = std::make_shared<Lattice<int>>(n, m);
        std::cout << "Created lattice " << n << "x" << m 
                  << " (int version) with address: " << lattice.get() << std::endl;
        return lattice;
    }, py::arg("n"), py::arg("m"), "Create Lattice<int> object");
    
    // Vector norm calculation function
    m.def("vector_norm", [](const std::vector<int>& v) {
        double sum = 0.0;
        for (const auto& x : v) {
            sum += static_cast<double>(x) * static_cast<double>(x);
        }
        return std::sqrt(sum);
    }, py::arg("v"), "Compute Euclidean norm of integer vector");
    
    m.def("vector_norm_double", [](const std::vector<double>& v) {
        double sum = 0.0;
        for (const auto& x : v) {
            sum += x * x;
        }
        return std::sqrt(sum);
    }, py::arg("v"), "Compute Euclidean norm of double vector");
    
    // Constants
    m.attr("__version__") = "0.1.0";
    m.attr("DEFAULT_ACTION_RANGE") = py::int_(5);
    m.attr("MAX_DIMENSION") = py::int_(200);
}