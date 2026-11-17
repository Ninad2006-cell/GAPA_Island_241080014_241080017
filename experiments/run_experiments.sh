set -e  # Exit immediately if any command fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "================================================================================"
echo "Parallel Genetic Algorithm Research - Experiment Automation"
echo "Working Directory: ${ROOT_DIR}"
echo "================================================================================"

# Step 1: Clean and Configure with Modern CMake
echo "[1/4] Configuring Modern CMake with CUDA 13.0 and sm_75..."
rm -rf "${BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75

# Step 2: Build All Executables
echo "[2/4] Compiling All Targets in Parallel..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

# Step 3: Run the Unified Benchmark Driver
echo "[3/4] Running Benchmark Suite..."
POP_SIZE=${1:-100}
GENE_LENGTH=${2:-10}
GENERATIONS=${3:-50}
TRIALS=${4:-3}
ISLANDS=${5:-4}
THREADS=${6:-4}
CSV_OUT="${RESULTS_DIR}/benchmark_results.csv"

"${BUILD_DIR}/benchmark" "${POP_SIZE}" "${GENE_LENGTH}" "${GENERATIONS}" "${TRIALS}" "${ISLANDS}" "${THREADS}" "${CSV_OUT}"

# Also run individual approach logs for raw console outputs
echo "Collecting individual approach trial logs in ${RESULTS_DIR}..."
for trial in $(seq 1 "${TRIALS}"); do
    SEED=$((42 + (trial - 1) * 10007))
    "${BUILD_DIR}/single" "${POP_SIZE}" "${GENE_LENGTH}" "${GENERATIONS}" "${SEED}" > "${RESULTS_DIR}/single_trial${trial}.txt" 2>&1
    "${BUILD_DIR}/sm"     "${POP_SIZE}" "${GENE_LENGTH}" "${GENERATIONS}" "${THREADS}" "${SEED}" > "${RESULTS_DIR}/sm_trial${trial}.txt" 2>&1
    "${BUILD_DIR}/m"      "${POP_SIZE}" "${GENE_LENGTH}" "${GENERATIONS}" "${THREADS}" "${SEED}" > "${RESULTS_DIR}/m_trial${trial}.txt" 2>&1
    "${BUILD_DIR}/mnm"    "${POP_SIZE}" "${GENE_LENGTH}" "${GENERATIONS}" "${THREADS}" "${SEED}" > "${RESULTS_DIR}/mnm_trial${trial}.txt" 2>&1
    "${BUILD_DIR}/island" "${POP_SIZE}" "${GENE_LENGTH}" "${GENERATIONS}" "${ISLANDS}" "${SEED}" > "${RESULTS_DIR}/island_trial${trial}.txt" 2>&1
done

# Step 4: Run Statistical Python Analysis and Generate Graphs
echo "[4/4] Generating Scientific Graphs and Summary Tables..."
python3 "${SCRIPT_DIR}/analyze_results.py" "${CSV_OUT}"

echo "================================================================================"
echo "All Experiments Successfully Completed!"
echo "CSV Data: ${CSV_OUT}"
echo "Visualizations: ${RESULTS_DIR}/graphs/"
echo "================================================================================"
