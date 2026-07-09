// run_cfm_e2e.cpp
//
// Inference of the end-to-end CFM model: standardization, the Euler loop,
// de-standardization and post-processing all live inside the ONNX graph.
// This program only samples x0 ~ N(0,1) (and u ~ U(0,1) when the graph uses
// stochastic rounding) and performs a single Run.
//
// Build with real ONNXRuntime:
//   g++ -O2 -std=c++17 run_cfm_e2e.cpp -o run_cfm_e2e -I$ORT/include -L$ORT/lib -lonnxruntime
// Build the self-test (no ONNXRuntime, the graph is emulated by a linear net):
//   g++ -O2 -std=c++17 -DSELFTEST run_cfm_e2e.cpp -o run_cfm_e2e_selftest

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef SELFTEST
#include <onnxruntime_cxx_api.h>
#endif

namespace {

// Row-major matrix read from the "name R C" + R rows of C values text format.
struct Mat {
  int rows = 0;
  int cols = 0;
  std::vector<float> data;

  float at(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
};

[[noreturn]] void fail(const std::string& msg) {
  std::cerr << msg << "\n";
  std::exit(1);
}

Mat read_matrix(std::istream& is, const char* expected) {
  std::string name;
  Mat m;
  is >> name >> m.rows >> m.cols;
  if (!is || name != expected) fail(std::string("Expected ") + expected + ", found " + name);

  m.data.resize(static_cast<size_t>(m.rows) * m.cols);
  for (float& v : m.data) is >> v;
  if (!is) fail(std::string("Truncated matrix: ") + expected);
  return m;
}

std::vector<float> read_vector(std::istream& is) {
  std::string name;
  int n = 0;
  is >> name >> n;
  std::vector<float> v(n);
  for (float& x : v) is >> x;
  return v;
}

std::string get_arg(int argc, char** argv, const std::string& key, const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (key == argv[i]) return argv[i + 1];
  return fallback;
}

}  // namespace

// =============================================================================
// The end-to-end network: (cond_phys, x0[, u]) -> x_phys
// =============================================================================
#ifdef SELFTEST
// Self-test emulation: reads scalers.txt + net.txt and redoes standardization,
// the integration loop and de-standardization with a linear core network.
struct E2E {
  bool stochastic = false;  // set by main from the golden file
  int cond_dim = 0, x_dim = 0, in_dim = 0, out_dim = 0, n_steps = 0;
  double cond_eps = 0.0;
  std::vector<float> cond_mean, cond_std, x_mean, x_std, b;
  Mat W;

  E2E(const std::string& scalers_path, const std::string& net_path, int nsteps) : n_steps(nsteps) {
    std::ifstream s(scalers_path);
    if (!s) fail("Cannot open " + scalers_path);
    std::string key;
    s >> key >> cond_dim;
    s >> key >> x_dim;
    s >> key >> cond_eps;
    cond_mean = read_vector(s);
    cond_std = read_vector(s);
    x_mean = read_vector(s);
    x_std = read_vector(s);

    std::ifstream f(net_path);
    if (!f) fail("Cannot open " + net_path);
    f >> key >> in_dim;
    f >> key >> out_dim;
    W = read_matrix(f, "W");
    b = read_vector(f);
  }

  std::vector<float> run(const std::vector<float>& cond_phys, const std::vector<float>& x0, int B,
                         const std::vector<float>& u = {}) const {
    std::vector<float> cs(static_cast<size_t>(B) * cond_dim);
    for (int n = 0; n < B; ++n)
      for (int j = 0; j < cond_dim; ++j) {
        const size_t i = static_cast<size_t>(n) * cond_dim + j;
        cs[i] = (cond_phys[i] - cond_mean[j]) / (cond_std[j] + static_cast<float>(cond_eps));
      }

    std::vector<float> x = x0;
    const float dt = 1.0f / n_steps;
    std::vector<float> h(in_dim);
    for (int step = 0; step < n_steps; ++step) {
      const float tk = (step + 0.5f) * dt;
      for (int n = 0; n < B; ++n) {
        for (int j = 0; j < x_dim; ++j) h[j] = x[static_cast<size_t>(n) * x_dim + j];
        h[x_dim] = tk;
        for (int j = 0; j < cond_dim; ++j) h[x_dim + 1 + j] = cs[static_cast<size_t>(n) * cond_dim + j];
        for (int o = 0; o < out_dim; ++o) {
          float acc = b[o];
          for (int i = 0; i < in_dim; ++i) acc += W.at(o, i) * h[i];
          x[static_cast<size_t>(n) * x_dim + o] += dt * acc;
        }
      }
    }

    for (int n = 0; n < B; ++n)
      for (int j = 0; j < x_dim; ++j) {
        const size_t i = static_cast<size_t>(n) * x_dim + j;
        x[i] = x[i] * x_std[j] + x_mean[j];
      }

    if (stochastic)
      for (size_t i = 0; i < x.size(); ++i) x[i] = std::floor(std::max(0.0f, x[i]) + u[i]);

    return x;
  }

  int xdim() const { return x_dim; }
  bool needs_u() const { return stochastic; }
};
#else
struct E2E {
  Ort::Env env;
  Ort::SessionOptions opts;
  Ort::Session session{nullptr};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<const char*> in_names;
  std::vector<const char*> out_names{"x_phys"};
  int cond_dim_ = 0, x_dim_ = 0;
  bool needs_u_ = false;

  explicit E2E(const std::string& model_path) : env(ORT_LOGGING_LEVEL_WARNING, "cfm_e2e") {
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = Ort::Session(env, model_path.c_str(), opts);

    // The graph has 2 inputs (cond_phys, x0), or 3 with stochastic rounding.
    // Detect it from the metadata so one binary works with both exports.
    const size_t n_in = session.GetInputCount();
    needs_u_ = (n_in == 3);
    in_names = {"cond_phys", "x0"};
    if (needs_u_) in_names.push_back("u");
    std::cout << "[ONNX-E2E] expected inputs: " << n_in
              << (needs_u_ ? " (stochastic rounding: u required)" : "") << "\n";

    const auto ci = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (ci.size() == 2 && ci[1] > 0) cond_dim_ = static_cast<int>(ci[1]);
    const auto xi = session.GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
    if (xi.size() == 2 && xi[1] > 0) x_dim_ = static_cast<int>(xi[1]);
  }

  std::vector<float> run(const std::vector<float>& cond_phys, const std::vector<float>& x0, int B,
                         const std::vector<float>& u = {}) {
    const int cd = cond_dim_ > 0 ? cond_dim_ : static_cast<int>(cond_phys.size() / B);
    const int xd = x_dim_ > 0 ? x_dim_ : static_cast<int>(x0.size() / B);
    if (needs_u_ && u.size() != static_cast<size_t>(B) * xd)
      fail("The model requires the input u with shape [" + std::to_string(B) + "," + std::to_string(xd) + "]");

    std::array<int64_t, 2> sc{B, cd}, sx{B, xd};
    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(cond_phys.data()), cond_phys.size(), sc.data(), 2));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(x0.data()), x0.size(), sx.data(), 2));
    if (needs_u_)
      inputs.emplace_back(Ort::Value::CreateTensor<float>(
          mem, const_cast<float*>(u.data()), u.size(), sx.data(), 2));

    auto out = session.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(),
                           out_names.data(), out_names.size());

    // The graph emits float32, or int32 when exported with --cast_int32.
    const auto info = out[0].GetTensorTypeAndShapeInfo();
    const size_t n = info.GetElementCount();
    std::vector<float> res(n);
    switch (info.GetElementType()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
        const float* p = out[0].GetTensorData<float>();
        std::copy(p, p + n, res.begin());
        break;
      }
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
        const int32_t* p = out[0].GetTensorData<int32_t>();
        std::copy(p, p + n, res.begin());
        break;
      }
      default:
        fail("Unhandled output type: " + std::to_string(static_cast<int>(info.GetElementType())));
    }
    return res;
  }

  int xdim() const { return x_dim_; }
  bool needs_u() const { return needs_u_; }
};
#endif

static int run_golden(E2E& net, const std::string& path) {
  std::ifstream g(path);
  if (!g) fail("Cannot open " + path);

  std::string key;
  int dummy = 0, stochastic = 0;
  g >> key >> dummy;       // n_steps
  g >> key >> dummy;       // internal_noise
  g >> key >> stochastic;  // stochastic

  const Mat cond = read_matrix(g, "cond_phys");
  const Mat x0 = read_matrix(g, "x0");
  const Mat u = stochastic ? read_matrix(g, "u") : Mat{};
  const Mat expected = read_matrix(g, "x_phys");
  const int B = cond.rows;

#ifdef SELFTEST
  net.stochastic = (stochastic != 0);
#else
  if (net.needs_u() != (stochastic != 0)) {
    std::cerr << "INCONSISTENT: the model " << (net.needs_u() ? "requires" : "does not require")
              << " u, the golden " << (stochastic ? "contains it" : "does not")
              << ". Regenerate the golden.\n";
    return 1;
  }
#endif

  const std::vector<float> got = net.run(cond.data, x0.data, B, u.data);

  // With an integer output (clamp/round inside the graph) the comparison is
  // discrete: count differing cells rather than only the maximum deviation.
  double max_abs = 0.0, max_rel = 0.0;
  long n_bad = 0;
  bool discrete = true;
  for (int n = 0; n < B; ++n)
    for (int j = 0; j < expected.cols; ++j) {
      const double v = got[static_cast<size_t>(n) * expected.cols + j];
      if (std::fabs(v - std::floor(v)) > 1e-6) discrete = false;
      const double a = std::fabs(v - expected.at(n, j));
      if (a > 0) ++n_bad;
      max_abs = std::max(max_abs, a);
      max_rel = std::max(max_rel, a / (std::fabs(expected.at(n, j)) + 1e-9));
    }

  const long n_tot = static_cast<long>(B) * expected.cols;
  if (discrete) {
    const bool ok = (n_bad == 0) || (max_abs <= 1 && static_cast<double>(n_bad) / n_tot < 1e-3);
    std::printf("Golden check  B=%d (discrete output)\n"
                "  differing cells = %ld/%ld\n  max|diff|       = %g\n  -> %s\n",
                B, n_bad, n_tot, max_abs, ok ? "OK" : "MISMATCH");
    return ok ? 0 : 2;
  }
  const bool ok = max_abs < 1e-3;
  std::printf("Golden check  B=%d\n  max|diff| = %.3e\n  max rel   = %.3e\n  -> %s\n",
              B, max_abs, max_rel, ok ? "OK" : "MISMATCH");
  return ok ? 0 : 2;
}

static int run_generation(E2E& net, const std::string& cond_path, unsigned seed) {
  std::ifstream cf(cond_path);
  if (!cf) fail("Cannot open " + cond_path);
  const Mat cond = read_matrix(cf, "cond_phys");
  const int B = cond.rows;

  int xd = net.xdim();
  if (xd <= 0) xd = 10;

  std::mt19937 rng(seed);
  std::normal_distribution<float> gauss(0.f, 1.f);
  std::vector<float> x0(static_cast<size_t>(B) * xd);
  for (float& z : x0) z = gauss(rng);

  // Stochastic rounding also needs u ~ U(0,1) with the same shape as x0.
  std::vector<float> u;
  if (net.needs_u()) {
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    u.resize(static_cast<size_t>(B) * xd);
    for (float& z : u) z = uni(rng);
  }

  const std::vector<float> x_phys = net.run(cond.data, x0, B, u);
  std::printf("Generated %d events. First rows:\n", B);
  for (int n = 0; n < std::min(B, 5); ++n) {
    std::printf("  ev%d:", n);
    for (int j = 0; j < xd; ++j) std::printf(" %+.4f", x_phys[static_cast<size_t>(n) * xd + j]);
    std::printf("\n");
  }
  return 0;
}

int main(int argc, char** argv) {
#ifdef SELFTEST
  E2E net(get_arg(argc, argv, "--scalers", "scalers.txt"), get_arg(argc, argv, "--net", "net.txt"),
          std::stoi(get_arg(argc, argv, "--n-steps", "100")));
  net.stochastic = (get_arg(argc, argv, "--stochastic", "0") == "1");
  std::cout << "[SELFTEST] emulated graph (linear network)\n";
#else
  E2E net(get_arg(argc, argv, "--model", "cfm_zdc_e2e.onnx"));
  std::cout << "[ONNX-E2E] model loaded\n";
#endif

  const std::string golden = get_arg(argc, argv, "--golden", "");
  if (!golden.empty()) return run_golden(net, golden);

  const std::string cond_path = get_arg(argc, argv, "--cond", "");
  if (cond_path.empty()) fail("Specify either --golden or --cond");

  return run_generation(net, cond_path, static_cast<unsigned>(std::stoul(get_arg(argc, argv, "--seed", "1234"))));
}