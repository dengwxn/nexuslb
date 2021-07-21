#include <gflags/gflags.h>
#include <glog/logging.h>
#include <time.h>
#include <yaml-cpp/yaml.h>

#include <boost/filesystem.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "nexus/backend/model_ins_simple.h"
#include "nexus/common/block_queue.h"
#include "nexus/common/device.h"
#include "nexus/common/model_db.h"
#include "nexus/common/typedef.h"
#include "nexus/proto/nnquery.pb.h"

DEFINE_int32(gpu, 0, "GPU device id");
DEFINE_string(framework, "tensorflow", "Framework");
DEFINE_string(model, "", "Model name");
DEFINE_int32(model_version, 1, "Version");
DEFINE_int32(min_batch, 1, "Minimum batch size");
DEFINE_int32(max_batch, 64, "Maximum batch size");
DEFINE_string(output, "", "Output file");
DEFINE_int32(height, 0, "Image height");
DEFINE_int32(width, 0, "Image width");

namespace nexus {
namespace backend {

using duration = std::chrono::microseconds;

class LoadTest {
 public:
  LoadTest(int gpu, const std::string& framework, const std::string& model_name,
           int model_version, int height = 0, int width = 0)
      : gpu_(gpu) {
    model_info_ = ModelDatabase::Singleton().GetModelInfo(framework, model_name,
                                                          model_version);
    CHECK(model_info_ != nullptr) << "Cannot find model info for " << framework
                                  << ":" << model_name << ":" << model_version;
    // Init model session
    model_sess_.set_framework(framework);
    model_sess_.set_model_name(model_name);
    model_sess_.set_version(model_version);
    model_sess_.set_latency_sla(50000);
    if (height > 0) {
      CHECK_GT(width, 0) << "Height and width must be set together";
      model_sess_.set_image_height(height);
      model_sess_.set_image_width(width);
    } else {
      if ((*model_info_)["resizable"] &&
          (*model_info_)["resizable"].as<bool>()) {
        // Set default image size for resizable CNN
        model_sess_.set_image_height(
            (*model_info_)["image_height"].as<uint32_t>());
        model_sess_.set_image_width(
            (*model_info_)["image_width"].as<uint32_t>());
      }
    }
    LOG(INFO) << model_sess_.DebugString();
    model_sessions_.push_back(ModelSessionToString(model_sess_));
    LOG(INFO) << "Profile model " << ModelSessionToProfileID(model_sess_);
    // Get test dataset
    cpu_device_ = DeviceManager::Singleton().GetCPUDevice();
#ifdef USE_GPU
    // Init GPU device
    NEXUS_CUDA_CHECK(cudaSetDevice(gpu_));
    gpu_device_ = DeviceManager::Singleton().GetGPUDevice(gpu_);
#else
    if (gpu_ != -1) {
      LOG(FATAL) << "The code is compiled without USE_GPU. Please set "
                    "`-gpu=-1` to profile on CPU.";
    }
#endif
  }

  void TestTime(int min_batch, int max_batch, const std::string output = "",
                int repeat = 10) {
    std::ostream* fout;
    if (output.length() == 0) {
      fout = &std::cout;
    } else {
      fout = new std::ofstream(output, std::ofstream::out);
    }
#ifdef USE_GPU
    *fout << gpu_device_->device_name() << "\n";
    *fout << gpu_device_->uuid() << "\n";
#endif

    ModelInstanceConfig config;
    config.add_model_session()->CopyFrom(model_sess_);
    size_t batch_size = 16;
    config.set_batch(batch_size);
    config.set_max_batch(batch_size);

    std::unique_ptr<ModelInstanceSimple> models[300];
    for (int i = 0; i < 10; i++) {
      auto beg = std::chrono::high_resolution_clock::now();
      CreateModelInstanceSimple(gpu_, config, ModelIndex(0), &models[i]);
      auto end = std::chrono::high_resolution_clock::now();
      auto forward = std::chrono::duration_cast<duration>(end - beg).count();
      std::cout << i + 1 << "," << forward << "\n";
    }
  }

  void TestLimit(int min_batch, int max_batch, const std::string output = "",
                 int repeat = 10) {
    std::ostream* fout;
    if (output.length() == 0) {
      fout = &std::cout;
    } else {
      fout = new std::ofstream(output, std::ofstream::out);
    }
#ifdef USE_GPU
    *fout << gpu_device_->device_name() << "\n";
    *fout << gpu_device_->uuid() << "\n";
#endif

    ModelInstanceConfig config;
    config.add_model_session()->CopyFrom(model_sess_);
    size_t batch_size = 16;
    config.set_batch(batch_size);
    config.set_max_batch(batch_size);

    std::unique_ptr<ModelInstanceSimple> models[300];
    for (int i = 0; i < 210; i++) {
      CreateModelInstanceSimple(gpu_, config, ModelIndex(0), &models[i]);
      models[i]->ForwardSimple(batch_size);
      auto bytes = models[i]->GetBytesInUse();
      auto peak_bytes = models[i]->GetPeakBytesInUse();
      std::cout << i + 1 << "," << bytes << "," << peak_bytes << "\n";
      std::cout << std::flush;
    }
  }

 private:
  template <class T>
  std::pair<float, float> GetStats(const std::vector<T>& lats) {
    float mean = 0.;
    float std = 0.;
    for (uint i = 0; i < lats.size(); ++i) {
      mean += lats[i];
    }
    mean /= lats.size();
    for (uint i = 0; i < lats.size(); ++i) {
      std += (lats[i] - mean) * (lats[i] - mean);
    }
    std = sqrt(std / (lats.size() - 1));
    return {mean, std};
  }

 private:
  int gpu_;
  ModelSession model_sess_;
  const YAML::Node* model_info_;
  std::string framework_;
  std::string model_name_;
  int version_;
  int height_;
  int width_;
  std::vector<std::string> model_sessions_;
  CPUDevice* cpu_device_;
#ifdef USE_GPU
  GPUDevice* gpu_device_;
#endif
};

}  // namespace backend
}  // namespace nexus

int main(int argc, char** argv) {
  using namespace nexus;
  using namespace nexus::backend;

  // log to stderr
  FLAGS_logtostderr = 1;
  // Init glog
  google::InitGoogleLogging(argv[0]);
  // Parse command line flags
  google::ParseCommandLineFlags(&argc, &argv, true);
  // Setup backtrace on segfault
  google::InstallFailureSignalHandler();
  // Check flags
  CHECK_GT(FLAGS_framework.length(), 0) << "Missing framework";
  CHECK_GT(FLAGS_model.length(), 0) << "Missing model";
  srand(time(NULL));

  LoadTest load(FLAGS_gpu, FLAGS_framework, FLAGS_model, FLAGS_model_version,
                FLAGS_height, FLAGS_width);
  load.TestTime(FLAGS_min_batch, FLAGS_max_batch, FLAGS_output);
  // load.TestLimit(FLAGS_min_batch, FLAGS_max_batch, FLAGS_output);
}
