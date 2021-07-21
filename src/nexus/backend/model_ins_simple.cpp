#include "nexus/backend/model_ins_simple.h"

#include <stdexcept>

#include "nexus/backend/sleep_model.h"
#include "nexus/common/sleep_profile.h"
#include "nexus/common/util.h"

#ifdef USE_TENSORFLOW
#include "nexus/backend/tensorflow_model_simple.h"
#endif

#include <glog/logging.h>

namespace nexus {
namespace backend {

void CreateModelInstanceSimple(int gpu_id, const ModelInstanceConfig& config,
                               ModelIndex model_index,
                               std::unique_ptr<ModelInstanceSimple>* model) {
  const std::string kSleepPrefix = "sleep#";
  auto beg = Clock::now();
  std::string framework = config.model_session(0).framework();
#ifdef USE_TENSORFLOW
  if (framework == "tensorflow") {
    model->reset(new TensorflowModelSimple(gpu_id, config, model_index));
  }
#endif
  auto end = Clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
  LOG(INFO) << "Loading model time: " << duration.count() << "ms";
}

ModelInstanceSimple::ModelInstanceSimple(int gpu_id,
                                         const ModelInstanceConfig& config,
                                         ModelIndex model_index)
    : gpu_id_(gpu_id),
      model_session_(config.model_session(0)),
      model_index_(model_index),
      batch_(config.batch()),
      max_batch_(config.max_batch()) {
  CHECK_GT(batch_, 0) << "batch must be greater than 0";
  CHECK_GE(max_batch_, batch_) << "max_batch must be greater than batch";
  std::string model_id = ModelSessionToModelID(model_session_);
  auto info = ModelDatabase::Singleton().GetModelInfo(model_id);
  CHECK(info != nullptr) << "Model not found in the database";
  model_info_ = *info;
  model_session_id_ = ModelSessionToString(model_session_);
  cpu_device_ = DeviceManager::Singleton().GetCPUDevice();
#ifdef USE_GPU
  gpu_device_ = DeviceManager::Singleton().GetGPUDevice(gpu_id);
#endif
  LOG(INFO) << "Construct model " << model_session_id_ << ", batch " << batch_
            << ", max batch " << max_batch_;
}

ModelInstanceSimple::~ModelInstanceSimple() {
  LOG(INFO) << "Deconstruct model " << model_session_id_;
}

void ModelInstanceSimple::set_batch(size_t batch) {
  CHECK_LE(batch, max_batch_) << "Batch size must be less than max_batch";
  batch_.store(batch);
}

ArrayPtr ModelInstanceSimple::CreateInputGpuArrayWithRawPointer(
    float* ptr, size_t nfloats) {
  LOG(ERROR) << "Don't support create input gpu array with raw pointer";
  return nullptr;
}

void ModelInstanceSimple::RemoveInputGpuArray(ArrayPtr arr) {
  LOG(WARNING) << "Don't support remove input gpu array";
}

void ModelInstanceSimple::ForwardAsync(std::shared_ptr<BatchTask> batch_task) {
  LOG(WARNING) << "Don't support async forward";
  Forward(batch_task);
}

void ModelInstanceSimple::WaitOutput(std::shared_ptr<BatchTask> batch_task) {
  LOG(WARNING) << "Don't support async forward";
}

uint64_t ModelInstanceSimple::GetPeakBytesInUse() {
  LOG(FATAL) << "GetPeakBytesInUse not implemented";
}

uint64_t ModelInstanceSimple::GetBytesInUse() {
  LOG(FATAL) << "GetBytesInUse not implemented";
}
}  // namespace backend
}  // namespace nexus
