// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_DEVICE_MANAGER_H_
#define OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_DEVICE_MANAGER_H_

#include <vector>

#include "open_spiel/abseil-cpp/absl/synchronization/mutex.h"
#include "open_spiel/algorithms/alpha_zero/vpnet.h"

namespace open_spiel::algorithms {

// Keeps track of a bunch of VPNet models, intended to be one per device, and
// gives them out based on usage. When you request a device you specify how much
// work you're going to give it, which is assumed done once the loan is
// returned.
class DeviceManager {
 public:
  DeviceManager() {}

  void AddDevice(VPNetModel model) {  // Not thread safe.
    devices.emplace_back(Device{std::move(model)});
  }

  // Acts as a pointer to the model, but lets the manager know when you're done.
  class DeviceLoan {
   public:
    // DeviceLoan is not public constructible and is move only.
    DeviceLoan(DeviceLoan&& other) = default;
    DeviceLoan& operator=(DeviceLoan&& other) = default;
    DeviceLoan(const DeviceLoan&) = delete;
    DeviceLoan& operator=(const DeviceLoan&) = delete;

    ~DeviceLoan() { manager_->Return(device_id_, requests_); }
    VPNetModel* operator->() { return model_; }

   private:
    DeviceLoan(DeviceManager* manager, VPNetModel* model, int device_id,
               int requests)
        : manager_(manager), model_(model), device_id_(device_id),
          requests_(requests) {}
    DeviceManager* manager_;
    VPNetModel* model_;
    int device_id_;
    int requests_;
    friend DeviceManager;
  };

  // Gives the device with the fewest outstanding requests.
  DeviceLoan Get(int requests, int device_id = -1) {
    absl::MutexLock lock(&m_);
    if (device_id < 0) {
      device_id = 0;
      for (int i = 1; i < devices.size(); ++i) {
        if (devices[i].requests < devices[device_id].requests) {
          device_id = i;
        }
      }
    }
    devices[device_id].requests += requests;
    return DeviceLoan(this, &devices[device_id].model, device_id, requests);
  }

  int Count() const { return devices.size(); }

 private:
  void Return(int device_id, int requests) {
    absl::MutexLock lock(&m_);
    devices[device_id].requests -= requests;
  }

  struct Device {
    VPNetModel model;
    int requests = 0;
  };

  std::vector<Device> devices;
  absl::Mutex m_;
};

}  // namespace open_spiel::algorithms

#endif  // OPEN_SPIEL_ALGORITHMS_ALPHA_ZERO_DEVICE_MANAGER_H_
