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

#include "open_spiel/algorithms/alpha_zero_torch/model.h"

#include <torch/torch.h>

#include <iostream>
#include <string>
#include <vector>

namespace open_spiel {
namespace algorithms {
namespace torch_az {

std::istream& operator>>(std::istream& stream, ModelConfig& config) {
  int channels;
  int height;
  int width;

  stream >> channels >> height >> width >> config.number_of_actions >>
      config.nn_depth >> config.nn_width >> config.learning_rate >>
      config.weight_decay >> config.nn_model;

  config.observation_tensor_shape = {channels, height, width};

  return stream;
}

std::ostream& operator<<(std::ostream& stream, const ModelConfig& config) {
  stream << config.observation_tensor_shape[0] << " "
         << config.observation_tensor_shape[1] << " "
         << config.observation_tensor_shape[2] << " "
         << config.number_of_actions << " " << config.nn_depth << " "
         << config.nn_width << " " << config.learning_rate << " "
         << config.weight_decay << " " << config.nn_model;
  return stream;
}

ResInputBlockImpl::ResInputBlockImpl(const ResInputBlockConfig& config)
    : conv_(torch::nn::Conv2dOptions(
                /*input_channels=*/config.input_channels,
                /*output_channels=*/config.filters,
                /*kernel_size=*/config.kernel_size)
                .stride(1)
                .padding(config.padding)
                .dilation(1)
                .groups(1)
                .bias(true)
                .padding_mode(torch::kZeros)),
      batch_norm_(torch::nn::BatchNorm2dOptions(
                      /*num_features=*/config.filters)
                      .eps(0.001)      // Make it the same as TF.
                      .momentum(0.01)  // Torch momentum = 1 - TF momentum.
                      .affine(true)
                      .track_running_stats(true)) {
  channels_ = config.input_channels;
  height_ = config.input_height;
  width_ = config.input_width;

  register_module("input_conv", conv_);
  register_module("input_batch_norm", batch_norm_);
}

torch::Tensor ResInputBlockImpl::forward(torch::Tensor x) {
  torch::Tensor output = x.view({-1, channels_, height_, width_});
  output = torch::relu(batch_norm_(conv_(output)));

  return output;
}

ResTorsoBlockImpl::ResTorsoBlockImpl(const ResTorsoBlockConfig& config,
                                     int layer)
    : conv1_(torch::nn::Conv2dOptions(
                 /*input_channels=*/config.input_channels,
                 /*output_channels=*/config.filters,
                 /*kernel_size=*/config.kernel_size)
                 .stride(1)
                 .padding(config.padding)
                 .dilation(1)
                 .groups(1)
                 .bias(true)
                 .padding_mode(torch::kZeros)),
      conv2_(torch::nn::Conv2dOptions(
                 /*input_channels=*/config.filters,
                 /*output_channels=*/config.filters,
                 /*kernel_size=*/config.kernel_size)
                 .stride(1)
                 .padding(config.padding)
                 .dilation(1)
                 .groups(1)
                 .bias(true)
                 .padding_mode(torch::kZeros)),
      batch_norm1_(torch::nn::BatchNorm2dOptions(
                       /*num_features=*/config.filters)
                       .eps(0.001)      // Make it the same as TF.
                       .momentum(0.01)  // Torch momentum = 1 - TF momentum.
                       .affine(true)
                       .track_running_stats(true)),
      batch_norm2_(torch::nn::BatchNorm2dOptions(
                       /*num_features=*/config.filters)
                       .eps(0.001)      // Make it the same as TF.
                       .momentum(0.01)  // Torch momentum = 1 - TF momentum.
                       .affine(true)
                       .track_running_stats(true)) {
  register_module("res_" + std::to_string(layer) + "_conv_1", conv1_);
  register_module("res_" + std::to_string(layer) + "_conv_2", conv2_);
  register_module("res_" + std::to_string(layer) + "_batch_norm_1",
                  batch_norm1_);
  register_module("res_" + std::to_string(layer) + "_batch_norm_2",
                  batch_norm2_);
}

torch::Tensor ResTorsoBlockImpl::forward(torch::Tensor x) {
  torch::Tensor residual = x;

  torch::Tensor output = torch::relu(batch_norm1_(conv1_(x)));
  output = batch_norm2_(conv2_(output));
  output += residual;
  output = torch::relu(output);

  return output;
}

ResOutputBlockImpl::ResOutputBlockImpl(const ResOutputBlockConfig& config)
    : value_conv_(torch::nn::Conv2dOptions(
                      /*input_channels=*/config.input_channels,
                      /*output_channels=*/config.value_filters,
                      /*kernel_size=*/config.kernel_size)
                      .stride(1)
                      .padding(config.padding)
                      .dilation(1)
                      .groups(1)
                      .bias(true)
                      .padding_mode(torch::kZeros)),
      value_batch_norm_(
          torch::nn::BatchNorm2dOptions(
              /*num_features=*/config.value_filters)
              .eps(0.001)      // Make it the same as TF.
              .momentum(0.01)  // Torch momentum = 1 - TF momentum.
              .affine(true)
              .track_running_stats(true)),
      value_linear1_(torch::nn::LinearOptions(
                         /*in_features=*/config.value_linear_in_features,
                         /*out_features=*/config.value_linear_out_features)
                         .bias(true)),
      value_linear2_(torch::nn::LinearOptions(
                         /*in_features=*/config.value_linear_out_features,
                         /*out_features=*/1)
                         .bias(true)),
      value_observation_size_(config.value_observation_size),
      policy_conv_(torch::nn::Conv2dOptions(
                       /*input_channels=*/config.input_channels,
                       /*output_channels=*/config.policy_filters,
                       /*kernel_size=*/config.kernel_size)
                       .stride(1)
                       .padding(config.padding)
                       .dilation(1)
                       .groups(1)
                       .bias(true)
                       .padding_mode(torch::kZeros)),
      policy_batch_norm_(
          torch::nn::BatchNorm2dOptions(
              /*num_features=*/config.policy_filters)
              .eps(0.001)      // Make it the same as TF.
              .momentum(0.01)  // Torch momentum = 1 - TF momentum.
              .affine(true)
              .track_running_stats(true)),
      policy_linear_(torch::nn::LinearOptions(
                         /*in_features=*/config.policy_linear_in_features,
                         /*out_features=*/config.policy_linear_out_features)
                         .bias(true)),
      policy_observation_size_(config.policy_observation_size) {
  register_module("value_conv", value_conv_);
  register_module("value_batch_norm", value_batch_norm_);
  register_module("value_linear_1", value_linear1_);
  register_module("value_linear_2", value_linear2_);
  register_module("policy_conv", policy_conv_);
  register_module("policy_batch_norm", policy_batch_norm_);
  register_module("policy_linear", policy_linear_);
}

std::vector<torch::Tensor> ResOutputBlockImpl::forward(torch::Tensor x,
                                                       torch::Tensor mask) {
  torch::Tensor value_output = torch::relu(value_batch_norm_(value_conv_(x)));
  value_output = value_output.view({-1, value_observation_size_});
  value_output = torch::relu(value_linear1_(value_output));
  value_output = torch::tanh(value_linear2_(value_output));

  torch::Tensor policy_logits =
      torch::relu(policy_batch_norm_(policy_conv_(x)));
  policy_logits = policy_logits.view({-1, policy_observation_size_});
  policy_logits = policy_linear_(policy_logits);
  policy_logits = torch::where(mask, policy_logits,
                               -(1 << 16) * torch::ones_like(policy_logits));

  return {value_output, policy_logits};
}

MLPOutputBlockImpl::MLPOutputBlockImpl(const int nn_width,
                                       const int policy_linear_out_features)
    : value_linear1_(torch::nn::LinearOptions(
                         /*in_features=*/nn_width,
                         /*out_features=*/nn_width)
                         .bias(true)),
      value_linear2_(torch::nn::LinearOptions(
                         /*in_features=*/nn_width,
                         /*out_features=*/1)
                         .bias(true)),
      policy_linear1_(torch::nn::LinearOptions(
                          /*input_channels=*/nn_width,
                          /*output_channels=*/nn_width)
                          .bias(true)),
      policy_linear2_(torch::nn::LinearOptions(
                          /*in_features=*/nn_width,
                          /*out_features=*/policy_linear_out_features)
                          .bias(true)) {
  register_module("value_linear_1", value_linear1_);
  register_module("value_linear_2", value_linear2_);
  register_module("policy_linear_1", policy_linear1_);
  register_module("policy_linear_2", policy_linear2_);
}

std::vector<torch::Tensor> MLPOutputBlockImpl::forward(torch::Tensor x,
                                                       torch::Tensor mask) {
  torch::Tensor value_output = torch::relu(value_linear1_(x));
  value_output = torch::tanh(value_linear2_(value_output));

  torch::Tensor policy_logits = torch::relu(policy_linear1_(x));
  policy_logits = policy_linear2_(policy_logits);
  policy_logits = torch::where(mask, policy_logits,
                               -(1 << 16) * torch::ones_like(policy_logits));

  return {value_output, policy_logits};
}

ModelImpl::ModelImpl(const ModelConfig& config, const std::string& device)
    : device_(device),
      num_torso_blocks_(config.nn_depth),
      weight_decay_(config.weight_decay) {
  // Save config.nn_model to class
  nn_model_ = config.nn_model;

  int input_size = 1;
  for (const auto& num : config.observation_tensor_shape) {
    if (num > 0) {
      input_size *= num;
    }
  }
  int channels = config.observation_tensor_shape[0];
  // Decide if resnet or MLP
  if (config.nn_model == "resnet") {
    int height = config.observation_tensor_shape[1];
    int width = config.observation_tensor_shape[2];

    ResInputBlockConfig input_config = {/*input_channels=*/channels,
                                        /*input_height=*/height,
                                        /*input_width=*/width,
                                        /*filters=*/config.nn_width,
                                        /*kernel_size=*/3,
                                        /*padding=*/1};

    ResTorsoBlockConfig residual_config = {/*input_channels=*/config.nn_width,
                                           /*filters=*/config.nn_width,
                                           /*kernel_size=*/3,
                                           /*padding=*/1};

    ResOutputBlockConfig output_config = {
        /*input_channels=*/config.nn_width,
        /*value_filters=*/1,
        /*policy_filters=*/2,
        /*kernel_size=*/1,
        /*padding=*/0,
        /*value_linear_in_features=*/1 * width * height,
        /*value_linear_out_features=*/config.nn_width,
        /*policy_linear_in_features=*/2 * width * height,
        /*policy_linear_out_features=*/config.number_of_actions,
        /*value_observation_size=*/1 * width * height,
        /*policy_observation_size=*/2 * width * height};

    layers_->push_back(ResInputBlock(input_config));
    for (int i = 0; i < num_torso_blocks_; i++) {
      layers_->push_back(ResTorsoBlock(residual_config, i));
    }
    layers_->push_back(ResOutputBlock(output_config));

    register_module("layers", layers_);

  } else if (config.nn_model == "mlp") {
    layers_->push_back(torch::nn::Linear(input_size, config.nn_width));
    for (int i = 0; i < num_torso_blocks_; i++) {
      layers_->push_back(torch::nn::Linear(config.nn_width, config.nn_width));
    }
    layers_->push_back(
        MLPOutputBlock(config.nn_width, config.number_of_actions));

    register_module("layers", layers_);
  } else {
    throw std::runtime_error("Unknown nn_model: " + config.nn_model);
  }
}

std::vector<torch::Tensor> ModelImpl::forward(torch::Tensor x,
                                              torch::Tensor mask) {
  std::vector<torch::Tensor> output = this->forward_(x, mask);
  return {output[0], torch::softmax(output[1], 1)};
}

std::vector<torch::Tensor> ModelImpl::losses(torch::Tensor inputs,
                                             torch::Tensor masks,
                                             torch::Tensor policy_targets,
                                             torch::Tensor value_targets) {
  std::vector<torch::Tensor> output = this->forward_(inputs, masks);

  torch::Tensor value_predictions = output[0];
  torch::Tensor policy_predictions = output[1];

  // Policy loss (cross-entropy).
  torch::Tensor policy_loss = torch::sum(
      -policy_targets * torch::log_softmax(policy_predictions, 1), -1);
  policy_loss = torch::mean(policy_loss);

  // Value loss (mean-squared error).
  torch::nn::MSELoss mse_loss;
  torch::Tensor value_loss = mse_loss(value_predictions, value_targets);

  // L2 regularization loss (weights only).
  torch::Tensor l2_regularization_loss = torch::full(
      {1, 1}, 0, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  for (auto& named_parameter : this->named_parameters()) {
    // named_parameter is essentially a key-value pair:
    //   {key, value} == {std::string name, torch::Tensor parameter}
    std::string parameter_name = named_parameter.key();

    // Do not include bias' in the loss.
    if (parameter_name.find("bias") != std::string::npos) {
      continue;
    }

    // Copy TensorFlow's l2_loss function.
    // https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    l2_regularization_loss +=
        weight_decay_ * torch::sum(torch::square(named_parameter.value())) / 2;
  }

  return {policy_loss, value_loss, l2_regularization_loss};
}

std::vector<torch::Tensor> ModelImpl::forward_(torch::Tensor x,
                                               torch::Tensor mask) {
  std::vector<torch::Tensor> output;
  if (this->nn_model_ == "resnet") {
    for (int i = 0; i < num_torso_blocks_ + 2; i++) {
      if (i == 0) {
        x = layers_[i]->as<ResInputBlock>()->forward(x);
      } else if (i >= num_torso_blocks_ + 1) {
        output = layers_[i]->as<ResOutputBlock>()->forward(x, mask);
      } else {
        x = layers_[i]->as<ResTorsoBlock>()->forward(x);
      }
    }
  } else if (this->nn_model_ == "mlp") {
    for (int i = 0; i < num_torso_blocks_ + 2; i++) {
      if (i == 0) {
        x = layers_[i]->as<torch::nn::Linear>()->forward(x);
      } else if (i >= num_torso_blocks_ + 1) {
        output = layers_[i]->as<MLPOutputBlockImpl>()->forward(x, mask);
      } else {
        x = layers_[i]->as<torch::nn::Linear>()->forward(x);
      }
    }
  } else {
    throw std::runtime_error("Unknown nn_model: " + this->nn_model_);
  }
  return output;
}

}  // namespace torch_az
}  // namespace algorithms
}  // namespace open_spiel
