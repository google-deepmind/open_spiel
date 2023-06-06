//
// Created by ramizouari on 06/06/23.
//
/*
==============================================================================
MIT License
Copyright 2022 Institute for Automotive Engineering of RWTH Aachen University.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================
*/

#include <iostream>
#include <vector>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/version.h>
#include <tensorflow/cc/saved_model/loader.h>
//include placeholder
#include <tensorflow/cc/ops/standard_ops.h>
using namespace std;

constexpr int kWidth=28, kHeight=28, kChannels=1;

int main(int argc, char **argv) {

    // create new session
    std::string model_dir = "model.pb";
    std::vector<string> input_names;
    std::vector<string> output_names;
    tensorflow::SavedModelBundle bundle; // Same with SavedModelBundle

// Create default options.
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;

// Load model.
    auto status = tensorflow::LoadSavedModel(
            session_options,
            run_options,
            model_dir,
            {"serve"},
            &bundle
    );
// An alias for SavedModel
    auto &model=bundle;
// Check if model has been loaded correctly.
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    }

// Get model signature.
    auto signatures = bundle.GetSignatures();
    if (!signatures.contains("serving_default")) {
        std::cerr << "Could not find serving_default in model signatures." << std::endl;
        return 1;
    }

// Get the inputs names.
    for (auto const & input : signatures.at("serving_default").inputs()) {
        input_names.push_back(input.second.name());
    }

// Get the outputs names.
    for (auto const & output : signatures.at("serving_default").outputs()) {
        output_names.push_back(output.second.name());
    }
    std::vector<tensorflow::Tensor>  inputs;  // Some compatible input vector.
    inputs.push_back(tensorflow::Tensor(tensorflow::DT_FLOAT, {1, kWidth, kHeight, kChannels})); // 1 image of 28x28x1
    for (int i=0;i< kWidth; i++){
        for (int j=0;j< kHeight; j++){
            for (int k=0;k< kChannels; k++){
                inputs[0].tensor<float, 4>()(0, i, j, k) = 1;
            }
        }
    }
    std::vector<tensorflow::Tensor>  outputs;

// Create a vector of pairs for associating inputs to their names.
    std::vector<std::pair<std::string, tensorflow::Tensor>> input_pairs;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        input_pairs.emplace_back(input_names.at(i), inputs.at(i));
    }

// Run the network.
    bundle.GetSession()->Run(
            input_pairs,
            output_names,
            {},
            &outputs
    );
    for(auto &output : outputs){
        for(int i=0;i<10;i++)
            std::cout<<output.matrix<float>()(0,i) << ' ';
        std::cout<<std::endl;
    }
    return 0;
}