// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FusedMultiTransformerKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const std::vector<const DenseTensor *> &ln_scales,
    const std::vector<const DenseTensor *> &ln_biases,
    const std::vector<const DenseTensor *> &qkv_weights,
    const std::vector<const DenseTensor *> &qkv_biases,
    const std::vector<const DenseTensor *> &linear_weights,
    const std::vector<const DenseTensor *> &linear_biases,
    const std::vector<const DenseTensor *> &ffn_ln_scales,
    const std::vector<const DenseTensor *> &ffn_ln_biases,
    const std::vector<const DenseTensor *> &ffn1_weights,
    const std::vector<const DenseTensor *> &ffn1_biases,
    const std::vector<const DenseTensor *> &ffn2_weights,
    const std::vector<const DenseTensor *> &ffn2_biases,
    const paddle::optional<std::vector<const DenseTensor *>> &cache_kvs,
    const paddle::optional<std::vector<const DenseTensor *>> &pre_caches,
    const paddle::optional<DenseTensor> &rotary_pos_emb,
    const paddle::optional<DenseTensor> &time_step,
    const paddle::optional<DenseTensor> &seq_lengths,
    const paddle::optional<DenseTensor> &attn_mask,
    bool pre_layer_norm,
    float epsilon,
    float dropout_rate,
    int rotary_emb_dims,
    bool is_test,
    const std::string &act_method,
    const std::string &dropout_implementation,
    bool trans_qkvw,
    int ring_id,
    std::vector<const DenseTensor *> cache_kv_outs,
    DenseTensor *out);

}  // namespace phi
