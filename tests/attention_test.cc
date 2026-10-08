#include <cstdlib>
#include <random>

#include "test_utils.h"
#include "ctranslate2/layers/attention.h"
#ifdef CT2_WITH_SYCL
#  include "xpu/utils.h"
#endif

class MockModel : public models::Model {
public:
  MockModel(dim_t num_heads, dim_t num_heads_kv) {
    const dim_t d_model = 64;
    const dim_t d_head = d_model / num_heads;
    
    std::vector<float> linear_0_data(num_heads * d_head * d_model, 0.01f);
    std::vector<float> linear_1_data(2 * num_heads_kv * d_head * d_model, 0.01f);
    
    register_variable("attn/linear_0/weight",
                      StorageView({num_heads * d_head, d_model}, linear_0_data));
    register_variable("attn/linear_1/weight",
                      StorageView({2 * num_heads_kv * d_head, d_model}, linear_1_data));
    register_variable("attn/linear_2/weight",
                      StorageView({d_model, num_heads * d_head}, DataType::FLOAT32));
    register_variable("attn/q_norm/gamma",
                      StorageView({d_model}, std::vector<float>(d_model, 1.0f)));
    register_variable("attn/k_norm/gamma",
                      StorageView({d_head}, std::vector<float>(d_head, 1.0f)));
    
    register_variable("attn/num_heads_kv",
                      StorageView(static_cast<int32_t>(num_heads_kv)));

    set_compute_type(ComputeType::FLOAT32, Device::CPU, 0, false);
  }
protected:
  std::unique_ptr<Model> clone() const override { return nullptr; }
};

class TestableAttention : public layers::MultiHeadAttention {
public:
  using MultiHeadAttention::MultiHeadAttention;
  using MultiHeadAttention::process_cross_attention;
};

class CrossAttentionTest : public ::testing::Test {
protected:
  static constexpr dim_t NUM_HEADS = 4;
  static constexpr dim_t D_MODEL = 64;
  static constexpr dim_t D_HEAD = D_MODEL / NUM_HEADS;
  static constexpr dim_t BATCH = 2;
  static constexpr dim_t Q_LEN = 6;
  static constexpr dim_t V_LEN = 8;

  float get_4d(const StorageView& view, dim_t b, dim_t h, dim_t t, dim_t d) {
    const auto& shape = view.shape();
    return view.data<float>()[b * shape[1] * shape[2] * shape[3] +
                              h * shape[2] * shape[3] + t * shape[3] + d];
  }

};

// MQA: All heads share same K/V
TEST_F(CrossAttentionTest, MultiQueryAttention) {
  MockModel model(NUM_HEADS, /*num_heads_kv=*/1);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);
  // Use non-uniform values to verify normalization is applied
  std::vector<float> value_data(BATCH * V_LEN * D_MODEL);
  for (size_t i = 0; i < value_data.size(); ++i)
    value_data[i] = static_cast<float>(i % 10 + 1);
  std::vector<float> fused_data(BATCH * Q_LEN * NUM_HEADS * D_HEAD);
  for (size_t i = 0; i < fused_data.size(); ++i)
    fused_data[i] = static_cast<float>(i % 10 + 1);
  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, value_data);
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, fused_data);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;
  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);
  // MQA: K/V are replicated to 4D format [batch, num_heads, time, d_head]
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  ASSERT_EQ(cached_values.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  // Verify K/V values are consistent across batch and time dimensions
  // (In MQA, there's only one set of K/V, so we just verify the tensor is valid)
  float k0 = get_4d(cached_keys, 0, 0, 0, 0);
  float v0 = get_4d(cached_values, 0, 0, 0, 0);
  EXPECT_NE(k0, 0.0f) << "K values should be non-zero after projection";
  EXPECT_NE(v0, 0.0f) << "V values should be non-zero after projection";
  // Verify q_norm and k_norm are applied (RMSNorm normalizes to ~1.0 magnitude)
  float q_val = q_proj.data<float>()[0];
  float k_val = cached_keys.data<float>()[0];
  EXPECT_GT(std::abs(q_val), 0.1f) << "q_norm should produce non-zero output";
  EXPECT_LT(std::abs(q_val), 2.0f) << "q_norm should normalize values";
  EXPECT_GT(std::abs(k_val), 0.1f) << "k_norm should produce non-zero output";
  EXPECT_LT(std::abs(k_val), 2.0f) << "k_norm should normalize values";
}

// GQA: Heads within same group share K/V
TEST_F(CrossAttentionTest, GroupedQueryAttention) {
  constexpr dim_t NUM_KV_HEADS = 2;
  constexpr dim_t HEADS_PER_GROUP = NUM_HEADS / NUM_KV_HEADS;

  MockModel model(NUM_HEADS, NUM_KV_HEADS);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);

  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, std::vector<float>(BATCH * V_LEN * D_MODEL, 1.0f));
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, DataType::FLOAT32);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;

  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);

  // GQA: After head replication, shape is [batch, num_heads, time, d_head]
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));

  // Heads in same group share K/V
  for (dim_t group = 0; group < NUM_KV_HEADS; ++group) {
    dim_t first = group * HEADS_PER_GROUP;
    float k_group = get_4d(cached_keys, 0, first, 0, 0);
    float v_group = get_4d(cached_values, 0, first, 0, 0);
    for (dim_t h = first + 1; h < first + HEADS_PER_GROUP; ++h) {
      EXPECT_EQ(get_4d(cached_keys, 0, h, 0, 0), k_group);
      EXPECT_EQ(get_4d(cached_values, 0, h, 0, 0), v_group);
    }
  }
}

// Merged self+cross attention (T5Gemma2 style): self-attention layer also
// projects encoder memory through a separate `memory_kv` linear and concatenates
// the result onto the self-attention K/V before softmax.
TEST(MergedAttentionTest, ForwardMergedProducesOutput) {
  constexpr dim_t NUM_HEADS = 4, NUM_KV = 1, D_HEAD = 16;
  constexpr dim_t D_MODEL = NUM_HEADS * D_HEAD;
  constexpr dim_t QKV_ROWS = (NUM_HEADS + 2 * NUM_KV) * D_HEAD;
  constexpr dim_t KV_ROWS = 2 * NUM_KV * D_HEAD;

  class MergedMockModel : public models::Model {
  public:
    MergedMockModel() {
      register_variable("attn/linear_0/weight",
                        StorageView({QKV_ROWS, D_MODEL}, std::vector<float>(QKV_ROWS * D_MODEL, 0.01f)));
      register_variable("attn/linear_1/weight",
                        StorageView({D_MODEL, NUM_HEADS * D_HEAD},
                                    std::vector<float>(D_MODEL * NUM_HEADS * D_HEAD, 0.01f)));
      register_variable("attn/memory_kv/weight",
                        StorageView({KV_ROWS, D_MODEL}, std::vector<float>(KV_ROWS * D_MODEL, 0.01f)));
      register_variable("attn/q_norm/gamma", StorageView({D_HEAD}, std::vector<float>(D_HEAD, 1.0f)));
      register_variable("attn/k_norm/gamma", StorageView({D_HEAD}, std::vector<float>(D_HEAD, 1.0f)));
      register_variable("attn/num_heads_kv", StorageView(static_cast<int32_t>(NUM_KV)));
      set_compute_type(ComputeType::FLOAT32, Device::CPU, 0, false);
    }
  protected:
    std::unique_ptr<Model> clone() const override { return nullptr; }
  };

  MergedMockModel model;
  layers::MultiHeadAttention attention(model, "attn", NUM_HEADS, /*self_attention=*/true);
  ASSERT_TRUE(attention.has_merged_encoder_attention());

  constexpr dim_t B = 1, Q_LEN = 1, MEM_LEN = 5;
  StorageView queries({B, Q_LEN, D_MODEL}, std::vector<float>(B * Q_LEN * D_MODEL, 1.0f));
  StorageView memory({B, MEM_LEN, D_MODEL}, std::vector<float>(B * MEM_LEN * D_MODEL, 1.0f));
  StorageView output(DataType::FLOAT32);
  StorageView self_k(DataType::FLOAT32), self_v(DataType::FLOAT32);
  StorageView mem_k(DataType::FLOAT32), mem_v(DataType::FLOAT32);

  attention.forward_merged(queries, &memory, nullptr, nullptr, output,
                           &self_k, &self_v, &mem_k, &mem_v, nullptr, nullptr, /*offset=*/0);

  EXPECT_EQ(output.shape(), (Shape{B, Q_LEN, D_MODEL}));
  EXPECT_EQ(mem_k.shape(), (Shape{B, NUM_HEADS, MEM_LEN, D_HEAD}));
  EXPECT_EQ(self_k.shape(), (Shape{B, NUM_HEADS, Q_LEN, D_HEAD}));
}

// MHA: Each head has independent K/V
TEST_F(CrossAttentionTest, StandardMultiHeadAttention) {
  MockModel model(NUM_HEADS, NUM_HEADS);
  TestableAttention attention(model, "attn", NUM_HEADS, false, false, true);

  StorageView queries({BATCH, Q_LEN, D_MODEL}, DataType::FLOAT32);
  StorageView values({BATCH, V_LEN, D_MODEL}, std::vector<float>(BATCH * V_LEN * D_MODEL, 1.0f));
  StorageView fused_proj({BATCH, Q_LEN, NUM_HEADS * D_HEAD}, DataType::FLOAT32);
  StorageView q_proj(DataType::FLOAT32), k_proj(DataType::FLOAT32), v_proj(DataType::FLOAT32);
  StorageView cached_keys(DataType::FLOAT32), cached_values(DataType::FLOAT32);
  dim_t beam = 1;

  attention.process_cross_attention(queries, values, fused_proj, q_proj, k_proj, v_proj,
                                    &cached_keys, &cached_values, nullptr, nullptr, beam);

  // Shape: [batch, num_heads, time, d_head] - each head has own K/V
  ASSERT_EQ(cached_keys.shape(), (Shape{BATCH, NUM_HEADS, V_LEN, D_HEAD}));
  ASSERT_EQ(cached_values.shape(), cached_keys.shape());
}

// Attention computes its scores for a block of queries at a time once they would be larger
// than CT2_ATTENTION_MAX_SCORES_BYTES. Each case runs once whole and once with one query per
// block, and the two must agree: SoftMax normalizes every query on its own, and the masks,
// position biases and relative positions are cut to the block's queries.
namespace {

  void set_max_scores_bytes(const char* value) {
#ifdef _WIN32
    _putenv_s("CT2_ATTENTION_MAX_SCORES_BYTES", value ? value : "");
#else
    if (value)
      setenv("CT2_ATTENTION_MAX_SCORES_BYTES", value, 1);
    else
      unsetenv("CT2_ATTENTION_MAX_SCORES_BYTES");
#endif
  }

  struct BlockedAttentionCase {
    std::string name;
    bool self_attention = true;
    bool multi_query = false;
    dim_t num_heads_kv = 4;
    bool relative_attention_bias = false;
    bool relative_positions = false;
    bool asymmetric_relative_positions = false;
    bool alibi = false;
    bool merged = false;  // forward_merged: self keys and encoder memory in one softmax
  };

  constexpr dim_t BLOCKED_HEADS = 4;
  constexpr dim_t BLOCKED_HEAD_DIM = 8;
  constexpr dim_t BLOCKED_MODEL = BLOCKED_HEADS * BLOCKED_HEAD_DIM;

  StorageView random_storage(std::mt19937& gen, Shape shape) {
    std::normal_distribution<float> dist(0.f, 0.3f);
    dim_t size = 1;
    for (const dim_t dim : shape)
      size *= dim;
    std::vector<float> values(size);
    for (auto& value : values)
      value = dist(gen);
    return StorageView(std::move(shape), values);
  }

  class BlockedAttentionModel : public models::Model {
  public:
    BlockedAttentionModel(const BlockedAttentionCase& c, ComputeType compute_type, Device device) {
      std::mt19937 gen(1234);
      const dim_t kv_heads = c.multi_query ? 1 : c.num_heads_kv;
      const dim_t kv_rows = 2 * kv_heads * BLOCKED_HEAD_DIM;
      if (c.self_attention) {
        register_variable("attn/linear_0/weight",
                          random_storage(gen, {BLOCKED_MODEL + kv_rows, BLOCKED_MODEL}));
        register_variable("attn/linear_1/weight",
                          random_storage(gen, {BLOCKED_MODEL, BLOCKED_MODEL}));
      } else {
        register_variable("attn/linear_0/weight",
                          random_storage(gen, {BLOCKED_MODEL, BLOCKED_MODEL}));
        register_variable("attn/linear_1/weight",
                          random_storage(gen, {kv_rows, BLOCKED_MODEL}));
        register_variable("attn/linear_2/weight",
                          random_storage(gen, {BLOCKED_MODEL, BLOCKED_MODEL}));
      }
      if (c.merged)
        register_variable("attn/memory_kv/weight", random_storage(gen, {kv_rows, BLOCKED_MODEL}));
      if (c.multi_query)
        register_variable("attn/multi_query", StorageView(static_cast<int8_t>(1)));
      else if (kv_heads != BLOCKED_HEADS)
        register_variable("attn/num_heads_kv", StorageView(static_cast<int32_t>(kv_heads)));
      if (c.relative_attention_bias) {
        register_variable("attn/relative_attention_bias", random_storage(gen, {32, BLOCKED_HEADS}));
        register_variable("attn/relative_attention_max_distance",
                          StorageView(static_cast<int32_t>(16)));
      }
      if (c.relative_positions) {
        register_variable("attn/relative_position_keys",
                          random_storage(gen, {2 * 3 + 1, BLOCKED_HEAD_DIM}));
        register_variable("attn/relative_position_values",
                          random_storage(gen, {2 * 3 + 1, BLOCKED_HEAD_DIM}));
      }
      if (c.asymmetric_relative_positions) {
        register_variable("attn/relative_asymmetric_position_keys",
                          random_storage(gen, {4 + 2 + 1, BLOCKED_HEAD_DIM}));
        register_variable("attn/relative_left_max_position", StorageView(static_cast<int32_t>(4)));
        register_variable("attn/relative_right_max_position", StorageView(static_cast<int32_t>(2)));
      }
      set_compute_type(compute_type, device, 0);
    }

  protected:
    std::unique_ptr<Model> clone() const override { return nullptr; }
  };

}

class BlockedAttentionTest : public ::testing::TestWithParam<FloatType> {
};

TEST_P(BlockedAttentionTest, BlocksMatchWhole) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;
#ifdef CT2_WITH_SYCL
  if (device == Device::XPU && !xpu::has_gpu())
    GTEST_SKIP() << "no XPU device";
#endif

  std::vector<BlockedAttentionCase> cases(10);
  cases[0].name = "self";
  cases[1].name = "cross";
  cases[1].self_attention = false;
  cases[2].name = "relative_attention_bias";
  cases[2].relative_attention_bias = true;
  cases[3].name = "relative_positions";
  cases[3].relative_positions = true;
  cases[4].name = "asymmetric_relative_positions";
  cases[4].asymmetric_relative_positions = true;
  cases[5].name = "alibi";
  cases[5].alibi = true;
  cases[6].name = "multi_query";  // time and head dimensions merged
  cases[6].multi_query = true;
  cases[7].name = "grouped_query";
  cases[7].num_heads_kv = 2;
  cases[8].name = "cross_multi_query";
  cases[8].self_attention = false;
  cases[8].multi_query = true;
  cases[9].name = "merged";
  cases[9].merged = true;
  cases[9].num_heads_kv = 2;

  constexpr dim_t batch = 2;
  constexpr dim_t num_queries = 7;
  constexpr dim_t num_memory = 9;
  const ComputeType compute_type = dtype == DataType::FLOAT16
    ? ComputeType::FLOAT16
    : ComputeType::FLOAT32;

  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    BlockedAttentionModel model(c, compute_type, device);
    model.set_device(device, 0);
    std::unique_ptr<layers::Alibi> alibi;
    if (c.alibi)
      alibi = std::make_unique<layers::Alibi>();
    const layers::MultiHeadAttention attention(model, "attn", BLOCKED_HEADS, c.self_attention,
                                               /*pre_norm=*/true, /*is_decoder=*/false,
                                               alibi.get());

    std::mt19937 gen(42);
    const StorageView queries =
      random_storage(gen, {batch, num_queries, BLOCKED_MODEL}).to(dtype).to(device);
    const StorageView memory =
      random_storage(gen, {batch, num_memory, BLOCKED_MODEL}).to(dtype).to(device);

    const auto mask = [&](const std::vector<int32_t>& lengths, bool mask_future, bool multi_query) {
      return layers::AttentionLayer::prepare_length_mask(
        StorageView({batch}, lengths, device), BLOCKED_HEADS, num_queries, mask_future, multi_query);
    };

    const auto run = [&](const char* max_scores_bytes, StorageView& output, StorageView& weights) {
      set_max_scores_bytes(max_scores_bytes);
      if (c.merged) {
        const StorageView self_mask = mask({num_queries, num_queries - 2}, true, false);
        const StorageView memory_mask = mask({num_memory, num_memory - 4}, false, false);
        StorageView self_keys(dtype, device), self_values(dtype, device);
        StorageView memory_keys(dtype, device), memory_values(dtype, device);
        attention.forward_merged(queries, &memory, &memory_mask, &self_mask, output,
                                 &self_keys, &self_values, &memory_keys, &memory_values,
                                 nullptr, nullptr, /*offset=*/0);
      } else if (c.self_attention) {
        const StorageView lengths_mask = mask({num_queries, num_queries - 2}, true, c.multi_query);
        attention(queries, queries, &lengths_mask, output, nullptr, nullptr, &weights);
      } else {
        const StorageView lengths_mask = mask({num_memory, num_memory - 4}, false, false);
        attention(queries, memory, &lengths_mask, output, nullptr, nullptr, &weights);
      }
      set_max_scores_bytes(nullptr);
    };

    StorageView whole_output(dtype, device), whole_weights(dtype, device);
    StorageView blocked_output(dtype, device), blocked_weights(dtype, device);
    run(nullptr, whole_output, whole_weights);
    run("1", blocked_output, blocked_weights);

    expect_storage_eq(blocked_output.to_float32(), whole_output.to_float32(), error);
    if (!c.merged)
      expect_storage_eq(blocked_weights.to_float32(), whole_weights.to_float32(), error);
  }
}

INSTANTIATE_TEST_SUITE_P(CPU, BlockedAttentionTest,
                         ::testing::Values(FloatType{Device::CPU, DataType::FLOAT32, 1e-5f}),
                         fp_test_name);
#ifdef CT2_WITH_SYCL
INSTANTIATE_TEST_SUITE_P(XPU, BlockedAttentionTest,
                         ::testing::Values(FloatType{Device::XPU, DataType::FLOAT32, 1e-4f},
                                           FloatType{Device::XPU, DataType::FLOAT16, 1e-2f}),
                         fp_test_name);
#endif
