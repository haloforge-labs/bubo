// DiffusionGemma tokenizer against Hugging Face `tokenizers` ids for the
// checkpoint's tokenizer.json. Needs DIFFUSION_GEMMA_MODEL=<checkpoint dir>;
// skips (77) without it.
#include "src/models/diffusion_gemma/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace dg = gufo::models::diffusion_gemma;

int main() {
  const char* root = std::getenv("DIFFUSION_GEMMA_MODEL");
  if (root == nullptr) {
    std::puts("SKIP: set DIFFUSION_GEMMA_MODEL");
    return 77;
  }
  const dg::Tokenizer tokenizer(root);
  struct Case {
    std::string text;
    std::vector<std::int32_t> ids;
  };
  // Reference ids: tokenizers.Tokenizer.from_file(...).encode(text,
  // add_special_tokens=False).ids
  const std::vector<Case> cases = {
      {"Hello world", {9259, 1902}},
      {"  leading spaces and  double  spaces   ",
       {138, 26016, 9952, 532, 138, 7902, 138, 35220, 139}},
      {"<bos><|turn>user\nWhy is the sky blue?<turn|>\n<|turn>model\n",
       {2, 105, 2364, 107, 11355, 563, 506, 7217, 3730, 236881, 106, 107, 105,
        4368, 107}},
      {"Unicode: caf\xc3\xa9 na\xc3\xafve \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e"
       "\xe3\x81\xae\xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88 "
       "\xf0\x9f\xa4\x96\xf0\x9f\x9a\x80 \xce\x95\xce\xbb\xce\xbb\xce\xb7\xce"
       "\xbd\xce\xb9\xce\xba\xce\xac \xd7\xa2\xd7\x91\xd7\xa8\xd7\x99\xd7\xaa",
       {123540, 236787, 33443, 120362, 33375, 238582, 236945, 95830, 236743,
        246143, 242015, 23883, 109959, 42220, 11173, 104154, 237270}},
      {"Weird bytes \x7f and control \x01\x02 chars",
       {1882, 2105, 17234, 236743, 251541, 532, 2256, 236743, 249732, 241247,
        56256}},
  };
  int failures = 0;
  for (const auto& c : cases) {
    const auto ids = tokenizer.Encode(c.text);
    if (ids != c.ids) {
      std::fprintf(stderr, "FAIL: encoding of \"%s\"\n", c.text.c_str());
      ++failures;
    }
    if (tokenizer.Decode(ids, false) != c.text) {
      std::fprintf(stderr, "FAIL: round trip of \"%s\"\n", c.text.c_str());
      ++failures;
    }
  }
  if (failures == 0)
    std::puts("PASS diffusion_gemma_tokenizer_test");
  return failures == 0 ? 0 : 1;
}
