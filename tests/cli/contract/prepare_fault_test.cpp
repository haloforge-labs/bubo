#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "src/cli/machine_output.hpp"
#include "src/cli/prepare/model_prepare.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "tests/prepare/fixture_catalogue.hpp"

namespace {
using namespace bubo::prepare;
using bubo::cli::MachineWriter;
using gufo::json::Value;
namespace fs = std::filesystem;
void Check(bool ok, const std::string& name) {
  if (!ok)
    throw std::runtime_error(name);
}
Value Plan(bool ready) {
  auto plan = Value::object();
  plan["component"] = "model";
  plan["ready"] = ready;
  plan["sources"] = Value::array();
  plan["licenses"] = Value::array();
  plan["paths"] = Value::object();
  plan["volumes"] = Value::array();
  return plan;
}
Value ApprovalRequest(const std::string& transcript) {
  Value request;
  std::istringstream lines(transcript);
  std::string line;
  while (std::getline(lines, line)) {
    const auto event = gufo::json::parse(line);
    if (event.member_str("type") == "approval_required")
      request = *event.find("payload");
  }
  return request;
}
class ResultFailure : public std::streambuf {
public:
  std::string bytes;
  bool saw_result = false;
  std::streamsize xsputn(const char* text, std::streamsize count) override {
    if (std::string_view(text, static_cast<std::size_t>(count))
            .find("\"type\":\"result\"") != std::string_view::npos) {
      saw_result = true;
      return 0;
    }
    bytes.append(text, static_cast<std::size_t>(count));
    return count;
  }
};
class Source : public Transport {
public:
  std::atomic<bool> entered{false}, release{false};
  TransferResult Perform(const Request&, const HeaderSink& header,
                         const BodySink& body,
                         const std::function<bool()>&) override {
    entered = true;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!release && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (!release)
      return {"test barrier timed out"};
    header({200,
            {{"content-length", "3"},
             {"etag", "\"abc\""},
             {"x-linked-etag", std::string(fixture::kAbcHash)},
             {"x-linked-size", "3"}}});
    const std::array<std::uint8_t, 3> bytes{'a', 'b', 'c'};
    body(bytes);
    return {};
  }
};
std::string ConcurrentPrepare(const fs::path& root) {
  Options options;
  const auto utf8_root = root.u8string();
  options.root = std::string(utf8_root.begin(), utf8_root.end());
  options.step = Step::Model;
  options.machine = true;
  options.yes = true;
  Source source;
  ModelTestEnvironment environment;
  auto entries = ModelCatalogue();
  for (auto& entry : entries) {
    entry.size = 3;
    entry.sha256 = fixture::kAbcHash;
  }
  environment.entries = entries;
  environment.transport = &source;
  environment.notices = {
      {"LICENSE", "abc", std::string(fixture::kAbcHash)},
      {"NOTICE", "abc", std::string(fixture::kAbcHash)},
      {"PROVENANCE.md", "abc", std::string(fixture::kAbcHash)}};
  environment.volume = [](const fs::path&) {
    return VolumeInfo{"fixture", 16ULL * 1024 * 1024 * 1024, "NTFS", false, {}};
  };
  environment.fingerprint = [](const Entry& entry, const fs::path& path) {
    return HashFile(path) == entry.sha256 ? std::string{} : "hash mismatch";
  };
  environment.download_options.wait = [](std::chrono::seconds) {};
  const PreparePhase phase = [&](const Options& o, std::istream& input,
                                 std::ostream& output, std::ostream& errors,
                                 PreparationPlan* preview, bool approved) {
    auto env = environment;
    env.cancelled = o.cancelled;
    return RunModelForTesting(o, env, input, output, errors, preview, approved);
  };
  int first_code = -1;
  std::ostringstream first_output;
  std::thread first([&] {
    std::ostringstream errors;
    std::istringstream input;
    MachineWriter writer(first_output);
    first_code = RunMachine(options, phase, {}, input, writer, errors, {},
                            [](std::string&) { return 0; });
  });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!source.entered && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  const bool entered = source.entered;
  std::ostringstream output, errors;
  std::istringstream input;
  MachineWriter writer(output);
  const auto start = std::chrono::steady_clock::now();
  const auto second_code =
      entered ? RunMachine(options, phase, {}, input, writer, errors, {},
                           [](std::string&) { return 0; })
              : -1;
  source.release = true;
  first.join();
  Check(entered && second_code == 8 && first_code == 0 &&
            output.str().find("prepare_locked") != std::string::npos &&
            std::chrono::steady_clock::now() - start < std::chrono::seconds(3),
        "two_concurrent_prepares_typed_lock");
  Check(fs::is_regular_file(NativePath(fs::u8path(ModelRoot(options))) /
                            "bubo-package.json"),
        "unicode_long_prepare_destination");
  return first_output.str() + output.str();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Check(argc == 2, "test needs its owned root");
    const auto root = fs::absolute(argv[1]);
    Check(!fs::exists(root), "test root must be fresh");
    fs::create_directories(root);
    std::ofstream transcript(root / "events.jsonl", std::ios::binary);
    unsigned passed = 0;
    {
      const auto lowmemory = ModelSetFromName("lowmemory");
      Check(lowmemory.has_value(), "lowmemory integrity fixture resolves");
      auto entries = ModelCatalogue(*lowmemory);
      for (auto& entry : entries) {
        entry.size = 3;
        entry.sha256 = fixture::kAbcHash;
      }
      Options options;
      options.root = (root / "lowmemory-integrity").string();
      options.step = Step::Model;
      options.model_set = *lowmemory;
      options.catalogue_id = "huihui-qwen38-flash-next-lowmemory-v1";
      options.offline = true;
      options.plan_only = true;
      const auto package = fs::path(ModelRoot(options));
      for (const auto& entry : entries) {
        const auto path = package / entry.destination;
        fs::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary) << "abc";
      }
      // This is an independent tiny file, never a hard link to real weights.
      std::ofstream(package / entries[1].destination, std::ios::binary)
          << "abd";
      ModelTestEnvironment environment;
      environment.entries = entries;
      environment.notices = {
          {"LICENSE", "abc", std::string(fixture::kAbcHash)},
          {"NOTICE", "abc", std::string(fixture::kAbcHash)},
          {"PROVENANCE.md", "abc", std::string(fixture::kAbcHash)}};
      environment.volume = [](const fs::path&) {
        return VolumeInfo{"fixture", 16ULL << 30, "NTFS", false, {}};
      };
      environment.fingerprint = [](const Entry& entry, const fs::path& path) {
        return HashFile(path) == entry.sha256 ? std::string{}
                                              : "fixture mismatch";
      };
      std::istringstream input;
      std::ostringstream output, errors;
      PreparationPlan plan;
      Check(RunModelForTesting(options, environment, input, output, errors,
                               &plan) == 4 &&
                errors.str().find("offline: cannot replace the damaged") !=
                    std::string::npos &&
                errors.str().find(entries[1].sha256) != std::string::npos &&
                HashFile(package / entries[1].destination) != entries[1].sha256,
            "damaged lowmemory shard is detected and retained by the integrity "
            "path");
      std::cout << "lowmemory_integrity: 1 passed, 0 failed\n";
      ++passed;
    }
    const auto run = [&](std::string name, int expected) {
      Options options;
      options.step = Step::Model;
      options.machine = true;
      options.root = root.string();
      std::ostringstream output, errors;
      std::istringstream input;
      MachineWriter writer(output);
      std::string stage = "approval";
      bool sent = false, complete = false, fault_sent = false;
      Value approval;
      const PreparePhase phase = [&](const Options& o, std::istream&,
                                     std::ostream&, std::ostream&,
                                     PreparationPlan* preview, bool) {
        if (preview) {
          if (complete) {
            stage = "completion";
            if (o.cancelled())
              return 9;
          }
          preview->ready = complete;
          preview->machine = Plan(complete);
          return 0;
        }
        if (!o.authorize_plan(Plan(false)))
          return 3;
        for (const auto* current :
             {"download", "copy", "verify", "extract", "promote"}) {
          stage = current;
          ReportProgress(o, "model", stage, "fixture", 1, 3);
          if (o.cancelled())
            return 9;  // Prove cancellation survives phase error mapping.
        }
        complete = true;
        return 0;
      };
      const auto read = [&](std::string& line) {
        if (!sent) {
          sent = true;
          if (name == "stdin_garbage") {
            line = "garbage";
            return 1;
          }
          if (name == "eof_approval")
            return -1;
          approval = Value::object();
          approval["action"] =
              name == "cancel_before_approval" ? "cancel" : "approve";
          approval["operation_id"] = name == "wrong_operation_id"
                                         ? std::string(32, '0')
                                         : writer.OperationId();
          approval["plan_digest"] =
              name == "stale_plan_digest"
                  ? std::string(64, '0')
                  : ApprovalRequest(output.str()).member_str("plan_digest");
          line = approval.dump();
          return 1;
        }
        if (fault_sent)
          return 0;
        if (name == "double_approve") {
          fault_sent = true;
          line = approval.dump();
          return 1;
        }
        if (name == "stdin_garbage_during_verify" && stage == "verify") {
          line = "{";
          fault_sent = true;
          return 1;
        }
        if (name == "cancel_during_verify" && stage == "verify") {
          auto cancel = Value::object();
          cancel["action"] = "cancel";
          cancel["operation_id"] = writer.OperationId();
          line = cancel.dump();
          fault_sent = true;
          return 1;
        }
        if (name == "eof_" + stage)
          return -1;
        return 0;
      };
      const auto code =
          RunMachine(options, phase, {}, input, writer, errors, {}, read);
      Check(code == expected, name + ": got exit " + std::to_string(code));
      Check(output.str().find(expected == 2 ? "invalid_arguments"
                                            : "cancelled") != std::string::npos,
            name + ": missing typed error");
      transcript << output.str();
      std::cout << name << ": 1 passed, 0 failed\n";
      ++passed;
    };
    for (const auto* name :
         {"stdin_garbage", "wrong_operation_id", "stale_plan_digest",
          "double_approve", "stdin_garbage_during_verify"})
      run(name, 2);
    for (const auto* name :
         {"cancel_before_approval", "cancel_during_verify", "eof_approval",
          "eof_download", "eof_copy", "eof_verify", "eof_extract",
          "eof_promote", "eof_completion"})
      run(name, 3);
    {
      // Large legal text and a source array each exceed the wire limit.
      auto plan = Plan(true);
      std::string text;
      for (unsigned i = 0; i < 200000; ++i)
        text += "é中🙂\r\n\"\\";
      auto legal = Value::object();
      legal["path"] = "licences/中-LICENSE";
      legal["sha256"] = std::string(64, 'a');
      legal["text"] = text;
      plan["licenses"].push_back(legal);
      for (unsigned i = 0; i < 300; ++i)
        plan["sources"].push_back(PlanSource(
            "file-" + std::to_string(i),
            "https://example.invalid/" + std::string(4096, 'x'), 3,
            std::string(64, 'a'), "fixture", "pinned", "中/é", "fixture"));
      plan["paths"]["model"] = "C:/é/中/" + std::string(32700, 'p');
      Options options;
      options.machine = true;
      options.step = Step::Model;
      std::ostringstream output, errors;
      std::istringstream input;
      MachineWriter writer(output);
      const PreparePhase phase = [&](const Options&, std::istream&,
                                     std::ostream&, std::ostream&,
                                     PreparationPlan* preview, bool) {
        preview->ready = true;
        preview->machine = plan;
        return 0;
      };
      Check(RunMachine(options, phase, {}, input, writer, errors) == 0,
            "huge_plan_component_records exit");
      std::istringstream lines(output.str());
      const auto digest = PlanDigest(plan);
      std::string line, restored;
      unsigned sources = 0, parts = 0, expected_parts = 0;
      Value paths;
      while (std::getline(lines, line)) {
        Check(line.size() + 1 <= 1024 * 1024 &&
                  line.find('\r') == std::string::npos,
              "event size / escaped CRLF");
        const auto event = gufo::json::parse(line);
        if (event.member_str("type") != "plan")
          continue;
        const auto& row = *event.find("payload");
        Check(row.member_size("part") == parts++, "contiguous parts");
        expected_parts = static_cast<unsigned>(row.member_size("parts"));
        Check(row.member_str("plan_digest") == digest, "full digest");
        sources += static_cast<unsigned>(row.find("sources")->size());
        for (const auto& item : row.find("licenses")->items()) {
          Check(item.member_size("text_offset") == restored.size() &&
                    item.member_size("text_bytes") == text.size(),
                "legal byte offsets");
          restored += item.member_str("text");
        }
        if (row.member_str("record_kind") == "path")
          paths = *row.find("paths");
      }
      Check(parts == expected_parts && sources == 300 && restored == text &&
                paths.dump() == plan.find("paths")->dump(),
            "huge_plan_component_records reconstruction");
      transcript << output.str();
      std::cout << "huge_plan_component_records: 1 passed, 0 failed\n"
                   "utf8_long_paths_licence_crlf: 1 passed, 0 failed\n";
      passed += 2;
    }
    transcript << ConcurrentPrepare(root / fs::u8path("concurrent-é-中") /
                                    std::string(100, 'a') /
                                    std::string(100, 'b'));
    std::cout << "two_concurrent_prepares_typed_lock: 1 passed, 0 failed\n";
    ++passed;
    std::cout << "unicode_long_prepare_destination: 1 passed, 0 failed\n";
    ++passed;
    {
      Options options;
      options.step = Step::Model;
      options.root = "relative-fixture";
      options.machine = true;
      const PreparePhase ready = [](const Options&, std::istream&,
                                    std::ostream&, std::ostream&,
                                    PreparationPlan* preview, bool) {
        preview->ready = true;
        preview->machine = Plan(true);
        return 0;
      };
      std::string published;
      options.publish_default = [&](const std::string& path) {
        published = path;
        return 0;
      };
      std::ostringstream output, errors;
      std::istringstream input;
      MachineWriter writer(output);
      Check(RunMachine(options, ready, {}, input, writer, errors) == 0 &&
                fs::u8path(published).is_absolute(),
            "relative_output_manifest_absolute");
      std::istringstream lines(output.str());
      std::string line;
      Value result;
      while (std::getline(lines, line)) {
        const auto event = gufo::json::parse(line);
        if (event.member_str("type") == "result")
          result = *event.find("payload");
      }
      Check(result.member_str("manifest_path") == published,
            "result and default pointer use the same absolute path");
      transcript << output.str();
      std::cout << "relative_output_manifest_absolute: 1 passed, 0 failed\n";
      ++passed;
      ResultFailure buffer;
      std::ostream failing(&buffer);
      MachineWriter failed_writer(failing);
      Check(RunMachine(options, ready, {}, input, failed_writer, errors) == 3 &&
                buffer.saw_result,
            "stdout_final_result_failure");
      transcript << buffer.bytes;
      std::cout << "stdout_final_result_failure: 1 passed, 0 failed\n";
      ++passed;
    }
    std::cout << "Prepare fault injection: " << passed << " passed, 0 failed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
