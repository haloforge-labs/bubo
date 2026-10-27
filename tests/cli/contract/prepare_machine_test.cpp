#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "src/cli/machine_output.hpp"
#include "src/cli/prepare/model_prepare.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "tests/prepare/fixture_catalogue.hpp"

namespace {
using namespace bubo::prepare;
using bubo::cli::MachineWriter;
using gufo::json::Value;
namespace fs = std::filesystem;
class Source : public Transport {
public:
  unsigned requests = 0, resumed = 0;
  std::function<void()> interrupt;
  TransferResult Perform(const Request& request, const HeaderSink& header,
                         const BodySink& body,
                         const std::function<bool()>&) override {
    ++requests;
    const auto range = request.headers.find("range");
    const bool resume = range != request.headers.end();
    resumed += resume;
    Response response{resume ? 206 : 200,
                      {{"content-length", resume ? "2" : "3"},
                       {"etag", "\"abc\""},
                       {"x-linked-etag", std::string(fixture::kAbcHash)},
                       {"x-linked-size", "3"}}};
    if (resume)
      response.headers["content-range"] = "bytes 1-2/3";
    if (!header(response))
      return {};
    const std::array<std::uint8_t, 3> bytes{'a', 'b', 'c'};
    if (interrupt && !resume) {
      body(std::span(bytes).first(1));
      interrupt();
      return {"fixture interruption"};
    }
    body(resume ? std::span(bytes).subspan(1) : std::span(bytes));
    return {};
  }
};
void Check(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Check(argc == 2, "test needs its owned root");
    const fs::path root = fs::absolute(argv[1]);
    Check(!fs::exists(root), "test root must be fresh");
    fs::create_directories(root);
    std::ofstream transcript(root / "events.jsonl", std::ios::binary);
    auto entries = ModelCatalogue();
    for (auto& entry : entries) {
      entry.size = 3;
      entry.sha256 = fixture::kAbcHash;
    }
    Source source;
    ModelTestEnvironment environment;
    environment.entries = entries;
    environment.transport = &source;
    environment.notices = {
        {"LICENSE", "abc", std::string(fixture::kAbcHash)},
        {"NOTICE", "abc", std::string(fixture::kAbcHash)},
        {"PROVENANCE.md", "abc", std::string(fixture::kAbcHash)}};
    environment.volume = [](const fs::path&) {
      return VolumeInfo{
          "fixture-volume", 16ULL * 1024 * 1024 * 1024, "NTFS", false, {}};
    };
    environment.fingerprint = [](const Entry& entry, const fs::path& path) {
      return HashFile(path) == entry.sha256 ? std::string{}
                                            : "fixture mismatch";
    };
    environment.download_options.wait = [](std::chrono::seconds) {};
    enum class InputCase { Approve, Decline, Eof, WorkEof, Cancel, BadDigest };
    unsigned checks = 0;
    {
      const char* retired_args[] = {"model", "--model-set", "hybrid"};
      const auto retired = Parse(retired_args, "fixture");
      Check(!retired && retired.error ==
                            "--model-set requires huihui, original or "
                            "lowmemory and cannot repeat",
            "retired set is rejected with the existing syntax error");
      const auto lowmemory = ModelSetFromName("lowmemory");
      Check(lowmemory.has_value(), "lowmemory resolves by name");
      Check(ModelSetFromPackageId("huihui-qwen38-flash-next-lowmemory-v1") ==
                    lowmemory &&
                ModelSetName(*lowmemory) == "lowmemory",
            "lowmemory resolves by package id and round trips");
      const char* args[] = {
          "model",    "--catalogue-id", "huihui-qwen38-flash-next-lowmemory-v1",
          "--output", "fixture",        "--plan",
          "--offline"};
      const auto parsed = Parse(args, "fixture");
      Check(parsed && parsed.options.model_set == *lowmemory &&
                parsed.options.plan_only && parsed.options.offline,
            "lowmemory catalogue id selects the offline fixed-set planner");
      auto options = parsed.options;
      options.root = (root / "lowmemory-reuse").string();
      options.machine = true;
      auto lowmemory_entries = ModelCatalogue(*lowmemory);
      const auto source_file = root / "lowmemory-abc.gguf";
      std::ofstream(source_file, std::ios::binary) << "abc";
      for (auto& entry : lowmemory_entries) {
        entry.size = 3;
        entry.sha256 = fixture::kAbcHash;
        const auto path = fs::path(ModelRoot(options)) / entry.destination;
        fs::create_directories(path.parent_path());
        fs::create_hard_link(source_file, path);
      }
      auto env = environment;
      env.entries = lowmemory_entries;
      env.package_id = PackageId(*lowmemory);
      env.recipe_id = RecipeId(*lowmemory);
      std::istringstream input;
      std::ostringstream output, errors;
      PreparationPlan plan;
      Check(RunModelForTesting(options, env, input, output, errors, &plan) == 0,
            "lowmemory hard links plan offline");
      const auto* states = plan.machine.find("file_states");
      Check(states && states->size() == 8 &&
                plan.text.find("Total download size: 0 bytes") !=
                    std::string::npos &&
                source.requests == 0,
            "lowmemory reuses all eight hard links without network or download "
            "bytes");
      for (const auto& state : states->items())
        Check(state.is_string() && state.str() == "reuse",
              "lowmemory reuse state");
      fs::remove(fs::path(ModelRoot(options)) /
                 lowmemory_entries[0].destination);
      options.offline = false;
      output.str("");
      errors.str("");
      for (const bool preview : {true, false}) {
        options.plan_only = preview;
        options.machine = preview;
        options.yes = true;
        output.str("");
        errors.str("");
        Check(RunModelForTesting(options, env, input, output, errors,
                                 preview ? &plan : nullptr) == 0 &&
                  errors.str().empty() && source.requests == (preview ? 0 : 1),
              "published lowmemory plans without acquisition and prepares "
              "through the in-process fixture transport");
      }
      // Use the actual pinned package for the published download plan. A plan
      // must never contact the repository, which is private until release.
      options.root = (root / "lowmemory-published-plan").string();
      options.plan_only = true;
      options.machine = true;
      options.yes = true;
      env.entries = ModelCatalogue(*lowmemory);
      Check(
          RunModelForTesting(options, env, input, output, errors, &plan) == 0 &&
              errors.str().empty() && source.requests == 1,
          "published lowmemory plans real pins without a network request");
      const auto* published_states = plan.machine.find("file_states");
      Check(published_states && published_states->size() == env.entries.size(),
            "published plan retains every actual package file");
      for (const auto& state : published_states->items())
        Check(state.is_string() && state.str() == "download",
              "missing published file is planned for download, never fetched");
      ++checks;
    }
    // Independent byte cases catch subtracting the margin twice and unsigned
    // underflow. Both actual planners serialize through PlanVolume.
    constexpr std::uint64_t margin = 2147483648ULL;
    for (const auto& volume : std::array<ConsentVolume, 5>{
             ConsentVolume{"enough", 1000, margin + 200},
             ConsentVolume{"exact", 200, margin + 200},
             ConsentVolume{"short", 100, margin + 200},
             ConsentVolume{"reuse", 1000, margin},
             ConsentVolume{"empty", 0, 0}}) {
      const auto row = PlanVolume(volume);
      const auto expected = volume.name == "enough"  ? 800ULL
                            : volume.name == "reuse" ? 1000ULL
                                                     : 0ULL;
      Check(row.member_size("free_after_download_bytes") == expected &&
                row.member_size("safety_margin_bytes") ==
                    (volume.name == "empty" ? 0 : margin),
            "remaining space excludes the margin and clamps at zero");
      ++checks;
    }
    {
      auto before = Value::object();
      before["volumes"] = Value::array();
      before["volumes"].push_back(PlanVolume({"drive", 1000, margin + 200}));
      auto after = Value::object();
      after["volumes"] = Value::array();
      after["volumes"].push_back(PlanVolume({"drive", 500, margin + 200}));
      Check(PlanDigest(before) == PlanDigest(after),
            "derived remaining space does not change approval identity");
      ++checks;
    }
    source.requests = 0;
    const auto run = [&](InputCase action) {
      Options options;
      options.root = root.string();
      options.step = Step::Model;
      options.machine = true;
      bool sent = false, interrupted = false;
      std::ostringstream out, err;
      std::istringstream unused;
      MachineWriter writer(out);
      const PreparePhase phase = [&](const Options& o, std::istream& input,
                                     std::ostream& output, std::ostream& errors,
                                     PreparationPlan* plan, bool approved) {
        auto env = environment;
        env.cancelled = o.cancelled;
        return RunModelForTesting(o, env, input, output, errors, plan,
                                  approved);
      };
      source.interrupt =
          (action == InputCase::Cancel || action == InputCase::WorkEof)
              ? std::function<void()>([&] { interrupted = true; })
              : std::function<void()>{};
      const auto read = [&](std::string& line) {
        if (sent) {
          if (action == InputCase::WorkEof)
            return -1;
          if (interrupted && action == InputCase::Cancel) {
            auto cancel = Value::object();
            cancel["action"] = "cancel";
            cancel["operation_id"] = writer.OperationId();
            line = cancel.dump();
            interrupted = false;
            return 1;
          }
          return 0;
        }
        if (action == InputCase::Eof)
          return -1;
        std::istringstream records(out.str());
        std::string row;
        Value request;
        while (std::getline(records, row)) {
          auto event = gufo::json::parse(row);
          if (event.member_str("type") == "approval_required")
            request = *event.find("payload");
        }
        Check(!request.is_null(), "approval record must precede input");
        auto approval = Value::object();
        approval["action"] =
            action == InputCase::Decline ? "decline" : "approve";
        approval["operation_id"] = writer.OperationId();
        approval["plan_digest"] = action == InputCase::BadDigest
                                      ? std::string(64, '0')
                                      : request.member_str("plan_digest");
        line = approval.dump();
        sent = true;
        return 1;
      };
      const auto code =
          RunMachine(options, phase, {}, unused, writer, err, {}, read);
      transcript << out.str();
      return code;
    };
    Check(run(InputCase::Decline) == 3 && !source.requests,
          "decline has no transfers");
    ++checks;
    Check(run(InputCase::Eof) == 3 && !source.requests,
          "approval EOF declines");
    ++checks;
    Check(run(InputCase::BadDigest) == 2 && !source.requests,
          "wrong digest rejects");
    ++checks;
    Check(run(InputCase::Cancel) == 3,
          "cancel uses existing download cancellation");
    ++checks;
    Check(run(InputCase::WorkEof) == 3, "work EOF cancels");
    ++checks;
    Check(run(InputCase::Approve) == 0, "approval installs from fake sources");
    ++checks;
    Check(source.resumed > 0,
          "resume sends Range from the retained fixture prefix");
    ++checks;
    const auto requests = source.requests;
    Check(run(InputCase::Eof) == 0 && source.requests == requests,
          "no-op has no approval or transfers");
    ++checks;
    // A changed plan must produce another approval before its phase can run.
    {
      Options options;
      options.root = root.string();
      options.step = Step::Model;
      options.machine = true;
      unsigned generation = 1, approvals = 0, last_request = 0;
      bool complete = false;
      std::ostringstream output, errors;
      std::istringstream input;
      MachineWriter writer(output);
      const auto plan = [&] {
        auto value = Value::object();
        value["component"] = "model";
        value["ready"] = complete;
        value["sources"] = Value::array();
        value["licenses"] = Value::array();
        value["paths"] = Value::object();
        value["volumes"] = Value::array();
        value["fixture_generation"] =
            static_cast<unsigned long long>(generation);
        return value;
      };
      const PreparePhase phase = [&](const Options& o, std::istream&,
                                     std::ostream&, std::ostream&,
                                     PreparationPlan* preview, bool) {
        if (preview) {
          preview->ready = complete;
          preview->machine = plan();
          return 0;
        }
        if (generation == 1)
          ++generation;
        if (!o.authorize_plan(plan()))
          return 3;
        complete = true;
        return 0;
      };
      const auto read = [&](std::string& line) {
        std::istringstream records(output.str());
        std::string record;
        while (std::getline(records, record)) {
          const auto event = gufo::json::parse(record);
          if (event.member_str("type") != "approval_required" ||
              event.member_size("seq") <= last_request)
            continue;
          last_request = static_cast<unsigned>(event.member_size("seq"));
          auto reply = Value::object();
          reply["action"] = "approve";
          reply["operation_id"] = writer.OperationId();
          reply["plan_digest"] =
              event.find("payload")->member_str("plan_digest");
          line = reply.dump();
          ++approvals;
          return 1;
        }
        return 0;
      };
      Check(RunMachine(options, phase, {}, input, writer, errors, {}, read) ==
                    0 &&
                approvals == 2,
            "changed plan requires fresh approval");
      ++checks;
      transcript << output.str();
    }
    // A machine error names its cause in `detail`: the failing phase's own
    // "prepare: E<code> <text>" line, without that prefix. The line still
    // reaches stderr. A line for another code is not the cause.
    {
      const auto error_of = [&](const char* line, int code, bool short_space,
                                std::string* stderr_text) {
        Options options;
        options.root = root.string();
        options.step = Step::Model;
        options.machine = true;
        std::ostringstream output, errors;
        std::istringstream input;
        MachineWriter writer(output);
        const PreparePhase phase = [&](const Options&, std::istream&,
                                       std::ostream&, std::ostream& err,
                                       PreparationPlan* preview, bool) {
          err << line;
          if (!short_space)
            return code;
          auto plan = Value::object();
          plan["component"] = "model";
          plan["ready"] = false;
          plan["sources"] = Value::array();
          plan["licenses"] = Value::array();
          plan["paths"] = Value::object();
          plan["volumes"] = Value::array();
          preview->ready = false;
          preview->machine = plan;
          preview->volumes.push_back({"C:", 100, 200});
          return 0;
        };
        const auto read = [](std::string&) { return -1; };
        RunMachine(options, phase, {}, input, writer, errors, {}, read);
        transcript << output.str();
        *stderr_text = errors.str();
        std::istringstream records(output.str());
        std::string record;
        Value payload;
        while (std::getline(records, record)) {
          const auto event = gufo::json::parse(record);
          if (event.member_str("type") == "error")
            payload = *event.find("payload");
        }
        Check(!payload.is_null(), "a failed prepare writes an error record");
        return payload;
      };
      std::string text;
      auto payload = error_of("prepare: E7 no existing ancestor for Q:\\Data\n",
                              7, false, &text);
      Check(payload.member_str("code") == "storage_failed" &&
                payload.find("detail") &&
                payload.member_str("detail") ==
                    "no existing ancestor for Q:\\Data" &&
                text.find("no existing ancestor") != std::string::npos,
            "storage error carries the phase's text as detail");
      ++checks;
      payload =
          error_of("prepare: E4 an earlier unrelated line\n", 7, false, &text);
      Check(payload.member_str("code") == "storage_failed" &&
                !payload.find("detail"),
            "a line for another code is not the detail");
      ++checks;
      payload = error_of("", 0, true, &text);
      Check(payload.member_str("code") == "storage_failed" &&
                payload.find("detail") &&
                payload.member_str("detail").find("Not enough free space on "
                                                  "C:") == 0 &&
                text.find("Not enough free space") != std::string::npos,
            "free-space refusal names the drive and the numbers");
      ++checks;
    }
    // Progress is throttled: a download reports once per network chunk, which
    // was one JSON line (and one writer thread) per chunk. At most one record
    // per 250 ms per file, and the record with bytes_done == bytes_total.
    {
      Options options;
      options.root = root.string();
      options.step = Step::Model;
      options.machine = true;
      std::ostringstream output, errors;
      std::istringstream input;
      MachineWriter writer(output);
      constexpr std::uint64_t total = 20000;
      const PreparePhase phase = [&](const Options& o, std::istream&,
                                     std::ostream&, std::ostream&,
                                     PreparationPlan* preview, bool) {
        for (std::uint64_t done = 1; done <= total; ++done)
          ReportProgress(o, "model", "download", "fixture-file", done, total);
        auto plan = Value::object();
        plan["component"] = "model";
        plan["ready"] = true;
        plan["sources"] = Value::array();
        plan["licenses"] = Value::array();
        plan["paths"] = Value::object();
        plan["volumes"] = Value::array();
        preview->ready = true;
        preview->machine = plan;
        return 0;
      };
      const auto read = [](std::string&) { return -1; };
      Check(
          RunMachine(options, phase, {}, input, writer, errors, {}, read) == 0,
          "progress fixture prepares");
      std::istringstream records(output.str());
      std::string record;
      std::uint64_t progress = 0, finals = 0;
      while (std::getline(records, record)) {
        const auto event = gufo::json::parse(record);
        if (event.member_str("type") != "progress")
          continue;
        ++progress;
        const auto& payload = *event.find("payload");
        finals += payload.member_size("bytes_done") == total;
      }
      // Two passes (preflight and completion) of 20000 updates each.
      Check(progress >= 2 && progress <= 40 && finals == 2,
            "progress is throttled and keeps the final record");
      ++checks;
    }
    // After `prepare runtime --runtime-dir`, the result names the variable that
    // makes later commands find the runtime, with the folder the run resolved.
    // Without --runtime-dir the default location serves and there is no field.
    for (const bool explicit_dir : {true, false}) {
      Options options;
      options.root = root.string();
      options.step = Step::Runtime;
      options.machine = true;
      options.yes = true;
      if (explicit_dir)
        options.runtime_dir = "C:\\fixture-runtime";
      std::ostringstream output, errors;
      std::istringstream input;
      MachineWriter writer(output);
      const PreparePhase phase = [&](const Options&, std::istream&,
                                     std::ostream&, std::ostream&,
                                     PreparationPlan* preview, bool) {
        auto plan = Value::object();
        plan["component"] = "runtime";
        plan["ready"] = true;
        plan["sources"] = Value::array();
        plan["licenses"] = Value::array();
        plan["paths"] = Value::object();
        plan["volumes"] = Value::array();
        preview->ready = true;
        preview->machine = plan;
        preview->runtime_dir = "C:\\fixture-runtime\\resolved";
        return 0;
      };
      const auto read = [](std::string&) { return -1; };
      Check(
          RunMachine(options, {}, phase, input, writer, errors, {}, read) == 0,
          "runtime-only machine run succeeds");
      std::istringstream records(output.str());
      std::string record;
      Value result;
      while (std::getline(records, record)) {
        const auto event = gufo::json::parse(record);
        if (event.member_str("type") == "result")
          result = *event.find("payload");
      }
      Check(!result.is_null(), "a successful run writes a result record");
      if (explicit_dir)
        Check(result.member_str("runtime_environment") ==
                  "BUBO_RUNTIME_DIR=C:\\fixture-runtime\\resolved",
              "result names BUBO_RUNTIME_DIR and the resolved folder");
      else
        Check(result.find("runtime_environment") == nullptr,
              "no field without --runtime-dir");
      ++checks;
    }
    transcript.close();
    std::cout << "Prepare machine: " << checks << " passed, 0 failed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
