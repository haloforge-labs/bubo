// Production CLI commands with an arranged OS probe, never a GPU server.
// Fixture provenance: data/memory96-derived.json. These are derived Windows
// arrangements, not measurements from a 96 GiB computer.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/cli/instance_control.hpp"
#include "src/cli/machine_adapter.hpp"
#include "src/cli/machine_warnings.hpp"
#include "src/cli/models_support.hpp"
#include "src/cli/prepare/prepare.hpp"
#include "src/cli/prepared_default.hpp"
#include "src/cli/serve/memory_policy.hpp"
#include "src/cli/serve/release_profile.hpp"
#include "src/cli/settings_support.hpp"
#include "src/cli/status.hpp"

namespace {
using namespace bubo::cli;
constexpr std::uint64_t GiB = 1073741824ULL;
std::string arrangement = "96-64";
std::uint64_t installed = 96 * GiB;
std::uint64_t pci = 0x10021586;
bool identity_known = true;
std::uint64_t host_available = 11172806656ULL;
std::optional<std::uint64_t> dedicated_override;
bool budget_known = true;
int dxgi_reads = 0;
MachineReading Reading(std::uint64_t value) {
  return AvailableReading(value,
                          "derived arrangement; see memory96-derived.json",
                          "test probe", "bytes", "derived, not sampled");
}
struct Probe final : MachineProbe {
  MachineReading None() {
    return UnavailableReading("arranged probe", "", "", "not sampled",
                              "Not part of this memory arrangement.");
  }
  MachineReading OverlayScheme() override { return None(); }
  MachineReading EffectivePowerMode() override { return None(); }
  MachineReading PowerSource() override { return None(); }
  MachineGpuReadings Gpu() override {
    MachineGpuReadings out;
    out.power = None();
    out.clock = None();
    return out;
  }
  MachineVgmReadings Vgm() override { return {None(), None(), None()}; }
  MachineReading HostPhysicalMemory() override {
    return Reading(33982058496ULL - (arrangement == "96-72" ? 8 * GiB : 0));
  }
  MachineReading HostAvailableMemory() override {
    return Reading(host_available);
  }
  MachineReading InstalledPhysicalMemory() override {
    return identity_known ? Reading(installed) : None();
  }
  MachineReading OemManufacturer() override { return None(); }
  MachineReading OemModel() override { return None(); }
  MachineReading BiosVendor() override { return None(); }
  MachineReading BiosVersion() override { return None(); }
  MachineReading BiosDate() override { return None(); }
  MachineDxgiReadings Dxgi() override {
    ++dxgi_reads;
    const auto carve_cut = arrangement == "96-64"   ? 32 * GiB
                           : arrangement == "96-72" ? 24 * GiB
                                                    : 0;
    const auto budget_cut = arrangement == "96-64"   ? 32 * GiB
                            : arrangement == "96-72" ? 28 * GiB
                                                     : 0;
    MachineDxgiAdapterInfo adapter;
    adapter.name = "AMD Radeon(TM) 8060S Graphics";
    adapter.vendor_id = static_cast<std::uint32_t>(pci >> 16);
    adapter.device_id =
        identity_known ? static_cast<std::uint32_t>(pci & 0xffff) : 0;
    adapter.dedicated_video_memory = 102892945408ULL - carve_cut;
    if (dedicated_override)
      adapter.dedicated_video_memory = *dedicated_override;
    adapter.shared_system_memory =
        (33982058496ULL - (arrangement == "96-72" ? 8 * GiB : 0)) / 2;
    adapter.has_adapter3 = true;
    adapter.query_ok = budget_known;
    adapter.budget = 119078668288ULL - budget_cut;
    adapter.nonlocal_budget = 0;
    auto readings =
        DxgiReadingsFromAdapters({adapter}, "", "derived, not sampled");
    // This arrangement models admission and capacity, not reservation offers.
    // Do not present the adapter struct's zero initializer as a reading.
    readings.available_for_reservation = None();
    return readings;
  }
};
int passed = 0, failed = 0;
void Check(bool condition, const std::string& what) {
  if (condition)
    ++passed;
  else {
    ++failed;
    std::cerr << "FAIL: " << what << '\n';
  }
}
const Value& At(const Value& value, const char* key) {
  static const Value none;
  const auto* child = value.find(key);
  return child ? *child : none;
}
const Value* DefaultRow(const Value& list) {
  const auto& rows = At(list, "catalogue");
  if (rows.is_array())
    for (const auto& row : rows.items())
      if (row.member_str("default_status") == "default")
        return &row;
  return nullptr;
}
void Cases(const std::filesystem::path& root) {
  const auto& catalogue = ShippedCatalogue();
  const auto* low = FindCatalogueEntry(catalogue.entries,
                                       "huihui-qwen38-flash-next-lowmemory-v1");
  const auto* standard = FindCatalogueEntry(
      catalogue.entries, "huihui-qwen38-flash-next-ud-q4-k-xl-v1");
  Check(low && standard, "real pinned catalogue packages exist");
  if (!low || !standard)
    return;
  for (const auto& name : {"96-64", "96-72", "128-96"}) {
    arrangement = name;
    installed = arrangement == "128-96" ? 128 * GiB : 96 * GiB;
    Probe probe;
    const auto budgets = ReadPlanningMemoryBudgets(probe);
    for (std::uint64_t sessions : {1ULL, 2ULL}) {
      EstimateRequest request;
      request.sessions = sessions;
      request.available_bytes = budgets.gpu_available;
      request.host_available_bytes = budgets.host_available;
      request.capacity_budgets = budgets;
      auto result = EstimateForCatalogueEntry(*low, root, request);
      const std::string label = arrangement + "/s" + std::to_string(sessions);
      Check(result.state == "estimated",
            label + " lower-memory estimate available");
      if (result.memory_estimate.is_null())
        continue;
      const auto& e = result.memory_estimate;
      Check(
          At(e, "weights").member_size("bytes") == 61064419200ULL,
          label + " calibrated main plus actual selected sidecar coefficients");
      const std::uint64_t expected =
          arrangement == "96-64"   ? (sessions == 1 ? 24576 : 8192)
          : arrangement == "96-72" ? (sessions == 1 ? 200000 : 139264)
                                   : (sessions == 1 ? 200000 : 262144);
      Check(e.member_size("context_tokens") == expected,
            label + " shared default-context recommendation");
      Check(At(e, "fits").is_bool() && At(e, "fits").as_bool(),
            label + " resolved startup fits");
      Check(At(e, "recommended").is_bool() && At(e, "recommended").as_bool(),
            label + " keeps unchanged cache and margin rules");
      request.context = 200000;
      const auto explicit_result =
          EstimateForCatalogueEntry(*low, root, request);
      Check(explicit_result.memory_estimate.member_size("context_tokens") ==
                200000,
            label + " explicit context is authoritative");
      const auto normal = EstimateForCatalogueEntry(*standard, root, request);
      Check(
          StartupMemoryRefusal(budgets, normal.memory_estimate, true) ==
              (installed == 96 * GiB
                   ? "The standard model does not fit the available memory. "
                     "Choose the lower-memory model."
                   : ""),
          label +
              " production pre-load refusal uses hardware and definite nonfit");
      if (installed == 96 * GiB)
        Check(!At(normal.memory_estimate, "fits").as_bool(),
              label + " standard default is a definite nonfit");
      else if (sessions == 1)
        Check(At(At(normal.memory_estimate, "gpu"), "need")
                      .member_size("bytes") == 97728669888ULL,
              "128 standard estimate remains exactly unchanged");
    }
    ModelsListOptions options;
    options.models_root = root;
    options.available_bytes = budgets.gpu_available;
    options.host_available_bytes = budgets.host_available;
    options.capacity_budgets = budgets;
    const auto list = BuildModelsList(options, catalogue.entries);
    const auto* chosen = DefaultRow(list);
    Check(chosen && chosen->member_str("id") ==
                        (installed == 96 * GiB ? low->id : standard->id),
          arrangement + " first-run catalogue default");
    for (const auto& row : At(list, "catalogue").items()) {
      if (row.member_str("id") != standard->id &&
          row.member_str("id") != "qwen38-flash-next-base-ud-q4-k-xl-v1")
        continue;
      Check(row.find("download_enabled")->as_bool() == (installed != 96 * GiB),
            arrangement +
                " full-size options remain visible with hardware-only "
                "eligibility");
      if (installed == 96 * GiB)
        Check(row.member_str("disabled_reason")
                      .find("Does not fit this computer: needs about 91 GiB") ==
                  0,
              arrangement + " refusal text uses the CLI estimate in GiB");
    }
    auto warnings = EvaluateMachineWarnings(ReadMachineWarningInputs(probe));
    bool reservation_warning = false;
    for (const auto& warning : warnings)
      reservation_warning |= warning.check == "gpu_reservation";
    Check(!reservation_warning,
          arrangement + " suitable reservation has no warning");
    bubo::prepare::Options prepare;
    prepare.local_app_data = (root / arrangement / "local").string();
    prepare.root = (root / arrangement / "downloads").string();
    const auto configured =
        bubo::prepare::ConfigureFirstRunModel(prepare, budgets);
    Check(configured == (installed == 96 * GiB),
          arrangement + " prepare uses the hardware first-run choice");
    if (configured) {
      Check(prepare.model_set == bubo::prepare::ModelSet::LowMemory &&
                static_cast<bool>(prepare.publish_default),
            "automatic lower-memory prepare has a publication callback");
      const auto manifest =
          root / arrangement / "downloaded" / low->id / "bubo-package.json";
      std::filesystem::create_directories(manifest.parent_path());
      std::filesystem::copy_file(
          std::filesystem::path(BUBO_MEMORY96_DATA_DIR) /
              "lowmemory-manifest.json",
          manifest, std::filesystem::copy_options::overwrite_existing);
      Check(prepare.publish_default(manifest.string()) == 0,
            "successful first-run manifest becomes the prepared default");
      Check(
          prepare.publish_default(manifest.string()) == 0,
          "default publication is idempotent for the same successful manifest");
      const auto selected =
          ResolvePreparedDefault({}, {}, {}, prepare.local_app_data);
      Check(selected.source == "prepared_default" &&
                selected.manifest == manifest,
            "subsequent plain serve resolves the lower-memory manifest");
      bubo::prepare::Options existing;
      existing.local_app_data = prepare.local_app_data;
      Check(!bubo::prepare::ConfigureFirstRunModel(existing, budgets),
            "an existing prepared selection is authoritative");
    }
    for (const auto step :
         {bubo::prepare::Step::Runtime, bubo::prepare::Step::Model}) {
      bubo::prepare::Options explicit_choice;
      explicit_choice.step = step;
      explicit_choice.local_app_data =
          (root / arrangement / "explicit").string();
      explicit_choice.model_set_chosen = step == bubo::prepare::Step::Model;
      Check(!bubo::prepare::ConfigureFirstRunModel(explicit_choice, budgets),
            "runtime-only or explicit model preparation is unchanged");
    }
    // ModelService sends the selected catalogue id even for the preselected
    // first-run model (models_windows.go jobArgs0). Exercise those real args.
    const auto gui_output = (root / arrangement / "gui-downloads").string();
    const auto gui_local = (root / arrangement / "gui-local").string();
    const char* gui_args[] = {"model", "--catalogue-id", low->id.c_str(),
                              "--output", gui_output.c_str()};
    auto gui_prepare = bubo::prepare::Parse(gui_args, gui_local);
    Check(static_cast<bool>(gui_prepare),
          "GUI first-run catalogue arguments parse");
    const bool gui_default =
        bubo::prepare::ConfigureFirstRunModel(gui_prepare.options, budgets);
    Check(
        gui_default == (installed == 96 * GiB),
        arrangement + " GUI-selected lower-memory model persists on first run");
    if (gui_default) {
      const auto manifest = std::filesystem::path(gui_output) / "models" /
                            low->id / "bubo-package.json";
      std::filesystem::create_directories(manifest.parent_path());
      std::filesystem::copy_file(std::filesystem::path(BUBO_MEMORY96_DATA_DIR) /
                                     "lowmemory-manifest.json",
                                 manifest);
      Check(gui_prepare.options.publish_default(manifest.string()) == 0 &&
                ResolvePreparedDefault({}, {}, {}, gui_local).manifest ==
                    manifest,
            "GUI first-run completion records the real lower-memory manifest");
    }
  }
  arrangement = "96-64";
  for (const auto bytes : {64 * GiB, 80 * GiB - 1, 80 * GiB, 112 * GiB - 1,
                           112 * GiB, 128 * GiB}) {
    installed = bytes;
    Probe probe;
    const auto budgets = ReadPlanningMemoryBudgets(probe);
    ModelsListOptions options{.models_root = root,
                              .available_bytes = budgets.gpu_available,
                              .host_available_bytes = budgets.host_available,
                              .capacity_budgets = budgets};
    const auto list = BuildModelsList(options, catalogue.entries);
    const auto* chosen = DefaultRow(list);
    const bool enabled = bytes >= 80 * GiB && bytes < 112 * GiB;
    Check(chosen &&
              chosen->member_str("id") == (enabled ? low->id : standard->id),
          "installed-memory band boundary " + std::to_string(bytes));
  }
  installed = 96 * GiB;
  for (const auto id : {0x10021586ULL, 0x10021587ULL, 0x10021588ULL,
                        0x10021589ULL, 0x10de1586ULL}) {
    pci = id;
    Probe probe;
    const auto budgets = ReadPlanningMemoryBudgets(probe);
    const auto list =
        BuildModelsList({.models_root = root,
                         .available_bytes = budgets.gpu_available,
                         .host_available_bytes = budgets.host_available,
                         .capacity_budgets = budgets},
                        catalogue.entries);
    const auto* chosen = DefaultRow(list);
    const bool enabled = id >= 0x10021586ULL && id <= 0x10021588ULL;
    Check(chosen &&
              chosen->member_str("id") == (enabled ? low->id : standard->id),
          "PCI identity boundary " + std::to_string(id));
  }
  pci = 0x10021586;
  identity_known = false;
  Probe probe;
  const auto budgets = ReadPlanningMemoryBudgets(probe);
  const auto list =
      BuildModelsList({.models_root = root,
                       .available_bytes = budgets.gpu_available,
                       .host_available_bytes = budgets.host_available,
                       .capacity_budgets = budgets},
                      catalogue.entries);
  Check(DefaultRow(list) && DefaultRow(list)->member_str("id") == standard->id,
        "unknown hardware does not activate 96 GB policy");
  identity_known = true;
  host_available = GiB;
  const auto short_budgets = ReadPlanningMemoryBudgets(probe);
  const auto short_list =
      BuildModelsList({.models_root = root,
                       .available_bytes = short_budgets.gpu_available,
                       .host_available_bytes = short_budgets.host_available,
                       .capacity_budgets = short_budgets},
                      catalogue.entries);
  Check(DefaultRow(short_list) &&
            DefaultRow(short_list)->member_str("id") == low->id,
        "96 hardware choice does not change with free-RAM pressure");
  EstimateRequest shortage;
  shortage.available_bytes = short_budgets.gpu_available;
  shortage.host_available_bytes = short_budgets.host_available;
  shortage.capacity_budgets = short_budgets;
  const auto short_estimate = EstimateForCatalogueEntry(*low, root, shortage);
  Check(!short_estimate.memory_estimate.is_null() &&
            !At(short_estimate.memory_estimate, "fits").as_bool(),
        "hardware choice does not erase a real host-floor refusal");
  Check(StartupMemoryRefusal(short_budgets, short_estimate.memory_estimate,
                             false) ==
            "Not enough memory is available for this setup. Set Variable "
            "Graphics Memory "
            "to the largest value offered (at least 64 GB), close other "
            "memory-heavy apps, and check again.",
        "production pre-load refusal preserves the lower-memory host floor");
  host_available = 11172806656ULL;
  for (const auto reservation : {63999999999ULL, 64000000000ULL}) {
    dedicated_override = reservation;
    const auto warnings =
        EvaluateMachineWarnings(ReadMachineWarningInputs(probe));
    bool found = false;
    for (const auto& warning : warnings) {
      if (warning.check != "gpu_reservation")
        continue;
      found = true;
      Check(warning.remedy ==
                "On a 96 GB computer, use the lower-memory model. "
                "Set Variable Graphics Memory to the largest value offered (72 "
                "GB if "
                "available, at least 64 GB), "
                "then restart Windows.",
            "96 GB advice has approved wording");
    }
    Check(found == (reservation < 64000000000ULL),
          "96 GB warning threshold is decimal 64000000000 bytes");
  }
  dedicated_override.reset();
  budget_known = false;
  const auto unknown_budget = ReadPlanningMemoryBudgets(probe);
  EstimateRequest unavailable;
  unavailable.available_bytes = unknown_budget.gpu_available;
  unavailable.host_available_bytes = unknown_budget.host_available;
  unavailable.capacity_budgets = unknown_budget;
  const auto unconfirmed = EstimateForCatalogueEntry(*low, root, unavailable);
  Check(!unconfirmed.memory_estimate.is_null() &&
            At(unconfirmed.memory_estimate, "fits").is_null() &&
            At(unconfirmed.memory_estimate, "recommended").is_null(),
        "unknown GPU budget never becomes a positive fit or recommendation");
  Check(StartupMemoryRefusal(unknown_budget, unconfirmed.memory_estimate, false)
            .empty(),
        "unknown fit is not an early definite-nonfit refusal");
  budget_known = true;
  dxgi_reads = 0;
  const auto review_budgets = ReadPlanningMemoryBudgets(probe);
  Check(dxgi_reads == 1,
        "planning samples selected DXGI identity and memory once");
  auto disabled_catalogue = catalogue.entries;
  for (auto& entry : disabled_catalogue)
    if (entry.id == low->id) {
      entry.download_enabled = false;
      entry.disabled_reason = "Contract fault: download unavailable.";
      entry.role = "supported";
      entry.default_status = "none";
    }
  const ModelsListOptions review_options{
      .models_root = root,
      .available_bytes = review_budgets.gpu_available,
      .host_available_bytes = review_budgets.host_available,
      .capacity_budgets = review_budgets};
  const auto disabled_list =
      BuildModelsList(review_options, disabled_catalogue);
  for (const auto& row : At(disabled_list, "catalogue").items())
    if (row.member_str("id") == low->id)
      Check(row.member_str("default_status") != "default" &&
                row.member_str("role") != "recommended",
            "disabled lower-memory row is never forced to default or "
            "recommended");
  auto missing_table = *standard;
  missing_table.id += "-missing-table-contract-fault";
  const auto unavailable_list =
      BuildModelsList(review_options, {missing_table});
  const auto& unavailable_row =
      At(unavailable_list, "catalogue").items().front();
  Check(unavailable_row.member_str("disabled_reason") ==
            "Does not fit this computer: needs more GPU memory than is "
            "available. "
            "Use the lower-memory model.",
        "unavailable full-size estimate retains a nonempty plain refusal");
  const auto review_list = BuildModelsList(review_options, catalogue.entries);
  for (const auto& row : At(review_list, "catalogue").items())
    if (row.member_str("id") == standard->id)
      Check(row.member_str("disabled_reason") ==
                "Does not fit this computer: needs about 91 GiB of GPU memory. "
                "Use the lower-memory model.",
            "full-size refusal omits ineffective graphics-memory advice");
  const auto advice = EvaluateMachineWarnings(
      {.gpu_reservation_bytes = 48 * GiB, .is_96gb_strix_halo = true});
  Check(
      !advice.empty() &&
          advice.front().remedy ==
              "On a 96 GB computer, use the lower-memory model. Set Variable "
              "Graphics Memory "
              "to "
              "the largest value offered (72 GB if available, at least 64 GB), "
              "then restart Windows.",
      "96 GB warning uses approved largest-value advice");
  for (bool enabled : {false, true}) {
    gufo::server::ReleaseResolution resolved;
    resolved.settings.push_back({.name = "--context", .value = "200000"});
    bool unsupported = false;
    try {
      gufo::server::ResolveDefaultSessionContext(resolved, 2, Value(), enabled);
    } catch (const ControlError& error) {
      unsupported = error.ExitCode() == 18;
    }
    Check(unsupported,
          "unknown multi-session estimate refuses identically with 96 policy=" +
              std::to_string(enabled));
    bool rejected = false;
    const auto nonfit = EstimateForCatalogueEntry(*low, root, shortage);
    try {
      gufo::server::ResolveDefaultSessionContext(
          resolved, 2, nonfit.memory_estimate, enabled);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Check(rejected,
          "nonfitting multi-session row refuses identically with 96 policy=" +
              std::to_string(enabled));
  }
  gufo::server::ReleaseInputs missing_inputs;
  missing_inputs.profile = "default";
  missing_inputs.architecture = low->architecture;
  missing_inputs.model_family = "qwen3.8-flash-next";
  missing_inputs.native_context = 262144;
  auto missing_resolved = gufo::server::ResolveReleaseSettings(missing_inputs);
  const auto unavailable_policy =
      ResolveServeMemoryPolicy(missing_resolved, "", root / "absent.gguf", 1,
                               review_budgets, false, true);
  Check(unavailable_policy.estimate.is_null() &&
            !unavailable_policy.estimate_reason.empty(),
        "inspect-only unavailable estimate remains null with its reason");
  // Same production policy serve calls, before any model loader is reached.
  for (const auto& name : {"96-64", "96-72", "128-96"}) {
    arrangement = name;
    installed = arrangement == "128-96" ? 128 * GiB : 96 * GiB;
    const auto policy_budgets = ReadPlanningMemoryBudgets(probe);
    for (const auto* entry : {low, standard}) {
      gufo::server::ReleaseInputs inputs;
      inputs.profile = "default";
      inputs.architecture = entry->architecture;
      inputs.model_family = "qwen3.8-flash-next";
      inputs.native_context = 262144;
      inputs.manifest_present = true;
      const auto folder = CatalogueEntryFolder(*entry, root);
      for (const auto& file :
           bubo::prepare::ModelCatalogue(*entry->flash_next_set))
        if (file.kind != bubo::prepare::Kind::MainShard)
          inputs.manifest_defaults[std::string(
              gufo::server::ModelArtefactSetting(file.kind))] =
              (folder / std::filesystem::u8path(file.destination)).string();
      bool refused = false, admitted = false;
      auto resolved = gufo::server::ResolveReleaseSettings(inputs);
      try {
        const auto policy = ResolveServeMemoryPolicy(resolved, entry->id, {}, 1,
                                                     policy_budgets, false);
        admitted = true;
        Check(
            policy.estimate.member_size("context_tokens") ==
                (entry == low && installed == 96 * GiB && arrangement == "96-64"
                     ? 24576
                     : 200000),
            arrangement +
                " serve policy resolves the same default as catalogue");
      } catch (const ControlError& error) {
        const auto& details = error.Details();
        auto artifact = details;
        artifact["exit_code"] = error.ExitCode();
        artifact["message"] = error.what();
        std::ofstream(root / (arrangement + "-memory-refusal.json"))
            << artifact.dump() << '\n';
        refused = error.ExitCode() == 19 &&
                  details.member_size("needed_bytes") >
                      details.member_size("available_bytes") &&
                  details.member_size("sessions") == 1 &&
                  details.member_size("context") == 200000;
      }
      Check(refused == (installed == 96 * GiB && entry == standard) &&
                admitted != refused,
            arrangement +
                " shared serve policy stops definite nonfit with typed memory "
                "details");
    }
  }
  arrangement = "96-64";
  installed = 96 * GiB;
  bubo::prepare::Options completed;
  completed.model_set = bubo::prepare::ModelSet::LowMemory;
  completed.catalogue_id = low->id;
  completed.root = (root / "completion").string();
  std::ostringstream message;
  std::string remembered;
  const auto completion = bubo::prepare::CompleteModelPreparation(
      completed, true, message, [&](const std::string& manifest) {
        remembered = manifest;
        return 0;
      });
  Check(completion == 0 && !remembered.empty() && message.str().empty(),
        "explicit first-run lower-memory completion remembers without claiming "
        "unchanged default");
  bubo::prepare::CompleteModelPreparation(completed, false, message,
                                          [](const std::string&) { return 0; });
  Check(message.str().find("saved default model is unchanged") !=
            std::string::npos,
        "outside first run an explicit alternative still explains unchanged "
        "default");
}
}  // namespace

namespace bubo::cli {
std::unique_ptr<MachineProbe> CreateSystemMachineProbe() {
  return std::make_unique<Probe>();
}
}  // namespace bubo::cli

int main(int argc, char** argv) {
  if (argc < 3)
    return 2;
  arrangement = argv[1];
  if (arrangement != "96-64" && arrangement != "96-72" &&
      arrangement != "128-96")
    return 2;
  installed = arrangement == "128-96" ? 128 * GiB : 96 * GiB;
  const std::string command = argv[2];
  if (command == "self-test") {
    if (argc != 4)
      return 2;
    const auto root =
        std::filesystem::u8path(argv[3]) /
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    Cases(root);
    std::cout << "Retained pointer/manifest artifacts: " << root.string()
              << '\n';
    std::cout << "Derived memory arrangements: " << passed << " passed, "
              << failed << " failed\n";
    return failed ? 1 : 0;
  }
  std::vector<const char*> args;
  for (int i = 3; i < argc; ++i)
    args.push_back(argv[i]);
  const auto local = EnvironmentUtf8("LOCALAPPDATA");
  if (command == "models")
    return RunModels(args, local, std::cout, std::cerr);
  if (command == "prepare")
    return bubo::prepare::Run(args, local, std::cout, std::cerr);
  if (command == "status")
    return RunStatus(
        args, local,
        std::filesystem::u8path(EnvironmentUtf8("BUBO_RUNTIME_DIR")), std::cout,
        std::cerr);
  return 2;
}
