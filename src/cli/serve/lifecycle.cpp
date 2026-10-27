#include "src/cli/serve/lifecycle.hpp"

#include <iostream>
#include <random>

#include "src/cli/machine_adapter.hpp"
#ifdef _WIN32
#include <unistd.h>
#endif

namespace bubo::cli {
namespace {
std::string RandomInstanceId() {
  std::random_device random;
  constexpr char hex[] = "0123456789abcdef";
  std::string id;
  for (unsigned i = 0; i != 16; ++i) {
    const auto byte = random() & 255;
    id += hex[byte >> 4];
    id += hex[byte & 15];
  }
  return id;
}
std::string PathText(const std::filesystem::path& path) {
  const auto text = path.u8string();
  return {text.begin(), text.end()};
}
}  // namespace

ServeLifecycle::ServeLifecycle(bool machine)
    : identity_(CurrentIdentity(RandomInstanceId())),
      snapshot_(Value::object()) {
  if (machine)
    writer_ = std::make_unique<MachineWriter>(std::cout);
  snapshot_["identity"] = IdentityJson(identity_);
  snapshot_["prepared"] = false;
  snapshot_["phase"] = "starting";
  snapshot_["versions"] = Product();
  snapshot_["endpoint"] = Value();
  snapshot_["endpoints"] = Value();
  snapshot_["api_key_env"] = "BUBO_API_KEY";
  snapshot_["served_model"] = Value();
  snapshot_["active_model"] = Value();
  snapshot_["client_snippets"] = Value();
  snapshot_["effective_settings"] = Value();
  for (const auto* field :
       {"memory_estimate", "memory_estimate_reason", "inference_defaults",
        "pi_snippet", "effective_switches", "prepared_models", "requests"})
    snapshot_[field] = Value();
  snapshot_["capabilities"] = Capabilities();
  snapshot_["connection"] = Value();
  snapshot_["memory_estimate_reason"] =
      "No coefficient table for this service.";
  snapshot_["log_path"] = Value();
  snapshot_["readiness"] = Value::object();
  snapshot_["readiness"]["runtime"] = false;
  snapshot_["readiness"]["model"] = false;
  snapshot_["readiness"]["server"] = false;
  snapshot_["last_error"] = Value();
  snapshot_["last_transition"] = UtcTime();
  snapshot_["sample_time"] = snapshot_["last_transition"];
  snapshot_["state_history"] = Value::array();
  for (const auto* field : {"counters", "last_completed_request", "sessions",
                            "queue", "slots", "memory"})
    snapshot_[field] = Value();
  if (writer_)
    output_thread_ = std::thread([this] { Drain(); });
  Transition("starting");
}

ServeLifecycle::~ServeLifecycle() {
  samples_closing_ = true;
  sample_wake_.notify_all();
  machine_wake_.notify_all();
  if (sample_thread_.joinable())
    sample_thread_.join();
  if (machine_thread_.joinable())
    machine_thread_.join();
  if (control_)
    control_->Shutdown();
  {
    std::lock_guard lock(mutex_);
    closing_ = true;
  }
  wake_.notify_one();
  if (output_thread_.joinable())
    output_thread_.join();
}

void ServeLifecycle::Configure(std::string endpoint, std::string served_model,
                               Value settings,
                               std::optional<std::filesystem::path> log_path,
                               bool listener_bound) {
  std::lock_guard lock(mutex_);
  // Only advertise SDK URLs from HttpServer::bound_endpoint() after bind.
  // OpenAI appends /chat/completions; Anthropic appends /v1/messages.
  snapshot_["endpoints"] = Value();
  if (listener_bound) {
    auto endpoints = Value::object();
    endpoints["openai_base_url"] = endpoint;
    endpoints["anthropic_base_url"] =
        endpoint.ends_with("/v1") ? endpoint.substr(0, endpoint.size() - 3)
                                  : endpoint;
    snapshot_["endpoints"] = std::move(endpoints);
  }
  snapshot_["endpoint"] = std::move(endpoint);
  snapshot_["served_model"] = std::move(served_model);
  snapshot_["effective_settings"] = std::move(settings);
  snapshot_["readiness"]["runtime"] = true;
  snapshot_["log_path"] = log_path ? Value(PathText(*log_path)) : Value();
  if (published_)
    PublishLocked();
}

void ServeLifecycle::PublishLocked(bool tombstone) {
  DiscoveryRecord record;
  record.identity = identity_;
  record.endpoint = snapshot_.member_str("endpoint");
  record.phase = snapshot_.member_str("phase");
  const auto path = snapshot_.member_str("log_path");
  if (!path.empty())
    record.log_path = path;
  record.last_error = snapshot_["last_error"];
  if (tombstone)
    WriteTombstone(record, record.last_error, {});
  else
    WriteDiscovery(record, {});
}

void ServeLifecycle::Transition(std::string type, Value payload) {
  std::lock_guard lock(mutex_);
  if (type != "warning" && type != "request_completed") {
    snapshot_["phase"] = type;
    snapshot_["last_transition"] = UtcTime();
    auto history = Value::array();
    const auto& previous = snapshot_["state_history"].items();
    const auto start = previous.size() >= 32 ? previous.size() - 31 : 0;
    for (std::size_t i = start; i < previous.size(); ++i)
      history.push_back(previous[i]);
    auto transition = Value::object();
    transition["type"] = type;
    transition["phase"] = type;
    transition["time"] = snapshot_["last_transition"];
    history.push_back(std::move(transition));
    snapshot_["state_history"] = std::move(history);
  }
  if (type == "ready") {
    snapshot_["prepared"] = true;
    snapshot_["readiness"]["runtime"] = true;
    snapshot_["readiness"]["model"] = true;
    snapshot_["readiness"]["server"] = true;
    payload = snapshot_;
  } else if (type == "stopping" || type == "stopped" || type == "error") {
    snapshot_["readiness"]["server"] = false;
    if (type == "stopped" || type == "error")
      snapshot_["readiness"]["model"] = false;
  }
  if (type == "error")
    snapshot_["last_error"] = payload;
  payload["identity"] = IdentityJson(identity_);
  // Reconcile all transitions through control and the atomic file even if
  // stdout failed. The queue has a byte cap as well as a record cap.
  try {
    if (type != "request_completed" && (published_ || type == "stopped"))
      PublishLocked(type == "stopped");
  } catch (const std::exception&) {
    // A disk failure must not prevent authenticated Stop from reaching the
    // shutdown callback, or prevent remaining machine records from draining.
    if (type != "error")
      snapshot_["last_error"] = Error("storage_failed", "control", "discovery",
                                      false, "Check the local data folder.");
  }
  if (!writer_ || output_failed_)
    return;
  const auto bytes = payload.dump().size();
  if (bytes >= 1024 * 1024 || events_.size() >= 64 ||
      queued_bytes_ + bytes > 2 * 1024 * 1024) {
    output_failed_ = true;
    events_.clear();
    queued_bytes_ = 0;
    return;
  }
  queued_bytes_ += bytes;
  events_.push_back({std::move(type), std::move(payload)});
  wake_.notify_one();
}

void ServeLifecycle::Drain() {
  for (;;) {
    Event event;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [this] { return closing_ || !events_.empty(); });
      if (events_.empty())
        return;
      event = std::move(events_.front());
      queued_bytes_ -= event.payload.dump().size();
      events_.pop_front();
    }
    bool written = false;
    try {
      auto envelope = IdentityJson(identity_);
      written = writer_->Write(event.type, std::move(event.payload), &envelope);
    } catch (...) {
    }
    if (!written) {
      std::lock_guard lock(mutex_);
      output_failed_ = true;
      events_.clear();
      queued_bytes_ = 0;
      return;
    }
  }
}

Value ServeLifecycle::Snapshot() const {
  // Control only copies published data. No sampler, scheduler, device probe
  // or sampling lock belongs on the pipe handler's response path.
  std::lock_guard lock(mutex_);
  auto value = snapshot_;
  throughput_.Apply(value, sampled_at_);
  value["sample_age_ms"] =
      sampled_at_ == std::chrono::steady_clock::time_point{}
          ? Value()
          : Value(std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - sampled_at_)
                      .count());
  return value;
}

void ServeLifecycle::SampleLocked() {
  // sample_mutex_ serializes callbacks and teardown, never control reads.
  std::function<Value()> sampler;
  {
    std::lock_guard lock(mutex_);
    sampler = sampler_;
  }
  // Stamp the beginning of observation: a blocked backend must not make old
  // counters look fresh when its session-state call finally returns.
  const auto sampled_at = std::chrono::steady_clock::now();
  const auto time = UtcTime();
  auto metrics = sampler ? sampler() : Value::object();
#ifdef _WIN32
  if (!metrics.find("memory") || metrics.find("memory")->is_null()) {
    const auto bytes = [](std::uint64_t amount) {
      return amount > 0 && amount <= 9007199254740991ULL ? Value(amount)
                                                         : Value();
    };
    auto memory = Value::object();
    memory["rss_bytes"] = bytes(gufo_resident_bytes());
    memory["host_available_bytes"] = bytes(gufo_available_physical_bytes());
    memory["unit"] = "bytes";
    memory["evidence"] = memory["rss_bytes"].is_null() &&
                                 memory["host_available_bytes"].is_null()
                             ? "unknown"
                             : "measured";
    metrics["memory"] = std::move(memory);
  }
#endif
  std::lock_guard lock(mutex_);
  for (const auto& [key, item] : metrics.members())
    snapshot_[key] = item;
  snapshot_["sample_time"] = time;
  sampled_at_ = sampled_at;
}

void ServeLifecycle::SampleLoop() {
  std::unique_lock sample_lock(sample_mutex_);
  while (!samples_closing_) {
    SampleLocked();
    sample_wake_.wait_for(sample_lock, std::chrono::milliseconds(100),
                          [this] { return samples_closing_.load(); });
  }
}

void ServeLifecycle::MachineLoop() {
  while (!samples_closing_) {
    // Fresh live readings on every pass, with their own observation times.
    // Expensive ADLX initialization belongs to the publisher, not each poll.
    auto probe = CreateSystemMachineProbe();
    auto machine = MachineReport(*probe);
    {
      std::lock_guard lock(mutex_);
      snapshot_["machine"] = std::move(machine);
    }
    std::unique_lock sample_lock(machine_mutex_);
    machine_wake_.wait_for(sample_lock, std::chrono::seconds(1),
                           [this] { return samples_closing_.load(); });
  }
}

void ServeLifecycle::SetSampler(std::function<Value()> sampler) {
  std::lock_guard sample_lock(sample_mutex_);
  {
    std::lock_guard lock(mutex_);
    sampler_ = std::move(sampler);
    throughput_ = ThroughputBaseline{};
  }
  // Establish the first real sample during startup, before accepting work.
  SampleLocked();
}

void ServeLifecycle::SetDetails(Value details) {
  std::lock_guard lock(mutex_);
  for (const auto& [key, value] : details.members())
    snapshot_[key] = value;
}
void ServeLifecycle::SetStop(std::function<void()> stop) {
  std::lock_guard lock(mutex_);
  stop_ = std::move(stop);
  if (stop_requested_ && stop_)
    stop_();
}
void ServeLifecycle::RequestStop() {
  if (stop_requested_.exchange(true))
    return;
  Transition("stopping");
  std::lock_guard lock(mutex_);
  if (stop_)
    stop_();
}
void ServeLifecycle::OpenControl() {
  // Startup can probe once. Live status subsequently reads the publications.
  auto probe = CreateSystemMachineProbe();
  {
    auto machine = MachineReport(*probe);
    std::lock_guard lock(mutex_);
    snapshot_["machine"] = std::move(machine);
  }
  sample_thread_ = std::thread([this] { SampleLoop(); });
  machine_thread_ = std::thread([this] { MachineLoop(); });
  control_ = std::make_unique<ControlServer>(
      identity_, [this] { return Snapshot(); }, [this] { RequestStop(); });
  control_->Start();
  std::lock_guard lock(mutex_);
  PublishLocked();
  published_ = true;
}
void ServeLifecycle::Finish() {
  if (finished_)
    return;
  if (!StopRequested())
    Transition("stopping");
  Transition("stopped");
  finished_ = true;
}
void ServeLifecycle::RecordAbruptExit() {
  std::lock_guard lock(mutex_);
  try {
    PublishLocked(true);
  } catch (const std::exception&) {
  }
}
}  // namespace bubo::cli
