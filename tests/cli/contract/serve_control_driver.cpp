// Isolated CPU process driver: actual lifecycle, local control and HTTP
// transport, with a backend that never loads model weights or a GPU runtime.
#ifndef BUBO_SERVE_TESTING
#error "This driver must not be compiled into a production package"
#endif

#include <arpa/inet.h>
#include <sys/socket.h>

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>

#include "src/cli/instance_control.hpp"
#include "src/cli/machine_adapter.hpp"
#include "src/cli/serve/asr_service.hpp"
#include "src/cli/serve/audio_asr_api.hpp"
#include "src/cli/serve/audio_tts_api.hpp"
#include "src/cli/serve/audio_websocket.hpp"
#include "src/cli/serve/http_server.hpp"
#include "src/cli/serve/image_api.hpp"
#include "src/cli/serve/lifecycle.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/npu_status_publisher.hpp"
#include "src/cli/serve/release_profile.hpp"
#include "src/cli/serve/serve_command.hpp"
#include "src/cli/serve/status_metrics.hpp"
#include "src/cli/serve/tts_service.hpp"
#include "src/cli/serve/video_api.hpp"
#include "src/cli/settings_support.hpp"
#include "src/cli/status.hpp"
#include "src/cli/stop.hpp"
#include "src/core/image.hpp"
#include "src/core/platform/socket.hpp"
#include "tests/cli/contract/fake_serve_backend.hpp"

// HttpServer is built from its production source in this CPU target. It owns
// no non-text service here; these unreachable adapters avoid pulling model
// runtimes into the test executable. Requests to them fail explicitly.
namespace gufo::server {
namespace {
HttpResponse UnavailableModality() {
  HttpResponse response;
  response.status = 503;
  response.reason = "Service Unavailable";
  response.body = "{\"error\":{\"code\":\"unsupported_test_modality\"}}";
  return response;
}
}  // namespace
bool VideoJobService::ready() const noexcept {
  return false;
}
bool TtsService::ready() const noexcept {
  return false;
}
std::string TtsService::model_id() const {
  return "unsupported_test_modality";
}
bool AsrService::ready() const noexcept {
  return false;
}
std::string AsrService::model_id() const {
  return "unsupported_test_modality";
}
bool IsVideoApiPath(std::string_view path) noexcept {
  return path.starts_with("/v1/videos");
}
bool IsAudioTtsApiPath(std::string_view path) noexcept {
  return path == "/v1/audio/speech";
}
bool IsAudioAsrApiPath(std::string_view path) noexcept {
  return path == "/v1/audio/transcriptions";
}
HttpResponse HandleVideoApiRequest(const HttpRequest&, VideoJobService&) {
  return UnavailableModality();
}
HttpResponse HandleAudioTtsApiRequest(const HttpRequest&, TtsService&) {
  return UnavailableModality();
}
HttpResponse HandleAudioAsrApiRequest(const HttpRequest&, AsrService&) {
  return UnavailableModality();
}
HttpResponse HandleImageApiRequest(const HttpRequest&, ImageService&) {
  return UnavailableModality();
}
HttpResponse HandleTtsWebSocket(const HttpRequest&, TtsService&) {
  return UnavailableModality();
}
HttpResponse HandleAsrWebSocket(const HttpRequest&, AsrService&) {
  return UnavailableModality();
}
}  // namespace gufo::server

namespace gufo::core {
std::vector<std::uint8_t> ReadImageUrl(std::string_view, ImageReadBudget&) {
  throw std::invalid_argument("The isolated text fake does not fetch images.");
}
}  // namespace gufo::core

namespace {
using bubo::cli::Value;
using namespace std::chrono_literals;

unsigned ParseUnsigned(std::string_view text) {
  unsigned value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size())
    throw std::invalid_argument("Expected an unsigned integer.");
  return value;
}

bool ReadinessCheck(int port) {
  const int fd = static_cast<int>(::socket(AF_INET, SOCK_STREAM, 0));
  if (fd < 0)
    return false;
  gufo::platform::SetSocketTimeout(fd, SO_RCVTIMEO, 2s);
  gufo::platform::SetSocketTimeout(fd, SO_SNDTIMEO, 2s);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<unsigned short>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
  bool ready = false;
  if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
      0) {
    constexpr std::string_view request =
        "GET /ready HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if (gufo::platform::SocketSend(fd, request.data(), request.size(), 0) > 0) {
      std::string response;
      while (response.size() < 4096 &&
             response.find("\r\n") == std::string::npos) {
        char buffer[1024];
        const auto length =
            gufo::platform::SocketRead(fd, buffer, sizeof(buffer));
        if (length <= 0)
          break;
        response.append(buffer, static_cast<std::size_t>(length));
      }
      ready = response.starts_with("HTTP/1.1 200 ");
    }
  }
  gufo::platform::CloseSocket(fd);
  return ready;
}

Value EffectiveSettings() {
  auto settings = Value::object();
  for (const auto* name : {"profile", "context", "sessions"}) {
    auto setting = Value::object();
    const Value effective =
        std::string_view(name) == "profile"
            ? Value("test-only")
            : Value(name == std::string_view("context") ? 4096 : 1);
    setting["requested"] = effective;
    setting["effective"] = effective;
    setting["source"] = "test_fixture";
    setting["reason"] = "Isolated CPU fake backend.";
    settings[name] = setting;
  }
  return settings;
}

int FakeImpl(std::span<const char* const> args,
             bubo::cli::ServeLifecycle* lifecycle_pointer,
             const std::optional<std::filesystem::path>& log_file) {
  if (!lifecycle_pointer)
    throw std::invalid_argument("The CPU fake requires local control.");
  auto& lifecycle = *lifecycle_pointer;
  // Clear callbacks before local backend/server resources are destroyed.
  // RunServeCommand publishes Stopped only after this runner returns.
  struct ClearHooks {
    bubo::cli::ServeLifecycle& lifecycle;
    ~ClearHooks() {
      lifecycle.SetStop({});
      lifecycle.SetSampler({});
    }
  } hooks{lifecycle};
  unsigned port = 0, load_delay = 0, stop_delay = 0, burst = 0, log_burst = 0;
  bool fake = false, hang_stop = false, imported = false, authenticated = false;
  bool npu_status_enabled = false;
  std::filesystem::path generation_gate;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view option = args[i];
    const auto value = [&]() -> std::string_view {
      if (++i >= args.size())
        throw std::invalid_argument("Missing option value.");
      return args[i];
    };
    if (option == "--fake-backend") {
      fake = true;
    } else if (option == "--fake-npu-status") {
      npu_status_enabled = true;
    } else if (option == "--fake-generation-gate") {
      generation_gate = std::filesystem::u8path(value());
    } else if (option == "--fake-rejected-bias") {
      gufo::server::ReleaseInputs inputs;
      inputs.explicit_flags["--think-marker-bias"] = std::string(value());
      gufo::server::ResolveReleaseSettings(inputs);
    } else if (option == "--fake-imported") {
      imported = true;
    } else if (option == "--fake-authenticated") {
      authenticated = true;
    } else if (option == "--fake-hang-stop") {
      hang_stop = true;
    } else if (option == "--port") {
      port = ParseUnsigned(value());
    } else if (option == "--host") {
      if (value() != "127.0.0.1")
        throw std::invalid_argument("The fake accepts loopback only.");
    } else if (option == "--fake-load-delay-ms") {
      load_delay = ParseUnsigned(value());
    } else if (option == "--fake-stop-delay-ms") {
      stop_delay = ParseUnsigned(value());
    } else if (option == "--fake-event-burst") {
      burst = ParseUnsigned(value());
    } else if (option == "--fake-log-burst") {
      log_burst = ParseUnsigned(value());
    } else if (option == "--log-level") {
      const auto level = gufo::server::LogLevelFromName(value());
      if (!level)
        throw std::invalid_argument("Invalid log level.");
      gufo::server::Logger::SetLevel(*level);
    } else {
      throw std::invalid_argument("Unknown fake serve option.");
    }
  }
  if (!fake || port == 0 || port > 65535 || (port >= 8081 && port <= 8086))
    throw std::invalid_argument("Select a test-only ephemeral loopback port.");
  const std::string endpoint =
      "http://127.0.0.1:" + std::to_string(port) + "/v1";
  auto backend =
      std::make_shared<bubo::cli::testing::FakeServeBackend>(generation_gate);
  lifecycle.Configure(endpoint, backend->model_id(), EffectiveSettings(),
                      log_file);
  lifecycle.SetSampler(
      [backend] { return gufo::server::RuntimeStatusMetrics(*backend); });
  bubo::cli::EndpointLease lease("127.0.0.1", static_cast<int>(port));
  lease.Acquire();
  bubo::cli::PortReservation reservation("127.0.0.1", static_cast<int>(port));
  reservation.Reserve();
  std::unique_ptr<bubo::cli::NpuStatusPublisher> npu_status;
  if (npu_status_enabled) {
    gufo::npu::NoteNpuStartup(false, "CPU fake startup denial");
    npu_status = std::make_unique<bubo::cli::NpuStatusPublisher>(lifecycle);
  }
  // Keep the real producer callback firing during the blocked-backend proof.
  std::jthread npu_updates([&](std::stop_token stop) {
    while (npu_status_enabled && !stop.stop_requested()) {
      gufo::npu::PublishNpuRuntimeStatus();
      std::this_thread::sleep_for(5ms);
    }
  });
  lifecycle.OpenControl();
  lifecycle.Transition("loading");
  const auto load_until =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(load_delay);
  while (std::chrono::steady_clock::now() < load_until &&
         !lifecycle.StopRequested())
    std::this_thread::sleep_for(5ms);
  if (lifecycle.StopRequested()) {
    return 0;
  }
  backend->Load();
  auto server = std::make_shared<gufo::server::HttpServer>(
      "127.0.0.1", static_cast<int>(port), backend, nullptr, nullptr, nullptr,
      gufo::server::HttpServerOptions{
          .on_request_completed = [&lifecycle, npu_status_enabled](Value row) {
            if (npu_status_enabled) {
              gufo::npu::NoteLastRequest({12.5, 0.25, 1024, 75});
              gufo::npu::PublishNpuRuntimeStatus();
            }
            lifecycle.Transition("request_completed", std::move(row));
          }});
  lifecycle.SetSampler([backend, server] {
    auto sample = gufo::server::RuntimeStatusMetrics(*backend);
    sample["requests"] = server->recent_requests();
    return sample;
  });
  lifecycle.SetStop([server] { server->request_stop(); });
  reservation.Release();
  std::string error;
  if (!server->start(&error))
    throw bubo::cli::ControlError(13, error);
  lifecycle.Configure(server->bound_endpoint(), backend->model_id(),
                      EffectiveSettings(), log_file, true);
  auto connection = Value::object();
  connection["endpoint"] = server->bound_endpoint();
  connection["model_id"] = backend->model_id();
  connection["context_tokens"] =
      static_cast<unsigned long long>(backend->max_context());
  connection["authenticated"] = authenticated;
  auto details = Value::object();
  details["connection"] = std::move(connection);
  details["active_model"] = bubo::cli::ActiveModelIdentity(
      std::filesystem::current_path() / "fixture-model.gguf",
      imported ? std::filesystem::path{}
               : std::filesystem::current_path() / "bubo-package.json",
      imported ? std::string_view{} : bubo::prepare::kPackageId);
  details["client_snippets"] =
      bubo::cli::ClientSnippets(backend->model_id(), authenticated);
  details["pi_snippet"] =
      bubo::cli::PiSnippet(server->bound_endpoint(), backend->model_id(),
                           backend->max_context(), authenticated);
  // Inject budgets only in the CPU fixture; production samples DXGI/host APIs.
  bubo::cli::MemoryEstimateInputs estimate;
  estimate.package_id = std::string(bubo::prepare::kPackageId);
  estimate.profile = "default";
  estimate.context = backend->max_context();
  estimate.sessions = 1;
  estimate.mtp = estimate.twin = estimate.draft_head = estimate.vision = true;
  // Owner's allocatable local DXGI budget, rather than the old dedicated cap.
  estimate.available_bytes = bubo::cli::MiBToBytes(110192);
  estimate.host_available_bytes = 11900000000ULL;
  estimate.npu_prefill = true;
  std::string estimate_reason;
  details["memory_estimate"] =
      bubo::cli::MemoryEstimate(estimate, &estimate_reason);
  details["memory_estimate_reason"] =
      estimate_reason.empty() ? Value() : Value(estimate_reason);
  lifecycle.SetDetails(std::move(details));
  std::thread listener([server] { server->run(false); });
  struct JoinListener {
    std::shared_ptr<gufo::server::HttpServer>& server;
    std::thread& listener;
    ~JoinListener() {
      if (listener.joinable()) {
        server->request_stop();
        listener.join();
      }
    }
  } listener_guard{server, listener};
  if (!ReadinessCheck(server->port())) {
    server->request_stop();
    listener.join();
    lifecycle.SetStop({});
    throw bubo::cli::ControlError(14, "Readiness check failed.");
  }
  lifecycle.Transition("ready");
  for (unsigned i = 0; i < log_burst; ++i)
    gufo::server::Logger::Error("fake-test-only", std::string(512, 'x'));
  if (log_burst)
    gufo::server::Logger::FlushSync(2s);
  for (unsigned i = 0; i < burst; ++i) {
    auto warning = Value::object();
    warning["code"] = "fake_output_pressure";
    warning["message"] = std::string(4096, 'x');
    lifecycle.Transition("warning", std::move(warning));
  }
  listener.join();
  lifecycle.SetStop({});
  // Simulate a backend cleanup hang after the production HTTP shutdown has
  // rejected admission and joined requests. Only the test's exact child can
  // be terminated by its parent harness; the Stop command never forces it.
  while (hang_stop)
    std::this_thread::sleep_for(100ms);
  std::this_thread::sleep_for(std::chrono::milliseconds(stop_delay));
  lifecycle.SetSampler({});
  server.reset();
  backend.reset();
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2)
    return 2;
  const std::span<const char* const> args(argv + 2,
                                          static_cast<std::size_t>(argc - 2));
  const std::string_view command = argv[1];
  const char* local = std::getenv("LOCALAPPDATA");
  if (command == "control-timings" && args.size() == 1) {
    auto times = Value::object();
    auto start = std::chrono::steady_clock::now();
    const auto mark = [&](const char* name) {
      const auto now = std::chrono::steady_clock::now();
      times[name] =
          std::chrono::duration<double, std::milli>(now - start).count();
      start = now;
    };
    const auto record = bubo::cli::ReadDiscovery(args[0], local ? local : "");
    if (!record)
      return 16;
    mark("discovery_ms");
    auto instance = bubo::cli::VerifiedConnect(*record);
    mark("identity_handshake_ms");
    const auto snapshot = instance.Snapshot();
    mark("pipe_snapshot_ms");
    const auto wire = snapshot.dump();
    mark("json_dump_ms");
    times["snapshot_bytes"] = wire.size();
    std::cout << times.dump() << '\n';
    return 0;
  }
  if (command == "probe-timings") {
    auto probe = bubo::cli::CreateSystemMachineProbe();
    auto times = Value::object();
    const auto measure = [&](const char* name, auto call) {
      const auto start = std::chrono::steady_clock::now();
      call();
      times[name] = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    };
    measure("overlay_ms", [&] { probe->OverlayScheme(); });
    measure("power_mode_ms", [&] { probe->EffectivePowerMode(); });
    measure("power_source_ms", [&] { probe->PowerSource(); });
    measure("gpu_adlx_ms", [&] { probe->Gpu(); });
    measure("vgm_ms", [&] { probe->Vgm(); });
    measure("dxgi_ms", [&] { probe->Dxgi(); });
    measure("host_ms", [&] { probe->HostPhysicalMemory(); });
    measure("oem_ms", [&] {
      probe->OemManufacturer();
      probe->OemModel();
      probe->BiosVendor();
      probe->BiosVersion();
      probe->BiosDate();
    });
    measure("bootstrap_total_ms",
            [&] { bubo::cli::StatusSnapshot(local ? local : "", {}); });
    std::cout << times.dump() << '\n';
    return 0;
  }
  if (command == "serve")
    return gufo::cli::RunServeCommand(args, FakeImpl);
  if (command == "status")
    return bubo::cli::RunStatus(args, local ? local : "", {}, std::cout,
                                std::cerr);
  if (command == "stop")
    return bubo::cli::RunStop(args, local ? local : "", std::cout, std::cerr);
  return 2;
}
