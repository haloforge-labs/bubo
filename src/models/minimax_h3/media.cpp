#include "src/models/minimax_h3/media.hpp"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef GUFO_FFMPEG_EXECUTABLE
#define GUFO_FFMPEG_EXECUTABLE "ffmpeg"
#endif

#ifndef GUFO_FFPROBE_EXECUTABLE
#define GUFO_FFPROBE_EXECUTABLE "ffprobe"
#endif

// POSIX requires the process environment through a mutable-pointer ABI even
// though posix_spawnp does not mutate it.
extern char**
    environ;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

namespace gufo::minimax_h3 {
namespace {

constexpr std::size_t kIoChunkBytes = 4U << 20U;
constexpr int kAudioChunkSamples = 4096;

bool EnsureSigpipeIgnored(std::string* error) {
  static std::once_flag once;
  static int failure = 0;
  std::call_once(once, [] {
    struct sigaction ignore{};
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    if (sigaction(SIGPIPE, &ignore, nullptr) != 0) {
      failure = errno;
    }
  });
  if (failure == 0) {
    return true;
  }
  if (error != nullptr) {
    *error = "cannot ignore SIGPIPE for FFmpeg writers: " +
             std::string(std::strerror(failure));
  }
  return false;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

const char* FfmpegProgram() {
  const char* override = std::getenv("GUFO_FFMPEG");
  return override != nullptr && *override != '\0' ? override
                                                  : GUFO_FFMPEG_EXECUTABLE;
}

const char* FfprobeProgram() {
  const char* override = std::getenv("GUFO_FFPROBE");
  return override != nullptr && *override != '\0' ? override
                                                  : GUFO_FFPROBE_EXECUTABLE;
}

std::filesystem::path PartialMediaPath(
    const std::filesystem::path& destination) {
  static std::atomic<std::uint64_t> sequence{0};
  const std::string name =
      destination.filename().string() + ".gufo-partial-" +
      std::to_string(static_cast<std::uint64_t>(getpid())) + "-" +
      std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + ".mp4";
  return destination.parent_path() / name;
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

void CloseFd(int* descriptor) {
  if (*descriptor >= 0) {
    while (close(*descriptor) != 0 && errno == EINTR) {
    }
    *descriptor = -1;
  }
}

bool MakeNonBlocking(int descriptor, std::string* error) {
  const int flags = fcntl(descriptor, F_GETFL);
  if (flags < 0 || fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) != 0) {
    SetError(error, "cannot configure FFmpeg media pipe: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  return true;
}

ssize_t CancellableWrite(int descriptor, const void* data, std::size_t bytes,
                         const CancellationToken* cancellation,
                         const std::atomic<bool>& stop) {
  while (!stop.load(std::memory_order_acquire) && !IsCancelled(cancellation)) {
    const ssize_t amount = write(descriptor, data, bytes);
    if (amount >= 0) {
      return amount;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      return -1;
    }
    struct pollfd readiness{static_cast<decltype(pollfd::fd)>(descriptor),
                            POLLOUT, 0};
    const int result = poll(&readiness, 1, 50);
    if (result < 0 && errno != EINTR) {
      return -1;
    }
    if (result > 0 && (readiness.revents & (POLLERR | POLLHUP | POLLNVAL))) {
      errno = EPIPE;
      return -1;
    }
  }
  errno = ECANCELED;
  return -1;
}

int AddCloseUnlessKept(posix_spawn_file_actions_t* actions, int descriptor,
                       int keep_a, int keep_b) {
  if (descriptor == keep_a || descriptor == keep_b) {
    return 0;
  }
  return posix_spawn_file_actions_addclose(actions, descriptor);
}

bool WaitForChild(pid_t child, int* status, std::string* error) {
  while (waitpid(child, status, 0) < 0) {
    if (errno == EINTR) {
      continue;
    }
    SetError(error, "cannot wait for media process: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  return true;
}

bool ParseInteger(std::string_view text, int* output) {
  if (text.empty()) {
    return false;
  }
  int value = 0;
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size()) {
    return false;
  }
  *output = value;
  return true;
}

bool ParseDouble(std::string_view text, double* output) {
  if (text.empty() || text == "N/A") {
    *output = 0.0;
    return true;
  }
  double value = 0.0;
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size() ||
      !std::isfinite(value)) {
    return false;
  }
  *output = value;
  return true;
}

bool ParseRate(std::string_view text, double* output) {
  const std::size_t slash = text.find('/');
  if (slash == std::string_view::npos) {
    return ParseDouble(text, output);
  }
  double numerator = 0.0;
  double denominator = 0.0;
  if (!ParseDouble(text.substr(0, slash), &numerator) ||
      !ParseDouble(text.substr(slash + 1), &denominator) ||
      denominator == 0.0) {
    return false;
  }
  *output = numerator / denominator;
  return std::isfinite(*output);
}

std::vector<char*> MutableArgv(std::span<std::string> storage) {
  std::vector<char*> arguments;
  arguments.reserve(storage.size() + 1);
  for (std::string& argument : storage) {
    arguments.push_back(argument.data());
  }
  arguments.push_back(nullptr);
  return arguments;
}

bool RunCapture(const char* program, std::vector<std::string> storage,
                std::string* output, std::string* error) {
  std::vector<char*> arguments = MutableArgv(storage);
  int stream[2] = {-1, -1};
  if (pipe(stream) != 0) {
    SetError(error, "cannot create media inspection pipe: " +
                        std::string(std::strerror(errno)));
    return false;
  }
  posix_spawn_file_actions_t actions;
  int code = posix_spawn_file_actions_init(&actions);
  if (code == 0) {
    code = posix_spawn_file_actions_adddup2(&actions, stream[1], STDOUT_FILENO);
  }
  if (code == 0) {
    code = posix_spawn_file_actions_addclose(&actions, stream[0]);
  }
  if (code == 0) {
    code = posix_spawn_file_actions_addclose(&actions, stream[1]);
  }
  pid_t child = -1;
  if (code == 0) {
    code = posix_spawnp(&child, program, &actions, nullptr, arguments.data(),
                        environ);
  }
  (void)posix_spawn_file_actions_destroy(&actions);
  CloseFd(&stream[1]);
  if (code != 0) {
    CloseFd(&stream[0]);
    SetError(error, "cannot start " + std::string(program) + ": " +
                        std::string(std::strerror(code)));
    return false;
  }
  output->clear();
  std::array<char, 4096> buffer{};
  while (true) {
    const ssize_t amount = read(stream[0], buffer.data(), buffer.size());
    if (amount < 0 && errno == EINTR) {
      continue;
    }
    if (amount <= 0) {
      break;
    }
    if (output->size() + static_cast<std::size_t>(amount) > (1U << 20U)) {
      CloseFd(&stream[0]);
      (void)kill(child, SIGTERM);
      int ignored = 0;
      (void)WaitForChild(child, &ignored, nullptr);
      SetError(error, "media inspection output exceeded 1 MiB");
      return false;
    }
    output->append(buffer.data(), static_cast<std::size_t>(amount));
  }
  CloseFd(&stream[0]);
  int status = 0;
  if (!WaitForChild(child, &status, error)) {
    return false;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    SetError(error,
             std::string(program) + " exited with status " +
                 std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
    return false;
  }
  return true;
}

using KeyValues = std::vector<std::pair<std::string, std::string>>;

KeyValues ParseKeyValues(std::string_view text) {
  KeyValues result;
  while (!text.empty()) {
    const std::size_t newline = text.find('\n');
    std::string_view line = text.substr(0, newline);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    const std::size_t equals = line.find('=');
    if (equals != std::string_view::npos) {
      result.emplace_back(line.substr(0, equals), line.substr(equals + 1));
    }
    if (newline == std::string_view::npos) {
      break;
    }
    text.remove_prefix(newline + 1);
  }
  return result;
}

const std::string* FindValue(const KeyValues& values, std::string_view key) {
  for (const auto& [candidate, value] : values) {
    if (candidate == key) {
      return &value;
    }
  }
  return nullptr;
}

}  // namespace

bool WriteSynchronizedMp4(const std::filesystem::path& path,
                          std::span<const std::uint8_t> rgb24, int frame_count,
                          const MediaVideoOptions& video,
                          std::span<const float> channel_major_pcm, int samples,
                          int channels, int sample_rate,
                          const CancellationToken* cancellation,
                          MediaMuxTelemetry* telemetry, std::string* error) {
  if (telemetry != nullptr) {
    *telemetry = {};
  }
  if (path.empty() || frame_count < 1 || video.input_width < 2 ||
      video.input_height < 2 || video.output_width < 2 ||
      video.output_height < 2 || (video.input_width & 1) != 0 ||
      (video.input_height & 1) != 0 || (video.output_width & 1) != 0 ||
      (video.output_height & 1) != 0 || video.fps < 1 || samples < 1 ||
      channels < 1 || sample_rate < 1) {
    SetError(error, "invalid MiniMax H3 media mux arguments");
    return false;
  }
  const std::uint64_t frame_bytes =
      static_cast<std::uint64_t>(video.input_width) * video.input_height * 3U;
  const std::uint64_t video_bytes =
      frame_bytes * static_cast<std::uint64_t>(frame_count);
  const std::uint64_t audio_elements =
      static_cast<std::uint64_t>(samples) * channels;
  if (video_bytes != rgb24.size() ||
      audio_elements != channel_major_pcm.size() ||
      video_bytes > std::numeric_limits<std::size_t>::max() ||
      audio_elements >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    SetError(error, "MiniMax H3 media stream geometry differs from buffers");
    return false;
  }
  if (IsCancelled(cancellation)) {
    if (telemetry != nullptr) {
      telemetry->cancelled = true;
    }
    SetError(error, "MiniMax H3 media mux cancelled before start");
    return false;
  }
  std::error_code filesystem_error;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
  }
  if (filesystem_error) {
    SetError(error, "cannot create media output directory: " +
                        filesystem_error.message());
    return false;
  }
  const std::filesystem::path partial_path = PartialMediaPath(path);
  std::filesystem::remove(partial_path, filesystem_error);
  filesystem_error.clear();

  int video_pipe[2] = {-1, -1};
  int audio_pipe[2] = {-1, -1};
  if (pipe(video_pipe) != 0 || pipe(audio_pipe) != 0) {
    const int saved = errno;
    CloseFd(&video_pipe[0]);
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[0]);
    CloseFd(&audio_pipe[1]);
    SetError(error, "cannot create FFmpeg media pipes: " +
                        std::string(std::strerror(saved)));
    return false;
  }
  const int audio_target =
      std::max({video_pipe[0], video_pipe[1], audio_pipe[0], audio_pipe[1]}) +
      1;
  const std::string size = std::to_string(video.input_width) + "x" +
                           std::to_string(video.input_height);
  const std::string output_size = std::to_string(video.output_width) + ":" +
                                  std::to_string(video.output_height);
  const std::string rate = std::to_string(video.fps);
  const std::string audio_rate = std::to_string(sample_rate);
  const std::string audio_channels = std::to_string(channels);
  const std::string audio_input = "pipe:" + std::to_string(audio_target);
  const std::string output_path = partial_path.string();
  std::vector<std::string> argument_storage = {
      "ffmpeg",
      "-y",
      "-loglevel",
      "error",
      "-f",
      "rawvideo",
      "-pixel_format",
      "rgb24",
      "-video_size",
      size,
      "-framerate",
      rate,
      "-i",
      "pipe:0",
      "-f",
      "f32le",
      "-ar",
      audio_rate,
      "-ac",
      audio_channels,
      "-i",
      audio_input,
      "-map",
      "0:v:0",
      "-map",
      "1:a:0",
      "-vf",
      "scale=" + output_size + ":flags=lanczos,setsar=1",
      "-c:v",
      "libx264",
      "-preset",
      "fast",
      "-crf",
      "18",
      "-pix_fmt",
      "yuv420p",
      "-c:a",
      "aac",
      "-b:a",
      "192k",
      "-movflags",
      "+faststart",
      output_path,
  };
  std::vector<char*> arguments = MutableArgv(argument_storage);
  posix_spawn_file_actions_t actions;
  int code = posix_spawn_file_actions_init(&actions);
  if (code == 0) {
    code =
        posix_spawn_file_actions_adddup2(&actions, video_pipe[0], STDIN_FILENO);
  }
  if (code == 0) {
    code =
        posix_spawn_file_actions_adddup2(&actions, audio_pipe[0], audio_target);
  }
  const std::array<int, 4> descriptors = {video_pipe[0], video_pipe[1],
                                          audio_pipe[0], audio_pipe[1]};
  for (const int descriptor : descriptors) {
    if (code == 0) {
      code =
          AddCloseUnlessKept(&actions, descriptor, STDIN_FILENO, audio_target);
    }
  }
  pid_t child = -1;
  if (code == 0) {
    code = posix_spawnp(&child, FfmpegProgram(), &actions, nullptr,
                        arguments.data(), environ);
  }
  (void)posix_spawn_file_actions_destroy(&actions);
  CloseFd(&video_pipe[0]);
  CloseFd(&audio_pipe[0]);
  if (code != 0) {
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[1]);
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "cannot start FFmpeg: " + std::string(std::strerror(code)));
    return false;
  }
  if (!MakeNonBlocking(video_pipe[1], error) ||
      !MakeNonBlocking(audio_pipe[1], error)) {
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[1]);
    (void)kill(child, SIGTERM);
    int ignored = 0;
    (void)WaitForChild(child, &ignored, nullptr);
    std::filesystem::remove(partial_path, filesystem_error);
    return false;
  }

  std::atomic<bool> stop{false};
  std::atomic<int> video_error{0};
  std::atomic<int> audio_error{0};
  std::uint64_t written_video = 0;
  std::uint64_t written_audio = 0;
  std::size_t maximum_audio_staging = 0;
  if (!EnsureSigpipeIgnored(error)) {
    CloseFd(&video_pipe[1]);
    CloseFd(&audio_pipe[1]);
    (void)kill(child, SIGTERM);
    int ignored = 0;
    (void)WaitForChild(child, &ignored, nullptr);
    std::filesystem::remove(partial_path, filesystem_error);
    return false;
  }

  std::thread video_thread([&] {
    const std::uint8_t* cursor = rgb24.data();
    std::size_t remaining = rgb24.size();
    while (remaining != 0 && !stop.load(std::memory_order_acquire)) {
      if (IsCancelled(cancellation)) {
        stop.store(true, std::memory_order_release);
        break;
      }
      const std::size_t request = std::min(remaining, kIoChunkBytes);
      const ssize_t amount =
          CancellableWrite(video_pipe[1], cursor, request, cancellation, stop);
      if (amount <= 0) {
        if (!IsCancelled(cancellation) &&
            !stop.load(std::memory_order_acquire)) {
          video_error.store(amount < 0 ? errno : EIO,
                            std::memory_order_release);
        }
        stop.store(true, std::memory_order_release);
        break;
      }
      cursor += amount;
      remaining -= static_cast<std::size_t>(amount);
      written_video += static_cast<std::uint64_t>(amount);
    }
    CloseFd(&video_pipe[1]);
  });
  std::thread audio_thread([&] {
    std::vector<float> interleaved(
        static_cast<std::size_t>(kAudioChunkSamples) * channels);
    maximum_audio_staging = interleaved.size() * sizeof(float);
    int offset = 0;
    while (offset < samples && !stop.load(std::memory_order_acquire)) {
      if (IsCancelled(cancellation)) {
        stop.store(true, std::memory_order_release);
        break;
      }
      const int chunk = std::min(kAudioChunkSamples, samples - offset);
      for (int sample = 0; sample < chunk; ++sample) {
        for (int channel = 0; channel < channels; ++channel) {
          interleaved[static_cast<std::size_t>(sample) * channels + channel] =
              channel_major_pcm[static_cast<std::size_t>(channel) * samples +
                                offset + sample];
        }
      }
      const std::uint8_t* cursor =
          reinterpret_cast<const std::uint8_t*>(interleaved.data());
      std::size_t remaining =
          static_cast<std::size_t>(chunk) * channels * sizeof(float);
      while (remaining != 0 && !stop.load(std::memory_order_acquire)) {
        const ssize_t amount = CancellableWrite(audio_pipe[1], cursor,
                                                remaining, cancellation, stop);
        if (amount <= 0) {
          if (!IsCancelled(cancellation) &&
              !stop.load(std::memory_order_acquire)) {
            audio_error.store(amount < 0 ? errno : EIO,
                              std::memory_order_release);
          }
          stop.store(true, std::memory_order_release);
          break;
        }
        cursor += amount;
        remaining -= static_cast<std::size_t>(amount);
        written_audio += static_cast<std::uint64_t>(amount);
      }
      offset += chunk;
    }
    CloseFd(&audio_pipe[1]);
  });
  video_thread.join();
  audio_thread.join();

  const bool cancelled = IsCancelled(cancellation);
  if (stop.load(std::memory_order_acquire) || cancelled) {
    (void)kill(child, SIGTERM);
  }
  int status = 0;
  const bool waited = WaitForChild(child, &status, error);
  if (telemetry != nullptr) {
    telemetry->video_bytes = written_video;
    telemetry->audio_bytes = written_audio;
    telemetry->maximum_audio_staging_bytes = maximum_audio_staging;
    telemetry->child_status =
        waited && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    telemetry->cancelled = cancelled;
  }
  if (!waited) {
    std::filesystem::remove(partial_path, filesystem_error);
    return false;
  }
  if (cancelled) {
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "MiniMax H3 media mux cancelled");
    return false;
  }
  const int writer_error = video_error.load(std::memory_order_acquire) != 0
                               ? video_error.load(std::memory_order_relaxed)
                               : audio_error.load(std::memory_order_acquire);
  if (writer_error != 0) {
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "cannot stream generated media to FFmpeg: " +
                        std::string(std::strerror(writer_error)));
    return false;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error,
             "FFmpeg exited with status " +
                 std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
    return false;
  }
  if (written_video != video_bytes ||
      written_audio != audio_elements * sizeof(float)) {
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "FFmpeg media streams ended before all bytes were sent");
    return false;
  }
  MediaProbe probe;
  const double expected_video_duration =
      static_cast<double>(frame_count) / video.fps;
  const double expected_audio_duration =
      static_cast<double>(samples) / sample_rate;
  if (!ProbeMediaFile(partial_path, &probe, error) ||
      probe.video_codec != "h264" || probe.audio_codec != "aac" ||
      probe.width != video.output_width ||
      probe.height != video.output_height ||
      std::abs(probe.video_frame_rate - video.fps) > 1.0e-6 ||
      probe.audio_channels != channels ||
      probe.audio_sample_rate != sample_rate ||
      std::abs(probe.video_start_seconds - probe.audio_start_seconds) >
          1.0 / sample_rate ||
      std::abs(probe.video_duration_seconds - expected_video_duration) >
          1.0 / video.fps ||
      std::abs(probe.audio_duration_seconds - expected_audio_duration) >
          2048.0 / sample_rate ||
      std::abs(probe.video_duration_seconds - probe.audio_duration_seconds) >
          0.075) {
    std::filesystem::remove(partial_path, filesystem_error);
    if (error != nullptr && error->empty()) {
      *error = "generated MiniMax H3 media violates the delivery contract";
    }
    return false;
  }
  std::filesystem::rename(partial_path, path, filesystem_error);
  if (filesystem_error) {
    const std::string message = filesystem_error.message();
    std::filesystem::remove(partial_path, filesystem_error);
    SetError(error, "cannot publish generated media: " + message);
    return false;
  }
  return true;
}

bool WriteSynchronizedMp4(const std::filesystem::path& path,
                          std::span<const std::uint8_t> rgb24, int frame_count,
                          int width, int height, int fps,
                          std::span<const float> channel_major_pcm, int samples,
                          int channels, int sample_rate,
                          const CancellationToken* cancellation,
                          MediaMuxTelemetry* telemetry, std::string* error) {
  return WriteSynchronizedMp4(path, rgb24, frame_count,
                              {.input_width = width,
                               .input_height = height,
                               .output_width = width,
                               .output_height = height,
                               .fps = fps},
                              channel_major_pcm, samples, channels, sample_rate,
                              cancellation, telemetry, error);
}

bool ProbeMediaFile(const std::filesystem::path& path, MediaProbe* probe,
                    std::string* error) {
  if (path.empty() || probe == nullptr) {
    SetError(error, "invalid MiniMax H3 media probe arguments");
    return false;
  }
  *probe = {};
  const std::string path_text = path.string();
  std::string video_output;
  std::vector<std::string> video_arguments = {
      "ffprobe",
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=codec_name,width,height,r_frame_rate,start_time,duration",
      "-of",
      "default=noprint_wrappers=1",
      path_text,
  };
  if (!RunCapture(FfprobeProgram(), std::move(video_arguments), &video_output,
                  error)) {
    return false;
  }
  const KeyValues video = ParseKeyValues(video_output);
  const std::string* video_codec = FindValue(video, "codec_name");
  const std::string* width = FindValue(video, "width");
  const std::string* height = FindValue(video, "height");
  const std::string* frame_rate = FindValue(video, "r_frame_rate");
  const std::string* video_start = FindValue(video, "start_time");
  const std::string* video_duration = FindValue(video, "duration");
  if (video_codec == nullptr || width == nullptr || height == nullptr ||
      frame_rate == nullptr || video_start == nullptr ||
      video_duration == nullptr || !ParseInteger(*width, &probe->width) ||
      !ParseInteger(*height, &probe->height) ||
      !ParseRate(*frame_rate, &probe->video_frame_rate) ||
      !ParseDouble(*video_start, &probe->video_start_seconds) ||
      !ParseDouble(*video_duration, &probe->video_duration_seconds)) {
    SetError(error, "FFprobe returned malformed MiniMax H3 video metadata");
    return false;
  }
  probe->video_codec = *video_codec;

  std::string audio_output;
  std::vector<std::string> audio_arguments = {
      "ffprobe",
      "-v",
      "error",
      "-select_streams",
      "a:0",
      "-show_entries",
      "stream=codec_name,channels,sample_rate,start_time,duration",
      "-of",
      "default=noprint_wrappers=1",
      path_text,
  };
  if (!RunCapture(FfprobeProgram(), std::move(audio_arguments), &audio_output,
                  error)) {
    return false;
  }
  const KeyValues audio = ParseKeyValues(audio_output);
  const std::string* audio_codec = FindValue(audio, "codec_name");
  const std::string* audio_channels = FindValue(audio, "channels");
  const std::string* audio_rate = FindValue(audio, "sample_rate");
  const std::string* audio_start = FindValue(audio, "start_time");
  const std::string* audio_duration = FindValue(audio, "duration");
  if (audio_codec == nullptr || audio_channels == nullptr ||
      audio_rate == nullptr || audio_start == nullptr ||
      audio_duration == nullptr ||
      !ParseInteger(*audio_channels, &probe->audio_channels) ||
      !ParseInteger(*audio_rate, &probe->audio_sample_rate) ||
      !ParseDouble(*audio_start, &probe->audio_start_seconds) ||
      !ParseDouble(*audio_duration, &probe->audio_duration_seconds)) {
    SetError(error, "FFprobe returned malformed MiniMax H3 audio metadata");
    return false;
  }
  probe->audio_codec = *audio_codec;
  return true;
}

}  // namespace gufo::minimax_h3
