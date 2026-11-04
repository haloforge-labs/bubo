#include "src/models/minimax_h3/media.hpp"

// FFmpeg muxing on Windows needs named-pipe input streams in place of the
// POSIX pipe:N descriptors; that lands with the MiniMax H3 bring-up phase.
// Until then the API fails loudly instead of silently producing nothing.

// FFmpeg muxing on Windows needs named-pipe input streams in place of the
// POSIX pipe:N descriptors; that lands with the MiniMax H3 bring-up phase.
// Until then the API fails loudly instead of silently producing nothing.
#include <string>
#include <utility>

namespace gufo::minimax_h3 {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

bool WriteSynchronizedMp4(const std::filesystem::path& path,
                          std::span<const std::uint8_t> rgb24, int frame_count,
                          const MediaVideoOptions& video,
                          std::span<const float> channel_major_pcm, int samples,
                          int channels, int sample_rate,
                          const CancellationToken* cancellation,
                          MediaMuxTelemetry* telemetry, std::string* error) {
  (void)path;
  (void)rgb24;
  (void)frame_count;
  (void)video;
  (void)channel_major_pcm;
  (void)samples;
  (void)channels;
  (void)sample_rate;
  (void)cancellation;
  if (telemetry != nullptr) {
    *telemetry = {};
  }
  SetError(error, "FFmpeg media muxing is not yet supported on Windows");
  return false;
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
  (void)path;
  if (probe != nullptr) {
    *probe = {};
  }
  SetError(error, "FFprobe media inspection is not yet supported on Windows");
  return false;
}

}  // namespace gufo::minimax_h3
