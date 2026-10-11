#include <curl/curl.h>
#include <openssl/evp.h>
#include <zlib.h>

#include <array>
#include <cassert>
#include <cstring>
#include <iostream>

int main() {
  // Version metadata only: no curl handle, session, socket, DNS or request.
  const auto* curl = curl_version_info(CURLVERSION_NOW);
  assert(curl && curl->ssl_version && (curl->features & CURL_VERSION_SSL));
  bool https = false;
  for (auto protocol = curl->protocols; *protocol; ++protocol)
    https |= std::strcmp(*protocol, "https") == 0;
  assert(https);
#ifdef _WIN32
  assert(std::strstr(curl->ssl_version, "Schannel"));
#endif
  std::array<unsigned char, 32> digest{};
  unsigned int length = 0;
  assert(EVP_Digest("abc", 3, digest.data(), &length, EVP_sha256(), nullptr) ==
         1);
  // Published SHA-256 abc known-answer vector, independent of this code.
  constexpr std::array<unsigned char, 32> expected{
      0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40,
      0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17,
      0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
  assert(length == expected.size() && digest == expected);
  // Fixed raw-DEFLATE stored block containing abc; no ZIP reader/own inflater.
  std::array<unsigned char, 8> deflate{0x01, 0x03, 0x00, 0xfc,
                                       0xff, 'a',  'b',  'c'};
  std::array<unsigned char, 3> output{};
  z_stream stream{};
  assert(inflateInit2(&stream, -15) == Z_OK);
  stream.next_in = deflate.data();
  stream.avail_in = static_cast<uInt>(deflate.size());
  stream.next_out = output.data();
  stream.avail_out = static_cast<uInt>(output.size());
  const auto status = inflate(&stream, Z_FINISH);
  assert(status == Z_STREAM_END && stream.total_in == deflate.size() &&
         stream.total_out == 3 && std::memcmp(output.data(), "abc", 3) == 0);
  assert(inflateEnd(&stream) == Z_OK);
  std::cout << "PASS: curl " << curl->version << " TLS=" << curl->ssl_version
            << " HTTPS available (no request); OpenSSL SHA-256 abc; zlib "
            << zlibVersion() << " raw DEFLATE inflateInit2(-15)\n";
}
