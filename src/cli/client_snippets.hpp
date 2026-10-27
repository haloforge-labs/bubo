#pragma once

#include <string>
#include <string_view>

#include "src/cli/machine_output.hpp"

namespace bubo::cli {
inline constexpr std::string_view kLoopbackDummyKey = "bubo-loopback";

// Addresses and credentials remain placeholders, including for wildcard/LAN
// listeners. The client selects a reachable address and explicitly fills a key.
inline Value ClientSnippets(std::string_view model, bool requires_key) {
  if (model.empty())
    return Value();
  auto snippets = Value::array();
  const auto add = [&](const char* kind, const char* title, std::string text) {
    auto row = Value::object();
    row["kind"] = kind;
    row["title"] = title;
    row["template"] = std::move(text);
    row["requires_key"] = requires_key;
    row["default_api_key"] =
        requires_key ? Value() : Value(std::string(kLoopbackDummyKey));
    snippets.push_back(std::move(row));
  };
  // JSON string escaping also produces a valid Python string literal.
  const auto model_json = Value(std::string(model)).dump();
  add("openai_python", "OpenAI Python SDK",
      "from openai import OpenAI\n\n"
      "client = OpenAI(base_url=\"{openai_base_url}\", api_key=\"{api_key}\")\n"
      "response = client.chat.completions.create(\n"
      "    model=" +
          model_json +
          ",\n"
          "    messages=[{\"role\": \"user\", \"content\": \"Hello\"}],\n"
          ")\nprint(response.choices[0].message.content)\n");
  auto body = Value::object();
  body["model"] = std::string(model);
  auto message = Value::object();
  message["role"] = "user";
  message["content"] = "Hello";
  body["messages"] = Value::array();
  body["messages"].push_back(message);
  // No curl template: a POSIX-quoted curl line fails in both cmd and
  // PowerShell, and the PowerShell template covers the Windows shell.
  // PowerShell single-quoted literals escape apostrophes by doubling them.
  std::string powershell_body;
  for (const char ch : body.dump())
    powershell_body += ch == '\'' ? "''" : std::string(1, ch);
  // Windows PowerShell 5.1 sends a string body as ISO-8859-1 unless the
  // content type names UTF-8.
  add("powershell_openai_chat", "PowerShell Invoke-RestMethod",
      "Invoke-RestMethod -Method Post `\n"
      "  -Uri '{openai_base_url}/chat/completions' `\n"
      "  -Headers @{ Authorization = 'Bearer {api_key}' } `\n"
      "  -ContentType 'application/json; charset=utf-8' `\n"
      "  -Body '" +
          powershell_body + "'\n");
  // auth_token= sends "Authorization: Bearer", which the server reads.
  // api_key= sends X-Api-Key, which it does not (a 401 when a key is set).
  add("anthropic_python", "Anthropic Python SDK",
      "from anthropic import Anthropic\n\n"
      "client = Anthropic(base_url=\"{anthropic_base_url}\", "
      "auth_token=\"{api_key}\")\n"
      "response = client.messages.create(\n"
      "    model=" +
          model_json +
          ", max_tokens=128,\n"
          "    messages=[{\"role\": \"user\", \"content\": \"Hello\"}],\n"
          ")\nprint(response.content)\n");
  return snippets;
}
}  // namespace bubo::cli
