#include "src/models/diffusion_gemma/chat.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <optional>

namespace gufo::models::diffusion_gemma {
namespace {

constexpr std::string_view kQuote = "<|\"|>";

// Jinja's `trim`: Python str.strip() whitespace.
std::string Trim(std::string_view text) {
  const auto space = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
  };
  std::size_t begin = 0, end = text.size();
  while (begin < end && space(text[begin]))
    ++begin;
  while (end > begin && space(text[end - 1]))
    --end;
  return std::string(text.substr(begin, end - begin));
}

std::string Upper(std::string text) {
  for (auto& c : text)
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return text;
}

std::string Lower(std::string text) {
  for (auto& c : text)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return text;
}

// Jinja's dictsort: members ordered by key, case-insensitively.
std::vector<const json::Value::Member*> DictSort(const json::Value& object) {
  std::vector<const json::Value::Member*> members;
  if (!object.is_object())
    return members;
  for (const auto& member : object.members())
    members.push_back(&member);
  std::stable_sort(members.begin(), members.end(),
                   [](const auto* a, const auto* b) {
                     return Lower(a->first) < Lower(b->first);
                   });
  return members;
}

// Jinja truthiness of a value that may be absent.
bool Truthy(const json::Value* value) {
  if (value == nullptr || value->is_null())
    return false;
  if (value->is_bool())
    return value->as_bool();
  if (value->is_string())
    return !value->str().empty();
  if (value->is_number())
    return value->as_double() != 0;
  if (value->is_array())
    return !value->items().empty();
  return !value->members().empty();
}

std::string Field(const json::Value& object, const char* key) {
  const auto* value = object.is_object() ? object.find(key) : nullptr;
  if (value == nullptr || value->is_null())
    return {};
  if (value->is_string())
    return value->str();
  return value->dump();
}

const json::Value* Find(const json::Value& object, const char* key) {
  return object.is_object() ? object.find(key) : nullptr;
}

// Python str() of a JSON number: integers without a fraction, otherwise
// the shortest round-trip form.
std::string PythonNumber(double value) {
  if (std::isfinite(value) && value == std::floor(value) &&
      std::fabs(value) < 1e16) {
    return std::to_string(static_cast<long long>(value));
  }
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
  return std::string(buffer, result.ptr);
}

std::string RequiredList(const json::Value* required) {
  std::string out = "required:[";
  bool first = true;
  for (const auto& item : required->items()) {
    if (!first)
      out += ',';
    first = false;
    out += std::string(kQuote) + (item.is_string() ? item.str() : item.dump()) +
           std::string(kQuote);
  }
  return out + "]";
}

// The template's format_parameters macro.
std::string FormatParameters(const json::Value& properties, bool filter_keys) {
  static const std::vector<std::string> kStandard = {
      "description", "type", "properties", "required", "nullable"};
  std::string out;
  bool found_first = false;
  for (const auto* member : DictSort(properties)) {
    const auto& key = member->first;
    const auto& value = member->second;
    if (filter_keys &&
        std::find(kStandard.begin(), kStandard.end(), key) != kStandard.end())
      continue;
    if (found_first)
      out += ',';
    found_first = true;
    out += key + ":{";
    bool add_comma = false;
    const auto comma = [&] {
      if (add_comma)
        out += ',';
      else
        add_comma = true;
    };
    if (Truthy(Find(value, "description"))) {
      out += "description:" + std::string(kQuote) +
             Field(value, "description") + std::string(kQuote);
      add_comma = true;
    }
    const auto* type_value = Find(value, "type");
    const std::string type =
        type_value && type_value->is_string() ? Upper(type_value->str()) : "";
    if (type == "STRING") {
      if (Truthy(Find(value, "enum"))) {
        comma();
        out += "enum:" + FormatArgument(*Find(value, "enum"), true);
      }
    } else if (type == "ARRAY") {
      const auto* items = Find(value, "items");
      if (items != nullptr && items->is_object() && Truthy(items)) {
        comma();
        out += "items:{";
        bool first_item = false;
        for (const auto* item : DictSort(*items)) {
          if (item->second.is_null())
            continue;
          if (first_item)
            out += ',';
          first_item = true;
          if (item->first == "properties") {
            out += "properties:{";
            if (item->second.is_object())
              out += FormatParameters(item->second, false);
            out += "}";
          } else if (item->first == "required") {
            out += RequiredList(&item->second);
          } else if (item->first == "type") {
            if (item->second.is_string()) {
              out +=
                  "type:" +
                  FormatArgument(json::Value(Upper(item->second.str())), true);
            } else {
              json::Value upper = json::Value::array();
              for (const auto& t : item->second.items())
                upper.push_back(
                    json::Value(Upper(t.is_string() ? t.str() : t.dump())));
              out += "type:" + FormatArgument(upper, true);
            }
          } else {
            out += item->first + ":" + FormatArgument(item->second, true);
          }
        }
        out += "}";
      }
    }
    if (Truthy(Find(value, "nullable"))) {
      comma();
      out += "nullable:true";
    }
    if (type == "OBJECT") {
      const auto* nested = Find(value, "properties");
      if (nested != nullptr && nested->is_object()) {
        comma();
        out += "properties:{" + FormatParameters(*nested, false) + "}";
      } else if (value.is_object()) {
        comma();
        out += "properties:{" + FormatParameters(value, true) + "}";
      }
      if (Truthy(Find(value, "required"))) {
        comma();
        out += RequiredList(Find(value, "required"));
      }
    }
    comma();
    out += "type:" + std::string(kQuote) + type + std::string(kQuote) + "}";
  }
  return out;
}

std::string ToolResponse(const std::string& name, const std::string& body) {
  return "<|tool_response>response:" + name +
         "{value:" + FormatArgument(json::Value(body), false) +
         "}<tool_response|>";
}

// The template's strip_thinking macro.
std::string StripChannels(std::string_view text) {
  std::string result;
  std::size_t start = 0;
  while (true) {
    const auto close = text.find("<channel|>", start);
    const auto part = text.substr(start, close == std::string_view::npos
                                             ? std::string_view::npos
                                             : close - start);
    const auto open = part.find("<|channel>");
    result += part.substr(0, open);
    if (close == std::string_view::npos)
      break;
    start = close + 10;
  }
  return Trim(result);
}

// Native call syntax: Gemma's format_argument without escaped keys.
class CallParser {
public:
  explicit CallParser(std::string_view text) : text_(text) {}

  std::optional<json::Value> Value() {
    Skip();
    if (Peek(kQuote)) {
      pos_ += kQuote.size();
      const auto end = text_.find(kQuote, pos_);
      if (end == std::string_view::npos)
        return std::nullopt;
      std::string s(text_.substr(pos_, end - pos_));
      pos_ = end + kQuote.size();
      return json::Value(std::move(s));
    }
    if (pos_ < text_.size() && text_[pos_] == '{')
      return Object();
    if (pos_ < text_.size() && text_[pos_] == '[') {
      ++pos_;
      json::Value array = json::Value::array();
      Skip();
      if (pos_ < text_.size() && text_[pos_] == ']') {
        ++pos_;
        return array;
      }
      while (true) {
        auto item = Value();
        if (!item)
          return std::nullopt;
        array.push_back(std::move(*item));
        Skip();
        if (pos_ >= text_.size())
          return std::nullopt;
        if (text_[pos_] == ',') {
          ++pos_;
          continue;
        }
        if (text_[pos_] == ']') {
          ++pos_;
          return array;
        }
        return std::nullopt;
      }
    }
    // A bare literal runs to the next delimiter.
    const auto end = text_.find_first_of(",}]", pos_);
    if (end == std::string_view::npos)
      return std::nullopt;
    const std::string literal = Trim(text_.substr(pos_, end - pos_));
    pos_ = end;
    if (literal == "true")
      return json::Value(true);
    if (literal == "false")
      return json::Value(false);
    if (literal == "null" || literal == "None")
      return json::Value();
    double number = 0;
    const auto parsed = std::from_chars(
        literal.data(), literal.data() + literal.size(), number);
    if (parsed.ec == std::errc() &&
        parsed.ptr == literal.data() + literal.size())
      return json::Value(number);
    return json::Value(literal);
  }

  std::optional<json::Value> Object() {
    Skip();
    if (pos_ >= text_.size() || text_[pos_] != '{')
      return std::nullopt;
    ++pos_;
    json::Value object = json::Value::object();
    Skip();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return object;
    }
    while (true) {
      Skip();
      std::string key;
      if (Peek(kQuote)) {
        pos_ += kQuote.size();
        const auto end = text_.find(kQuote, pos_);
        if (end == std::string_view::npos)
          return std::nullopt;
        key = std::string(text_.substr(pos_, end - pos_));
        pos_ = end + kQuote.size();
        Skip();
        if (pos_ >= text_.size() || text_[pos_] != ':')
          return std::nullopt;
        ++pos_;
      } else {
        const auto colon = text_.find(':', pos_);
        if (colon == std::string_view::npos)
          return std::nullopt;
        key = Trim(text_.substr(pos_, colon - pos_));
        pos_ = colon + 1;
      }
      auto value = Value();
      if (!value)
        return std::nullopt;
      object[key] = std::move(*value);
      Skip();
      if (pos_ >= text_.size())
        return std::nullopt;
      if (text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (text_[pos_] == '}') {
        ++pos_;
        return object;
      }
      return std::nullopt;
    }
  }

private:
  bool Peek(std::string_view s) const {
    return text_.substr(pos_, s.size()) == s;
  }
  void Skip() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_])))
      ++pos_;
  }

  std::string_view text_;
  std::size_t pos_{0};
};

}  // namespace

std::string FormatArgument(const json::Value& value, bool escape_keys) {
  if (value.is_null())
    return "null";
  if (value.is_string())
    return std::string(kQuote) + value.str() + std::string(kQuote);
  if (value.is_bool())
    return value.as_bool() ? "true" : "false";
  if (value.is_number())
    return PythonNumber(value.as_double());
  if (value.is_object()) {
    std::string out = "{";
    bool first = true;
    for (const auto* member : DictSort(value)) {
      if (!first)
        out += ',';
      first = false;
      out += escape_keys
                 ? std::string(kQuote) + member->first + std::string(kQuote)
                 : member->first;
      out += ":" + FormatArgument(member->second, escape_keys);
    }
    return out + "}";
  }
  std::string out = "[";
  bool first = true;
  for (const auto& item : value.items()) {
    if (!first)
      out += ',';
    first = false;
    out += FormatArgument(item, escape_keys);
  }
  return out + "]";
}

std::string FormatToolDeclaration(const json::Value& tool) {
  const auto* function = Find(tool, "function");
  const json::Value empty = json::Value::object();
  const auto& f = function != nullptr ? *function : empty;
  std::string out = "declaration:" + Field(f, "name") +
                    "{description:" + std::string(kQuote) +
                    Field(f, "description") + std::string(kQuote);
  const auto* params = Find(f, "parameters");
  if (Truthy(params)) {
    out += ",parameters:{";
    const auto* properties = Find(*params, "properties");
    if (Truthy(properties))
      out += "properties:{" + FormatParameters(*properties, false) + "},";
    const auto* required = Find(*params, "required");
    if (Truthy(required))
      out += RequiredList(required) + ",";
    if (Truthy(Find(*params, "type")))
      out += "type:" + std::string(kQuote) + Upper(Field(*params, "type")) +
             std::string(kQuote) + "}";
  }
  if (const auto* response = Find(f, "response")) {
    out += ",response:{";
    if (Truthy(Find(*response, "description")))
      out += "description:" + std::string(kQuote) +
             Field(*response, "description") + std::string(kQuote) + ",";
    if (Upper(Field(*response, "type")) == "OBJECT")
      out += "type:" + std::string(kQuote) + Upper(Field(*response, "type")) +
             std::string(kQuote) + "}";
  }
  return out + "}";
}

std::string RenderChat(const std::vector<Message>& messages,
                       const std::vector<json::Value>& tools, bool thinking) {
  std::string out = "<bos>";
  std::string prev_type;  // "", "think", "tool", "tool_call", "tool_response"
  std::size_t first = 0;
  const bool leading_system =
      !messages.empty() &&
      (messages[0].role == "system" || messages[0].role == "developer");
  if (thinking || !tools.empty() || leading_system) {
    out += "<|turn>system\n";
    if (thinking) {
      out += "<|think|>\n";
      prev_type = "think";
    }
    if (leading_system) {
      out += Trim(messages[0].content);
      first = 1;
    }
    if (!tools.empty()) {
      for (const auto& tool : tools)
        out += "<|tool>" + Trim(FormatToolDeclaration(tool)) + "<tool|>";
      prev_type = "tool";
    }
    out += "<turn|>\n";
  }
  std::string prev_non_tool_role;
  for (std::size_t i = first; i < messages.size(); ++i) {
    const auto& message = messages[i];
    if (message.role == "tool") {
      // Rendered after the assistant call that requested it. The template's
      // prev_non_tool_role is updated for tool messages too, so an assistant
      // message after tool results opens a new model turn.
      prev_non_tool_role = message.role;
      continue;
    }
    prev_type.clear();
    const bool model = message.role == "assistant";
    const std::string role = model ? "model" : message.role;
    const bool continue_turn = model && prev_non_tool_role == "assistant";
    if (!continue_turn)
      out += "<|turn>" + role + "\n";
    for (const auto& call : message.tool_calls) {
      out += "<|tool_call>call:" + call.name + "{";
      bool first_arg = true;
      for (const auto* argument : DictSort(call.arguments)) {
        if (!first_arg)
          out += ',';
        first_arg = false;
        out += argument->first + ":" + FormatArgument(argument->second, false);
      }
      out += "}<tool_call|>";
      prev_type = "tool_call";
    }
    bool responded = false;
    if (!message.tool_calls.empty()) {
      for (std::size_t k = i + 1;
           k < messages.size() && messages[k].role == "tool"; ++k) {
        const auto& follow = messages[k];
        std::string name = follow.name.empty() ? "unknown" : follow.name;
        for (const auto& call : message.tool_calls)
          if (!call.id.empty() && call.id == follow.tool_call_id)
            name = call.name;
        out += ToolResponse(name, follow.content);
        responded = true;
        prev_type = "tool_response";
      }
    }
    const std::string captured =
        model ? StripChannels(message.content) : Trim(message.content);
    out += captured;
    const bool has_content = !Trim(captured).empty();
    std::string next_role;
    for (std::size_t j = i + 1; j < messages.size(); ++j)
      if (messages[j].role != "tool") {
        next_role = messages[j].role;
        break;
      }
    const bool continues_into_next = model && next_role == "assistant" &&
                                     (message.tool_calls.empty() || responded);
    if (prev_type == "tool_call" && !responded) {
      out += "<|tool_response>";
    } else if (continues_into_next) {
    } else if (!(responded && !has_content && next_role.empty())) {
      out += "<turn|>\n";
    }
    prev_non_tool_role = message.role;
  }
  if (prev_type != "tool_response" && prev_type != "tool_call")
    out += "<|turn>model\n";
  else if (prev_type == "tool_response" && thinking)
    out += "<|channel>thought\n";
  return out;
}

std::string NativeToolCallsToXml(std::string_view text, std::size_t* held) {
  constexpr std::string_view kOpen = "<|tool_call>";
  constexpr std::string_view kClose = "<tool_call|>";
  *held = std::string_view::npos;
  std::string out;
  std::size_t cursor = 0;
  while (true) {
    const auto open = text.find(kOpen, cursor);
    if (open == std::string_view::npos) {
      out += text.substr(cursor);
      return out;
    }
    out += text.substr(cursor, open - cursor);
    const auto close = text.find(kClose, open);
    if (close == std::string_view::npos) {
      *held = open;  // Still being generated.
      return out;
    }
    std::string_view body =
        text.substr(open + kOpen.size(), close - open - kOpen.size());
    std::string xml;
    if (body.starts_with("call:")) {
      body.remove_prefix(5);
      const auto brace = body.find('{');
      if (brace != std::string_view::npos) {
        const std::string name = Trim(body.substr(0, brace));
        CallParser parser(body.substr(brace));
        if (const auto arguments = parser.Object();
            arguments && !name.empty()) {
          xml = "<tool_call>\n<function=" + name + ">\n";
          for (const auto& [key, value] : arguments->members())
            xml += "<parameter=" + key + ">\n" +
                   (value.is_string() ? value.str() : value.dump()) +
                   "\n</parameter>\n";
          xml += "</function>\n</tool_call>";
        }
      }
    }
    // A malformed call stays visible as text rather than vanishing.
    out += xml.empty()
               ? std::string(text.substr(open, close + kClose.size() - open))
               : xml;
    cursor = close + kClose.size();
  }
}

std::string StripThinking(const std::string& text) {
  const auto open = text.find("<|channel>");
  if (open == std::string::npos)
    return text;
  const auto close = text.find("<channel|>", open);
  if (close == std::string::npos)
    return text.substr(0, open);
  return text.substr(0, open) + text.substr(close + 10);
}

std::vector<std::int32_t> StripThinkingTokens(
    std::span<const std::int32_t> tokens, std::int32_t open,
    std::int32_t close) {
  std::vector<std::int32_t> out;
  out.reserve(tokens.size());
  bool inside = false;
  for (const auto token : tokens) {
    if (!inside && token == open) {
      inside = true;
    } else if (inside) {
      if (token == close)
        inside = false;
    } else {
      out.push_back(token);
    }
  }
  return out;
}

}  // namespace gufo::models::diffusion_gemma
