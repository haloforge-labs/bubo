#include "src/cli/arg_parser.hpp"

#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

void TestBasicFlagsAndOptions() {
  gufo::cli::ArgParser parser("test_app", "A test application");
  bool verbose = false;
  std::string model = "default.gguf";
  int threads = 4;
  float temperature = 0.5f;

  parser.AddFlag("-v", "--verbose", "Enable verbose mode", "General", &verbose);
  parser.AddOption("-m", "--model", "PATH", "Path to model", "Model", &model);
  parser.AddOption("-t", "--threads", "N", "Number of threads", "Compute",
                   &threads);
  parser.AddOption("--temp", "--temperature", "T", "Sampling temperature",
                   "Sampling", &temperature);

  const std::array<const char*, 7> args = {"-v", "--model", "custom.gguf", "-t",
                                           "8",  "--temp",  "0.9"};

  std::string err;
  assert(parser.Parse(args, &err));
  assert(verbose == true);
  assert(model == "custom.gguf");
  assert(threads == 8);
  assert(temperature > 0.89f && temperature < 0.91f);
}

void TestListParsing() {
  gufo::cli::ArgParser parser("test_bench");
  std::vector<std::size_t> depths;
  parser.AddOption("", "--n-depth", "LIST", "List of depths", "Bench", &depths);

  const std::array<const char*, 2> args = {"--n-depth", "4096,8192,16384"};
  std::string err;
  assert(parser.Parse(args, &err));
  assert((depths == std::vector<std::size_t>{4096, 8192, 16384}));

  const std::array<const char*, 2> invalid_args = {"--n-depth", "4096,invalid"};
  assert(!parser.Parse(invalid_args, &err));
  assert(!err.empty());
}

void TestPositionals() {
  gufo::cli::ArgParser parser("test_prompt");
  std::string prompt;
  std::string model;
  parser.AddOption("-m", "--model", "PATH", "Model path", "Model", &model);
  parser.JoinPositionals(&prompt);

  const std::array<const char*, 4> args = {"-m", "model.gguf", "Hello",
                                           "world"};
  std::string err;
  assert(parser.Parse(args, &err));
  assert(model == "model.gguf");
  assert(prompt == "Hello world");
}

void TestHelpFormatting() {
  gufo::cli::ArgParser parser("gufo run", "Run inference on Strix Halo");
  std::string model;
  bool interactive = false;
  parser.AddOption("-m", "--model", "PATH", "Model file", "Model", &model);
  parser.AddFlag("-i", "--interactive", "Interactive chat", "Execution",
                 &interactive);

  std::string help = parser.FormatHelp();
  assert(help.find("Usage: gufo run [OPTIONS]") != std::string::npos);
  assert(help.find("Model:") != std::string::npos);
  assert(help.find("Execution:") != std::string::npos);
  assert(help.find("--model") != std::string::npos);
  assert(help.find("--interactive") != std::string::npos);
}

void TestUnexpectedPositionalsAndParserReuse() {
  gufo::cli::ArgParser parser("test");
  std::string text;
  bool verbose = false;
  parser.AddOption("", "--text", "TEXT", "Text", "Input", &text);
  parser.AddFlag("-v", "--verbose", "Verbose", "General", &verbose);
  std::string error;
  const char* unexpected[] = {"typo"};
  assert(!parser.Parse(unexpected, &error));
  assert(error.find("typo") != std::string::npos);
  const char* help[] = {"--help"};
  assert(parser.Parse(help, &error));
  assert(parser.IsHelpRequested() && error.empty());
  const char* value[] = {"--text", "-v"};
  assert(parser.Parse(value, &error));
  assert(!parser.IsHelpRequested());
  assert(text == "-v" && !verbose);
  assert(parser.WasSupplied("--text") && !parser.WasSupplied("-v"));
  const char* inline_value[] = {"--text=audio"};
  assert(parser.Parse(inline_value, &error));
  assert(text == "audio");
  assert(parser.WasSupplied("--text"));
  assert(parser.Parse(help, &error));
  assert(!parser.WasSupplied("--text"));
}

void TestCommandGrammarPolicies() {
  using Parser = gufo::cli::ArgParser;
  Parser parser("test");
  parser.SetImplicitHelp(false).SetInlineValues(false);
  bool help = false;
  unsigned flags = 0;
  std::string text, error;
  parser.AddFlag("", "--help", "Help", "General", &help);
  parser.AddCustomFlag("", "--flag", "Count", "General",
                       [&](auto, auto, auto*) {
                         ++flags;
                         return true;
                       });
  parser.AddOption("", "--text", "TEXT", "Text", "General", &text);
  parser.SetErrorHandler([](Parser::ParseError kind, std::string_view arg) {
    return std::to_string(static_cast<int>(kind)) + ":" + std::string(arg);
  });
  const char* invalid[] = {"--help", "--text=inline"};
  assert(!parser.Parse(invalid, &error));
  assert(help && !parser.IsHelpRequested());
  assert(error == "0:--text=inline");
  const char* missing[] = {"--text"};
  assert(!parser.Parse(missing, &error) && error == "1:--text");
  const char* repeated[] = {"--flag", "--flag"};
  assert(parser.Parse(repeated, &error) && flags == 2);
  parser.SetInlineValues(true);
  const char* flag_value[] = {"--flag=true"};
  assert(!parser.Parse(flag_value, &error) && error == "2:--flag=true");
  std::vector<std::string> delegated;
  parser.SetUnrecognizedHandler([&](std::string_view arg, auto*) {
    delegated.emplace_back(arg);
    return true;
  });
  const char* mixed[] = {"llm", "--unknown=1", "--text", "--flag", "--json"};
  assert(parser.Parse(mixed, &error) && text == "--flag" && flags == 2);
  assert(
      (delegated == std::vector<std::string>{"llm", "--unknown=1", "--json"}));
  assert(Parser::Contains(mixed, {"--json"}));
  assert(!Parser::Contains(mixed, {"--unknown"}));
  assert(Parser::Contains(mixed, {"--unknown"}, true));
  assert(Parser::Contains(mixed, {"--flag"}));  // Raw value-position detection.
  parser.SetOptionInlineValues("--flag", false);
  assert(parser.Parse(flag_value, &error));
  assert(delegated.back() == "--flag=true" && flags == 2);
  const wchar_t* wide[] = {L"--model", L"--jsonl", L"--bundle=a"};
  assert(Parser::Contains(wide, {L"--jsonl"}));
  assert(!Parser::Contains(wide, {L"--bundle"}));
  assert(Parser::Contains(wide, {L"--bundle"}, true));
  const char* help_first[] = {"--help", "--text"};
  parser.SetImplicitHelp(true);
  assert(parser.Parse(help_first, &error) && parser.IsHelpRequested());
}

int main() {
  TestBasicFlagsAndOptions();
  TestListParsing();
  TestPositionals();
  TestHelpFormatting();
  TestUnexpectedPositionalsAndParserReuse();
  TestCommandGrammarPolicies();
  std::cout << "ArgParser: 6 cases passed, 0 failed.\n";
  return 0;
}
