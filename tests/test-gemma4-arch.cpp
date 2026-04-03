#include "llama-arch.h"

#include <cstdlib>
#include <iostream>
#include <string>

static int failures = 0;
static int tests = 0;

static void check(const char * name, bool cond) {
    ++tests;
    if (!cond) {
        ++failures;
        std::cerr << "FAIL: " << name << "\n";
    } else {
        std::cout << "PASS: " << name << "\n";
    }
}

int main() {
    check("gemma3n arch from string",
          llm_arch_from_string("gemma3n") == LLM_ARCH_GEMMA3N);

    check("gemma4 arch from string",
          llm_arch_from_string("gemma4") == LLM_ARCH_GEMMA4);

    check("gemma3n arch name",
          std::string(llama_model_arch_name(LLM_ARCH_GEMMA3N)) == "gemma3n");

    check("gemma4 arch name",
          std::string(llama_model_arch_name(LLM_ARCH_GEMMA4)) == "gemma4");

    check("unknown arch unchanged",
          llm_arch_from_string("nonexistent_arch") == LLM_ARCH_UNKNOWN);

    check("existing gemma3 still works",
          llm_arch_from_string("gemma3") == LLM_ARCH_GEMMA3);

    std::cout << "\n" << tests << " tests, " << failures << " failures\n";
    return failures > 0 ? 1 : 0;
}
