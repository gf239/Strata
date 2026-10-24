// include/strata/flint/engine.hpp - the Recursive Flint (GGUF architecture flint) engine path as a library: `strata`
// hands a flint model to it at the top of main() (the family switch), and strata-flint is a thin wrapper around the
// same entry point.  See src/flint/flint_engine.cu.
#pragma once

namespace strata::flint {

/// The engine's command line (strata's flag names where they mean the same thing: --native / --model, --max-context,
/// --prefill / --chunk, --serve, --tokens, --max-new, --no-token-graph).  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::flint
