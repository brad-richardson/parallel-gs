// SC1 (B): stable, serializable compute-pipeline variant keys.
//
// A key fully determines the DeferredPipelineCompile that Granite's
// update_hash_compute_pipeline hashes (program + user spec mask/values +
// subgroup state + robustness), so a list recorded with PGS_VARIANT_LOG=1 on
// one launch can be precompiled on the next. Keys are GPU/driver-specific
// (subgroup sizes, program hashes) — record and replay on the same device.
//
// Text form (one per line; see log_pgs_variant_compute in Granite):
//   c1:<prog16>:<mask2>:<specs>:<imask2>:<ispecs>:<sgctl>:<sgmin>:<sgmax>:<sgfull>:<robust>
// where <specs> is "id=value,id=value,..." (masked ids only) or "-".
#pragma once
#include <stdint.h>

namespace ParallelGS
{

struct PgsVariantKey
{
	uint64_t prog_hash = 0;
	uint32_t spec_mask = 0;
	uint32_t specs[8] = {};
	uint32_t ispec_mask = 0;
	uint32_t ispecs[4] = {};
	uint32_t sg_control = 0;
	uint32_t sg_min_log2 = 0;
	uint32_t sg_max_log2 = 0;
	uint32_t sg_full_group = 0;
	uint32_t robust = 0;
};

// Parses a c1:... key. Accepts a bare key or a full "[pgs-variant] key=..."
// log line, so a recording log greps straight into a list file.
// Returns false for graphics keys (g1:...), setting *is_graphics, and for
// malformed lines (is_graphics false).
bool pgs_parse_variant_key(const char *line, PgsVariantKey &key, bool &is_graphics);

} // namespace ParallelGS
