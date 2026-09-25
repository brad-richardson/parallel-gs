// GB9 knob 1 (ssx3 fork): PGS_HIER_BINNING env override for hierarchical
// binning. Single source of truth shared by GSRenderer (behavior,
// gs_renderer.cpp) and the PS2Recomp [gs-path] logger (reporting), so the
// printed rule always matches the effective rule.
#pragma once
#include <cstdlib>
#include <cstring>

namespace ParallelGS
{

enum class PgsHierBinningMode
{
	Auto, // Unset (or "auto", or unrecognized): today's platform behavior.
	Force, // Run the standard hier-if-large rule on every platform, so the
	       // Mac runs the Odin's binning rule (Fable review §4, knob 1).
	Off // Flat binning (hier target 1) on every platform.
};

inline PgsHierBinningMode pgs_hier_binning_mode()
{
	const char *value = std::getenv("PGS_HIER_BINNING");
	if (value == nullptr || value[0] == '\0')
		return PgsHierBinningMode::Auto;
	if (std::strcmp(value, "force") == 0)
		return PgsHierBinningMode::Force;
	if (std::strcmp(value, "off") == 0)
		return PgsHierBinningMode::Off;
	return PgsHierBinningMode::Auto;
}

// Effective rule: true = flat-always (hier target 1), false = hier-if-large.
// Mirrors the platform branches in get_target_hierarchical_binning.
inline bool pgs_hier_binning_flat_always(PgsHierBinningMode mode)
{
#if defined(__APPLE__)
	// Broken Metal drivers can't deal with the hierarchical binning for some
	// reason (see get_target_hierarchical_binning), so Apple stays flat
	// unless forced.
	return mode != PgsHierBinningMode::Force;
#else
	return mode == PgsHierBinningMode::Off;
#endif
}

} // namespace ParallelGS
