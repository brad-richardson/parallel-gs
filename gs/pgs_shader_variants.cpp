#include "pgs_shader_variants.hpp"
#include <stdlib.h>
#include <string.h>

namespace ParallelGS
{

static bool parse_u32(const char *&p, uint32_t &out, int base, uint32_t max_value)
{
	char *end = nullptr;
	unsigned long value = strtoul(p, &end, base);
	if (end == p || value > max_value)
		return false;
	out = uint32_t(value);
	p = end;
	return true;
}

static bool parse_u64_hex(const char *&p, uint64_t &out)
{
	char *end = nullptr;
	unsigned long long value = strtoull(p, &end, 16);
	if (end == p)
		return false;
	out = uint64_t(value);
	p = end;
	return true;
}

// Parses "id=value,id=value,..." or "-". ids must be < max_id and (when
// required_mask != 0) members of required_mask; values are full u32.
static bool parse_spec_list(const char *&p, uint32_t *values, uint32_t max_id, uint32_t required_mask)
{
	if (*p == '-')
	{
		p++;
		return required_mask == 0;
	}

	uint32_t seen = 0;
	for (;;)
	{
		uint32_t id = 0, value = 0;
		if (!parse_u32(p, id, 10, max_id - 1))
			return false;
		if (*p != '=')
			return false;
		p++;
		if (!parse_u32(p, value, 10, 0xffffffffu))
			return false;
		if (seen & (1u << id))
			return false;
		seen |= 1u << id;
		values[id] = value;
		if (*p != ',')
			break;
		p++;
	}
	return seen == required_mask;
}

bool pgs_parse_variant_key(const char *line, PgsVariantKey &key, bool &is_graphics)
{
	is_graphics = false;
	if (!line)
		return false;

	// Accept full log lines ("... key=c1:... ...") as well as bare keys.
	const char *p = strstr(line, "key=");
	p = p ? p + 4 : line;
	while (*p == ' ' || *p == '\t')
		p++;

	if (strncmp(p, "g1:", 3) == 0)
	{
		is_graphics = true;
		return false;
	}
	if (strncmp(p, "c1:", 3) != 0)
		return false;
	p += 3;

	PgsVariantKey k;
	if (!parse_u64_hex(p, k.prog_hash) || *p != ':')
		return false;
	p++;
	if (!parse_u32(p, k.spec_mask, 16, 0xff) || *p != ':')
		return false;
	p++;
	if (!parse_spec_list(p, k.specs, 8, k.spec_mask) || *p != ':')
		return false;
	p++;
	if (!parse_u32(p, k.ispec_mask, 16, 0xf) || *p != ':')
		return false;
	p++;
	if (!parse_spec_list(p, k.ispecs, 4, k.ispec_mask) || *p != ':')
		return false;
	p++;
	if (!parse_u32(p, k.sg_control, 10, 1) || *p != ':')
		return false;
	p++;
	if (!parse_u32(p, k.sg_min_log2, 10, 7) || *p != ':')
		return false;
	p++;
	if (!parse_u32(p, k.sg_max_log2, 10, 7) || *p != ':')
		return false;
	p++;
	if (!parse_u32(p, k.sg_full_group, 10, 1) || *p != ':')
		return false;
	p++;
	if (!parse_u32(p, k.robust, 10, 1))
		return false;
	// A full log line continues with " ms=..."; a bare key ends here
	// (trailing whitespace/newline is fine).
	if (*p == ' ' || *p == '\t')
	{
		const char *q = p;
		while (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n')
			q++;
		if (*q != '\0' && strncmp(q, "ms=", 3) != 0)
			return false;
	}
	else if (*p != '\r' && *p != '\n' && *p != '\0')
		return false;

	key = k;
	return true;
}

} // namespace ParallelGS
