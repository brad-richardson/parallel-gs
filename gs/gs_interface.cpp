// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#include "page_tracker.hpp"
#include "gs_interface.hpp"
#include "gs_util.hpp"
#include "shaders/swizzle_utils.h"
#include "muglm/muglm_impl.hpp"
#include "gs_registers_debug.hpp"
#include <cstdlib> // G40: std::getenv

namespace ParallelGS
{
// G40 local diagnostic (not upstream): sub-vsync A->B observation wall at
// boundary 1 (composite-flush ordinal 1 == pass 0, vsync #1, phase 0).
// Env-gated (PGS_G40_WALL=1), default silent, observation-only: kick/tex
// counters + LOGI + VRAM/staging reads + ONE forced submit+wait per probe
// firing (composite ordinals {0,1}; ordinal 0 is the load-control).
// Unset: one cached getenv branch per hook, zero lines, zero submits.
static bool g40_enabled()
{
	static bool en = std::getenv("PGS_G40_WALL") != nullptr;
	return en;
}
static unsigned g40_next_flush_ord = 0;
static unsigned g40_next_comp_ord = 0;
static bool g40_busy = false;
static unsigned g40_k_seen = 0;
static unsigned g40_k_adc = 0;
static unsigned g40_k_deg = 0;
static unsigned g40_k_bb = 0;
static unsigned g40_k_fuse = 0;
static unsigned g40_k_acc = 0;
static unsigned g40_tex_reuse = 0;
static bool g40_tex_created = false;
static uint64_t g40_tex_hash = 0;
static unsigned g40_tex_tbp0 = 0;
static unsigned g40_tex_tbw = 0;
static unsigned g40_tex_psm = 0;
static unsigned g40_tex_levels = 0;
static unsigned g40_tex_samples = 0;
static bool g40_tex_longterm = false;
static bool g40_last_valid = false;
static unsigned g40_last_ford = 0;
static uint64_t g40_last_f = 0;
static uint32_t g40_last_z = 0;
static uint8_t g40_last_head[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
static void g40_fnv(const uint8_t *p, size_t n, uint64_t &fnv, uint32_t &nz)
{
	fnv = 1469598103934665603ull; // G29-E1 truncated basis (comparability)
	nz = 0;
	for (size_t i = 0; i < n; i++)
	{
		fnv ^= p[i];
		fnv *= 1099511628211ull;
		nz += p[i] != 0;
	}
}
static const char *g40_reason_str(FlushReason r)
{
	switch (r)
	{
	case FlushReason::FBPointer:
		return "FBPointer";
	case FlushReason::Overflow:
		return "Overflow";
	case FlushReason::TextureHazard:
		return "TextureHazard";
	case FlushReason::CopyHazard:
		return "CopyHazard";
	case FlushReason::SubmissionFlush:
		return "SubmissionFlush";
	case FlushReason::PressureFlush:
		return "PressureFlush";
	case FlushReason::HostAccess:
		return "HostAccess";
	default:
		return "?";
	}
}
// G41 local diagnostic (not upstream): masked-write canary at boundary 1
// (composite-flush ordinal 1) + per-flush instance census (scratch-overlap
// proof) + O4m hash-input breakdown. Env-gated (PGS_G41_CANARY=1), default
// silent. The canary kicks small synthetic sprites into scratch VRAM pages
// (492..511, never B) through the live draw path, reads the arena back
// after a forced submit+wait, then RESTORES the saved bytes, so the game
// observes zero perturbation (G31 / G30-vpage / scanouts expected 0-diff).
static bool g41_enabled()
{
	static bool en = std::getenv("PGS_G41_CANARY") != nullptr;
	return en;
}
static unsigned g41_next_ford = 0;
static unsigned g41_next_comp_ord = 0;
static bool g41_busy = false;
enum { G41_ARENA_BASE_PAGE = 492, G41_ARENA_NPAGES = 20 };
static void g41_fnv(const uint8_t *p, size_t n, uint64_t &fnv, uint32_t &nz)
{
	fnv = 1469598103934665603ull; // G29-E1 truncated basis (comparability)
	nz = 0;
	for (size_t i = 0; i < n; i++)
	{
		fnv ^= p[i];
		fnv *= 1099511628211ull;
		nz += p[i] != 0;
	}
}
// G41-R1: kick-time composite register snapshot (C5 replays the composite's
// exact draw state; live regs at flush-end already belong to the next pass).
static RegisterState g41_snap_regs = {};
static bool g41_snap_valid = false;
GSInterface::GSInterface()
	: tracker(*this), renderer(tracker)
{
	setup_handlers();
	reset_context_state_registers();

	// Ensure that default states will trigger a dirty flag.
	render_pass.instances[0].frame.desc.PSM = 0x3f;
	render_pass.instances[0].zbuf.desc.PSM = 0x7;
}

void GSInterface::reset_context_state()
{
	flush();
	reset_context_state_registers();
}

void GSInterface::reset_context_state_registers()
{
	registers = {};
	for (auto &p : paths)
		p = {};
	reset_vertex_queue();
	clobber_register_state();
}

bool GSInterface::init(Vulkan::Device *device, const GSOptions &options)
{
	vram_size = options.vram_size;
	uint32_t num_pages = vram_size / PageSize;
	tracker.set_num_pages(num_pages);
	uint32_t num_pages_u32 = (num_pages + 31) / 32;
	sync_host_vram_blocks.resize(num_pages * PGS_BLOCKS_PER_PAGE / 32);
	sync_vram_host_pages.resize(num_pages_u32);
	block_buffer.reserve(num_pages_u32);

	if (!renderer.init(device, options))
		return false;

	set_super_sampling_rate(options.super_sampling,
	                        options.ordered_super_sampling,
	                        options.super_sampled_textures);

	renderer.reserve_primitive_buffers(MaxPrimitivesPerFlush);
	render_pass.positions = renderer.get_reserved_vertex_positions();
	render_pass.attributes = renderer.get_reserved_vertex_attributes();
	render_pass.prim = renderer.get_reserved_primitive_attributes();
	return true;
}

void GSInterface::set_super_sampling_rate(SuperSampling super_sampling,
                                          bool ordered_grid, bool super_sampled_textures_)
{
	super_sampled_textures = super_sampled_textures_;
	super_sampling = SuperSampling(std::min<uint32_t>(
			uint32_t(super_sampling), uint32_t(renderer.get_max_supported_super_sampling())));

	switch (super_sampling)
	{
	case SuperSampling::X1:
		sampling_rate_x_log2 = 0;
		sampling_rate_y_log2 = 0;
		super_sampled_textures = false;
		break;

	case SuperSampling::X2:
		sampling_rate_x_log2 = 0;
		sampling_rate_y_log2 = 1;
		break;

	case SuperSampling::X4:
		if (ordered_grid)
		{
			sampling_rate_x_log2 = 1;
			sampling_rate_y_log2 = 1;
		}
		else
		{
			sampling_rate_x_log2 = 0;
			sampling_rate_y_log2 = 2;
		}
		break;

	case SuperSampling::X8:
		sampling_rate_x_log2 = 1;
		sampling_rate_y_log2 = 2;
		break;

	case SuperSampling::X16:
		if (ordered_grid)
		{
			sampling_rate_x_log2 = 2;
			sampling_rate_y_log2 = 2;
		}
		else
		{
			sampling_rate_x_log2 = 1;
			sampling_rate_y_log2 = 3;
		}
		break;
	}

	renderer.invalidate_super_sampling_state(sampling_rate_x_log2, sampling_rate_y_log2);
}

static bool write_mask_is_16bit_channel_slice(uint32_t psm, uint32_t color_mask)
{
	// Only accept all or nothing masking, which is probably a case where game does not intend channel shuffle.
	if (psm != PSMCT32 && psm != PSMCT24 && psm != PSMZ32 && psm != PSMZ24)
	{
		uint32_t fb_mask = color_mask & 0xf8f8f8;
		if (fb_mask != 0xf8f8f8 && fb_mask != 0)
			return true;
	}

	return false;
}

void GSInterface::flush_render_pass(FlushReason reason)
{
	// G40 wall entry: snapshot only (no map: a pre-read would recurse into
	// mark_submission_timeline and consume the pending prims in an inner
	// flush. O2 pre-bytes come from the O2x post-flush chain, see end.).
	// Inner (recursive) calls see g40_busy and skip all G40 work.
	bool g40_outer = !g40_busy && g40_enabled();
	unsigned g40_ford = 0;
	bool g40_is_comp = false;
	int g40_cord = -1;
	unsigned g40_entry_prims = 0;
	Vulkan::ImageHandle g40_tex_hold;
	uint32_t g40_tex_w = 0;
	uint32_t g40_tex_h = 0;
	if (g40_outer)
	{
		g40_ford = g40_next_flush_ord++;
		g40_entry_prims = render_pass.primitive_count;
		if (render_pass.primitive_count)
		{
			for (uint32_t g40_i = 0; g40_i < render_pass.num_instances; g40_i++)
			{
				if (render_pass.instances[g40_i].frame.desc.FBP == 112)
					g40_is_comp = true;
			}
		}
		if (g40_is_comp)
		{
			g40_busy = true;
			g40_cord = int(g40_next_comp_ord++);
			if (!render_pass.held_images.empty() && !render_pass.tex_infos.empty() && render_pass.held_images[0])
			{
				g40_tex_hold = render_pass.held_images[0];
				g40_tex_w = render_pass.held_images[0]->get_width();
				g40_tex_h = render_pass.held_images[0]->get_height();
			}
		}
	}
	// G41 entry: independent composite ordinal + per-flush instance census
	// (FRAME+Z identity, bb page spans, scratch-arena overlap flags).
	// Inner (recursive) calls see g41_busy and skip all G41 work.
	bool g41_outer = !g41_busy && g41_enabled();
	bool g41_is_comp = false;
	int g41_cord = -1;
	unsigned g41_ford = 0;
	unsigned g41_entry_prims = 0;
	bool g41_has_texinfo = false;
	TexInfo g41_texinfo = {};
	unsigned g41_tex_count = 0;
	if (g41_outer)
	{
		g41_ford = g41_next_ford++;
		g41_entry_prims = render_pass.primitive_count;
		if (render_pass.primitive_count)
		{
			for (uint32_t g41_i = 0; g41_i < render_pass.num_instances; g41_i++)
			{
				if (render_pass.instances[g41_i].frame.desc.FBP == 112)
					g41_is_comp = true;
			}
			for (uint32_t g41_i = 0; g41_i < render_pass.num_instances; g41_i++)
			{
				const auto &g41_inst = render_pass.instances[g41_i];
				if (g41_inst.bb.z < 0 || g41_inst.bb.w < 0)
					continue;
				int g41_bx0 = g41_inst.bb.x < 0 ? 0 : g41_inst.bb.x;
				int g41_by0 = g41_inst.bb.y < 0 ? 0 : g41_inst.bb.y;
				int g41_bx1 = g41_inst.bb.z < 0 ? 0 : g41_inst.bb.z;
				int g41_by1 = g41_inst.bb.w < 0 ? 0 : g41_inst.bb.w;
				unsigned g41_fbp = g41_inst.frame.desc.FBP;
				unsigned g41_fbw = g41_inst.frame.desc.FBW;
				unsigned g41_stride = g41_fbw ? g41_fbw : 1;
				unsigned g41_flo = g41_fbp + unsigned(g41_bx0 >> int(g41_inst.fb_page_width_log2)) +
				                   unsigned(g41_by0 >> int(g41_inst.fb_page_height_log2)) * g41_stride;
				unsigned g41_fhi = g41_fbp + unsigned(g41_bx1 >> int(g41_inst.fb_page_width_log2)) +
				                   unsigned(g41_by1 >> int(g41_inst.fb_page_height_log2)) * g41_stride;
				unsigned g41_zbp = g41_inst.zbuf.desc.ZBP;
				unsigned g41_zlo = g41_zbp + unsigned(g41_bx0 >> int(g41_inst.z_page_width_log2)) +
				                   unsigned(g41_by0 >> int(g41_inst.z_page_height_log2)) * g41_stride;
				unsigned g41_zhi = g41_zbp + unsigned(g41_bx1 >> int(g41_inst.z_page_width_log2)) +
				                   unsigned(g41_by1 >> int(g41_inst.z_page_height_log2)) * g41_stride;
				unsigned g41_fov = (g41_flo < 512 && g41_fhi >= 492) ? 1 : 0;
				unsigned g41_zov = (g41_inst.z_write && g41_zlo < 512 && g41_zhi >= 492) ? 1 : 0;
				LOGI("G41: census ford=%u prims=%u inst=%u/%u FBP=%u FBW=%u PSM=%u MSK=%08x ZBP=%u ZPSM=%u ZMSK=%u zwr=%u zsen=%u cwm=%08x bb=%d,%d,%d,%d fpg=%u..%u zpg=%u..%u fov=%u zov=%u reason=%s.\n",
				     g41_ford, g41_entry_prims, g41_i, render_pass.num_instances,
				     g41_fbp, g41_fbw, (unsigned)g41_inst.frame.desc.PSM,
				     (unsigned)g41_inst.frame.desc.FBMSK,
				     g41_zbp, (unsigned)g41_inst.zbuf.desc.PSM,
				     (unsigned)g41_inst.zbuf.desc.ZMSK,
				     (unsigned)g41_inst.z_write, (unsigned)g41_inst.z_sensitive,
				     (unsigned)g41_inst.color_write_mask,
				     g41_inst.bb.x, g41_inst.bb.y, g41_inst.bb.z, g41_inst.bb.w,
				     g41_flo, g41_fhi, g41_zlo, g41_zhi, g41_fov, g41_zov,
				     g40_reason_str(reason));
			}
		}
		if (g41_is_comp)
		{
			g41_busy = true;
			g41_cord = int(g41_next_comp_ord++);
			if (!render_pass.tex_infos.empty())
			{
				g41_texinfo = render_pass.tex_infos[0].info;
				g41_has_texinfo = true;
				g41_tex_count = (unsigned)render_pass.tex_infos.size();
			}
		}
	}
	ParallelGS::RenderPass rp = {};

	if (render_pass.primitive_count)
	{
		rp.num_primitives = render_pass.primitive_count;

		rp.states = render_pass.state_vectors.data();
		rp.num_states = render_pass.state_vectors.size();
		rp.allow_blend_demote = hacks.allow_blend_demote;

		rp.textures = render_pass.tex_infos.data();
		rp.num_textures = render_pass.tex_infos.size();

		uint32_t binning_cost = 0;

		for (uint32_t i = 0; i < render_pass.num_instances; i++)
		{
			auto &inst = render_pass.instances[i];
			uint32_t tile_width = ((inst.bb.z - inst.bb.x) >> PGS_FB_SWIZZLE_WIDTH_LOG2) + 1;
			uint32_t tile_height = ((inst.bb.w - inst.bb.y) >> PGS_FB_SWIZZLE_HEIGHT_LOG2) + 1;
			binning_cost += tile_width * tile_height * rp.num_primitives;
		}

		// Somewhat arbitrary. Try to balance binning load.
		if (binning_cost < 10 * 1000)
			rp.coarse_tile_size_log2 = 3;
		else if (binning_cost < 10 * 1000 * 1000)
			rp.coarse_tile_size_log2 = 4;
		else if (binning_cost < 100 * 1000 * 1000)
			rp.coarse_tile_size_log2 = 5;
		else
			rp.coarse_tile_size_log2 = 6;

		if (sampling_rate_y_log2 != 0 && rp.coarse_tile_size_log2 > 3)
			rp.coarse_tile_size_log2 -= 1;

		for (uint32_t i = 0; i < render_pass.num_instances; i++)
		{
			auto &inst = render_pass.instances[i];
			uint32_t coarse_tiles_width = ((inst.bb.z - inst.bb.x) >> rp.coarse_tile_size_log2) + 1;
			uint32_t coarse_tiles_height = ((inst.bb.w - inst.bb.y) >> rp.coarse_tile_size_log2) + 1;

			// Try to avoid overflowing the 64 MiB sub-allocation limit in Granite when
			// allocating binning list.
			// Just a mild performance optimization, will still work without this heuristic.
			// TODO: Maybe expose something in Granite to make this less hard-coded.
			VkDeviceSize primitive_list_size = coarse_tiles_width * coarse_tiles_height * rp.num_primitives * sizeof(uint16_t);
			while (primitive_list_size > 48 * 1024 * 1024)
			{
				rp.coarse_tile_size_log2 += 1;
				primitive_list_size /= 4;
			}
		}

		rp.num_instances = render_pass.num_instances;

		// It's possible the last RP instance was added, but there are no primitives yet, since
		// we ended up flushing before we could expand the BB.
		if (render_pass.instances[rp.num_instances - 1].bb.z < 0)
			rp.num_instances--;
		assert(rp.num_instances);

		for (uint32_t i = 0; i < rp.num_instances; i++)
		{
			auto &inst = render_pass.instances[i];
			auto &pass = rp.instances[i];

			assert(inst.bb.x <= inst.bb.z && inst.bb.y <= inst.bb.w);
			assert(inst.bb.x >= 0 && inst.bb.y >= 0);
			assert(inst.bb.z < 2048 && inst.bb.w < 2048);

			pass.fb.frame = inst.frame;
			pass.fb.z = inst.zbuf;
			assert(inst.bb.z < std::max<int>(1, pass.fb.frame.desc.FBW) * PGS_BUFFER_WIDTH_SCALE);

			pass.base_x = inst.bb.x;
			pass.base_y = inst.bb.y;
			pass.coarse_tiles_width = ((inst.bb.z - inst.bb.x) >> rp.coarse_tile_size_log2) + 1;
			pass.coarse_tiles_height = ((inst.bb.w - inst.bb.y) >> rp.coarse_tile_size_log2) + 1;

			// This should be possible to vary based on dynamic usage.
			// If there are only trivial UI passes, we should make it single-sampled.
			pass.sampling_rate_x_log2 = sampling_rate_x_log2;
			pass.sampling_rate_y_log2 = sampling_rate_y_log2;

			// Any FBMASK that masks more than the global mask must be demoted from OPAQUE.
			pass.opaque_fbmask = ~inst.color_write_mask;
			pass.channel_shuffle = inst.has_channel_shuffle ||
			                       write_mask_is_16bit_channel_slice(inst.frame.desc.PSM, inst.color_write_mask);

			// If we're super sampling textures, we can avoid a ton of common SSAA issues which
			// arise when doing single-sampled textures resolving on top of super-sampled framebuffer data.
			// If we're rendering field aware upscaling, we essentially need to force super-sampling everywhere all the time.
			if (!render_pass.field_aware_rendering &&
			    (!super_sampled_textures || !render_pass.tex_infos_has_super_samples))
			{
				// This case is to handle certain channel shuffling effects which render with 16-bit over a 32-bit FB
				// using 0x3fff FBMSK. This ends up slicing the green channel and trying to resolve super-sampling in 16-bit
				// domain leads to bogus results.
				// If a channel is considered "odd" w.r.t. masking, force single-sampled rendering.
				// Don't apply this fixup for 24/32-bit bpp, since there are no reasonable shuffle effects
				// that operate on those bit-depths. Try to avoid false positives.
				if ((sampling_rate_x_log2 || sampling_rate_y_log2) && pass.channel_shuffle)
				{
					pass.sampling_rate_x_log2 = 0;
					pass.sampling_rate_y_log2 = 0;
				}
				else if (inst.z_feedback)
				{
					// If we're doing Z-feedback effects, it's not safe to run super-sampled since there are too many glitches in play
					// for it to be viable. When super-sampled Z is converted to color, then downsampled, there will be bleeding
					// across geometry, and we have no way of resolving this other than forwarding super-sampled textures
					// everywhere.
					pass.sampling_rate_x_log2 = 0;
					pass.sampling_rate_y_log2 = 0;
				}
				else if (inst.z_write && inst.zbuf.desc.ZBP == inst.frame.desc.FBP)
				{
					// If we're doing color/Z aliasing like this, we're not doing normal rendering,
					// and any super sampling state is likely to get clobbered hard either way.
					// There's no useful use case for rendering 3D with this configuration,
					// so be careful and just disable SSAA.
					pass.sampling_rate_x_log2 = 0;
					pass.sampling_rate_y_log2 = 0;
				}
			}

			pass.z_sensitive = inst.z_sensitive;
			pass.z_write = inst.z_write;
		}

		rp.feedback_mode = render_pass.feedback_mode;
		rp.feedback_texture_psm = render_pass.feedback_psm;
		rp.feedback_texture_cpsm = render_pass.feedback_cpsm;

		// Affects shader variants.
		rp.has_aa1 = render_pass.has_aa1;
		rp.has_scanmsk = render_pass.has_scanmsk;

		// Debug stuff
		rp.feedback_color = debug_mode.feedback_render_target;

		if (debug_mode.feedback_render_target)
			for (uint32_t i = 0; i < render_pass.num_instances && !rp.feedback_depth; i++)
				rp.feedback_depth = render_pass.instances[i].z_sensitive;

		switch (debug_mode.draw_mode)
		{
		case DebugMode::DrawDebugMode::Strided:
			// Try to balance debuggability so there's not a million events to step through
			// while being able to identify a faulty primitive.
			rp.debug_capture_stride = 16;
			break;

		case DebugMode::DrawDebugMode::Full:
			rp.debug_capture_stride = 1;
			break;

		default:
			break;
		}

		rp.label_key = render_pass.label_key++;
		rp.flush_reason = reason;
		//////

		{
			// G11b local experiment hook (not upstream): recorded-batch identity.
			LOGI("G11: record prims=%u inst=%u tex=%u tex0=%u\n", rp.num_primitives,
			     rp.num_instances, rp.num_textures, (unsigned)render_pass.tex0_infos.size());
			for (uint32_t g11_i = 0; g11_i < rp.num_instances; g11_i++)
				LOGI("G11:   inst %u FRAME FBP=%u FBW=%u PSM=%u\n", g11_i,
				     (unsigned)rp.instances[g11_i].fb.frame.desc.FBP,
				     (unsigned)rp.instances[g11_i].fb.frame.desc.FBW,
				     (unsigned)rp.instances[g11_i].fb.frame.desc.PSM);
			for (uint32_t g11_j = 0; g11_j < render_pass.tex0_infos.size(); g11_j++)
				LOGI("G11:   tex %u TBP0=%u TBW=%u TPSM=%u CBP=%u\n", g11_j,
				     (unsigned)render_pass.tex0_infos[g11_j].TBP0,
				     (unsigned)render_pass.tex0_infos[g11_j].TBW,
				     (unsigned)render_pass.tex0_infos[g11_j].PSM,
				     (unsigned)render_pass.tex0_infos[g11_j].CBP);
		}
		renderer.flush_rendering(rp);

		// Need to call this after flush rendering to ensure that the images
		// have been cached properly.
		promote_render_pass_to_backbuffer(rp);

		TRACE_HEADER("FLUSH RENDER", rp);
	}

	// If we used memoization of CLUT and ended on an old instance, we need to notify the renderer
	// so it uses the current index as read input to future updates.
	renderer.rewind_clut_instance(render_pass.clut_instance);
	render_pass.latest_clut_instance = render_pass.clut_instance;

	render_pass.held_images.clear();
	render_pass.texture_map.clear();
	render_pass.tex_infos.clear();
	render_pass.tex_infos_has_super_samples = false;
	render_pass.tex0_infos.clear();
	render_pass.state_vector_map.clear();
	render_pass.state_vectors.clear();
	render_pass.primitive_count = 0;
	render_pass.pending_palette_updates = 0;

	RenderPassState::Instance instance = render_pass.instances[render_pass.current_instance];

	render_pass.instances[0] = {};
	render_pass.num_instances = 1;
	render_pass.current_instance = 0;
	auto &inst = render_pass.instances[0];
	inst.frame = instance.frame;
	inst.zbuf = instance.zbuf;
	inst.fb_page_width_log2 = instance.fb_page_width_log2;
	inst.fb_page_height_log2 = instance.fb_page_height_log2;
	inst.z_page_width_log2 = instance.z_page_width_log2;
	inst.z_page_height_log2 = instance.z_page_height_log2;

	render_pass.last_triangle_is_parallelogram_candidate = false;
	render_pass.feedback_mode = RenderPass::Feedback::None;
	render_pass.has_aa1 = false;
	render_pass.has_scanmsk = false;
	render_pass.has_hazardous_short_term_texture_caching = false;
	render_pass.has_optimized_short_term_texture_caching = false;
	state_tracker.dirty_flags = STATE_DIRTY_ALL_BITS;
	//state_tracker.current_copy_cache_hazard_counter = 0;

	renderer.reserve_primitive_buffers(MaxPrimitivesPerFlush);
	render_pass.positions = renderer.get_reserved_vertex_positions();
	render_pass.attributes = renderer.get_reserved_vertex_attributes();
	render_pass.prim = renderer.get_reserved_primitive_attributes();
	if (g40_outer) // G40 wall end: O1/O5/O4m every flush, O3/O3b/O4 at cord {0,1}
	{
		LOGI("G40: O1 ford=%u cord=%d prims=%u submitted=%u inst=%u tex=%u seen=%u adc=%u deg=%u bb=%u fuse=%u acc=%u.\n",
		     g40_ford, g40_cord, g40_entry_prims, (unsigned)rp.num_primitives,
		     (unsigned)rp.num_instances, (unsigned)rp.num_textures,
		     g40_k_seen, g40_k_adc, g40_k_deg, g40_k_bb, g40_k_fuse, g40_k_acc);
		g40_k_seen = 0;
		g40_k_adc = 0;
		g40_k_deg = 0;
		g40_k_bb = 0;
		g40_k_fuse = 0;
		g40_k_acc = 0;
		if (g40_is_comp)
		{
			LOGI("G40: O5 ford=%u cord=%d reason=%s label=%u states=%u tex=%u fbmode=%u tilelog2=%u.\n",
			     g40_ford, g40_cord, g40_reason_str(reason), (unsigned)rp.label_key,
			     (unsigned)rp.num_states, (unsigned)rp.num_textures, (unsigned)rp.feedback_mode,
			     (unsigned)rp.coarse_tile_size_log2);
			for (uint32_t g40_i = 0; g40_i < rp.num_instances; g40_i++)
			{
				LOGI("G40: O5 ford=%u cord=%d inst=%u FBP=%u FBW=%u PSM=%u FBMSK=%08x ZBP=%u ZMSK=%u opq=%08x chshuf=%u zsens=%u zwr=%u ssx=%u ssy=%u base=%u,%u ctiles=%ux%u.\n",
				     g40_ford, g40_cord, g40_i,
				     (unsigned)rp.instances[g40_i].fb.frame.desc.FBP,
				     (unsigned)rp.instances[g40_i].fb.frame.desc.FBW,
				     (unsigned)rp.instances[g40_i].fb.frame.desc.PSM,
				     (unsigned)rp.instances[g40_i].fb.frame.desc.FBMSK,
				     (unsigned)rp.instances[g40_i].fb.z.desc.ZBP,
				     (unsigned)rp.instances[g40_i].fb.z.desc.ZMSK,
				     (unsigned)rp.instances[g40_i].opaque_fbmask,
				     (unsigned)rp.instances[g40_i].channel_shuffle,
				     (unsigned)rp.instances[g40_i].z_sensitive,
				     (unsigned)rp.instances[g40_i].z_write,
				     (unsigned)rp.instances[g40_i].sampling_rate_x_log2,
				     (unsigned)rp.instances[g40_i].sampling_rate_y_log2,
				     (unsigned)rp.instances[g40_i].base_x, (unsigned)rp.instances[g40_i].base_y,
				     (unsigned)rp.instances[g40_i].coarse_tiles_width,
				     (unsigned)rp.instances[g40_i].coarse_tiles_height);
			}
			LOGI("G40: O4m ford=%u cord=%d texdims=%ux%u created=%u longterm=%u tbp0=%u tbw=%u psm=%u lv=%u smp=%u reuse=%u hash=%016llx.\n",
			     g40_ford, g40_cord, g40_tex_w, g40_tex_h,
			     (unsigned)g40_tex_created, (unsigned)g40_tex_longterm,
			     g40_tex_tbp0, g40_tex_tbw, g40_tex_psm, g40_tex_levels, g40_tex_samples,
			     g40_tex_reuse, (unsigned long long)g40_tex_hash);
			g40_tex_reuse = 0;
			if (g40_cord == 0 || g40_cord == 1)
			{
				// G40 O2: literal pre-composite B from the O2x chain (B is
				// untouched between flushes: scenes are single-inst FBP0
				// (G11) and vpage excludes 112..223 except via composite).
				if (g40_last_valid)
				{
					LOGI("G40: O2 ford=%u cord=%d B pre-fnv=%016llx nz=%u head=%02x%02x%02x%02x%02x%02x%02x%02x srcford=%u.\n",
					     g40_ford, g40_cord, (unsigned long long)g40_last_f, (unsigned)g40_last_z,
					     g40_last_head[0], g40_last_head[1], g40_last_head[2], g40_last_head[3],
					     g40_last_head[4], g40_last_head[5], g40_last_head[6], g40_last_head[7],
					     g40_last_ford);
				}
				else
				{
					LOGI("G40: O2 ford=%u cord=%d B pre=load (no prior flush; eea04488c453e75b/149721 receipted).\n",
					     g40_ford, g40_cord);
				}
				// G40 probe: record GPU-side copies, force full submit + wait,
				// then read host B (O3), gpu B (O3b) and texture bytes (O4).
				bool g40_rec = renderer.g40_record_probes(g40_tex_hold.get(), g40_tex_w, g40_tex_h);
				uint64_t g40_tl = tracker.mark_submission_timeline(FlushReason::HostAccess);
				renderer.flush_submit(g40_tl);
				renderer.wait_timeline(g40_tl);
				{
					const size_t g40_base = 112 * 8192;
					const size_t g40_n = 112 * 8192;
					const uint8_t *g40_vram = static_cast<const uint8_t *>(map_vram_read(g40_base, g40_n));
					if (g40_vram)
					{
						uint64_t g40_f = 0;
						uint32_t g40_z = 0;
						g40_fnv(g40_vram, g40_n, g40_f, g40_z);
						LOGI("G40: O3 ford=%u cord=%d rec=%u B post-fnv=%016llx nz=%u head=%02x%02x%02x%02x%02x%02x%02x%02x.\n",
						     g40_ford, g40_cord, (unsigned)g40_rec,
						     (unsigned long long)g40_f, (unsigned)g40_z,
						     g40_vram[0], g40_vram[1], g40_vram[2], g40_vram[3],
						     g40_vram[4], g40_vram[5], g40_vram[6], g40_vram[7]);
					}
					else
					{
						LOGI("G40: O3 ford=%u cord=%d rec=%u map failed.\n",
						     g40_ford, g40_cord, (unsigned)g40_rec);
					}
				}
				{
					uint64_t g40_tfnv = 0;
					uint64_t g40_gfnv = 0;
					uint32_t g40_tnz = 0;
					uint32_t g40_gnz = 0;
					uint8_t g40_thead[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
					uint8_t g40_ghead[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
					bool g40_fin = renderer.g40_finish_probes(g40_tfnv, g40_tnz, g40_thead,
					                                          g40_gfnv, g40_gnz, g40_ghead);
					LOGI("G40: O3b ford=%u cord=%d fin=%u gpuB-fnv=%016llx nz=%u head=%02x%02x%02x%02x%02x%02x%02x%02x.\n",
					     g40_ford, g40_cord, (unsigned)g40_fin,
					     (unsigned long long)g40_gfnv, (unsigned)g40_gnz,
					     g40_ghead[0], g40_ghead[1], g40_ghead[2], g40_ghead[3],
					     g40_ghead[4], g40_ghead[5], g40_ghead[6], g40_ghead[7]);
					LOGI("G40: O4 ford=%u cord=%d fin=%u tex-fnv=%016llx nz=%u head=%02x%02x%02x%02x%02x%02x%02x%02x dims=%ux%u.\n",
					     g40_ford, g40_cord, (unsigned)g40_fin,
					     (unsigned long long)g40_tfnv, (unsigned)g40_tnz,
					     g40_thead[0], g40_thead[1], g40_thead[2], g40_thead[3],
					     g40_thead[4], g40_thead[5], g40_thead[6], g40_thead[7],
					     g40_tex_w, g40_tex_h);
				}
			}
			g40_busy = false;
		}
		else if (g40_entry_prims > 0)
		{
			// G40 O2x: inter-flush B at scene-flush end (safe: prims reset;
			// busy-guarded against recursion). Pure read expected (A-only
			// work leaves no B writes pending); the bytes self-report.
			g40_busy = true;
			{
				const size_t g40_base = 112 * 8192;
				const size_t g40_n = 112 * 8192;
				const uint8_t *g40_vram = static_cast<const uint8_t *>(map_vram_read(g40_base, g40_n));
				if (g40_vram)
				{
					uint64_t g40_f = 0;
					uint32_t g40_z = 0;
					g40_fnv(g40_vram, g40_n, g40_f, g40_z);
					g40_last_valid = true;
					g40_last_ford = g40_ford;
					g40_last_f = g40_f;
					g40_last_z = g40_z;
					for (unsigned g40_b = 0; g40_b < 8; g40_b++)
						g40_last_head[g40_b] = g40_vram[g40_b];
					LOGI("G40: O2x ford=%u B postflush-fnv=%016llx nz=%u head=%02x%02x%02x%02x%02x%02x%02x%02x.\n",
					     g40_ford, (unsigned long long)g40_f, (unsigned)g40_z,
					     g40_vram[0], g40_vram[1], g40_vram[2], g40_vram[3],
					     g40_vram[4], g40_vram[5], g40_vram[6], g40_vram[7]);
				}
				else
				{
					LOGI("G40: O2x ford=%u map failed.\n", g40_ford);
				}
			}
			g40_busy = false;
		}
	}
	if (g41_outer) // G41 end: cord join + (at cord 1) hash breakdown + canary
	{
		if (g41_is_comp)
		{
			int g41_fire = g40_outer ? g40_cord : g41_cord;
			auto g41_dump_state = [this](const char *g41_tag, int g41_cord_tag) {
				LOGI("G41: st %s cord=%d ofx=%d ofy=%d scilo=%d,%d scihi=%d,%d scixfb=%d wrap=%u dirty=%08x ss=%u,%u prim=%016llx frame0=%016llx scis0=%016llx xyof0=%016llx test0=%016llx zbuf0=%016llx tex0_0=%016llx alpha0=%016llx ac=%u.\n",
				     g41_tag, g41_cord_tag,
				     render_pass.ofx, render_pass.ofy,
				     render_pass.scissor_lo.x, render_pass.scissor_lo.y,
				     render_pass.scissor_hi.x, render_pass.scissor_hi.y,
				     render_pass.scissor_hi_x_fb, (unsigned)render_pass.can_fb_wraparound,
				     (unsigned)state_tracker.dirty_flags,
				     sampling_rate_x_log2, sampling_rate_y_log2,
				     (unsigned long long)registers.prim.bits,
				     (unsigned long long)registers.ctx[0].frame.bits,
				     (unsigned long long)registers.ctx[0].scissor.bits,
				     (unsigned long long)registers.ctx[0].xyoffset.bits,
				     (unsigned long long)registers.ctx[0].test.bits,
				     (unsigned long long)registers.ctx[0].zbuf.bits,
				     (unsigned long long)registers.ctx[0].tex0.bits,
				     (unsigned long long)registers.ctx[0].alpha.bits,
				     (unsigned)registers.prmodecont.desc.AC);
			};
			LOGI("G41: cord g41=%d g40=%d ford=%u prims=%u tex=%u.\n",
			     g41_cord, g40_cord, g41_ford, g41_entry_prims, g41_tex_count);
			g41_dump_state("cord", g41_fire);
			if (g41_fire == 1)
			{
				{
					// G41 (b): O4m hash-input breakdown at cord 1. The last
					// texture resolve before this flush is the composite's
					// (single-texture pass, tex=1); recompute the hash from
					// the same 8 inputs to prove the breakdown is complete.
					const TextureDescriptor &g41_d = state_tracker.last_texture_descriptor;
					Util::Hasher g41_h;
					g41_h.u64(g41_d.tex0.bits);
					g41_h.u64(g41_d.tex1.bits);
					g41_h.u64(g41_d.texa.bits);
					g41_h.u64(g41_d.miptbp1_3.bits);
					g41_h.u64(g41_d.miptbp4_6.bits);
					g41_h.u64(g41_d.clamp.bits);
					g41_h.u64(g41_d.palette_bank);
					g41_h.u32(g41_d.samples);
					unsigned g41_lctx = registers.prim.desc.CTXT & 1;
					LOGI("G41: hash inputs tex0=%016llx tex1=%016llx texa=%016llx mb13=%016llx mb46=%016llx clamp=%016llx bank=%u smp=%u recomp=%016llx TBP0=%u TBW=%u TPSM=%u.\n",
					     (unsigned long long)g41_d.tex0.bits,
					     (unsigned long long)g41_d.tex1.bits,
					     (unsigned long long)g41_d.texa.bits,
					     (unsigned long long)g41_d.miptbp1_3.bits,
					     (unsigned long long)g41_d.miptbp4_6.bits,
					     (unsigned long long)g41_d.clamp.bits,
					     g41_d.palette_bank, g41_d.samples,
					     (unsigned long long)g41_h.get(),
					     (unsigned)g41_d.tex0.desc.TBP0, (unsigned)g41_d.tex0.desc.TBW,
					     (unsigned)g41_d.tex0.desc.PSM);
					LOGI("G41: hash rect x=%u y=%u w=%u h=%u lv=%u ctx=%u rtex0=%016llx rtex1=%016llx rclamp=%016llx rmb13=%016llx rmb46=%016llx rtexa=%016llx.\n",
					     g41_d.rect.x, g41_d.rect.y, g41_d.rect.width, g41_d.rect.height,
					     g41_d.rect.levels, g41_lctx,
					     (unsigned long long)registers.ctx[g41_lctx].tex0.bits,
					     (unsigned long long)registers.ctx[g41_lctx].tex1.bits,
					     (unsigned long long)registers.ctx[g41_lctx].clamp.bits,
					     (unsigned long long)registers.ctx[g41_lctx].miptbl_1_3.bits,
					     (unsigned long long)registers.ctx[g41_lctx].miptbl_4_6.bits,
					     (unsigned long long)registers.texa.bits);
					if (g41_has_texinfo)
					{
						LOGI("G41: hash texinfo n=%u sizes=%.1f,%.1f,%.6f,%.6f region=%.1f,%.1f,%.1f,%.1f bias=%.6f,%.6f arrayed=%d flags=%d.\n",
						     g41_tex_count,
						     g41_texinfo.sizes.x, g41_texinfo.sizes.y,
						     g41_texinfo.sizes.z, g41_texinfo.sizes.w,
						     g41_texinfo.region.x, g41_texinfo.region.y,
						     g41_texinfo.region.z, g41_texinfo.region.w,
						     g41_texinfo.bias.x, g41_texinfo.bias.y,
						     g41_texinfo.arrayed, g41_texinfo.flags);
					}
					else
					{
						LOGI("G41: hash texinfo EMPTY n=%u.\n", g41_tex_count);
					}
				}
				{
					// G41 canary: 5 synthetic sprites (C1..C5) into scratch
					// arena pages 492..511 (never B) through the live draw
					// path. Per-cell register + VRAM save/restore; G40 rows
					// suppressed (g40_busy held) so game O-rows stay 0-diff.
					g40_busy = true;
					unsigned g41_s_kseen = g40_k_seen;
					unsigned g41_s_kadc = g40_k_adc;
					unsigned g41_s_kdeg = g40_k_deg;
					unsigned g41_s_kbb = g40_k_bb;
					unsigned g41_s_kfuse = g40_k_fuse;
					unsigned g41_s_kacc = g40_k_acc;
					unsigned g41_s_treuse = g40_tex_reuse;
					bool g41_s_tmade = g40_tex_created;
					uint64_t g41_s_thash = g40_tex_hash;
					unsigned g41_s_ttbp0 = g40_tex_tbp0;
					unsigned g41_s_ttbw = g40_tex_tbw;
					unsigned g41_s_tpsm = g40_tex_psm;
					unsigned g41_s_tlv = g40_tex_levels;
					unsigned g41_s_tsmp = g40_tex_samples;
					bool g41_s_tlt = g40_tex_longterm;
					VertexPosition g41_vq_pos[3];
					VertexAttribute g41_vq_attr[3];
					for (unsigned g41_vqi = 0; g41_vqi < 3; g41_vqi++)
					{
						g41_vq_pos[g41_vqi] = vertex_queue.pos[g41_vqi];
						g41_vq_attr[g41_vqi] = vertex_queue.attr[g41_vqi];
					}
					unsigned g41_vq_n = vertex_queue.count;
					StateTracker g41_saved_st = state_tracker;
					auto g41_saved_pf = render_pass.potential_feedback;
					auto g41_saved_porder = render_pass.last_triangle_parallelogram_order;
					ivec2 g41_saved_scilo = render_pass.scissor_lo;
					ivec2 g41_saved_scihi = render_pass.scissor_hi;
					int g41_saved_scixfb = render_pass.scissor_hi_x_fb;
					bool g41_saved_wrap = render_pass.can_fb_wraparound;
					g41_dump_state("cstart", 1);
					const size_t g41_abase = size_t(G41_ARENA_BASE_PAGE) * 8192;
					const size_t g41_an = size_t(G41_ARENA_NPAGES) * 8192;
					bool g41_pre_ok = false;
					std::vector<uint8_t> g41_pre(g41_an, 0);
					{
						const uint8_t *g41_a = static_cast<const uint8_t *>(map_vram_read(g41_abase, g41_an));
						if (!g41_a)
						{
							LOGI("G41: canary ABORT arena pre-read map failed.\n");
						}
						else
						{
							memcpy(g41_pre.data(), g41_a, g41_an);
							g41_pre_ok = true;
							uint64_t g41_pf = 0;
							uint32_t g41_pz = 0;
							g41_fnv(g41_pre.data(), g41_an, g41_pf, g41_pz);
							LOGI("G41: arena pre-fnv=%016llx nz=%u head=%02x%02x%02x%02x%02x%02x%02x%02x vq=%u.\n",
							     (unsigned long long)g41_pf, (unsigned)g41_pz,
							     g41_pre[0], g41_pre[1], g41_pre[2], g41_pre[3],
							     g41_pre[4], g41_pre[5], g41_pre[6], g41_pre[7],
							     g41_vq_n);
						}
					}
					static const unsigned g41_fbp[5] = { 492, 496, 500, 504, 508 };
					static const unsigned g41_psm[5] = { 0, 0, 1, 1, 1 };
					static const uint32_t g41_msk[5] = { 0, 0xff000000u, 0, 0xff000000u, 0xff000000u };
					for (unsigned g41_cell = 0; g41_cell < 5 && g41_pre_ok; g41_cell++)
					{
						bool g41_c5 = g41_cell == 4;
						unsigned g41_lctx = registers.prim.desc.CTXT & 1;
						RegisterAddr g41_a_frame = g41_lctx ? RegisterAddr::FRAME_2 : RegisterAddr::FRAME_1;
						RegisterAddr g41_a_zbuf = g41_lctx ? RegisterAddr::ZBUF_2 : RegisterAddr::ZBUF_1;
						RegisterAddr g41_a_test = g41_lctx ? RegisterAddr::TEST_2 : RegisterAddr::TEST_1;
						RegisterAddr g41_a_scis = g41_lctx ? RegisterAddr::SCISSOR_2 : RegisterAddr::SCISSOR_1;
						RegisterAddr g41_a_xyof = g41_lctx ? RegisterAddr::XYOFFSET_2 : RegisterAddr::XYOFFSET_1;
						RegisterAddr g41_a_fba = g41_lctx ? RegisterAddr::FBA_2 : RegisterAddr::FBA_1;
						uint64_t g41_s_prim = registers.prim.bits;
						uint64_t g41_s_rgbaq = registers.rgbaq.bits;
						uint64_t g41_s_st = registers.st.bits;
						uint64_t g41_s_uv = registers.uv.bits;
						uint64_t g41_s_frame = registers.ctx[g41_lctx].frame.bits;
						uint64_t g41_s_zbuf = registers.ctx[g41_lctx].zbuf.bits;
						uint64_t g41_s_test = registers.ctx[g41_lctx].test.bits;
						uint64_t g41_s_scis = registers.ctx[g41_lctx].scissor.bits;
						uint64_t g41_s_xyof = registers.ctx[g41_lctx].xyoffset.bits;
						uint64_t g41_s_fba = registers.ctx[g41_lctx].fba.bits;
						uint64_t g41_s_pabe = registers.pabe.bits;
						RegisterState g41_live_all = registers;
						bool g41_void = false;
						if (!g41_c5)
						{
							Reg64<PRIMBits> g41_p;
							g41_p.bits = 0;
							g41_p.desc.PRIM = 6;
							g41_p.desc.CTXT = g41_lctx;
							write_register(RegisterAddr::PRIM, g41_p.bits);
							write_register(RegisterAddr::PRMODE, g41_p.bits);
							Reg64<RGBAQBits> g41_c;
							g41_c.bits = 0;
							g41_c.desc.R = 0x11;
							g41_c.desc.G = 0x22;
							g41_c.desc.B = 0x33;
							g41_c.desc.A = 0x44;
							g41_c.desc.Q = 1.0f;
							write_register(RegisterAddr::RGBAQ, g41_c.bits);
							write_register(g41_a_xyof, uint64_t(0));
							Reg64<SCISSORBits> g41_sc;
							g41_sc.bits = 0;
							g41_sc.desc.SCAX0 = 0;
							g41_sc.desc.SCAY0 = 0;
							g41_sc.desc.SCAX1 = 63;
							g41_sc.desc.SCAY1 = 63;
							write_register(g41_a_scis, g41_sc.bits);
							write_register(g41_a_test, uint64_t(0));
							write_register(g41_a_fba, uint64_t(0));
							write_register(RegisterAddr::PABE, uint64_t(0));
							Reg64<FRAMEBits> g41_f;
							g41_f.bits = 0;
							g41_f.desc.FBP = g41_fbp[g41_cell];
							g41_f.desc.FBW = 8;
							g41_f.desc.PSM = g41_psm[g41_cell];
							g41_f.desc.FBMSK = g41_msk[g41_cell];
							write_register(g41_a_frame, g41_f.bits);
							Reg64<ZBUFBits> g41_z;
							g41_z.bits = 0;
							g41_z.desc.ZBP = 511;
							g41_z.desc.ZMSK = 1;
							write_register(g41_a_zbuf, g41_z.bits);
						}
						else
						{
							// G41-R1: C5 replays the composite's kick-time
							// registers (last FBP112 append), redirected to
							// scratch (FBP) + fixed geometry.
							unsigned g41_sctx = g41_snap_regs.prim.desc.CTXT & 1;
							if (!g41_snap_valid)
							{
								LOGI("G41: C5 NO-SNAPSHOT (void).\n");
								g41_void = true;
							}
							else if (g41_sctx != g41_lctx)
							{
								LOGI("G41: C5 ctxt-mismatch snap=%u live=%u (void).\n", g41_sctx, g41_lctx);
								g41_void = true;
							}
							else
							{
								const RegisterState &g41_src = g41_snap_regs;
								write_register(RegisterAddr::PRIM, g41_src.prim.bits);
								write_register(RegisterAddr::PRMODE, g41_src.prim.bits);
								Reg64<RGBAQBits> g41_rq;
								g41_rq.bits = g41_src.rgbaq.bits;
								g41_rq.desc.Q = 1.0f;
								write_register(RegisterAddr::RGBAQ, g41_rq.bits);
								write_register(RegisterAddr::ST, g41_src.st.bits);
								write_register(RegisterAddr::UV, g41_src.uv.bits);
								write_register(RegisterAddr::FOG, g41_src.fog.bits);
								write_register(RegisterAddr::PRMODECONT, g41_src.prmodecont.bits);
								write_register(RegisterAddr::TEXCLUT, g41_src.texclut.bits);
								write_register(RegisterAddr::TEXA, g41_src.texa.bits);
								write_register(RegisterAddr::FOGCOL, g41_src.fogcol.bits);
								write_register(RegisterAddr::DIMX, g41_src.dimx.bits);
								write_register(RegisterAddr::DTHE, g41_src.dthe.bits);
								write_register(RegisterAddr::COLCLAMP, g41_src.colclamp.bits);
								write_register(RegisterAddr::PABE, g41_src.pabe.bits);
								write_register(RegisterAddr::SCANMSK, g41_src.scanmsk.bits);
								write_register(g41_sctx ? RegisterAddr::TEX0_2 : RegisterAddr::TEX0_1, g41_src.ctx[g41_sctx].tex0.bits);
								write_register(g41_sctx ? RegisterAddr::TEX1_2 : RegisterAddr::TEX1_1, g41_src.ctx[g41_sctx].tex1.bits);
								write_register(g41_sctx ? RegisterAddr::CLAMP_2 : RegisterAddr::CLAMP_1, g41_src.ctx[g41_sctx].clamp.bits);
								write_register(g41_sctx ? RegisterAddr::MIPTBP1_2 : RegisterAddr::MIPTBP1_1, g41_src.ctx[g41_sctx].miptbl_1_3.bits);
								write_register(g41_sctx ? RegisterAddr::MIPTBP2_2 : RegisterAddr::MIPTBP2_1, g41_src.ctx[g41_sctx].miptbl_4_6.bits);
								write_register(g41_sctx ? RegisterAddr::ALPHA_2 : RegisterAddr::ALPHA_1, g41_src.ctx[g41_sctx].alpha.bits);
								write_register(g41_a_test, g41_src.ctx[g41_sctx].test.bits);
								write_register(g41_a_fba, g41_src.ctx[g41_sctx].fba.bits);
								Reg64<FRAMEBits> g41_f;
								g41_f.bits = g41_src.ctx[g41_sctx].frame.bits;
								g41_f.desc.FBP = g41_fbp[g41_cell];
								write_register(g41_a_frame, g41_f.bits);
								write_register(g41_a_zbuf, g41_src.ctx[g41_sctx].zbuf.bits);
								write_register(g41_a_xyof, uint64_t(0));
								Reg64<SCISSORBits> g41_sc;
								g41_sc.bits = 0;
								g41_sc.desc.SCAX0 = 0;
								g41_sc.desc.SCAY0 = 0;
								g41_sc.desc.SCAX1 = 63;
								g41_sc.desc.SCAY1 = 63;
								write_register(g41_a_scis, g41_sc.bits);
								LOGI("G41: C5 snap prim=%016llx frame=%016llx test=%016llx zbuf=%016llx tex0=%016llx alpha=%016llx fba=%016llx pabe=%016llx.\n",
								     (unsigned long long)g41_src.prim.bits,
								     (unsigned long long)g41_src.ctx[g41_sctx].frame.bits,
								     (unsigned long long)g41_src.ctx[g41_sctx].test.bits,
								     (unsigned long long)g41_src.ctx[g41_sctx].zbuf.bits,
								     (unsigned long long)g41_src.ctx[g41_sctx].tex0.bits,
								     (unsigned long long)g41_src.ctx[g41_sctx].alpha.bits,
								     (unsigned long long)g41_src.ctx[g41_sctx].fba.bits,
								     (unsigned long long)g41_src.pabe.bits);
							}
						}
						if (g41_void) continue;
						reset_vertex_queue();
						Reg64<XYZBits> g41_v0;
						g41_v0.bits = 0;
						g41_v0.desc.X = 0;
						g41_v0.desc.Y = 0;
						g41_v0.desc.Z = 0;
						Reg64<XYZBits> g41_v1;
						g41_v1.bits = 0;
						g41_v1.desc.X = 31 << 4;
						g41_v1.desc.Y = 31 << 4;
						g41_v1.desc.Z = 0;
						if (g41_c5)
						{
							Reg64<UVBits> g41_u0;
							g41_u0.bits = 0;
							g41_u0.desc.U = 0;
							g41_u0.desc.V = 0;
							Reg64<STBits> g41_s0;
							g41_s0.bits = 0;
							write_register(RegisterAddr::UV, g41_u0.bits);
							write_register(RegisterAddr::ST, g41_s0.bits);
						}
						write_register(RegisterAddr::XYZ2, g41_v0.bits);
						if (g41_c5)
						{
							Reg64<UVBits> g41_u1;
							g41_u1.bits = 0;
							g41_u1.desc.U = 511;
							g41_u1.desc.V = 511;
							Reg64<STBits> g41_s1;
							g41_s1.bits = 0;
							g41_s1.desc.S = 511.0f;
							g41_s1.desc.T = 511.0f;
							write_register(RegisterAddr::UV, g41_u1.bits);
							write_register(RegisterAddr::ST, g41_s1.bits);
						}
						write_register(RegisterAddr::XYZ2, g41_v1.bits);
						unsigned g41_acc = render_pass.primitive_count;
						{
							const auto &g41_kinst = render_pass.instances[render_pass.current_instance];
							LOGI("G41: C%u kick acc=%u ctx=%u prim=%016llx FBP=%u FBW=%u PSM=%u MSK=%08x ZBP=%u ZMSK=%u bb=%d,%d,%d,%d.\n",
							     g41_cell + 1, g41_acc, g41_lctx,
							     (unsigned long long)registers.prim.bits,
							     (unsigned)g41_kinst.frame.desc.FBP,
							     (unsigned)g41_kinst.frame.desc.FBW,
							     (unsigned)g41_kinst.frame.desc.PSM,
							     (unsigned)g41_kinst.frame.desc.FBMSK,
							     (unsigned)g41_kinst.zbuf.desc.ZBP,
							     (unsigned)g41_kinst.zbuf.desc.ZMSK,
							     g41_kinst.bb.x, g41_kinst.bb.y, g41_kinst.bb.z, g41_kinst.bb.w);
						}
						flush_render_pass(FlushReason::SubmissionFlush);
						uint64_t g41_tl = tracker.mark_submission_timeline(FlushReason::HostAccess);
						renderer.flush_submit(g41_tl);
						renderer.wait_timeline(g41_tl);
						{
							const uint8_t *g41_a = static_cast<const uint8_t *>(map_vram_read(g41_abase, g41_an));
							if (!g41_a)
							{
								LOGI("G41: C%u verd acc=%u map failed.\n", g41_cell + 1, g41_acc);
							}
							else
							{
								unsigned g41_nchg = 0;
								unsigned g41_chg0 = 0;
								for (unsigned g41_pg = 0; g41_pg < G41_ARENA_NPAGES; g41_pg++)
								{
									if (memcmp(g41_a + g41_pg * 8192, &g41_pre[g41_pg * 8192], 8192) != 0)
									{
										if (g41_nchg == 0)
											g41_chg0 = G41_ARENA_BASE_PAGE + g41_pg;
										g41_nchg++;
									}
								}
								LOGI("G41: C%u verd acc=%u landed=%u nchg=%u chg0=%u fbp=%u.\n",
								     g41_cell + 1, g41_acc, g41_nchg ? 1 : 0, g41_nchg, g41_chg0,
								     g41_fbp[g41_cell]);
								unsigned g41_shown = 0;
								for (unsigned g41_pg = 0; g41_pg < G41_ARENA_NPAGES && g41_shown < 8; g41_pg++)
									{
									if (memcmp(g41_a + g41_pg * 8192, &g41_pre[g41_pg * 8192], 8192) != 0)
									{
										uint64_t g41_cf = 0;
										uint32_t g41_cz = 0;
										g41_fnv(g41_a + g41_pg * 8192, 8192, g41_cf, g41_cz);
										const uint8_t *g41_cp = g41_a + g41_pg * 8192;
										LOGI("G41: C%u chg page=%u fnv=%016llx nz=%u head=%02x%02x%02x%02x%02x%02x%02x%02x.\n",
										     g41_cell + 1, G41_ARENA_BASE_PAGE + g41_pg,
										     (unsigned long long)g41_cf, (unsigned)g41_cz,
										     g41_cp[0], g41_cp[1], g41_cp[2], g41_cp[3],
										     g41_cp[4], g41_cp[5], g41_cp[6], g41_cp[7]);
										g41_shown++;
									}
								}
							}
						}
						{
							uint8_t *g41_w = static_cast<uint8_t *>(map_vram_write(g41_abase, g41_an));
							if (!g41_w)
							{
								LOGI("G41: C%u restored=MAPFAIL.\n", g41_cell + 1);
							}
							else
							{
								memcpy(g41_w, g41_pre.data(), g41_an);
								end_vram_write(g41_abase, g41_an);
								uint64_t g41_tl2 = tracker.mark_submission_timeline(FlushReason::HostAccess);
								renderer.flush_submit(g41_tl2);
								renderer.wait_timeline(g41_tl2);
								const uint8_t *g41_v = static_cast<const uint8_t *>(map_vram_read(g41_abase, g41_an));
								unsigned g41_rok = (g41_v && memcmp(g41_v, g41_pre.data(), g41_an) == 0) ? 1 : 0;
								LOGI("G41: C%u restored=%u.\n", g41_cell + 1, g41_rok);
							}
						}
						if (g41_c5 && !g41_void)
						{
							write_register(RegisterAddr::FOG, g41_live_all.fog.bits);
							write_register(RegisterAddr::PRMODECONT, g41_live_all.prmodecont.bits);
							write_register(RegisterAddr::TEXCLUT, g41_live_all.texclut.bits);
							write_register(RegisterAddr::TEXA, g41_live_all.texa.bits);
							write_register(RegisterAddr::FOGCOL, g41_live_all.fogcol.bits);
							write_register(RegisterAddr::DIMX, g41_live_all.dimx.bits);
							write_register(RegisterAddr::DTHE, g41_live_all.dthe.bits);
							write_register(RegisterAddr::COLCLAMP, g41_live_all.colclamp.bits);
							write_register(RegisterAddr::SCANMSK, g41_live_all.scanmsk.bits);
							write_register(g41_lctx ? RegisterAddr::TEX0_2 : RegisterAddr::TEX0_1, g41_live_all.ctx[g41_lctx].tex0.bits);
							write_register(g41_lctx ? RegisterAddr::TEX1_2 : RegisterAddr::TEX1_1, g41_live_all.ctx[g41_lctx].tex1.bits);
							write_register(g41_lctx ? RegisterAddr::CLAMP_2 : RegisterAddr::CLAMP_1, g41_live_all.ctx[g41_lctx].clamp.bits);
							write_register(g41_lctx ? RegisterAddr::MIPTBP1_2 : RegisterAddr::MIPTBP1_1, g41_live_all.ctx[g41_lctx].miptbl_1_3.bits);
							write_register(g41_lctx ? RegisterAddr::MIPTBP2_2 : RegisterAddr::MIPTBP2_1, g41_live_all.ctx[g41_lctx].miptbl_4_6.bits);
							write_register(g41_lctx ? RegisterAddr::ALPHA_2 : RegisterAddr::ALPHA_1, g41_live_all.ctx[g41_lctx].alpha.bits);
						}
						write_register(g41_a_frame, g41_s_frame);
						write_register(g41_a_zbuf, g41_s_zbuf);
						write_register(g41_a_test, g41_s_test);
						write_register(g41_a_scis, g41_s_scis);
						write_register(g41_a_xyof, g41_s_xyof);
						write_register(g41_a_fba, g41_s_fba);
						write_register(RegisterAddr::PABE, g41_s_pabe);
						write_register(RegisterAddr::RGBAQ, g41_s_rgbaq);
						write_register(RegisterAddr::ST, g41_s_st);
						write_register(RegisterAddr::UV, g41_s_uv);
						write_register(RegisterAddr::PRIM, g41_s_prim);
						write_register(RegisterAddr::PRMODE, g41_s_prim);
						reset_vertex_queue();
					}
					state_tracker = g41_saved_st;
					render_pass.potential_feedback = g41_saved_pf;
					render_pass.last_triangle_parallelogram_order = g41_saved_porder;
					render_pass.scissor_lo = g41_saved_scilo;
					render_pass.scissor_hi = g41_saved_scihi;
					render_pass.scissor_hi_x_fb = g41_saved_scixfb;
					render_pass.can_fb_wraparound = g41_saved_wrap;
					g41_dump_state("cend", 1);
					for (unsigned g41_vqi = 0; g41_vqi < 3; g41_vqi++)
					{
						vertex_queue.pos[g41_vqi] = g41_vq_pos[g41_vqi];
						vertex_queue.attr[g41_vqi] = g41_vq_attr[g41_vqi];
					}
					vertex_queue.count = g41_vq_n;
					g40_k_seen = g41_s_kseen;
					g40_k_adc = g41_s_kadc;
					g40_k_deg = g41_s_kdeg;
					g40_k_bb = g41_s_kbb;
					g40_k_fuse = g41_s_kfuse;
					g40_k_acc = g41_s_kacc;
					g40_tex_reuse = g41_s_treuse;
					g40_tex_created = g41_s_tmade;
					g40_tex_hash = g41_s_thash;
					g40_tex_tbp0 = g41_s_ttbp0;
					g40_tex_tbw = g41_s_ttbw;
					g40_tex_psm = g41_s_tpsm;
					g40_tex_levels = g41_s_tlv;
					g40_tex_samples = g41_s_tsmp;
					g40_tex_longterm = g41_s_tlt;
					g40_busy = false;
				}
			}
			g41_busy = false;
		}
	}
}

void GSInterface::flush(PageTrackerFlushFlags flags, FlushReason reason)
{
	if ((flags & PAGE_TRACKER_FLUSH_HOST_VRAM_SYNC_BIT) != 0)
	{
		block_buffer.clear();
		for (size_t i = 0, n = sync_host_vram_blocks.size(); i < n; i++)
		{
			Util::for_each_bit(sync_host_vram_blocks[i], [i, this](uint32_t bit) {
				block_buffer.push_back(i * 32 + bit);
			});
			sync_host_vram_blocks[i] = 0;
		}

		if (!block_buffer.empty())
			renderer.flush_host_vram_copy(block_buffer.data(), block_buffer.size());

		TRACE_HEADER("FLUSH HOST VRAM", Reg64<DummyBits>{0});
	}

	if ((flags & PAGE_TRACKER_FLUSH_COPY_BIT) != 0)
	{
		if ((flags & (PAGE_TRACKER_FLUSH_CACHE_BIT | PAGE_TRACKER_FLUSH_FB_BIT | PAGE_TRACKER_FLUSH_WRITE_BACK_BIT)) != 0)
		{
			TRACE_HEADER("FLUSH COPY", Reg64<DummyBits>{0});
			renderer.flush_transfer();
		}
		else
		{
			// If we're not flushing anything beyond copies, it means we're just resolving a WAW hazard internally.
			TRACE_HEADER("BARRIER COPY", Reg64<DummyBits>{0});
			renderer.transfer_overlap_barrier();
		}
	}

	if ((flags & PAGE_TRACKER_FLUSH_CACHE_BIT) != 0)
	{
		TRACE_HEADER("FLUSH CACHE UPLOAD", Reg64<DummyBits>{0});
		renderer.flush_cache_upload();
		// VRAM may have changed, so need to reset memoization state.
		render_pass.num_memoized_palettes = 0;
	}

	if ((flags & PAGE_TRACKER_FLUSH_FB_BIT) != 0)
		flush_render_pass(reason);

	if ((flags & PAGE_TRACKER_FLUSH_WRITE_BACK_BIT) != 0)
	{
		TRACE_HEADER("FLUSH WRITE BACK", Reg64<DummyBits>{0});
		block_buffer.clear();
		for (size_t i = 0, n = sync_vram_host_pages.size(); i < n; i++)
		{
			Util::for_each_bit(sync_vram_host_pages[i], [i, this](uint32_t bit) {
				block_buffer.push_back(i * 32 + bit);
			});
			sync_vram_host_pages[i] = 0;
		}

		if (!block_buffer.empty())
			renderer.flush_readback(block_buffer.data(), block_buffer.size());
	}

	// For pressure flushes, we need to ensure GPU is kicked.
	if (reason == FlushReason::PressureFlush)
	{
		uint64_t value = tracker.mark_submission_timeline();
		renderer.flush_submit(value);
		if (debug_mode.deterministic_timeline_query)
			renderer.wait_timeline(value);
	}
}

void GSInterface::sync_host_vram_page(uint32_t page_index, uint32_t block_mask)
{
	sync_host_vram_blocks[page_index] |= block_mask;
}

void GSInterface::sync_vram_host_page(uint32_t page_index)
{
	sync_vram_host_pages[page_index / 32] |= 1u << (page_index & 31);
}

void GSInterface::sync_shadow_page(uint32_t page_index)
{
	renderer.mark_shadow_page_sync(page_index);
}

void GSInterface::rewrite_forwarded_clut_upload(
		const ContextState &ctx, PaletteUploadDescriptor &upload,
		uint32_t &palette_width, uint32_t &palette_height)
{
	// Horrible engine-specific speed hack. Try to forward some extreme edge cases where
	// game renders to FB only to use that as a palette.
	// It's basically CSM2 mode with a scaled palette coordinate, so attempt to rewrite the descriptor to that style.
	// Absolute insanity, but what you gonna do *shrug*.
	// Without this, we end up with one full flush per primitive almost, making everything sub 5 fps ...

	auto &fb_instance = render_pass.instances[render_pass.current_instance];
	auto &desc = ctx.tex0.desc;

	// Match an extremely specific code pattern.
	if (uint32_t(ctx.tex0.desc.CBP) == fb_instance.frame.desc.FBP * PGS_BLOCKS_PER_PAGE &&
	    fb_instance.frame.desc.PSM == PSMCT32 &&
	    ctx.tex0.desc.CSM == TEX0Bits::CSM_LAYOUT_RECT &&
	    fb_instance.color_write_mask == 0xffffff && render_pass.primitive_count >= 2 &&
	    desc.CPSM == PSMCT32 && desc.PSM == PSMT4)
	{
		auto &prim_a = render_pass.prim[render_pass.primitive_count - 2];
		auto &prim_b = render_pass.prim[render_pass.primitive_count - 1];

		if (prim_a.state != prim_b.state || prim_a.tex != prim_b.tex)
			return;

		// This engine renders two horizontal lines, representing the palette.
		if ((prim_a.state & (1u << STATE_BIT_LINE)) == 0)
			return;

		// Check for any unexpected state which could cause the heuristic to fail.
		if ((prim_a.state & ((1u << STATE_BIT_PERSPECTIVE) | (1u << STATE_BIT_Z_TEST) | (1u << STATE_BIT_Z_WRITE))) != 0)
			return;

		// Make sure the primitive we're testing targets the current FB.
		auto a_fb_index = prim_a.state >> STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET;
		if (a_fb_index != render_pass.current_instance)
			return;

		// And that it's sampling from a texture.
		auto a_state_index = prim_a.state & ((1u << STATE_INDEX_BIT_COUNT) - 1u);
		auto &state = render_pass.state_vectors[a_state_index];
		if ((state.combiner & COMBINER_TME_BIT) == 0)
			return;

		if (((state.combiner >> COMBINER_MODE_OFFSET) & ((1u << COMBINER_MODE_BITS) - 1u)) != COMBINER_DECAL)
			return;

		// Sanity check, ignore any blending-like state.
		// Allow ATE, since the engine does that, but it always passes the test, so *shrug*.
		if ((state.blend_mode & (BLEND_MODE_ABE_BIT | BLEND_MODE_DATE_BIT | BLEND_MODE_DATM_BIT)) != 0)
			return;

		auto &pos0 = render_pass.positions[3 * render_pass.primitive_count - 5];
		auto &pos1 = render_pass.positions[3 * render_pass.primitive_count - 6];
		auto &pos2 = render_pass.positions[3 * render_pass.primitive_count - 2];
		auto &pos3 = render_pass.positions[3 * render_pass.primitive_count - 3];

		// Ensure the full 16 color palette is written as expected.
		if (pos0.pos.x != 0 || pos0.pos.y != 0)
			return;
		if (pos1.pos.x <= 7 * PGS_SUBPIXELS || pos1.pos.y != 0)
			return;
		if (pos2.pos.x != 0 || pos2.pos.y != PGS_SUBPIXELS)
			return;
		if (pos3.pos.x <= 7 * PGS_SUBPIXEL_BITS || pos3.pos.y != PGS_SUBPIXELS)
			return;

		auto &attr0 = render_pass.attributes[3 * render_pass.primitive_count - 5];
		auto &attr1 = render_pass.attributes[3 * render_pass.primitive_count - 6];
		auto &attr2 = render_pass.attributes[3 * render_pass.primitive_count - 2];
		auto &attr3 = render_pass.attributes[3 * render_pass.primitive_count - 3];

		// CSM2 can only sample from a single line.
		if (attr0.uv.y != attr1.uv.y || attr0.uv.y != attr2.uv.y || attr0.uv.y != attr3.uv.y)
			return;

		// Make sure the coordinates are increasing.
		if (attr1.uv.x <= attr0.uv.x || attr3.uv.x <= attr0.uv.x)
			return;

		// Make sure the sampling is aligned to COU alignment (can be relaxed if need be).
		if ((attr0.uv.x & (PGS_SUBPIXELS - 1)))
			return;

		// Rewrite the upload.
		upload.csm2_x_scale = float(attr3.uv.x - attr0.uv.x) / (float(pos3.pos.x) + 8 * PGS_SUBPIXELS);
		upload.csm2_x_bias = attr0.uv.x >> PGS_SUBPIXEL_BITS;
		upload.tex0.desc.CSM = TEX0Bits::CSM_LAYOUT_LINE;

		uint32_t tex_index = prim_a.tex & ((1u << TEX_TEXTURE_INDEX_BITS) - 1u);
		if (render_pass.tex0_infos[tex_index].PSM != PSMCT32)
			return;

		upload.tex0.desc.CBP = render_pass.tex0_infos[tex_index].TBP0;
		upload.texclut.desc.CBW = render_pass.tex0_infos[tex_index].TBW;
		upload.texclut.desc.COU = 0;
		upload.texclut.desc.COV = attr0.uv.y >> PGS_SUBPIXEL_BITS;
		upload.csm1_reference_base = fb_instance.frame.desc.FBP * PGS_BLOCKS_PER_PAGE;
		upload.csm1_mask = 0xff000000u;

		palette_width = ((attr3.uv.x - 1) >> PGS_SUBPIXEL_BITS) + 1u - upload.csm2_x_bias;
		palette_height = 1;

		// We won't be needing these primitives later. Rendering 1k+ primitives on one tile
		// is a serious performance drain.
		prim_a.bb = i16vec4(0, 0, -1, -1);
		prim_b.bb = i16vec4(0, 0, -1, -1);
	}
}

void GSInterface::handle_clut_upload(uint32_t ctx_index)
{
	auto &ctx = registers.ctx[ctx_index];
	auto &desc = ctx.tex0.desc;
	bool load_clut = false;

	auto psm = uint32_t(desc.PSM);
	auto cpsm = uint32_t(desc.CPSM);
	auto csa = uint32_t(desc.CSA);

	// Fixup buggy case with PSMCT24, which is not supported.
	if (cpsm == PSMCT24)
		cpsm = PSMCT32;

	// Only upload if PSM is valid.
	if (!is_palette_format(psm))
		return;

	auto CLD = uint32_t(desc.CLD);

	switch (CLD)
	{
	case TEX0Bits::CLD_LOAD:
		load_clut = true;
		break;
	case TEX0Bits::CLD_LOAD_WRITE_CBP0:
	case TEX0Bits::CLD_LOAD_WRITE_CBP1:
		load_clut = true;
		registers.cached_cbp[CLD & 1] = uint32_t(desc.CBP);
		break;
	case TEX0Bits::CLD_COMPARE_LOAD_CBP0:
	case TEX0Bits::CLD_COMPARE_LOAD_CBP1:
		load_clut = registers.cached_cbp[CLD & 1] != uint32_t(desc.CBP);
		registers.cached_cbp[CLD & 1] = uint32_t(desc.CBP);
		break;
	default:
		break;
	}

	if (!load_clut)
		return;

	// If there's a partial transfer in-flight, flush it.
	// The write should technically happen as soon as we write HWREG.
	// It's possible CLUT upload will depend on this.
	// TODO: Could hazard check this, but ... w/e. Hazards between copy and cache isn't that bad.
	if (transfer_state.host_to_local_active &&
	    transfer_state.host_to_local_payload.size() > transfer_state.last_flushed_qwords)
	{
#ifdef PARALLEL_GS_DEBUG
		LOGW("Flushing partial transfer due to palette read.\n");
#endif
		flush_pending_transfer(true);
	}

	PageRectCLUT page = {};
	uint32_t palette_width, palette_height;

	// Only target lower bank.
	if (cpsm == PSMCT32)
		csa &= 15;

	if (psm == PSMT8 || psm == PSMT8H)
	{
		if (desc.CSM != TEX0Bits::CSM_LAYOUT_RECT)
		{
			palette_width = 256;
			palette_height = 1;
		}
		else
		{
			palette_width = 16;
			palette_height = 16;
		}

		page.csa_mask = 0xffffu << csa;
	}
	else
	{
		if (desc.CSM != TEX0Bits::CSM_LAYOUT_RECT)
		{
			palette_width = 16;
			palette_height = 1;
		}
		else
		{
			palette_width = 8;
			palette_height = 4;
		}

		page.csa_mask = 1u << csa;
	}

	// For 32-bit color, read upper CLUT bank as well.
	if (cpsm == PSMCT32)
		page.csa_mask |= page.csa_mask << 16;

	// Queue up palette upload.
	PaletteUploadDescriptor palette_desc = {};
	palette_desc.texclut = registers.texclut;
	palette_desc.tex0.desc = desc;
	palette_desc.tex0.desc.CPSM = cpsm;

	// Normalize fields we don't care about.
	palette_desc.tex0.desc.TBP0 = 0;
	palette_desc.tex0.desc.TFX = 0;
	palette_desc.tex0.desc.TW = 0;
	palette_desc.tex0.desc.TH = 0;
	palette_desc.tex0.desc.TCC = 0;
	palette_desc.tex0.desc.TBW = 0;
	palette_desc.tex0.desc.CLD = 0;
	palette_desc.tex0.desc.CSA = csa;
	palette_desc.csm2_x_scale = 1.0f;

	rewrite_forwarded_clut_upload(ctx, palette_desc, palette_width, palette_height);

	uint32_t x_offset = desc.CSM == TEX0Bits::CSM_LAYOUT_LINE ? palette_desc.texclut.desc.COU * TEX0Bits::COU_SCALE : 0;
	uint32_t y_offset = desc.CSM == TEX0Bits::CSM_LAYOUT_LINE ? palette_desc.texclut.desc.COV : 0;
	x_offset += palette_desc.csm2_x_bias;

	auto clut_page = compute_page_rect(uint32_t(palette_desc.tex0.desc.CBP), x_offset, y_offset,
	                                   palette_width, palette_height,
	                                   palette_desc.texclut.desc.CBW,
	                                   cpsm);

	page.base_page = clut_page.base_page;
	page.page_width = clut_page.page_width;
	page.page_height = clut_page.page_height;
	page.page_stride = clut_page.page_stride;
	page.block_mask = clut_page.block_mask;
	page.write_mask = clut_page.write_mask;

	tracker.mark_texture_read(page);
	tracker.register_cached_clut_clobber(page);

	// Try to find a memoized palette. In case game constantly uploads CLUT redundantly.
	// This is very common, and this optimization is extremely important.
	uint32_t punchthrough_candidate = UINT32_MAX;
	palette_desc.incoming_clut_instance = render_pass.clut_instance;

	for (uint32_t i = render_pass.num_memoized_palettes; i; i--)
	{
		auto &memoized = render_pass.memoized_palettes[i - 1];

		// Try to optimize for a certain pattern where a game is doing
		// 256 color, 16 color, 256 color, 16 color, etc.

		// If a later update wrote something that this update did not write, we have diverging history.
		// Normally, games don't seem to use CSA offsets much, so this should be okay?
		if ((memoized.csa_mask & ~page.csa_mask) != 0)
		{
			// More than one candidate, ignore.
			// Also, if the 256 color entry doesn't clobber the full CSA bank, we cannot know for sure there
			// aren't any secondary CLUT updates that need to be preserved somehow.
			if (punchthrough_candidate != UINT32_MAX || memoized.csa_mask != UINT32_MAX)
				break;

			punchthrough_candidate = memoized.clut_instance;
			continue;
		}

		if (memoized.csa_mask == page.csa_mask &&
		    memoized.upload.texclut.bits == palette_desc.texclut.bits &&
		    memoized.upload.tex0.bits == palette_desc.tex0.bits &&
		    memoized.upload.csm2_x_scale == palette_desc.csm2_x_scale &&
		    memoized.upload.csm2_x_bias == palette_desc.csm2_x_bias &&
		    memoized.upload.csm1_mask == palette_desc.csm1_mask &&
		    memoized.upload.csm1_reference_base == palette_desc.csm1_reference_base)
		{
			// We found the candidate, but we must be appending our 16 color write on top of the same CLUT
			// state we used to have at the time of CLUT commit.
			if (punchthrough_candidate != UINT32_MAX && punchthrough_candidate != memoized.upload.incoming_clut_instance)
				break;

			if (memoized.clut_instance != render_pass.clut_instance)
				mark_texture_state_dirty();
			render_pass.clut_instance = memoized.clut_instance;

			// Move to end.
			if (i < render_pass.num_memoized_palettes)
			{
				memmove(render_pass.memoized_palettes + i - 1,
				        render_pass.memoized_palettes + i,
				        (render_pass.num_memoized_palettes - i) * sizeof(render_pass.memoized_palettes[0]));

				auto &last_memoized = render_pass.memoized_palettes[render_pass.num_memoized_palettes - 1];
				last_memoized.csa_mask = page.csa_mask;
				last_memoized.upload = palette_desc;
				last_memoized.clut_instance = render_pass.clut_instance;
			}

			return;
		}
	}

	render_pass.clut_instance = renderer.update_palette_cache(palette_desc);
	bool replacing_clut = render_pass.latest_clut_instance == render_pass.clut_instance;
	render_pass.latest_clut_instance = render_pass.clut_instance;
	mark_texture_state_dirty();

	if (replacing_clut)
	{
		// If we replaced an existing memoization entry which went unused, it's no longer part of the cache.
		for (uint32_t i = 0; i < render_pass.num_memoized_palettes; i++)
		{
			if (render_pass.memoized_palettes[i].clut_instance == render_pass.clut_instance)
			{
				render_pass.num_memoized_palettes--;

				// Avoid case where incoming CLUT == new CLUT instance, which can happen if we have a replacing update.
				// The memoization cache will know about the true incoming CLUT instance.
				palette_desc.incoming_clut_instance = render_pass.memoized_palettes[i].upload.incoming_clut_instance;

				if (i < render_pass.num_memoized_palettes)
				{
					memmove(render_pass.memoized_palettes + i,
					        render_pass.memoized_palettes + i + 1,
					        (render_pass.num_memoized_palettes - i) * sizeof(render_pass.memoized_palettes[0]));
				}

				break;
			}
		}
	}

	// Maintain a sliding window.
	if (render_pass.num_memoized_palettes == NumMemoizedPalettes)
	{
		memmove(render_pass.memoized_palettes, render_pass.memoized_palettes + 1,
		        sizeof(render_pass.memoized_palettes) - sizeof(render_pass.memoized_palettes[0]));
		render_pass.num_memoized_palettes--;
	}

	TRACE_INDEXED("MEMOIZE CLUT", render_pass.num_memoized_palettes, palette_desc);
	auto &memoized = render_pass.memoized_palettes[render_pass.num_memoized_palettes++];
	memoized.clut_instance = render_pass.clut_instance;
	memoized.csa_mask = page.csa_mask;
	memoized.upload = palette_desc;

	TRACE("CACHE CLUT", palette_desc);

	if (!replacing_clut)
	{
		render_pass.pending_palette_updates++;
		if (render_pass.pending_palette_updates >= CLUTInstances)
			tracker.flush_render_pass(FlushReason::Overflow);
	}
}

void GSInterface::handle_tex0_write(uint32_t ctx_index)
{
	handle_clut_upload(ctx_index);
	tracker.flush_if_memory_pressure();
}

void GSInterface::handle_miptbl_gen(uint32_t ctx_index)
{
	auto &ctx = registers.ctx[ctx_index];
	auto &tex0 = ctx.tex0.desc;
	auto &tex1 = ctx.tex1.desc;

	if (!tex1.MTBA)
		return;

	// Auto-generate MIPTBL1 when TEX0 is written, and MTBA is set.

	uint32_t base = tex0.TBP0;
	uint32_t TW = tex0.TW;
	uint32_t TH = tex0.TH;
	uint32_t W = 1u << TW;
	uint32_t H = 1u << TH;
	uint32_t row_length_64 = W / 64;

	auto layout = get_data_structure(uint32_t(tex0.PSM));
	uint32_t num_blocks = (W >> layout.block_width_log2) * (H >> layout.block_height_log2);
	base += num_blocks;

	num_blocks /= 4;
	row_length_64 /= 2;
	ctx.miptbl_1_3.desc.TBP1 = base;
	ctx.miptbl_1_3.desc.TBW1 = row_length_64;
	base += num_blocks;

	num_blocks /= 4;
	row_length_64 /= 2;
	ctx.miptbl_1_3.desc.TBP2 = base;
	ctx.miptbl_1_3.desc.TBW2 = row_length_64;
	base += num_blocks;

	ctx.miptbl_1_3.desc.TBP3 = base;
	ctx.miptbl_1_3.desc.TBW3 = row_length_64;

	state_tracker.dirty_flags |= STATE_DIRTY_TEX_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT;
}

void GSInterface::shift_vertex_queue()
{
	// Ring-buffer feels overkill. Should lower to some straight forward SIMD moves.
	if (vertex_queue.count == 3)
	{
		vertex_queue.pos[0] = vertex_queue.pos[1];
		vertex_queue.attr[0] = vertex_queue.attr[1];
		vertex_queue.pos[1] = vertex_queue.pos[2];
		vertex_queue.attr[1] = vertex_queue.attr[2];
		vertex_queue.count = 2;
	}
}

void GSInterface::vertex_kick_xyz(Reg64<XYZBits> xyz)
{
	shift_vertex_queue();
	auto &pos = vertex_queue.pos[vertex_queue.count];
	auto &attr = vertex_queue.attr[vertex_queue.count];

	pos.pos.x = int(xyz.desc.X) - render_pass.ofx;
	pos.pos.y = int(xyz.desc.Y) - render_pass.ofy;
	pos.z = xyz.desc.Z;

	attr.st.x = registers.st.desc.S;
	attr.st.y = registers.st.desc.T;
	attr.q = registers.rgbaq.desc.Q;
	attr.rgba = registers.rgbaq.words[0];
	attr.fog = float(registers.fog.desc.FOG);
	attr.uv = u16vec2(registers.uv.desc.U, registers.uv.desc.V);

	vertex_queue.count++;
	TRACE_INDEXED("VERT", vertex_queue.count, xyz);
}

void GSInterface::vertex_kick_xyzf(Reg64<XYZFBits> xyzf)
{
	shift_vertex_queue();

	auto &pos = vertex_queue.pos[vertex_queue.count];
	auto &attr = vertex_queue.attr[vertex_queue.count];

	pos.pos.x = int(xyzf.desc.X) - render_pass.ofx;
	pos.pos.y = int(xyzf.desc.Y) - render_pass.ofy;
	pos.z = xyzf.desc.Z;

	attr.st.x = registers.st.desc.S;
	attr.st.y = registers.st.desc.T;
	attr.q = registers.rgbaq.desc.Q;
	attr.rgba = registers.rgbaq.words[0];
	attr.fog = float(xyzf.desc.F);
	attr.uv = u16vec2(registers.uv.desc.U, registers.uv.desc.V);

	vertex_queue.count++;
	TRACE_INDEXED("VERT", vertex_queue.count, xyzf);
}

bool GSInterface::get_and_clear_dirty_flag(StateDirtyFlags flags)
{
	bool ret = (state_tracker.dirty_flags & flags) != 0;
	if (ret)
		state_tracker.dirty_flags &= ~flags;
	return ret;
}

void GSInterface::mark_render_pass_has_texture_feedback(const TEX0Bits &tex0, RenderPass::Feedback mode)
{
	if (render_pass.feedback_mode != RenderPass::Feedback::None && mode != render_pass.feedback_mode)
		tracker.flush_render_pass(FlushReason::TextureHazard);

	if (render_pass.feedback_mode != RenderPass::Feedback::None)
	{
		if (uint32_t(tex0.PSM) != render_pass.feedback_psm ||
		    (is_palette_format(render_pass.feedback_psm) &&
		     render_pass.feedback_cpsm != uint32_t(tex0.CPSM)))
		{
			tracker.flush_render_pass(FlushReason::TextureHazard);
		}
	}

	if (render_pass.feedback_mode == RenderPass::Feedback::None)
	{
		render_pass.feedback_mode = mode;
		render_pass.feedback_psm = uint32_t(tex0.PSM);
		render_pass.feedback_cpsm = is_palette_format(render_pass.feedback_psm) ? uint32_t(tex0.CPSM) : 0;
	}

	if (render_pass.feedback_mode == RenderPass::Feedback::Depth)
		render_pass.instances[render_pass.current_instance].z_feedback = true;
}

void GSInterface::check_frame_buffer_state()
{
	auto &prim = registers.prim;
	auto &ctx = registers.ctx[prim.desc.CTXT];

	if (!get_and_clear_dirty_flag(STATE_DIRTY_FB_BIT))
	{
		assert(render_pass.instances[render_pass.current_instance].frame.desc.compat(ctx.frame.desc));
		assert(render_pass.instances[render_pass.current_instance].zbuf.desc.compat(ctx.zbuf.desc));
		return;
	}

	auto *inst = &render_pass.instances[render_pass.current_instance];
	bool fb_delta = !inst->frame.desc.compat(ctx.frame.desc);
	bool z_delta = !inst->zbuf.desc.compat(ctx.zbuf.desc);

	if (render_pass.primitive_count && (fb_delta || (z_delta && inst->z_sensitive)))
	{
		render_pass.current_instance = render_pass.num_instances;

		bool can_fuse_render_pass = !render_pass.has_hazardous_short_term_texture_caching;

		// If we have short-term cached textures, any framebuffer change will need to invalidate those textures,
		// and therefore end the render pass.
		// Short-term textures are effectively defer-invalidated until next framebuffer change.
		if (can_fuse_render_pass)
		{
			for (uint32_t instance = 0; instance < render_pass.num_instances; instance++)
			{
				auto &test_inst = render_pass.instances[instance];
				if (test_inst.frame.desc.compat(ctx.frame.desc) &&
				    (!test_inst.z_sensitive || test_inst.zbuf.desc.compat(ctx.zbuf.desc)))
				{
					render_pass.current_instance = instance;
					break;
				}
			}
		}

		if (render_pass.current_instance == render_pass.num_instances)
		{
			// Allocate new offset instance if we can, otherwise, we're forced to flush early.
			if (render_pass.num_instances < MaxRenderPassInstances && can_fuse_render_pass)
			{
				render_pass.instances[render_pass.current_instance] = {};
				render_pass.num_instances++;
				tracker.invalidate_texture_cache(render_pass.clut_instance);
				if (render_pass.has_optimized_short_term_texture_caching)
					tracker.invalidate_fb_write_short_term_references();
			}
			else
			{
				flush_pending_transfer(true);
				tracker.flush_render_pass(FlushReason::FBPointer);
			}
		}

		// Force data structures to be updated in the new context.
		fb_delta = true;
		z_delta = true;
		render_pass.last_triangle_is_parallelogram_candidate = false;
		inst = &render_pass.instances[render_pass.current_instance];
	}

	if (fb_delta)
	{
		auto fb_layout = get_data_structure(ctx.frame.desc.PSM);
		inst->fb_page_width_log2 = fb_layout.page_width_log2;
		inst->fb_page_height_log2 = fb_layout.page_height_log2;
		inst->frame = ctx.frame;
		state_tracker.dirty_flags |= STATE_DIRTY_POTENTIAL_FEEDBACK_REGION_BIT;
	}

	if (z_delta)
	{
		auto z_layout = get_data_structure(ctx.zbuf.desc.PSM);
		inst->z_page_width_log2 = z_layout.page_width_log2;
		inst->z_page_height_log2 = z_layout.page_height_log2;
		inst->zbuf = ctx.zbuf;
		state_tracker.dirty_flags |= STATE_DIRTY_POTENTIAL_FEEDBACK_REGION_BIT;
	}

	// This is treated as an implicit texflush. Just makes sure we recheck the texture when changing FB.
	if (fb_delta || z_delta)
		state_tracker.texflush_counter++;

	assert(inst->frame.desc.compat(ctx.frame.desc));
	assert(inst->zbuf.desc.compat(ctx.zbuf.desc));
}

uint32_t GSInterface::find_or_place_unique_state_vector(const StateVector &state)
{
	uint32_t state_index;

	auto &last_state = state_tracker.last_state_vector;
	if (!render_pass.state_vectors.empty() &&
	    state.blend_mode == last_state.blend_mode &&
	    state.combiner == last_state.combiner &&
	    state.dimx.x == last_state.dimx.x &&
	    state.dimx.y == last_state.dimx.y)
	{
		return state_tracker.last_state_index;
	}

	Util::Hasher hasher;

	hasher.u32(state.blend_mode);
	hasher.u32(state.combiner);
	hasher.u32(state.dimx.x);
	hasher.u32(state.dimx.y);

	auto *cached_state_index = render_pass.state_vector_map.find(hasher.get());
	if (cached_state_index)
	{
		state_index = cached_state_index->get();
	}
	else
	{
		state_index = uint32_t(render_pass.state_vectors.size());
		TRACE_INDEXED("STATE", state_index, state);
		render_pass.state_vectors.push_back(state);
		render_pass.state_vector_map.emplace_replace(hasher.get(), state_index);
	}

	last_state = state;
	state_tracker.last_state_index = state_index;

	return state_index;
}

uint32_t GSInterface::drawing_kick_update_state_vector()
{
	if (!get_and_clear_dirty_flag(STATE_DIRTY_STATE_BIT))
		return state_tracker.last_state_index;

	auto &prim = registers.prim;
	auto &ctx = registers.ctx[prim.desc.CTXT];

	StateVector state = {};

	// Dither enable
	if (registers.dthe.desc.DTHE)
	{
		state.blend_mode |= BLEND_MODE_DTHE_BIT;
		state.dimx.x = registers.dimx.words[0];
		state.dimx.y = registers.dimx.words[1];
	}

	if (ctx.test.desc.ATE && ctx.test.desc.ATST != ATST_ALWAYS) // ALWAYS pass is meaningless.
	{
		// This degenerates into Z write disable, and color passes as normal.
		// No need to ever do the test.
		bool implied_z_mask = ctx.test.desc.ATST == ATST_NEVER &&
		                      ctx.test.desc.AFAIL == AFAIL_FB_ONLY;

		if (!implied_z_mask)
		{
			state.blend_mode |= BLEND_MODE_ATE_BIT;
			state.blend_mode |= ctx.test.desc.ATST << BLEND_MODE_ATE_MODE_OFFSET;
			state.blend_mode |= ctx.test.desc.AFAIL << BLEND_MODE_AFAIL_MODE_OFFSET;
		}
	}

	state.blend_mode |= ctx.test.desc.DATE ? BLEND_MODE_DATE_BIT : 0;
	state.blend_mode |= ctx.test.desc.DATM ? BLEND_MODE_DATM_BIT : 0;

	// Enabling AA1 seems to imply alpha blending?
	if (prim.desc.ABE || prim.desc.AA1)
	{
		state.blend_mode |= ctx.alpha.desc.A << BLEND_MODE_A_MODE_OFFSET;
		state.blend_mode |= ctx.alpha.desc.B << BLEND_MODE_B_MODE_OFFSET;
		state.blend_mode |= ctx.alpha.desc.C << BLEND_MODE_C_MODE_OFFSET;
		state.blend_mode |= ctx.alpha.desc.D << BLEND_MODE_D_MODE_OFFSET;
	}

	if (prim.desc.ABE)
	{
		state.blend_mode |= BLEND_MODE_ABE_BIT;
		state.blend_mode |= registers.pabe.desc.PABE ? BLEND_MODE_PABE_BIT : 0;
	}

	state.blend_mode |= registers.colclamp.desc.CLAMP ? BLEND_MODE_COLCLAMP_BIT : 0;
	state.blend_mode |= ctx.fba.desc.FBA ? BLEND_MODE_FB_ALPHA_BIT : 0;

	if (prim.desc.TME)
	{
		state.combiner |= COMBINER_TME_BIT;
		state.combiner |= ctx.tex0.desc.TCC ? COMBINER_TCC_BIT : 0;
		state.combiner |= uint32_t(ctx.tex0.desc.TFX) << COMBINER_MODE_OFFSET;
	}

	state.combiner |= prim.desc.FGE ? COMBINER_FOG_BIT : 0;
	return find_or_place_unique_state_vector(state);
}

void GSInterface::update_texture_page_rects()
{
	auto &prim = registers.prim;
	auto &ctx = registers.ctx[prim.desc.CTXT];
	auto psm = uint32_t(ctx.tex0.desc.PSM);
	auto &tex = state_tracker.tex;

	// Mark that we're starting a read. This will check for any hazards and flush render pass if need be.
	for (uint32_t level = 0; level < tex.rect.levels; level++)
	{
		tex.page_rects[level] = compute_page_rect(
				tex.levels[level].base,
				tex.rect.x >> level,
				tex.rect.y >> level,
				tex.rect.width >> level,
				tex.rect.height >> level,
				tex.levels[level].stride,
				psm);
	}
}

void GSInterface::texture_page_rects_read_safe_region()
{
	PageRect rect = {};
	rect.base_page = render_pass.potential_feedback.base_page;
	rect.page_width = render_pass.potential_feedback.max_safe_page + 1;
	rect.page_height = 1;
	rect.page_stride = 0;
	rect.block_mask = UINT32_MAX;
	rect.write_mask = UINT32_MAX;

	tracker.mark_texture_read(rect);
}

void GSInterface::texture_page_rects_read_region(const ivec4 &uv_bb)
{
	auto &prim = registers.prim;
	auto &ctx = registers.ctx[prim.desc.CTXT];

	assert(render_pass.is_potential_feedback);
	auto &feedback = render_pass.potential_feedback;

	assert(state_tracker.tex.rect.levels == 1);
	auto tex_base_page = uint32_t(ctx.tex0.desc.TBP0) / PGS_BLOCKS_PER_PAGE;

	assert(uv_bb.z >= 0);
	assert(uv_bb.w >= 0);
	uint32_t max_page_x = uint32_t(uv_bb.z) >> feedback.page_width_log2;
	uint32_t max_page_y = uint32_t(uv_bb.w) >> feedback.page_height_log2;

	if (!get_and_clear_dirty_flag(STATE_DIRTY_POTENTIAL_FEEDBACK_REGION_BIT))
	{
		// Don't have to loop over all pages if we know we won't collide.
		// It's only safe to skip the page traversal if we have checked once already
		// and frame buffer pointer isn't stale.
		if (max_page_x + max_page_y * feedback.page_stride <= feedback.max_safe_page)
			return;
	}

	// Clamp the hazard region so we don't falsely invalidate the texture.
	PageRect rect = {};
	rect.base_page = tex_base_page;
	rect.page_width = max_page_x + 1 + feedback.width_bias;
	rect.page_height = max_page_y + 1;
	rect.page_stride = feedback.page_stride;
	rect.block_mask = UINT32_MAX;
	rect.write_mask = UINT32_MAX;

	tracker.mark_texture_read(rect);
}

void GSInterface::texture_page_rects_read_full()
{
	auto &tex = state_tracker.tex;
	for (uint32_t level = 0; level < tex.rect.levels; level++)
		tracker.mark_texture_read(state_tracker.tex.page_rects[level]);
}

void GSInterface::invalidate_texture_hash(Util::Hash hash, bool clut)
{
	if (!clut)
	{
		// Any CLUT texture will make palette bank part of the hash.
		auto *tex = render_pass.texture_map.find(hash);
		if (tex)
			tex->valid = false;
	}

	mark_texture_state_dirty();
}

void GSInterface::forget_in_render_pass_memoization()
{
	// Forget any palette memoization.
	render_pass.num_memoized_palettes = 0;
	mark_texture_state_dirty();
}

void GSInterface::recycle_image_handle(Vulkan::ImageHandle image)
{
	// If we're debugging we don't want the confusion of aliased image handles.
	// It screws with debug label names.
	if (!debug_mode.feedback_render_target)
		renderer.recycle_image_handle(std::move(image));
}

uint64_t GSInterface::query_timeline()
{
	return renderer.query_timeline();
}

void GSInterface::mark_texture_state_dirty()
{
	state_tracker.last_texture_index = UINT32_MAX;
	state_tracker.last_texture_index_valid_at_texflush = 0;
	state_tracker.dirty_flags |= STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT;
}

void GSInterface::mark_texture_state_dirty_with_flush()
{
	mark_texture_state_dirty();
	state_tracker.texflush_counter++;
}

uint32_t GSInterface::drawing_kick_update_texture(FBFeedbackMode feedback_mode, const ivec4 &uv_bb, const ivec4 &bb)
{
	if (!get_and_clear_dirty_flag(STATE_DIRTY_TEX_BIT))
	{
		assert(state_tracker.last_texture_index != UINT32_MAX);
		assert(state_tracker.last_texture_index_valid_at_texflush == state_tracker.texflush_counter);
		if (g40_enabled()) // G40 O4m
			g40_tex_reuse++;
		return state_tracker.last_texture_index;
	}

	auto &prim = registers.prim;
	auto &ctx = registers.ctx[prim.desc.CTXT];

	if (feedback_mode == FBFeedbackMode::Pixel)
	{
		mark_render_pass_has_texture_feedback(ctx.tex0.desc,
		                                      render_pass.is_color_feedback ?
		                                      RenderPass::Feedback::Color : RenderPass::Feedback::Depth);

		// Lock the current CLUT upload as sensitive.
		if (is_palette_format(render_pass.feedback_psm))
			renderer.mark_clut_read(render_pass.clut_instance);

		// Special index indicating on-tile feedback.
		// We could add a different sentinel for depth feedback.
		// 1024k CLUT instances and 32 sub-banks. Fits in 15 bits. Use bit 15 MSB to mark feedback texture.
		return (1u << (TEX_TEXTURE_INDEX_BITS - 1u)) | (render_pass.clut_instance * 32 + uint32_t(ctx.tex0.desc.CSA));
	}

	TextureDescriptor desc = {};

	// Disregard texture state that does not affect upload.
	desc.tex0 = ctx.tex0;
	desc.tex1 = ctx.tex1;
	desc.clamp = ctx.clamp;
	desc.samples = 1;

	auto psm = uint32_t(desc.tex0.desc.PSM);
	auto cpsm = uint32_t(desc.tex0.desc.CPSM);

	// Fixup buggy case with PSMCT24, which is not supported.
	if (cpsm == PSMCT24)
	{
		cpsm = PSMCT32;
		desc.tex0.desc.CPSM = PSMCT32;
	}

	uint32_t csa_mask = 0;

	if (is_palette_format(psm))
	{
		desc.palette_bank = render_pass.clut_instance;
		desc.latest_palette_bank = render_pass.latest_clut_instance;

		// Only allowed CPSM formats are CT32 and CT16(S).
		if (cpsm != PSMCT32)
			desc.texa = registers.texa;
		else
			desc.tex0.desc.CSA &= 15;

		if (psm == PSMT8 || psm == PSMT8H)
			csa_mask = 0xffffu;
		else
			csa_mask = 1u;

		csa_mask <<= uint32_t(desc.tex0.desc.CSA);

		// For 32-bit color, read upper CLUT bank as well.
		if (cpsm == PSMCT32)
		{
			csa_mask &= 0xffff;
			csa_mask |= csa_mask << 16;
		}

		// Lock the current CLUT upload as sensitive.
		renderer.mark_clut_read(render_pass.clut_instance);
	}
	else
	{
		// Don't care about palette.
		desc.tex0.desc.CPSM = 0;
		desc.tex0.desc.CSA = 0;
		// TODO: May be possible to replace TEXA with image view swizzle if 0 or 0xff.
		// 0x80 is more common alpha value on PS2 though, so probably not worth the hassle.
		if (psm != PSMCT32 && psm != PSMZ32)
			desc.texa = registers.texa;
	}

	// Only affects shading
	desc.tex0.desc.TCC = 0;
	desc.tex0.desc.TFX = 0;

	// Only affects palette upload
	desc.tex0.desc.CBP = 0;
	desc.tex0.desc.CSM = 0;
	desc.tex0.desc.CLD = 0;

	// As a general rule we should cache a texture long term which allows us to track COPY hazards properly,
	// but in feedback scenarios where there is overlap between
	// the UV BB and rendering BB, we temporarily suspend hazard tracking until we can prove a well-defined
	// rendering pattern where render region and sampling region is disjoint.
	bool long_term_cache_texture = true;

	if (feedback_mode == FBFeedbackMode::Sliced)
	{
		// If game explicitly clamps the rect to a small region, it's likely doing well-defined feedbacks.
		// E.g. Tales of Abyss main menu ping-pong blurs.
		// This code is quite flawed, and I'm not sure what the correct solution is yet.
		if (PRIMType(prim.desc.PRIM) == PRIMType::Sprite)
		{
			// If game is using sprites, it's more likely than not it's doing explicit mip blurs, etc, so cache those.
			// The main problem we always want to avoid is heavy random triangle soup geometry that does feedback.
			long_term_cache_texture = !render_pass.current_primitive_is_channel_shuffle;

			// Very crude heuristic. Some games have full-screen warping effects where
			// they intend to render small sprites at a bias. If we force hazards in this case, performance collapses.
			// Very strange not to use triangles here, but this heuristic should pick up these cases.
			// For geniune mip-mapping scenarios, FBW will likely not be large (>= 256 pixels),
			// and tiny primitives are likely not used.
			if (long_term_cache_texture &&
			    ctx.frame.desc.FBW >= 4 &&
			    uv_bb.z - uv_bb.x < 16 && uv_bb.w - uv_bb.y < 16 &&
			    bb.z - bb.x < 16 && bb.w - bb.y < 16)
			{
				// If the BB and UV_BB are very far part, over a page's worth, we are probably relying on proper feedback
				// and not a simple warp effect.
				ivec4 hazard_bb(
						std::max<int>(uv_bb.x, bb.x),
						std::max<int>(uv_bb.y, bb.y),
						std::min<int>(uv_bb.z, bb.z),
						std::min<int>(uv_bb.w, bb.w));

				long_term_cache_texture = (hazard_bb.z + 64 < hazard_bb.x) || (hazard_bb.w + 64 < hazard_bb.y);
			}

			if (long_term_cache_texture && desc.tex1.desc.MMAG)
			{
				// Check case for in-place blur. If a texture is being sampled at a very tiny bias with linear filter, assume
				// the game is abusing some texcache behavior to make it work somehow, or at least look okay.
				// It tends to look very broken, especially when using SSAA textures.
				int u_offset = std::abs(uv_bb.x - bb.x);
				int v_offset = std::abs(uv_bb.y - bb.y);
				int u_size = uv_bb.z - uv_bb.x;
				int v_size = uv_bb.w - uv_bb.y;
				int x_size = bb.z - bb.x;
				int y_size = bb.w - bb.y;

				// For linear filter, UV bb is generally 2 extra pixels.
				u_size -= 2;
				v_size -= 2;

				if (u_offset <= 1 && v_offset <= 1 && u_size == x_size && v_size == y_size)
					long_term_cache_texture = false;
			}
		}
		else if (desc.clamp.desc.WMS == CLAMPBits::REGION_CLAMP && desc.clamp.desc.WMT == CLAMPBits::REGION_CLAMP)
		{
			ivec4 clamped_uv_bb(
					int(desc.clamp.desc.MINU),
					int(desc.clamp.desc.MINV),
					int(desc.clamp.desc.MAXU),
					int(desc.clamp.desc.MAXV));

			ivec4 hazard_bb(
					std::max<int>(clamped_uv_bb.x, bb.x),
					std::max<int>(clamped_uv_bb.y, bb.y),
					std::min<int>(clamped_uv_bb.z, bb.z),
					std::min<int>(clamped_uv_bb.w, bb.w));

			long_term_cache_texture = hazard_bb.x > hazard_bb.z || hazard_bb.y > hazard_bb.w;
		}
		else
		{
			if (render_pass.last_triangle_is_parallelogram_candidate)
			{
				// Questionable heuristic. If it looks like we're going to be rendering sprites, be a bit more aggressive.
				ivec4 hazard_bb(
						std::max<int>(uv_bb.x, bb.x),
						std::max<int>(uv_bb.y, bb.y),
						std::min<int>(uv_bb.z, bb.z),
						std::min<int>(uv_bb.w, bb.w));

				long_term_cache_texture = hazard_bb.x > hazard_bb.z || hazard_bb.y > hazard_bb.w;
			}
			else
			{
				// Questionable, but it seems almost impossible to do this correctly and fast.
				// Need to emulate the PS2 texture cache exactly, which is just insane.
				// This should be fine in most cases.
				long_term_cache_texture = false;
			}
		}
	}
	else if (feedback_mode == FBFeedbackMode::BypassHazards)
	{
		// Only hold this texture for the duration of the render pass.
		long_term_cache_texture = false;
	}

	auto TW = uint32_t(desc.tex0.desc.TW);
	auto TH = uint32_t(desc.tex0.desc.TH);
	uint32_t width = 1u << TW;
	uint32_t height = 1u << TH;

	auto page_height = get_data_structure(psm).page_height;

	if (render_pass.is_potential_feedback &&
	    width > uint32_t(desc.tex0.desc.TBW) * PGS_BUFFER_WIDTH_SCALE &&
	    desc.tex0.desc.TBW != 0 &&
	    height > page_height)
	{
		// Speculate that we can clamp the image region.
		// This is mostly a performance workaround, especially when using SSAA textures.
		// Only do this if height is large enough that stride is even meaningful.
		if (desc.clamp.desc.WMS == CLAMPBits::CLAMP || desc.clamp.desc.WMS == CLAMPBits::REPEAT)
		{
			desc.clamp.desc.WMS = CLAMPBits::REGION_CLAMP;
			desc.clamp.desc.MINU = 0;
			desc.clamp.desc.MAXU = uint32_t(desc.tex0.desc.TBW * PGS_BUFFER_WIDTH_SCALE) - 1;
		}
		else if (desc.clamp.desc.WMS == CLAMPBits::REGION_CLAMP)
		{
			desc.clamp.desc.MINU = std::min<uint32_t>(desc.clamp.desc.MINU, desc.tex0.desc.TBW * PGS_BUFFER_WIDTH_SCALE - 1);
			desc.clamp.desc.MAXU = std::min<uint32_t>(desc.clamp.desc.MAXU, desc.tex0.desc.TBW * PGS_BUFFER_WIDTH_SCALE - 1);
		}
	}

	// In sliced mode with clamping, we can clamp harder based on uv_bb.
	// In this path, we're guaranteed to not hit wrapping with region clamp.
	// For repeat, give up. Should not happen (hopefully).
	if (feedback_mode == FBFeedbackMode::Sliced && long_term_cache_texture &&
	    desc.clamp.desc.WMS != CLAMPBits::REGION_REPEAT &&
	    desc.clamp.desc.WMT != CLAMPBits::REGION_REPEAT)
	{
		// Narrow the texture size for purposes of reducing load, since we'll be discarding this texture right away.
		if (desc.clamp.desc.WMS == CLAMPBits::REGION_CLAMP)
		{
			// Further clamp the range.
			desc.clamp.desc.MINU = std::max<int>(
					int(desc.clamp.desc.MINU), std::min<int>(uv_bb.x, int(desc.clamp.desc.MAXU)));
			desc.clamp.desc.MAXU = std::min<int>(
					int(desc.clamp.desc.MAXU), std::max<int>(uv_bb.z, int(desc.clamp.desc.MINU)));
		}
		else if (desc.clamp.desc.WMS == CLAMPBits::CLAMP || (uv_bb.z < int(width) && uv_bb.x >= 0))
		{
			// Invent a clamp.
			// If we have repeat, we must observe those semantics accurately.
			desc.clamp.desc.WMS = CLAMPBits::REGION_CLAMP;
			desc.clamp.desc.MINU = std::max<int>(0, uv_bb.x);
			desc.clamp.desc.MAXU = std::min<int>(int(width) - 1, uv_bb.z);
		}

		if (desc.clamp.desc.WMT == CLAMPBits::REGION_CLAMP)
		{
			// Further clamp the range.
			desc.clamp.desc.MINV = std::max<int>(
					int(desc.clamp.desc.MINV), std::min<int>(uv_bb.y, int(desc.clamp.desc.MAXV)));
			desc.clamp.desc.MAXV = std::min<int>(
					int(desc.clamp.desc.MAXV), std::max<int>(uv_bb.w, int(desc.clamp.desc.MINV)));
		}
		else if (desc.clamp.desc.WMT == CLAMPBits::CLAMP || (uv_bb.w < int(height) && uv_bb.y >= 0))
		{
			// Invent a clamp.
			// If we have repeat, we must observe those semantics accurately.
			desc.clamp.desc.WMT = CLAMPBits::REGION_CLAMP;
			desc.clamp.desc.MINV = std::max<int>(0, uv_bb.y);
			desc.clamp.desc.MAXV = std::min<int>(int(height) - 1, uv_bb.w);
		}
	}

	// Ignore {MIN,MAX}{U,V} if region modes are not used.
	if (!desc.clamp.desc.has_horizontal_region())
	{
		// Normalize these so we don't create duplicate textures for different clamp modes.
		desc.clamp.desc.MINU = 0;
		desc.clamp.desc.MAXU = 0;
		desc.clamp.desc.WMS = CLAMPBits::CLAMP;
	}

	if (!desc.clamp.desc.has_vertical_region())
	{
		// Normalize these so we don't create duplicate textures for different clamp modes.
		desc.clamp.desc.MINV = 0;
		desc.clamp.desc.MAXV = 0;
		desc.clamp.desc.WMT = CLAMPBits::CLAMP;
	}

	// No point in uploading mips if we never access it.
	if (!desc.tex1.desc.mmin_has_mipmap() || hacks.disable_mipmaps)
		desc.tex1.desc.MXL = 0;

	// Memoize this computation.
	state_tracker.tex.rect = desc.rect = GSRenderer::compute_effective_texture_rect(desc);
	state_tracker.tex.levels[0].base = desc.tex0.desc.TBP0;
	state_tracker.tex.levels[0].stride = desc.tex0.desc.TBW;

	if (desc.rect.levels >= 2)
	{
		desc.miptbp1_3.desc.TBP1 = state_tracker.tex.levels[1].base = ctx.miptbl_1_3.desc.TBP1;
		desc.miptbp1_3.desc.TBW1 = state_tracker.tex.levels[1].stride = ctx.miptbl_1_3.desc.TBW1;
	}

	if (desc.rect.levels >= 3)
	{
		desc.miptbp1_3.desc.TBP2 = state_tracker.tex.levels[2].base = ctx.miptbl_1_3.desc.TBP2;
		desc.miptbp1_3.desc.TBW2 = state_tracker.tex.levels[2].stride = ctx.miptbl_1_3.desc.TBW2;
	}

	if (desc.rect.levels >= 4)
	{
		desc.miptbp1_3.desc.TBP3 = state_tracker.tex.levels[3].base = ctx.miptbl_1_3.desc.TBP3;
		desc.miptbp1_3.desc.TBW3 = state_tracker.tex.levels[3].stride = ctx.miptbl_1_3.desc.TBW3;
	}

	if (desc.rect.levels >= 5)
	{
		desc.miptbp4_6.desc.TBP1 = state_tracker.tex.levels[4].base = ctx.miptbl_4_6.desc.TBP1;
		desc.miptbp4_6.desc.TBW1 = state_tracker.tex.levels[4].stride = ctx.miptbl_4_6.desc.TBW1;
	}

	if (desc.rect.levels >= 6)
	{
		desc.miptbp4_6.desc.TBP2 = state_tracker.tex.levels[5].base = ctx.miptbl_4_6.desc.TBP2;
		desc.miptbp4_6.desc.TBW2 = state_tracker.tex.levels[5].stride = ctx.miptbl_4_6.desc.TBW2;
	}

	if (desc.rect.levels >= 7)
	{
		desc.miptbp4_6.desc.TBP3 = state_tracker.tex.levels[6].base = ctx.miptbl_4_6.desc.TBP3;
		desc.miptbp4_6.desc.TBW3 = state_tracker.tex.levels[6].stride = ctx.miptbl_4_6.desc.TBW3;
	}

	// Only affects shading.
	desc.tex1.desc.LCM = 0;
	desc.tex1.desc.MMAG = 0;
	desc.tex1.desc.MMIN = 0;
	desc.tex1.desc.MTBA = 0;
	desc.tex1.desc.L = 0;
	desc.tex1.desc.K = 0;

	update_texture_page_rects();

	// Be quite conservative when we decide to attempt super sampled textures.
	// This is mostly only useful in:
	// - Correctly resolving per-pixel effects which use some form of pixel testing.
	// - Preserving per sample data during blit passes, etc.
	if (desc.rect.levels == 1 && super_sampled_textures &&
	    get_bits_per_pixel(desc.tex0.desc.PSM) >= 8 &&
	    PRIMType(prim.desc.PRIM) == PRIMType::Sprite &&
	    tracker.texture_may_super_sample(state_tracker.tex.page_rects[0]))
	{
		desc.samples = 1u << (sampling_rate_x_log2 + sampling_rate_y_log2);
	}

	Util::Hasher hasher;
	hasher.u64(desc.tex0.bits);
	hasher.u64(desc.tex1.bits);
	hasher.u64(desc.texa.bits);
	hasher.u64(desc.miptbp1_3.bits);
	hasher.u64(desc.miptbp4_6.bits);
	hasher.u64(desc.clamp.bits);
	// Palette bank needs to be part of hash key.
	// If the same texture is being used with different palettes things break really fast.
	// We need to be able to hold different variants of the same texture in the memoization structure.
	// The page tracker never keeps more than one variant alive however, so the multiple variants only
	// live as long as we can maintain the render pass.
	hasher.u64(desc.palette_bank);
	hasher.u32(desc.samples);
	auto *cached_index = render_pass.texture_map.find(hasher.get());

	// For explicit feedback, we have to be super careful, and we skip these checks.
	// This is mostly relevant for potential feedback and textures placed at an address
	// slightly above the frame buffer pointer for whatever reason.
	bool skip_hazard_checks = cached_index && cached_index->valid &&
	                          cached_index->valid_at_texflush == state_tracker.texflush_counter &&
	                          feedback_mode == FBFeedbackMode::None;

	if (!skip_hazard_checks)
	{
		if (render_pass.is_potential_feedback)
		{
			// Only keep this alive until end of render pass, or a copy hazard occurs.
			long_term_cache_texture = false;
			texture_page_rects_read_region(uv_bb);
		}
		else if (long_term_cache_texture)
		{
			texture_page_rects_read_full();
		}

		// The render pass might have been flushed, have to requery.
		cached_index = render_pass.texture_map.find(hasher.get());

		// We started long-term, but now we're rendering on top of it in sliced mode, cannot use this reference.
		if (cached_index && cached_index->valid && cached_index->long_term && !long_term_cache_texture)
		{
			texture_page_rects_read_full();
			cached_index = render_pass.texture_map.find(hasher.get());
		}
	}

	if (!long_term_cache_texture)
		render_pass.has_hazardous_short_term_texture_caching = true;

	if (state_tracker.last_texture_index != UINT32_MAX &&
	    !render_pass.tex_infos.empty() &&
	    state_tracker.last_texture_descriptor == desc)
	{
		state_tracker.last_texture_index_valid_at_texflush = state_tracker.texflush_counter;
		if (cached_index && cached_index->valid)
			cached_index->valid_at_texflush = state_tracker.texflush_counter;
		if (g40_enabled()) // G40 O4m
			g40_tex_reuse++;
		return state_tracker.last_texture_index;
	}

	uint32_t texture_index;
	bool g40_tex_made = false; // G40 O4

	if (cached_index && cached_index->valid)
	{
		texture_index = cached_index->index;
		if (g40_enabled()) // G40 O4m
			g40_tex_reuse++;
	}
	else
	{
		// If we're not caching in the page tracker, we have to at least do hazard tracking on the first read from VRAM.
		// Any subsequent read from this texture will ignore hazard tracking.
		if (render_pass.is_potential_feedback)
			texture_page_rects_read_safe_region();
		else if (!long_term_cache_texture || skip_hazard_checks)
			texture_page_rects_read_full();

		if (!debug_mode.disable_sampler_feedback && long_term_cache_texture)
		{
			// Framebuffer textures should be analyzed properly.
			// It's very unlikely that these textures will persist across multiple render passes anyway.
			// For very large textures that are probably not 1k in reality should also get analysis passes.
			// Update long_term_cache_texture after setting hazardous_short_term, since this
			// is purely done for optimization purposes, and not due to potential feedback.
			uint32_t pixel_count = desc.rect.width * desc.rect.height;
			const uint32_t pixel_threshold = tracker.texture_may_super_sample(state_tracker.tex.page_rects[0]) ?
			                                 128 * 128 : 512 * 512;

			// If width/height is larger than the stride, and the stride matters,
			// it's a good sign that the texture is not really this large. Opt-in to sampler feedback.
			if (desc.rect.levels == 1 &&
			    (pixel_count > pixel_threshold || desc.rect.width >= 1024 || desc.rect.height >= 1024 ||
			     (desc.rect.width > desc.tex0.desc.TBW * 64 && desc.rect.height > page_height)))
			{
				long_term_cache_texture = false;
				render_pass.has_optimized_short_term_texture_caching = true;
			}
		}

		auto image = tracker.find_cached_texture(hasher.get());
		if (!image)
		{
			TRACE("CACHE IMAGE", desc);
			desc.hash = hasher.get();
			image = renderer.create_cached_texture(desc);
			g40_tex_made = true; // G40 O4

			// Long-term references can persist across render passes, and intended for normal resource textures.
			// They will generally be invalidated when it's overwritten by a copy or FB write.
			if (long_term_cache_texture)
			{
				if (desc.rect.levels == 1 &&
				    state_tracker.tex.page_rects[0].page_width == 1 &&
				    state_tracker.tex.page_rects[0].page_height == 1)
				{
					state_tracker.last_cpu_compatible_cache_TBP0 = desc.tex0.desc.TBP0;
				}
				else
					state_tracker.last_cpu_compatible_cache_TBP0 = UINT32_MAX;

				// CPU upload is not possible if we hit SSAA texture path.
				if (desc.samples == 1)
				{
					if (tracker.register_cached_texture(state_tracker.tex.page_rects, desc.rect.levels,
														csa_mask, render_pass.clut_instance,
														hasher.get(), image) == PageTracker::UploadStrategy::CPU)
					{
						renderer.promote_cached_texture_upload_cpu(state_tracker.tex.page_rects[0]);
					}
				}
			}
			else
			{
				// Potential feedback textures are handled explicitly w.r.t. FB hazards,
				// but we still need to consider potential copy hazards.
				tracker.register_short_term_cached_texture(state_tracker.tex.page_rects, desc.rect.levels, hasher.get());
				recycle_image_handle(image);
			}

			renderer.commit_cached_texture(render_pass.tex_infos.size(),
			                               desc.rect.levels == 1 && !long_term_cache_texture &&
			                               !debug_mode.disable_sampler_feedback);
		}

		texture_index = render_pass.tex_infos.size();

		if (cached_index)
		{
			cached_index->valid_at_texflush = state_tracker.texflush_counter;
			cached_index->index = texture_index;
			cached_index->valid = true;
		}
		else
		{
			render_pass.texture_map.emplace_replace(hasher.get(), texture_index,
				state_tracker.texflush_counter, long_term_cache_texture);
		}

		TextureInfo info = {};
		info.view = &image->get_view();
		info.info.sizes = vec4(float(width), float(height),
							   1.0f / float(info.view->get_view_width()),
							   1.0f / float(info.view->get_view_height()));

		if (uint32_t(desc.clamp.desc.WMS) == CLAMPBits::CLAMP)
		{
			info.info.region.x = 0.0f;
			info.info.region.z = float(info.view->get_view_width()) - 1.0f;
		}
		else if (uint32_t(desc.clamp.desc.WMS) == CLAMPBits::REGION_CLAMP)
		{
			info.info.region.x = float(uint32_t(desc.clamp.desc.MINU));
			info.info.region.z = float(uint32_t(desc.clamp.desc.MAXU));
		}

		if (uint32_t(desc.clamp.desc.WMT) == CLAMPBits::CLAMP)
		{
			info.info.region.y = 0.0f;
			info.info.region.w = float(info.view->get_view_height()) - 1.0f;
		}
		else if (uint32_t(desc.clamp.desc.WMT) == CLAMPBits::REGION_CLAMP)
		{
			info.info.region.y = float(uint32_t(desc.clamp.desc.MINV));
			info.info.region.w = float(uint32_t(desc.clamp.desc.MAXV));
		}

		info.info.bias.x = -float(desc.rect.x) * info.info.sizes.z;
		info.info.bias.y = -float(desc.rect.y) * info.info.sizes.w;

		info.info.arrayed = int(desc.samples > 1);
		info.info.flags = long_term_cache_texture ? TEX_INFO_LONG_TERM_REFERENCE : 0;
		if (info.info.arrayed)
			render_pass.tex_infos_has_super_samples = true;

		// Common pattern for esoteric channel re-mapping. Don't try to be clever here. Force sample mapping.
		if (info.info.arrayed && is_palette_format(psm) && desc.clamp.desc.has_region_repeat())
			info.info.flags |= TEX_INFO_FORCE_SAMPLE_MAPPING;

		render_pass.tex_infos.push_back(info);
		render_pass.tex0_infos.push_back(ctx.tex0.desc);
		render_pass.held_images.push_back(std::move(image));
	}

	if (g40_enabled()) // G40 O4: full-resolve snapshot (first use of a texture per pass)
	{
		g40_tex_created = g40_tex_made;
		g40_tex_hash = hasher.get();
		g40_tex_tbp0 = desc.tex0.desc.TBP0;
		g40_tex_tbw = desc.tex0.desc.TBW;
		g40_tex_psm = desc.tex0.desc.PSM;
		g40_tex_levels = desc.rect.levels;
		g40_tex_samples = desc.samples;
		g40_tex_longterm = long_term_cache_texture;
	}
	state_tracker.last_texture_descriptor = desc;
	state_tracker.last_texture_index = texture_index;
	state_tracker.last_texture_index_valid_at_texflush = state_tracker.texflush_counter;
	state_tracker.texflush_counter_pending = false;
	return texture_index;
}

void GSInterface::drawing_kick_update_state(FBFeedbackMode feedback_mode, const ivec4 &uv_bb, const ivec4 &bb)
{
	if (!get_and_clear_dirty_flag(STATE_DIRTY_PRIM_TEMPLATE_BIT))
		return;

	auto &prim = registers.prim;
	auto &ctx = registers.ctx[prim.desc.CTXT];

	auto &p = state_tracker.prim_template;
	p = {};

	if (prim.desc.TME)
	{
		uint32_t tex_index = drawing_kick_update_texture(feedback_mode, uv_bb, bb);
		p.tex = tex_index << TEX_TEXTURE_INDEX_OFFSET;
		p.tex |= ctx.tex1.desc.MMAG == TEX1Bits::LINEAR ? TEX_SAMPLER_MAG_LINEAR_BIT : 0;
		p.tex |= ctx.clamp.desc.has_horizontal_clamp() ? TEX_SAMPLER_CLAMP_S_BIT : 0;
		p.tex |= ctx.clamp.desc.has_vertical_clamp() ? TEX_SAMPLER_CLAMP_T_BIT : 0;

		if (tex_index < MaxTextures)
		{
			auto &info = render_pass.tex_infos[tex_index].info;
			if (info.arrayed)
			{
				p.tex |= TEX_PER_SAMPLE_BIT;
				if ((info.flags & TEX_INFO_FORCE_SAMPLE_MAPPING) != 0)
					p.tex |= TEX_SAMPLE_MAPPING_BIT;
			}
		}

		if (ctx.tex1.desc.mmin_has_mipmap() && !hacks.disable_mipmaps)
		{
			p.tex2 = ctx.tex1.desc.LCM << TEX2_FIXED_LOD_OFFSET;
			p.tex2 |= ctx.tex1.desc.L << TEX2_L_OFFSET;
			p.tex2 |= ctx.tex1.desc.K << TEX2_K_OFFSET;
			p.tex |= ctx.tex1.desc.MXL << TEX_MAX_MIP_LEVEL_OFFSET;

			switch (ctx.tex1.desc.MMIN)
			{
			case TEX1Bits::LINEAR:
				p.tex |= TEX_SAMPLER_MIN_LINEAR_BIT;
				break;
			case TEX1Bits::NEAREST_MIPMAP_LINEAR:
				p.tex |= TEX_SAMPLER_MIPMAP_LINEAR_BIT;
				break;
			case TEX1Bits::LINEAR_MIPMAP_NEAREST:
				p.tex |= TEX_SAMPLER_MIN_LINEAR_BIT;
				break;
			case TEX1Bits::LINEAR_MIPMAP_LINEAR:
				p.tex |= TEX_SAMPLER_MIN_LINEAR_BIT | TEX_SAMPLER_MIPMAP_LINEAR_BIT;
				break;
			default:
				break;
			}
		}
		else
		{
			// Always flag fixed LOD so we can do early perspective divide.
			p.tex2 = 1u << TEX2_FIXED_LOD_OFFSET;
		}

		if ((p.tex & (1u << (TEX_TEXTURE_INDEX_BITS - 1))) != 0)
		{
			p.tex2 |= registers.texa.desc.AEM << TEX2_FEEDBACK_AEM_OFFSET;
			p.tex2 |= registers.texa.desc.TA0 << TEX2_FEEDBACK_TA0_OFFSET;
			p.tex2 |= registers.texa.desc.TA1 << TEX2_FEEDBACK_TA1_OFFSET;
		}
	}

	// Update state after updating texture state, since reading a texture may cause a flush,
	// which resets the state vectors.
	p.state = drawing_kick_update_state_vector() << STATE_INDEX_BIT_OFFSET;

	if (ctx.test.desc.ZTE == TESTBits::ZTE_ENABLED)
	{
		if (ctx.test.desc.has_z_test())
		{
			p.state |= 1u << STATE_BIT_Z_TEST;
			p.state |= ctx.test.desc.ZTST == TESTBits::ZTST_GREATER ? (1u << STATE_BIT_Z_TEST_GREATER) : 0;
		}

		// It's possible to use alpha test as a quirky way to disable Z writes.
		bool implied_z_mask = ctx.test.desc.ATE != 0 &&
		                      ctx.test.desc.ATST == ATST_NEVER &&
		                      ctx.test.desc.AFAIL != AFAIL_ZB_ONLY;

		if (!implied_z_mask && ctx.zbuf.desc.ZMSK == 0)
			p.state |= 1u << STATE_BIT_Z_WRITE;
	}

	bool color_write_needs_previous_pixels = false;

	// AA1 implies alpha-blending of some sort.
	if (prim.desc.ABE || prim.desc.AA1)
	{
		// If any of the blend factors use dst color, it's not opaque.
		// It's still possible to abuse blender to do extra math while remaining opaque.
		if (ctx.alpha.desc.A == BLEND_RGB_DEST ||
		    ctx.alpha.desc.B == BLEND_RGB_DEST ||
		    ctx.alpha.desc.C == BLEND_ALPHA_DEST ||
		    ctx.alpha.desc.D == BLEND_RGB_DEST)
		{
			color_write_needs_previous_pixels = true;
		}
	}

	if (ctx.test.desc.DATE)
	{
		color_write_needs_previous_pixels = true;
	}
	else if (ctx.test.desc.ATE && ctx.test.desc.ATST != ATST_ALWAYS)
	{
		// ATST of NEVER will usually trigger a degenerate primitive.
		// If this primitive cannot write Z, an AFAIL of FB_ONLY is completely irrelevant.
		bool can_write_z = (p.state & (1u << STATE_BIT_Z_WRITE)) != 0u;
		bool can_mask_color = ctx.test.desc.AFAIL != AFAIL_FB_ONLY;
		bool can_mask_z = can_write_z && ctx.test.desc.AFAIL != AFAIL_ZB_ONLY;
		if (can_mask_color || can_mask_z)
			color_write_needs_previous_pixels = true;
	}

	// FBMASK is in a similar situation where we might not be considered OPAQUE, but if all primitives
	// in a pass use the same FBMASK, we can be considered opaque. Need to defer this decision until triangle setup time.

	// If we're in a feedback situation,
	// we cannot be opaque since sampling a texture essentially becomes blending.
	if ((render_pass.is_color_feedback || render_pass.is_depth_feedback) && feedback_mode == FBFeedbackMode::Pixel)
		color_write_needs_previous_pixels = true;

	// If OPAQUE, the frame buffer color content is fully written if Z test passes.
	// Final output does not depend on previous color data at all.
	if (!color_write_needs_previous_pixels)
		p.state |= 1u << STATE_BIT_OPAQUE;

	// Points and Sprites don't get AA1.
	// I assume blending still happens (since AA1 implies blending),
	// but we won't compute any fancy coverage.
	if (prim.desc.enables_aa1())
	{
		p.state |= 1u << STATE_BIT_MULTISAMPLE;
		render_pass.has_aa1 = true;
	}

	if (registers.scanmsk.desc.has_mask())
	{
		p.state |= 1u << (STATE_BIT_SCANMSK_EVEN + registers.scanmsk.desc.MSK - SCANMSKBits::MSK_SKIP_EVEN);
		render_pass.has_scanmsk = true;

		// Heuristic to catch known games that hit this.
		// If game is rendering a tall scanmsk-ed primitive, assume this is needed.
		if (bb.y + 400 < bb.w)
		{
			// Somewhat of a hack to avoid having to turn off force_progressive manually.
			// Scanmask punches a hole in the progressive assumption.
			state_tracker.has_complex_scanmsk_timeout = 4;
		}
	}

	if (!prim.desc.FST)
		p.state |= 1u << STATE_BIT_PERSPECTIVE;
	if (prim.desc.IIP)
		p.state |= 1u << STATE_BIT_IIP;
	if (prim.desc.FIX)
		p.state |= 1u << STATE_BIT_FIX;
}

PageRect GSInterface::compute_fb_rect() const
{
	auto &inst = render_pass.instances[render_pass.current_instance];
	// We know this BB is not degenerate already.
	assert(inst.bb.x <= inst.bb.z);
	assert(inst.bb.y <= inst.bb.w);
	auto bb_page = inst.bb >> ivec2(inst.fb_page_width_log2, inst.fb_page_height_log2).xyxy();

	PageRect page = {};

	page.base_page = inst.frame.desc.FBP;
	page.page_width = bb_page.z - bb_page.x + 1;
	page.page_height = bb_page.w - bb_page.y + 1;

	// We may benefit from sub-page FB write tracking.
	// Don't enter this path by default, since it's more expensive to setup.
	// It assumes we'll need sub-block tracking.
	if (page.page_width == 1 && page.page_height == 1)
	{
		return compute_page_rect(inst.frame.desc.FBP * PGS_BLOCKS_PER_PAGE,
		                         inst.bb.x, inst.bb.y, inst.bb.z - inst.bb.x + 1,
		                         inst.bb.w - inst.bb.y + 1, inst.frame.desc.FBW,
		                         inst.frame.desc.PSM);
	}

	page.page_stride = inst.frame.desc.FBW;
	page.base_page += bb_page.x + bb_page.y * page.page_stride;
	page.block_mask = UINT32_MAX;
	page.write_mask = psm_word_write_mask(inst.frame.desc.PSM);

	return page;
}

PageRect GSInterface::compute_z_rect() const
{
	auto &inst = render_pass.instances[render_pass.current_instance];
	// We know this BB is not degenerate already.
	assert(inst.bb.x <= inst.bb.z);
	assert(inst.bb.y <= inst.bb.w);
	auto bb_page = inst.bb >> ivec2(inst.z_page_width_log2, inst.z_page_height_log2).xyxy();

	PageRect page = {};

	page.base_page = inst.zbuf.desc.ZBP;
	page.page_width = bb_page.z - bb_page.x + 1;
	page.page_height = bb_page.w - bb_page.y + 1;

	// We may benefit from sub-page FB write tracking.
	// Don't enter this path by default, since it's more expensive to setup.
	// It assumes we'll need sub-block tracking.
	if (page.page_width == 1 && page.page_height == 1)
	{
		return compute_page_rect(inst.zbuf.desc.ZBP * PGS_BLOCKS_PER_PAGE,
		                         inst.bb.x, inst.bb.y, inst.bb.z - inst.bb.x + 1,
		                         inst.bb.w - inst.bb.y + 1, inst.frame.desc.FBW,
		                         inst.zbuf.desc.PSM | ZBUFBits::PSM_MSB);
	}

	page.page_stride = inst.frame.desc.FBW;
	page.base_page += bb_page.x + bb_page.y * page.page_stride;
	page.block_mask = UINT32_MAX;
	page.write_mask = psm_word_write_mask(inst.zbuf.desc.PSM);

	return page;
}

bool GSInterface::draw_is_degenerate()
{
	if (!get_and_clear_dirty_flag(STATE_DIRTY_DEGENERATE_BIT))
		return state_tracker.degenerate_draw;

	auto &prim = registers.prim;
	auto &ctx = registers.ctx[prim.desc.CTXT];

	// Degenerate scissor.
	if (ctx.scissor.desc.SCAX0 > ctx.scissor.desc.SCAX1 ||
	    ctx.scissor.desc.SCAY0 > ctx.scissor.desc.SCAY1)
	{
		state_tracker.degenerate_draw = true;
		return true;
	}

	// We never pass the depth test.
	if (ctx.test.desc.ZTE == TESTBits::ZTE_ENABLED && ctx.test.desc.ZTST == TESTBits::ZTST_NEVER)
	{
		state_tracker.degenerate_draw = true;
		return true;
	}

	// We force alpha test to fail, and fail mode is to keep FB contents -> no side effects.
	if (ctx.test.desc.ATE && ctx.test.desc.ATST == ATST_NEVER && ctx.test.desc.AFAIL == AFAIL_KEEP)
	{
		state_tracker.degenerate_draw = true;
		return true;
	}

	// If ATEST_NEVER is used, it functions like a secondary write mask.
	// Compute the effective static write mask, and figure out if we can discard the primitive.
	uint32_t fbmsk = ctx.frame.desc.FBMSK;
	uint32_t zmsk = ctx.zbuf.desc.ZMSK;

	if (ctx.frame.desc.PSM == PSMCT24 || ctx.frame.desc.PSM == PSMZ24)
		fbmsk |= 0xff << 24;

	if (ctx.test.desc.ATE && ctx.test.desc.ATST == ATST_NEVER)
	{
		switch (ctx.test.desc.AFAIL)
		{
		case AFAIL_ZB_ONLY:
			// Only Z is written.
			fbmsk |= UINT32_MAX;
			break;

		case AFAIL_KEEP:
			// Nothing is written. Obvious degenerate draw.
			fbmsk |= UINT32_MAX;
			zmsk |= 1;
			break;

		case AFAIL_FB_ONLY:
			// RGBA is written.
			zmsk |= 1;
			break;

		case AFAIL_RGB_ONLY:
			// RGB is written. A is masked. For 24-bit, A masking is implied.
			fbmsk |= 0xff << 24;
			break;

		default:
			break;
		}
	}

	// Color is effectively masked here, functioning as a noop.
	if (prim.desc.ABE && ctx.alpha.desc.D == BLEND_RGB_DEST && ctx.alpha.desc.A == ctx.alpha.desc.B)
		fbmsk |= 0xffffff;

	// Any write is ignored. PS2 rendering does not have side effects.
	// Undefined ZTE seems to mean ignore depth completely.

	bool read_only_color = fbmsk == UINT32_MAX;
	bool read_only_depth = zmsk || ctx.test.desc.ZTE == TESTBits::ZTE_UNDEFINED;
	state_tracker.degenerate_draw = read_only_color && read_only_depth;
	return state_tracker.degenerate_draw;
}

bool GSInterface::state_is_z_sensitive() const
{
	auto &prim = registers.prim;
	auto &ctx = registers.ctx[prim.desc.CTXT];

	if (ctx.test.desc.ZTE == TESTBits::ZTE_ENABLED)
	{
		// We need to read depth.
		if (ctx.test.desc.has_z_test())
			return true;

		bool implied_z_mask = ctx.test.desc.ATE != 0 &&
		                      ctx.test.desc.ATST == ATST_NEVER &&
		                      ctx.test.desc.AFAIL != AFAIL_ZB_ONLY;

		// We need to write depth.
		// ZTST_NEVER will trigger degenerate draw and won't hit this path.
		if (!implied_z_mask && ctx.zbuf.desc.ZMSK == 0)
			return true;
	}

	return false;
}

void GSInterface::update_color_feedback_state()
{
	if (!get_and_clear_dirty_flag(STATE_DIRTY_FEEDBACK_BIT))
	{
		// If we're in feedback, we have to recheck state every draw. We expect that anyway
		// since FB will likely have to be flushed every draw ...
		if (render_pass.is_color_feedback || render_pass.is_depth_feedback)
			state_tracker.dirty_flags |= STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT;
		return;
	}

	auto &prim = registers.prim;
	auto &ctx = registers.ctx[prim.desc.CTXT];
	auto &inst = render_pass.instances[render_pass.current_instance];

	render_pass.is_color_feedback = false;
	render_pass.is_depth_feedback = false;
	render_pass.is_awkward_color_feedback = false;
	render_pass.is_potential_feedback = false;
	render_pass.is_potential_channel_shuffle = false;

	if (!prim.desc.TME)
		return;

	// Mip-mapping is too weird to deal with.
	if (ctx.tex1.desc.has_mipmap() && !hacks.disable_mipmaps)
		return;

	auto tbp0 = uint32_t(ctx.tex0.desc.TBP0);
	auto tex_psm = uint32_t(ctx.tex0.desc.PSM);

	if (ctx.frame.desc.PSM == tex_psm &&
	    (tex_psm == PSMCT16 || tex_psm == PSMCT16S || tex_psm == PSMZ16 || tex_psm == PSMZ16S))
	{
		render_pass.is_potential_channel_shuffle = true;
	}

	// A game might use REGION_CLAMP to align the effective "base pointer" of the texture to the frame buffer.
	if (!is_palette_format(tex_psm))
	{
		if (ctx.clamp.desc.WMS == CLAMPBits::REGION_CLAMP && ctx.clamp.desc.MINU != 0)
			tbp0 += PGS_BLOCKS_PER_PAGE * (uint32_t(ctx.clamp.desc.MINU) >> get_data_structure(tex_psm).page_width_log2);
		if (ctx.clamp.desc.WMT == CLAMPBits::REGION_CLAMP && ctx.clamp.desc.MINV != 0)
			tbp0 += PGS_BLOCKS_PER_PAGE * uint32_t(ctx.tex0.desc.TBW) * (uint32_t(ctx.clamp.desc.MINV) >> get_data_structure(tex_psm).page_height_log2);
	}

	const bool equal_address_color = tbp0 == ctx.frame.desc.FBP * PGS_BLOCKS_PER_PAGE;
	// Only consider feedbacks if we're actually Z sensitive, i.e. there is a depth buffer to read in the first place.
	const bool equal_address_depth = tbp0 == ctx.zbuf.desc.ZBP * PGS_BLOCKS_PER_PAGE &&
	                                 (inst.z_sensitive || state_is_z_sensitive());

	if (equal_address_color)
	{
		if (swizzle_compat_key(tex_psm) != swizzle_compat_key(ctx.frame.desc.PSM) ||
		    uint32_t(ctx.tex0.desc.TBW) != ctx.frame.desc.FBW)
		{
			// If we have overlapping address, but the swizzling format is off, this is pure mayhem, and we force-disable caching.
			// It makes zero sense to try anything reasonable here.
			render_pass.is_color_feedback = true;
			render_pass.is_awkward_color_feedback = true;
			return;
		}
	}

	if (equal_address_depth)
	{
		if (swizzle_compat_key(tex_psm) != swizzle_compat_key(ctx.zbuf.desc.PSM | ZBUFBits::PSM_MSB) ||
		    uint32_t(ctx.tex0.desc.TBW) != ctx.frame.desc.FBW)
		{
			// Mayhem. We don't really care unless there is content in the wild that relies on this to work well.
			return;
		}
	}

	if (ctx.clamp.desc.WMS == CLAMPBits::REGION_REPEAT || ctx.clamp.desc.WMT == CLAMPBits::REGION_REPEAT)
	{
		// Anything repeat region is too messy.
		return;
	}

	// Check for potential feedback case.
	if (!equal_address_color && !equal_address_depth)
	{
		// If TBP < FBP we may still have a potential feedback caused by game using randomly large TW/TH
		// and not using REGION_CLAMP properly. E.g. a 1024x1024 texture with 32-bit will cover the entirety of VRAM.
		// The end of a texture may straddle into the frame buffer
		// even if game never intends to actually sample from that region.
		// In this case, there's no reasonable way it will work, so try to clamp the page rect to avoid false hazards.
		// This will break if game actually intended to sample like this, but it seems extremely unlikely in practice.
		// TODO: A more proper solution is to do analysis of UV bb per draw when we're in the "potential feedback" case.

		bool is_potential_color_feedback = false;
		bool is_potential_depth_feedback = false;
		compute_has_potential_feedback(ctx.tex0.desc, ctx.clamp.desc,
		                               ctx.frame.desc, ctx.zbuf.desc,
		                               vram_size / PGS_PAGE_ALIGNMENT_BYTES,
		                               is_potential_color_feedback, is_potential_depth_feedback);

		bool existing_z_write = inst.z_write;
		if (existing_z_write)
		{
			// Only accept existing Z writes in the render pass if we keep appending to it,
			// i.e. there are no FB pointer changes. Otherwise, we may falsely consider it a feedback.
			existing_z_write = inst.frame.desc.compat(ctx.frame.desc) &&
			                   inst.zbuf.desc.compat(ctx.zbuf.desc);
		}

		// Cannot rely on existing_z_write fully since this is called before we commit Z-state.
		bool has_z_write = existing_z_write || (state_is_z_sensitive() && ctx.zbuf.desc.ZMSK == 0);

		uint32_t tex_write_mask = psm_word_write_mask(tex_psm);
		uint32_t fb_write_mask = psm_word_write_mask(inst.frame.desc.PSM);
		uint32_t z_write_mask = psm_word_write_mask(inst.zbuf.desc.PSM);

		// If aliasing with 8H and 24, that is fine.
		if ((tex_write_mask & fb_write_mask) == 0)
			is_potential_color_feedback = false;
		if ((tex_write_mask & z_write_mask) == 0 || !has_z_write)
			is_potential_depth_feedback = false;

		render_pass.is_potential_feedback = is_potential_color_feedback || is_potential_depth_feedback;

		if (render_pass.is_potential_feedback)
		{
			auto layout = get_data_structure(ctx.tex0.desc.PSM);

			render_pass.potential_feedback.base_page = uint32_t(ctx.tex0.desc.TBP0) / PGS_BLOCKS_PER_PAGE;
			render_pass.potential_feedback.page_width_log2 = layout.page_width_log2;
			render_pass.potential_feedback.page_height_log2 = layout.page_height_log2;
			render_pass.potential_feedback.page_stride = ctx.tex0.desc.TBW;
			if (ctx.tex0.desc.PSM == PSMT4 || ctx.tex0.desc.PSM == PSMT8)
				render_pass.potential_feedback.page_stride >>= 1;
			// Potential straddle.
			render_pass.potential_feedback.width_bias = (ctx.tex0.desc.TBP0 & (PGS_BLOCKS_PER_PAGE - 1)) != 0 ? 1 : 0;

			uint32_t num_safe_pages = UINT32_MAX;
			if (is_potential_color_feedback)
			{
				uint32_t safe_color_pages = ctx.frame.desc.FBP - render_pass.potential_feedback.base_page;
				safe_color_pages &= vram_size / PGS_PAGE_ALIGNMENT_BYTES - 1;
				num_safe_pages = std::min<uint32_t>(num_safe_pages, safe_color_pages);
			}

			if (is_potential_depth_feedback)
			{
				uint32_t safe_depth_pages = ctx.zbuf.desc.ZBP - render_pass.potential_feedback.base_page;
				safe_depth_pages &= vram_size / PGS_PAGE_ALIGNMENT_BYTES - 1;
				num_safe_pages = std::min<uint32_t>(num_safe_pages, safe_depth_pages);
			}

			if (render_pass.potential_feedback.width_bias >= num_safe_pages)
				render_pass.is_potential_feedback = false;
			else
				render_pass.potential_feedback.max_safe_page = num_safe_pages - 1u - render_pass.potential_feedback.width_bias;
		}
	}
	else
	{
		// If we're in proper feedback, we have to recheck state every draw, since texture state depends on per-primitive UVs.
		render_pass.is_color_feedback = equal_address_color &&
		                                (psm_word_write_mask(tex_psm) & psm_word_write_mask(ctx.frame.desc.PSM)) != 0;
		render_pass.is_depth_feedback = equal_address_depth &&
		                                (psm_word_write_mask(tex_psm) & psm_word_write_mask(ctx.zbuf.desc.PSM)) != 0;

		state_tracker.dirty_flags |= STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT;
	}
}

template <bool quad, unsigned num_vertices, bool conservative>
static void compute_uv_bb(const VertexAttribute *attr, const ContextState &ctx, const PRIMBits &prim, ivec4 &uv_bb,
                          ivec2 *uvs, bool *needs_perspective)
{
	int width = 1 << int(ctx.tex0.desc.TW);
	int height = 1 << int(ctx.tex0.desc.TH);
	auto fwidth = float(width * 16);
	auto fheight = float(height * 16);
	ivec2 local_uvs[3];

	if (needs_perspective)
		*needs_perspective = false;
	if (!uvs)
		uvs = local_uvs;

	if (prim.FST)
	{
		uvs[0] = ivec2(attr[0].uv);
		uvs[1] = ivec2(attr[1].uv);
		if (!quad)
			uvs[2] = ivec2(attr[2].uv);
	}
	else
	{
		// If we have perspective, we cannot assume pixel correctness.
		// For sprite, Q is flat, and we only use Q0 anyway.
		if (!quad && needs_perspective)
			if (attr[0].q != attr[1].q || attr[1].q != attr[2].q)
				*needs_perspective = true;

		float inv_q0 = 1.0f / attr[0].q;
		float inv_q1 = 1.0f / attr[1].q;
		uvs[0] = ivec2(vec2(fwidth, fheight) * (attr[0].st * inv_q0));
		uvs[1] = ivec2(vec2(fwidth, fheight) * (attr[1].st * inv_q1));

		if (!quad)
		{
			float inv_q2 = 1.0f / attr[2].q;
			uvs[2] = ivec2(vec2(fwidth, fheight) * (attr[2].st * inv_q2));
		}
	}

	ivec2 uv_min = min(uvs[0], uvs[1]);
	ivec2 uv_max = max(uvs[0], uvs[1]);
	if (!quad)
	{
		uv_min = min(uv_min, uvs[2]);
		uv_max = max(uv_max, uvs[2]);
	}

	if (conservative)
	{
		if (ctx.tex1.desc.MMAG != 0)
		{
			// Consider linear filtering if using that. Expand the BB appropriately.
			uv_min -= ivec2(1 << (PGS_SUBPIXEL_BITS - 1));
			uv_max += ivec2((1 << (PGS_SUBPIXEL_BITS - 1)) - 1);
		}
		else if (!prim.FST)
		{
			// Consider FP rounding errors.
			uv_min -= ivec2(1);
			uv_max += ivec2(1);
		}
	}

	// This can safely become a REGION_CLAMP.
	uv_bb = ivec4(uv_min, uv_max) >> PGS_SUBPIXEL_BITS;

	if (!conservative)
	{
		// The bottom-right pixels tend to be unused due to raster rules.
		// If there's only a one pixel overlap region, we can safely ignore that since
		// it's going to be a false positive in 99.999999% of cases ...
		uv_bb.z = std::max<int>(uv_bb.z - 1, 0);
		uv_bb.w = std::max<int>(uv_bb.w - 1, 0);
	}
}

template <bool quad, unsigned num_vertices>
GSInterface::FBFeedbackMode
GSInterface::deduce_color_feedback_mode(const VertexPosition *pos, const VertexAttribute *attr,
                                        const ContextState &ctx, const PRIMBits &prim,
                                        ivec4 &uv_bb, const ivec4 &bb)
{
	// Sprite and triangle is fine. Line is not ok.
	constexpr bool can_feedback = num_vertices == 3 || (quad && num_vertices == 2);
	if (!can_feedback)
		return FBFeedbackMode::None;

	if (render_pass.is_awkward_color_feedback)
		return FBFeedbackMode::BypassHazards;

	int width = 1 << int(ctx.tex0.desc.TW);
	int height = 1 << int(ctx.tex0.desc.TH);
	bool needs_perspective = false;

	ivec2 uvs[3] = {};
	compute_uv_bb<quad, num_vertices, true>(attr, ctx, prim, uv_bb, uvs, &needs_perspective);

	// Check if we're sampling outside the texture's range. In this case we get clamp or repeat,
	// and we cannot assume 1:1 pixel mapping.
	// We'll allow equal, since bottom-right pixels won't get rendered usually.
	// Any line with linear filtering is probably not pixel feedback.
	// Anything with perspective won't work with Pixel mode either.
	if (needs_perspective)
		return FBFeedbackMode::Sliced;

	// Based on the primitive BB, if the region clamp contains the full primitive BB, we cannot observe clamping,
	// so ignore the effect.
	if (uint32_t(ctx.clamp.desc.WMS) == CLAMPBits::REGION_CLAMP)
	{
		int minu = int(ctx.clamp.desc.MINU);
		int maxu = int(ctx.clamp.desc.MAXU);
		if (bb.x < minu || bb.z > maxu)
			return FBFeedbackMode::Sliced;
	}
	else if (bb.z >= width)
		return FBFeedbackMode::Sliced;

	if (uint32_t(ctx.clamp.desc.WMT) == CLAMPBits::REGION_CLAMP)
	{
		int minv = int(ctx.clamp.desc.MINV);
		int maxv = int(ctx.clamp.desc.MAXV);
		if (bb.y < minv || bb.w > maxv)
			return FBFeedbackMode::Sliced;
	}
	else if (bb.w >= height)
		return FBFeedbackMode::Sliced;

	ivec2 uv0_delta = uvs[0] - pos[0].pos;
	ivec2 uv1_delta = uvs[1] - pos[1].pos;
	ivec2 min_delta = min(uv0_delta, uv1_delta);
	ivec2 max_delta = max(uv0_delta, uv1_delta);

	if (!quad)
	{
		ivec2 uv2_delta = uvs[2] - pos[2].pos;
		min_delta = min(min_delta, uv2_delta);
		max_delta = max(max_delta, uv2_delta);
	}

	int min_delta2 = min(min_delta.x, min_delta.y);
	int max_delta2 = max(max_delta.x, max_delta.y);

	if (ctx.tex1.desc.MMAG == TEX1Bits::LINEAR)
	{
		// Must land on pixel center for LINEAR to work.
		if (min_delta2 != (1 << (PGS_SUBPIXEL_BITS - 1)) || max_delta2 != (1 << (PGS_SUBPIXEL_BITS - 1)))
			return FBFeedbackMode::Sliced;
	}
	else
	{
		// The UV offset must be in range of [0, 2^SUBPIXEL_BITS - 1]. This guarantees snapping with NEAREST.
		// 8 is ideal. That means pixel centers during interpolation will land exactly in the center of the texel.
		if (min_delta2 < 0 || max_delta2 >= (1 << PGS_SUBPIXEL_BITS))
			return FBFeedbackMode::Sliced;
	}

	// Perf go brrrrrrr.
	return FBFeedbackMode::Pixel;
}

template <bool list_primitive, bool fan_primitive, bool quad, unsigned num_vertices>
void GSInterface::drawing_kick_append()
{
	auto &prim = registers.prim;
	auto &ctx = registers.ctx[prim.desc.CTXT];
	if (g41_enabled() && ctx.frame.desc.FBP == 112)
	{
		g41_snap_regs = registers;
		g41_snap_valid = true;
	}

	VertexAttribute attr[3];
	VertexPosition pos[3];

	if (num_vertices == 1)
	{
		pos[0] = vertex_queue.pos[vertex_queue.count - 1];
		attr[0] = vertex_queue.attr[vertex_queue.count - 1];

		// According to TJnotJT: half pixel points round up.
		// Make use of top-left rules here to contruct a primitive
		// which has bottom-right wins rules.
		pos[0].pos.x -= (1 << (PGS_SUBPIXEL_BITS - 1)) - 1;
		pos[0].pos.y -= (1 << (PGS_SUBPIXEL_BITS - 1)) - 1;

		pos[1] = pos[0];
		pos[1].pos.x += 1 << PGS_SUBPIXEL_BITS;
		pos[1].pos.y += 1 << PGS_SUBPIXEL_BITS;
	}
	else if (num_vertices == 2)
	{
		for (uint32_t i = 0; i < num_vertices; i++)
		{
			pos[i] = vertex_queue.pos[vertex_queue.count - 1 - i];
			attr[i] = vertex_queue.attr[vertex_queue.count - 1 - i];
		}
	}
	else if (num_vertices == 3)
	{
		for (uint32_t i = 0; i < num_vertices; i++)
		{
			pos[i] = vertex_queue.pos[2 - i];
			attr[i] = vertex_queue.attr[2 - i];
		}
	}

	ivec2 lo_pos = muglm::min(pos[0].pos, pos[1].pos);
	ivec2 hi_pos = muglm::max(pos[0].pos, pos[1].pos);

	// Take into account line expansion just to be safe.
	constexpr bool is_line = !quad && num_vertices == 2;

	if (!quad && !is_line)
	{
		lo_pos = muglm::min(pos[2].pos, lo_pos);
		hi_pos = muglm::max(pos[2].pos, hi_pos);
	}

	auto pre_snap_lo = lo_pos;
	auto pre_snap_hi = hi_pos;

	hi_pos -= 1;
	// Tighten the bounding box according to top-left raster rules.
	if (!render_pass.field_aware_rendering)
		lo_pos += (1 << int(PGS_SUBPIXEL_BITS - sampling_rate_y_log2)) - 1;

	lo_pos >>= int(PGS_SUBPIXEL_BITS);
	hi_pos >>= int(PGS_SUBPIXEL_BITS);

	// We implement lines as a parallelogram which expand X or Y by +/- 0.5
	// pixel or 1.0 pixel depending on AA1 mode.
	// Triangle AA1 is very similar to line AA since it's modelled as line AA on edges,
	// creating some kind of "feathered" rasterization.
	if (is_line || registers.prim.desc.enables_aa1())
	{
		lo_pos -= ivec2(1);
		hi_pos += ivec2(1);
	}

	if (get_and_clear_dirty_flag(STATE_DIRTY_SCISSOR_BIT))
	{
		ivec2 sci_lo = ivec2(ctx.scissor.desc.SCAX0, ctx.scissor.desc.SCAY0);
		ivec2 sci_hi = ivec2(ctx.scissor.desc.SCAX1, ctx.scissor.desc.SCAY1);
		// This is somewhat dubious, but there's no logical reason to render outside one page's worth of width
		// when using FBW = 0 for whatever reason. Duplicating page writes would wreak havoc.
		render_pass.scissor_lo = sci_lo;
		render_pass.scissor_hi = sci_hi;
		int fbw_deduced_width = std::max<int>(1, int(ctx.frame.desc.FBW)) * PGS_BUFFER_WIDTH_SCALE;
		render_pass.scissor_hi_x_fb = std::min<int>(sci_hi.x, fbw_deduced_width - 1);
		render_pass.can_fb_wraparound = ctx.frame.desc.FBW != 0 && render_pass.scissor_hi_x_fb < render_pass.scissor_hi.x;
	}

	lo_pos = muglm::max(lo_pos, render_pass.scissor_lo);
	hi_pos = muglm::min(hi_pos, render_pass.scissor_hi);
	ivec4 bb = ivec4(lo_pos, hi_pos);

	// Check for degenerate BB. Can happen if primitive is clipped away completely by scissor.
	if (bb.z < bb.x || bb.w < bb.y)
	{
		if (g40_enabled()) // G40 O1
			g40_k_bb++;
		TRACE("Degenerate BB", bb);
		return;
	}

	bool is_parallelogram_candidate = false;
	if (num_vertices == 3)
	{
		// If two adjacent triangles look a lot like a sprite we can fuse the two triangles into a quad,
		// which is more efficient for our renderer.
		// Also, skip any hazard checking when doing this.
		// This helps avoid a lot of false positives when doing feedback rendering with two triangles which form a quad.
		// Also speeds up raster / binning since we only have to consider one primitive.

		ivec3 order;
		is_parallelogram_candidate = triangle_is_parallelogram_candidate(
				pos, attr, pre_snap_lo, pre_snap_hi, prim.desc, order);

		// If no state changed, try to match the parallelogram.
		if (state_tracker.dirty_flags == 0 && is_parallelogram_candidate &&
		    render_pass.last_triangle_is_parallelogram_candidate &&
		    triangles_form_parallelogram(pos, attr, order,
		                                 render_pass.positions + (render_pass.primitive_count - 1) * 3,
		                                 render_pass.attributes + (render_pass.primitive_count - 1) * 3,
		                                 render_pass.last_triangle_parallelogram_order,
		                                 prim.desc))
		{
			auto &state = render_pass.prim[render_pass.primitive_count - 1].state;
			state |= (1u << STATE_BIT_PARALLELOGRAM) |
			         (render_pass.last_triangle_parallelogram_order.x << STATE_PARALLELOGRAM_PROVOKING_OFFSET);

			render_pass.last_triangle_is_parallelogram_candidate = false;
			if (g40_enabled()) // G40 O1
				g40_k_fuse++;
			TRACE("Promote Parallelogram", DummyBits{});
			return;
		}

		render_pass.last_triangle_parallelogram_order = order;
	}

	update_color_feedback_state();
	auto feedback_mode = FBFeedbackMode::None;
	ivec4 uv_bb = {};
	if (render_pass.is_color_feedback || render_pass.is_depth_feedback)
	{
		// Some heuristics would like to know about this.
		render_pass.last_triangle_is_parallelogram_candidate = is_parallelogram_candidate;
		feedback_mode = deduce_color_feedback_mode<quad, num_vertices>(pos, attr, ctx, prim.desc, uv_bb, bb);
	}
	else if (render_pass.is_potential_feedback || (render_pass.is_potential_channel_shuffle && quad))
		compute_uv_bb<quad, num_vertices, false>(attr, ctx, prim.desc, uv_bb, nullptr, nullptr);

	render_pass.current_primitive_is_channel_shuffle = false;
	if (quad && render_pass.is_potential_channel_shuffle)
	{
		// Channel shuffles are special PS2 hacks that abuse the swizzle layout of 16-bit formats.
		// X coordinate [0, 8) and [8, 15) map to each halve of a 32-bit color word.
		// This can be used to copy R/G into B/A or vice versa.
		// If we detect this case, assume it's not a real feedback.
		// If we don't detect this, it's 20+ RPs per shuffle, which is brutal.
		// If all pixels land within the same 8-pixel column, this is a clear channel shuffle case.
		// Also sanity check that uv_bb is horizontally XOR 8 pixels to be even more safe.
		if ((bb.x & ~7) == (bb.z & ~7) && (uv_bb.x ^ 8) == bb.x && uv_bb.y == bb.y)
			render_pass.current_primitive_is_channel_shuffle = true;
	}

	// If there's a partial transfer in-flight, flush it.
	// The write should technically happen as soon as we write HWREG.
	// This can trigger a texture invalidation. We need to do it here, before checking for texture dirty state.
	if (prim.desc.TME && transfer_state.host_to_local_active &&
	    transfer_state.host_to_local_payload.size() > transfer_state.last_flushed_qwords)
	{
#ifdef PARALLEL_GS_DEBUG
		LOGW("Flushing partial transfer due to texture read ...\n");
#endif
		flush_pending_transfer(true);
	}

	// Even if no state changes, we have to consider potential hazards.
	// If a hazard does occur, dirty bits will be set appropriately,
	// re-triggering state checks.
	check_frame_buffer_state();

	if (prim.desc.TME && (state_tracker.dirty_flags & STATE_DIRTY_TEX_BIT) == 0)
	{
		// TEXFLUSH while keeping the texture state frozen is curious.
		if (state_tracker.texflush_counter_pending)
		{
			state_tracker.texflush_counter++;
			state_tracker.texflush_counter_pending = false;
		}

		// Have to make sure it's still safe to read the texture we're using.
		// Only do this when dirty flag is not set. Otherwise, we'll check it when resolving texture index anyway.
		if (state_tracker.texflush_counter != state_tracker.last_texture_index_valid_at_texflush ||
		    feedback_mode != FBFeedbackMode::None)
		{
			if (render_pass.is_potential_feedback)
				texture_page_rects_read_region(uv_bb);
			else
				texture_page_rects_read_full();

			if ((state_tracker.dirty_flags & STATE_DIRTY_TEX_BIT) == 0)
				state_tracker.last_texture_index_valid_at_texflush = state_tracker.texflush_counter;
		}
	}

	drawing_kick_update_state(feedback_mode, uv_bb, bb);

	auto &fb_instance = render_pass.instances[render_pass.current_instance];

	const auto &prim_state = state_tracker.prim_template;

	PrimitiveAttribute prim_attr;
	prim_attr.tex = prim_state.tex;
	prim_attr.tex2 = prim_state.tex2;
	prim_attr.state = prim_state.state;
	prim_attr.fbmsk = ctx.frame.desc.FBMSK;
	prim_attr.fogcol = registers.fogcol.words[0];
	prim_attr.alpha = (ctx.alpha.desc.FIX << ALPHA_AFIX_OFFSET) |
	                  (ctx.test.desc.AREF << ALPHA_AREF_OFFSET);

	if (quad)
	{
		prim_attr.state |= 1u << STATE_BIT_PARALLELOGRAM;
		prim_attr.state |= 1u << STATE_BIT_SPRITE;
		prim_attr.state |= 1u << STATE_BIT_SNAP_RASTER;
		prim_attr.state |= 1u << STATE_BIT_SNAP_ATTRIBUTE;
		prim_attr.state &= ~(1u << STATE_BIT_MULTISAMPLE);
	}
	else if (is_line)
	{
		prim_attr.state |= 1u << STATE_BIT_PARALLELOGRAM;
		prim_attr.state |= 1u << STATE_BIT_LINE;
		// Lines always have less than full coverage, if using AA1, never write Z.
		if ((prim_attr.state & (1u << STATE_BIT_MULTISAMPLE)) != 0)
			prim_attr.state &= ~(1u << STATE_BIT_Z_WRITE);
	}

	if (num_vertices == 1)
	{
		// Don't interpolate anything.
		prim_attr.state |= 1u << STATE_BIT_FIX;
		// Don't think we can reasonably upscale a point. Games can rely on the rounding to generate an exact pixel.
		prim_attr.state |= 1u << STATE_BIT_SNAP_RASTER;
		prim_attr.state |= 1u << STATE_BIT_SNAP_ATTRIBUTE;
	}

	prim_attr.state |= render_pass.current_instance << STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET;

	// If our damage region expands, then mark hazards.
	// This avoids spam where we have to remark pages as dirty every single draw.
	bool rp_expands = false;
	bool is_z_sensitive = state_is_z_sensitive();

	// We go from no Z pages to at least read-only Z.
	if (!fb_instance.z_sensitive && is_z_sensitive)
	{
		fb_instance.z_sensitive = true;
		rp_expands = true;
	}

	// We go from read-only Z to read-write Z.
	if (is_z_sensitive && ctx.zbuf.desc.ZMSK == 0 && !fb_instance.z_write)
	{
		fb_instance.z_write = true;
		// With Z writes existing, we might have a feedback we didn't have before.
		state_tracker.dirty_flags |= STATE_DIRTY_FEEDBACK_BIT;
		rp_expands = true;
	}

	// Color write mask increases, redamage all pages.
	uint32_t write_mask = ~ctx.frame.desc.FBMSK;
	if ((write_mask & fb_instance.color_write_mask) != write_mask)
	{
		fb_instance.color_write_mask |= write_mask;
		rp_expands = true;
	}

	auto &current_bb = fb_instance.bb;

	if (render_pass.can_fb_wraparound && bb.x > render_pass.scissor_hi_x_fb)
	{
		// Esoteric edge case where a game is attempting to render beyond the width of the frame buffer.
		// What would happen in this case is a wraparound effect where it appears
		// as-if the primitives render on the next page in Y direction instead.
		// The simplest possible fix is to just fix-up the coordinates here.
		// This isn't technically correct in scenarios where there is partial wraparound on the framebuffer's edge,
		// but that would lead to absolute nonsense outcomes and there are no known cases of that happening.

		// Shift the primitive in place, and hope for the best.
		int x_offset = -int(ctx.frame.desc.FBW << fb_instance.fb_page_width_log2);
		int y_offset = int(1u << fb_instance.fb_page_height_log2);

		// It's overwhelmingly likely to resolve in one iteration,
		// so don't bother trying to be cute with integer divisions.
		while (bb.x > render_pass.scissor_hi_x_fb)
		{
			bb.x += x_offset;
			bb.y += y_offset;
			bb.z += x_offset;
			bb.w += y_offset;

			for (uint32_t i = 0; i < num_vertices; i++)
			{
				pos[i].pos.x += x_offset << PGS_SUBPIXEL_BITS;
				pos[i].pos.y += y_offset << PGS_SUBPIXEL_BITS;
			}
		}

		bb.z = std::min<int>(bb.z, render_pass.scissor_hi_x_fb);
		bb.x = std::max<int>(bb.x, 0);

		if (bb.z < bb.x)
		{
			if (g40_enabled()) // G40 O1
				g40_k_bb++;
			TRACE("Degenerate BB", bb);
			return;
		}
	}
	else if (render_pass.can_fb_wraparound)
	{
		bb.z = std::min<int>(bb.z, render_pass.scissor_hi_x_fb);
	}

	assert(bb.z < int(std::max<int>(1, fb_instance.frame.desc.FBW) * PGS_BUFFER_WIDTH_SCALE));
	assert(bb.z < int(std::max<int>(1, ctx.frame.desc.FBW) * PGS_BUFFER_WIDTH_SCALE));

	// Expand render pass BB.
	// If we expand, damage pages.
	// Writing fine-grained FB results is too costly on CPU,
	// but it is an option if we have to in certain scenarios.
	if (bb.x < current_bb.x) { rp_expands = true; current_bb.x = bb.x; }
	if (bb.y < current_bb.y) { rp_expands = true; current_bb.y = bb.y; }
	if (bb.z > current_bb.z) { rp_expands = true; current_bb.z = bb.z; }
	if (bb.w > current_bb.w) { rp_expands = true; current_bb.w = bb.w; }

	if (rp_expands)
	{
		// Damage pages.
		// This is very conservative, and potentially can trigger hazards which should not exist,
		// but this seems unlikely without solid proof that games care.
		auto fb_rect = compute_fb_rect();
		fb_rect.write_mask &= fb_instance.color_write_mask;
		tracker.mark_fb_write(fb_rect);

		if (fb_instance.z_sensitive)
		{
			auto z_rect = compute_z_rect();
			if (fb_instance.z_write)
				tracker.mark_fb_write(z_rect);
			else
				tracker.mark_fb_read(z_rect);
		}
	}

	prim_attr.bb = i16vec4(bb);

	TRACE("Primitive", prim_attr);
	TRACE("DRAW", render_pass.primitive_count);

	render_pass.prim[render_pass.primitive_count] = prim_attr;
	memcpy(render_pass.positions + 3 * render_pass.primitive_count, pos, sizeof(pos));
	memcpy(render_pass.attributes + 3 * render_pass.primitive_count, attr, sizeof(attr));
	render_pass.primitive_count++;
	if (g40_enabled()) // G40 O1
		g40_k_acc++;
	// Commit this here as well. Need to do it after flushing state, since that may reset any tracking state.
	render_pass.last_triangle_is_parallelogram_candidate = is_parallelogram_candidate;
	if (render_pass.current_primitive_is_channel_shuffle)
		render_pass.instances[render_pass.current_instance].has_channel_shuffle = true;

	// Mark state as explicitly not dirty now. If we ended up flushing render pass due to e.g. texture state,
	// some dirty bits will remain set, despite not actually being dirty.
	state_tracker.dirty_flags = 0;
}

template <bool list_primitive, bool fan_primitive, bool quad, unsigned num_vertices>
void GSInterface::drawing_kick_maintain_queue()
{
	static_assert(!fan_primitive || !list_primitive, "Cannot be both fan and list primitive.");
	static_assert(num_vertices <= 3 && num_vertices >= 1, "Num vertices out of range.");
	static_assert(!quad || num_vertices != 3, "Cannot have quad primitive with 3 vertices.");

	if (fan_primitive)
	{
		vertex_queue.pos[1] = vertex_queue.pos[2];
		vertex_queue.attr[1] = vertex_queue.attr[2];
		vertex_queue.count = 2;
	}
	else if (list_primitive)
	{
		vertex_queue.count = 0;
	}

	// Strip primitive will shift queue on next vertex kick.
}

template <bool list_primitive, bool fan_primitive, bool quad, unsigned num_vertices>
void GSInterface::drawing_kick_primitive(bool adc)
{
	if (vertex_queue.count < num_vertices)
		return;

	if (g40_enabled()) // G40 O1	
	{
		g40_k_seen++;
		g40_k_adc += adc ? 1 : 0;
	}
	if (!adc)
	{
		if (!draw_is_degenerate())
			drawing_kick_append<list_primitive, fan_primitive, quad, num_vertices>();
		else
		{
			if (g40_enabled()) // G40 O1
				g40_k_deg++;
			TRACE("Degenerate Draw", DummyBits{});
		}
	}

	// We seem to do queue maintenance regardless after a vertex kick.
	drawing_kick_maintain_queue<list_primitive, fan_primitive, quad, num_vertices>();
}

void GSInterface::drawing_kick_invalid(bool)
{
	// Flush the queue, no nothing otherwise.
	vertex_queue.count = 0;
}

void GSInterface::drawing_kick(bool adc)
{
	(this->*draw_handler)(adc);
	post_draw_kick_handler();
}

template <PRIMType PRIM>
void GSInterface::drawing_kick(bool adc)
{
	// constexpr dispatch
	switch (PRIM)
	{
	case PRIMType::Point:
		drawing_kick_primitive<true, false, true, 1>(adc);
		break;

	case PRIMType::LineList:
		drawing_kick_primitive<true, false, false, 2>(adc);
		break;

	case PRIMType::LineStrip:
		drawing_kick_primitive<false, false, false, 2>(adc);
		break;

	case PRIMType::TriangleList:
		drawing_kick_primitive<true, false, false, 3>(adc);
		break;

	case PRIMType::TriangleStrip:
		drawing_kick_primitive<false, false, false, 3>(adc);
		break;

	case PRIMType::TriangleFan:
		drawing_kick_primitive<false, true, false, 3>(adc);
		break;

	case PRIMType::Sprite:
		drawing_kick_primitive<true, false, true, 2>(adc);
		break;

	default:
		break;
	}

	post_draw_kick_handler();
}

void GSInterface::post_draw_kick_handler()
{
	// If we have buffered up too much, flush out automatically now.
	if (render_pass.pending_palette_updates >= (CLUTInstances - 1) ||
	    render_pass.primitive_count >= MaxPrimitivesPerFlush ||
		render_pass.tex_infos.size() >= MaxTextures ||
		render_pass.state_vectors.size() >= MaxStateVectors)
	{
		flush_pending_transfer(true);
		tracker.flush_render_pass(FlushReason::Overflow);
	}

	tracker.flush_if_memory_pressure();
}

void GSInterface::reset_vertex_queue()
{
	vertex_queue.count = 0;
}

template <int CTX>
void GSInterface::a_d_TEX2(uint64_t payload)
{
	auto &ctx = registers.ctx[CTX];
	auto &preserve = ctx.tex0.desc;

	Reg64<TEX0Bits> tex0{payload};
	tex0.desc.TBP0 = preserve.TBP0;
	tex0.desc.TBW = preserve.TBW;
	tex0.desc.TW = preserve.TW;
	tex0.desc.TH = preserve.TH;
	tex0.desc.TCC = preserve.TCC;
	tex0.desc.TFX = preserve.TFX;

	if (CTX == 0)
		a_d_TEX0_1(tex0.bits);
	else
		a_d_TEX0_2(tex0.bits);
}

void GSInterface::check_pending_transfer()
{
	if (transfer_state.host_to_local_active &&
	    transfer_state.host_to_local_payload.size() >= transfer_state.required_qwords)
	{
		flush_pending_transfer(false);
	}
}

void GSInterface::flush_pending_transfer(bool keep_alive)
{
	if (transfer_state.host_to_local_active &&
	    transfer_state.host_to_local_payload.size() > transfer_state.last_flushed_qwords)
	{
#ifdef PARALLEL_GS_DEBUG
		if (transfer_state.copy.bitbltbuf.bits != registers.bitbltbuf.bits)
			LOGW("Mismatch in bitbltbuf state.\n");
		if (transfer_state.copy.trxpos.bits != registers.trxpos.bits)
			LOGW("Mismatch in trxpos state.\n");
		if (transfer_state.copy.trxreg.bits != registers.trxreg.bits)
			LOGW("Mismatch in trxreg state.\n");
#endif

		auto dst_rect = compute_page_rect(transfer_state.copy.bitbltbuf.desc.DBP,
										  transfer_state.copy.trxpos.desc.DSAX,
										  transfer_state.copy.trxpos.desc.DSAY,
		                                  transfer_state.copy.trxreg.desc.RRW,
		                                  transfer_state.copy.trxreg.desc.RRH,
		                                  transfer_state.copy.bitbltbuf.desc.DBW,
		                                  transfer_state.copy.bitbltbuf.desc.DPSM);

		bool copy_cpu = false;

		transfer_state.copy.host_data = transfer_state.host_to_local_payload.data();
		transfer_state.copy.host_data_size = transfer_state.host_to_local_payload.size() * sizeof(uint64_t);
		transfer_state.copy.host_data_size_offset = transfer_state.last_flushed_qwords * sizeof(uint64_t);
		transfer_state.copy.host_data_size_required = transfer_state.required_qwords * sizeof(uint64_t);

		// This is an arbitrarily chosen value to deal with known problematic games.
		constexpr uint32_t CopyCacheHazardLimit = 400;

		// Can we resolve this on CPU timeline?
		// Only bother with cases which are known to fix hazards.
		// Also, only bother with simple cases. No partial copies, etc ...
		if (dst_rect.page_width == 1 && dst_rect.page_height == 1 &&
		    transfer_state.copy.host_data_size_offset == 0 &&
		    transfer_state.copy.host_data_size == transfer_state.copy.host_data_size_required)
		{
			if (transfer_state.copy.bitbltbuf.desc.DBP == state_tracker.last_cpu_compatible_cache_TBP0)
			{
				state_tracker.current_copy_cache_hazard_counter++;

				if (state_tracker.max_copy_cache_hazard_counter < CopyCacheHazardLimit &&
				    state_tracker.current_copy_cache_hazard_counter == CopyCacheHazardLimit)
				{
					LOGI("Detected bad copy -> cache -> copy -> cache pattern. Injecting CPU wait workaround.\n");
				}

				if (state_tracker.current_copy_cache_hazard_counter > state_tracker.max_copy_cache_hazard_counter)
				{
#ifdef PARALLEL_GS_DEBUG
					LOGW("Increasing copy cache limit to %u.\n", state_tracker.current_copy_cache_hazard_counter);
#endif
					state_tracker.max_copy_cache_hazard_counter = state_tracker.current_copy_cache_hazard_counter;
				}
			}
			else
				state_tracker.current_copy_cache_hazard_counter = 0;

			// Attempt a non-blocking write.
			copy_cpu = tracker.acquire_host_write(dst_rect, renderer.query_timeline());

			// Be conservative, we only want to do this for games that really need it.
			// We must have seen at least a case of N back-to-back copies which all triggered a hazard.
			// This is a hint that the game is doing something deeply questionable.
			// This number of barriers isn't all that bad on first frame (we want to avoid 1000+ barriers).
			// Once we've observed this case, we will start to stall in order to ensure fixed performance.
			bool heuristic_force_cpu_wait = state_tracker.max_copy_cache_hazard_counter >= CopyCacheHazardLimit &&
			                                state_tracker.current_copy_cache_hazard_counter > 4;

			if (!copy_cpu && heuristic_force_cpu_wait)
			{
				if (!tracker.page_is_copy_cached_sensitive(dst_rect))
				{
					// We want to avoid real race conditions. We'll just end up doing work out of order.
					uint64_t host_timeline = tracker.get_submitted_host_write_timeline(dst_rect);
					renderer.wait_timeline(host_timeline);

					// Highly speculative, but if the only concern is a stray FB write, we assume that the ordering
					// between these two is not important, since we'll be overwriting the value anyway.
					// Avoids some extremely bad feedback loops which don't seem to be relevant.
					copy_cpu = true;

					// This is unsynced, but forward to the promotion algorithms that we can ignore hazards.
					tracker.commit_punchthrough_host_write(dst_rect);
				}
				else
				{
					// If we've observed the same upload -> cache pattern over and over again,
					// we assume it's best to stall and do CPU forwarded uploads instead.
					uint64_t host_timeline = tracker.get_host_write_timeline(dst_rect);
					if (host_timeline == UINT64_MAX)
					{
						host_timeline = tracker.mark_submission_timeline(FlushReason::HostAccess);
						renderer.flush_submit(host_timeline);
					}
					renderer.wait_timeline(host_timeline);
					copy_cpu = tracker.acquire_host_write(dst_rect, renderer.query_timeline());
				}

				assert(copy_cpu);
			}
		}
		else
		{
			state_tracker.current_copy_cache_hazard_counter = 0;
		}

		if (copy_cpu)
		{
			// TODO: vram_upload.
			void *mapped = renderer.begin_host_vram_access();
			switch (transfer_state.copy.bitbltbuf.desc.DPSM)
			{
#define PSM_DISPATCH(psm) \
		case psm: \
			vram_upload<psm>(mapped, transfer_state.copy.host_data, \
			                 transfer_state.copy.bitbltbuf.desc.DBP, transfer_state.copy.bitbltbuf.desc.DBW, \
			                 transfer_state.copy.trxpos.desc.DSAX, transfer_state.copy.trxpos.desc.DSAY, \
			                 transfer_state.copy.trxreg.desc.RRW, transfer_state.copy.trxreg.desc.RRH, vram_size - 1); \
			break
			PSM_DISPATCH(PSMT4);
			PSM_DISPATCH(PSMT4HL);
			PSM_DISPATCH(PSMT4HH);
			PSM_DISPATCH(PSMT8);
			PSM_DISPATCH(PSMT8H);
			PSM_DISPATCH(PSMCT16);
			PSM_DISPATCH(PSMCT16S);
			PSM_DISPATCH(PSMZ16);
			PSM_DISPATCH(PSMZ16S);
			PSM_DISPATCH(PSMCT32);
			PSM_DISPATCH(PSMZ32);
			PSM_DISPATCH(PSMCT24);
			PSM_DISPATCH(PSMZ24);
#undef PSM_DISPATCH
			default:
#ifdef PARALLEL_GS_DEBUG
				LOGW("Unsupported CPU upload format %u, falling back.\n", transfer_state.copy.bitbltbuf.desc.DPSM);
#endif
				copy_cpu = false;
				break;
			}

			if (copy_cpu)
				tracker.commit_host_write(dst_rect);
		}

		if (!copy_cpu)
		{
			tracker.mark_transfer_write(dst_rect);
			renderer.copy_vram(transfer_state.copy, dst_rect);
		}

		// Very possible we just have to flush early and we never receive more image data until
		// game kicks a new transfer.
		transfer_state.last_flushed_qwords = uint32_t(transfer_state.host_to_local_payload.size());
		tracker.invalidate_texture_cache(render_pass.clut_instance);
		invalidate_promoted_backbuffer(transfer_state.copy.bitbltbuf.desc.DBP / PGS_BLOCKS_PER_PAGE);

		TRACE_HEADER("VRAM COPY", transfer_state.copy);
	}

	if (!keep_alive)
	{
		transfer_state.host_to_local_payload.clear();
		transfer_state.last_flushed_qwords = 0;
		transfer_state.host_to_local_active = false;
	}
}

void GSInterface::read_transfer_fifo(void *data, uint32_t num_128b_words)
{
	uint32_t to_copy = std::min<uint32_t>(num_128b_words, transfer_state.fifo_readback_128b_size - transfer_state.fifo_readback_128b_offset);

	if (to_copy)
	{
		memcpy(data,
		       transfer_state.fifo_readback.data() + 16 * transfer_state.fifo_readback_128b_offset,
		       to_copy * 16);
	}

	// Assume we'll read 0 if we read past the FIFO.
	if (to_copy < num_128b_words)
		memset(static_cast<uint8_t *>(data) + 16 * to_copy, 0, (num_128b_words - to_copy) * 16);

	transfer_state.fifo_readback_128b_offset += to_copy;
}

void GSInterface::init_transfer()
{
	flush_pending_transfer(false);

	transfer_state.copy.trxdir = registers.trxdir;
	transfer_state.copy.trxreg = registers.trxreg;
	transfer_state.copy.trxpos = registers.trxpos;
	transfer_state.copy.bitbltbuf = registers.bitbltbuf;

	auto XDIR = transfer_state.copy.trxdir.desc.XDIR;

	if (XDIR == LOCAL_TO_LOCAL)
	{
		auto dst_rect = compute_page_rect(transfer_state.copy.bitbltbuf.desc.DBP,
		                                  transfer_state.copy.trxpos.desc.DSAX,
		                                  transfer_state.copy.trxpos.desc.DSAY,
		                                  transfer_state.copy.trxreg.desc.RRW,
		                                  transfer_state.copy.trxreg.desc.RRH,
		                                  transfer_state.copy.bitbltbuf.desc.DBW,
		                                  transfer_state.copy.bitbltbuf.desc.DPSM);

		auto src_rect = compute_page_rect(transfer_state.copy.bitbltbuf.desc.SBP,
		                                  transfer_state.copy.trxpos.desc.SSAX,
		                                  transfer_state.copy.trxpos.desc.SSAY,
		                                  transfer_state.copy.trxreg.desc.RRW,
		                                  transfer_state.copy.trxreg.desc.RRH,
		                                  transfer_state.copy.bitbltbuf.desc.SBW,
		                                  transfer_state.copy.bitbltbuf.desc.SPSM);

		transfer_state.copy.needs_shadow_vram = tracker.mark_transfer_copy(dst_rect, src_rect);
		tracker.invalidate_texture_cache(render_pass.clut_instance);
		renderer.copy_vram(transfer_state.copy, dst_rect);
		invalidate_promoted_backbuffer(transfer_state.copy.bitbltbuf.desc.DBP / PGS_BLOCKS_PER_PAGE);
	}
	else if (XDIR == HOST_TO_LOCAL)
	{
		transfer_state.required_qwords =
				(transfer_state.copy.trxreg.desc.RRW *
				transfer_state.copy.trxreg.desc.RRH *
				get_bits_per_pixel(transfer_state.copy.bitbltbuf.desc.DPSM)) / 64;

		transfer_state.host_to_local_active = transfer_state.required_qwords != 0;
		transfer_state.copy.needs_shadow_vram = false;
		// Await writes to HWREG.
	}
	else if (XDIR == LOCAL_TO_HOST)
	{
		uint32_t required_bytes = (transfer_state.copy.trxreg.desc.RRW *
		                           transfer_state.copy.trxreg.desc.RRH *
		                           get_bits_per_pixel(transfer_state.copy.bitbltbuf.desc.SPSM)) / 8;

		transfer_state.fifo_readback.reserve(required_bytes);
		transfer_state.fifo_readback_128b_offset = 0;
		transfer_state.fifo_readback_128b_size = required_bytes / 16;

		auto src_rect = compute_page_rect(transfer_state.copy.bitbltbuf.desc.SBP,
										  transfer_state.copy.trxpos.desc.SSAX,
										  transfer_state.copy.trxpos.desc.SSAY,
										  transfer_state.copy.trxreg.desc.RRW,
										  transfer_state.copy.trxreg.desc.RRH,
										  transfer_state.copy.bitbltbuf.desc.SBW,
										  transfer_state.copy.bitbltbuf.desc.SPSM);

		uint64_t host_timeline = tracker.get_host_read_timeline(src_rect);

		if (hacks.unsynced_readbacks && renderer.query_timeline() < host_timeline)
		{
			memset(transfer_state.fifo_readback.data(), 0, required_bytes);
			return;
		}

		if (host_timeline == UINT64_MAX)
		{
			host_timeline = tracker.mark_submission_timeline(FlushReason::HostAccess);
			renderer.flush_submit(host_timeline);
			if (debug_mode.deterministic_timeline_query)
				renderer.wait_timeline(host_timeline);
		}
		renderer.wait_timeline(host_timeline);

		void *mapped = renderer.begin_host_vram_access();

		switch (transfer_state.copy.bitbltbuf.desc.SPSM)
		{
#define PSM_DISPATCH(psm) \
		case psm: \
			vram_readback<psm>(transfer_state.fifo_readback.data(), mapped, \
			                   transfer_state.copy.bitbltbuf.desc.SBP, transfer_state.copy.bitbltbuf.desc.SBW, \
			                   transfer_state.copy.trxpos.desc.SSAX, transfer_state.copy.trxpos.desc.SSAY, \
			                   transfer_state.copy.trxreg.desc.RRW, transfer_state.copy.trxreg.desc.RRH, vram_size - 1); \
			break
			PSM_DISPATCH(PSMCT32);
			PSM_DISPATCH(PSMZ32);
			PSM_DISPATCH(PSMCT24);
			PSM_DISPATCH(PSMZ24);
			PSM_DISPATCH(PSMCT16);
			PSM_DISPATCH(PSMCT16S);
			PSM_DISPATCH(PSMZ16);
			PSM_DISPATCH(PSMZ16S);
			PSM_DISPATCH(PSMT8);
			PSM_DISPATCH(PSMT8H);
#undef PSM_DISPATCH

		default:
			LOGW("Unrecognized FIFO readback PSM: %u\n", transfer_state.copy.bitbltbuf.desc.SPSM);
			break;
		}
	}
}

void GSInterface::update_draw_handler()
{
	switch (PRIMType(registers.prim.desc.PRIM))
	{
	case PRIMType::Point:
		draw_handler = &GSInterface::drawing_kick_primitive<true, false, true, 1>;
		break;

	case PRIMType::LineList:
		draw_handler = &GSInterface::drawing_kick_primitive<true, false, false, 2>;
		break;

	case PRIMType::LineStrip:
		draw_handler = &GSInterface::drawing_kick_primitive<false, false, false, 2>;
		break;

	case PRIMType::TriangleList:
		draw_handler = &GSInterface::drawing_kick_primitive<true, false, false, 3>;
		break;

	case PRIMType::TriangleStrip:
		draw_handler = &GSInterface::drawing_kick_primitive<false, false, false, 3>;
		break;

	case PRIMType::TriangleFan:
		draw_handler = &GSInterface::drawing_kick_primitive<false, true, false, 3>;
		break;

	case PRIMType::Sprite:
		draw_handler = &GSInterface::drawing_kick_primitive<true, false, true, 2>;
		break;

	case PRIMType::Invalid:
		draw_handler = &GSInterface::drawing_kick_invalid;
		break;
	}
}

void GSInterface::update_optimized_gif_handler(uint32_t path)
{
	auto &hand = optimized_draw_handler[path];
	hand = nullptr;

	auto &gif_path = paths[path];

	// Only care about PACKED
	if (gif_path.tag.FLG != GIFTagBits::PACKED || gif_path.tag.NLOOP == 0)
		return;

	static const OptimizedPacketHandler STQRGBAXYZHandlers[] = {
		&GSInterface::packed_STQRGBAXYZ<false, PRIMType(0), 1>,
		&GSInterface::packed_STQRGBAXYZ<false, PRIMType(1), 1>,
		&GSInterface::packed_STQRGBAXYZ<false, PRIMType(2), 1>,
		&GSInterface::packed_STQRGBAXYZ<false, PRIMType(3), 1>,
		&GSInterface::packed_STQRGBAXYZ<false, PRIMType(4), 1>,
		&GSInterface::packed_STQRGBAXYZ<false, PRIMType(5), 1>,
		&GSInterface::packed_STQRGBAXYZ<false, PRIMType(6), 1>,
		&GSInterface::packed_STQRGBAXYZ<false, PRIMType(7), 1>,
	};

	static const OptimizedPacketHandler STQRGBAXYZFHandlers[] = {
		&GSInterface::packed_STQRGBAXYZ<true, PRIMType(0), 1>,
		&GSInterface::packed_STQRGBAXYZ<true, PRIMType(1), 1>,
		&GSInterface::packed_STQRGBAXYZ<true, PRIMType(2), 1>,
		&GSInterface::packed_STQRGBAXYZ<true, PRIMType(3), 1>,
		&GSInterface::packed_STQRGBAXYZ<true, PRIMType(4), 1>,
		&GSInterface::packed_STQRGBAXYZ<true, PRIMType(5), 1>,
		&GSInterface::packed_STQRGBAXYZ<true, PRIMType(6), 1>,
		&GSInterface::packed_STQRGBAXYZ<true, PRIMType(7), 1>,
	};

	static const OptimizedPacketHandler UVRGBAXYZHandlers[] = {
		&GSInterface::packed_UVRGBAXYZ<false, PRIMType(0), 1>,
		&GSInterface::packed_UVRGBAXYZ<false, PRIMType(1), 1>,
		&GSInterface::packed_UVRGBAXYZ<false, PRIMType(2), 1>,
		&GSInterface::packed_UVRGBAXYZ<false, PRIMType(3), 1>,
		&GSInterface::packed_UVRGBAXYZ<false, PRIMType(4), 1>,
		&GSInterface::packed_UVRGBAXYZ<false, PRIMType(5), 1>,
		&GSInterface::packed_UVRGBAXYZ<false, PRIMType(6), 1>,
		&GSInterface::packed_UVRGBAXYZ<false, PRIMType(7), 1>,
	};

	static const OptimizedPacketHandler UVRGBAXYZFHandlers[] = {
		&GSInterface::packed_UVRGBAXYZ<true, PRIMType(0), 1>,
		&GSInterface::packed_UVRGBAXYZ<true, PRIMType(1), 1>,
		&GSInterface::packed_UVRGBAXYZ<true, PRIMType(2), 1>,
		&GSInterface::packed_UVRGBAXYZ<true, PRIMType(3), 1>,
		&GSInterface::packed_UVRGBAXYZ<true, PRIMType(4), 1>,
		&GSInterface::packed_UVRGBAXYZ<true, PRIMType(5), 1>,
		&GSInterface::packed_UVRGBAXYZ<true, PRIMType(6), 1>,
		&GSInterface::packed_UVRGBAXYZ<true, PRIMType(7), 1>,
	};

	static const OptimizedPacketHandler ADONLYHandlers[] = {
		&GSInterface::packed_ADONLY<16>,
		&GSInterface::packed_ADONLY<1>,
		&GSInterface::packed_ADONLY<2>,
		&GSInterface::packed_ADONLY<3>,
		&GSInterface::packed_ADONLY<4>,
		&GSInterface::packed_ADONLY<5>,
		&GSInterface::packed_ADONLY<6>,
		&GSInterface::packed_ADONLY<7>,
		&GSInterface::packed_ADONLY<8>,
		&GSInterface::packed_ADONLY<9>,
		&GSInterface::packed_ADONLY<10>,
		&GSInterface::packed_ADONLY<11>,
		&GSInterface::packed_ADONLY<12>,
		&GSInterface::packed_ADONLY<13>,
		&GSInterface::packed_ADONLY<14>,
		&GSInterface::packed_ADONLY<15>,
	};

	constexpr uint64_t STQRGBAXYZ2_Mask =
			(uint32_t(GIFAddr::ST) << 0) |
			(uint32_t(GIFAddr::RGBAQ) << 4) |
			(uint32_t(GIFAddr::XYZ2) << 8);

	constexpr uint64_t STQRGBAXYZF2_Mask =
			(uint32_t(GIFAddr::ST) << 0) |
			(uint32_t(GIFAddr::RGBAQ) << 4) |
			(uint32_t(GIFAddr::XYZF2) << 8);

	constexpr uint64_t STQRGBAXYZ2_TriList_Mask =
			STQRGBAXYZ2_Mask | (STQRGBAXYZ2_Mask << 12) | (STQRGBAXYZ2_Mask << 24);
	constexpr uint64_t STQRGBAXYZF2_TriList_Mask =
			STQRGBAXYZF2_Mask | (STQRGBAXYZF2_Mask << 12) | (STQRGBAXYZF2_Mask << 24);

	constexpr uint64_t STQRGBAXYZ2_LineList_Mask =
			STQRGBAXYZ2_Mask | (STQRGBAXYZ2_Mask << 12);
	constexpr uint64_t STQRGBAXYZF2_LineList_Mask =
			STQRGBAXYZF2_Mask | (STQRGBAXYZF2_Mask << 12);

	constexpr uint64_t UVRGBAXYZ2_Mask =
			(uint32_t(GIFAddr::UV) << 0) |
			(uint32_t(GIFAddr::RGBAQ) << 4) |
			(uint32_t(GIFAddr::XYZ2) << 8);

	constexpr uint64_t UVRGBAXYZF2_Mask =
			(uint32_t(GIFAddr::UV) << 0) |
			(uint32_t(GIFAddr::RGBAQ) << 4) |
			(uint32_t(GIFAddr::XYZF2) << 8);

	constexpr uint32_t STXYZFSTRGBAXYZF_Mask =
			(uint32_t(GIFAddr::ST) << 0) |
			(uint32_t(GIFAddr::XYZF2) << 4) |
			(uint32_t(GIFAddr::ST) << 8) |
			(uint32_t(GIFAddr::RGBAQ) << 12) |
			(uint32_t(GIFAddr::XYZF2) << 16);

	constexpr uint32_t STXYZFSTRGBAXYZ_Mask =
			(uint32_t(GIFAddr::ST) << 0) |
			(uint32_t(GIFAddr::XYZ2) << 4) |
			(uint32_t(GIFAddr::ST) << 8) |
			(uint32_t(GIFAddr::RGBAQ) << 12) |
			(uint32_t(GIFAddr::XYZ2) << 16);

	uint32_t nreg = gif_path.tag.NREG;

	if (nreg == 3 && (gif_path.tag.REGS & 0xfff) == STQRGBAXYZ2_Mask)
	{
		// STQRGBAXYZ2 - Super common STQ comes before RGBA since that's how you update Q correctly,
		// and obviously XYZ2 is the vert/draw kick, so it has to be last.
		hand = STQRGBAXYZHandlers[registers.prim.desc.PRIM];
	}
	else if (nreg == 3 && (gif_path.tag.REGS & 0xfff) == STQRGBAXYZF2_Mask)
	{
		// STQRGBAXYZF2 - Super common STQ comes before RGBA since that's how you update Q correctly,
		// and obviously XYZ2 is the vert/draw kick, so it has to be last.
		hand = STQRGBAXYZFHandlers[registers.prim.desc.PRIM];
	}
	else if (nreg == 3 && (gif_path.tag.REGS & 0xfff) == UVRGBAXYZ2_Mask)
	{
		hand = UVRGBAXYZHandlers[registers.prim.desc.PRIM];
	}
	else if (nreg == 3 && (gif_path.tag.REGS & 0xfff) == UVRGBAXYZF2_Mask)
	{
		hand = UVRGBAXYZFHandlers[registers.prim.desc.PRIM];
	}
	else if (nreg == 5 &&
	         (gif_path.tag.REGS & 0xfffff) == STXYZFSTRGBAXYZF_Mask &&
	         PRIMType(registers.prim.desc.PRIM) == PRIMType::Sprite)
	{
		// Makes sense for sprite rendering. No need to specify RGBA twice. Seen in Legaia 2.
		hand = &GSInterface::packed_STXYZSTRGBAXYZ_sprite<true>;
	}
	else if (nreg == 5 &&
	         (gif_path.tag.REGS & 0xfffff) == STXYZFSTRGBAXYZ_Mask &&
	         PRIMType(registers.prim.desc.PRIM) == PRIMType::Sprite)
	{
		// Makes sense for sprite rendering. No need to specify RGBA twice. Seen in Legaia 2.
		hand = &GSInterface::packed_STXYZSTRGBAXYZ_sprite<false>;
	}
	else if (nreg == 6 &&
	         (gif_path.tag.REGS & 0xffffffull) == STQRGBAXYZ2_LineList_Mask &&
	         PRIMType(registers.prim.desc.PRIM) == PRIMType::LineList)
	{
		// Makes sense for linelist.
		hand = &GSInterface::packed_STQRGBAXYZ<false, PRIMType::LineList, 2>;
	}
	else if (nreg == 6 &&
	         (gif_path.tag.REGS & 0xffffffull) == STQRGBAXYZF2_LineList_Mask &&
	         PRIMType(registers.prim.desc.PRIM) == PRIMType::LineList)
	{
		// Makes sense for linelist.
		hand = &GSInterface::packed_STQRGBAXYZ<true, PRIMType::LineList, 2>;
	}
	else if (nreg == 9 &&
	         (gif_path.tag.REGS & 0xfffffffffull) == STQRGBAXYZ2_TriList_Mask &&
	         PRIMType(registers.prim.desc.PRIM) == PRIMType::TriangleList)
	{
		// Makes sense for trilist.
		hand = &GSInterface::packed_STQRGBAXYZ<false, PRIMType::TriangleList, 3>;
	}
	else if (nreg == 9 &&
	         (gif_path.tag.REGS & 0xfffffffffull) == STQRGBAXYZF2_TriList_Mask &&
	         PRIMType(registers.prim.desc.PRIM) == PRIMType::TriangleList)
	{
		// Makes sense for trilist.
		hand = &GSInterface::packed_STQRGBAXYZ<true, PRIMType::TriangleList, 3>;
	}
	else
	{
		constexpr uint64_t ad_only_mask = uint64_t(GIFAddr::A_D) * 0x1111111111111111ull;
		uint64_t reg_mask = nreg == 0 ? UINT64_MAX : ((1ull << (gif_path.tag.NREG * 4)) - 1);
		if ((gif_path.tag.REGS & reg_mask) == (ad_only_mask & reg_mask))
			hand = ADONLYHandlers[gif_path.tag.NREG];
	}
}

void GSInterface::a_d_PRIM(uint64_t payload)
{
	Reg64<PRIMBits> prim(payload);
	bool prim_delta = registers.prim.desc.PRIM != prim.desc.PRIM;

	if (registers.prmodecont.desc.AC)
	{
		if (registers.prim.desc.CTXT != prim.desc.CTXT)
		{
			state_tracker.dirty_flags |= STATE_DIRTY_DEGENERATE_BIT |
			                             STATE_DIRTY_PRIM_TEMPLATE_BIT |
			                             STATE_DIRTY_TEX_BIT |
			                             STATE_DIRTY_FB_BIT |
			                             STATE_DIRTY_FEEDBACK_BIT |
			                             STATE_DIRTY_SCISSOR_BIT;

			render_pass.ofx = int32_t(registers.ctx[prim.desc.CTXT].xyoffset.desc.OFX);
			render_pass.ofy = int32_t(registers.ctx[prim.desc.CTXT].xyoffset.desc.OFY);
		}

		update_internal_register(registers.prim.bits, payload,
		                         STATE_DIRTY_FEEDBACK_BIT |
		                         STATE_DIRTY_PRIM_TEMPLATE_BIT |
		                         STATE_DIRTY_TEX_BIT |
		                         STATE_DIRTY_STATE_BIT);

		if (!registers.prim.desc.TME)
			state_tracker.dirty_flags &= ~STATE_DIRTY_TEX_BIT;
	}
	else
	{
		if (registers.prim.desc.PRIM != prim.desc.PRIM && prim.desc.AA1)
			state_tracker.dirty_flags |= STATE_DIRTY_PRIM_TEMPLATE_BIT;
		registers.prim.desc.PRIM = prim.desc.PRIM;
	}

	if (prim_delta)
	{
		update_draw_handler();
		// If we're updating PRIM, optimized draw handler is either nullptr anyway,
		// or we're in ADONLY, in which case the optimized handler
		// does not care about PRIM register at all.
		// We don't really know (or should need to know) which GIFPath we're executing in here,
		// so don't try to be clever.
	}

	reset_vertex_queue();

	TRACE("PRIM", registers.prim);
}

void GSInterface::a_d_RGBAQ(uint64_t payload)
{
	registers.rgbaq.bits = payload;
	TRACE("RGBAQ", registers.rgbaq);
}

void GSInterface::a_d_RGBAQUndocumented(uint64_t payload)
{
	// Ridge Racer V.
	a_d_RGBAQ(payload);
}

void GSInterface::a_d_ST(uint64_t payload)
{
	registers.st.bits = payload;
	TRACE("ST", registers.st);
}

void GSInterface::a_d_UV(uint64_t payload)
{
	registers.uv.bits = payload;
	TRACE("UV", registers.uv);
}

void GSInterface::a_d_XYZF2(uint64_t payload)
{
	vertex_kick_xyzf(payload);
	drawing_kick(false);
}

void GSInterface::a_d_XYZ2(uint64_t payload)
{
	vertex_kick_xyz(payload);
	drawing_kick(false);
}

void GSInterface::a_d_TEX0_1(uint64_t payload)
{
	update_internal_register(registers.ctx[0].tex0.bits, payload,
	                         STATE_DIRTY_FEEDBACK_BIT | STATE_DIRTY_STATE_BIT |
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("TEX0_1", registers.ctx[0].tex0);
	handle_tex0_write(0);
	handle_miptbl_gen(0);
}

void GSInterface::a_d_TEX0_2(uint64_t payload)
{
	update_internal_register(registers.ctx[1].tex0.bits, payload,
	                         STATE_DIRTY_FEEDBACK_BIT | STATE_DIRTY_STATE_BIT |
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("TEX0_2", registers.ctx[1].tex0);
	handle_tex0_write(1);
	handle_miptbl_gen(1);
}

void GSInterface::a_d_CLAMP_1(uint64_t payload)
{
	update_internal_register(registers.ctx[0].clamp.bits, payload,
	                         STATE_DIRTY_FEEDBACK_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("CLAMP_1", registers.ctx[0].clamp);
}

void GSInterface::a_d_CLAMP_2(uint64_t payload)
{
	update_internal_register(registers.ctx[1].clamp.bits, payload,
	                         STATE_DIRTY_FEEDBACK_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("CLAMP_2", registers.ctx[1].clamp);
}

void GSInterface::a_d_FOG(uint64_t payload)
{
	registers.fog.bits = payload;
	TRACE("FOG", registers.fog);
}

void GSInterface::a_d_XYZF3(uint64_t payload)
{
	vertex_kick_xyzf(payload);
	drawing_kick(true);
}

void GSInterface::a_d_XYZ3(uint64_t payload)
{
	vertex_kick_xyz(payload);
	drawing_kick(true);
}

void GSInterface::a_d_TEX1_1(uint64_t payload)
{
	update_internal_register(registers.ctx[0].tex1.bits, payload,
	                         STATE_DIRTY_FEEDBACK_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("TEX1_1", registers.ctx[0].tex1);
}

void GSInterface::a_d_TEX1_2(uint64_t payload)
{
	update_internal_register(registers.ctx[1].tex1.bits, payload,
	                         STATE_DIRTY_FEEDBACK_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("TEX1_2", registers.ctx[1].tex1);
}

void GSInterface::a_d_TEX2_1(uint64_t payload) { a_d_TEX2<0>(payload); }
void GSInterface::a_d_TEX2_2(uint64_t payload) { a_d_TEX2<1>(payload); }
void GSInterface::a_d_XYOFFSET_1(uint64_t payload)
{
	registers.ctx[0].xyoffset.bits = payload;
	TRACE("XYOFFSET_1", registers.ctx[0].xyoffset);

	if (registers.prim.desc.CTXT == 0)
	{
		render_pass.ofx = int32_t(registers.ctx[0].xyoffset.desc.OFX);
		render_pass.ofy = int32_t(registers.ctx[0].xyoffset.desc.OFY);
	}
}

void GSInterface::a_d_XYOFFSET_2(uint64_t payload)
{
	registers.ctx[1].xyoffset.bits = payload;
	TRACE("XYOFFSET_2", registers.ctx[1].xyoffset);

	if (registers.prim.desc.CTXT == 1)
	{
		render_pass.ofx = int32_t(registers.ctx[1].xyoffset.desc.OFX);
		render_pass.ofy = int32_t(registers.ctx[1].xyoffset.desc.OFY);
	}
}

void GSInterface::a_d_PRMODECONT(uint64_t payload)
{
	registers.prmodecont.bits = payload;
	TRACE("PRMODECONT", registers.prmodecont);
}

void GSInterface::a_d_PRMODE(uint64_t payload)
{
	if (!registers.prmodecont.desc.AC)
	{
		Reg64<PRIMBits> prim{payload};
		prim.desc.PRIM = registers.prim.desc.PRIM;

		if (registers.prim.desc.CTXT != prim.desc.CTXT)
		{
			state_tracker.dirty_flags |= STATE_DIRTY_DEGENERATE_BIT |
			                             STATE_DIRTY_PRIM_TEMPLATE_BIT |
			                             STATE_DIRTY_TEX_BIT |
			                             STATE_DIRTY_FB_BIT |
			                             STATE_DIRTY_FEEDBACK_BIT |
			                             STATE_DIRTY_SCISSOR_BIT;

			render_pass.ofx = int32_t(registers.ctx[prim.desc.CTXT].xyoffset.desc.OFX);
			render_pass.ofy = int32_t(registers.ctx[prim.desc.CTXT].xyoffset.desc.OFY);
		}

		update_internal_register(registers.prim.bits, prim.bits,
		                         STATE_DIRTY_FEEDBACK_BIT |
		                         STATE_DIRTY_PRIM_TEMPLATE_BIT |
		                         STATE_DIRTY_TEX_BIT |
		                         STATE_DIRTY_STATE_BIT);

		if (!registers.prim.desc.TME)
			state_tracker.dirty_flags &= ~STATE_DIRTY_TEX_BIT;

		TRACE("PRMODE", registers.prim);
	}
}
void GSInterface::a_d_TEXCLUT(uint64_t payload)
{
	registers.texclut.bits = payload;
	TRACE("TEXCLUT", registers.texclut);
}

void GSInterface::a_d_SCANMSK(uint64_t payload)
{
	update_internal_register(registers.scanmsk.bits, payload,
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("SCANMSK", registers.scanmsk);
}

void GSInterface::a_d_MIPTBP1_1(uint64_t payload)
{
	update_internal_register(registers.ctx[0].miptbl_1_3.bits, payload,
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("MIPTBP1_1", registers.ctx[0].miptbl_1_3);
}

void GSInterface::a_d_MIPTBP1_2(uint64_t payload)
{
	update_internal_register(registers.ctx[1].miptbl_1_3.bits, payload,
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("MIPTBP1_2", registers.ctx[1].miptbl_1_3);
}

void GSInterface::a_d_MIPTBP2_1(uint64_t payload)
{
	update_internal_register(registers.ctx[0].miptbl_4_6.bits, payload,
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("MIPTBP2_1", registers.ctx[0].miptbl_4_6);
}

void GSInterface::a_d_MIPTBP2_2(uint64_t payload)
{
	update_internal_register(registers.ctx[1].miptbl_4_6.bits, payload,
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("MIPTBP2_2", registers.ctx[1].miptbl_4_6);
}

void GSInterface::a_d_TEXA(uint64_t payload)
{
	update_internal_register(registers.texa.bits, payload,
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_TEX_BIT);
	TRACE("TEXA", registers.texa);
}

void GSInterface::a_d_FOGCOL(uint64_t payload)
{
	registers.fogcol.bits = payload;
	TRACE("FOGCOL", registers.fogcol);
}

void GSInterface::a_d_TEXFLUSH(uint64_t)
{
	// We cannot rely on TEXFLUSH fully, unfortunately.
	// We'll have to rely on our own tracking,
	// but there are some edge cases where TEXFLUSH is a useful heuristic.
	TRACE("TEXFLUSH", Reg64<DummyBits>{0});

	// This is relevant for textures which are held during a render pass.
	// If game never uses texflush, we should probably ignore any potential feedback hazards which are very unlikely to be real hazards.
	// Newly seen textures ignore texflush however, since games screw that up regularly.
	// We don't want to commit to updating the counter just yet however.
	// We want to make sure this TEXFLUSH is actually targeting a real, cached texture that needs to be invalidated.
	// If it's a new texture we have yet to cache, we will speculate that this is most likely
	// a TEXFLUSH that attempts to invalidate the new texture, and we should not increase the counter.
	state_tracker.texflush_counter_pending = true;
}

void GSInterface::a_d_SCISSOR_1(uint64_t payload)
{
	update_internal_register(registers.ctx[0].scissor.bits, payload,
	                         STATE_DIRTY_SCISSOR_BIT | STATE_DIRTY_DEGENERATE_BIT);
	TRACE("SCISSOR_1", registers.ctx[0].scissor);
}

void GSInterface::a_d_SCISSOR_2(uint64_t payload)
{
	update_internal_register(registers.ctx[1].scissor.bits, payload,
	                         STATE_DIRTY_SCISSOR_BIT | STATE_DIRTY_DEGENERATE_BIT);
	TRACE("SCISSOR_2", registers.ctx[1].scissor);
}

void GSInterface::a_d_ALPHA_1(uint64_t payload)
{
	update_internal_register(registers.ctx[0].alpha.bits, payload,
	                         STATE_DIRTY_STATE_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_DEGENERATE_BIT);
	TRACE("ALPHA_1", registers.ctx[0].alpha);
}

void GSInterface::a_d_ALPHA_2(uint64_t payload)
{
	update_internal_register(registers.ctx[1].alpha.bits, payload,
	                         STATE_DIRTY_STATE_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_DEGENERATE_BIT);
	TRACE("ALPHA_2", registers.ctx[1].alpha);
}

void GSInterface::a_d_DIMX(uint64_t payload)
{
	update_internal_register(registers.dimx.bits, payload, STATE_DIRTY_STATE_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("DIMX", registers.dimx);
}

void GSInterface::a_d_DTHE(uint64_t payload)
{
	update_internal_register(registers.dthe.bits, payload, STATE_DIRTY_STATE_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("DTHE", registers.dthe);
}

void GSInterface::a_d_COLCLAMP(uint64_t payload)
{
	update_internal_register(registers.colclamp.bits, payload, STATE_DIRTY_STATE_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("COLCLAMP", registers.colclamp);
}

void GSInterface::a_d_TEST_1(uint64_t payload)
{
	update_internal_register(registers.ctx[0].test.bits, payload,
	                         STATE_DIRTY_DEGENERATE_BIT |
	                         STATE_DIRTY_STATE_BIT |
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("TEST_1", registers.ctx[0].test);
}

void GSInterface::a_d_TEST_2(uint64_t payload)
{
	update_internal_register(registers.ctx[1].test.bits, payload,
	                         STATE_DIRTY_DEGENERATE_BIT |
	                         STATE_DIRTY_STATE_BIT |
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("TEST_2", registers.ctx[1].test);
}

void GSInterface::a_d_PABE(uint64_t payload)
{
	update_internal_register(registers.pabe.bits, payload, STATE_DIRTY_STATE_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("PABE", registers.pabe);
}

void GSInterface::a_d_FBA_1(uint64_t payload)
{
	update_internal_register(registers.ctx[0].fba.bits, payload, STATE_DIRTY_STATE_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("FBA_1", registers.ctx[0].fba);
}

void GSInterface::a_d_FBA_2(uint64_t payload)
{
	update_internal_register(registers.ctx[1].fba.bits, payload, STATE_DIRTY_STATE_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("FBA_2", registers.ctx[1].fba);
}

void GSInterface::update_internal_register(uint64_t &reg, uint64_t value, StateDirtyFlags flags)
{
	if (reg != value)
	{
		reg = value;
		TRACE("DIRTY", flags);
		state_tracker.dirty_flags |= flags;
	}
}

void GSInterface::a_d_FRAME_1(uint64_t payload)
{
	// G11 local experiment hook (not upstream): draw-target identity.
	if (registers.ctx[0].frame.bits != payload)
	{
		Reg64<FRAMEBits> g11_f;
		g11_f.bits = payload;
		LOGI("G11: FRAME_1 FBP=%u FBW=%u PSM=%u FBMSK=%08x\n",
		     (unsigned)g11_f.desc.FBP, (unsigned)g11_f.desc.FBW,
		     (unsigned)g11_f.desc.PSM, (unsigned)g11_f.desc.FBMSK);
	}
	update_internal_register(registers.ctx[0].frame.bits, payload,
	                         STATE_DIRTY_DEGENERATE_BIT | STATE_DIRTY_FEEDBACK_BIT |
	                         STATE_DIRTY_TEX_BIT | STATE_DIRTY_FB_BIT |
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_SCISSOR_BIT);
	TRACE("FRAME_1", registers.ctx[0].frame);
}

void GSInterface::a_d_FRAME_2(uint64_t payload)
{
	// G11 local experiment hook (not upstream): draw-target identity.
	if (registers.ctx[1].frame.bits != payload)
	{
		Reg64<FRAMEBits> g11_f;
		g11_f.bits = payload;
		LOGI("G11: FRAME_2 FBP=%u FBW=%u PSM=%u FBMSK=%08x\n",
		     (unsigned)g11_f.desc.FBP, (unsigned)g11_f.desc.FBW,
		     (unsigned)g11_f.desc.PSM, (unsigned)g11_f.desc.FBMSK);
	}
	update_internal_register(registers.ctx[1].frame.bits, payload,
	                         STATE_DIRTY_DEGENERATE_BIT | STATE_DIRTY_FEEDBACK_BIT |
	                         STATE_DIRTY_TEX_BIT | STATE_DIRTY_FB_BIT |
	                         STATE_DIRTY_PRIM_TEMPLATE_BIT | STATE_DIRTY_SCISSOR_BIT);
	TRACE("FRAME_2", registers.ctx[1].frame);
}

void GSInterface::a_d_ZBUF_1(uint64_t payload)
{
	update_internal_register(registers.ctx[0].zbuf.bits, payload,
	                         STATE_DIRTY_FEEDBACK_BIT | STATE_DIRTY_DEGENERATE_BIT |
	                         STATE_DIRTY_TEX_BIT | STATE_DIRTY_FB_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("ZBUF_1", registers.ctx[0].zbuf);
}

void GSInterface::a_d_ZBUF_2(uint64_t payload)
{
	update_internal_register(registers.ctx[1].zbuf.bits, payload,
	                         STATE_DIRTY_FEEDBACK_BIT | STATE_DIRTY_DEGENERATE_BIT |
	                         STATE_DIRTY_TEX_BIT | STATE_DIRTY_FB_BIT | STATE_DIRTY_PRIM_TEMPLATE_BIT);
	TRACE("ZBUF_2", registers.ctx[1].zbuf);
}

void GSInterface::a_d_BITBLTBUF(uint64_t payload)
{
	registers.bitbltbuf.bits = payload;
	TRACE("BITBLTBUF", registers.bitbltbuf);
}

void GSInterface::a_d_TRXPOS(uint64_t payload)
{
	registers.trxpos.bits = payload;
	TRACE("TRXPOS", registers.trxpos);
}

void GSInterface::a_d_TRXREG(uint64_t payload)
{
	registers.trxreg.bits = payload;
	TRACE("TRXREG", registers.trxreg);
}

void GSInterface::a_d_TRXDIR(uint64_t payload)
{
	registers.trxdir.bits = payload;
	TRACE("TRXDIR", registers.trxdir);
	init_transfer();
}

// Normally this is written by GIFTag + IMAGE, which effectively spams HWREG with data,
// but nothing stops application from writing HWREG on its own.
void GSInterface::a_d_HWREG(uint64_t payload)
{
	if (transfer_state.host_to_local_active)
	{
		transfer_state.host_to_local_payload.push_back(payload);
		// Flush out transfer if enough data has been received.
		check_pending_transfer();
	}
}

void GSInterface::a_d_HWREG_multi(const uint64_t *payload, size_t count)
{
	if (transfer_state.host_to_local_active)
	{
		transfer_state.host_to_local_payload.insert(transfer_state.host_to_local_payload.end(),
													payload, payload + count);
		// Flush out transfer if enough data has been received.
		check_pending_transfer();
	}
}

// For debugging?
void GSInterface::a_d_SIGNAL(uint64_t payload)
{
	if (signal_interface && signal_interface->on_signal(payload))
		flush();
}

void GSInterface::a_d_FINISH(uint64_t payload)
{
	if (signal_interface && signal_interface->on_finish(payload))
		flush();
}

void GSInterface::a_d_LABEL(uint64_t payload)
{
	if (signal_interface && signal_interface->on_label(payload))
		flush();
}

void GSInterface::reglist_nop(uint64_t) {}
void GSInterface::packed_nop(const void *) {}

template <GSInterface::RegListHandler Handler>
void GSInterface::packed_a_d_forward(const void *words)
{
	(this->*Handler)(*static_cast<const uint64_t *>(words));
}

void GSInterface::packed_RGBAQ(const void *words)
{
	auto &rgba = *static_cast<const PackedRGBAQBits *>(words);
	registers.rgbaq.desc.R = rgba.R;
	registers.rgbaq.desc.G = rgba.G;
	registers.rgbaq.desc.B = rgba.B;
	registers.rgbaq.desc.A = rgba.A;
	registers.rgbaq.desc.Q = registers.internal_q;

	TRACE("RGBAQ", registers.rgbaq);
}

void GSInterface::packed_ST(const void *words)
{
	auto &st = *static_cast<const PackedSTBits *>(words);
	registers.st.desc.S = st.S;
	registers.st.desc.T = st.T;
	registers.internal_q = st.Q;

	TRACE("ST", registers.st);
}

void GSInterface::packed_UV(const void *words)
{
	auto &uv = *static_cast<const PackedUVBits *>(words);
	registers.uv.desc.U = uv.U;
	registers.uv.desc.V = uv.V;
	TRACE("UV", registers.uv);
}

template <bool ADC>
void GSInterface::packed_XYZF(const void *words)
{
	auto &xyzf = *static_cast<const PackedXYZFBits *>(words);
	bool adc = ADC || xyzf.ADC;

	Reg64<XYZFBits> bits = {};
	bits.desc.X = xyzf.X;
	bits.desc.Y = xyzf.Y;
	bits.desc.Z = xyzf.Z;
	bits.desc.F = xyzf.F;
	vertex_kick_xyzf(bits);

	TRACE("ADC", adc);
	drawing_kick(adc);
}

template <bool ADC>
void GSInterface::packed_XYZ(const void *words)
{
	auto &xyz = *static_cast<const PackedXYZBits *>(words);
	bool adc = ADC || xyz.ADC;

	Reg64<XYZBits> bits = {};
	bits.desc.X = xyz.X;
	bits.desc.Y = xyz.Y;
	bits.desc.Z = xyz.Z;
	vertex_kick_xyz(bits);

	TRACE("ADC", adc);
	drawing_kick(adc);
}

template <bool ADC, bool FOG, PRIMType PRIM>
void GSInterface::packed_XYZ(const void *words)
{
	bool adc;

	if (FOG)
	{
		auto &xyzf = *static_cast<const PackedXYZFBits *>(words);
		Reg64<XYZFBits> bits = {};
		bits.desc.X = xyzf.X;
		bits.desc.Y = xyzf.Y;
		bits.desc.Z = xyzf.Z;
		bits.desc.F = xyzf.F;
		vertex_kick_xyzf(bits);
		adc = ADC || xyzf.ADC;
	}
	else
	{
		auto &xyz = *static_cast<const PackedXYZBits *>(words);
		Reg64<XYZBits> bits = {};
		bits.desc.X = xyz.X;
		bits.desc.Y = xyz.Y;
		bits.desc.Z = xyz.Z;
		vertex_kick_xyz(bits);
		adc = ADC || xyz.ADC;
	}

	TRACE("ADC", adc);
	drawing_kick<PRIM>(adc);
}

template <bool FOG, PRIMType PRIM, int factor>
void GSInterface::packed_STQRGBAXYZ(const void *words_, uint32_t num_vertices)
{
	const auto *words = static_cast<const uint8_t *>(words_);
	num_vertices *= factor;

	for (uint32_t i = 0; i < num_vertices; i++, words += 48)
	{
		packed_ST(words + 0);
		packed_RGBAQ(words + 16);
		packed_XYZ<false, FOG, PRIM>(words + 32);
	}
}

template <bool FOG, PRIMType PRIM, int factor>
void GSInterface::packed_UVRGBAXYZ(const void *words_, uint32_t num_vertices)
{
	const auto *words = static_cast<const uint8_t *>(words_);
	num_vertices *= factor;

	for (uint32_t i = 0; i < num_vertices; i++, words += 48)
	{
		packed_UV(words + 0);
		packed_RGBAQ(words + 16);
		packed_XYZ<false, FOG, PRIM>(words + 32);
	}
}

template <bool FOG>
void GSInterface::packed_STXYZSTRGBAXYZ_sprite(const void *words_, uint32_t num_sprites)
{
	const auto *words = static_cast<const uint8_t *>(words_);
	for (uint32_t i = 0; i < num_sprites; i++, words += 80)
	{
		packed_ST(words + 0);
		packed_XYZ<false, FOG, PRIMType::Sprite>(words + 16);
		packed_ST(words + 32);
		packed_RGBAQ(words + 48);
		packed_XYZ<false, FOG, PRIMType::Sprite>(words + 64);
	}
}

void GSInterface::packed_FOG(const void *words)
{
	auto &fog = *static_cast<const PackedFOGBits *>(words);
	registers.fog.desc.FOG = fog.F;
	TRACE("FOG", registers.fog);
}

void GSInterface::setup_handlers()
{
	for (auto &h : ad_handlers)
		h = &GSInterface::reglist_nop;
	for (auto &h : reglist_handlers)
		h = &GSInterface::reglist_nop;
	for (auto &h : packed_handlers)
		h = &GSInterface::packed_nop;

	draw_handler = &GSInterface::drawing_kick_invalid;

#define DECL_REG(reg, addr) ad_handlers[addr] = &GSInterface::a_d_##reg;
#include "gs_register_addr.hpp"
#undef DECL_REG

	reglist_handlers[int(GIFAddr::PRIM)] = &GSInterface::a_d_PRIM;
	reglist_handlers[int(GIFAddr::RGBAQ)] = &GSInterface::a_d_RGBAQ;
	reglist_handlers[int(GIFAddr::ST)] = &GSInterface::a_d_ST;
	reglist_handlers[int(GIFAddr::UV)] = &GSInterface::a_d_UV;
	reglist_handlers[int(GIFAddr::XYZF2)] = &GSInterface::a_d_XYZF2;
	reglist_handlers[int(GIFAddr::XYZ2)] = &GSInterface::a_d_XYZ2;
	reglist_handlers[int(GIFAddr::TEX0_1)] = &GSInterface::a_d_TEX0_1;
	reglist_handlers[int(GIFAddr::TEX0_2)] = &GSInterface::a_d_TEX0_2;
	reglist_handlers[int(GIFAddr::CLAMP_1)] = &GSInterface::a_d_CLAMP_1;
	reglist_handlers[int(GIFAddr::CLAMP_2)] = &GSInterface::a_d_CLAMP_2;
	reglist_handlers[int(GIFAddr::FOG)] = &GSInterface::a_d_FOG;
	reglist_handlers[int(GIFAddr::XYZF3)] = &GSInterface::a_d_XYZF3;
	reglist_handlers[int(GIFAddr::XYZ3)] = &GSInterface::a_d_XYZ3;

	packed_handlers[int(GIFAddr::PRIM)] = &GSInterface::packed_a_d_forward<&GSInterface::a_d_PRIM>;
	packed_handlers[int(GIFAddr::RGBAQ)] = &GSInterface::packed_RGBAQ;
	packed_handlers[int(GIFAddr::ST)] = &GSInterface::packed_ST;
	packed_handlers[int(GIFAddr::UV)] = &GSInterface::packed_UV;
	packed_handlers[int(GIFAddr::TEX0_1)] = &GSInterface::packed_a_d_forward<&GSInterface::a_d_TEX0_1>;
	packed_handlers[int(GIFAddr::TEX0_2)] = &GSInterface::packed_a_d_forward<&GSInterface::a_d_TEX0_2>;
	packed_handlers[int(GIFAddr::CLAMP_1)] = &GSInterface::packed_a_d_forward<&GSInterface::a_d_CLAMP_1>;
	packed_handlers[int(GIFAddr::CLAMP_2)] = &GSInterface::packed_a_d_forward<&GSInterface::a_d_CLAMP_2>;
	packed_handlers[int(GIFAddr::FOG)] = &GSInterface::packed_FOG;
	packed_handlers[int(GIFAddr::XYZF2)] = &GSInterface::packed_XYZF<false>;
	packed_handlers[int(GIFAddr::XYZ2)] = &GSInterface::packed_XYZ<false>;
	packed_handlers[int(GIFAddr::XYZF3)] = &GSInterface::packed_XYZF<true>;
	packed_handlers[int(GIFAddr::XYZ3)] = &GSInterface::packed_XYZ<true>;
}

void *GSInterface::map_vram_write(size_t offset, size_t size)
{
	if (!size)
		return nullptr;

	size_t begin_page = offset / PageSize;
	size_t end_page = (offset + size - 1) / PageSize;

	PageRect page_rect = {};
	page_rect.base_page = begin_page;
	page_rect.page_width = end_page - begin_page + 1;
	page_rect.page_height = 1;
	page_rect.block_mask = UINT32_MAX;
	page_rect.write_mask = UINT32_MAX;

	uint64_t host_write_timeline = tracker.get_host_write_timeline(page_rect);
	if (host_write_timeline == UINT64_MAX)
	{
		host_write_timeline = tracker.mark_submission_timeline(FlushReason::HostAccess);
		renderer.flush_submit(host_write_timeline);
	}

	renderer.wait_timeline(host_write_timeline);

	return static_cast<uint8_t *>(renderer.begin_host_vram_access()) + offset;
}

void GSInterface::end_vram_write(size_t offset, size_t size)
{
	if (!size)
		return;

	size_t begin_page = offset / PageSize;
	size_t end_page = (offset + size - 1) / PageSize;

	PageRect page_rect = {};
	page_rect.base_page = begin_page;
	page_rect.page_width = end_page - begin_page + 1;
	page_rect.page_height = 1;
	page_rect.block_mask = UINT32_MAX;
	page_rect.write_mask = UINT32_MAX;

	renderer.end_host_write_vram_access();
	tracker.commit_host_write(page_rect);
}

const void *GSInterface::map_vram_read(size_t offset, size_t size)
{
	if (!size)
		return nullptr;

	size_t begin_page = offset / PageSize;
	size_t end_page = (offset + size - 1) / PageSize;

	PageRect page_rect = {};
	page_rect.base_page = begin_page;
	page_rect.page_width = end_page - begin_page + 1;
	page_rect.page_height = 1;
	page_rect.block_mask = UINT32_MAX;
	page_rect.write_mask = UINT32_MAX;

	uint64_t host_read_timeline = tracker.get_host_read_timeline(page_rect);
	if (host_read_timeline == UINT64_MAX)
	{
		host_read_timeline = tracker.mark_submission_timeline(FlushReason::HostAccess);
		renderer.flush_submit(host_read_timeline);
	}

	renderer.wait_timeline(host_read_timeline);

	return static_cast<const uint8_t *>(renderer.begin_host_vram_access()) + offset;
}

void GSInterface::flush()
{
	// G11 local experiment hook (not upstream): read-only accumulator
	// census at submit time. Log-only; no behavior change.
	LOGI("G11: flush() acc prims=%u instances=%u states=%u tex=%u\n",
	     render_pass.primitive_count, render_pass.num_instances,
	     (unsigned)render_pass.state_vectors.size(), (unsigned)render_pass.tex_infos.size());
	flush_pending_transfer(true);
	uint64_t value = tracker.mark_submission_timeline();
	renderer.flush_submit(value);
	if (debug_mode.deterministic_timeline_query)
		renderer.wait_timeline(value);
}

void GSInterface::clobber_register_state()
{
	state_tracker.dirty_flags = STATE_DIRTY_ALL_BITS;
	update_draw_handler();
	// We don't know which path will start executing so we cannot infer anything from pending GIFTags.
	// Defer until we receive a fresh GIFTag header.
	for (uint32_t i = 0; i < 4; i++)
		update_optimized_gif_handler(i);
}

void GSInterface::write_register(RegisterAddr addr, uint64_t payload)
{
	(this->*ad_handlers[int(addr)])(payload);
}

template <int count>
void GSInterface::packed_ADONLY(const void *words, uint32_t num_loops)
{
	auto *ad = static_cast<const Reg128<PackedADBits> *>(words);
	for (uint32_t i = 0; i < num_loops; i++)
		for (int j = 0; j < count; j++, ad++)
			write_register(RegisterAddr(ad->desc.ADDR), ad->desc.data);
}

void GSInterface::gif_transfer(uint32_t path_index, const void *data, size_t size)
{
	// Transfers are in units of 128 bits.
	assert(path_index < 4);
	assert(size % 16 == 0);
	size /= 16;
	auto &path = paths[path_index];

	if (size == 0)
		return;

	const auto *qwords = static_cast<const GIFTagBits *>(data);
	const auto *word64 = static_cast<const uint64_t *>(data);

	// This can be optimized a lot, but keep it simple for now.

	uint32_t nreg = path.tag.NREG == 0 ? 16 : path.tag.NREG;

	for (size_t i = 0; i < size; )
	{
		bool needs_gif_tag = path.loop == path.tag.NLOOP;
		if (needs_gif_tag)
		{
			path.tag = qwords[i];
			TRACE_HEADER("GIFTag", path.tag);
			if (path.tag.FLG == GIFTagBits::PACKED && path.tag.PRE != 0 && path.tag.NLOOP)
			{
				// Set PRIM register.
				a_d_PRIM(path.tag.PRIM);
			}

			update_optimized_gif_handler(path_index);

			path.loop = 0;
			path.reg = 0;
			i++;
			nreg = path.tag.NREG == 0 ? 16 : path.tag.NREG;

			if (path.tag.NLOOP)
				registers.internal_q = 1.0f;
		}
		else
		{
			if (path.reg == 0 && optimized_draw_handler[path_index])
			{
				// Should this divide be optimized to use divide by constant trick?
				uint32_t nloops_to_run = std::min<uint32_t>(size / nreg, path.tag.NLOOP - path.loop);
				(this->*optimized_draw_handler[path_index])(&qwords[i], nloops_to_run);
				i += nloops_to_run * nreg;
				path.loop += nloops_to_run;
			}
			else if (path.tag.FLG == GIFTagBits::PACKED)
			{
				auto addr = uint32_t(path.tag.REGS >> (4 * path.reg)) & 0xf;
				path.reg++;

				if (GIFAddr(addr) == GIFAddr::A_D)
				{
					auto *ad = reinterpret_cast<const Reg128<PackedADBits> *>(&qwords[i]);
					write_register(RegisterAddr(ad->desc.ADDR), ad->desc.data);
				}
				else
					(this->*packed_handlers[addr])(&qwords[i]);

				i++;

				bool end_of_loop = path.reg == nreg;
				if (end_of_loop)
				{
					path.loop++;
					path.reg = 0;
				}
			}
			else if (path.tag.FLG == GIFTagBits::REGLIST)
			{
				// Number of 128-bit words is ceil(NLOOP * NREG / 2).
				// Loops can be tightly packed if NREG is odd.

				for (uint32_t j = 0; j < 2; j++)
				{
					auto addr = uint32_t(path.tag.REGS >> (4 * path.reg)) & 0xf;
					path.reg++;
					(this->*reglist_handlers[addr])(word64[2 * i + j]);

					bool end_of_loop = path.reg == nreg;
					if (end_of_loop)
					{
						path.loop++;
						path.reg = 0;
						if (path.loop == path.tag.NLOOP)
							break;
					}
				}

				i++;
			}
			else
			{
				// IMAGE
				// Spam HWREG.
				auto num_loops = std::min<size_t>(size - i, path.tag.NLOOP - path.loop);
				a_d_HWREG_multi(word64 + 2 * i, num_loops * 2);
				i += num_loops;
				path.loop += num_loops;
			}
		}
	}
}

RegisterState &GSInterface::get_register_state()
{
	return registers;
}

const RegisterState &GSInterface::get_register_state() const
{
	return registers;
}

PrivRegisterState &GSInterface::get_priv_register_state()
{
	return priv_registers;
}

const PrivRegisterState &GSInterface::get_priv_register_state() const
{
	return priv_registers;
}

GIFPath &GSInterface::get_gif_path(uint32_t path)
{
	return paths[path];
}

const GIFPath &GSInterface::get_gif_path(uint32_t path) const
{
	return paths[path];
}

void GSInterface::set_debug_mode(const DebugMode &mode)
{
	debug_mode = mode;
	renderer.set_enable_timestamps(mode.timestamps);
}

void GSInterface::set_hacks(const Hacks &hacks_)
{
	hacks = hacks_;

	if (!hacks.backbuffer_promotion)
	{
		for (auto &b : promoted_backbuffers)
			b = {};
		num_promoted_backbuffers = 0;
	}
}

PromotedBackbuffer *GSInterface::find_promoted_backbuffer(uint32_t fbp)
{
	for (uint32_t i = 0; i < num_promoted_backbuffers; i++)
		if (promoted_backbuffers[i].FBP == fbp)
			return &promoted_backbuffers[i];

	return nullptr;
}

void GSInterface::invalidate_promoted_backbuffer(uint32_t fbp)
{
	for (uint32_t i = 0; i < num_promoted_backbuffers; i++)
		if (promoted_backbuffers[i].FBP == fbp)
			promoted_backbuffers[i].img.reset();
}

void GSInterface::promote_render_pass_to_backbuffer(const RenderPass &rp)
{
	if (!hacks.backbuffer_promotion)
		return;

	for (uint32_t instance = 0; instance < rp.num_instances; instance++)
	{
		auto &inst = rp.instances[instance];
		uint32_t fbp = inst.fb.frame.desc.FBP;
		auto *promoted = find_promoted_backbuffer(fbp);
		if (!promoted)
			continue;

		promoted->img.reset();

		ivec2 lo = ivec2(INT32_MAX);
		ivec2 hi = ivec2(INT32_MIN);
		bool is_valid_blit = true;
		uint32_t promoted_tex_index = UINT32_MAX;

		for (uint32_t prim = 0; prim < rp.num_primitives; prim++)
		{
			uint32_t state = render_pass.prim[prim].state;
			uint32_t state_index = (state >> STATE_INDEX_BIT_OFFSET) & ((1 << STATE_INDEX_BIT_COUNT) - 1);
			uint32_t prim_instance = (state >> STATE_VERTEX_RENDER_PASS_INSTANCE_OFFSET) &
			                         ((1 << STATE_VERTEX_RENDER_PASS_INSTANCE_COUNT) - 1);

			if (prim_instance != instance)
				continue;

			if ((state & (1u << STATE_BIT_SPRITE)) == 0)
			{
				is_valid_blit = false;
				break;
			}

			auto &s = render_pass.state_vectors[state_index];
			if ((s.combiner & COMBINER_TME_BIT) == 0)
			{
				is_valid_blit = false;
				break;
			}

			uint32_t tex_index = (render_pass.prim[prim].tex >> TEX_TEXTURE_INDEX_OFFSET) &
			                     ((1 << TEX_TEXTURE_INDEX_BITS) - 1);

			// Ignore texture feedback.
			if (tex_index >= rp.num_textures)
			{
				is_valid_blit = false;
				break;
			}

			if (promoted_tex_index != UINT32_MAX && tex_index != promoted_tex_index)
			{
				is_valid_blit = false;
				break;
			}

			promoted_tex_index = tex_index;

			// Ignore if mipmapped.
			if (rp.textures[tex_index].view->get_image().get_create_info().levels > 1)
			{
				is_valid_blit = false;
				break;
			}

			ivec2 uv0;
			ivec2 uv1;

			// Crude heuristic, just look at the bounding box being sampled.
			// Promote that region to be the new backbuffer.
			if ((state & (1 << STATE_BIT_PERSPECTIVE)) != 0)
			{
				auto &attr0 = render_pass.attributes[3 * prim + 0];
				auto &attr1 = render_pass.attributes[3 * prim + 1];
				constexpr float rounding_epsilon = 1.0f / 1024.0f;
				uv0 = ivec2((attr0.st / attr0.q) * rp.textures[tex_index].info.sizes.xy() + rounding_epsilon);
				uv1 = ivec2((attr1.st / attr1.q) * rp.textures[tex_index].info.sizes.xy() + rounding_epsilon);
			}
			else
			{
				uv0 = ivec2(render_pass.attributes[3 * prim + 0].uv) >> PGS_SUBPIXEL_BITS;
				uv1 = ivec2(render_pass.attributes[3 * prim + 1].uv) >> PGS_SUBPIXEL_BITS;
			}

			lo = muglm::min(lo, uv0);
			lo = muglm::min(lo, uv1);
			hi = muglm::max(hi, uv0);
			hi = muglm::max(hi, uv1);
		}

		if (!is_valid_blit)
			continue;

		auto &img = render_pass.tex_infos[promoted_tex_index].view->get_image();

		// For purposes of promoting frame to field rendering more gracefully, assume
		// that any field based blit should round the blit rect to something reasonable.
		// Assume that if the height is close enough to these standard progressive heights, we should snap to that.
		// Crude heuristic, but it's very hard to deduce frame to field blending without making some assumptions.
		if (lo.y <= 2)
			lo.y = 0;

		static const int snapped_heights[] = { 224 * 2, 240 * 2, 256 * 2, 288 * 2 };

		if (lo.y == 0)
		{
			for (int snapped_height : snapped_heights)
			{
				if (std::abs(hi.y - snapped_height) <= 2)
				{
					hi.y = snapped_height;
					break;
				}
			}
		}

		lo = muglm::clamp(lo, ivec2(0), ivec2(img.get_width(), img.get_height()));
		hi = muglm::clamp(hi, ivec2(0), ivec2(img.get_width(), img.get_height()));

		// Assume bottom-right rules leading to rounding down here.
		// Missing a pixel is better than including too many pixels since it may contain garbage.

		ivec2 extent = hi - lo;
		if (extent.x > 0 && extent.y > 0)
		{
			// The cached texture has been recycled, so need to make a copy of it for later scanout.
			// We don't know yet if we need to resolve the multisamples, so just copy all samples as-is.
			// This should happen at most once per frame, so it's not a big deal.
			promoted->img = renderer.copy_cached_texture(
					img, VkRect2D{{lo.x, lo.y}, {uint32_t(extent.x), uint32_t(extent.y)}});
		}
	}
}

void GSInterface::register_backbuffer_promotion_fbp(uint32_t fbp)
{
	for (uint32_t i = 0; i < num_promoted_backbuffers; i++)
	{
		if (promoted_backbuffers[i].FBP == fbp)
		{
			if (i)
			{
				auto tmp = std::move(promoted_backbuffers[i]);
				for (uint32_t j = i; j; j--)
					promoted_backbuffers[j] = std::move(promoted_backbuffers[j - 1]);
				promoted_backbuffers[0] = std::move(tmp);
			}
			return;
		}
	}

	if (num_promoted_backbuffers == MaxPromotedBackbuffers)
		num_promoted_backbuffers--;
	for (uint32_t j = num_promoted_backbuffers; j; j--)
		promoted_backbuffers[j] = std::move(promoted_backbuffers[j - 1]);
	promoted_backbuffers[0] = {};
	promoted_backbuffers[0].FBP = fbp;
	num_promoted_backbuffers++;
}

ScanoutResult GSInterface::vsync(const VSyncInfo &info_)
{
	auto info = info_;
	auto ffmd = priv_registers.smode2.FFMD;

	// Somewhat of a hack to avoid having to turn off force_progressive manually.
	if (state_tracker.has_complex_scanmsk_timeout)
	{
		info.force_progressive = false;
		// If the game stops using complex scanmasks, eventually flip back to normal operation.
		state_tracker.has_complex_scanmsk_timeout--;
	}

	const Vulkan::Image *promoted1 = nullptr;
	const Vulkan::Image *promoted2 = nullptr;
	if (priv_registers.pmode.EN1)
	{
		auto *promoted = find_promoted_backbuffer(priv_registers.dispfb1.FBP);
		promoted1 = promoted ? promoted->img.get() : nullptr;
	}

	if (priv_registers.pmode.EN2)
	{
		auto *promoted = find_promoted_backbuffer(priv_registers.dispfb2.FBP);
		promoted2 = promoted ? promoted->img.get() : nullptr;
	}

	// If the promoted buffer looks like it was downsampled from a full frame, we can patch away FFMD
	// and avoid unnecessary deinterlacing, also avoids doing field-adaptive rendering by mistake,
	// since the game is clearly not actually field rendered.
	if ((promoted1 && promoted1->get_height() > 350) ||
	    (promoted2 && promoted2->get_height() > 350))
	{
		priv_registers.smode2.FFMD = 0;
	}

	// If the game is field rendered, and we want high-res scanout,
	// snap to field resolution rather than native pixels.
	// Ideally we should be able to achieve a "perfect" deinterlacer this way,
	// since we can jitter a full pixel rather than half pixel.
	render_pass.field_aware_rendering = info.high_resolution_scanout &&
	                                    sampling_rate_y_log2 &&
	                                    priv_registers.smode2.FFMD &&
	                                    priv_registers.smode2.INT &&
	                                    priv_registers.smode1.CMOD != SMODE1Bits::CMOD_PROGRESSIVE;

	renderer.set_field_aware_super_sampling(render_pass.field_aware_rendering);


	auto result = renderer.vsync(priv_registers, info,
	                             sampling_rate_x_log2, sampling_rate_y_log2,
								 promoted1, promoted2);
	{
		// G31 local diagnostic (not upstream): in-vsync() (b1)/(b2) split
		// probe. Promotion is statically OFF for this run (Hacks defaults
		// false + set_hacks() uncalled anywhere), so promoted1/2 are
		// predicted null on all 16 vsyncs; this dump confirms on-device
		// and captures the DISPFB1-region VRAM bytes at sample time.
		// Region B (FBP112 pages 112..223) is predicted == load (no
		// writer ever touches it, G30); region A (FBP0) is the
		// live-write control. FNV-1a reuses the SAME truncated basis as
		// G29/G30 (G29-E1). Read-only: one host map + wait AFTER the
		// vsync's own submission (no reordering); promotion identity is
		// logged but no GPU readback is attempted (a present promoted
		// image routes to OTHER/stop regardless -- checksum deferred).
		static unsigned g31_seq = 0;
		unsigned g31_p1null = promoted1 == nullptr ? 1 : 0;
		unsigned g31_p1w = g31_p1null ? 0 : promoted1->get_width();
		unsigned g31_p1h = g31_p1null ? 0 : promoted1->get_height();
		unsigned g31_p1l = g31_p1null ? 0 : (unsigned)promoted1->get_create_info().layers;
		unsigned g31_p1f = g31_p1null ? 0 : (unsigned)promoted1->get_format();
		unsigned g31_p2null = promoted2 == nullptr ? 1 : 0;
		LOGI("G31: state seq=%u EN1=%u EN2=%u DISPFB1=%u/%u/%u/%u/%u DSP1=%u/%u/%u/%u/%u/%u SM=%u/%u/%u nprom=%u hack=%d p1null=%u p1=%ux%ux%ux%u p2null=%u phase=%u.\n",
		     g31_seq,
		     (unsigned)priv_registers.pmode.EN1, (unsigned)priv_registers.pmode.EN2,
		     (unsigned)priv_registers.dispfb1.FBP, (unsigned)priv_registers.dispfb1.FBW,
		     (unsigned)priv_registers.dispfb1.PSM, (unsigned)priv_registers.dispfb1.DBX,
		     (unsigned)priv_registers.dispfb1.DBY,
		     (unsigned)priv_registers.display1.DW, (unsigned)priv_registers.display1.DH,
		     (unsigned)priv_registers.display1.MAGH, (unsigned)priv_registers.display1.MAGV,
		     (unsigned)priv_registers.display1.DX, (unsigned)priv_registers.display1.DY,
		     (unsigned)priv_registers.smode1.CMOD, (unsigned)priv_registers.smode2.INT,
		     (unsigned)priv_registers.smode2.FFMD,
		     num_promoted_backbuffers, int(hacks.backbuffer_promotion),
		     g31_p1null, g31_p1w, g31_p1h, g31_p1l, g31_p1f, g31_p2null, info.phase);
		const size_t g31_n = 224 * 8192;
		const size_t g31_half = 917504;
		const uint8_t *g31_vram = static_cast<const uint8_t *>(map_vram_read(0, g31_n));
		if (g31_vram)
		{
			uint64_t g31_fa = 1469598103934665603ull;
			uint64_t g31_fb = 1469598103934665603ull;
			size_t g31_za = 0;
			size_t g31_zb = 0;
			for (size_t g31_i = 0; g31_i < g31_half; g31_i++)
			{
				g31_fa ^= g31_vram[g31_i];
				g31_fa *= 1099511628211ull;
				g31_za += g31_vram[g31_i] != 0;
				g31_fb ^= g31_vram[g31_half + g31_i];
				g31_fb *= 1099511628211ull;
				g31_zb += g31_vram[g31_half + g31_i] != 0;
			}
			LOGI("G31: bytes seq=%u A fnv=%016llx nz=%u head=%02x%02x%02x%02x%02x%02x%02x%02x B fnv=%016llx nz=%u head=%02x%02x%02x%02x%02x%02x%02x%02x.\n",
			     g31_seq, (unsigned long long)g31_fa, (unsigned)g31_za,
			     g31_vram[0], g31_vram[1], g31_vram[2], g31_vram[3],
			     g31_vram[4], g31_vram[5], g31_vram[6], g31_vram[7],
			     (unsigned long long)g31_fb, (unsigned)g31_zb,
			     g31_vram[g31_half], g31_vram[g31_half + 1], g31_vram[g31_half + 2], g31_vram[g31_half + 3],
			     g31_vram[g31_half + 4], g31_vram[g31_half + 5], g31_vram[g31_half + 6], g31_vram[g31_half + 7]);
		}
		else
		{
			LOGI("G31: bytes seq=%u map failed.\n", g31_seq);
		}
		g31_seq++;
	}

	if (hacks.backbuffer_promotion && result.image)
	{
		if (priv_registers.pmode.EN1)
			register_backbuffer_promotion_fbp(priv_registers.dispfb1.FBP);
		if (priv_registers.pmode.EN2)
			register_backbuffer_promotion_fbp(priv_registers.dispfb2.FBP);
	}

	// Restore FFMD state.
	priv_registers.smode2.FFMD = ffmd;
	return result;
}

bool GSInterface::vsync_can_skip(const VSyncInfo &info) const
{
	return state_tracker.has_complex_scanmsk_timeout == 0 && renderer.vsync_can_skip(priv_registers, info);
}

FlushStats GSInterface::consume_flush_stats()
{
	return renderer.consume_flush_stats();
}

double GSInterface::get_accumulated_timestamps(TimestampType type) const
{
	return renderer.get_accumulated_timestamps(type);
}

void GSInterface::set_signal_interface(SignalInterface *iface)
{
	signal_interface = iface;
}
}
