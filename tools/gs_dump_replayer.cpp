// SPDX-FileCopyrightText: 2024 Arntzen Software AS
// SPDX-FileContributor: Hans-Kristian Arntzen
// SPDX-FileContributor: Runar Heyer
// SPDX-License-Identifier: LGPL-3.0+

#include "gs_renderer.hpp"
#include "device.hpp"
#include "context.hpp"
#include "gs_dump_parser.hpp"
#include "global_managers_init.hpp"
#include "filesystem.hpp"
#include "thread_group.hpp"
#include "gs_interface.hpp"
#include "gs_dump_generator.hpp"
#include "cli_parser.hpp"
#include "timer.hpp"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>

using namespace Vulkan;
using namespace ParallelGS;
using namespace Util;

static void print_help()
{
	LOGI("Usage: parallel-gs-replayer <dump.gs>\n"
		"\t[--ssaa <rate>]\n"
		"\t[--strided]\n"
		"\t[--full]\n"
		"\t[--iterations <count>]\n"
		"\t[--high-res-scanout]\n"
		"\t[--ssaa-textures]\n"
		"\t[--conservative-crtc]\n"
		"\t[--disable-sampler-feedback]\n");
}

// G42 local experiment hook (not upstream): Mesa Turnip contrast on Adreno.
// Env-gated (PGS_G42_TURNIP=<Turnip driver .so path>), default = system
// loader (zero behavior change). Adrenotools-format Turnip builds export a
// single OBJECT symbol "HMI" (legacy pointer-layout hw module: tag@0, api
// versions@4, id@8, name@16, author@24, methods@32); we open it as the
// Vulkan HAL module and hand its GetInstanceProcAddr to Granite's loader.
// Every Turnip-path failure exits non-zero: a silent fallback to the
// system driver would mislabel the run.
struct G42HalModuleMethods
{
	int (*open)(const void *module, const char *id, void **device);
};
struct G42HalModule
{
	uint32_t tag;
	uint16_t module_api_version;
	uint16_t hal_api_version;
	const char *id;
	const char *name;
	const char *author;
	G42HalModuleMethods *methods;
};
struct G42HalDevice
{
	uint32_t tag;
	uint32_t version;
	void *module;
	int (*close)(void *dev);
	void *enumerate_instance_ext;
	void *create_instance;
	PFN_vkGetInstanceProcAddr get_instance_proc_addr;
};

static bool g42_init_loader()
{
	const char *turnip = getenv("PGS_G42_TURNIP");
	LOGI("G42: Vulkan loader: %s.\n", turnip ? turnip : "(system)");
	if (!turnip)
		return Context::init_loader(nullptr);
	void *mod = dlopen(turnip, RTLD_NOW | RTLD_LOCAL);
	if (!mod)
	{
		LOGE("G42: dlopen failed: %s.\n", dlerror());
		return false;
	}
	G42HalModule *hmi = reinterpret_cast<G42HalModule *>(dlsym(mod, "HMI"));
	if (!hmi)
	{
		LOGE("G42: no HMI symbol: %s.\n", dlerror());
		return false;
	}
	LOGI("G42: HMI tag=%08x id=%s name=%s author=%s methods=%p.\n",
	     hmi->tag, hmi->id ? hmi->id : "(null)",
	     hmi->name ? hmi->name : "(null)",
	     hmi->author ? hmi->author : "(null)",
	     (const void *)hmi->methods);
	if (hmi->tag != 0x48574d54 || !hmi->methods || !hmi->methods->open)
	{
		LOGE("G42: HMI layout mismatch.\n");
		return false;
	}
	void *dev = nullptr;
	int rc = hmi->methods->open(hmi, "vulkan0", &dev);
	LOGI("G42: HAL open rc=%d dev=%p.\n", rc, dev);
	if (rc != 0 || !dev)
	{
		LOGE("G42: HAL open failed.\n");
		return false;
	}
	G42HalDevice *vdev = reinterpret_cast<G42HalDevice *>(dev);
	LOGI("G42: HAL dev tag=%08x get_proc=%p.\n",
	     vdev->tag, (const void *)vdev->get_instance_proc_addr);
	if (!vdev->get_instance_proc_addr)
	{
		LOGE("G42: HAL has no GetInstanceProcAddr.\n");
		return false;
	}
	return Context::init_loader(vdev->get_instance_proc_addr);
}

int main(int argc, char **argv)
{
	std::string dump_path;
	DebugMode debug_mode;
	debug_mode.feedback_render_target = true;
	debug_mode.draw_mode = DebugMode::DrawDebugMode::None;
	unsigned total_iterations = 1;
	bool high_res_scanout = false;
	bool conservative_crtc = false;
	GSOptions opts = {};

	CLICallbacks cbs;
	cbs.add("--help", [&](CLIParser &parser) { parser.end(); print_help(); });
	cbs.add("--ssaa", [&](CLIParser &parser) { opts.super_sampling = SuperSampling(parser.next_uint()); });
	cbs.add("--strided", [&](CLIParser &) { debug_mode.draw_mode = DebugMode::DrawDebugMode::Strided; });
	cbs.add("--full", [&](CLIParser &) { debug_mode.draw_mode = DebugMode::DrawDebugMode::Full; });
	cbs.add("--iterations", [&](CLIParser &parser) { total_iterations = parser.next_uint(); });
	cbs.add("--high-res-scanout", [&](CLIParser &) { high_res_scanout = true; });
	cbs.add("--ssaa-textures", [&](CLIParser &) { opts.super_sampled_textures = true; });
	cbs.add("--disable-sampler-feedback", [&](CLIParser &) { debug_mode.disable_sampler_feedback = true; });
	cbs.add("--conservative-crtc", [&](CLIParser &) { conservative_crtc = true; });
	cbs.default_handler = [&](const char *arg) { dump_path = arg; };

	CLIParser cli_parser(std::move(cbs), argc - 1, argv + 1);
	if (!cli_parser.parse())
	{
		print_help();
		return EXIT_FAILURE;
	}

	if (cli_parser.is_ended_state())
		return EXIT_SUCCESS;

	if (dump_path.empty())
	{
		LOGE("Must provide dump.\n");
		print_help();
		return EXIT_FAILURE;
	}

	if (!g42_init_loader())
		return EXIT_FAILURE;

	Context ctx;
	ctx.set_num_thread_indices(1);
	if (!ctx.init_instance_and_device(nullptr, 0, nullptr, 0,
	                                  CONTEXT_CREATION_ENABLE_PUSH_DESCRIPTOR_BIT |
	                                  CONTEXT_CREATION_ENABLE_DESCRIPTOR_HEAP_BIT |
									  CONTEXT_CREATION_ENABLE_DESCRIPTOR_BUFFER_BIT))
		return EXIT_FAILURE;

	Device device;
	device.set_context(ctx);
	device.init_frame_contexts(4);

	GSInterface iface;
	if (!iface.init(&device, opts))
		return EXIT_FAILURE;

	bool use_rdoc = Device::init_renderdoc_capture();
	// G26 local narrowing (not upstream): deliver ONLY disable_sampler_feedback;
	// force feedback_render_target back to the interface default (false) so the
	// G24 ride-along is removed. Field surgery on the parsed struct: every other
	// parsed field passes through unchanged, one variable vs G24.
	debug_mode.feedback_render_target = false;
	iface.set_debug_mode(debug_mode);
	LOGI("G26: debug_mode delivered (disable_sampler_feedback=%d, feedback_render_target=%d, use_rdoc=%d).\n", int(debug_mode.disable_sampler_feedback), int(debug_mode.feedback_render_target), int(use_rdoc));
	if (use_rdoc)
	{
		device.begin_renderdoc_capture();
	}

	GSDumpParser parser;
	if (!parser.open(argv[1], 4 * 1024 * 1024, &iface))
		return EXIT_FAILURE;

	unsigned iterations = 0;
	uint64_t start_ns = 0;
	unsigned vsyncs = 0;

	// G8 local experiment hook (not upstream): first-iterate scanout stash +
	// per-vsync flush stats on the last pass. Loop-body execution count (and the
	// wall-per-unit timing derived from it) is unchanged vs G7: in-loop work is
	// counter reads, one log line, and a handle swap; PPM readback/write happens
	// after end_ns. Readback mirrors gs_repro_replayer.cpp (as G7's hook did).
	// G10 local experiment hook (not upstream): bounded last-pass scanout
	// series (one PPM per iterate-true vsync) + per-pass stats resets so
	// cold/warm passes separate (G8's stacked takes are corrected in G10).
	// Loop-body execution count unchanged; PPM I/O stays after end_ns.
	auto save_scanout_ppm = [&](ScanoutResult shot, const std::string &ppm_path, const char *tag) {
		if (!shot.image)
		{
			LOGE("G8: no image in %s vsync result.\n", tag);
			return;
		}
		BufferHandle shot_readback;
		{
			auto cmd = device.request_command_buffer();
			cmd->image_barrier(*shot.image, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
			                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
			                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

			BufferCreateInfo shot_bufinfo = {};
			shot_bufinfo.size = shot.image->get_width() * shot.image->get_height() * sizeof(uint32_t);
			shot_bufinfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			shot_bufinfo.domain = BufferDomain::CachedHost;
			shot_readback = device.create_buffer(shot_bufinfo);
			cmd->copy_image_to_buffer(*shot_readback, *shot.image, 0, {},
			                          { shot.image->get_width(), shot.image->get_height(), 1 },
			                          0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 });
			cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
			             VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
			device.submit(cmd);
			device.wait_idle();
		}

		auto *shot_pixels = static_cast<const uint8_t *>(
				device.map_host_buffer(*shot_readback, MEMORY_ACCESS_READ_BIT));

		// G29 local diagnostic (not upstream): readback-vs-render ladder. P1
		// (above) is the original CachedHost readback. P2 re-reads the same
		// image through a coherent Host buffer (names a CachedHost-map zero);
		// P3 re-reads through a coherent Host buffer behind a full-mask
		// barrier (names a GPU-visibility zero). Checksums + nonzero counts go
		// to logcat only; files still carry P1 bytes. A once-only VRAM
		// checksum (direct mapped read, no image copy) proves/disproves
		// upload+raster landing upstream of scanout.
		auto g29_checksum = [](const uint8_t *px, size_t n, uint64_t *fnv, size_t *nz) {
			uint64_t h = 1469598103934665603ull;
			size_t c = 0;
			for (size_t i = 0; i < n; i++)
			{
				h ^= px[i];
				h *= 1099511628211ull;
				c += px[i] != 0;
			}
			*fnv = h;
			*nz = c;
		};
		{
			size_t shot_n = size_t(shot.image->get_width()) * shot.image->get_height() * 4;
			uint64_t p1_fnv = 0, p2_fnv = 0, p3_fnv = 0;
			size_t p1_nz = 0, p2_nz = 0, p3_nz = 0;
			g29_checksum(shot_pixels, shot_n, &p1_fnv, &p1_nz);
			for (int pass = 2; pass <= 3; pass++)
			{
				BufferHandle alt_readback;
				const uint8_t *alt_pixels = nullptr;
				{
					auto cmd = device.request_command_buffer();
					if (pass == 2)
					{
						cmd->image_barrier(*shot.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
						                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
					}
					else
					{
						cmd->image_barrier(*shot.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
						                   VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
						                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
						                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
					}
					BufferCreateInfo alt_bufinfo = {};
					alt_bufinfo.size = shot_n;
					alt_bufinfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
					alt_bufinfo.domain = BufferDomain::Host;
					alt_readback = device.create_buffer(alt_bufinfo);
					cmd->copy_image_to_buffer(*alt_readback, *shot.image, 0, {},
					                          { shot.image->get_width(), shot.image->get_height(), 1 },
					                          0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 });
					cmd->barrier(VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
					             VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
					device.submit(cmd);
					device.wait_idle();
				}
				if (alt_readback)
				{
					alt_pixels = static_cast<const uint8_t *>(
							device.map_host_buffer(*alt_readback, MEMORY_ACCESS_READ_BIT));
				}
				if (alt_pixels)
				{
					if (pass == 2)
						g29_checksum(alt_pixels, shot_n, &p2_fnv, &p2_nz);
					else
						g29_checksum(alt_pixels, shot_n, &p3_fnv, &p3_nz);
				}
			}
			LOGI("G29: ladder %s P1 fnv=%016llx nz=%llu P2 fnv=%016llx nz=%llu P3 fnv=%016llx nz=%llu (n=%llu).\n",
			     tag, (unsigned long long)p1_fnv, (unsigned long long)p1_nz,
			     (unsigned long long)p2_fnv, (unsigned long long)p2_nz,
			     (unsigned long long)p3_fnv, (unsigned long long)p3_nz, (unsigned long long)shot_n);
			static bool g29_vram_done = false;
			if (!g29_vram_done)
			{
				g29_vram_done = true;
				size_t vram_n = opts.vram_size;
				const uint8_t *vram = static_cast<const uint8_t *>(iface.map_vram_read(0, vram_n));
				if (vram)
				{
					uint64_t v_fnv = 0;
					size_t v_nz = 0;
					g29_checksum(vram, vram_n, &v_fnv, &v_nz);
					LOGI("G29: vram fnv=%016llx nz=%llu (n=%llu).\n",
					     (unsigned long long)v_fnv, (unsigned long long)v_nz, (unsigned long long)vram_n);
				}
				else
				{
					LOGI("G29: vram map failed.\n");
				}
			}
		}
		unsigned shot_w = shot.image->get_width(), shot_h = shot.image->get_height();
		FILE *ppm = fopen(ppm_path.c_str(), "wb");
		if (ppm)
		{
			fprintf(ppm, "P6\n%u %u\n255\n", shot_w, shot_h);
			for (unsigned i = 0; i < shot_w * shot_h; i++)
				fwrite(shot_pixels + i * 4, 1, 3, ppm);
			fclose(ppm);
			LOGI("G8: wrote %s-vsync scanout %ux%u (internal %ux%u, mode %ux%u, interlaced %d, phase %u) to %s.\n",
			     tag, shot_w, shot_h, shot.internal_width, shot.internal_height,
			     shot.mode_width, shot.mode_height, (int)shot.interlaced, shot.interlace_phase,
			     ppm_path.c_str());
		}
		else
		{
			LOGE("G8: failed to write %s.\n", ppm_path.c_str());
		}
	};

	// Baseline: discard counters accumulated during open/state load so the
	// per-vsync series below starts clean. Counter read/reset only.
	iface.consume_flush_stats();
	std::vector<ScanoutResult> g10_series;
	g10_series.reserve(16);
	unsigned g8_seq = 0;

	do
	{
		bool g8_last_pass = (iterations == total_iterations - 1);
		g8_seq = 0;
		// G10: per-pass reset (restart() rewinds the file, not the stats).
		iface.consume_flush_stats();
		LOGI("G10: pass %u stats reset\n", iterations);
		for (;;)
		{
			LOGI("Running frame ...\n");
			if (iterations > 0)
				vsyncs++;
			bool g8_more = parser.iterate_until_vsync(high_res_scanout, conservative_crtc);
			if (!g8_more)
				break;
			{
				FlushStats g8_st = iface.consume_flush_stats();
				LOGI("G10: pass %u vsync #%u prims=%u passes=%u pal=%u copies=%u copy_threads=%u copy_barriers=%u scratch=%llu img=%llu\n",
				     iterations, g8_seq, g8_st.num_primitives, g8_st.num_render_passes, g8_st.num_palette_updates,
				     g8_st.num_copies, g8_st.num_copy_threads, g8_st.num_copy_barriers,
				     (unsigned long long)g8_st.allocated_scratch_memory,
				     (unsigned long long)g8_st.allocated_image_memory);
				if (g8_last_pass)
					g10_series.push_back(parser.consume_vsync_result());
			}
			g8_seq++;
		}

		// G30 local diagnostic (not upstream): post-run VRAM content probe.
		// G29's VRAM checksum read LOAD state (this loop's trailing
		// restart() below reloads all 4 MB before G29's post-loop read --
		// host load FNV matches G29's line byte-exact). This dump runs
		// BEFORE the reload, on the last pass only: per-8KB-page FNV-1a
		// (SAME truncated basis as G29, G29-E1) + nonzero count + first 8
		// bytes, to logcat only. Read-only; images/regs untouched (adds
		// <1 s inside the timed region -- expect inflated ms/VBlank).
		if (g8_last_pass)
		{
			auto g30_checksum = [](const uint8_t *px, size_t n, uint64_t *fnv, size_t *nz) {
				uint64_t h = 1469598103934665603ull;
				size_t c = 0;
				for (size_t i = 0; i < n; i++)
				{
					h ^= px[i];
					h *= 1099511628211ull;
					c += px[i] != 0;
				}
				*fnv = h;
				*nz = c;
			};
			size_t g30_n = opts.vram_size;
			const uint8_t *g30_vram = static_cast<const uint8_t *>(iface.map_vram_read(0, g30_n));
			if (g30_vram)
			{
				for (size_t g30_p = 0; g30_p < g30_n / 8192; g30_p++)
				{
					const uint8_t *pg = g30_vram + g30_p * 8192;
					uint64_t f = 0;
					size_t z = 0;
					g30_checksum(pg, 8192, &f, &z);
					LOGI("G30: vpage pass=%u page=%03u fnv=%016llx nz=%u head=%02x%02x%02x%02x%02x%02x%02x%02x.\n",
					     iterations, (unsigned)g30_p, (unsigned long long)f, (unsigned)z,
					     pg[0], pg[1], pg[2], pg[3], pg[4], pg[5], pg[6], pg[7]);
				}
			}
			else
			{
				LOGI("G30: vram map failed.\n");
			}
		}

		if (!parser.restart())
		{
			LOGE("Failed to rewind capture.\n");
			break;
		}

		HeapBudget budget[VK_MAX_MEMORY_HEAPS] = {};
		device.get_memory_budget(budget);

		for (uint32_t i = 0; i < device.get_memory_properties().memoryHeapCount; i++)
		{
			LOGI("Memory usage - Heap %u - %s - %llu MiB / %llu MiB\n", i,
			     (device.get_memory_properties().memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 ? "DEVICE" : "HOST",
			     static_cast<unsigned long long>(budget[i].tracked_usage / (1024 * 1024)),
			     static_cast<unsigned long long>(budget[i].device_usage / (1024 * 1024)));
		}

		if (iterations == 0)
			start_ns = Util::get_current_time_nsecs();
	} while (++iterations < total_iterations);
	uint64_t end_ns = Util::get_current_time_nsecs();

	double total_time = double(end_ns - start_ns) * 1e-9;
	LOGI("Total time per VBlank: %.3f ms\n", 1e3 * total_time / double(vsyncs));

	// G10: bounded series (last pass, one PPM per iterate-true) + G8's
	// FIRST/LAST saves kept for continuity (series front/back).
	// After end_ns; timing untouched.
	if (!g10_series.empty())
	{
		for (size_t g10_i = 0; g10_i < g10_series.size(); g10_i++)
		{
			char g10_tag[32];
			snprintf(g10_tag, sizeof(g10_tag), "vsync%zu", g10_i);
			save_scanout_ppm(g10_series[g10_i], dump_path + ".g10-" + g10_tag + ".ppm", g10_tag);
		}
		save_scanout_ppm(g10_series.front(), dump_path + ".g8-first.ppm", "first");
		save_scanout_ppm(g10_series.back(), dump_path + ".g8-last.ppm", "last");
	}
	else
		LOGE("G10: no iterate-true on the last pass; no scanout series.\n");

	LOGI("Done!\n");

	if (use_rdoc)
		device.end_renderdoc_capture();
}
