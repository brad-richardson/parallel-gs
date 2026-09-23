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
#include <stddef.h>
#include <string.h>
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
	// G43 fix (not upstream): G42 omitted hw_device_t's reserved block
	// (AOSP hardware.h, LP64: tag@0 version@4 module@8 reserved[12]@16..112
	// close@112), so every hwvulkan_device_t op read 96 bytes early and
	// GetInstanceProcAddr came from a reserved (zero) slot. Layout now
	// matches hwvulkan.h: close@112 EnumerateInstanceExtensionProperties@120
	// CreateInstance@128 GetInstanceProcAddr@136.
	uint64_t reserved[12];
	int (*close)(void *dev);
	void *enumerate_instance_ext;
	void *create_instance;
	PFN_vkGetInstanceProcAddr get_instance_proc_addr;
};
#if __SIZEOF_POINTER__ == 8
static_assert(offsetof(G42HalDevice, close) == 112, "G43: hw_device_t::close offset");
static_assert(offsetof(G42HalDevice, enumerate_instance_ext) == 120, "G43: EnumerateInstanceExtensionProperties offset");
static_assert(offsetof(G42HalDevice, create_instance) == 128, "G43: CreateInstance offset");
static_assert(offsetof(G42HalDevice, get_instance_proc_addr) == 136, "G43: GetInstanceProcAddr offset");
#endif

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
	// G43 (not upstream): mapped range of the Turnip .so, so the four
	// device pointers below can be checked against it (in-.so code vs
	// null/garbage). dladdr gives base+file; on Android the full range
	// comes from /proc/self/maps.
	{
		Dl_info g43_info = {};
		if (dladdr(hmi, &g43_info) && g43_info.dli_fname)
		{
			LOGI("G43: HMI maps at base=%p file=%s.\n",
			     g43_info.dli_fbase, g43_info.dli_fname);
#ifdef __ANDROID__
			FILE *maps = fopen("/proc/self/maps", "r");
			if (maps)
			{
				char line[1024];
				while (fgets(line, sizeof(line), maps))
					if (strstr(line, g43_info.dli_fname))
						LOGI("G43: map %s", line);
				fclose(maps);
			}
#endif
		}
		else
		{
			LOGI("G43: dladdr(HMI) failed.\n");
		}
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
	LOGI("G43: HAL dev tag=%08x close=%p enum_ext=%p create_inst=%p get_proc=%p.\n",
	     vdev->tag, (const void *)vdev->close,
	     vdev->enumerate_instance_ext, vdev->create_instance,
	     (const void *)vdev->get_instance_proc_addr);
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

	do
	{
		do
		{
			LOGI("Running frame ...\n");
			if (iterations > 0)
				vsyncs++;
		} while (parser.iterate_until_vsync(high_res_scanout, conservative_crtc));

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

	LOGI("Done!\n");

	if (use_rdoc)
		device.end_renderdoc_capture();
}
