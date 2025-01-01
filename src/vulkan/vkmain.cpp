#include "vkmain.hpp"
#include "common.h"

const int BM = 64;
const int BN = 64;
const int BK = 16;
const int TM = 4;
const int TN = 4;

bool checkValidationLayerSupport() {
	std::vector<const char*> requiredLayers = { "VK_LAYER_KHRONOS_validation" };

	uint32_t layerCount;
	auto result = vk::enumerateInstanceLayerProperties(&layerCount, nullptr);
	assert(result == vk::Result::eSuccess);

	std::vector<vk::LayerProperties> availableLayers(layerCount);
	result = vk::enumerateInstanceLayerProperties(&layerCount, availableLayers.data());
	assert(result == vk::Result::eSuccess);

	for (const char* layerName : requiredLayers) {
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound) {
			return false;
		}
	}
	return true;
}

struct PushConstants {
	int M;
	int N;
	int K;
	float alpha;
	float beta;
};

int main(int argc, char *argv[]) {

	////////////////////////////////////////////////////////////////////////
	//                          VULKAN INSTANCE                           //
	////////////////////////////////////////////////////////////////////////
	vk::ApplicationInfo AppInfo{
		"VulkanCompute",      // Application Name
		1,                    // Application Version
		nullptr,              // Engine Name or nullptr
		0,                    // Engine Version
		VK_API_VERSION_1_2    // Vulkan API version
	};

	const std::vector<const char*> Layers = { "VK_LAYER_KHRONOS_validation" };
	vk::InstanceCreateInfo InstanceCreateInfo(
		vk::InstanceCreateFlags(), // Flags
		&AppInfo,                  // Application Info
		Layers.size(),             // Layers count
		Layers.data()              // Layers
	);

	// Check for validation layer support
	if (!checkValidationLayerSupport()) {
		std::cerr << "Validation layers requested but not available!" << std::endl;
		return -1;
	}
	vk::Instance Instance = vk::createInstance(InstanceCreateInfo);


	////////////////////////////////////////////////////////////////////////
	//                          PHYSICAL DEVICE                           //
	////////////////////////////////////////////////////////////////////////
	vk::PhysicalDevice PhysicalDevice = Instance.enumeratePhysicalDevices().front();
	vk::PhysicalDeviceProperties DeviceProps = PhysicalDevice.getProperties();
	const uint32_t ApiVersion = DeviceProps.apiVersion;
	vk::PhysicalDeviceLimits DeviceLimits = DeviceProps.limits;

	if (!DeviceLimits.timestampComputeAndGraphics) {
		std::cerr << "Timestamp queries not supported!" << std::endl;
		return -1;
	}

#ifdef DEBUG
	std::cout << "Device Name    : " << DeviceProps.deviceName << std::endl;
	std::cout << "Vulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "." << VK_VERSION_MINOR(ApiVersion) << "." << VK_VERSION_PATCH(ApiVersion) << std::endl;
	std::cout << "Max Compute Shared Memory Size: " << DeviceLimits.maxComputeSharedMemorySize / 1024 << " KB" << std::endl;
#endif // DEBUG



	////////////////////////////////////////////////////////////////////////
	//                            QUEUE FAMILY                            //
	////////////////////////////////////////////////////////////////////////
	std::vector<vk::QueueFamilyProperties> QueueFamilyProps = PhysicalDevice.getQueueFamilyProperties();
	auto PropIt = std::find_if(QueueFamilyProps.begin(), QueueFamilyProps.end(), [](const vk::QueueFamilyProperties& Prop) {
		return Prop.queueFlags & vk::QueueFlagBits::eCompute;
	});
	const uint32_t ComputeQueueFamilyIndex = std::distance(QueueFamilyProps.begin(), PropIt);

#ifdef DEBUG
	std::cout << "Compute Queue Family Index: " << ComputeQueueFamilyIndex << std::endl;
#endif // DEBUG


	////////////////////////////////////////////////////////////////////////
	//                               DEVICE                               //
	////////////////////////////////////////////////////////////////////////
	float queuePriorities = 1.0f;
	vk::DeviceQueueCreateInfo DeviceQueueCreateInfo(
		vk::DeviceQueueCreateFlags(),   // Flags
		ComputeQueueFamilyIndex,        // Queue Family Index
		1,                              // Number of Queues
		&queuePriorities
	);
	vk::DeviceCreateInfo DeviceCreateInfo(
		vk::DeviceCreateFlags(),   // Flags
		1,
		&DeviceQueueCreateInfo      // Device Queue Create Info struct
	);
	vk::Device Device = PhysicalDevice.createDevice(DeviceCreateInfo);

	////////////////////////////////////////////////////////////////////////
	//                         Allocating Memory                          //
	////////////////////////////////////////////////////////////////////////
	// Create the required buffers for the application
	// Allocate the memory to back the buffers
	// Bind the buffers to the memory

	// K, N, M of the matrices
	int N = 6;
	int SIZE[] = {128, 256, 512, 1024, 2048, 4096};
	long max_size = SIZE[N - 1];

	int repeat_times = 5;

	// Create buffers
	const uint32_t NumElements = max_size;
	const uint32_t MaxBufferSize = NumElements * NumElements * sizeof(float);

	vk::BufferCreateInfo BufferCreateInfo{
		vk::BufferCreateFlags(),                    // Flags
		MaxBufferSize,                                 // Size
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,    // Usage
		vk::SharingMode::eExclusive,                // Sharing mode
		1,                                          // Number of queue family indices
		&ComputeQueueFamilyIndex                    // List of queue family indices
	};
	vk::Buffer InBufferA = Device.createBuffer(BufferCreateInfo);
	vk::Buffer InBufferB = Device.createBuffer(BufferCreateInfo);
	vk::Buffer OutBuffer = Device.createBuffer(BufferCreateInfo);

	vk::BufferCreateInfo StagingBufferCreateInfo{
		vk::BufferCreateFlags(),
		MaxBufferSize,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		vk::SharingMode::eExclusive,
		1,
		&ComputeQueueFamilyIndex
	};
	vk::Buffer StagingBuffer = Device.createBuffer(StagingBufferCreateInfo);

	// Memory req
	vk::MemoryRequirements InBufferAMemoryRequirements = Device.getBufferMemoryRequirements(InBufferA);
	vk::MemoryRequirements InBufferBMemoryRequirements = Device.getBufferMemoryRequirements(InBufferB);
	vk::MemoryRequirements OutBufferMemoryRequirements = Device.getBufferMemoryRequirements(OutBuffer);
	vk::MemoryRequirements StagingBufferMemoryRequirements = Device.getBufferMemoryRequirements(StagingBuffer);

	// query
	vk::PhysicalDeviceMemoryProperties MemoryProperties = PhysicalDevice.getMemoryProperties();

	uint32_t MemoryTypeIndex = uint32_t(~0);
	vk::DeviceSize MemoryHeapSize = uint32_t(~0);
	for (uint32_t CurrentMemoryTypeIndex = 0; CurrentMemoryTypeIndex < MemoryProperties.memoryTypeCount; ++CurrentMemoryTypeIndex)
	{
		vk::MemoryType MemoryType = MemoryProperties.memoryTypes[CurrentMemoryTypeIndex];
		if ((vk::MemoryPropertyFlagBits::eHostVisible & MemoryType.propertyFlags) &&
			(vk::MemoryPropertyFlagBits::eHostCoherent & MemoryType.propertyFlags))
		{
			MemoryHeapSize = MemoryProperties.memoryHeaps[MemoryType.heapIndex].size;
			MemoryTypeIndex = CurrentMemoryTypeIndex;
			break;
		}
	}

#ifdef DEBUG
	std::cout << "Memory Type Index: " << MemoryTypeIndex << std::endl;
	std::cout << "Memory Heap Size : " << MemoryHeapSize / 1024 / 1024 / 1024 << " GB" << std::endl;
#endif // DEBUG

	uint32_t GPUMemoryTypeIndex = uint32_t(~0);
	vk::DeviceSize GPUMemoryHeapSize = uint32_t(~0);
	for (uint32_t CurrentMemoryTypeIndex = 0; CurrentMemoryTypeIndex < MemoryProperties.memoryTypeCount; ++CurrentMemoryTypeIndex)
	{
		vk::MemoryType MemoryType = MemoryProperties.memoryTypes[CurrentMemoryTypeIndex];
		if ((vk::MemoryPropertyFlagBits::eDeviceLocal & MemoryType.propertyFlags))
		{
			GPUMemoryHeapSize = MemoryProperties.memoryHeaps[MemoryType.heapIndex].size;
			GPUMemoryTypeIndex = CurrentMemoryTypeIndex;
			break;
		}
	}

#ifdef DEBUG
	std::cout << "GPU Memory Type Index: " << MemoryTypeIndex << std::endl;
	std::cout << "GPU Memory Heap Size : " << MemoryHeapSize / 1024 / 1024 / 1024 << " GB" << std::endl;
#endif // DEBUG


	// Allocate memory
	vk::MemoryAllocateInfo InBufferAMemoryAllocateInfo(InBufferAMemoryRequirements.size, GPUMemoryTypeIndex);
	vk::MemoryAllocateInfo InBufferBMemoryAllocateInfo(InBufferBMemoryRequirements.size, GPUMemoryTypeIndex);
	vk::MemoryAllocateInfo OutBufferMemoryAllocateInfo(OutBufferMemoryRequirements.size, GPUMemoryTypeIndex);
	vk::MemoryAllocateInfo StagingBufferMemoryAllocateInfo(StagingBufferMemoryRequirements.size, MemoryTypeIndex);

	vk::DeviceMemory InBufferAMemory = Device.allocateMemory(InBufferAMemoryAllocateInfo);
	vk::DeviceMemory InBufferBMemory = Device.allocateMemory(InBufferBMemoryAllocateInfo);
	vk::DeviceMemory OutBufferMemory = Device.allocateMemory(OutBufferMemoryAllocateInfo);
	vk::DeviceMemory StagingBufferMemory = Device.allocateMemory(StagingBufferMemoryAllocateInfo);


	// Map memory and write
	float* StagingBufferPtr = static_cast<float *>(Device.mapMemory(StagingBufferMemory, 0, MaxBufferSize));

	// for (uint32_t I = 0; I < NumElements; ++I) {
	// StagingBufferPtr[I] = I - 0.1f;
	// }


	// Bind buffers to memory
	Device.bindBufferMemory(InBufferA, InBufferAMemory, 0);
	Device.bindBufferMemory(InBufferB, InBufferBMemory, 0);
	Device.bindBufferMemory(OutBuffer, OutBufferMemory, 0);
	Device.bindBufferMemory(StagingBuffer, StagingBufferMemory, 0);

	////////////////////////////////////////////////////////////////////////
	//                              PIPELINE                              //
	////////////////////////////////////////////////////////////////////////

	// Shader module
	std::vector<char> ShaderContents;
	if (std::ifstream ShaderFile{ "shaders/07_sgemm_vectorize2.comp.spv", std::ios::binary | std::ios::ate }) {
		const size_t FileSize = ShaderFile.tellg();
		ShaderFile.seekg(0);
		ShaderContents.resize(FileSize, '\0');
		ShaderFile.read(ShaderContents.data(), FileSize);
	}

	vk::ShaderModuleCreateInfo ShaderModuleCreateInfo(
		vk::ShaderModuleCreateFlags(),                                // Flags
		ShaderContents.size(),                                        // Code size
		reinterpret_cast<const uint32_t*>(ShaderContents.data()));    // Code
	vk::ShaderModule ShaderModule = Device.createShaderModule(ShaderModuleCreateInfo);

	// Descriptor Set Layout
	// The layout of data to be passed to pipelin
	const std::vector<vk::DescriptorSetLayoutBinding> DescriptorSetLayoutBinding = {
		{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
		{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
		{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
	};
	vk::DescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(
		vk::DescriptorSetLayoutCreateFlags(),
		DescriptorSetLayoutBinding);
	vk::DescriptorSetLayout DescriptorSetLayout = Device.createDescriptorSetLayout(DescriptorSetLayoutCreateInfo);

	vk::PushConstantRange pushConstantRange(
		vk::ShaderStageFlagBits::eCompute,
		0,
		sizeof(PushConstants)
	);
	std::vector<vk::PushConstantRange> pushConstantRanges = {pushConstantRange};


	// Pipeline Layout
	vk::PipelineLayoutCreateInfo PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), DescriptorSetLayout, pushConstantRange);
	vk::PipelineLayout PipelineLayout = Device.createPipelineLayout(PipelineLayoutCreateInfo);
	vk::PipelineCache PipelineCache = Device.createPipelineCache(vk::PipelineCacheCreateInfo());

	// Compute Pipeline
	vk::PipelineShaderStageCreateInfo PipelineShaderCreateInfo(
		vk::PipelineShaderStageCreateFlags(),  // Flags
		vk::ShaderStageFlagBits::eCompute,     // Stage
		ShaderModule,                          // Shader Module
		"main"                                 // Shader Entry Point
	);
	vk::ComputePipelineCreateInfo ComputePipelineCreateInfo(
		vk::PipelineCreateFlags(),    // Flags
		PipelineShaderCreateInfo,     // Shader Create Info struct
		PipelineLayout                // Pipeline Layout
	);
	vk::Pipeline ComputePipeline = Device.createComputePipeline(PipelineCache, ComputePipelineCreateInfo).value;

	////////////////////////////////////////////////////////////////////////
	//                          DESCRIPTOR SETS                           //
	////////////////////////////////////////////////////////////////////////
	// Descriptor sets must be allocated in a vk::DescriptorPool, so we need to create one first
	vk::DescriptorPoolSize DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 3);
	vk::DescriptorPoolCreateInfo DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 1, DescriptorPoolSize);
	vk::DescriptorPool DescriptorPool = Device.createDescriptorPool(DescriptorPoolCreateInfo);

	// Allocate descriptor sets, update them to use buffers:
	vk::DescriptorSetAllocateInfo DescriptorSetAllocInfo(DescriptorPool, 1, &DescriptorSetLayout);
	const std::vector<vk::DescriptorSet> DescriptorSets = Device.allocateDescriptorSets(DescriptorSetAllocInfo);
	vk::DescriptorSet DescriptorSet = DescriptorSets.front();
	vk::DescriptorBufferInfo InBufferAInfo(InBufferA, 0, MaxBufferSize);
	vk::DescriptorBufferInfo InBufferBInfo(InBufferB, 0, MaxBufferSize);
	vk::DescriptorBufferInfo OutBufferInfo(OutBuffer, 0, MaxBufferSize);

	const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
		{DescriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &InBufferAInfo},
		{DescriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &InBufferBInfo},
		{DescriptorSet, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &OutBufferInfo},
	};
	Device.updateDescriptorSets(WriteDescriptorSets, {});

	////////////////////////////////////////////////////////////////////////
	//                         SUBMIT WORK TO GPU                         //
	////////////////////////////////////////////////////////////////////////


	// Create QueryPool for timestamping
	std::array<uint64_t, 2> queryResults;

	vk::QueryPoolCreateInfo queryPoolInfo(vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, static_cast<uint32_t>(queryResults.size()));
	vk::QueryPool queryPool = Device.createQueryPool(queryPoolInfo);


	// Command Pool
	vk::CommandPoolCreateInfo CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, ComputeQueueFamilyIndex);

	vk::CommandPool CommandPool = Device.createCommandPool(CommandPoolCreateInfo);
	// Allocate Command buffer from Pool
	vk::CommandBufferAllocateInfo CommandBufferAllocInfo(
		CommandPool,                         // Command Pool
		vk::CommandBufferLevel::ePrimary,    // Level
		1);                                  // Num Command Buffers

	const std::vector<vk::CommandBuffer> CmdBuffers = Device.allocateCommandBuffers(CommandBufferAllocInfo);
	vk::CommandBuffer CmdBuffer = CmdBuffers.front();

	// Record commands
	vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eSimultaneousUse);


	for(int s=0; s<N; s++) {
		int size = SIZE[s];

		CmdBuffer.reset();
		CmdBuffer.begin(CmdBufferBeginInfo);

		CmdBuffer.resetQueryPool(queryPool, 0, 2);

		CmdBuffer.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, queryPool, 0);

		randomize_matrix(StagingBufferPtr, size * size);

		vk::BufferCopy CopyRegion(0, 0, size * size * sizeof(float));
		CmdBuffer.copyBuffer(StagingBuffer, InBufferA, { CopyRegion });

		vk::MemoryBarrier memoryBarrier(
			vk::AccessFlagBits::eTransferWrite,  // Source access mask 
			vk::AccessFlagBits::eShaderRead      // Destination access mask
		);

		CmdBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,  // Source stage
			vk::PipelineStageFlagBits::eComputeShader,  // Destination stage
			vk::DependencyFlags(),
			1, &memoryBarrier,
			0, nullptr,
			0, nullptr
		);

#ifdef VERBOSE
		// Map output buffer and read results
		std::cout << "INPUT A: ";
		for (uint32_t I = 0; I < NumElements * NumElements; ++I) {
			std::cout << StagingBufferPtr[I] << " ";
		}
		std::cout << std::endl;
#endif

		randomize_matrix(StagingBufferPtr, size * size);

		CmdBuffer.copyBuffer(StagingBuffer, InBufferB, { CopyRegion });

		CmdBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,  // Source stage
			vk::PipelineStageFlagBits::eComputeShader,  // Destination stage
			vk::DependencyFlags(),
			1, &memoryBarrier,
			0, nullptr,
			0, nullptr
		);

#ifdef VERBOSE
		// Map output buffer and read results
		std::cout << "INPUT B: ";
		for (uint32_t I = 0; I < NumElements * NumElements; ++I) {
			std::cout << StagingBufferPtr[I] << " ";
		}
		std::cout << std::endl;
#endif


		CmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, ComputePipeline);
		CmdBuffer.bindDescriptorSets(
			vk::PipelineBindPoint::eCompute,    // Bind point
			PipelineLayout,                  // Pipeline Layout
			0,                               // First descriptor set
			{ DescriptorSet },               // List of descriptor sets
			{});                             // Dynamic offsets


		PushConstants pushConstantsData {
			size,
			size,
			size,
			0.5,
			3.0,
		};

		// Push constants into the command buffer
		CmdBuffer.pushConstants(
			PipelineLayout, 
			vk::ShaderStageFlagBits::eCompute,
			0,
			sizeof(pushConstantsData),
			&pushConstantsData
		);

		CmdBuffer.dispatch(CEIL_DIV(pushConstantsData.N, BN), CEIL_DIV(pushConstantsData.M, BM), 1);

		vk::MemoryBarrier postComputeBarrier(
			vk::AccessFlagBits::eShaderWrite,     // Source access mask
			vk::AccessFlagBits::eTransferRead     // Destination access mask
		);

		CmdBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader,  // Source stage mask
			vk::PipelineStageFlagBits::eTransfer,  // Destination stage mask
			vk::DependencyFlags(),
			1, &postComputeBarrier,
			0, nullptr,
			0, nullptr
		);

		CmdBuffer.copyBuffer(OutBuffer, StagingBuffer, { CopyRegion });

		CmdBuffer.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, queryPool, 1);

		CmdBuffer.end();

		// Fence and submit
		vk::Queue Queue = Device.getQueue(ComputeQueueFamilyIndex, 0);
		vk::Fence Fence = Device.createFence(vk::FenceCreateInfo());
		vk::SubmitInfo SubmitInfo(
			0,                // Num Wait Semaphores
			nullptr,        // Wait Semaphores
			nullptr,        // Pipeline Stage Flags
			1,              // Num Command Buffers
			&CmdBuffer);    // List of command buffers

		float elapsed_time = 0.0f;
		for(int i=0; i<repeat_times; i++) {
			Device.resetFences(Fence);

			Queue.submit({ SubmitInfo }, Fence);
			(void) Device.waitForFences(
				{ Fence },             // List of fences
				true,               // Wait All
				uint64_t(-1));      // Timeout

			vk::Result result = Device.getQueryPoolResults(queryPool,
						  0,
						  2,
						  queryResults.size() * sizeof(uint64_t),
						  queryResults.data(),
						  sizeof(uint64_t),
						  vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
			assert(result == vk::Result::eSuccess);

			elapsed_time += (float(queryResults[1] - queryResults[0]) * DeviceProps.limits.timestampPeriod) / 1000000.0f;

		}

		elapsed_time = elapsed_time / 1000.0f; // convert result from ms to sec
		uint64_t flops = uint64_t(2) * pushConstantsData.M * pushConstantsData.N * pushConstantsData.K;  // uint64_t needed to promote the calculation to long;

		std::cout << "Average elapsed time: " << elapsed_time / repeat_times << "s, performance: " << repeat_times * flops * 1e-9 / elapsed_time << "GFLOPS. , size: " << pushConstantsData.M << std::endl;

		Device.destroyFence(Fence);
	}


#ifdef VERBOSE
	std::cout << "OUTPUT: ";
	for (uint32_t I = 0; I < NumElements * NumElements; ++I) {
		std::cout << StagingBufferPtr[I] << " ";
	}
	std::cout << std::endl;
#endif


	Device.unmapMemory(StagingBufferMemory);
	////////////////////////////////////////////////////////////////////////
	//                              CLEANUP                               //
	////////////////////////////////////////////////////////////////////////
	Device.destroyQueryPool(queryPool);
	Device.resetCommandPool(CommandPool, vk::CommandPoolResetFlags());
	Device.destroyDescriptorSetLayout(DescriptorSetLayout);
	Device.destroyPipelineLayout(PipelineLayout);
	Device.destroyPipelineCache(PipelineCache);
	Device.destroyShaderModule(ShaderModule);
	Device.destroyPipeline(ComputePipeline);
	Device.destroyDescriptorPool(DescriptorPool);
	Device.destroyCommandPool(CommandPool);
	Device.destroyBuffer(StagingBuffer);
	Device.destroyBuffer(InBufferA);
	Device.destroyBuffer(InBufferB);
	Device.destroyBuffer(OutBuffer);
	Device.freeMemory(InBufferAMemory);
	Device.freeMemory(InBufferBMemory);
	Device.freeMemory(OutBufferMemory);
	Device.freeMemory(StagingBufferMemory);
	Device.destroy();
	Instance.destroy();
	return 0;
}
