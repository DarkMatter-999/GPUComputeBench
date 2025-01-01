BUILD_DIR=./build

VULKAN_SRC_DIR=./src/vulkan
HIP_SRC_DIR=./src/hip

CXX = g++
CXXFLAGS = -std=c++17 -Wall -I. -I$(VULKAN_SDK_PATH)/include
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib -lglfw -lvulkan

HIPCC = hipcc
HIPCCFLAGS = -Wall -I$(HIP_SRC_DIR)


VULKAN_SRC_FILES = $(wildcard $(VULKAN_SRC_DIR)/*.cpp)
VULKAN_OBJ_FILES = $(patsubst $(VULKAN_SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(VULKAN_SRC_FILES))
SHADER_FILES = $(wildcard $(VULKAN_SRC_DIR)/shaders/*.vert $(VULKAN_SRC_DIR)/shaders/*.frag $(VULKAN_SRC_DIR)/shaders/*.comp)
SPV_FILES = $(patsubst $(VULKAN_SRC_DIR)/shaders/%.vert,$(BUILD_DIR)/shaders/%.vert.spv,$(SHADER_FILES)) \
	    $(patsubst $(VULKAN_SRC_DIR)/shaders/%.frag,$(BUILD_DIR)/shaders/%.frag.spv,$(SHADER_FILES)) \
	    $(patsubst $(VULKAN_SRC_DIR)/shaders/%.comp,$(BUILD_DIR)/shaders/%.comp.spv,$(SHADER_FILES))

HIP_SRC_FILES = $(wildcard $(HIP_SRC_DIR)/*.cu)
HIP_OBJ_FILES = $(patsubst $(HIP_SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(HIP_SRC_FILES))

TARGET = bench

# Vulkan Target
$(TARGET)-vk: $(VULKAN_OBJ_FILES) $(SPV_FILES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/$(TARGET) $(VULKAN_OBJ_FILES) $(LDFLAGS)

$(BUILD_DIR)/%.o: $(VULKAN_SRC_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/shaders/%.vert.spv: $(VULKAN_SRC_DIR)/shaders/%.vert
	mkdir -p $(BUILD_DIR)/shaders
	$(VULKAN_SDK_PATH)/bin/glslc $< -o $@

$(BUILD_DIR)/shaders/%.frag.spv: $(VULKAN_SRC_DIR)/shaders/%.frag
	mkdir -p $(BUILD_DIR)/shaders
	$(VULKAN_SDK_PATH)/bin/glslc $< -o $@

$(BUILD_DIR)/shaders/%.comp.spv: $(VULKAN_SRC_DIR)/shaders/%.comp
	mkdir -p $(BUILD_DIR)/shaders
	$(VULKAN_SDK_PATH)/bin/glslc $< -o $@

# ROCm HIP Target
$(TARGET)-hip: $(HIP_OBJ_FILES)
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(HIPCCFLAGS) -o $(BUILD_DIR)/$(TARGET) $(HIP_OBJ_FILES)

$(BUILD_DIR)/%.o: $(HIP_SRC_DIR)/%.cu
	mkdir -p $(BUILD_DIR)
	$(HIPCC) $(HIPCCFLAGS) -c -o $@ $<

.PHONY: test clean shader run


shader: $(SPV_FILES)

run:
	cd $(BUILD_DIR) && ./$(TARGET)
clean:
	rm -rf $(BUILD_DIR)

