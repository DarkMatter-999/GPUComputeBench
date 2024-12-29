BUILD_DIR=./build
VULKAN_SRC_DIR=./src/vulkan

CXX = g++
CXXFLAGS = -std=c++17 -Wall -I. -I$(VULKAN_SDK_PATH)/include
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib -lglfw -lvulkan

SRC_FILES = $(wildcard $(VULKAN_SRC_DIR)/*.cpp)
OBJ_FILES = $(patsubst $(VULKAN_SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))
SHADER_FILES = $(wildcard $(VULKAN_SRC_DIR)/shaders/*.vert $(VULKAN_SRC_DIR)/shaders/*.frag $(VULKAN_SRC_DIR)/shaders/*.comp)
SPV_FILES = $(patsubst $(VULKAN_SRC_DIR)/shaders/%.vert,$(BUILD_DIR)/shaders/%.vert.spv,$(SHADER_FILES)) \
	    $(patsubst $(VULKAN_SRC_DIR)/shaders/%.frag,$(BUILD_DIR)/shaders/%.frag.spv,$(SHADER_FILES)) \
	    $(patsubst $(VULKAN_SRC_DIR)/shaders/%.comp,$(BUILD_DIR)/shaders/%.comp.spv,$(SHADER_FILES))
TARGET = bench

$(TARGET): $(OBJ_FILES) $(SPV_FILES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/$(TARGET) $(OBJ_FILES) $(LDFLAGS)

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

.PHONY: test clean shader run

shader: $(SPV_FILES)

run:
	cd $(BUILD_DIR) && ./$(TARGET)
clean:
	rm -rf $(BUILD_DIR)

