# 原生加速库（Python 通过 ctypes 调用）
#   Windows: mingw32-make          -> prebuilt\satstream.dll
#   Linux  : make                  -> prebuilt/libsatstream.so
CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O3 -DNDEBUG -fPIC -Wall -Wextra -Wno-unused-result -Isrc -Ithird_party
LDFLAGS  ?=
BUILD    := build
OUT      := prebuilt

SRC := src/decode.cpp src/img.cpp src/motion.cpp src/tracker.cpp src/speed.cpp \
       src/runtime.cpp src/c_api.cpp
OBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRC))

ifeq ($(filter Windows_NT,$(OS)),)
  TARGET := $(OUT)/libsatstream.so
  SLIB   := -shared
  MKDIR  := mkdir -p
else
  TARGET := $(OUT)/satstream.dll
  SLIB   := -shared -static-libgcc -static-libstdc++ -Wl,--export-all-symbols
  MKDIR  := cmd /c if not exist
endif

.PHONY: all clean dirs
all: $(TARGET)

dirs:
ifeq ($(filter Windows_NT,$(OS)),)
	@mkdir -p $(BUILD) $(OUT)
else
	@cmd /c if not exist $(BUILD) mkdir $(BUILD)
	@cmd /c if not exist $(OUT) mkdir $(OUT)
endif

$(BUILD)/%.o: src/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -DSFLY_BUILD_DLL -c $< -o $@

$(TARGET): $(OBJ) | dirs
	$(CXX) $(SLIB) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD)