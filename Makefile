CC = gcc
NVCC = nvcc -ccbin=gcc
BUILD = release
CFLAGS_RELEASE = -O3 -Ofast -march=native -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -ggdb3 -fPIC
CFLAGS_DEBUG = -Wno-unused-result -O0 -ggdb3 -fPIC
NVCC_CFLAGS_RELEASE = --generate-code arch=compute_86,code=[compute_86,sm_86] --generate-line-info -Xcompiler="-O3 -Ofast -march=native -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -ggdb3 -fPIC" --expt-relaxed-constexpr
NVCC_CFLAGS_DEBUG = -O0 -Xcompiler -fPIC

INCLUDES = -I include/ -I kernels/include/ -I third_party/OpenBLAS/include/
KERNELS_INCLUDES = -I kernels/src/

LDLIBS = -lm -lopenblas
LDFLAGS = -L third_party/OpenBLAS/lib/
NVCC_LDLIBS = -lcudart -lcublas
NVCC_LDFLAGS = -L /usr/local/cuda/lib64

SRC_DIR = src
KERNELS_DIR = kernels
INCLUDE_DIR = include
BUILD_DIR = build
LIBRARY_NAME = gpt
PLATFORM = $(shell uname -s)

# check if gcc is clang as in the case of macos
GCC_IS_CLANG=$(shell $(CC) --version | grep -i clang >/dev/null && echo yes || echo no)
CLANG_IS_GCC=$(shell $(CC) --version | grep -i GCC >/dev/null && echo yes || echo no)
_CC = $(CC)

ifeq ($(CC), gcc)
ifeq ($(GCC_IS_CLANG), yes)
_CC = clang
endif
endif

ifeq ($(CC), clang)
ifeq ($(CLANG_IS_GCC), yes)
_CC = gcc
endif
endif

ifeq ($(_CC), gcc)
CFLAGS_RELEASE += -fopenmp -DOMP
CFLAGS_DEBUG += -fopenmp -DOMP
LDLIBS += -lgomp
endif

ifeq ($(_CC), clang)
LDLIBS += -lomp
endif

ifeq ($(PLATFORM), Darwin)
SHARED_SUFFIX = dylib
BREW_PATH = $(shell brew --prefix)
INCLUDES += -I $(BREW_PATH)/opt/libomp/include -I $(BREW_PATH)/opt/argp-standalone/include
LDFLAGS += -L $(BREW_PATH)/opt/libomp/lib -L $(BREW_PATH)/opt/argp-standalone/lib
LDLIBS += -largp
endif

ifeq ($(PLATFORM), Linux)
SHARED_SUFFIX = so
ifeq ($(_CC), clang)
CFLAGS_RELEASE += -Xclang -fopenmp -DOMP
CFLAGS_DEBUG += -Xclang -fopenmp -DOMP
endif
LDLIBS += $(NVCC_LDLIBS)
LDFLAGS += $(NVCC_LDFLAGS)
endif

ifeq ($(BUILD), release)
CFLAGS = $(CFLAGS_RELEASE)
NVCC_CFLAGS = $(NVCC_CFLAGS_RELEASE)
else ifeq ($(BUILD), debug)
CFLAGS = $(CFLAGS_DEBUG)
NVCC_CFLAGS = $(NVCC_CFLAGS_DEBUG)
else
$(error Invalid BUILD '$(BUILD)', expected 'release' or 'debug')
endif

SHARED_LIB = lib$(LIBRARY_NAME).$(SHARED_SUFFIX)
# Check if Valgrind is installed
VALGRIND := $(shell command -v valgrind)

# Find all source files in the src directory
SRCS = $(wildcard $(SRC_DIR)/*.c)
COMMON_SRCS = $(wildcard $(KERNELS_DIR)/src/common/*.c)
CORE_SRCS = $(wildcard $(KERNELS_DIR)/src/core/*.c)
CPU_SRCS = $(wildcard $(KERNELS_DIR)/src/cpu/*.c)
CUDA_SRCS = $(wildcard $(KERNELS_DIR)/src/cuda/*.cu)

# Generate object file names from source file names
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))
COMMON_OBJS = $(patsubst $(KERNELS_DIR)/src/common/%.c, $(KERNELS_DIR)/build/common/%.o, $(COMMON_SRCS))
CORE_OBJS = $(patsubst $(KERNELS_DIR)/src/core/%.c, $(KERNELS_DIR)/build/core/%.o, $(CORE_SRCS))
CPU_OBJS = $(patsubst $(KERNELS_DIR)/src/cpu/%.c, $(KERNELS_DIR)/build/cpu/%.o, $(CPU_SRCS))
CUDA_OBJS = $(patsubst $(KERNELS_DIR)/src/cuda/%.cu, $(KERNELS_DIR)/build/cuda/%.o, $(CUDA_SRCS))

# Find all C files in the project root directory
ROOT_SRCS = $(wildcard ./*.c)

# Generate executable names from root source file names
EXES = $(patsubst ./%.c, ./%, $(ROOT_SRCS))

.PHONY: all clean third_party shared_lib root_binaries setup valgrind_training valgrind_inference tests

# Default rule to build the shared library and root binaries
all: setup third_party shared_lib root_binaries
ifeq "$(PLATFORM)" "Darwin"
	@echo "\nExecute the following command in terminal to setup DYLD_LIBRARY_PATH environment variable\n"
	@echo 'export DYLD_LIBRARY_PATH=$(BREW_PATH)/opt/libomp/lib:$(BREW_PATH)/opt/argp-standalone/lib:third_party/OpenBLAS/lib:$$DYLD_LIBRARY_PATH'
	@echo "\nThe above command is needed for executables to load the dynamic libraries at runtime."
endif

ifeq "$(PLATFORM)" "Linux"
	@echo "\nExecute the following command in terminal to setup LD_LIBRARY_PATH environment variable\n"
	@echo 'export LD_LIBRARY_PATH=third_party/OpenBLAS/lib:$$LD_LIBRARY_PATH'
	@echo "\nThe above command is needed for executables to load the dynamic libraries at runtime."
endif

# Compile rule for object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) $(KERNELS_INCLUDES) -c $< -o $@

# Compile rule for object files in kernels/src/common/
$(KERNELS_DIR)/build/common/%.o: $(KERNELS_DIR)/src/common/%.c
	$(CC) $(CFLAGS) $(INCLUDES) $(KERNELS_INCLUDES) -c $< -o $@

# Compile rule for object files in kernels/src/core/
$(KERNELS_DIR)/build/core/%.o: $(KERNELS_DIR)/src/core/%.c
	$(CC) $(CFLAGS) $(INCLUDES) $(KERNELS_INCLUDES) -c $< -o $@

# Compile rule for object files in kernels/src/cpu/
$(KERNELS_DIR)/build/cpu/%.o: $(KERNELS_DIR)/src/cpu/%.c
	$(CC) $(CFLAGS) $(INCLUDES) $(KERNELS_INCLUDES) -c $< -o $@

# Compile rule for object files in kernels/src/cuda/
$(KERNELS_DIR)/build/cuda/%.o: $(KERNELS_DIR)/src/cuda/%.cu
	$(NVCC) $(NVCC_CFLAGS) $(INCLUDES) $(KERNELS_INCLUDES) -c $< -o $@

# Rule to create the shared library
$(SHARED_LIB): $(OBJS) $(COMMON_OBJS) $(CORE_OBJS) $(CPU_OBJS) $(CUDA_OBJS)
	$(CC) -o $(SHARED_LIB) $(LDFLAGS) $(LDLIBS) $^ -shared

# Compile rule for root source files into executables
$(EXES): $(ROOT_SRCS) $(SHARED_LIB)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -L . -o $@ $@.c -l$(LIBRARY_NAME) $(LDLIBS)

# Rule to build third_party libraries
third_party:
	@if ! test -d third_party/OpenBLAS/lib;\
		then cd third_party/OpenBLAS && make -j && make PREFIX=../../third_party/OpenBLAS install;\
	fi
# Rule to build the shared library
shared_lib: $(SHARED_LIB)

# Rule to build the root binaries
root_binaries: $(EXES)

tests: 
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) tests/test_bmm.c $(LDLIBS)

valgrind_inference: setup all
ifeq "$(VALGRIND)" ""
	@echo "Valgrind is not installed on your system!"
ifeq "$(PLATFORM)" "Darwin"
	@echo "If you are using brew, valgrind can be installed using 'brew install valgrind'"
	@exit 1
endif
ifeq "$(PLATFORM)" "Linux"
	@echo "Valgrind can be installed using 'sudo apt install valgrind'"
	@exit 1
endif
endif
	@if ! test -f "model/valgrind-model.bin"; then \
		python3 model.py --block-size=128 --layers=1 --heads=1 --embd=128 --name=valgrind-model; \
	fi
	valgrind -s --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./infer_gpt --load-checkpoint=model/valgrind-model.bin --tokenizer=tokenizer.bin --prompt="[50256, 17250]" --max-tokens=1

valgrind_training: setup all
ifeq "$(VALGRIND)" ""
	@echo "Valgrind is not installed on your system!"
ifeq "$(PLATFORM)" "Darwin"
	@echo "If you are using brew, valgrind can be installed using 'brew install valgrind'"
	@exit 1
endif
ifeq "$(PLATFORM)" "Linux"
	@echo "Valgrind can be installed using 'sudo apt install valgrind'"
	@exit 1
endif
endif
	@if ! test -f "model/valgrind-model.bin"; then \
		python3 model.py --block-size=128 --layers=1 --heads=1 --embd=128 --name=valgrind-model; \
	fi
	valgrind -s --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./train_gpt --train-data=/home/vishalr/Desktop/gpt.c/data/tinyshakespeare/tinyshakespeare_train.bin --val-data=/home/vishalr/Desktop/gpt.c/data/tinyshakespeare/tinyshakespeare_val.bin --max-epochs=1 --log-dir=model_checkpoints --output=gpt2 --batch-size=4 --block-size=128 --lr=3e-4 --load-checkpoint=model/valgrind-model.bin

# Rule to create the build directory if it doesn't exist
setup: third_party
	@if ! test -d $(BUILD_DIR);\
		then echo "\033[93mSetting up $(BUILD_DIR) directory...\033[0m"; mkdir -p build;\
	fi
	@if ! test -d $(KERNELS_DIR)/build;\
		then echo "\033[93mSetting up $(KERNELS_DIR)/build directory...\033[0m"; mkdir -p kernels/build;\
	fi
	@if ! test -d $(KERNELS_DIR)/buid/common;\
		then echo "\033[93mSetting up $(KERNELS_DIR)/build/common directory...\033[0m"; mkdir -p kernels/build/common;\
	fi
	@if ! test -d $(KERNELS_DIR)/buid/core;\
		then echo "\033[93mSetting up $(KERNELS_DIR)/build/core directory...\033[0m"; mkdir -p kernels/build/core;\
	fi
	@if ! test -d $(KERNELS_DIR)/build/cpu;\
		then echo "\033[93mSetting up $(KERNELS_DIR)/build/cpu directory...\033[0m"; mkdir -p kernels/build/cpu;\
	fi
	@if ! test -d $(KERNELS_DIR)/build/cuda;\
		then echo "\033[93mSetting up $(KERNELS_DIR)/build/cuda directory...\033[0m"; mkdir -p kernels/build/cuda;\
	fi

# Clean rule to remove all generated files
clean:
	rm -rf $(BUILD_DIR) $(KERNELS_DIR)/build a.out $(SHARED_LIB) $(EXES)