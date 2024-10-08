CC = gcc
BUILD = release
CFLAGS_RELEASE = -O3 -Ofast -march=native -Wno-unused-result -ggdb3 -fPIC
CFLAGS_DEBUG = -Wno-unused-result -O0 -ggdb3 -fPIC

INCLUDES = -I include/ -I third_party/OpenBLAS/include/
LDLIBS = -lm -lopenblas
LDFLAGS = -L third_party/OpenBLAS/lib/

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
LIBRARY_NAME = gpt
PLATFORM = $(shell uname -s)

# check if gcc is clang as in the case of macos
GCC_IS_CLANG=$(shell $(CC) --version | grep -i clang >/dev/null && echo yes || echo no)
CLANG_IS_GCC=$(shell $(CC) --version | grep -i GCC >/dev/null && echo yes || echo no)
ifeq ($(CC), gcc)
ifeq ($(GCC_IS_CLANG), yes)
$(info Using clang instead of gcc. gcc is aliased to clang on your system.)
_CC = clang
endif
endif

ifeq ($(CC), clang)
ifeq ($(CLANG_IS_GCC), yes)
$(info Using gcc instead of clang. clang is aliased to gcc on your system.)
_CC = gcc
endif
endif

ifeq ($(_CC), gcc)
CFLAGS_RELEASE += -fopenmp -DOMP
CFLAGS_DEBUG += -fopenmp -DOMP
LDLIBS += -lgomp
endif

ifeq ($(_CC),clang)
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
endif

ifeq ($(BUILD), release)
CFLAGS = $(CFLAGS_RELEASE)
else ifeq ($(BUILD), debug)
CFLAGS = $(CFLAGS_DEBUG)
else
$(error Invalid BUILD '$(BUILD)', expected 'release' or 'debug')
endif

SHARED_LIB = lib$(LIBRARY_NAME).$(SHARED_SUFFIX)
# Check if Valgrind is installed
VALGRIND := $(shell command -v valgrind)

# Find all source files in the src directory
SRCS = $(wildcard $(SRC_DIR)/*.c)

# Generate object file names from source file names
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRCS))

# Find all C files in the project root directory
ROOT_SRCS = $(wildcard ./*.c)

# Generate executable names from root source file names
EXES = $(patsubst ./%.c, ./%, $(ROOT_SRCS))

.PHONY: all clean third_party shared_lib root_binaries setup valgrind_inference tests

# Default rule to build the shared library and root binaries
all: setup third_party shared_lib root_binaries

# Compile rule for object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Rule to create the shared library
$(SHARED_LIB): $(OBJS)
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
	valgrind -s --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./infer_gpt --load-checkpoint=model/valgrind-model.bin --tokenizer=tokenizer.bin --prompt="[50256, 17250]" --max-tokens=10

# Rule to create the build directory if it doesn't exist
setup: third_party
	@if ! test -d $(BUILD_DIR);\
		then echo "\033[93mSetting up build directory...\033[0m"; mkdir -p build;\
	fi

# Clean rule to remove all generated files
clean:
	rm -rf $(BUILD_DIR) a.out $(SHARED_LIB) $(EXES)
