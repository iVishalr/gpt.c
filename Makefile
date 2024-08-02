CC = gcc
# CFLAGS = -O3 -Ofast -march=native -Wno-unused-result -ggdb3 -fPIC -fopenmp -DOMP
CFLAGS = -Wno-unused-result -O0 -ggdb3 -fPIC -fopenmp -DOMP
INCLUDES = -I include/ -I /opt/OpenBLAS/include/
LDLIBS = -lm -lopenblas -lgomp
LDFLAGS = -L /opt/OpenBLAS/lib

SRC_DIR = src
INCLUDE_DIR = include
BUILD = build
LIB_NAME=gpt
SHARED_SUFFIX=dll
PLATFORM=$(shell uname -s)
ifeq "$(PLATFORM)" "Darwin"
    SHARED_SUFFIX=dylib
endif
ifeq "$(PLATFORM)" "Linux"
    SHARED_SUFFIX=so
endif
SHARED_LIB = lib$(LIB_NAME).$(SHARED_SUFFIX)

# Find all source files in the src directory
SRCS = $(wildcard $(SRC_DIR)/*.c)

# Generate object file names from source file names
OBJS = $(patsubst $(SRC_DIR)/%.c, $(BUILD)/%.o, $(SRCS))

# Find all C files in the project root directory
ROOT_SRCS = $(wildcard ./*.c)

# Generate executable names from root source file names
EXES = $(patsubst ./%.c, ./%, $(ROOT_SRCS))

.PHONY: all clean shared_lib root_binaries setup valgrind tests

# Default rule to build the shared library and root binaries
all: setup shared_lib root_binaries

# Compile rule for object files
$(BUILD)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -c $< -o $@ $(LDLIBS)

# Rule to create the shared library
$(SHARED_LIB): $(OBJS)
	$(CC) -o $(SHARED_LIB) $(LDFLAGS) $(LDLIBS) $^ -shared

# Compile rule for root source files into executables
$(EXES): $(ROOT_SRCS) $(SHARED_LIB)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $@.c -L . -l$(LIB_NAME) $(LDLIBS)

# Rule to build the shared library
shared_lib: $(SHARED_LIB)

# Rule to build the root binaries
root_binaries: $(EXES)

tests: 
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) tests/test_bmm.c $(LDLIBS)

valgrind: setup all
	valgrind -s --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./train_gpt

# Rule to create the build directory if it doesn't exist
setup:
	@if ! test -d $(BUILD);\
		then echo "\033[93mSetting up build directory...\033[0m"; mkdir -p build;\
	fi

# Clean rule to remove all generated files
clean:
	rm -rf $(BUILD) a.out $(SHARED_LIB) $(EXES)
