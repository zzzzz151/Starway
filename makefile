CXX := clang++
WARNINGS := -Wall -Wextra -Werror -Wunused -Wconversion -Wsign-conversion -Wshadow -Wpedantic -Wold-style-cast
CXXFLAGS := -std=c++23 -march=native -O3 -shared $(WARNINGS)
SUFFIX := .dll

ifneq ($(OS), Windows_NT)
	SUFFIX = .so
	CXXFLAGS += -fPIC
endif

all:
	$(CXX) $(CXXFLAGS) dataloader/dataloader.cpp -o dataloader$(SUFFIX)
