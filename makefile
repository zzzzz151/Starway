CXX := clang++
WARNINGS := -Wall -Wextra -Werror -Wunused -Wconversion -Wsign-conversion -Wshadow -Wpedantic -Wold-style-cast
CXXFLAGS := -std=c++23 -march=native -O3 -ferror-limit=1000 $(WARNINGS)
DATALOADER_CXXFLAGS = $(CXXFLAGS) -shared

ifeq ($(OS),Windows_NT)
    EXT := .exe
    DATALOADER_EXT := .dll

	FORMAT_COMMAND := @powershell -Command "Get-ChildItem -Recurse -Include *.cpp, *.hpp, *.cu \
		| ForEach-Object { clang-format -i $$_.FullName }"
else
    EXT :=
    DATALOADER_EXT := .so

	FORMAT_COMMAND := find . -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' | xargs clang-format -i

	CXXFLAGS += -stdlib=libc++
	DATALOADER_CXXFLAGS += -fPIC
endif

recompile:

format:
	$(FORMAT_COMMAND)

chess-test: recompile
	$(CXX) $(CXXFLAGS) cpp/chess/test.cpp -o chess-test$(EXT)
	./chess-test$(EXT)

converter: recompile
	$(CXX) $(CXXFLAGS) cpp/converter/montyformat_to_starway.cpp -o montyformat_to_starway$(EXT)

dataloader: recompile
	$(CXX) $(DATALOADER_CXXFLAGS) cpp/dataloader/dataloader.cpp -o dataloader$(DATALOADER_EXT)
