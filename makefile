.PHONY: format chess-test converter dataloader

CXX := clang++
WARNINGS := -Wall -Wextra -Werror -Wunused -Wconversion -Wsign-conversion -Wshadow -Wpedantic -Wold-style-cast
CXXFLAGS := -std=c++23 -march=native -O3 -ferror-limit=1000 $(WARNINGS)

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
chess-test:
	$(CXX) $(CXXFLAGS) cpp/chess/test.cpp -o cpp/chess/test
converter:
	$(CXX) $(CXXFLAGS) cpp/converter/montyformat_to_starway.cpp -o montyformat_to_starway
	$(CXX) $(CXXFLAGS) cpp/converter/interleave.cpp -o interleave
dataloader:
	$(CXX) $(CXXFLAGS) -shared -fPIC cpp/dataloader/dataloader.cpp -o dataloader.so
