CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2
BIN      := barenn

all: $(BIN)

$(BIN): barenn.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(BIN)
	./$(BIN)

validate: $(BIN)
	./scripts/validate.sh ./$(BIN)

bench: $(BIN)
	./scripts/bench.sh ./$(BIN)

clean:
	rm -f $(BIN)

.PHONY: all run validate bench clean
