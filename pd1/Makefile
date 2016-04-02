all: bin

bin: main.cu ../util.h
	nvcc -o bin -I.. main.cu

debug: main.cu ../util.h
	nvcc -o bin_dbg -I.. -D_DEBUG main.cu

clean:
	rm bin
	rm bin_dbg

test: debug
	./bin_dbg < ~/tests/1pd/test1.in

test_release: bin
	./bin < ~/tests/1pd/test1.in
