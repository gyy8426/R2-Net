all: src/cuda/roi_align.cu.o
	python3 build.py

src/cuda/roi_align.cu.o: src/cuda/roi_align_kernel.cu
	$(MAKE) -C src/cuda

clean:
	$(MAKE) -C src/cuda clean
