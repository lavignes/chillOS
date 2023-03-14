AS := riscv64-unknown-elf-as
LD := riscv64-unknown-elf-ld
OBJCOPY := riscv64-unknown-elf-objcopy

RUN := qemu-system-riscv64 -M virt -smp 4 -m 2G -serial mon:stdio -bios none -kernel kernel.bin

AS_FLAGS := -I kernel -g -L -march=rv64ima
LD_FLAGS := -T kernel.ld

KERNEL_SRC := $(wildcard kernel/*.s)
KERNEL_OBJ := $(KERNEL_SRC:.s=.o)

all: kernel.bin

%.o: %.s
	$(AS) $(AS_FLAGS) -c $< -o $@

kernel.elf: $(KERNEL_OBJ)
	$(LD) $(LD_FLAGS) $(KERNEL_OBJ) -o $@

kernel.bin: kernel.elf
	$(OBJCOPY) -O binary kernel.elf kernel.bin

clean:
	rm -f kernel.bin kernel.elf $(KERNEL_OBJ)

run: kernel.bin
	$(RUN)

debug: kernel.bin
	$(RUN) -S -gdb tcp::9000
