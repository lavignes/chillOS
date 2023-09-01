AS := riscv64-unknown-elf-as
CC := riscv64-unknown-elf-gcc
LD := riscv64-unknown-elf-ld
OBJCOPY := riscv64-unknown-elf-objcopy

RUN := qemu-system-riscv64 -M virt -smp 4 -m 2G -serial mon:stdio -bios none -kernel kernel.bin

AS_FLAGS := -I kernel -g -march=rv64ima
CC_FLAGS := -march=rv64ima -mabi=lp64 -g -fPIC -nostdlib -nodefaultlibs -fno-builtin \
			-Wimplicit -Werror -Wall -Wextra -Wno-unused-function -Wno-unused-parameter \
			-Wstrict-aliasing
LD_FLAGS := -T kernel.ld

KERNEL_ASM_SRC := $(wildcard kernel/*.s)
KERNEL_CHL_SRC := $(wildcard kernel/*.chl)
KERNEL_ASM_OBJ := $(KERNEL_ASM_SRC:.s=.o)
KERNEL_CHL_OBJ := $(KERNEL_CHL_SRC:.chl=.o)
KERNEL_C_SRC := $(KERNEL_CHL_SRC:.chl=.c)
KERNEL_C_PKG := $(KERNEL_CHL_SRC:.chl=.pkg)

all: kernel.bin

#.PRECIOUS: $(KERNEL_C_SRC)
#.PRECIOUS: $(KERNEL_C_PKG)

%.o: %.s
	$(AS) $(AS_FLAGS) -c $< -o $@

%.c: %.chl $(KERNEL_C_PKG)
	./chill.py -c $<

%.pkg: %.chl
	./chill.py -p $<

%.o: %.c
	$(CC) $(CC_FLAGS) -c $< -o $@

kernel.elf: $(KERNEL_ASM_OBJ) $(KERNEL_CHL_OBJ)
	$(LD) $(LD_FLAGS) $^ -o $@

kernel.bin: kernel.elf
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f kernel.bin kernel.elf $(KERNEL_ASM_OBJ) $(KERNEL_C_SRC) $(KERNEL_CHL_OBJ) $(KERNEL_C_PKG)

run: kernel.bin
	$(RUN)

debug: kernel.bin
	$(RUN) -S -gdb tcp::9000
