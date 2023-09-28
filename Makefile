AS := riscv64-unknown-elf-as
CC := riscv64-unknown-elf-gcc
LD := riscv64-unknown-elf-ld
OBJCOPY := riscv64-unknown-elf-objcopy

RUN := qemu-system-riscv64 -M virt -smp 4 -m 2G -serial mon:stdio -bios none -kernel kernel.bin

AS_FLAGS := -I kernel -g -march=rv64ima
CC_FLAGS := -march=rv64ima -mabi=lp64 -g -fPIC -nostdlib -nodefaultlibs -fno-builtin \
			-ffreestanding \
			-Wimplicit -Werror -Wall -Wextra -Wno-unused-function -Wno-unused-parameter -Wno-unused-variable \
			-Wno-unused-label \
			-Wstrict-aliasing
LD_FLAGS := -T kernel.ld

KERNEL_ASM_SRC := $(wildcard kernel/*.s)
KERNEL_CHL_SRC := $(wildcard kernel/*.chl)
KERNEL_ASM_OBJ := $(KERNEL_ASM_SRC:.s=.o)
KERNEL_CHL_OBJ := $(KERNEL_CHL_SRC:.chl=.o)
KERNEL_C_SRC := $(KERNEL_CHL_SRC:.chl=.c)
KERNEL_C_PKG := $(KERNEL_CHL_SRC:.chl=.pkg)
KERNEL_C_ASM := $(KERNEL_CHL_SRC:.chl=.asm)

STDLIB_ASM_SRC := $(wildcard stdlib/*.s)
STDLIB_CHL_SRC := $(wildcard stdlib/*.chl)
STDLIB_ASM_OBJ := $(KERNEL_ASM_SRC:.s=.o)
STDLIB_CHL_OBJ := $(KERNEL_CHL_SRC:.chl=.o)
STDLIB_C_SRC := $(KERNEL_CHL_SRC:.chl=.c)
STDLIB_C_PKG := $(KERNEL_CHL_SRC:.chl=.pkg)
STDLIB_C_ASM := $(KERNEL_CHL_SRC:.chl=.asm)

all: kernel.bin

#.PRECIOUS: $(KERNEL_C_SRC)
#.PRECIOUS: $(KERNEL_C_PKG)

%.o: %.s
	$(AS) $(AS_FLAGS) -c $< -o $@

%.c: %.chl $(KERNEL_C_PKG) $(STDLIB_C_PKG)
	./chill.py -c $<

%.pkg: %.chl
	./chill.py -p $<

%.o: %.c
	$(CC) $(CC_FLAGS) -c $< -o $@

%.asm: %.c
	$(CC) $(filter-out -g,$(CC_FLAGS)) -O3 -S $< -o $@

asm: $(KERNEL_C_ASM) $(STDLIB_C_ASM)

kernel.elf: $(KERNEL_ASM_OBJ) $(KERNEL_CHL_OBJ) $(STDLIB_ASM_OBJ) $(STDLIB_CHL_OBJ)
	$(LD) $(LD_FLAGS) $^ -o $@

kernel.bin: kernel.elf
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f kernel.bin kernel.elf \
		$(KERNEL_ASM_OBJ) $(KERNEL_C_SRC) $(KERNEL_CHL_OBJ) $(KERNEL_C_PKG) $(KERNEL_C_ASM) \
		$(STDLIB_ASM_OBJ) $(STDLIB_C_SRC) $(STDLIB_CHL_OBJ) $(STDLIB_C_PKG) $(STDLIB_C_ASM)

run: kernel.bin
	$(RUN)

debug: kernel.bin
	$(RUN) -S -gdb tcp::9000
