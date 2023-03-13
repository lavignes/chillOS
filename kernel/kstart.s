.include "asm-common.inc"

.section .init
.global kstart
kstart:
    la sp, KMEM_STACK_TOP

    mv a0, a1       # fdt/dtb addr is in a1
    call kmem_init

    tail khalt

STRING_DEFINE HALTING, "system is halting...\n"
.global khalt
khalt:
    ld a0, HALTING
    ld a1, HALTING + 8
    call uart_write
1:
    j 1b

