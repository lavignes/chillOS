.include "string.inc"

.section .init
.global kstart
kstart:
    # if we are not hart 0, go chill over there
    csrr t0, mhartid
    bnez t0, hart_park

    la sp, KMEM_STACK_TOP

    # fdt/dtb addr is in a1
    mv a0, a1
    call k0mem_init
    call k0mem_alloc_init

    ld a0, TEST_FORMAT
    ld a1, TEST_FORMAT + 8
    li t0, 0x42
    addi sp, sp, -8
    sd t0, 0(sp)
    nop
    call uart_print
    addi sp, sp, 8

    tail kpanic

STRING_DEFINE TEST_FORMAT, "number:%x\n"

# park the harts over here until we have semaphore to release them
hart_park:
    wfi
    j hart_park

# TODO: need to write panic routine that kills all harts
STRING_DEFINE KERNEL_PANIC, "kenel panic!\n"
.global kpanic
kpanic:
    ld a0, KERNEL_PANIC
    ld a1, KERNEL_PANIC + 8
    call uart_write
1:
    j 1b

