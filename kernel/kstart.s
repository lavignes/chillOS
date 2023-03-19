.include "pushpop.inc"
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

    tail khalt

# park the harts over here until we have semaphore to release them
hart_park:
    wfi
    j hart_park


.global kpanic
kpanic:
    tail khalt

# TODO: need to write panic routine that kills all harts
STRING_DEFINE HALTING, "system is halting...\n"
.global khalt
khalt:
    ld a0, HALTING
    ld a1, HALTING + 8
    call uart_write
1:
    j 1b

