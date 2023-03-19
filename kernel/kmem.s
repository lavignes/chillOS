.include "pushpop.inc"
.include "string.inc"

# max number of memory ranges used for heaps
.equ KMEM_MEM_BLOCKS_SIZE, 32

# array of (start,end) pairs of memory ranges
kmem_mem_blocks:
    .zero 16 * KMEM_MEM_BLOCKS_SIZE
kmem_mem_blocks_len:
    .dword 0

.equ DTB_MAGIC, 0xD00DFEED

.global k0mem_init
k0mem_init:
    pushd ra
    pushd s1
    csrw satp, zero     # disable all virtual memory and protection

    mv s1, a0
    lwu a0, 0(s1)       # load magic
    call be_to_lew

    li t0, DTB_MAGIC
    beq a0, t0, 1f
    tail panic_bad_magic
1:
    mv a0, s1
    call init_mem_blocks

    popd s1
    popd ra
    ret

init_mem_blocks:
    # for now, we just hardcode the memory map
    # from kernel stack top to +2GB
    # TODO: parse the DTB nodes to find the memory range
    la t0, KMEM_STACK_TOP
    la t1, kmem_mem_blocks
    sd t0, 0(t1)
    li t0, 0xFFFFFFFF # 0x80000000 + 2GB
    sd t0, 8(t1)

    # 1 entry
    li t0, 1
    la t1, kmem_mem_blocks_len
    sd t0, 0(t1)
    ret

STRING_DEFINE BAD_MAGIC, "DTB init error: bad magic\n"
panic_bad_magic:
    ld a0, BAD_MAGIC
    ld a1, BAD_MAGIC + 8
    call uart_write
    tail kpanic

be_to_lew:
    andi t0, a0, 0xFF
    slli t0, t0, 24
    srli t1, a0, 24
    or t0, t0, t1
    srli t1, a0, 16
    andi t1, t1, 0xFF
    slli t1, t1, 8
    or t0, t0, t1
    srli t1, a0, 8
    andi t1, t1, 0xFF
    slli t1, t1, 16
    or a0, t0, t1
    ret

klock_aquire:
    li t0, 1
1:
    amoswap.w.aq t0, t0, (a0)
    bnez t0, 1b
    ret

klock_release:
    amoswap.w.rl zero, zero, (a0)
    ret

# array of watermarks
block_watermarks:
    .zero 8 * KMEM_MEM_BLOCKS_SIZE

active_block:
    .dword 0

mem_lock:
    .word 0

.global k0mem_alloc_init
k0mem_alloc_init:
    mv t0, zero
    la t1, block_watermarks
    la t2, kmem_mem_blocks
    la t3, kmem_mem_blocks_len
    ld t3, 0(t3)
1:
    ld t4, 0(t2)    # load mem block start
    sd t4, 0(t1)    # store in watermark

    addi t1, t1, 8
    addi t2, t2, 16
    addi t0, t0, 1

    blt t0, t3, 1b
    ret

.global
kmem_free:
    ret

.global kmem_alloc
kmem_alloc:
    pushd ra
    pushd s1

    mv s1, a0       # backup memory size in s1
    la a0, mem_lock
    call klock_aquire
1:
    la t0, active_block # get index into watermarks
    ld t0, 0(t0)
    slli t0, t0, 3          # shifted to multiply by 8

    la t1, block_watermarks
    add t2, t1, t0
    ld t2, 0(t2)    # get watermark value

    la t1, kmem_mem_blocks
    add t3, t1, t0
    addi t3, t3, 8
    ld t3, 0(t3)    # get end of mem block

    add t4, t2, s1  # get new potential watermark value

    blt t4, t3, 2f  # if new value is less than block end, then allocate

    # otherwise, need to increment active block
    la t0, active_block
    ld t1, 0(t1)
    addi t1, t1, 1
    sd t1, 0(t1)

    j 1b    # retry next block
2:
    la t0, block_watermarks     # update watermark
    sd t4, 0(t0)

    mv s1, t2           # backup pointer in s1
    la a0, mem_lock
    call klock_release

    mv a0, s1   # return pointer
    popd s1
    popd ra
    ret
