.include "asm-common.inc"

# max number of memory ranges used for heaps
.equ KMEM_MEM_BLOCKS_SIZE, 32

# array of (start,end) pairs of memory ranges
kmem_mem_blocks:
    .zero 16 * KMEM_MEM_BLOCKS_SIZE
kmem_mem_blocks_len:
    .dword 0

.equ DTB_MAGIC, 0xD00DFEED

.global kmem_init
kmem_init:
    pushd ra
    pushd s1
    csrw satp, zero     # disable all virtual memory and protection

    mv s1, a0
    lwu a0, 0(s1)       # load magic
    call be_to_lew

    li t0, DTB_MAGIC
    beq a0, t0, 1f
    tail halt_bad_magic
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
    sd t0, 0(t1)

    # 1 entry
    li t0, 1
    la t1, kmem_mem_blocks_len
    sd t0, 0(t1)
    ret

STRING_DEFINE BAD_MAGIC, "DTB init error: bad magic\n"
halt_bad_magic:
    ld a0, BAD_MAGIC
    ld a1, BAD_MAGIC + 8
    call uart_write
    tail khalt

be_to_lew:
    mv t1, zero
    andi t0, a0, 0x0FF
    slli t1, t0, 24
    srli a0, a0, 8
    andi t0, a0, 0x0FF
    slli t0, t0, 16
    or t1, t0, t1
    srli a0, a0, 8
    andi t0, a0, 0x0FF
    slli t0, t0, 8
    or t1, t0, t1
    srli a0, a0, 8
    andi t0, a0, 0x0FF
    or a0, t1, t0
    ret

