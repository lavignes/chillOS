# vim: ft=riscv sw=8 ts=8 cc=80 noet

.include "kdt.inc"

.section .text

.equ DTB_HEADER_MAGIC, 0xd00dfeed
.equ DTB_HEADER_COMPAT_VERSION, 16

.equ DTB_HEADER_FIELD_MAGIC, 0
.equ DTB_HEADER_FIELD_MEM_RESERVATION, 16
.equ DTB_HEADER_FIELD_COMPAT_VERSION, 24
.equ DTB_HEADER_FIELD_STRUCT, 36

/*

	so what I want to do is have the caller pass a sort of handle struct.
	we'll add a method like kDtbHandleInit that will intialize the handle
	at the root of the Dtb tree.

	then we can use methods to walk the tree and read properties of the
	current node handle.

*/

DTB_HEADER_ADDR:
	.dword 0

.global kdtInit
kdtInit:
	addi sp, sp, -16
	sd ra, 8(sp)
	sd fp, 0(sp)
	addi fp, sp, 16

	mv s1, a0
	lwu a0, DTB_HEADER_FIELD_MAGIC(s1)
	call swapEndianWord
	li t0, DTB_HEADER_MAGIC
	beq a0, t0, 1f
	tail kPanic # todo handle errors better
1:	lwu a0, DTB_HEADER_FIELD_COMPAT_VERSION(s1)
	call swapEndianWord
	li t0, DTB_HEADER_COMPAT_VERSION
	beq a0, t0, 1f
	tail kPanic # todo handle errors better
1:	la t0, DTB_HEADER_ADDR
	sd s1, 0(t0)

    	ld ra, 8(sp)
    	ld fp, 0(sp)
    	addi sp, sp, 16
	ret

.global kdtNodeInit
kdtNodeInit:
	addi sp, sp, -16
	sd ra, 8(sp)
	sd fp, 0(sp)
	addi fp, sp, 16

	mv s1, a0
	# no parent for root node
	sd zero, _KDT_NODE_FIELD_PARENT(s1)

	la t0, DTB_HEADER_ADDR
	ld s2, 0(t0)
	lwu a0, DTB_HEADER_FIELD_STRUCT(s2)
	call swapEndianWord
	# compute pointer to root node as we only have a base and offset
	# have the header (s2) and offset
	add a0, a0, s2
	sd a0, _KDT_NODE_FIELD_CURRENT(s1)

    	ld ra, 8(sp)
    	ld fp, 0(sp)
    	addi sp, sp, 16
	ret

swapEndianWord:
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
