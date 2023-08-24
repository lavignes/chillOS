# vim: ft=riscv sw=8 ts=8 cc=80 noet

.include "abi.inc"
.include "kdt.inc"

.section .text

.equ DTB_HEADER_MAGIC, 0xd00dfeed
.equ DTB_HEADER_COMPAT_VERSION, 16

.equ DTB_HEADER_FIELD_MAGIC, 0
.equ DTB_HEADER_FIELD_MEM_RESERVATION, 16
.equ DTB_HEADER_FIELD_COMPAT_VERSION, 24
.equ DTB_HEADER_FIELD_STRUCT, 36

.equ DTB_TOKEN_BEGIN_NODE, 1
.equ DTB_TOKEN_END_NODE, 2
.equ DTB_TOKEN_PROP, 3
.equ DTB_TOKEN_NOP, 4
.equ DTB_TOKEN_END, 9

DTB_HEADER_ADDR:
	.dword 0

.global kdtInit
kdtInit:
	ABI_PREAMBLE_RA_S0

	mv s0, a0
	lwu a0, DTB_HEADER_FIELD_MAGIC(s0)
	call swapEndianWord
	li t0, DTB_HEADER_MAGIC
	beq a0, t0, 1f
	tail kPanic # todo handle errors better
1:	lwu a0, DTB_HEADER_FIELD_COMPAT_VERSION(s0)
	call swapEndianWord
	li t0, DTB_HEADER_COMPAT_VERSION
	beq a0, t0, 1f
	tail kPanic # todo handle errors better
1:	la t0, DTB_HEADER_ADDR
	sd s0, 0(t0)

	ABI_POSTAMBLE_RA_S0
	ret

.global kdtNodeInit
kdtNodeInit:
	ABI_PREAMBLE_RA_S2

	mv s0, a0
	# no parent for root node
	sd zero, _KDT_NODE_FIELD_PARENT(s0)

	la t0, DTB_HEADER_ADDR
	ld s1, 0(t0)
	lwu a0, DTB_HEADER_FIELD_STRUCT(s1)
	call swapEndianWord
	# compute pointer to root node as we only have a base and offset
	# have the header (s1) and offset
	add s1, a0, s1

# TODO: all this could be shared for other operations when initing a node

	# nodes can be prefixed with an arbitrary num of nop tokens
	# step forward until we get to the node name
	li s2, DTB_TOKEN_BEGIN_NODE
1:	lwu a0, 0(s1)
	addi s1, s1, 4
	call swapEndianWord
	bne a0, s2, 1b
	# save the start of the node name
	sd s1, _KDT_NODE_FIELD_CURRENT(s0)

	# find the first prop, it will be after null-terminated str
1:	lb t0, 0(s1)
	addi s1, s1, 1
	beqz t0, 1b
	# align to next word
	addi s1, s1, 3
	andi s1, s1, -4

	# step foward until the prop token
	li s2, DTB_TOKEN_PROP
1:	lwu a0, 0(s1)
	addi s1, s1, 4
	call swapEndianWord
	bne a0, s2, 1b
	# save the start of the first prop
	sd s1, _KDT_NODE_FIELD_PROPS(s0)

	ABI_POSTAMBLE_RA_S2
	ret

.global kdtNodeGetName
kdtNodeGetName:
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
