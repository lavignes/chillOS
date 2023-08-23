# vim: ft=riscv sw=8 ts=8 cc=80 noet
.include "kdt.inc"

.section .init

.global kStart
kStart:
	csrr t0, mhartid
	bnez t0, hartPark
	la sp, KMEM_STACK_TOP

	mv a0, a1
	call kdtInit

	la a0, root_node
	call kdtNodeInit

	tail kPanic

root_node:
	.space KDT_NODE_SIZE
	.align 4

.global kPanic
kPanic:
	wfi
	j kPanic

.global kSpinLock
kSpinLock:
	li t0, 1
1:	amoswap.d.aq t0, t0, (a0)
	bnez t0, 1b
	ret

.global kSpinUnlock
kSpinUnlock:
	amoswap.d.rl zero, zero, (a0)
	ret

# place other harts here until we have a semaphore
# to wake them up
hartPark:
	wfi
	j hartPark

