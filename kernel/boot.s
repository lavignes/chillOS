# vim: ft=riscv sw=8 ts=8 cc=80 noet
.section .boot

.align 2
.global _Kentry
_Kentry:
	csrr t0, mhartid
	bnez t0, _Khartpark

	la sp, _KSTACK_TOP
	mv a0, sp
	# fdt is already in a1
	tail _ZN1K5startE

.align 2
_Khartpark:
	wfi
	j _Khartpark

