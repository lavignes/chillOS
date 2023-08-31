# vim: ft=riscv sw=8 ts=8 cc=80 noet
.section .init

.global _Kentry
_Kentry:
	csrr t0, mhartid
	bnez t0, _Khartpark

	la sp, _KSTACK_TOP
	mv a0, a1
	call _ZN1K5startE
	tail _ZN1K4haltE

_Khartpark:
	wfi
	j _Khartpark
