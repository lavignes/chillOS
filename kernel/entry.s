# vim: ft=riscv sw=8 ts=8 cc=80 noet
.section .init

.global _kentry
_kentry:
	csrr t0, mhartid
	bnez t0, _khartpark

	la sp, _KSTACK_TOP
	mv a0, a1
	call _ZN1k5startE
	call _ZN1k4haltE

_khartpark:
	wfi
	j _khartpark
