.section .text

# compiler builtins

.global _ZN2U813read_volatileE
_ZN2U813read_volatileE:
	lbu a0, 0(a0)
	ret

.global _ZN2U814write_volatileE
_ZN2U814write_volatileE:
	sb a1, 0(a0)
	ret

.global _ZN3U1613read_volatileE
_ZN3U1613read_volatileE:
	lhu a0, 0(a0)
	ret

.global _ZN3U1614write_volatileE
_ZN3U1614write_volatileE:
	sh a1, 0(a0)
	ret

.global _ZN3U3213read_volatileE
_ZN3U3213read_volatileE:
	lwu a0, 0(a0)
	ret

.global _ZN3U3214write_volatileE
_ZN3U3214write_volatileE:
	sw a1, 0(a0)
	ret

.global _ZN3U6413read_volatileE
_ZN3U6413read_volatileE:
	ld a0, 0(a0)
	ret

.global _ZN3U6414write_volatileE
_ZN3U6414write_volatileE:
	sd a1, 0(a0)
	ret

.global _ZN4UInt13read_volatileE
_ZN4UInt13read_volatileE:
	ld a0, 0(a0)
	ret

.global _ZN4UInt14write_volatileE
_ZN4UInt14write_volatileE:
	sd a1, 0(a0)
	ret
