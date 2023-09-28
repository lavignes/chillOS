.section .text

.global _ZN5Linux8sys_openE
_ZN5Linux8sys_openE:
    mov %rax, 2
    syscall
    ret

.global _ZN5Linux8sys_readE
_ZN5Linux8sys_readE:
    mov %rax, 0
    syscall
    ret

.global _ZN5Linux9sys_writeE
_ZN5Linux9sys_writeE:
    mov %rax, 1
    syscall
    ret

.global _ZN5Linux9sys_closeE
_ZN5Linux9sys_closeE:
    mov %rax, 3
    syscall
    ret

.global _ZN5Linux8sys_exitE
_ZN5Linux8sys_exitE:
    mov %rax, 60
    syscall
    ret
