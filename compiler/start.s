.section .text

.global _start
_start:
    xor %ebp, %ebp
    mov %edi, (%rsp)
    lea %rsi, 8(%rsp)
    lea %rdx, 16(%rsp, %rdi, 8)
    jmp _ZN4main4mainE
