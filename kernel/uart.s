# TODO: i want to get the uart address from the dtb
.include "pushpop.inc"

.equ UART_BASE, 0x10000000

.global uart_write
uart_write:
    li t0, UART_BASE
    add a1, a0, a1  # ptr to string end is now in a1
1:
    beq a0, a1, 2f     # if len == 0, exit
    lbu t1, 0(a0)   # read char from mem
    sb t1, 0(t0)    # write char to uart
    addi a0, a0, 1
    j 1b
2:
    ret

hex_chars:
    .ascii "0123456789abcdef"
    .align 4

.global uart_print
uart_print:
    mv fp, sp
    pushd fp
    pushd ra
    pushd s1
    pushd s2
    pushd s3
    mv s1, a0
    mv s2, a1
    li s3, UART_BASE
    add s2, s1, s2  # ptr to string end is now in s2
Lloop:
    beq s1, s2, Lreturn # if end of string, end
    lbu t0, 0(s1)       # read char

    li t1, '%'
    beq t0, t1, Lformat # if not a '%' then print normally
    sb t0, 0(s3)
    j Lnext_char

Lformat:
    addi s1, s1, 1
    beq s1, s2, Lreturn # if this is end of string, just exit
    lbu t0, 0(s1)       # read specifier

    li t1, '%'
    bne t0, t1, Lhex  # if '%', print literal '%'
    sb t1, 0(s3)
    j Lnext_char

Lhex:
    li t1, 'x'
    bne t0, t1, Lnext_char # if 'x', print 64-bit hex

    poprd t0, fp        # read hex value off arg stack
    li t1, 60
1:
    srl t2, t0, t1      # shift nibble into LSBs
    andi t2, t2, 0xF    # mask out
    la t3, hex_chars
    add t2, t3, t2
    lbu t2, 0(t2)       # read hex char
    sb t2, 0(s3)        # write char
    addi t1, t1, -4     # decrease shift
    bnez t1, 1b         # get next nibble
    j Lnext_char

Lnext_char:
    addi s1, s1, 1
    j Lloop
Lreturn:
    popd s3
    popd s2
    popd s1
    popd ra
    popd fp
    ret
