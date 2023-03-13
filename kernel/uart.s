# TODO: i want to get the uart address from the dtb
.equ UART_BASE, 0x10000000

.global uart_write
uart_write:
    li t0, UART_BASE
1:
    beqz a1, 2f     # if len == 0, exit
    lbu t1, 0(a0)   # read char from mem
    sb t1, 0(t0)    # write char to uart
    addi a1, a1, -1
    addi a0, a0, 1
    j 1b
2:
    ret
