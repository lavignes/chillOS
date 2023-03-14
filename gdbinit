set architecture riscv:rv64
tui new-layout riscv64 {-horizontal {src 4 asm 3} 2 regs 1} 2 status 0 cmd 1

layout riscv64
tui reg all
focus cmd

file kernel.elf
target remote localhost:9000
