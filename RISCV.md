# RISC-V General Reference 

Programmer's references for RISC-V kind of suck. They are messy and don't contain
a comprehensive list of all instructions and pseudo-instructions in one place.

This document is a practical reference covering RISC-V as I use and need it.

## Registers

| Register | ABI Name | Description             | Saver  |
|----------|----------|-------------------------|--------|
| x0       | zero     | always zero             |        |
| x1       | ra       | return address          | caller |
| x2       | sp       | stack pointer           | callee |
| x3       | gp       | global pointer          |        |
| x4       | tp       | thread pointer          |        |
| x5       | t0       | temporary               | caller |
| x6       | t1       | temporary               | caller |
| x7       | t2       | temporary               | caller |
| x8       | s0 / fp  | saved / frame pointer   | callee |
| x9       | s1       | saved                   | callee |
| x10      | a0       | argument / return value | caller |
| x11      | a1       | argument / return value | caller |
| x12      | a2       | argument                | caller |
| x13      | a3       | argument                | caller |
| x14      | a4       | argument                | caller |
| x15      | a5       | argument                | caller |
| x16      | a6       | argument                | caller |
| x17      | a7       | argument                | caller |
| x18      | s2       | saved                   | callee |
| x19      | s3       | saved                   | callee |
| x20      | s4       | saved                   | callee |
| x21      | s5       | saved                   | callee |
| x22      | s6       | saved                   | callee |
| x23      | s7       | saved                   | callee |
| x24      | s8       | saved                   | callee |
| x25      | s9       | saved                   | callee |
| x26      | s10      | saved                   | callee |
| x27      | s11      | saved                   | callee |
| x28      | t3       | temporary               | caller |
| x29      | t4       | temporary               | caller |
| x30      | t5       | temporary               | caller |
| x31      | t6       | temporary               | caller |
| pc       |          | program counter         |        |

## RV32I Instructions

| Format                | Name                               | Pseudocode                                      |
|-----------------------|------------------------------------|-------------------------------------------------|
| lui rd, imm           | load upper immediate               | rd <- sext(imm[31:12] << 12)                    |
| auipc rc, offset      | add upper immediate to pc          | rd <- pc + sext(imm[31:12] << 12)               |
| jal rd, offset        | jump and link                      | rd <- pc + 4<br>pc <- pc + sext(offset)         |
| jalr rd, rs1, offset  | jump and link register             | rd <- pc + 4<br>pc <- (rs1 + sext(offset)) & ~1 |
| beq rs1, rs2, offset  | branch equal                       | if rs1 == rs2 then pc <- pc + sext(offset)      |
| bne rs1, rs2, offset  | branch not equal                   | if rs1 != rs2 then pc <- pc + sext(offset)      |
| blt rs1, rs2, offset  | branch less than                   | if rs1 < rs2 then pc <- pc + sext(offset)       |
| bge rs1, rs2, offset  | branch greater or equal            | if rs1 >= rs2 then pc <- pc + sext(offset)      |
| bltu rs1, rs2, offset | branch less than (unsigned)        | if rs1 < rs2 then pc <- pc + sext(offset)       |
| bgeu rs1, rs2, offset | branch greater or equal (unsigned) | if rs1 >= rs2 then pc <- pc + sext(offset)      |
| lb rd, offset(rs1)    | load byte                          | rd <- sext(M\[rs1 + sext(offset)\][7:0])        |
| lh rd, offset(rs1)    | load half word                     | rd <- sext(M\[rs1 + sext(offset)\][15:0])       |
| lw rd, offset(rs1)    | load word                          | rd <- sext(M\[rs1 + sext(offset)\][31:0]        |
| lbu rd, offset(rs1)   | load byte (unsigned)               | rd <- M\[rs1 + sext(offset)\][7:0]              |
| lhu rd, offset(rs1)   | load half word (unsigned)          | rd <- M\[rs1 + sext(offset)\][15:0]             |
| sb rs2, offset(rs1)   | store byte                         | M[rs1 + sext(offset)] <- rs2[7:0]               |
| sh rs2, offset(rs1)   | store half word                    | M[rs1 + sext(offset)] <- rs2[15:0]              |
| sw rs2, offset(rs1)   | store word                         | M[rs1 + sext(offset)] <- rs2[31:0]              |
| addi rd, rs1, imm     | add immediate                      | rd <- rs1 + sext(imm)                           |
| slti rd, rs1, imm     | set less than immediate            | rd <- iXLEN(rs1) < iXLEN(sext(imm))             |
| sltiu rd, rs1, imm    | set less than immediate (unsigned) | rd <- rs1 < sext(imm)                           |
| xori rd, rs1, imm     | xor immediate                      | rd <- rs1 ^ sext(imm)                           |
| ori rd. rs1, imm      | or immediate                       | rd <- rs1 \| sext(imm)                          |
| andi rd. rs1, imm     | and immediate                      | rd <- rs1 & sext(imm)                           |
| slli rd, rs1, imm     | shift left logical immediate       | rd <- rs1 << imm                                |
| srli rd, rs1, imm     | shift right logical immediate      | rd <- rs1 >> imm                                |
| srai rd, rs1, imm     | shift right arithmetic immediate   | rd <- iXLEN(rs1) >> imm                         |
| add rd, rs1, rs2      | add                                | rd <- rs1 + rs2                                 |
| sub rd, rs1, rs2      | subtract                           | rd <- rs1 - rs2                                 |
| sll rd, rs1, rs2      | shift left logical                 | rd <- rs1 << rs2                                |
| slt rd, rs1, rs2      | set less than                      | rd <- iXLEN(rs1) < iXLEN(rs2)                   |
| sltu rd, rs1, rs2     | set less than (unsigned)           | rd <- rs1 < rs2                                 |
| xor rd, rs1, rs2      | xor                                | rd <- rs1 ^ rs2                                 |
| srl rd, rs1, rs2      | shift right logical                | rd <- rs1 >> rs2                                |
| sra rd, rs1, rs2      | shift right arithmetic             | rd <- iXLEN(rs1) >> rs2                         |
| or rd, rs1, rs2       | or                                 | rd <- rs1 \| rs2                                |
| and rd, rs1, rs2      | and                                | rd <- rs1 & rs2                                 |
| fence pred, succ      | memory barrier on any of i/o/r/w   | ensure pred (iorw) commits before succ (iorw)   |
|-----------------------|------------------------------------|-------------------------------------------------|
| fence.i               | instruction memory barrier         | ensure previous instr writes are committed      |

## RV64I Instructions

| Format              | Name                                  | Pseudocode                             |
|---------------------|---------------------------------------|----------------------------------------|
| lwu rd, offset(rs1) | load word (unsigned)                  | rd <- M\[rs1 + sext(offset)\][31:0]    |
| ld rd, offset(rs1)  | load dword                            | rd <- M\[rs1 + sext(offset)\][63:0]    |
| sd rs2, offset(rs1) | store dword                           | M[rs1 + sext(offset)] <- rs2[63:0]     |
| addiw rd, rs1, imm  | add immediate word                    | rd <- sext((rs1 + sext(imm))[31:0])    |
| slliw rd, rs1, imm  | shift left logical immediate word     | rd <- sext((rs1 << imm)[31:0])         |
| srliw rd, rs1, imm  | shift right logical immediate word    | rd <- sext(rs1[31:0] >> imm)           |
| sraiw rd, rs1, imm  | shift right arithmetic immediate word | rd <- sext(i32(rs1[31:0]) >> imm)      |
| addw rd, rs1, rs2   | add word                              | rd <- sext((rs1 + rs2)[31:0])          |
| subw rd, rs1, rs2   | subtract word                         | rd <- sext((rs1 - rs2)[31:0])          |
| sllw rd, rs1, rs2   | shift left logical word               | rd <- sext((rs1 << rs2[4:0])[31:0])    |
| srlw rd, rs1, rs2   | shift right logical word              | rd <- sext(rs1[31:0] >> rs2[4:0])      |
| sraw rd, rs1, rs2   | shift right arithmetic word           | rd <- sext(i32(rs1[31:0]) >> rs2[4:0]) |

## RV32M Instructions

| Format              | Name                               | Pseudocode                                     |
|---------------------|------------------------------------|------------------------------------------------|
| mul rd, rs1, rs2    | multiply                           | rd <- rs1 * rs2                                |
| mulh rd, rs1, rs2   | multiply high (signed, signed)     | rd <- iXLEN((iXLEN(rs1) * iXLEN(rs2))) >> XLEN |
| mulhsu rd, rs1, rs2 | multiply high (signed, unsigned)   | rd <- iXLEN((iXLEN(rs1) * rs2)) >> XLEN        |
| mulhu rd, rs1, rs2  | multiply high (unsigned, unsigned) | rd <- (rs1 * rs2) >> XLEN                      |
| div rd, rs1, rs2    | divide (signed)                    | rd <- iXLEN(rs1) / iXLEN(rs2)                  |
| divu rd, rs1, rs2   | divide (unsigned)                  | rd <- rs1 / rs2                                |
| rem rd, rs1, rs2    | remainder (signed)                 | rd <- iXLEN(rs1) % iXLEN(rs2)                  |
| remu rd, rs1, rs2   | remainder (unsigned)               | rd <- rs1 % rs2                                |

## RV64M Instructions

| Format             | Name                      | Pseudocode                |
|--------------------|---------------------------|---------------------------|
| mulw rd, rs1, rs2  | multiply word             | rd <- rs1 * rs2           |
| divw rd, rs1, rs2  | divide word               | rd <- i32(rs1) / i32(rs2) |
| divuw rd, rs1, rs2 | divide (unsigned) word    | rd <- rs1 / rs2           |
| remw rd, rs1, rs2  | remainder (signed) word   | rd <- i32(rs1) % i32(rs2) |
| remuw rd, rs1, rs2 | remainder (unsigned) word | rd <- rs1 % rs2           |

## Pseudo-Instructions

| Format                     | Name                                 | Notes                       |
|----------------------------|--------------------------------------|-----------------------------|
| la rd, symbol              | load address                         |                             |
| lla rd, symbol             | load local address                   | never position-independent  |
| lga rd, symbol             | load global address                  | always position-independent |
| l{b,h,w,d} rd, symbol      | load global                          |                             |
| s{b,h,w,d} rd, symbol, rs1 | store global                         |                             |
| nop                        | no operation                         |                             |
| li rd, imm                 | load immediate                       |                             |
| mv rd, rs1                 | copy register                        |                             |
| not rd, rs1                | one's complement                     |                             |
| neg rd, rs1                | two's complement                     |                             |
| negw rd, rs1               | two's complement word                |                             |
| sext.b rd, rs1             | sign-extend byte                     |                             |
| sext.h rd, rs1             | sign-extend half word                |                             |
| sext.w rd, rs1             | sign-extend word                     |                             |
| zext.b rd, rs1             | zero-extend byte                     |                             |
| zext.h rd, rs1             | zero-extend half word                |                             |
| zext.w rd, rs1             | zero-extend word                     |                             |
| seqz rd, rs1               | set if equal zero                    |                             |
| snez rd, rs1               | set if not equal to zero             |                             |
| sltz rd, rs1               | set if less than zero                |                             |
| sgtz rd, rs1               | set if greater than zero             |                             |
| beqz rs1, offset           | branch equal to zero                 |                             |
| bnez rs1, offset           | branch not equal to zero             |                             |
| blez rs1, offset           | branch less than or equal to zero    |                             |
| bgez rs1, offset           | branch greater than or equal to zero |                             |
| bltz rs1, offset           | branch less than zero                |                             |
| bgtz rs1, offset           | branch greater than zero             |                             |
| bgt rs1, rs2, offset       | branch greater than                  |                             |
| ble rs1, rs2, offset       | branch less than or equal            |                             |
| bgtu rs1, rs2, offset      | branch greater than (unsigned)       |                             |
| bleu rs1, rs2, offset      | branch less then or equal (unsigned) |                             |
| j offset                   | jump                                 |                             |
| jal offset                 | jump and link                        |                             |
| jr rs1                     | jump register                        |                             |
| jalr rs1                   | jump and link register               |                             |
| ret                        | return                               |                             |
| call offset                | call                                 |                             |
| tail offset                | tail call                            |                             |
| fence                      | fence all memory and i/o             |                             |

## Control and Status Registers

| Name     | Privilege | Description                                 |
|----------|-----------|---------------------------------------------|
| mhartid  | MRO       | id of hardware thread                       |
| mstatus  | MRW       | machine status register                     |
| mie      | MRW       | machine interrupt enable                    |
| mtvec    | MRW       | machine trap vector                         |
| mscratch | MRW       | machine trap scratch register               |
| mepc     | MRW       | machine exception program counter           |
| mcause   | MRW       | machine trap cause                          |
| mtval    | MRW       | machine bad address or instruction          |
| satp     | MRW       | supervisor address translation & protection |

### Instructions

| Format             | Name                           |
|--------------------|--------------------------------|
| csrrw rd, csr, rs1 | csr atomic read write          |
| csrrs rd, csr, rs1 | csr atomic read and set bits   |
| csrrc rd, csr, rs1 | csr atomic read and clear bits |

In general, use the pseudo-instructions instead

### Pseudo-Instructions

| Format         | Name                             |
|----------------|----------------------------------|
| csrr rd, csr   | read csr                         |
| csrw rd, csr   | write csr                        |
| csrs csr, rs1  | set bits in csr                  |
| csrc csr, rs1  | clear bits in csr                |
| csrwi csr, imm | write csr with immediate         |
| csrsi csr, imm | set bits in csr with immediate   |
| csrci csr, imm | clear bits in csr with immediate |

## Other Privileged Instructions

| Format     | Name                                                   |
|------------|--------------------------------------------------------|
| ecall      | exception call                                         |
| ebreak     | exception break                                        |
| uret       | return to user mode                                    |
| sret       | return to supervisor mode                              |
| mret       | return to machine mode                                 |
| wfi        | wait for interrupt                                     |
| sfence.vma | fence for virtual memory translation table (tlb flush) |
