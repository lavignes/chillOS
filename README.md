## setup

* install toolchain (`binutils-riscv64-unknown-elf` on ubuntu-derivatives) 
* install `gdb-multiarch`
* install qemu `qemu-system-riscv64` (`qemu-system-misc` on ubuntu-derivatives)

## run

```
make run
```

## debug

```
make debug
```

```
gdb-multiarch -x tools/gdbinit
```
