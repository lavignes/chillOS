ENTRY(kstart);

. = 0x80000000;

SECTIONS {
	.text : ALIGN(4K) {
		*(.init);
		*(.text);
        . += 4096;
        PROVIDE(KMEM_STACK_TOP = .);
	}
}
