# vim: ft=riscv sw=8 ts=8 cc=80 noet
.ifndef KDT_INC
.equ KDT_INC, 1

.struct 0
# pointer to parent node
_KDT_NODE_FIELD_PARENT:		.struct . + 8
# pointer to node name
_KDT_NODE_FIELD_CURRENT:	.struct . + 8
# pointer to current prop
_KDT_NODE_FIELD_PROPS:		.struct . + 8
# pointer to current child
_KDT_NODE_FIELD_CHILDREN:	.struct . + 8
KDT_NODE_SIZE:
.previous


.endif # KDT_INC
