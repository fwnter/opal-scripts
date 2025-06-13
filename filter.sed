/^#{20,}$/p
/^##### .* #####$/{p;d}
/^[[:space:]]*Analysis time:/p
/^[[:space:]]*evaluation time:/p
/^[[:space:]]*running[[:space:]].*analysis[[:space:]]*$/,/^[[:space:]]*evaluation time:/p