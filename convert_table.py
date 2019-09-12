import typing

table = r"""
\begin{table}[H]
	\centering
	\caption{OPQ Deployment}
	\begin{tabular}{|c|c|c|c|}
		\hline 
		Box & Location & Coordinates & Description \\ 
		\hline
		1000 & POST (CSDL) & -157.816237,21.297438 & Same line as 1002, 1009, 1010, 1013. Sensitive computer equipment. \\
		\hline
		1001 & Hamilton (Basement 3F) & -157.816173,21.300332 & Same line as 1008. Ease of access. \\
		\hline
		1002 & POST (ICSpace) & -157.826327,21.30046 & Same line as 1000, 1009, 1010, 1013. Sensitive computer equipment.   \\
		\hline
		1003 & IT Building & -157.826327,21.30046 & Same line as 1004. Sensitive computer equipment. \\
		\hline
		1004 & Building 37 & -157.817821,21.298732 & Same line as 1003. LAVA Lab. \\
		\hline
		1005 & Parking Structure (Phase II) & -157.819234, 21.296042 & Same line as 1011, 1012. Proximity to PV. \\
		\hline
		1006 & Frog I & -157.823122,21.298050 & Same line as 1007, 1014. Proximity to PV. PV ground truth access. \\
		\hline
		1007 & Frog II & -157.822819,21.298046 & Same line as 1006, 1014. Proximity to PV. PV ground truth access.  \\
		\hline
		1008 & Ag. Engineering & -157.816107,21.301564 & Same line as 1001. Known PQ issues. \\
		\hline
		1009 & Watanabe & -157.815817,21.298351 & Same line as 1000, 1002, 1010, 1013. Sensitive physics instruments. \\
		\hline
		1010 & Holmes & -157.816104,21.297011 & Same line as 1000, 1002, 1009, 1013. Machining shops and electronic equipment. \\
		\hline
		1011 & Parking Structure (Phase I) & -157.817430,21.295338 & Same line as 1005, 1012. Proximity to PV. \\
		\hline
		1012 & Law Library & -157.817361,21.296328 & Same line as 1005, 1011. \\
		\hline
		1013  & Kennedy Theater & -157.815225,21.299282 & Same line as 1000, 1002, 1009, 1010. Demanding electronic equipment. \\
		\hline
		1014 & Sinclair Library & -157.820499,21.298475 & Same line as 1006, 1007. \\
		\hline
		1015 & Architecture Portables & -157.813644,21.301092 & Edge of campus. Portable buildings. \\
		\hline
	\end{tabular} 
	\label{table:OpqDeployment}
\end{table}

"""


def find_idx(lst: typing.List[str],
             predicate: typing.Callable[[str], bool]) -> int:
    i = 0
    for v in lst:
        if predicate(v):
            return i
        i += 1

    return -1


def tabs_at_idx(lst: typing.List[str], idx: int) -> str:
    return lst[idx].count("\t") * "\t"


if __name__ == "__main__":
    table = table.replace(r"\begin{tabular}", r"\begin{tabularx}{\textwidth}")
    table = table.replace(r"\end{tabular}", r"\end{tabularx}")
    table_lines = list(filter(lambda line: r"\hline" not in line, table.split("\n")))

    # Insert top rule
    top_rule_idx = find_idx(table_lines, lambda line: r"\begin{tabularx}{\textwidth}" in line) + 1
    rule_tabs = tabs_at_idx(table_lines, top_rule_idx)
    table_lines.insert(top_rule_idx, rule_tabs + r"\toprule")

    # Insert mid rule
    mid_rule_idx = find_idx(table_lines, lambda line: r"\toprule" in line) + 2
    table_lines.insert(mid_rule_idx, rule_tabs + r"\midrule")

    # Insert bottom rule
    bottom_rule_idx = find_idx(table_lines, lambda line: r"\end{tabularx}" in line)
    table_lines.insert(bottom_rule_idx, rule_tabs + r"\bottomrule")

    # Fix bold top row
    header_idx = find_idx(table_lines, lambda line: r"\toprule" in line) + 1
    header_line = table_lines[header_idx]
    header_line_tabs = header_line.count("\t") * "\t"
    header_values = header_line.strip().split()
    header_values = list(filter(lambda v: v not in {'&', '\\\\'}, header_values))
    header_values = list(map(lambda v: r"\textbf{%s}" % v, header_values))
    updated_header_line = header_line_tabs + " & ".join(header_values) + r" \\"
    table_lines[header_idx] = updated_header_line

    for line in table_lines:
        # Fix alignment
        if r"\begin{tabularx}{\textwidth}" in line:
            line = line.replace("|", "")
            line = line.replace("c", "X")

        print(line)
