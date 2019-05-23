#!/usr/bin/env python3

import glob
import subprocess
import sys

if __name__ == "__main__":
    tex_files = glob.glob("*.tex")
    print("Checking spelling of {} tex source files...".format(len(tex_files)))
    errors = []

    for tex_file in tex_files:
        result = subprocess.run(["hunspell", "-u3", "-d", "en_US", "-p", "dict.txt", "-t", tex_file], stdout=subprocess.PIPE)
        stdout = result.stdout
        if len(stdout) > 0:
            errs = list(map(lambda line: line.strip(), stdout.strip().split(b"\n")))
            errors.extend(errs)

        warnings = ["todo", "seven", "layer", "layers", "7"]
        # Check for todo as well
        with open(tex_file, "r") as fin:
            for line_no, line in enumerate(fin.readlines()):
                l = line.strip().lower()
                for warning in warnings:
                    if warning in l:
                        print("WARN: {} {}:{} -- {}".format(warning, tex_file, line_no, line))

    if len(errors) > 0:
        print("Errors found...")
        for error in errors:
            print(error)
        sys.exit(1)
    else:
        print("No errors found...")
        sys.exit(0)

