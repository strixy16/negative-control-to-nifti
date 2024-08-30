#!/bin/bash

HNSCC_DIR="/cluster/projects/radiomics/PublicDatasets/procdata/HeadNeck/TCIA_HNSCC/converted_niftis/"
empty_count=0
non_empty_count=0

for dir in "$HNSCC_DIR"/*/CT; do
    if [ -d "$dir" ]; then
        if [ -z "$(ls -A "$dir")" ]; then
            ((empty_count++))
        else
            ((non_empty_count++))
        fi
    fi
done

echo "Number of directories with empty CT: $empty_count"
echo "Number of directories with non-empty CT: $non_empty_count"