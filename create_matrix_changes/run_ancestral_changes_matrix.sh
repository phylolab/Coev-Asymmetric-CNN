#!/bin/bash
fastaName=${arrIN[0]}
pathTree=${arrIN[1]}
pathChanges=${arrIN[2]}
typeFasta=${arrIN[4]}

Rscript create_ancestral_changes_array.r $fastaName $pathTree $pathChanges $typeFasta
