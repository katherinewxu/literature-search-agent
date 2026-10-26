#!/bin/bash
# Frozen benchmark build: SYNERGY+ source -> unfiltered store -> cached OpenAlex abstract backfill -> audit.
set -euo pipefail
r=/scratch/users/$USER/literature-search-agent
cd ~/literature-search-agent
.venv/bin/litsearch data build --source synergy-source   --input ~/.synergy_dataset_source/synergy-dataset-plus   --reviews Appenzeller-Herzog_2019,Bos_2018,Brouwer_2019,Donners_2021,Hall_2011,Leenaars_2019,Meijboom_2021,Muthu_2020   --review-metadata $r/synergy/raw_rest/metadata/review_metadata.csv   --output $r/data/synergy-v3-base
.venv/bin/litsearch data backfill-abstracts --data $r/data/synergy-v3-base --output $r/data/synergy-v3 --cache $r/cache/openalex
.venv/bin/litsearch data audit --data $r/data/synergy-v3 > $r/data/synergy-v3-audit.json
